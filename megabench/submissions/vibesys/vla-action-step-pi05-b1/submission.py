import ctypes
import hashlib
import os
import subprocess

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "pi05_kernel.cu")
_NVCC_FLAGS = ["-O3", "-std=c++17", "-gencode", "arch=compute_100a,code=sm_100a", "-shared", "-Xcompiler", "-fPIC"]

_INPUT_ORDER = [
    "action_in_weight", "action_in_bias", "time_in_weight", "time_in_bias",
    "time_out_weight", "time_out_bias", "action_out_weight", "action_out_bias",
    "wq", "wk", "wv", "wo", "wg", "wu", "wd",
    "attn_norm_weight", "attn_norm_bias", "ffn_norm_weight", "ffn_norm_bias",
    "final_norm_weight", "final_norm_bias", "noise", "prefix_k", "prefix_v", "prefix_mask",
]


_TL_EVENTS = ["temb", "time_in", "time_out", "mod", "norm_attn", "qkv", "rope", "attn_scores", "attn_pv",
              "conv", "o_proj", "norm_ffn", "gate_up", "gelu", "down", "final_norm", "action_out"]
_TL_NAMES = _TL_EVENTS + [f"unused{i}" for i in range(len(_TL_EVENTS), 32)] + ["bar_" + e for e in _TL_EVENTS]
_TL_RECORDS = 9000  # records per CTA writer (>= 4 per phase occurrence)


def _compile(extra=()):
    flags = [*_NVCC_FLAGS, *extra]
    src = open(_SRC, "rb").read() + open(os.path.join(_HERE, "timeline_probe.cuh"), "rb").read()
    key = hashlib.sha256(src + " ".join(flags).encode()).hexdigest()[:16]
    outdir = os.path.join(_HERE, ".build")
    os.makedirs(outdir, exist_ok=True)
    so = os.path.join(outdir, f"pi05_{key}.so")
    if not os.path.exists(so):
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.run(["nvcc", *flags, _SRC, "-o", tmp], check=True, timeout=900)
        os.replace(tmp, so)
    return so


def build(case: dict):
    return _build(case, False)


def build_timeline(case: dict):
    return _build(case, True), list(_TL_NAMES)


def _build(case: dict, timeline: bool):
    p = case["params"]
    assert p["layers"] == 18 and p["expert_hidden"] == 1024 and p["horizon"] == 50
    lib = ctypes.CDLL(_compile(["-DMEGABENCH_TIMELINE"] if timeline else []))
    lib.pi05_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
    lib.pi05_launch.restype = ctypes.c_int
    lib.pi05_nbar.restype = ctypes.c_int
    nbar = lib.pi05_nbar()
    dev = torch.device("cuda")
    grid = torch.cuda.get_device_properties(dev).multi_processor_count
    f32 = dict(device=dev, dtype=torch.float32)
    bf = dict(device=dev, dtype=torch.bfloat16)
    z = torch.zeros
    scratch = [
        z(10 * 1024, **f32), z(10 * 1024, **f32), z(10 * 1024, **f32), z(10 * 37 * 3072, **f32),  # temb hid1 cond mod
        z(50 * 2560, **f32), z(50 * 1024, **f32), z(50 * 8192, **f32), z(50 * 1024, **f32), z(16 * 50 * 2048, **f32),  # ... Yd Opart[16][50][2048]
    ]
    inv = 1.0 / (10000.0 ** (torch.arange(0, 256, 2, dtype=torch.int64).float() / 256))
    inv = inv.to(dev).contiguous()
    h = z(50 * 1024, **bf)
    X1 = z(64 * 1024, **bf)
    Xo = z(64 * 2048, **bf)
    X2 = z(64 * 4096, **bf)
    St = z(8 * 50 * 16 * 2, **f32)  # chunk stats (m, l) [8][50][16]
    bar = z(4, device=dev, dtype=torch.int32)
    cs = z(50 * 128 * 2, **f32)  # rope cos/sin table (float2), rebuilt in-kernel every call
    Qr = z(50 * 2048, **bf)      # roped Q
    Kr = z(50 * 256, **bf)       # roped action-K
    keep = scratch + [inv, h, X1, Xo, X2, St, bar, cs, Qr, Kr]
    n_in = len(_INPUT_ORDER)
    arr = (ctypes.c_void_p * (n_in + 1 + len(scratch) + 1 + 5 + 1 + 3 + 2))()
    k = n_in + 1
    for t in scratch:
        arr[k] = t.data_ptr(); k += 1
    arr[k] = inv.data_ptr(); k += 1
    for t in (h, X1, Xo, X2, St):
        arr[k] = t.data_ptr(); k += 1
    arr[k] = bar.data_ptr(); k += 1
    for t in (cs, Qr, Kr):
        arr[k] = t.data_ptr(); k += 1
    tl = None
    if timeline:
        words = 1 + grid * _TL_RECORDS
        tl = torch.zeros(words, device=dev, dtype=torch.int64)
        keep.append(tl)
        arr[k] = tl.data_ptr(); arr[k + 1] = words
    state = {"epoch": 0}

    def run(inputs):
        out = torch.empty((50, 32), **f32)
        for i, name in enumerate(_INPUT_ORDER):
            arr[i] = inputs[name].data_ptr()
        arr[n_in] = out.data_ptr()
        if tl is not None:
            tl.zero_()
        stream = torch.cuda.current_stream().cuda_stream
        err = lib.pi05_launch(ctypes.cast(arr, ctypes.c_void_p), grid, state["epoch"] & 0xFFFFFFFF, stream)
        if err != 0:
            raise RuntimeError(f"pi05 launch failed: {err}")
        state["epoch"] = (state["epoch"] + nbar * grid) & 0xFFFFFFFF
        run._keep = keep
        if timeline:
            return tl
        return {"actions": out}

    return run
