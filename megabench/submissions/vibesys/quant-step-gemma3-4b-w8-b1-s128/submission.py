"""Gemma3-4B W8A16 decode step as a single persistent CUDA kernel (mk/megakernel.cu)."""
import ctypes
import hashlib
import math
import os
import shutil
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "mk" / "megakernel.cu"
_PROBE = _HERE / "timeline_probe.cuh"
_NAMES = ["ln1", "ln2", "ln3", "ln4", "qn", "kn", "fnorm"]
_EVENTS = ["qkv", "attn", "oproj", "gateup", "down", "lmhead"]
_TL_RECORDS = 34 * 5 * 2 + 8


def _nvcc():
    p = shutil.which("nvcc")
    for c in (p, "/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.1/bin/nvcc"):
        if c and os.path.exists(c):
            return c
    raise RuntimeError("nvcc not found")


def _compile(timeline: bool, ctx: int, window: int) -> Path:
    h = hashlib.sha256()
    h.update(_SRC.read_bytes())
    h.update(_PROBE.read_bytes())
    h.update(f"{timeline}{ctx}{window}".encode())
    tag = h.hexdigest()[:16]
    out = _HERE / "mk" / "cache" / f"mk_{tag}.so"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    cmd = [_nvcc(), "-O3", "-std=c++17", "-arch=sm_100a", "--cudart", "shared", "-shared",
           "-Xcompiler", "-fPIC", f"-I{_HERE}", f"-DCTX={ctx}", f"-DWIN={window}"]
    if timeline:
        cmd.append("-DMEGABENCH_TIMELINE")
    cmd += ["-o", str(tmp), str(_SRC)]
    subprocess.run(cmd, check=True)
    os.replace(tmp, out)
    return out


def _rope_table(ctx, device):
    rows = []
    for theta, factor in ((10_000.0, 1.0), (1_000_000.0, 8.0)):
        width = 256
        inv = theta ** (-torch.arange(0, width, 2, device=device, dtype=torch.float64) / width)
        phase = torch.outer(torch.tensor([ctx], device=device).to(torch.float64) / factor, inv)
        phase = torch.cat((phase, phase), dim=-1)
        rows.append(phase.cos().float().to(torch.bfloat16).reshape(-1))
        rows.append(phase.sin().float().to(torch.bfloat16).reshape(-1))
    return torch.cat(rows).contiguous()


def _make(case, timeline):
    p = case["params"]
    assert (p["hidden"], p["layers"], p["intermediate"], p["vocab"], p["head_dim"], p["q_heads"], p["kv_heads"], p["bits"], p["batch"]) \
        == (2560, 34, 10240, 262144, 256, 8, 4, 8, 1)
    ctx = int(p["context"])
    lib = ctypes.CDLL(str(_compile(timeline, ctx, int(p["local_window"]))))
    lib.mk_launch.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.mk_launch.restype = ctypes.c_int
    assert lib.mk_init() == 0
    npar = lib.mk_pcount()
    dev = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    assert 107 <= sms <= 160  # o-proj row-balanced split: 16..23 rows per CTA
    emb_scale = float(torch.tensor(math.sqrt(p["hidden"]), dtype=torch.bfloat16))
    bf, f32 = torch.bfloat16, torch.float32
    s = {
        "rope": _rope_table(ctx, dev),
        "qkv": torch.zeros(4096, dtype=bf, device=dev), "attn": torch.zeros(2048, dtype=bf, device=dev),
        "obuf": torch.zeros(2560, dtype=bf, device=dev), "act": torch.zeros(10240, dtype=bf, device=dev),
        "mlp": torch.zeros(2560, dtype=bf, device=dev),
        "bcnt": torch.zeros(1, dtype=torch.int32, device=dev), "bgen": torch.zeros(1, dtype=torch.int32, device=dev),
        "ticket": torch.zeros(1, dtype=torch.int32, device=dev),
        "pv": torch.zeros(sms, dtype=f32, device=dev), "pi": torch.zeros(sms, dtype=torch.int32, device=dev),
        "dpart": torch.zeros(160 * 16 * 16, dtype=torch.int64, device=dev),
        "apart": torch.zeros(32 * 256 + 64 + 32, dtype=f32, device=dev),  # split-KV partials: o[32][256], (m, z)[32], attn->oproj done counter
    }
    tl_words = 1 + sms * _TL_RECORDS
    order = ["token", "embed", "ln1", "ln2", "ln3", "ln4", "qn", "kn", "fnorm", "wq", "wk", "wv", "wo", "wg", "wu", "wd",
             "wq_scale", "wk_scale", "wv_scale", "wo_scale", "wg_scale", "wu_scale", "wd_scale", "kcache", "vcache"]

    def launch(inputs, tl):
        outs = {"logits": torch.empty(262144, dtype=f32, device=dev), "next_token": torch.empty((), dtype=torch.int64, device=dev),
                "k_write": torch.empty(34, 4, 256, dtype=bf, device=dev), "v_write": torch.empty(34, 4, 256, dtype=bf, device=dev)}
        ptrs = [inputs[k].data_ptr() for k in order]
        ptrs += [s[k].data_ptr() for k in ("rope", "qkv", "attn", "obuf", "act", "mlp")]
        ptrs += [outs["logits"].data_ptr(), outs["next_token"].data_ptr(), outs["k_write"].data_ptr(), outs["v_write"].data_ptr()]
        ptrs += [s[k].data_ptr() for k in ("bcnt", "bgen", "ticket", "pv", "pi", "dpart")]
        ptrs.append(tl.data_ptr() if tl is not None else 0)
        ptrs.append(s["apart"].data_ptr())
        assert len(ptrs) == npar
        arr = (ctypes.c_void_p * npar)(*ptrs)
        rc = lib.mk_launch(arr, emb_scale, tl_words if tl is not None else 0, sms, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"mk_launch failed: {rc}")
        return outs

    return launch, tl_words, dev


def build(case: dict):
    launch, _, _ = _make(case, False)

    def run(inputs: dict):
        return launch(inputs, None)
    return run


def build_timeline(case: dict):
    launch, tl_words, dev = _make(case, True)

    def run(inputs: dict):
        tl = torch.zeros(tl_words, dtype=torch.uint64, device=dev)
        launch(inputs, tl)
        return tl
    return run, list(_EVENTS)
