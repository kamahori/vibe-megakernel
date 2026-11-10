"""Qwen3-30B-A3B decode step as ONE persistent cooperative CUDA kernel (mk.cu)."""
import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_SCRATCH = 64 + 5120 * 2 + 4096 * 2 + 2048 * 2 + 8 * 768 * 2 + 128 * 4 + 256 * 4 + 256 * 4 + 64 \
    + 5120 * 4 + 2048 * 4 + 8 * 2 * 768 * 4 + 8 * 2048 * 2 + 2048 * 2
_EVENTS = ["qkv", "attention", "wo", "router", "gate_up", "down", "lm_head", "barrier"]
_libs = {}


def _lib(timeline: bool):
    key = bool(timeline)
    if key in _libs:
        return _libs[key]
    srcs = [_DIR / "mk.cu", _DIR / "timeline_probe.cuh"]
    h = hashlib.sha256()
    for s in srcs:
        h.update(s.read_bytes())
    h.update(b"tl" if timeline else b"rel")
    so = _DIR / "_build" / f"mk_{h.hexdigest()[:16]}.so"
    if not so.exists():
        so.parent.mkdir(exist_ok=True)
        nvcc = shutil.which("nvcc") or "/usr/local/cuda-13.1/bin/nvcc"
        tmp = so.with_suffix(f".{os.getpid()}.tmp")
        cmd = [nvcc, "-gencode", "arch=compute_100a,code=sm_100a", "-O3", "-std=c++17", f"-I{_DIR}", "-shared", "-Xcompiler", "-fPIC",
               str(_DIR / "mk.cu"), "-o", str(tmp)]
        if timeline:
            cmd.insert(1, "-DMEGABENCH_TIMELINE")
        subprocess.run(cmd, check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    lib.mk_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.mk_launch.restype = ctypes.c_int
    _libs[key] = lib
    return lib


def _make(case: dict, timeline: bool):
    p = case["params"]
    assert p["batch"] == 1 and p["context"] == 128 and p["layers"] == 48
    dev = torch.device("cuda")
    lib = _lib(timeline)
    D, theta, pos = p["head_dim"], 1e6, p["context"]
    inv = theta ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    f = torch.outer(torch.tensor([float(pos)], dtype=torch.float64), inv)
    emb = torch.cat([f, f], dim=-1)[0]
    cosb = emb.cos().float().to(dev).contiguous()
    sinb = emb.sin().float().to(dev).contiguous()
    scratch = torch.zeros(_SCRATCH, dtype=torch.uint8, device=dev)
    tl_words = 1 + 148 * 1300
    names = ("ln1", "ln2", "fnorm", "wq", "wk", "wv", "wo", "qn", "kn", "wrt", "wg", "wu", "wd", "embed",
             "lm_head", "kcache", "vcache")
    arr = (ctypes.c_ulonglong * 27)()

    def run(inputs):
        layers = p["layers"]
        logits = torch.empty(p["vocab"], dtype=torch.float32, device=dev)
        nxt = torch.empty((), dtype=torch.int64, device=dev)
        kw = torch.empty(layers, p["kv_heads"], D, dtype=torch.bfloat16, device=dev)
        vw = torch.empty_like(kw)
        tl = torch.zeros(tl_words, dtype=torch.uint64, device=dev) if timeline else None
        for i, n in enumerate(names):
            assert inputs[n].is_contiguous()
            arr[i] = inputs[n].data_ptr()
        for i, t in enumerate((inputs["token"], cosb, sinb, kw, vw, logits, nxt, scratch)):
            arr[17 + i] = t.data_ptr()
        arr[25] = tl.data_ptr() if timeline else 0
        arr[26] = tl_words
        err = lib.mk_launch(ctypes.addressof(arr), torch.cuda.current_stream().cuda_stream)
        if err:
            raise RuntimeError(f"mk_launch failed: cuda error {err}")
        out = {"logits": logits, "next_token": nxt, "k_write": kw, "v_write": vw}
        return tl if timeline else out

    return run


def build(case: dict):
    return _make(case, False)


def build_timeline(case: dict):
    return _make(case, True), _EVENTS
