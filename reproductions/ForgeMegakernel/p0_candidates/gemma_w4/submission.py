"""Single-launch Gemma 3 W4 decode candidate."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None
_BARRIERS_PER_CALL = 1 + 34 * 10 + 2


def _extension():
    global _EXT
    if _EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
        build_dir = Path("/tmp/gemma_w4_persistent_v1")
        build_dir.mkdir(parents=True, exist_ok=True)
        _EXT = load(
            name="gemma_w4_persistent_v1",
            sources=[str(Path(__file__).with_name("gemma_kernel.cu"))],
            build_directory=str(build_dir),
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


def build(case: dict):
    p = case["params"]
    expected = (p["layers"], p["hidden"], p["intermediate"],
                p["q_heads"], p["kv_heads"], p["head_dim"],
                p["vocab"], p["context"], p["bits"])
    if expected != (34, 2560, 10240, 8, 4, 256, 262144, 128, 4):
        raise ValueError("unsupported Gemma geometry")

    extension = _extension()
    scratch = None
    sync = None
    calls = 0

    def run(inputs: dict):
        nonlocal scratch, sync, calls
        device = inputs["token"].device
        if scratch is None:
            scratch = torch.empty((44544,), device=device, dtype=torch.float32)
            sync = torch.tensor([0, 0], device=device, dtype=torch.int32)
        logits = torch.empty((262144,), device=device, dtype=torch.float32)
        next_token = torch.empty((), device=device, dtype=torch.int64)
        k_write = torch.empty((34, 4, 256), device=device, dtype=torch.bfloat16)
        v_write = torch.empty_like(k_write)
        extension.step(inputs, scratch, sync, logits, next_token,
                       k_write, v_write, calls * _BARRIERS_PER_CALL)
        calls += 1
        return {"logits": logits, "next_token": next_token,
                "k_write": k_write, "v_write": v_write}

    return run
