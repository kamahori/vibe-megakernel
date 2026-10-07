"""One-launch Qwen3 decode for the fixed public MegaBench geometry."""

from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import load


_extension = None


def _kernel():
    global _extension
    if _extension is None:
        build_dir = "/tmp/dense_step_qwen3_build"
        os.makedirs(build_dir, exist_ok=True)
        _extension = load(
            name="dense_step_qwen3_persistent",
            sources=[str(Path(__file__).with_name("src") / "dense_step.cu")],
            build_directory=build_dir,
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _extension


def build(case: dict):
    p = case["params"]
    assert (p["batch"], p["context"], p["layers"], p["hidden"],
            p["q_heads"], p["kv_heads"], p["head_dim"],
            p["intermediate"], p["vocab"]) == (
                1, 128, 28, 1024, 16, 8, 128, 3072, 151936)
    ext = _kernel()

    def run(inputs: dict):
        device = inputs["token"].device
        logits = torch.empty((p["vocab"],), device=device, dtype=torch.float32)
        next_token = torch.empty((), device=device, dtype=torch.int64)
        k_write = torch.empty((p["layers"], p["kv_heads"], p["head_dim"]),
                              device=device, dtype=torch.bfloat16)
        v_write = torch.empty_like(k_write)
        scratch = torch.empty((25000,), device=device, dtype=torch.float32)
        ext.run(inputs["token"], inputs["ln1"], inputs["wq"], inputs["wk"],
                inputs["wv"], inputs["qn"], inputs["kn"], inputs["wo"],
                inputs["ln2"], inputs["wg"], inputs["wu"], inputs["wd"],
                inputs["fnorm"], inputs["embed"], inputs["kcache"],
                inputs["vcache"], logits, next_token, k_write, v_write,
                scratch)
        return {"logits": logits, "next_token": next_token,
                "k_write": k_write, "v_write": v_write}

    return run
