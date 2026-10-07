"""Trusted input generator and PyTorch oracle for ready whole-model cases."""

from __future__ import annotations

import importlib

import torch

from ..cases import Case


Inputs = dict[str, torch.Tensor]
Outputs = dict[str, torch.Tensor]

TASK_MODULES = {
    "dense_step": "dense", "moe_step": "moe", "quant_step": "gemma",
    "spec_target_step": "eagle3", "gptoss_step": "gptoss",
    "hybrid_step": "hybrid", "vl_decode_step": "vision",
    "spec_full_iteration": "speculative",
    "distributed_step": "distributed",
    "deepseek_v32_step": "frontier", "glm52_step": "frontier",
    "kimi_k3_step": "kimi",
}


def make_inputs(case: Case, seed: int, device: str = "cpu") -> Inputs:
    if not case.ready:
        raise NotImplementedError(f"{case.id}: {case.note}")
    if case.family not in TASK_MODULES:
        raise NotImplementedError(f"no input generator for {case.id}")
    module = importlib.import_module(f"{__package__}.{TASK_MODULES[case.family]}")
    return module.make_inputs(case, seed, device)


def reference(case: Case, t: Inputs) -> Outputs:
    if not case.ready:
        raise NotImplementedError(f"{case.id}: {case.note}")
    if case.family not in TASK_MODULES:
        raise NotImplementedError(f"no PyTorch oracle for {case.id}")
    module = importlib.import_module(f"{__package__}.{TASK_MODULES[case.family]}")
    return module.reference(case, t)
