"""Trusted input generator and PyTorch oracle for ready whole-model cases."""

from __future__ import annotations

import torch

from ..cases import Case


Inputs = dict[str, torch.Tensor]
Outputs = dict[str, torch.Tensor]


def make_inputs(case: Case, seed: int, device: str = "cpu") -> Inputs:
    if not case.ready:
        raise NotImplementedError(f"{case.id}: {case.note}")
    if case.family == "dense_step" and case.model == "Qwen/Qwen3-0.6B":
        from .dense import make_inputs as make_dense_inputs
        return make_dense_inputs(case, seed, device)
    if case.family == "moe_step" and case.model == "Qwen/Qwen3-30B-A3B":
        from .moe import make_inputs as make_moe_inputs
        return make_moe_inputs(case, seed, device)
    if case.family == "quant_step" and case.model == "google/gemma-3-4b-it":
        from .gemma import make_inputs as make_gemma_inputs
        return make_gemma_inputs(case, seed, device)
    if case.family == "spec_target_step" and "EAGLE3" in case.model:
        from .eagle3 import make_inputs as make_spec_inputs
        return make_spec_inputs(case, seed, device)
    raise NotImplementedError(f"no input generator for {case.id}")


def reference(case: Case, t: Inputs) -> Outputs:
    if not case.ready:
        raise NotImplementedError(f"{case.id}: {case.note}")
    if case.family == "dense_step" and case.model == "Qwen/Qwen3-0.6B":
        from .dense import reference as dense_reference
        return dense_reference(case, t)
    if case.family == "moe_step" and case.model == "Qwen/Qwen3-30B-A3B":
        from .moe import reference as moe_reference
        return moe_reference(case, t)
    if case.family == "quant_step" and case.model == "google/gemma-3-4b-it":
        from .gemma import reference as gemma_reference
        return gemma_reference(case, t)
    if case.family == "spec_target_step" and "EAGLE3" in case.model:
        from .eagle3 import reference as spec_reference
        return spec_reference(case, t)
    raise NotImplementedError(f"no PyTorch oracle for {case.id}")
