"""Benchmark an MPK MLP subgraph at MegaBench P0 model geometries.

This is a component diagnostic, not a MegaBench case score. The graph computes
gate/up linear -> SiLU product -> down linear plus residual with BF16 tensors.
It does not include attention, routing, quantization, or output logits.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

from megabench.harness.benchmark import _measure


GEOMETRIES = {
    "dense-step-qwen3-06b-b1-s128": (1, 1024, 3072),
    "moe-step-qwen3-30b-a3b-b1-s128": (1, 2048, 768),
    "spec-target-step-llama31-8b-k4": (5, 4096, 14336),
}


def grid_for_linear(output_dim: int) -> int:
    if output_dim % 96 == 0:
        return output_dim // 96
    if output_dim % 64 == 0:
        return output_dim // 64
    raise ValueError(f"unsupported output dimension {output_dim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=GEOMETRIES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    batch, hidden, intermediate = GEOMETRIES[args.case]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    experiment_tmp = args.output_dir / "tmp"
    experiment_tmp.mkdir()
    os.environ["TMPDIR"] = str(experiment_tmp.resolve())
    tempfile.tempdir = str(experiment_tmp.resolve())

    gen = torch.Generator(device="cuda").manual_seed(20261001)

    def random(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (torch.randn(shape, generator=gen, device="cuda") * scale).bfloat16()

    x = random((batch, hidden), 0.2)
    residual = random((batch, hidden), 0.2)
    w_gate = random((intermediate, hidden), hidden ** -0.5)
    w_up = random((intermediate, hidden), hidden ** -0.5)
    w_down = random((hidden, intermediate), intermediate ** -0.5)
    mid = torch.zeros((batch, 2 * intermediate), dtype=torch.bfloat16, device="cuda")
    activated = torch.zeros((batch, intermediate), dtype=torch.bfloat16, device="cuda")
    out = torch.zeros((batch, hidden), dtype=torch.bfloat16, device="cuda")

    workers, schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=workers,
                  num_local_schedulers=schedulers,
                  max_num_batched_tokens=batch,
                  max_num_batched_requests=batch,
                  max_num_pages=batch)
    pk = PersistentKernel(**params)
    try:
        x_dt = pk.attach_input(x, name="x")
        residual_dt = pk.attach_input(residual, name="residual")
        gate_dt = pk.attach_input(w_gate, name="w_gate")
        up_dt = pk.attach_input(w_up, name="w_up")
        down_dt = pk.attach_input(w_down, name="w_down")
        mid_dt = pk.attach_input(mid, name="mid")
        activated_dt = pk.attach_input(activated, name="activated")
        out_dt = pk.attach_input(out, name="out")
        grid = grid_for_linear(2 * intermediate)
        fused_dt = pk.shuffle_tensors(
            inputs=[gate_dt, up_dt], shuffled_dim=0,
            num_groups=grid // 2, name="w_gate_up")
        block = (256, 1, 1)
        pk.linear_layer(x_dt, fused_dt, mid_dt, (grid, 1, 1), block)
        pk.silu_mul_layer(mid_dt, activated_dt, (grid // 2, 1, 1), block)
        pk.linear_with_residual_layer(
            activated_dt, down_dt, residual_dt, out_dt,
            (hidden // 64, 1, 1), block)

        pk.compile(output_dir=str(args.output_dir.resolve()))
        pk()
        torch.cuda.synchronize()
        gate = x.float() @ w_gate.float().T
        up = x.float() @ w_up.float().T
        expected = torch.nn.functional.silu(gate) * up
        expected = expected @ w_down.float().T + residual.float()
        diff = (out.float() - expected).abs()
        max_abs = diff.max().item()
        max_rel = (diff / expected.abs().clamp_min(1e-2)).max().item()
        torch.testing.assert_close(out.float(), expected, atol=0.003, rtol=0.03)

        reset_request = pk.init_func.__self__.init_request_func

        def run_once() -> None:
            reset_request()
            pk()

        out.zero_()
        run_once()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, atol=0.003, rtol=0.03)
        timing = _measure(lambda _: run_once(), {}, "cuda", args.warmup, args.reps)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            run_once()
            torch.cuda.synchronize()
        kernels = [event.name for event in trace.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        result = {
            "scope": "MPK MLP component only; not a full MegaBench case",
            "case_geometry": args.case,
            "shape": {"batch": batch, "hidden": hidden,
                      "intermediate": intermediate},
            "max_abs_error_vs_fp32_reference": max_abs,
            "max_relative_error_floor_1e-2": max_rel,
            "timing": timing,
            "gpu_kernel_count": len(kernels),
            "gpu_kernel_names": kernels,
            "request_reset_included_in_timing": True,
            "second_call_recomputed_after_clearing_output": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        }
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        pk.finalize()


if __name__ == "__main__":
    main()
