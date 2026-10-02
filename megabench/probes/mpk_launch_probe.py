"""Profile one native MPK launch against MegaBench's GPU-kernel audit.

Run with a source-built MPK on PYTHONPATH and a visible CUDA GPU. This is a
diagnostic, not a MegaBench submission or a score for any challenge case.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("offline", "onepass"), default="offline")
    args = parser.parse_args()

    experiment_tmp = Path(__file__).resolve().parents[1] / "experiments" / "tmp"
    experiment_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(experiment_tmp)
    tempfile.tempdir = str(experiment_tmp)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MPK launch probe")

    batch, hidden = 1, 4096
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    workers, schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(mode=args.mode, test_mode=True, num_workers=workers,
                  num_local_schedulers=schedulers)
    pk = PersistentKernel(**params)
    x_dt = pk.attach_input(x, name="x")
    weight_dt = pk.attach_input(weight, name="weight")
    out_dt = pk.attach_input(out, name="out")
    pk.rmsnorm_layer(x_dt, weight_dt, out_dt,
                     grid_dim=(batch, 1, 1), block_dim=(256, 1, 1))

    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="mpk-launch-probe-",
                                         dir=experiment_tmp) as tmp:
            compile_and_profile(pk, x, weight, out, Path(tmp))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        compile_and_profile(pk, x, weight, out, args.output_dir)


def compile_and_profile(pk: PersistentKernel, x: torch.Tensor,
                        weight: torch.Tensor, out: torch.Tensor,
                        output_dir: Path) -> None:
    try:
        pk.compile(output_dir=str(output_dir))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            pk()
            torch.cuda.synchronize()
        kernels = [event.name for event in trace.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                    + 1e-5) * weight.float()).bfloat16()
        torch.testing.assert_close(out, expected, atol=0.05, rtol=0)
        original = out.clone()
        out.zero_()
        pk.init_func.__self__.init_request_func()
        pk()
        torch.cuda.synchronize()
        second_call_recomputed = torch.allclose(out, original, atol=0.05, rtol=0)
        out.zero_()
        replacement = torch.randn_like(x)
        pk.init_func.__self__.init_request_func()
        pk(x=replacement)
        torch.cuda.synchronize()
        kwarg_rebound = not torch.allclose(out, original, atol=0.05, rtol=0)
        x.copy_(replacement)
        out.zero_()
        pk.init_func.__self__.init_request_func()
        pk()
        torch.cuda.synchronize()
        replacement_expected = (
            replacement.float() * torch.rsqrt(
                replacement.float().square().mean(-1, keepdim=True) + 1e-5)
            * weight.float()).bfloat16()
        binding = {
            "second_call_recomputed_after_reset": second_call_recomputed,
            "call_kwarg_rebound_input": kwarg_rebound,
            "copy_into_attached_buffer_max_abs_error":
                (out.float() - replacement_expected.float()).abs().max().item(),
        }
        (output_dir / "binding_check.json").write_text(json.dumps(binding, indent=2))
        print(f"MPK GPU kernel executions: {len(kernels)}")
        print(f"Input binding check: {binding}")
        for name in kernels:
            print(name)
    finally:
        pk.finalize()


if __name__ == "__main__":
    main()
