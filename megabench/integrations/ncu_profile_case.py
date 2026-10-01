"""Run one ready MegaBench candidate inside an Nsight Compute capture window.

The external ncu MCP server accepts one executable path and no arguments. A
session-specific executable can call this module with its fixed case and
submission paths. Profiling starts after input creation and JIT warmup. Use
the MCP profile tool's kernel_filter to select the candidate kernel, since
PyTorch setup can launch unrelated GPU kernels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..cases import select_cases
from ..harness.runner import _load_submission
from ..workloads import make_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7301)
    args = parser.parse_args(argv)

    case = select_cases("all", [args.case])[0]
    if not case.ready:
        parser.error(f"case is not ready: {case.id}")
    submission = args.submission.resolve(strict=True)

    import torch

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required for ncu profiling")
    with torch.inference_mode():
        inputs = make_inputs(case, args.seed, "cuda:0")
        candidate = _load_submission(submission, case)
        candidate(inputs)  # compile and warm the submission outside the capture
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            candidate(inputs)
            torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
    print(f"Captured one steady-state run of {case.id}: {submission}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
