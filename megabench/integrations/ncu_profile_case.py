"""Run one ready MegaBench candidate inside an Nsight Compute capture window.

The external ncu MCP server accepts one executable path and no arguments. A
session-specific executable can call this module with its fixed case and
submission paths. Profiling starts after input creation and JIT warmup. Use
the MCP profile tool's kernel_filter to select the candidate kernel, since
PyTorch setup can launch unrelated GPU kernels.
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import timedelta
from pathlib import Path

from ..cases import select_cases
from ..harness.runner import _load_submission
from ..workloads import make_inputs


def capture(rank: int, contract: dict, submission: str, seed: int, rendezvous: str | None = None) -> None:
    import torch
    import torch.distributed as dist
    from ..cases import Case
    from ..tasks.parallel import initialize

    case = Case(**contract)
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    execution = None
    if rendezvous:
        dist.init_process_group('nccl', init_method=rendezvous, rank=rank,
                                world_size=case.gpus, timeout=timedelta(minutes=10))
        ctx = initialize(case)
        execution = {'rank': rank, 'world_size': case.gpus, 'tp_rank': ctx.tp_rank,
                     'ep_rank': ctx.ep_rank, 'tp_group': ctx.tp_group, 'ep_group': ctx.ep_group,
                     'backend': 'nccl', 'device': device}
    try:
        with torch.inference_mode():
            inputs = make_inputs(case, seed, device)
            candidate = _load_submission(Path(submission), case, execution=execution)
            candidate(inputs)
            torch.cuda.synchronize(device)
            if rendezvous:
                dist.barrier()
            torch.cuda.cudart().cudaProfilerStart()
            try:
                candidate(inputs)
                torch.cuda.synchronize(device)
            finally:
                torch.cuda.cudart().cudaProfilerStop()
    finally:
        if rendezvous:
            dist.destroy_process_group()


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
    if torch.cuda.device_count() < case.gpus:
        parser.error(f"{case.id} requires {case.gpus} visible GPUs")
    if case.gpus == 1:
        capture(0, case.to_dict(), str(submission), args.seed)
    else:
        with tempfile.TemporaryDirectory(prefix='megabench-ncu-') as directory:
            torch.multiprocessing.spawn(capture, nprocs=case.gpus, join=True,
                args=(case.to_dict(), str(submission), args.seed,
                      (Path(directory) / 'rendezvous').as_uri()))
    print(f"Captured one steady-state run of {case.id}: {submission}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
