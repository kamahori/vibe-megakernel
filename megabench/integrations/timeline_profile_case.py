"""Record one warmed per-block phase timeline of a MegaBench candidate.

A candidate opts in by exporting ``build_timeline(case: dict) -> (run, event_names)``
next to ``build``. ``run(inputs)`` launches a build whose probes are enabled (for
example ``-DMEGABENCH_TIMELINE`` with ``timeline_probe.cuh``, or FlashInfer's
``FLASHINFER_ENABLE_PROFILER`` macros) and returns the zeroed-then-filled ``uint64``
or ``int64`` CUDA buffer of that call. The first call is warmup; the second is saved.

Probes add a timer read and a store each, so this build is slower than the scored one
and can shift where cost lands. Use it to find the phase and blocks on the critical
path, and confirm any change with ``check_candidate``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from ..cases import select_cases
from ..workloads import make_inputs
from . import timeline


def _load_timeline_build(path: Path, contract: dict):
    spec = importlib.util.spec_from_file_location("megabench_candidate_timeline", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import submission {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    build = getattr(module, "build_timeline", None)
    if not callable(build):
        raise TypeError("submission must export build_timeline(case: dict) -> (run, event_names) "
                        "to record a timeline; see megabench/integrations/timeline_probe.cuh")
    run, names = build(contract)
    if not callable(run):
        raise TypeError("build_timeline(case) must return (callable run(inputs), event names)")
    return run, [str(name) for name in names]


def _save(buffer, names: list[str], output: Path) -> str:
    import torch

    if buffer.dtype not in (torch.uint64, torch.int64) or buffer.dim() != 1:
        raise TypeError(f"the timeline run must return a 1-D uint64 or int64 tensor; got "
                        f"{buffer.dtype} with shape {tuple(buffer.shape)}")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "timeline.u64"
    raw.write_bytes(buffer.detach().view(torch.int64).cpu().numpy().tobytes())
    (output / "events.json").write_text(json.dumps(names))
    decoded = timeline.decode_buffer(raw.read_bytes(), names)
    (output / "trace.json").write_text(json.dumps(timeline.to_chrome_trace(decoded)))
    rows = timeline.phase_table(decoded)
    summary = timeline.summarize(decoded, top=len(rows) or 1)
    (output / "summary.json").write_text(json.dumps({"phases": rows, "summary": summary}, indent=2))
    return timeline.format_table(rows)


def capture(rank: int, contract: dict, submission: str, seed: int, output: str,
            rendezvous: str | None = None) -> None:
    import torch
    import torch.distributed as dist
    from ..cases import Case
    from ..tasks.parallel import initialize

    case = Case(**contract)
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    contract = case.to_dict()
    if rendezvous:
        dist.init_process_group('nccl', init_method=rendezvous, rank=rank,
                                world_size=case.gpus, timeout=timedelta(minutes=10))
        ctx = initialize(case)
        contract["execution"] = {'rank': rank, 'world_size': case.gpus, 'tp_rank': ctx.tp_rank,
                                 'ep_rank': ctx.ep_rank, 'tp_group': ctx.tp_group,
                                 'ep_group': ctx.ep_group, 'backend': 'nccl', 'device': device}
    try:
        with torch.inference_mode():
            inputs = make_inputs(case, seed, device)
            run, names = _load_timeline_build(Path(submission), contract)
            run(inputs)
            torch.cuda.synchronize(device)
            if rendezvous:
                dist.barrier()
            buffer = run(inputs)
            torch.cuda.synchronize(device)
        target = Path(output) / (f"rank{rank}" if rendezvous else "")
        table = _save(buffer, names, target)
        if rank == 0:
            print(table)
        print(f"rank {rank}: wrote {target}/timeline.u64, trace.json (ui.perfetto.dev), summary.json")
    finally:
        if rendezvous:
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7301)
    args = parser.parse_args(argv)

    case = select_cases("all", [args.case])[0]
    if not case.ready:
        parser.error(f"case is not ready: {case.id}")
    submission = args.submission.resolve(strict=True)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")

    import torch

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required to record a timeline")
    if torch.cuda.device_count() < case.gpus:
        parser.error(f"{case.id} requires {case.gpus} visible GPUs")
    if case.gpus == 1:
        capture(0, case.to_dict(), str(submission), args.seed, str(args.output))
    else:
        with tempfile.TemporaryDirectory(prefix='megabench-timeline-') as directory:
            torch.multiprocessing.spawn(capture, nprocs=case.gpus, join=True,
                args=(case.to_dict(), str(submission), args.seed, str(args.output),
                      (Path(directory) / 'rendezvous').as_uri()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
