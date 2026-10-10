"""Nsight Systems cross-check: GPU kernel durations of one decode step at b1-s128.

Run each side under ``nsys profile --capture-range=cudaProfilerApi``; only the
timed calls between ``cudaProfilerStart``/``Stop`` are captured:

* ``mpk`` (MPK venv): the PR #10 ``online_notoken`` probe (one call = one decode
  step at step 128 over 129 positions). Weights, KV and metadata are loaded once
  outside the capture. Each call launches MPK's prepare, worker and scheduler
  kernels; the worker kernel's duration is the GPU time of the step without
  host launch latency or the host synchronization.
* ``opus`` (.venv): the ``opus`` arm's ``direct`` step, one cooperative launch per
  call, synchronized after each call so the launches do not overlap.

``summarize`` reads the ``cuda_gpu_trace`` CSV that ``nsys stats`` writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

CASE_ID = "dense-step-qwen3-06b-b1-s128"


def _profiler(on: bool) -> None:
    import torch
    (torch.cuda.cudart().cudaProfilerStart if on else torch.cuda.cudart().cudaProfilerStop)()


def run_mpk(reps: int, warmup: int, seed: int, compile_root: str) -> None:
    import torch
    from megabench.cases import select_cases
    from megabench.probes.mpk_native_qwen3_probe import NativeQwen3
    from megabench.tasks.workloads import make_inputs
    case = select_cases("all", [CASE_ID])[0]
    values = make_inputs(case, seed, "cuda")
    cand = NativeQwen3(case, values, case.params["layers"], Path(compile_root))
    cand.load(values)

    def once():
        cand.meta["step"].fill_(cand.context)
        cand.meta["paged_kv_last_page_len_buffer"].fill_(cand.context + 1)
        torch.cuda.synchronize()
        cand.pk()
        torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(warmup):
            once()
        _profiler(True)
        for _ in range(reps):
            once()
        _profiler(False)


def run_opus(reps: int, warmup: int, seed: int) -> None:
    import torch
    from megabench.sota.arms.opus import OpusArm
    from megabench.sota.shapes import Shape, make_shape_inputs
    shape = Shape(1, 128)
    inputs = make_shape_inputs(shape, seed, "cuda")
    step = OpusArm().prepare(shape, inputs, "direct", "cuda")
    step.set_chained(False)
    for _ in range(warmup):
        step.launch()
    torch.cuda.synchronize()
    _profiler(True)
    for _ in range(reps):
        step.launch()
        torch.cuda.synchronize()
    _profiler(False)
    step.close()


def _short(name: str) -> str:
    """A stable short label for a (possibly templated) kernel name."""
    base = name.split("(")[0].split("<")[0].strip()
    return base.split(" ")[-1].split("::")[-1] or name[:60]


def summarize(csv_path: Path) -> dict:
    """Per-kernel duration statistics (us) and, for MPK, per-call GPU spans."""
    rows = list(csv.DictReader(open(csv_path, newline="")))
    if not rows:
        return {"kernels": {}, "n_rows": 0}
    keys = rows[0].keys()
    start_k = next(k for k in keys if k.lower().startswith("start"))
    dur_k = next(k for k in keys if k.lower().startswith("duration"))
    name_k = next(k for k in keys if k.lower() == "name")
    kernels: dict[str, list[float]] = {}
    events = []
    for r in rows:
        name = r[name_k]
        if name.startswith("[CUDA"):  # memcpy / memset rows
            continue
        start, dur = float(r[start_k]), float(r[dur_k])
        label = _short(name)
        kernels.setdefault(label, []).append(dur / 1e3)
        if "at::native" not in name:  # the harness's own fill_ kernels are not MPK work
            events.append((start, start + dur, label))
    out = {"n_rows": len(rows), "kernels": {}}
    for label, d in kernels.items():
        d = sorted(d)
        out["kernels"][label] = {"n": len(d), "median_us": statistics.median(d),
                                 "min_us": d[0], "max_us": d[-1]}
    # MPK: one call = prepare -> worker + scheduler; span = first start to last end.
    events.sort()
    preps = [e for e in events if "prepare" in e[2].lower()]
    if preps:
        spans = []
        for i, p in enumerate(preps):
            nxt = preps[i + 1][0] if i + 1 < len(preps) else float("inf")
            call = [e for e in events if p[0] <= e[0] < nxt]
            spans.append((max(e[1] for e in call) - p[0]) / 1e3)
        spans.sort()
        out["mpk_call_span_us"] = {"n": len(spans), "median_us": statistics.median(spans),
                                   "min_us": spans[0], "max_us": spans[-1]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mpk", "opus"):
        p = sub.add_parser(name)
        p.add_argument("--reps", type=int, default=50)
        p.add_argument("--warmup", type=int, default=10)
        p.add_argument("--seed", type=int, default=7301)
        if name == "mpk":
            p.add_argument("--compile-root",
                           default="/raid/garv901/.cache/megabench_sota/mpk_builds")
    s = sub.add_parser("summarize")
    s.add_argument("csv", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "mpk":
        run_mpk(args.reps, args.warmup, args.seed, args.compile_root)
    elif args.cmd == "opus":
        run_opus(args.reps, args.warmup, args.seed)
    else:
        json.dump(summarize(args.csv), sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
