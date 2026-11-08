"""Measure the run-to-run noise of the latency VibeSys optimizes, to set ``RELATIVE_NOISE``.

``relative_noise`` is the relative standard deviation of one benchmark run: VibeSys retains a
checkpoint only when it beats the best by more than that fraction, and its verdict on an idea
treats it as the floor of one run's sigma (see ``RELATIVE_NOISE``). It must therefore cover how
much one unchanged kernel's reading varies between runs, including runs on different GPUs,
because a candidate and its base are measured in separate jobs. This tool repeats the
evaluator's own benchmark measurement (``BENCHMARK_SETTINGS``), each run in a fresh process as
VibeSys runs it, and estimates that sigma per case from the variation within each GPU and
between the GPUs' means.

Measure (inside a one-GPU allocation; ``PYTHONPATH`` holds the benchmark checkout)::

    python -m megabench.integrations.vibesys_noise_calibration run \\
        --case tts-frame-step-csm-1b-b1-s128 --submission path/to/submission.py \\
        --runs 10 --output noise/csm.jsonl

Cover the node's GPUs by pinning one job per GPU index (on cayenne, ``gpu:idxN``)::

    for i in 0 1 2 3 4 5 6 7; do
      sbatch --gres=gpu:idx$i:1 --time=00:20:00 --wrap="python -m \\
        megabench.integrations.vibesys_noise_calibration run --case ... --runs 3 \\
        --submission ... --output noise/csm.jsonl"
    done

Summarize every result file::

    python -m megabench.integrations.vibesys_noise_calibration summarize noise/*.jsonl

Calibration on 2026-11-07 (B200 x 8, final kernels of the 2026-10-09 multimodal campaign):
kernel latency varied by under 0.25% (CV) within one GPU; the per-GPU means varied by 0.16%
(CSM), 0.28% (pi0.5) and 0.96% (Waypoint) (CV), with the slowest GPU 0.4%, 0.9% and 2.2% above
the fastest. The suggested sigma is 0.005, 0.005 and 0.010. The eager baseline varied by up to
4.6% (CV) within one GPU, which is why the latency, not the speedup over eager, is the metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from .vibesys_case_evaluator import BENCHMARK_SETTINGS, RELATIVE_NOISE, kernel_latency_ms

# A suggested sigma is rounded up to this step, so it does not track one GPU's last digit.
_SUGGESTION_STEP = 0.005


def _cv(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.fmean(values)


def measure_once(case_id: str, submission: Path) -> dict:
    """One benchmark measurement exactly as the evaluator's benchmark mode takes it."""
    import torch

    from ..cases import select_cases
    from ..harness.runner import evaluate_case

    case = select_cases("all", [case_id])[0]
    started = time.time()
    result = evaluate_case(case, submission.resolve(), device="cuda:0", **BENCHMARK_SETTINGS)
    props = torch.cuda.get_device_properties(0)
    return {
        "case": case_id,
        "status": result["status"],
        "kernel_latency_ms": kernel_latency_ms(result),
        "kernel_runs_ms": result.get("candidate_timing", {}).get("cuda_event_ms"),
        "eager_ms": result.get("reference_timing", {}).get("cuda_event_p50_ms"),
        "speedup": result.get("speedup_vs_best_baseline_cuda_event"),
        "submission_sha256": result.get("submission_sha256"),
        "gpu_uuid": str(getattr(props, "uuid", "")),
        "gpu_name": props.name,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "wall_s": round(time.time() - started, 1),
    }


def summarize(rows: list[dict]) -> dict:
    """Per-case variation of valid latency readings, within each GPU and across GPU means.

    One run's sigma is estimated as the root sum of squares of the variation between GPU
    means and the largest variation within one GPU: a run on a random GPU carries both.
    """
    readings: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    eager: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("kernel_latency_ms") is None:
            continue
        readings[row["case"]][row.get("gpu_uuid") or "unknown"].append(row["kernel_latency_ms"])
        if row.get("eager_ms") is not None:
            eager[row["case"]].append(row["eager_ms"])
    report = {}
    for case, by_gpu in sorted(readings.items()):
        means = [statistics.fmean(values) for values in by_gpu.values()]
        within = max((_cv(values) for values in by_gpu.values() if len(values) > 1),
                     default=0.0)
        between = _cv(means) if len(means) > 1 else 0.0
        sigma = math.hypot(between, within)
        report[case] = {
            "gpus": len(by_gpu),
            "runs": sum(len(values) for values in by_gpu.values()),
            "mean_ms": statistics.fmean(means),
            "max_within_gpu_cv": within,
            "across_gpu_cv": between,
            "across_gpu_spread": max(means) / min(means) - 1,
            "eager_cv": _cv(eager[case]) if len(eager[case]) > 1 else None,
            "run_sigma": sigma,
            "suggested_relative_noise": math.ceil(sigma / _SUGGESTION_STEP) * _SUGGESTION_STEP,
        }
    return report


def _print_report(report: dict) -> None:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.2f}%"

    print(f"current RELATIVE_NOISE = {RELATIVE_NOISE}")
    for case, item in report.items():
        print(f"{case}: {item['runs']} runs on {item['gpus']} GPU(s), "
              f"{item['mean_ms']:.3f} ms; within-GPU CV <= {pct(item['max_within_gpu_cv'])}, "
              f"across-GPU CV {pct(item['across_gpu_cv'])} "
              f"(spread {pct(item['across_gpu_spread'])}), "
              f"eager CV {pct(item['eager_cv'])}; run sigma {pct(item['run_sigma'])}, "
              f"suggested relative_noise {item['suggested_relative_noise']:.3f}")
    if len(report) > 1:
        print("suggested for all cases: "
              f"{max(item['suggested_relative_noise'] for item in report.values()):.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="repeat the benchmark measurement in fresh processes")
    run.add_argument("--case", required=True)
    run.add_argument("--submission", type=Path, required=True)
    run.add_argument("--runs", type=int, default=10)
    run.add_argument("--output", type=Path, required=True)
    once = commands.add_parser("once", help=argparse.SUPPRESS)
    once.add_argument("--case", required=True)
    once.add_argument("--submission", type=Path, required=True)
    once.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("summarize", help="summarize result files")
    report.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.command == "once":
        row = measure_once(args.case, args.submission)
        with args.output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps({key: row[key] for key in ("case", "status", "kernel_latency_ms",
                                                    "eager_ms", "wall_s")}))
        return 0
    if args.command == "run":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for index in range(args.runs):
            # A fresh process per run, as VibeSys starts one per benchmark.
            done = subprocess.run(
                [sys.executable, "-m", __spec__.name, "once", "--case", args.case,
                 "--submission", str(args.submission), "--output", str(args.output)],
                check=False)
            if done.returncode != 0:
                print(f"run {index + 1} failed with exit code {done.returncode}",
                      file=sys.stderr)
        return 0
    rows = [json.loads(line) for path in args.results
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _print_report(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
