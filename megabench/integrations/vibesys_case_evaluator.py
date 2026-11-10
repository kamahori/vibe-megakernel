"""VibeSys adapter for exactly one assigned MegaBench case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataclasses import replace

from ..cases import load_regimes, select_cases
from ..harness.correctness import check_trials
from ..harness.runner import _load_submission, evaluate_case


# The metric VibeSys optimizes: the candidate's own CUDA-event p50 step time, lower is better.
# It names no field inside ``result``, because VibeSys searches nested JSON for the metric
# when the top level lacks it, and a failed candidate must have no reading at all.
METRIC = "kernel_latency_ms"
# The relative standard deviation of one benchmark run (objectives.toml [pareto]
# relative_noise). VibeSys uses it twice: a checkpoint is retained only when it beats the best
# by more than this fraction, and the framework verdict on an idea treats it as the floor of
# one run's sigma and asks for a gain beyond 2 * sigma * sqrt(1/m + 1/n) over m candidate and n
# base runs (2.8% at one run a side, 2.0% at two). It is a sigma, not the widest gap: kernel
# time varies by under 0.25% (CV) on one GPU, and its per-GPU means vary by 0.16% (CSM),
# 0.28% (pi0.5) and 0.96% (Waypoint) across the 8 B200s of one node (2026-11-07), which matters
# because a candidate and its base can land on different GPUs.
RELATIVE_NOISE = 0.01
# How the benchmark mode measures; vibesys_noise_calibration repeats exactly this.
BENCHMARK_SETTINGS = {"trials": 3, "warmup": 1, "reps": 3, "graph_baseline": False}


def kernel_latency_ms(result: dict) -> float | None:
    """The step time VibeSys compares, or None when the candidate has no valid reading.

    Only an ``ok_provisional`` result is a trusted megakernel measurement; anything else
    (incorrect, over the launch budget, unverified, an error) reports no latency, so it can
    never win on speed. The eager baseline is deliberately not involved: it varies by several
    percent between runs and would add its noise to every comparison.
    """
    if result.get("status") != "ok_provisional":
        return None
    latency = result.get("candidate_timing", {}).get("cuda_event_p50_ms")
    return float(latency) if latency is not None else None


def progress_score(result: dict) -> float:
    """The legacy saturating score, kept for reports; VibeSys no longer optimizes it."""
    speedup = result.get("speedup_vs_best_baseline_cuda_event")
    if result.get("status") != "ok_provisional" or speedup is None:
        return 0.0
    return 1.0 + speedup / (1.0 + speedup)


def benchmark_output(case_id: str, result: dict) -> dict:
    """The result file VibeSys reads back, without raw per-rank records.

    Multi-GPU results keep every rank's raw record, including the eager
    reference's launch audit, under ``ranks``. Their summaries are already in
    ``correctness`` and ``launch_audit``, so the result file omits the copies.
    """
    compact = {key: value for key, value in result.items() if key != "ranks"}
    if "regimes" in compact:  # the per-sequence contexts are already in result["case"]
        compact["regimes"] = [{key: value for key, value in row.items() if key != "contexts"}
                              for row in compact["regimes"]]
    output = {"case": case_id, "status": result.get("status")}
    latency = kernel_latency_ms(result)
    if latency is not None:
        output[METRIC] = latency
    return output | {"progress_score": progress_score(result), "result": compact}


def main(case_id: str, argv: list[str] | None = None,
         regimes_path: Path | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("accuracy", "benchmark"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    case = select_cases("all", [case_id])[0]
    if regimes_path is not None:
        case = replace(case, regimes=load_regimes(regimes_path))
    submission = args.submission.resolve()
    if not submission.is_file():
        parser.error(f"missing submission: {submission}")
    if args.mode == "accuracy":
        import torch

        try:
            if case.gpus > 1:
                result = evaluate_case(case, submission, device="cuda:0", trials=3,
                                       warmup=0, reps=1, graph_baseline=False)
                if result.get("correctness", {}).get("status") != "pass":
                    raise AssertionError(result.get("reason", result["status"]))
            else:
                # Each regime shape is built and checked on its own.
                for shape in ([sub for _, sub in case.regime_cases()]
                              if case.regimes else [case]):
                    candidate = _load_submission(submission, shape)
                    with torch.inference_mode():
                        check_trials(shape, candidate, "cuda:0", 3)
        except Exception as exc:
            print(json.dumps({"case": case_id, "status": "fail",
                              "reason": f"{type(exc).__name__}: {exc}"}))
            return 1
        print(json.dumps({"case": case_id, "status": "pass"}))
        return 0
    if args.output_json is None:
        parser.error("benchmark requires --output-json")
    try:
        result = evaluate_case(case, submission, device="cuda:0", **BENCHMARK_SETTINGS)
    except Exception as exc:
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    output = benchmark_output(case_id, result)
    with args.output_json.open("x", encoding="utf-8") as file:
        json.dump(output, file)
    summary = {key: output[key] for key in ("case", "status", METRIC, "progress_score")
               if key in output}
    if METRIC not in output:  # VibeSys rejects the run for the missing metric; say why
        summary["reason"] = result.get("reason") or (
            f"status {result.get('status')!r} is not ok_provisional, so the candidate has "
            f"no {METRIC} reading")
    print(json.dumps(summary))
    return 0
