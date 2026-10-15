"""VibeSys adapter for exactly one assigned MegaBench case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cases import select_cases
from ..harness.correctness import check_trials
from ..harness.runner import _load_submission, evaluate_case


def benchmark_output(case_id: str, result: dict, score: float) -> dict:
    """The result file VibeSys reads back, without raw per-rank records.

    Multi-GPU results keep every rank's raw record, including the eager
    reference's launch audit, under ``ranks``. Their summaries are already in
    ``correctness`` and ``launch_audit``, so the result file omits the copies.
    """
    compact = {key: value for key, value in result.items() if key != "ranks"}
    return {"case": case_id, "progress_score": score, "result": compact}


def progress_score(case, result: dict, mode: str = "speedup") -> float:
    """VibeSys selects rounds by this score; higher is better, 0 for failures.

    ``speedup``: 1 + s/(1+s) for the speedup s over the best baseline. It
    saturates at large s and inherits the eager baseline's timing noise.
    ``sol``: 1 + floor/candidate, the fraction of the case's speed-of-light
    floor (``megabench sol``) reached by the candidate's CUDA-event p50. It is
    linear in the goal and independent of baseline timing; cases without a
    floor fall back to ``speedup``.
    """
    if result.get("status") != "ok_provisional":
        return 0.0
    if mode == "sol":
        from ..sol import estimate_case

        floor = estimate_case(case)
        floor_ms = floor.get("floor_ms", floor.get("hbm_floor_ms"))
        candidate_ms = (result.get("candidate_timing") or {}).get("cuda_event_p50_ms")
        if floor_ms and candidate_ms:
            return 1.0 + floor_ms / candidate_ms
    elif mode != "speedup":
        raise ValueError(f"unknown score mode: {mode}")
    speedup = result.get("speedup_vs_best_baseline_cuda_event")
    return 1.0 + speedup / (1.0 + speedup)


def main(case_id: str, argv: list[str] | None = None, score_mode: str = "speedup") -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("accuracy", "benchmark"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    case = select_cases("all", [case_id])[0]
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
                candidate = _load_submission(submission, case)
                with torch.inference_mode():
                    check_trials(case, candidate, "cuda:0", 3)
        except Exception as exc:
            print(json.dumps({"case": case_id, "status": "fail",
                              "reason": f"{type(exc).__name__}: {exc}"}))
            return 1
        print(json.dumps({"case": case_id, "status": "pass"}))
        return 0
    if args.output_json is None:
        parser.error("benchmark requires --output-json")
    try:
        result = evaluate_case(case, submission, device="cuda:0", trials=3,
                               warmup=1, reps=3, graph_baseline=False)
    except Exception as exc:
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    score = progress_score(case, result, score_mode)
    output = benchmark_output(case_id, result, score)
    with args.output_json.open("x", encoding="utf-8") as file:
        json.dump(output, file)
    print(json.dumps({"case": case_id, "progress_score": score,
                      "status": result["status"]}))
    return 0
