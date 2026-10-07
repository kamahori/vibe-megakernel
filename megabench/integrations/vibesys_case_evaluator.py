"""VibeSys adapter for exactly one assigned MegaBench case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cases import select_cases
from ..harness.correctness import check_trials
from ..harness.runner import _load_submission, evaluate_case


def main(case_id: str, argv: list[str] | None = None) -> int:
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
    speedup = result.get("speedup_vs_best_baseline_cuda_event")
    score = 1.0 + speedup / (1.0 + speedup) if result["status"] == "ok_provisional" else 0.0
    output = {"case": case_id, "progress_score": score, "result": result}
    with args.output_json.open("x", encoding="utf-8") as file:
        json.dump(output, file)
    print(json.dumps({"case": case_id, "progress_score": score,
                      "status": result["status"]}))
    return 0
