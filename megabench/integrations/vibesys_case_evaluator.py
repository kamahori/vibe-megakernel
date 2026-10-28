"""VibeSys adapter for exactly one assigned MegaBench case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataclasses import replace

from ..cases import load_regimes, select_cases
from ..harness.correctness import check_trials
from ..harness.runner import _load_submission, evaluate_case


def benchmark_output(case_id: str, result: dict, score: float) -> dict:
    """The result file VibeSys reads back, without raw per-rank records.

    Multi-GPU results keep every rank's raw record, including the eager
    reference's launch audit, under ``ranks``. Their summaries are already in
    ``correctness`` and ``launch_audit``, so the result file omits the copies.
    """
    compact = {key: value for key, value in result.items() if key != "ranks"}
    if "regimes" in compact:  # the per-sequence contexts are already in result["case"]
        compact["regimes"] = [{key: value for key, value in row.items() if key != "contexts"}
                              for row in compact["regimes"]]
    return {"case": case_id, "progress_score": score, "result": compact}


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
        result = evaluate_case(case, submission, device="cuda:0", trials=3,
                               warmup=1, reps=3, graph_baseline=False)
    except Exception as exc:
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    speedup = result.get("speedup_vs_best_baseline_cuda_event")
    score = 1.0 + speedup / (1.0 + speedup) if result["status"] == "ok_provisional" else 0.0
    output = benchmark_output(case_id, result, score)
    with args.output_json.open("x", encoding="utf-8") as file:
        json.dump(output, file)
    print(json.dumps({"case": case_id, "progress_score": score,
                      "status": result["status"]}))
    return 0
