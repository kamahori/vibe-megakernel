"""VibeSys adapter for exactly one assigned MegaBench case."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from ..cases import select_cases
from ..harness.correctness import _compare
from ..harness.runner import _load_submission, evaluate_case
from ..workloads import make_inputs, reference


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

        scenarios = ("raw", "full", "partial") if case.family == "spec_target_step" else ("seed1", "seed2")
        try:
            candidate = _load_submission(submission, case)
            with torch.inference_mode():
                for scenario in scenarios:
                    values = make_inputs(case, secrets.randbits(32), "cuda:0")
                    if scenario in ("full", "partial"):
                        from ..tasks.eagle3 import set_acceptance_scenario
                        set_acceptance_scenario(case, values,
                                                case.params["draft_depth"] if scenario == "full" else 1)
                    originals = {name: value.detach().to("cpu", copy=True)
                                 for name, value in values.items()}
                    expected = reference(case, values)
                    actual = candidate(values)
                    torch.cuda.synchronize()
                    for name, value in values.items():
                        if not torch.equal(value.cpu(), originals[name]):
                            raise AssertionError(f"candidate mutated input {name}")
                    _compare(expected, actual, case, "cuda:0")
                    del values, originals, expected, actual
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
