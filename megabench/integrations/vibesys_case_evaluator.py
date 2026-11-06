"""VibeSys adapter for exactly one assigned MegaBench case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dataclasses import replace

from ..cases import load_regimes, select_cases
from ..harness.correctness import check_trials
from ..harness.runner import _load_submission, evaluate_case


def pinned_score(result: dict, pin_dir: Path | None, key: dict) -> tuple[float, dict | None]:
    """The progress score against one best-baseline time pinned per run.

    The baselines are re-measured at each evaluation and drift between rounds,
    so a faster kernel could score below a slower one measured earlier. The
    first successful evaluation in *pin_dir* writes its measured best baseline
    (the fastest of eager, graph and compiled); every later evaluation with the
    same *key* scores against that value. Without *pin_dir* the score uses the
    measured baseline, as before.
    """
    if result.get("status") != "ok_provisional":
        return 0.0, None
    measured = result["speedup_vs_best_baseline_cuda_event"]
    if pin_dir is None:
        return 1.0 + measured / (1.0 + measured), None
    cand = result["candidate_timing"]["cuda_event_p50_ms"]
    measured_best = measured * cand
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = Path(pin_dir) / f"score-baseline-{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    pinned_now = False
    if not path.is_file():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"key": key, "best_baseline_cuda_event_p50_ms": measured_best},
                                  sort_keys=True, default=str))
        try:
            os.link(tmp, path)  # fails if another evaluation pinned first
            pinned_now = True
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
    pinned = json.loads(path.read_text())["best_baseline_cuda_event_p50_ms"]
    speedup = pinned / cand
    info = {"pinned_best_baseline_cuda_event_p50_ms": pinned,
            "measured_best_baseline_cuda_event_p50_ms": measured_best,
            "speedup_vs_pinned_baseline_cuda_event": speedup,
            "pinned_now": pinned_now, "file": str(path)}
    return 1.0 + speedup / (1.0 + speedup), info


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
    pin_dir = os.environ.get("MEGABENCH_BASELINE_CACHE")
    key = {"case": case.to_dict()}
    if pin_dir and result.get("status") == "ok_provisional":
        import torch

        key["gpu"] = torch.cuda.get_device_name(0)
    score, pin = pinned_score(result, Path(pin_dir) if pin_dir else None, key)
    if pin is not None:
        result["score_baseline"] = pin
    output = benchmark_output(case_id, result, score)
    with args.output_json.open("x", encoding="utf-8") as file:
        json.dump(output, file)
    print(json.dumps({"case": case_id, "progress_score": score,
                      "status": result["status"]}))
    return 0
