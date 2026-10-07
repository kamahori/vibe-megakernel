"""Grade the experimental MPK hybrid with the full oracle and relaxed launches."""

from __future__ import annotations

import argparse
import json
import secrets
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from megabench.cases import select_cases
from megabench.harness.benchmark import _audit_launches, _measure
from megabench.harness.correctness import check_trials
from megabench.probes.mpk_hybrid_baseline import build
from megabench.tasks.workloads import make_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    case = select_cases("all", [args.case])[0]
    result = {
        "case": case.to_dict(),
        "scope": "full-task MPK hybrid baseline; relaxed GPU launch count",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "trials_requested": args.trials,
    }
    try:
        build_start = time.perf_counter()
        candidate = build(case.to_dict())
        result["build_ms"] = (time.perf_counter() - build_start) * 1000
        with torch.inference_mode():
            result["correctness"], result["first_call_ms"] = check_trials(
                case, candidate, "cuda", args.trials)
            perf_inputs = make_inputs(case, secrets.randbits(32), "cuda")
            result["candidate_timing"] = _measure(
                candidate, perf_inputs, "cuda", args.warmup, args.reps)
            result["launch_audit"] = _audit_launches(
                candidate, perf_inputs, "cuda", 100000)
            result["lm_head_fallback_fraction"] = candidate.head.last_fallback_fraction
            if candidate.head.diagnostics:
                result["lm_head_diagnostics"] = candidate.head.diagnostics
        result["status"] = "pass_relaxed"
    except Exception as exc:
        result["status"] = "incorrect" if isinstance(exc, AssertionError) else "error"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key in ("status", "reason", "build_ms", "candidate_timing",
                                 "launch_audit")}, indent=2))
    if result["status"] != "pass_relaxed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
