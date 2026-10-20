"""Run the Trainium harness end to end on development geometries.

Run inside the Neuron image on a trn2 host, from the repository root:

    python -m megabench.verify_neuron [--full] [--case ID ...]

Without ``--full`` each P0 TP4 case shrinks to two layers. The eager TP
baseline is evaluated as a submission: it must pass correctness and fail the
launch audit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .cases import select_cases
from .harness.neuron import evaluate_neuron

SUBMISSION = Path(__file__).parent / "examples" / "neuron_tp_baseline_submission.py"


def development(case):
    return replace(case, params=case.params | {"layers": 2})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs="*")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    cases = select_cases("p0", args.case, backend="neuron")
    failed = 0
    for case in cases:
        case = case if args.full else development(case)
        result = evaluate_neuron(case, SUBMISSION, trials=args.trials, warmup=2, reps=5)
        audit = result.get("launch_audit", {}).get("per_rank", [{}])[0]
        summary = {"case": case.id, "layers": case.params["layers"], "status": result["status"],
                   "reason": str(result.get("reason", ""))[:600],
                   "trials": len(result.get("correctness", {}).get("trials", [])),
                   "baseline_unavailable": result.get("baseline_unavailable"),
                   "candidate_p50_ms": result.get("candidate_timing", {}).get("cuda_event_p50_ms"),
                   "audit_nki_calls": audit.get("nki_kernel_calls"),
                   "audit_other_ops": audit.get("other_device_op_count")}
        print(json.dumps(summary), flush=True)
        failed += result["status"] != "non_megakernel"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
