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


def audit_selftest() -> int:
    """One NKI kernel per step passes the launch audit; any other device work fails it."""
    import nki
    import nki.isa as nisa
    import nki.language as nl
    import torch
    from torch_neuronx import nki_op

    from .harness.neuron import _audit

    @nki.jit
    def add(a, b):
        a_sb = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
        b_sb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_sb, src=a)
        nisa.dma_copy(dst=b_sb, src=b)
        c_sb = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_sb, data1=a_sb, data2=b_sb, op=nl.add)
        out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=c_sb)
        return out

    @nki_op("megabench_verify::add", mutates_args={})
    def add_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return add(a, b)

    compiled = torch.compile(lambda v: {"y": v["a"] + v["b"]}, backend="neuron", dynamic=False, fullgraph=True)
    inputs = {"a": torch.randn(128, 512, device="neuron:0"), "b": torch.randn(128, 512, device="neuron:0")}
    expected = {
        "nki_op": (lambda v: {"y": add_op(v["a"], v["b"])}, "within_budget"),
        "direct nki.jit": (lambda v: {"y": add(v["a"], v["b"])}, "within_budget"),
        "two nki_op calls": (lambda v: {"y": add_op(add_op(v["a"], v["b"]), v["b"])}, "launch_budget_failed"),
        "nki_op + eager add": (lambda v: {"y": add_op(v["a"], v["b"]) + 1}, "launch_budget_failed"),
        "eager only": (lambda v: {"y": v["a"] + v["b"]}, "launch_budget_failed"),
        "torch.compile": (compiled, "launch_budget_failed"),
    }
    failed = 0
    for label, (run, want) in expected.items():
        run(inputs)
        result = _audit(run, inputs, "neuron:0", 1)
        ok = result["status"] == want
        failed += not ok
        print(json.dumps({"audit": label, "status": result["status"], "expected": want, "ok": ok,
                          "nki_kernel_calls": result["nki_kernel_calls"],
                          "other_device_ops": result["other_device_ops"],
                          "compiled_regions": result["compiled_regions"]}), flush=True)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs="*")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--audit-selftest", action="store_true")
    args = parser.parse_args()
    if args.audit_selftest:
        return 1 if audit_selftest() else 0
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
