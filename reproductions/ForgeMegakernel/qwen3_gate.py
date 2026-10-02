"""Trusted Qwen3 development gate using the independent MegaBench evaluator.

MegaBench supplies synthetic weights. This gate can establish toolchain and
development-step correctness, but never claims the paper's checkpoint oracle.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


CASE_ID = "dense-step-qwen3-06b-b1-s128"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(request: dict[str, Any], candidate: Path, round_dir: Path,
             *, python: str = sys.executable) -> dict[str, Any]:
    milestone = request["milestone"]
    run_id = request["run_id"]
    envelope: dict[str, Any] = {
        "status": "abstain", "suite": milestone, "run_id": run_id,
        "candidate_sha256": request["candidate_sha256"],
        "summary": "GPU gate not run", "metrics": {}, "details": {},
    }
    checkpoint = Path(request["checkpoint_path"]).resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != request["checkpoint_sha256"]:
        return envelope | {"status": "fail", "summary": "pinned checkpoint changed"}
    submission = candidate / "submission.py"
    for name in ("submission.py", "dense_binding.cpp", "dense_launch.cu", "dense_kernel.cu"):
        if not (candidate / name).is_file():
            return envelope | {"status": "fail", "summary": f"missing candidate source: {name}"}
    if not torch.cuda.is_available():
        return envelope | {"summary": "CUDA is not visible; no Forge or MegaBench GPU verdict"}

    if milestone == "M0":
        build_script = (
            "from pathlib import Path; import sys; "
            "from megabench.cases import select_cases; "
            "from megabench.harness.runner import _load_submission; "
            "case=select_cases('all',[sys.argv[2]])[0]; "
            "run=_load_submission(Path(sys.argv[1]),case); "
            "assert callable(run)"
        )
        process = subprocess.run([python, "-c", build_script, str(submission), CASE_ID],
                                 cwd=REPO_ROOT, text=True, capture_output=True,
                                 timeout=900, check=False)
        (round_dir / "toolchain.stdout").write_text(process.stdout)
        (round_dir / "toolchain.stderr").write_text(process.stderr)
        if process.returncode:
            return envelope | {"status": "fail", "summary": "Qwen3 CUDA extension build/import failed"}
        envelope["status"] = "pass"
        envelope["summary"] = "Qwen3 CUDA extension built and imported on a CUDA host"
        envelope["metrics"] = {"toolchain_ok": True}
        envelope["details"] = {"build_only": True, "gpu_step_executed": False}
        return envelope

    report_path = round_dir / "megabench_dense.jsonl"
    command = [
        python, "-m", "megabench", "evaluate", "--case", CASE_ID,
        "--submission", str(submission), "--method", "forge-dev-gate",
        "--session-id", f"forge-{run_id}", "--device", "cuda:0",
        "--trials", "3", "--warmup", "1", "--reps", "3",
        "--timeout", "1800", "--output", str(report_path),
    ]
    process = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True,
                             timeout=1900, check=False)
    (round_dir / "megabench.stdout").write_text(process.stdout)
    (round_dir / "megabench.stderr").write_text(process.stderr)
    if process.returncode or not report_path.is_file():
        return envelope | {"status": "fail", "summary": "MegaBench evaluator failed to produce a report"}
    records = [json.loads(line) for line in report_path.read_text().splitlines() if line.strip()]
    if len(records) != 1:
        return envelope | {"status": "fail", "summary": "MegaBench report must contain one case"}
    record = records[0]
    if record.get("case", {}).get("id") != CASE_ID:
        return envelope | {"status": "fail", "summary": "MegaBench evaluated a different case"}
    correctness = record.get("correctness", {}).get("status") == "pass"
    audit = record.get("launch_audit", {})
    launches = audit.get("gpu_kernel_count")
    event_ms = record.get("candidate_timing", {}).get("cuda_event_p50_ms")
    metrics = {
        "gpu_executed": True,
        "toolchain_ok": correctness and audit.get("status") == "within_budget",
        "megabench_correct": correctness,
        "launches_per_step": launches,
        "candidate_cuda_event_us": event_ms * 1000 if isinstance(event_ms, (int, float)) else None,
        "checkpoint_sha256": request["checkpoint_sha256"],
    }
    envelope["metrics"] = metrics
    envelope["details"] = {"megabench_status": record.get("status"),
                           "megabench_report": str(report_path),
                           "synthetic_weights": True}
    if not correctness or audit.get("status") != "within_budget":
        envelope["status"] = "fail"
        envelope["summary"] = "MegaBench correctness or launch audit failed"
    else:
        envelope["summary"] = "MegaBench development step passed; checkpoint Forge gate unavailable"
    return envelope


def main() -> None:
    request = json.load(sys.stdin)
    result = evaluate(request, Path(os.environ["FORGE_CANDIDATE_DIR"]),
                      Path(os.environ["FORGE_ROUND_DIR"]))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
