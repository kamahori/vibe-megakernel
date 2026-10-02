"""Fresh Codex CLI editor and reviewer adapters for a Forge campaign.

The requested model is pinned to gpt-6-sol. This adapter does not run a gate;
the campaign invokes a separate trusted gate after the editor exits.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EDITOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["hypothesis"],
    "properties": {"hypothesis": {"type": "string"}},
}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "summary"],
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail", "abstain"]},
        "summary": {"type": "string"},
    },
}

EDITOR_INSTRUCTIONS = """You are one fresh ForgeMegakernel coding round. The JSON below is the
campaign context; inspect the candidate source and task files before editing.

Work in this order:
1. Read the current milestone requirement and guidance. Treat earlier milestones
   as invariants. For a MegaBench adaptation, follow adaptation_guidance and do
   not interpret a MegaBench pass as a Forge paper milestone pass.
2. Read last_gate_summary if present. Separate measured values and failed checks
   from the gate's prose. If correctness or a structural check failed, locate the
   first failing operator, layer, lane, or schedule edge. If those checks passed
   but latency or MBU is poor, use the measured bottleneck to choose one change
   to work distribution, dependencies, or memory staging. If there is no prior
   measurement, establish the simplest candidate for the current milestone.
3. State one falsifiable hypothesis: the source change, why it addresses that
   specific gate result, and which gate metric or check should improve. Make a
   focused edit in the candidate directory. Keep the interface, reference,
   evaluator, gate, campaign rules, and protected files untouched. Do not add a
   special case for a gate input or replace model work with precomputed output.
4. Run relevant local build or checks if available. A local 'CUDA not visible'
   result is an abstention, not evidence of correctness or speed. The campaign
   runs its independent GPU gate after this round. Never invent latency, MBU,
   profiler counters, correctness, or milestone status.

Return JSON with a nonempty hypothesis describing the edit and predicted
observable effect. The campaign records the diff and independently measured
result; you do not decide whether the round is kept.
"""

REVIEW_INSTRUCTIONS = """You are a fresh, read-only reviewer of one passing
ForgeMegakernel development round. Independently inspect the candidate source,
the complete diff, the gate result, source manifest when supplied, and run IDs.

Check that edits stay within the allowed candidate surface; that the code path
measured by the gate actually includes the claimed change, including compiled
helpers and active preprocessor branches; and that the candidate does not
special-case gate inputs, bypass model work, or alter the evaluator. Compare the
claim with the gate's measured correctness, launch, latency, and other available
metrics. A gate pass alone does not prove the claimed optimization. The official
gate may have run on a GPU even if the editor's sandbox reported CUDA unavailable.

Return pass only when the available diff and evidence support the round. Return
fail for a demonstrated defect. Return abstain when a material claim cannot be
verified from the supplied evidence, and name the missing evidence in summary.
Do not edit files. Return JSON with status and a short, specific summary.
"""


def _invoke(role: str, request: dict[str, object], codex: str) -> dict[str, object]:
    candidate = Path(os.environ["FORGE_CANDIDATE_DIR"]).resolve()
    round_dir = Path(os.environ["FORGE_ROUND_DIR"]).resolve()
    if not candidate.is_dir() or not round_dir.is_dir():
        raise ValueError("candidate and round directories must exist")
    if role == "edit":
        prompt = EDITOR_INSTRUCTIONS + "\nCampaign context:\n" + json.dumps(request, indent=2)
        schema, sandbox = EDITOR_SCHEMA, "workspace-write"
    else:
        prompt = REVIEW_INSTRUCTIONS + "\nRound evidence:\n" + json.dumps(request, indent=2)
        schema, sandbox = REVIEW_SCHEMA, "read-only"
    with tempfile.TemporaryDirectory(prefix="forge_codex_") as temporary:
        schema_path = Path(temporary) / "schema.json"
        output_path = Path(temporary) / "result.json"
        schema_path.write_text(json.dumps(schema))
        command = [
            codex, "exec", "--model", "gpt-6-sol", "--sandbox", sandbox,
            "--cd", str(candidate), "--skip-git-repo-check", "--ephemeral",
            "--output-schema", str(schema_path),
            "--output-last-message", str(output_path), "-",
        ]
        process = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                 cwd=candidate, check=False,
                                 timeout=int(os.environ.get("FORGE_CODEX_TIMEOUT_SECONDS", "3600")))
        (round_dir / f"codex_{role}.stdout").write_text(process.stdout)
        (round_dir / f"codex_{role}.stderr").write_text(process.stderr)
        if process.returncode:
            raise RuntimeError(f"Codex {role} process exited {process.returncode}")
        response = json.loads(output_path.read_text())
    if not isinstance(response, dict):
        raise ValueError("Codex output must be a JSON object")
    if role == "edit":
        if not isinstance(response.get("hypothesis"), str) or not response["hypothesis"].strip():
            raise ValueError("Codex editor omitted its hypothesis")
        Path(os.environ["FORGE_AGENT_REPORT"]).write_text(json.dumps(response) + "\n")
    else:
        if response.get("status") not in {"pass", "fail", "abstain"}:
            raise ValueError("Codex review omitted a valid status")
        if not isinstance(response.get("summary"), str):
            raise ValueError("Codex review omitted a summary")
        response["run_id"] = request["run_id"]
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("edit", "review"))
    parser.add_argument("--codex", default="codex")
    args = parser.parse_args()
    request = json.load(sys.stdin)
    result = _invoke(args.role, request, args.codex)
    if args.role == "review":
        print(json.dumps(result))


if __name__ == "__main__":
    main()
