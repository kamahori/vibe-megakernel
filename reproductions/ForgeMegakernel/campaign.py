"""Fresh-process agent campaign with protected, milestone-gated advancement.

External commands are argv arrays, not shell snippets. The gate command must be
an independently maintained GPU evaluator that prints a JSON gate envelope.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .config import ModelConfig, accessed_bytes
from .milestones import Cell, MILESTONES, MILESTONE_CARDS, MILESTONE_GUIDANCE, check_milestone


def _json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _files(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in a candidate: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = path.read_bytes()
        elif not path.is_dir():
            raise ValueError(f"special file in candidate: {path}")
    return result


def _diff(before: Path, after: Path) -> str:
    left, right = _files(before), _files(after)
    lines: list[str] = []
    for name in sorted(left.keys() | right.keys()):
        if left.get(name) == right.get(name):
            continue
        a = left.get(name, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        b = right.get(name, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        change = list(difflib.unified_diff(a, b, fromfile=f"accepted/{name}", tofile=f"candidate/{name}"))
        if not change:
            change = [f"Binary change: {name} {hashlib.sha256(left.get(name, b'')).hexdigest()} -> "
                      f"{hashlib.sha256(right.get(name, b'')).hexdigest()}\n"]
        lines.extend(change)
    return "".join(lines)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        for name, data in sorted(_files(path).items()):
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(data)
    else:
        raise FileNotFoundError(path)
    return digest.hexdigest()


def _command(argv: list[str], cwd: Path, stdin: str, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, input=stdin, text=True, capture_output=True,
                          env={**os.environ, **env}, timeout=timeout, check=False)


def _envelope(output: str) -> dict[str, Any]:
    value = json.loads(output)
    if not isinstance(value, dict) or value.get("status") not in {"pass", "fail", "abstain"}:
        raise ValueError("gate/reviewer must print a JSON object with pass/fail/abstain status")
    return value


def _cell(spec: dict[str, Any]) -> Cell:
    return Cell(ModelConfig(**spec["model"]), spec["batch"], spec["context"],
                spec["peak_gbs"], spec["checkpoint_sha256"], spec["mbu_floors"],
                spec.get("timing_scope", "end-to-end"))


def _validate_spec(spec: dict[str, Any]) -> None:
    _cell(spec)
    for key in ("agent_command", "gate_command", "review_command"):
        if not isinstance(spec.get(key), list) or not spec[key] or not all(isinstance(x, str) and x for x in spec[key]):
            raise ValueError(f"{key} must be a nonempty argv array")
    if not isinstance(spec.get("rules"), str):
        raise ValueError("rules must be a string")
    if not isinstance(spec.get("seed"), str):
        raise ValueError("seed must name a candidate directory")
    if not isinstance(spec.get("checkpoint_path"), str):
        raise ValueError("checkpoint_path must name the pinned model file")


def run_campaign(spec_path: Path, campaign_dir: Path, rounds: int = 1) -> list[dict[str, Any]]:
    """Run new rounds without touching the seed or overwriting prior rounds."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text())
    _validate_spec(spec)
    cell = _cell(spec)
    seed = (spec_path.parent / spec["seed"]).resolve()
    checkpoint = (spec_path.parent / spec["checkpoint_path"]).resolve()
    if not checkpoint.is_file() or _digest(checkpoint) != cell.checkpoint_sha256:
        raise ValueError("checkpoint file is missing or differs from the pinned SHA-256")
    campaign_dir = campaign_dir.resolve()
    if not seed.is_dir() or campaign_dir == seed or seed in campaign_dir.parents or campaign_dir in seed.parents:
        raise ValueError("seed must be an existing directory separate from campaign_dir")
    _files(seed)
    spec_digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    state_path = campaign_dir / "state.json"
    if campaign_dir.exists():
        if not state_path.is_file():
            raise FileExistsError("campaign directory exists without state.json")
        state = json.loads(state_path.read_text())
        if state["spec_sha256"] != spec_digest:
            raise ValueError("campaign specification changed; start a new campaign")
        accepted_path = (campaign_dir / state["accepted"]).resolve()
        if campaign_dir not in accepted_path.parents or not accepted_path.is_dir():
            raise ValueError("accepted candidate path escapes or is missing from campaign")
        if _digest(accepted_path) != state["accepted_sha256"]:
            raise ValueError("accepted candidate changed since the previous round")
    else:
        campaign_dir.mkdir(parents=True)
        shutil.copytree(seed, campaign_dir / "seed", symlinks=False)
        _files(campaign_dir / "seed")
        state = {"spec_sha256": spec_digest, "accepted": "seed", "milestone": 0,
                 "accepted_sha256": _digest(campaign_dir / "seed"),
                 "next_round": 1, "last_gate_summary": None}
        _json(state_path, state)
    protected = [Path(__file__).parent.resolve(), spec_path, checkpoint]
    protected.extend((spec_path.parent / name).resolve() for name in spec.get("protected_paths", []))
    protected_before = {str(path): _digest(path) for path in protected}
    records: list[dict[str, Any]] = []
    timeout = int(spec.get("command_timeout_seconds", 3600))

    for _ in range(rounds):
        milestone = state["milestone"]
        if milestone >= len(MILESTONES):
            break
        run_id = f"{state['next_round']:04d}"
        round_dir = campaign_dir / "rounds" / run_id
        round_dir.mkdir(parents=True, exist_ok=False)
        accepted = campaign_dir / state["accepted"]
        candidate = round_dir / "candidate"
        shutil.copytree(accepted, candidate, symlinks=False)
        prompt = json.dumps({
            "milestone_matrix": [
                {"name": name, "requirement": card}
                for name, card in zip(MILESTONES, MILESTONE_CARDS)
            ],
            "current_milestone": milestone,
            "current_requirement": MILESTONE_CARDS[milestone],
            "current_guidance": MILESTONE_GUIDANCE[milestone],
            "cell": {"model": spec["model"], "batch": cell.batch, "context": cell.context,
                     "mbu_floors": dict(cell.mbu_floors), "peak_gbs": cell.peak_gbs,
                     "timing_scope": cell.timing_scope},
            "last_gate_summary": state["last_gate_summary"], "rules": spec["rules"],
            "candidate_dir": str(candidate),
            "agent_report": str(round_dir / "agent_report.json"),
        }, indent=2)
        (round_dir / "prompt.json").write_text(prompt + "\n")
        env = {"FORGE_CANDIDATE_DIR": str(candidate), "FORGE_ROUND_DIR": str(round_dir),
               "FORGE_MILESTONE": f"M{milestone}", "FORGE_RUN_ID": run_id,
               "FORGE_AGENT_REPORT": str(round_dir / "agent_report.json"),
               "FORGE_CODEX_TIMEOUT_SECONDS": str(timeout),
               "PYTHONPATH": os.pathsep.join((
                   str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", ""))),
               }
        record: dict[str, Any] = {"run_id": run_id, "milestone": f"M{milestone}",
                                  "gate_status": "not_run", "review_status": "not_run",
                                  "decision": "revert"}
        try:
            agent = _command(spec["agent_command"], candidate, prompt, env, timeout)
            (round_dir / "agent.stdout").write_text(agent.stdout)
            (round_dir / "agent.stderr").write_text(agent.stderr)
            if any(_digest(path) != protected_before[str(path)] for path in protected):
                raise RuntimeError("agent changed a protected evaluator or specification path")
            if agent.returncode:
                raise RuntimeError(f"agent exited {agent.returncode}")
            report = json.loads((round_dir / "agent_report.json").read_text())
            if not isinstance(report, dict) or not isinstance(report.get("hypothesis"), str) or not report["hypothesis"].strip():
                raise ValueError("agent report requires a hypothesis")
            record["hypothesis"] = report["hypothesis"]
            diff = _diff(accepted, candidate)
            (round_dir / "diff.patch").write_text(diff)
            record["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
            if not diff:
                raise ValueError("agent made no candidate change")
            record["candidate_sha256_before_gate"] = _digest(candidate)
            gate = _command(spec["gate_command"], candidate, json.dumps({
                "milestone": f"M{milestone}", "run_id": run_id,
                "nominal_bytes": accessed_bytes(cell.model, cell.batch, cell.context),
                "candidate_sha256": record["candidate_sha256_before_gate"],
                "checkpoint_sha256": cell.checkpoint_sha256,
                "checkpoint_path": str(checkpoint),
                "timing_scope": cell.timing_scope,
            }), env, timeout)
            (round_dir / "gate.stderr").write_text(gate.stderr)
            if any(_digest(path) != protected_before[str(path)] for path in protected):
                raise RuntimeError("gate changed a protected evaluator or specification path")
            if _digest(candidate) != record["candidate_sha256_before_gate"]:
                raise RuntimeError("gate changed candidate files")
            if gate.returncode:
                raise RuntimeError(f"gate exited {gate.returncode}")
            gate_result = _envelope(gate.stdout)
            if (gate_result.get("suite") != f"M{milestone}" or
                    gate_result.get("run_id") != run_id or
                    gate_result.get("candidate_sha256") != record["candidate_sha256_before_gate"] or
                    not isinstance(gate_result.get("details"), dict)):
                raise ValueError("gate envelope is missing the expected suite, run ID, candidate digest, or details")
            _json(round_dir / "gate.json", gate_result)
            record["gate_status"] = gate_result["status"]
            evidence = gate_result.get("metrics", {})
            if not isinstance(evidence, dict):
                raise ValueError("gate metrics must be a JSON object")
            issues, derived = check_milestone(milestone, evidence, cell)
            record["derived_metrics"] = derived
            record["gate_issues"] = issues
            state["last_gate_summary"] = {"status": gate_result["status"],
                                          "summary": gate_result.get("summary"),
                                          "metrics": evidence,
                                          "derived_metrics": derived, "issues": issues}
            if gate_result["status"] == "pass" and not issues:
                review_input = json.dumps({"run_id": run_id, "milestone": f"M{milestone}",
                                           "diff": diff, "gate": gate_result,
                                           "derived_metrics": derived})
                review = _command(spec["review_command"], candidate, review_input, env, timeout)
                (round_dir / "review.stderr").write_text(review.stderr)
                if any(_digest(path) != protected_before[str(path)] for path in protected):
                    raise RuntimeError("reviewer changed a protected evaluator or specification path")
                if _digest(candidate) != record["candidate_sha256_before_gate"]:
                    raise RuntimeError("reviewer changed candidate files")
                if review.returncode:
                    raise RuntimeError(f"reviewer exited {review.returncode}")
                review_result = _envelope(review.stdout)
                if review_result.get("run_id") != run_id:
                    raise ValueError("review result has a missing or mismatched run ID")
                _json(round_dir / "review.json", review_result)
                record["review_status"] = review_result["status"]
                if review_result["status"] == "pass":
                    record["decision"] = "keep"
                    state["accepted"] = str(candidate.relative_to(campaign_dir))
                    state["accepted_sha256"] = record["candidate_sha256_before_gate"]
                    state["milestone"] += 1
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
            record["error"] = str(exc)
        try:
            record["candidate_sha256"] = _digest(candidate)
        except (OSError, ValueError) as exc:
            record["candidate_error"] = str(exc)
        _json(round_dir / "round.json", record)
        state["next_round"] += 1
        _json(state_path, state)
        with (campaign_dir / "ledger.jsonl").open("a") as ledger:
            ledger.write(json.dumps(record, sort_keys=True) + "\n")
        records.append(record)
        if "error" in record and "protected" in record["error"]:
            break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    for record in run_campaign(args.spec, args.campaign_dir, args.rounds):
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
