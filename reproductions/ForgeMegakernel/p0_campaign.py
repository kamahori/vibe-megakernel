"""Run the Forge edit/gate/review loop on independent MegaBench P0 cases.

This is a pipeline transfer experiment, not a paper milestone verdict: the
current P0 fixtures use synthetic weights and the MegaBench correctness gate.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from megabench.cases import select_cases
from megabench.harness.runner import _contract_digest

from .campaign import _digest, _json
from .milestones import MILESTONES, MILESTONE_CARDS, MILESTONE_GUIDANCE
from .p0_workspace import make_workspace


ROOT = Path(__file__).resolve().parents[2]
METHOD = "forge-codex-gpt6sol"


def _sources(workspace: Path) -> dict[str, bytes]:
    """The only files a P0 coding agent may change in its workspace."""
    result: dict[str, bytes] = {}
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if relative.parts[0] in {".git", ".venv", "reports", ".cache",
                                 ".torch_extensions", "__pycache__"}:
            continue
        if any(part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"unexpected workspace symlink: {relative}")
        if not path.is_file():
            continue
        if (len(relative.parts) == 1 and
                (relative.name in {"submission.py", "kernel.py", "notes.md"} or
                 relative.suffix in {".cu", ".cuh", ".cpp", ".h"})) or (
                     relative.parts[0] == "src" and
                     relative.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}):
            result[str(relative)] = path.read_bytes()
    return result


def _protected(workspace: Path) -> str:
    digest = hashlib.sha256()
    source_names = _sources(workspace).keys()
    venv = workspace / ".venv"
    digest.update(b".venv\0" + (os.readlink(venv).encode() if venv.is_symlink() else b"absent"))
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if relative.parts[0] in {".git", ".venv", "reports", ".cache",
                                 ".torch_extensions", "__pycache__"}:
            continue
        if any(part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"unexpected workspace symlink: {relative}")
        if path.is_file() and str(relative) not in source_names:
            digest.update(str(relative).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _diff(before: dict[str, bytes], after: dict[str, bytes]) -> str:
    lines: list[str] = []
    for name in sorted(before.keys() | after.keys()):
        if before.get(name) == after.get(name):
            continue
        left = before.get(name, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        right = after.get(name, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        lines.extend(difflib.unified_diff(left, right, fromfile=f"accepted/{name}",
                                          tofile=f"candidate/{name}"))
    return "".join(lines)


def _source_digest(sources: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(sources.items()):
        digest.update(name.encode() + b"\0" + content)
    return digest.hexdigest()


def _restore(workspace: Path, before: dict[str, bytes]) -> None:
    for name in _sources(workspace).keys() - before.keys():
        (workspace / name).unlink()
    for name, contents in before.items():
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)


def _run(argv: list[str], cwd: Path, stdin: str, env: dict[str, str],
         timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True,
                          text=True, env={**os.environ, **env}, timeout=timeout)


def _head(workspace: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace,
                                   text=True).strip()


def run_p0_case(case_id: str, campaign_dir: Path, *, rounds: int = 1,
                codex: str = "codex", timeout: int = 3600,
                device: str = "cuda:0") -> list[dict[str, Any]]:
    """Run one case in one fresh workspace; preserve every round's evidence."""
    case = select_cases("p0", [case_id])[0]
    if case.suite != "p0" or not case.ready:
        raise ValueError("case must be a ready P0 cell")
    if rounds < 1 or timeout < 1:
        raise ValueError("rounds and timeout must be positive")
    campaign_dir = campaign_dir.resolve()
    if campaign_dir.exists():
        state = json.loads((campaign_dir / "state.json").read_text())
        workspace = campaign_dir / "workspace"
        if state.get("case_id") != case_id or state.get("method") != METHOD:
            raise ValueError("campaign case or method differs from requested run")
        if "accepted_source_sha256" not in state or "protected_sha256" not in state:
            clean = subprocess.check_output(["git", "status", "--porcelain"],
                                            cwd=workspace, text=True)
            if clean:
                raise ValueError("legacy campaign workspace is not clean")
            state["accepted_source_sha256"] = _source_digest(_sources(workspace))
            state["protected_sha256"] = _protected(workspace)
            _json(campaign_dir / "state.json", state)
        if (not workspace.is_dir() or _head(workspace) != state["accepted_commit"] or
                _source_digest(_sources(workspace)) != state["accepted_source_sha256"] or
                _protected(workspace) != state["protected_sha256"]):
            raise ValueError("accepted workspace differs from campaign state")
    else:
        campaign_dir.mkdir(parents=True)
        workspace = make_workspace(case_id, campaign_dir / "workspace")
        initial_head = _head(workspace)
        state = {"case_id": case_id, "method": METHOD, "next_round": 1,
                 "accepted_commit": initial_head,
                 "accepted_source_sha256": _source_digest(_sources(workspace)),
                 "protected_sha256": _protected(workspace),
                 "last_gate_summary": None}
        _json(campaign_dir / "state.json", state)
    if device.startswith("cuda") and not torch.cuda.is_available():
        record = {"case_id": case_id, "status": "abstain", "reason": "CUDA not visible",
                  "agent_started": False, "rounds": 0}
        if not (campaign_dir / "preflight.json").exists():
            _json(campaign_dir / "preflight.json", record)
        return [record]

    records: list[dict[str, Any]] = []
    contract_sha256 = _contract_digest()
    pipeline_sha256 = _digest(Path(__file__).parent)
    for _ in range(rounds):
        run_id = f"{state['next_round']:04d}"
        round_dir = campaign_dir / "rounds" / run_id
        round_dir.mkdir(parents=True, exist_ok=False)
        before = _sources(workspace)
        protected = _protected(workspace)
        prompt = {
            "milestone_matrix": [{"name": name, "requirement": card}
                                 for name, card in zip(MILESTONES, MILESTONE_CARDS)],
            "current_milestone": "M5 (MegaBench adaptation)",
            "current_requirement": MILESTONE_CARDS[5],
            "current_guidance": MILESTONE_GUIDANCE[5],
            "adaptation_guidance": (
                "This P0 case uses synthetic weights and the official MegaBench gate. "
                "Make the candidate correct and within its launch budget first. "
                "If the previous gate passed, use its CUDA-event latency and "
                "speedup against the baseline to choose one performance change. "
                "A correct one-launch kernel can still be much too slow: inspect "
                "launch geometry, work per CTA, serial loops, redundant weight "
                "or cache reads, and idle SMs. Preserve correctness and compare "
                "the next measured result. MegaBench does not establish Forge "
                "checkpoint, precision, traffic, MBU, or M6-M9 milestones."
            ),
            "task": (workspace / "TASK.md").read_text(),
            "case": case.to_dict(),
            "last_gate_summary": state["last_gate_summary"],
            "rules": ("Edit only submission.py, kernel.py, notes.md, root CUDA/C++ helpers, "
                      "or source files under src/. Use ./check_candidate for local feedback. "
                      "The trusted MegaBench evaluation runs after the edit. "
                      "A MegaBench pass is a P0 development pass, not a Forge paper milestone pass."),
            "candidate_dir": str(workspace),
            "agent_report": str(round_dir / "agent_report.json"),
        }
        _json(round_dir / "prompt.json", prompt)
        env = {"FORGE_CANDIDATE_DIR": str(workspace),
               "FORGE_ROUND_DIR": str(round_dir),
               "FORGE_AGENT_REPORT": str(round_dir / "agent_report.json"),
               "FORGE_CODEX_TIMEOUT_SECONDS": str(timeout),
               "PYTHONPATH": str(ROOT)}
        record: dict[str, Any] = {"case_id": case_id, "run_id": run_id,
                                  "agent_model": "gpt-6-sol", "decision": "revert",
                                  "gate_status": "not_run", "review_status": "not_run"}
        try:
            agent = _run([sys.executable, "-m", "reproductions.ForgeMegakernel.codex_adapter",
                          "edit", "--codex", codex], workspace, json.dumps(prompt), env, timeout)
            (round_dir / "agent.stdout").write_text(agent.stdout)
            (round_dir / "agent.stderr").write_text(agent.stderr)
            if agent.returncode:
                raise RuntimeError(f"agent exited {agent.returncode}")
            if (_protected(workspace) != protected or _head(workspace) != state["accepted_commit"] or
                    _contract_digest() != contract_sha256 or _digest(Path(__file__).parent) != pipeline_sha256):
                raise RuntimeError("agent changed protected workspace files or git history")
            report = json.loads((round_dir / "agent_report.json").read_text())
            if not isinstance(report.get("hypothesis"), str) or not report["hypothesis"].strip():
                raise ValueError("agent did not report a hypothesis")
            record["hypothesis"] = report["hypothesis"]
            after = _sources(workspace)
            diff = _diff(before, after)
            (round_dir / "diff.patch").write_text(diff)
            if not diff:
                raise ValueError("agent made no source change")
            record["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
            record["candidate_source_sha256"] = _source_digest(after)
            for name, content in after.items():
                destination = round_dir / "candidate" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            result_path = round_dir / "megabench.jsonl"
            build_cache = round_dir / "gate_torch_extensions"
            command = [sys.executable, "-m", "megabench", "evaluate", "--case", case_id,
                       "--submission", str(workspace / "submission.py"),
                       "--method", METHOD, "--session-id", f"{campaign_dir.name}-{case_id}",
                       "--device", device, "--timeout", str(timeout),
                       "--output", str(result_path)]
            gate = _run(command, ROOT, "", {
                **env, "TORCH_EXTENSIONS_DIR": str(build_cache),
            }, timeout + 60)
            (round_dir / "gate.stdout").write_text(gate.stdout)
            (round_dir / "gate.stderr").write_text(gate.stderr)
            if not result_path.is_file():
                raise RuntimeError(f"MegaBench evaluator exited {gate.returncode}")
            if (_protected(workspace) != protected or _sources(workspace) != after or
                    _contract_digest() != contract_sha256 or _digest(Path(__file__).parent) != pipeline_sha256):
                raise RuntimeError("gate changed workspace source or protected files")
            result = json.loads(result_path.read_text().strip())
            session_id = f"{campaign_dir.name}-{case_id}"
            if (result.get("case") != case.to_dict() or
                    result.get("contract_sha256") != contract_sha256 or
                    result.get("session") != {"id": session_id, "method": METHOD,
                                              "case_id": case_id}):
                raise ValueError("MegaBench result does not match the candidate and pinned contract")
            status = result.get("status", "invalid")
            if (status == "ok_provisional" or result.get("submission_sha256") is not None) and (
                    result.get("submission_sha256") != hashlib.sha256(
                        after["submission.py"]).hexdigest()):
                raise ValueError("MegaBench result does not match the candidate submission")
            record["gate_status"] = status
            measured = {"status": status, "reason": result.get("reason"),
                        "correctness": result.get("correctness"),
                        "launch_audit": result.get("launch_audit"),
                        "candidate_timing": result.get("candidate_timing"),
                        "speedup_vs_best_baseline_cuda_event":
                            result.get("speedup_vs_best_baseline_cuda_event")}
            state["last_gate_summary"] = measured
            if status == "ok_provisional":
                manifest = {name: hashlib.sha256(content).hexdigest()
                            for name, content in sorted(after.items())}
                review_input = {"run_id": run_id, "case_id": case_id, "diff": diff,
                                "gate": result, "source_manifest": manifest,
                                "source_digest": record["candidate_source_sha256"],
                                "fresh_build_cache": str(build_cache)}
                review = _run([sys.executable, "-m", "reproductions.ForgeMegakernel.codex_adapter",
                               "review", "--codex", codex], workspace,
                              json.dumps(review_input), env, timeout)
                (round_dir / "review.stdout").write_text(review.stdout)
                (round_dir / "review.stderr").write_text(review.stderr)
                if review.returncode:
                    raise RuntimeError(f"reviewer exited {review.returncode}")
                if (_protected(workspace) != protected or _sources(workspace) != after or
                        _contract_digest() != contract_sha256 or _digest(Path(__file__).parent) != pipeline_sha256):
                    raise RuntimeError("reviewer changed workspace source or protected files")
                verdict = json.loads(review.stdout)
                if verdict.get("run_id") != run_id or verdict.get("status") not in {"pass", "fail", "abstain"}:
                    raise ValueError("reviewer returned an invalid verdict")
                _json(round_dir / "review.json", verdict)
                record["review_status"] = verdict["status"]
                if verdict["status"] == "pass":
                    subprocess.run(["git", "add", "-A", "--", "."], cwd=workspace, check=True)
                    subprocess.run(["git", "-c", "user.name=ForgeMegakernel",
                                    "-c", "user.email=forge@example.invalid", "commit", "-q",
                                    "-m", f"Keep {case_id} round {run_id}"],
                                   cwd=workspace, check=True)
                    state["accepted_commit"] = _head(workspace)
                    state["accepted_source_sha256"] = _source_digest(after)
                    record["decision"] = "keep"
        except (OSError, ValueError, KeyError, RuntimeError,
                subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            record["error"] = str(exc)
        if record["decision"] == "revert":
            if _protected(workspace) == protected and _head(workspace) == state["accepted_commit"]:
                _restore(workspace, before)
            else:
                record["error"] = "protected workspace changed; manual inspection required"
        state["next_round"] += 1
        _json(round_dir / "round.json", record)
        _json(campaign_dir / "state.json", state)
        with (campaign_dir / "ledger.jsonl").open("a") as ledger:
            ledger.write(json.dumps(record, sort_keys=True) + "\n")
        records.append(record)
        if record.get("error") == "protected workspace changed; manual inspection required":
            break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="seconds allowed for each agent, gate, and review process")
    args = parser.parse_args()
    for record in run_p0_case(args.case, args.campaign_dir, rounds=args.rounds,
                              codex=args.codex, device=args.device,
                              timeout=args.timeout):
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
