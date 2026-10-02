"""Create an isolated MegaBench P0 workspace for a Forge coding round."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from megabench.cases import Case, select_cases


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILES = {
    "dense_step": ("dense.py", "references/qwen3.py"),
    "moe_step": ("moe.py", "common.py", "references/qwen3.py",
                 "references/moe.py"),
    "quant_step": ("gemma.py", "common.py"),
    "spec_target_step": ("eagle3.py", "common.py"),
}
OUTPUT_KEYS = {
    "dense_step": ("logits", "next_token", "k_write", "v_write"),
    "moe_step": ("logits", "next_token", "k_write", "v_write"),
    "quant_step": ("logits", "next_token", "k_write", "v_write"),
    "spec_target_step": ("logits", "accepted_count", "committed_count",
                         "committed_tokens", "cache_length", "target_features",
                         "k_write", "v_write"),
}


def _reference(workspace: Path, case: Case) -> str:
    target = workspace / "reference/megabench"
    files = REFERENCE_FILES[case.family]
    package_files = ["__init__.py", "tasks/__init__.py"]
    if any(name.startswith("references/") for name in files):
        package_files.append("tasks/references/__init__.py")
    for relative in (*package_files, *(f"tasks/{name}" for name in files)):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "megabench" / relative, destination)
    (target / "cases.py").write_text(
        '"""Case metadata for this reference snapshot."""\n'
        "from types import SimpleNamespace\n"
        "Case = SimpleNamespace\n"
        f"CASE = Case(**{case.to_dict()!r})\n")
    (workspace / "reference/README.md").write_text(
        "# Assigned PyTorch reference\n\n"
        "The task and its imported helpers are copied under `megabench/tasks/`. "
        "The trusted MegaBench gate lives outside this workspace.\n")
    return files[0].removesuffix(".py")


def _tools(workspace: Path, case: Case) -> None:
    python_path = ROOT / ".venv/bin/python"
    python = shlex.quote(str(python_path if python_path.is_file() else Path(sys.executable).resolve()))
    repo = shlex.quote(str(ROOT))
    submission = shlex.quote(str(workspace / "submission.py"))
    case_id = shlex.quote(case.id)
    common = ("#!/bin/sh\nset -eu\n"
              f"export PYTHONPATH={repo}\n"
              f"cd {shlex.quote(str(workspace))}\n")
    check = workspace / "check_candidate"
    check.write_text(
        common + "run_id=$(date -u +%Y%m%dT%H%M%S)-$$\n"
        f"exec {python} -m megabench evaluate --case {case_id} "
        f"--submission {submission} --method local-check "
        "--session-id \"local-$run_id\" --device cuda:0 "
        "--trials 3 --warmup 1 --reps 3 --timeout 1200 "
        "--output \"reports/check-$run_id.jsonl\" \"$@\"\n")
    check.chmod(0o755)
    profile = workspace / "profile_candidate"
    profile.write_text(
        common + f"exec {python} -m megabench.integrations.ncu_profile_case "
        f"--case {case_id} --submission {submission} \"$@\"\n")
    profile.chmod(0o755)


def make_workspace(case_id: str, output: Path) -> Path:
    """Create a new single-case source surface without copying the trusted gate."""
    case = select_cases("p0", [case_id])[0]
    if not case.ready or case.family not in REFERENCE_FILES:
        raise ValueError(f"unsupported P0 case: {case_id}")
    workspace = output.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / ".gitignore").write_text(
        ".venv/\nreports/\n.cache/\n.torch_extensions/\n"
        "__pycache__/\n*.log\n*.ncu-rep\n")
    shared_venv = ROOT / ".venv"
    if shared_venv.is_dir():
        (workspace / ".venv").symlink_to(shared_venv, target_is_directory=True)
    (workspace / "reports").mkdir()
    (workspace / "submission.py").write_text(
        "def build(case: dict):\n    raise NotImplementedError(case['id'])\n")
    module = _reference(workspace, case)
    (workspace / "case.json").write_text(json.dumps(case.to_dict(), indent=2) + "\n")
    (workspace / "TASK.md").write_text(
        f"# {case.id}\n\n"
        f"Implement one complete {case.model} {case.phase} step using the geometry "
        "in `case.json`. Read "
        f"`reference/megabench/tasks/{module}.py` and its included helpers for "
        "input layouts and PyTorch reference math.\n\n"
        "Edit `submission.py` and your own helper files. Export "
        "`build(case: dict) -> run(inputs: dict) -> outputs: dict`. "
        f"Return {', '.join(f'`{key}`' for key in OUTPUT_KEYS[case.family])}. "
        "Compute from the current runtime weights, token, KV cache, and any "
        "draft inputs; preserve all inputs. The complete step must use at most "
        "one GPU kernel launch.\n\n"
        "Use `./check_candidate` for local correctness, launch auditing, and "
        "timing, and `./profile_candidate` for optional Nsight Compute feedback. "
        "The independent campaign gate runs after the coding round.\n")
    _tools(workspace, case)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(["git", "-c", "user.name=MegaBench",
                    "-c", "user.email=megabench@example.invalid", "commit", "-q",
                    "-m", f"Prepare {case_id} Forge workspace"],
                   cwd=workspace, check=True)
    return workspace
