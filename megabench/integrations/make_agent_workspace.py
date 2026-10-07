"""Create a small, single-case workspace for a kernel-writing agent."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ..cases import Case, select_cases
from .make_vibesys_case_task import make_task


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
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


def _python(benchmark_root: Path) -> Path:
    shared = benchmark_root / ".venv/bin/python"
    return shared if shared.is_file() else Path(sys.executable).resolve()


def _write_reference(project: Path, benchmark_root: Path, case: Case) -> str:
    target = project / "reference/megabench"
    files = REFERENCE_FILES[case.family]
    package_files = ["__init__.py", "tasks/__init__.py"]
    if any(name.startswith("references/") for name in files):
        package_files.append("tasks/references/__init__.py")
    for relative in package_files:
        source = benchmark_root / "megabench" / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative in files:
        source = benchmark_root / "megabench/tasks" / relative
        destination = target / "tasks" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (target / "cases.py").write_text(
        "\"\"\"Case metadata for this reference snapshot.\"\"\"\n"
        "from types import SimpleNamespace\n"
        "Case = SimpleNamespace\n"
        f"CASE = Case(**{case.to_dict()!r})\n")
    module = files[0].removesuffix(".py")
    (project / "reference/README.md").write_text(
        "# Assigned PyTorch reference\n\n"
        f"Read `megabench/tasks/{module}.py` for input generation and the "
        "PyTorch oracle. Its small helper imports are included here. "
        "`megabench.cases.CASE` contains only this case's public geometry.\n\n"
        "The reference snapshot is for inspection and local experiments. "
        "The check and profile scripts use the trusted benchmark outside "
        "this workspace.\n")
    return module


def _write_tools(project: Path, benchmark_root: Path, case: Case) -> None:
    python = shlex.quote(str(_python(benchmark_root)))
    trusted = shlex.quote(str(benchmark_root))
    submission = shlex.quote(str(project / "submission.py"))
    case_id = shlex.quote(case.id)
    common = ("#!/bin/sh\nset -eu\n"
              f"export PYTHONPATH={trusted}\n"
              f"cd {shlex.quote(str(project))}\n")
    check = project / "check_candidate"
    check.write_text(
        common + "run_id=$(date -u +%Y%m%dT%H%M%S)-$$\n"
        f"exec {python} -m megabench evaluate --case {case_id} "
        f"--submission {submission} --method local-check "
        "--session-id \"local-$run_id\" --device cuda:0 "
        "--trials 3 --warmup 1 --reps 3 --timeout 1200 "
        "--output \"reports/check-$run_id.jsonl\" \"$@\"\n")
    check.chmod(0o755)
    profile = project / "profile_candidate"
    profile.write_text(
        common + f"exec {python} -m megabench.integrations.ncu_profile_case "
        f"--case {case_id} --submission {submission} \"$@\"\n")
    profile.chmod(0o755)
    sol = project / "sol_info"
    sol.write_text(
        common + f"exec {python} -m megabench sol --case {case_id} \"$@\"\n")
    sol.chmod(0o755)


def _write_kernelagent_task(project: Path, benchmark_root: Path,
                            case: Case) -> None:
    (project / "problem.txt").write_text(
        (project / "TASK.md").read_text()
        + "\nFor KernelAgent, write `kernel.py` with "
        "`kernel_function(inputs: dict) -> dict`. Run `test.py` for local "
        "correctness. The campaign driver wraps this function as a MegaBench "
        "`build(case)` submission.\n")
    scenarios = (("eagle3_raw", "accept_full", "accept_one")
                 if case.family == "spec_target_step" else ("seeded",))
    (project / "test.py").write_text(
        "import sys\nimport torch\n"
        f"sys.path.insert(0, {str(benchmark_root)!r})\n"
        "from kernel import kernel_function\n"
        "from megabench.cases import select_cases\n"
        "from megabench.workloads import make_inputs, reference\n"
        "from megabench.runner import _compare, _audit_launches\n"
        f"case = select_cases('all', [{case.id!r}])[0]\n"
        "with torch.inference_mode():\n"
        f"    for index, scenario in enumerate({scenarios!r}):\n"
        "        inputs = make_inputs(case, 104729 + index, 'cuda:0')\n"
        "        if scenario in ('accept_full', 'accept_one'):\n"
        "            from megabench.tasks.eagle3 import set_acceptance_scenario\n"
        "            prefix = case.params['draft_depth'] if scenario == 'accept_full' else 1\n"
        "            set_acceptance_scenario(case, inputs, prefix)\n"
        "        originals = {name: value.detach().to('cpu', copy=True) "
        "for name, value in inputs.items()}\n"
        "        expected = reference(case, inputs)\n"
        "        actual = kernel_function(inputs)\n"
        "        torch.cuda.synchronize()\n"
        "        _compare(expected, actual, case, 'cuda:0')\n"
        "        assert all(torch.equal(inputs[name].cpu(), before) "
        "for name, before in originals.items())\n"
        "        audit = _audit_launches(kernel_function, inputs, 'cuda:0', 1)\n"
        "        assert audit['status'] == 'within_budget', audit\n")


def make_workspace(case_id: str, output: Path, *, agent: str = "vibesys",
                   benchmark_root: Path = BENCHMARK_ROOT) -> Path:
    if agent not in ("vibesys", "kernelagent", "plain"):
        raise ValueError(f"unsupported agent: {agent}")
    case = select_cases("all", [case_id])[0]
    if not case.ready or case.family not in REFERENCE_FILES:
        raise ValueError(f"case has no minimal reference bundle: {case_id}")
    benchmark_root = benchmark_root.resolve()
    project = output.resolve()
    project.mkdir(parents=True, exist_ok=False)
    (project / ".gitignore").write_text(
        ".venv/\nreports/\n.cache/\n.torch_extensions/\n"
        "__pycache__/\n*.log\n*.ncu-rep\n/agent.toml\n")
    shared_venv = benchmark_root / ".venv"
    if shared_venv.is_dir():
        (project / ".venv").symlink_to(shared_venv, target_is_directory=True)
    (project / "reports").mkdir()
    submission_source = (
        "def build(case: dict):\n"
        "    from kernel import kernel_function\n"
        "    return kernel_function\n"
        if agent == "kernelagent" else
        "def build(case: dict):\n    raise NotImplementedError(case['id'])\n"
    )
    (project / "submission.py").write_text(submission_source)
    module = _write_reference(project, benchmark_root, case)
    (project / "case.json").write_text(json.dumps(case.to_dict(), indent=2) + "\n")
    interface = (
        "Write `kernel.py` with `kernel_function(inputs: dict) -> dict`. "
        "The campaign driver wraps it as a MegaBench submission. "
        if agent == "kernelagent" else
        "Edit `submission.py` and your own helper files. Export "
        "`build(case: dict) -> run(inputs: dict) -> outputs: dict`. "
    )
    (project / "TASK.md").write_text(
        f"# {case.id}\n\n"
        f"Implement one complete {case.model} {case.phase} step with the "
        f"public geometry in `case.json`. Read `reference/megabench/tasks/{module}.py` "
        "and its included helpers for input layouts and exact PyTorch math.\n\n"
        + interface +
        f"Return {', '.join(f'`{key}`' for key in OUTPUT_KEYS[case.family])}. "
        "Compute from current runtime weights, token, KV cache, and any draft "
        "inputs; preserve all inputs. The complete step must use at most "
        "one GPU kernel launch.\n\n"
        + ("Use `test.py` for local correctness. " if agent == "kernelagent"
           else "") +
        "Use `./check_candidate` for local correctness, launch auditing, "
        "and timing. " +
        "Use `./profile_candidate` with Nsight Compute for optional "
        "kernel feedback. The capture command must set `TMPDIR=/tmp`, "
        "`TMP=/tmp`, and `TEMP=/tmp` inside the agent sandbox and pass "
        "`--profile-from-start off` to capture the warmed marker window. "
        "Use `./sol_info --json` for an optimistic HBM "
        "speed-of-light floor and byte breakdown; pass `--result PATH` for "
        "the gap to a checked CUDA-event result. The default assumes one "
        "B200 GPU at 8 TB/s. These commands load the trusted benchmark outside "
        "this workspace; final grading is run separately.\n")
    _write_tools(project, benchmark_root, case)
    if agent == "vibesys":
        shutil.copy2(benchmark_root / "megabench/integrations/vibesys-agent.example.toml",
                     project / "agent.toml")
        make_task(case_id, project, benchmark_root)
    elif agent == "kernelagent":
        _write_kernelagent_task(project, benchmark_root, case)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=project, check=True)
    subprocess.run(["git", "add", "-A"], cwd=project, check=True)
    subprocess.run(["git", "-c", "user.name=MegaBench",
                    "-c", "user.email=megabench@example.invalid", "commit", "-q",
                    "-m", f"Prepare {case_id} agent workspace"],
                   cwd=project, check=True)
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", choices=("vibesys", "kernelagent", "plain"),
                        default="vibesys")
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT,
                        help="trusted benchmark snapshot used by workspace tools")
    args = parser.parse_args()
    print(make_workspace(args.case, args.output, agent=args.agent,
                         benchmark_root=args.benchmark_root))


if __name__ == "__main__":
    main()
