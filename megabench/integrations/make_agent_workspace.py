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
from ..sol import WORK_MODELS
from .make_vibesys_case_task import make_task


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILES = {
    "dense_step": ("dense.py", "common.py", "references/qwen3.py"),
    "moe_step": ("moe.py", "common.py", "references/qwen3.py",
                 "references/moe.py"),
    "quant_step": ("gemma.py", "common.py"),
    "spec_target_step": ("eagle3.py", "common.py"),
    "gptoss_step": ("gptoss.py", "quantization.py", "common.py"),
    "hybrid_step": ("hybrid.py", "common.py"),
    "vl_decode_step": ("vision.py", "gemma.py", "common.py"),
    "spec_full_iteration": ("speculative.py", "eagle3.py", "common.py"),
    "distributed_step": ("distributed.py", "parallel.py", "common.py"),
    "deepseek_v32_step": ("frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "glm52_step": ("frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "kimi_k3_step": ("kimi.py", "frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "kimi_k3_layer": ("kimi_layer.py", "kimi.py", "frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "glm53_flash_step": ("glm53.py", "frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "tts_frame_step": ("csm.py", "common.py"),
    "vla_action_step": ("pi05.py", "vision.py", "common.py"),
    "world_frame_step": ("waypoint.py", "common.py"),
    "megamoe_layer": ("megamoe.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
}
OUTPUT_KEYS = {
    "dense_step": ("logits", "next_token", "k_write", "v_write"),
    "moe_step": ("logits", "next_token", "k_write", "v_write"),
    "quant_step": ("logits", "next_token", "k_write", "v_write"),
    "spec_target_step": ("logits", "accepted_count", "committed_count",
                         "committed_tokens", "cache_length", "target_features",
                         "k_write", "v_write"),
    "gptoss_step": ("logits", "next_token", "k_write", "v_write", "expert_ids"),
    "hybrid_step": ("logits", "next_token", "k_write", "v_write",
                    "recurrent_state", "conv_state", "cache_length"),
    "vl_decode_step": ("logits", "next_token", "k_write", "v_write"),
    "distributed_step": ("logits", "next_token", "k_write", "v_write"),
    "deepseek_v32_step": ("logits", "next_token", "kv_write", "pe_write", "index_k_write", "sparse_indices", "expert_ids"),
    "glm52_step": ("logits", "next_token", "kv_write", "kv_scale_write", "pe_write", "index_k_write", "index_scale_write", "sparse_indices", "expert_ids"),
    "kimi_k3_step": ("logits", "next_token", "kv_write", "pe_write", "recurrent_state", "conv_state", "expert_ids", "cache_length"),
    "kimi_k3_layer": ("prefix", "expert_ids"),
    "glm53_flash_step": ("logits", "next_token", "kv_write", "index_k_write", "index_gate_write", "index_scores",
                         "sparse_mask", "recurrent_state", "conv_state", "expert_ids"),
    "tts_frame_step": ("codes", "logits", "k_write", "v_write"),
    "vla_action_step": ("actions",),
    "world_frame_step": ("latent", "k_write", "v_write"),
    "megamoe_layer": ("y",),
    "spec_full_iteration": ("logits", "proposed_tokens", "tree_parents", "accepted_count",
                            "proposed_count", "committed_count", "committed_tokens",
                            "cache_length", "draft_cache_length", "target_features",
                            "draft_hidden", "k_write", "v_write", "draft_k_write", "draft_v_write"),
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
    # Keep the real dataclass: hybrid fixtures call dataclasses.replace().
    # Copy only its definition and the assigned geometry, never the catalog.
    definition = (benchmark_root / "megabench/cases.py").read_text().split(
        "CASES: tuple[Case, ...] = (", 1)[0]
    (target / "cases.py").write_text(definition + f"CASE = Case(**{case.to_dict()!r})\n")
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
    if case.gpus > 1:
        (project / "test.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(benchmark_root)!r})\n"
            "from megabench.cases import select_cases\n"
            "from megabench.harness.runner import evaluate_case\n"
            "if __name__ == '__main__':\n"
            f"    case = select_cases('all', [{case.id!r}])[0]\n"
            "    result = evaluate_case(case, Path(__file__).with_name('submission.py'), "
            "device='cuda:0', trials=3, warmup=1, reps=3)\n"
            "    assert result['status'] == 'ok_provisional', result\n")
        return
    scenarios = (("eagle3_raw", "accept_full", "accept_one")
                 if case.family in ("spec_target_step", "spec_full_iteration") else ("seeded",))
    spec_module = "speculative" if case.family == "spec_full_iteration" else "eagle3"
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
        f"            from megabench.tasks.{spec_module} import set_acceptance_scenario\n"
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
    outputs = OUTPUT_KEYS[case.family]
    if case.family == "distributed_step" and "qwen" in case.model.lower():
        outputs += ("expert_ids",)
    if case.family == "kimi_k3_layer":
        from ..tasks.kimi_layer import attention_kind
        writes = ("kv_write", "pe_write") if attention_kind(case) == "mla" else ("recurrent_state", "conv_state")
        outputs = outputs[:1] + writes + outputs[1:]
    topology = (
        f"The evaluator starts {case.gpus} rank processes with data-parallel attention and EP={case.ep}. "
        "`build(case)` receives `case['execution']`: rank, world_size, tp_rank, ep_rank, "
        "device, backend, and live tp_group/ep_group handles. Each rank decodes its own "
        f"{case.params['batch']} sequences with replicated attention weights and owns "
        f"{case.params['experts'] // case.ep} contiguous experts. Routed latent rows must reach "
        "the ranks that own their experts and return to their source rank (the reference uses "
        "all-to-all). Complete all communication inside run. "
        "The budget includes NCCL kernels and is one GPU launch per rank. "
        "Use an allocation with all required GPUs visible. Nsight capture of spawned "
        "ranks requires `--target-processes all`.\n\n" if case.family == "kimi_k3_layer" else
        f"The evaluator starts {case.gpus} rank processes with EP={case.ep}. "
        "`build(case)` receives `case['execution']`: rank, world_size, tp_rank, ep_rank, "
        f"device, backend, and live tp_group/ep_group handles. Each rank holds {case.params['tokens']} "
        f"tokens with their routing and owns {case.params['experts'] // case.ep} contiguous experts. "
        "Token rows must reach the ranks that own their experts and the expert outputs must "
        "return to their source rank (the reference uses all-to-all). Complete all "
        "communication inside run. The budget includes NCCL kernels and is one GPU launch per rank. "
        "Use an allocation with all required GPUs visible. Nsight capture of spawned "
        "ranks requires `--target-processes all`.\n\n" if case.family == "megamoe_layer" else
        f"The evaluator starts {case.gpus} rank processes, with TP={case.tp}, EP={case.ep}. "
        "`build(case)` receives `case['execution']`: rank, world_size, tp_rank, ep_rank, "
        "device, backend, and live tp_group/ep_group handles. "
        "Weights and vocabulary logits use the reference's rank-local layouts. "
        "Compressed MLA KV and DSA index state are replicated; conventional KV and KDA heads are sharded. "
        "`next_token` is the global greedy token. Complete all communication inside run. "
        "The budget includes NCCL kernels and is one GPU launch per rank. "
        "Use an allocation with all required GPUs visible. Nsight capture of spawned "
        "ranks requires `--target-processes all`.\n\n" if case.gpus > 1 else "")
    (project / "TASK.md").write_text(
        f"# {case.id}\n\n" +
        (f"Implement one {case.model} decoder layer (layer {case.params['layer']}) {case.phase} step with the "
         if case.family == "kimi_k3_layer" else
         f"Implement the routed experts of one {case.model} MoE layer, dispatch through combine, with the "
         if case.family == "megamoe_layer" else
         f"Implement one complete {case.model} {case.phase} step with the ") +
        f"public geometry in `case.json`. Read `reference/megabench/tasks/{module}.py` "
        "and its included helpers for input layouts and exact PyTorch math.\n\n"
        + interface +
        f"Return {', '.join(f'`{key}`' for key in outputs)}. "
        "Compute from current runtime weights, token, KV cache, and any draft "
        "inputs; preserve all inputs. The complete step must use at most "
        "one GPU kernel launch per rank.\n\n" + topology
        + ("Use `test.py` for local correctness. " if agent == "kernelagent"
           else "") +
        "Use `./check_candidate` for local correctness, launch auditing, "
        "and timing. " +
        "Use `./profile_candidate` with Nsight Compute for optional "
        "kernel feedback. The capture command must set `TMPDIR=/tmp`, "
        "`TMP=/tmp`, and `TEMP=/tmp` inside the agent sandbox and pass "
        "`--profile-from-start off` to capture the warmed marker window. "
        "Use `./sol_info --json` for an optimistic " +
        ("speed-of-light floor (the largest of the HBM, tensor-core and NVLink "
         "bounds) with byte and FLOP breakdowns" if case.family in WORK_MODELS
         else "HBM speed-of-light floor and byte breakdown") +
        "; pass `--result PATH` for "
        "the gap to a checked CUDA-event result where a traffic model is available. "
        "The default bandwidth is 8 TB/s per B200. These commands load the trusted benchmark outside "
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
