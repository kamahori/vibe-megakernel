"""Run Claude Code, Codex or opencode on one MegaBench case inside a container, under Slurm.

On this host containers run only inside Slurm jobs (rootless Podman), and Slurm
refuses CPU-only jobs, so each agent round is one GPU job: the agent's
container holds the case's GPUs for the whole round, then the round's
submission is graded in a fresh container in the same job.

Subcommands:

- ``build-image``: stage ``megabench/docker/agent.Dockerfile``, the frozen
  requirements, the egress proxy and the host's ``claude``, ``codex`` and
  ``opencode`` in one build context and build the image in a Slurm job (``--submit``).
- ``prepare``: make a new run directory: a trusted snapshot of ``megabench/``
  from one git ref (past-run docs, submissions and examples removed), a fresh
  ``plain`` workspace for the case, an empty agent ``HOME`` with the agent's
  settings, and the job scripts.
- ``smoke``: submit a short job that starts the agent container exactly as a
  round does, without a credential, and checks what it can see and reach.
- ``submit``: submit the rounds as a chain of Slurm jobs.

What the agent container sees: the workspace (read-write) and the trusted
snapshot (read-only) at their host paths, its own ``HOME``, the job's GPUs and
the image. Not the repository, the host ``HOME``, other runs, the grading state
or the credential files. It has no network of its own (``--network none``); its
only way out is an allowlist proxy (``egress_proxy.py``) that runs in the job
outside the container, reached through a bind-mounted Unix socket: by default
only the model API of the chosen agent. Web search and fetch tools are turned off in the agent's
settings, because they run on the provider's side, past the proxy.

Grading runs the trusted ``megabench evaluate`` in a container without network
on a copy of the workspace, with a baseline cache that the agent cannot see.
Nothing that the agent wrote runs on the host: the host side of the job only
copies files (``cp -a``, links not followed).

Trainium (cases of backend ``neuron``): the same flow on a host with Docker and no Slurm.
``build-image --platform neuron`` builds ``megabench/docker/neuron/agent.Dockerfile`` on the
Neuron image; each container gets ``/dev/neuron0`` and ``NEURON_RT_VISIBLE_CORES`` (the run's
NeuronCores); ``--scheduler local`` (the default without ``sbatch``) runs a round in the
foreground under a lock on the chip, so runs queue one after another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from ..cases import select_cases
from .clean_vibesys_run import DENIED_TOOLS, build_trusted

REPO = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO / "megabench/docker"
IMAGE_REPOSITORY = "localhost/megabench-agent"
NEURON_IMAGE_REPOSITORY = "localhost/megabench-agent-neuron"
NEURON_DOCKER_DIR = DOCKER_DIR / "neuron"
# The local scheduler's lock: one round (or smoke or rescore) at a time holds the accelerator.
LOCAL_LOCK = Path(os.environ.get("MEGABENCH_LOCAL_LOCK", "/tmp/megabench-accelerator.lock"))
IMAGE_VENV = "/opt/venv"
AGENT_HOME = "/home/agent"
PROXY_PORT = 3128
# Hosts each agent needs for its model API. Others are refused by the proxy.
DEFAULT_ALLOW = {
    "claude": ("api.anthropic.com",),
    "codex": ("api.openai.com", "chatgpt.com", "auth.openai.com"),
    "opencode": (),  # the provider's host, from OPENCODE_PROVIDERS
}
# OpenAI-compatible providers for opencode. Prices are USD per million tokens (from the
# provider's /v1/models on 2026-11-23); opencode uses them for the cost in its JSON events.
OPENCODE_PROVIDERS = {
    "coralbricks": {
        "name": "Coral Bricks",
        "base_url": "https://inference.coralbricks.ai/v1",
        "env": "CORALBRICKS_API_KEY",
        "default_model": "deepseek-v4.1-flash",
        "models": {
            "deepseek-v4.1-flash": {"input": 0.19, "output": 0.6, "cache_read": 0.008},
            "deepseek-v4.1-flash-fast": {"input": 0.45, "output": 1.8, "cache_read": 0.009},
            "deepseek-v4.1-flash-value": {"input": 0.005, "output": 0.57, "cache_read": 0.003},
        },
        "context": 1048576,
        "output": 65536,
        "default_effort": "high",  # the API's default is no reasoning
    },
}
CREDENTIAL_ENV = {
    "claude-oauth-token": "CLAUDE_CODE_OAUTH_TOKEN",
    "anthropic-api-key": "ANTHROPIC_API_KEY",
    "openai-api-key": "OPENAI_API_KEY",
    "codex-auth-json": None,  # copied to $HOME/.codex/auth.json at round start
    "coralbricks-api-key": OPENCODE_PROVIDERS["coralbricks"]["env"],
}
CREDENTIAL_AGENT = {"claude-oauth-token": "claude", "anthropic-api-key": "claude",
                    "openai-api-key": "codex", "codex-auth-json": "codex",
                    "coralbricks-api-key": "opencode"}
CREDENTIAL_PROVIDER = {"coralbricks-api-key": "coralbricks"}
DEFAULT_GOAL = (
    "the task in TASK.md is done: submission.py passes ./check_candidate (correct outputs, "
    "within the GPU launch budget), and either its measured step time is within 1.5x of the "
    "./sol_info floor or at least three distinct optimizations tried after the first passing "
    "version each gave less than a 2% speedup")
KICKOFF = (
    "Read TASK.md and work on it. You are in a container with the case's GPUs and no "
    "internet access; the commands in TASK.md (./check_candidate, ./profile_candidate, "
    "./timeline_candidate, ./sol_info, ./primdb) run here directly. Keep all files in this "
    "workspace and commit your progress with git as you go. A separate trusted evaluation "
    "grades submission.py after the session. The session has a hard limit of about {minutes} "
    "minutes of wall time, GPU commands included; keep a correct submission.py in place early "
    "and improve it from there.")
# opencode has no /goal: with --goal-style command the harness plays its part. Whenever
# `opencode run` stops with time left, it resumes the session with this message, until the
# agent's last reply says GOAL MET or the round's time runs out.
GOAL_CHECK = (
    "The harness is checking the goal of this session: {goal}. If the goal holds now, verify it "
    "(run ./check_candidate) and end your reply with the line GOAL MET. Otherwise do not stop: "
    "keep working toward the goal, and use ./time_left to see the time you have.")
RESUME = ("You are resuming an earlier session on this task in a new time slot. Check the state "
          "of the workspace and continue toward the goal.")
# --bare: the workspace keeps only the case's contract and its reference, and becomes a
# VibeSys task (`.vibesys/tasks/<case>/`) whose check and benchmark load the trusted
# benchmark. No tool, skill or note of this repository's method stays in it, and its git
# history starts after they are gone.
BARE_KEEP = ("TASK.md", "case.json", "submission.py", "reference")
BARE_TASK_TAIL = """\
This repository is a VibeSys task: `.vibesys/tasks/{case}/` holds its check (the outputs
against the PyTorch reference, on fresh inputs, and the launch budget) and its benchmark (the
step's kernel latency in milliseconds, lower is better). Both load a trusted, read-only copy of
the benchmark outside this repository. The machine has {gpus} NVIDIA B200 GPU(s) (sm_100a),
CUDA 13 with `nvcc` and Nsight Compute (`ncu`), and PyTorch in the default Python.

After the session a separate trusted evaluation grades `submission.py` as it stands at the head
of the branch `vibesys/result/{case}`.
"""
BARE_TASK_TAIL_NEURON = """\
This repository is a VibeSys task: `.vibesys/tasks/{case}/` holds its check (the outputs
against the PyTorch reference, on fresh inputs, and the launch budget) and its benchmark (the
step's kernel latency in milliseconds, lower is better). Both load a trusted, read-only copy of
the benchmark outside this repository. The machine is an AWS trn2.3xlarge: one Trainium2 chip
(LNC=2: four logical NeuronCores of 24 GiB HBM each), of which this task uses {cores}
(`NEURON_RT_VISIBLE_CORES={core_range}`). The default Python has PyTorch with torch-neuronx
(device `neuron`), NKI (`nki`) and `neuronx-cc`; the Neuron tools (`neuron-ls`,
`neuron-explorer`) are in `/opt/aws/neuron/bin`.

After the session a separate trusted evaluation grades `submission.py` as it stands at the head
of the branch `vibesys/result/{case}`.
"""
BARE_GITIGNORE = (".cache/\n.torch_extensions/\n__pycache__/\n*.ncu-rep\n*.nsys-rep\n"
                  "*.neff\n*.ntff\nneuronxcc-*/\n")
BARE_OBJECTIVE = """\
# {case}: a megakernel for one {model} {phase} step

This is a megakernel task. Write `submission.py`, and any helper files of your own, so that
`build(case)` returns `run(inputs)`, which computes the whole step that
`reference/megabench/tasks/{module}.py` defines, for the geometry in `case.json`, in at most
{launches} GPU kernel launch(es) per step: one kernel that runs the whole step as phases. The
score is the step's kernel latency (CUDA events, p50), lower is better. `TASK.md` states the
outputs.

Rules:

- Every output must match the PyTorch reference within the case's tolerances, on fresh random
  inputs and weights each time; integer outputs such as tokens must match exactly.
- Compute every output from the current inputs and weights on every call. Never modify an
  input.
- At most {launches} GPU kernel launch(es) per step, counted by the profiler; the check fails a
  step over the budget.
- Edit only `submission.py` and files you add. Never edit `.vibesys/`, `case.json` or
  `reference/`.
- Use the GPU(s) of this machine: {gpus} B200.
"""
BARE_OBJECTIVE_NEURON = """\
# {case}: a megakernel for one {model} {phase} step

This is a megakernel task on AWS Trainium2. Write `submission.py`, and any helper files of your
own, so that `build(case)` returns `run(inputs)`, which computes the whole step that
`reference/megabench/tasks/{module}.py` defines, for the geometry in `case.json`, in at most
{launches} NKI kernel call(s) per step and rank: one kernel that runs the whole step as phases.
The score is the step's kernel latency (Neuron device events, p50, the slowest rank), lower is
better. `TASK.md` states the outputs and how the step is split over ranks.

Rules:

- Every output must match the PyTorch reference within the case's tolerances, on fresh random
  inputs and weights each time; integer outputs such as tokens must match exactly.
- Compute every output from the current inputs and weights on every call. Never modify an
  input.
- At most {launches} NKI kernel call(s) per step and rank (a `torch_neuronx.nki_op` or `nki.jit`
  kernel), and no other device work: no eager PyTorch ops, `torch.compile` graphs or
  `torch.distributed` collectives on Neuron tensors. The check audits this.
- Edit only `submission.py` and files you add. Never edit `.vibesys/`, `case.json` or
  `reference/`.
- Use the NeuronCores of this machine: {cores} logical NeuronCore(s) of one Trainium2 chip.
"""
# Cases whose trusted inputs take minutes to build on the CPU (the Trainium grader builds them
# there): Waypoint's 128-frame history cache is ~4 min per trial, so a 3-trial check and a
# 5 x 3-trial measure need longer limits, and the benchmark times one trial per sample (the
# check still runs three correctness trials).
SLOW_CASE_LIMITS = {
    "world-frame-step-waypoint15-1b-f128-trn2-nc1": {
        "accuracy_timeout": 3000, "benchmark_timeout": 3000,
        "benchmark_settings": {"trials": 1}},
}
BARE_EVALUATOR = '''\
"""The check and the benchmark of one MegaBench case, against a trusted benchmark.

evaluate.py accuracy --submission submission.py
evaluate.py benchmark --submission submission.py --output-json PATH
"""
import argparse
import json
import os
import sys

TRUSTED = {trusted!r}
CASE = {case!r}
os.environ["PYTHONPATH"] = TRUSTED  # the evaluator's own subprocesses import it too
# Each of a measure's 5 samples is already a fresh process; no extra stability probes.
os.environ.setdefault("MEGABENCH_STABILITY_PROCESSES", "1")
sys.path.insert(0, TRUSTED)

from pathlib import Path  # noqa: E402

from megabench.cases import select_cases  # noqa: E402
from megabench.harness.runner import evaluate_case  # noqa: E402
from megabench.integrations.vibesys_case_evaluator import (  # noqa: E402
    BENCHMARK_SETTINGS, METRIC, kernel_latency_ms, stabilize)


def summary(result):
    out = {{"case": CASE, "status": result.get("status"), "reason": result.get("reason")}}
    for key in ("correctness", "launch_audit", "input_mutation"):
        if key in result:
            out[key] = result[key]
    timing = result.get("candidate_timing") or {{}}
    if "cuda_event_p50_ms" in timing:
        out["cuda_event_p50_ms"] = timing["cuda_event_p50_ms"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("accuracy", "benchmark"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    case = select_cases("all", [CASE])[0]
    submission = args.submission.resolve()
    if args.mode == "accuracy":
        result = evaluate_case(case, submission, device="cuda:0", trials=3, warmup=1, reps=1,
                               graph_baseline=False, neuron_baseline=False)
        print(json.dumps(summary(result), default=str))
        return 0 if result.get("status") == "ok_provisional" else 1
    if args.output_json is None:
        parser.error("benchmark needs --output-json")
    # On Trainium the eager baseline is slow and the metric does not use it; grading times it.
    result = evaluate_case(case, submission, device="cuda:0",
                           **dict(BENCHMARK_SETTINGS, **{bench_override}), neuron_baseline=False)
    stabilize(result, case, CASE, submission, None)
    output = summary(result)
    latency = kernel_latency_ms(result)
    if latency is not None:
        output[METRIC] = latency
    with args.output_json.open("x", encoding="utf-8") as file:
        json.dump(output, file, default=str)
    print(json.dumps(output, default=str))
    return 0 if latency is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
# The vibesys-cli placement of the agent container: this machine, with the job's GPUs.
# Trainium: vibesys-cli 0.3.0 has no Neuron resource; the task runs its jobs on this machine
# (the agent container, which holds the run's NeuronCores), one at a time.
VIBESYS_CLI_TOML_NEURON = """\
version = "0.2"

[placements.local]
kind = "local"
max_jobs = 1
"""
VIBESYS_CLI_TOML = """\
version = "0.2"

[placements.local]
kind = "local"
[placements.local.resources.gpu]
count = {gpus}
"""
VIBESYS_KICKOFF = ("/vibesys:optimize {case} --megakernel-divide-and-conquer "
                   "--budget time {budget}m --fanout {fanout}")
VIBESYS_GOAL = ("the /vibesys:optimize run on task {case} has given its final report, because "
                "its {budget}-minute time budget (from the start of this session) is spent or "
                "an external factor stopped all progress")

# ./time_left in the workspace: the agent budget left over all rounds (time between rounds,
# while the next job waits for GPUs, does not count). RUN/control/round.json is written by
# the host at the start of each round.
TIME_LEFT = """#!/bin/sh
exec python3 - "{control}/round.json" <<'PY'
import json, sys, time
r = json.load(open(sys.argv[1]))
spent = time.time() - r["start"]
slot = max(0, r["slot_seconds"] - spent) / 60
total = max(0, r["budget_seconds"] - r["used_seconds_before"] - spent) / 60
print(f"{{int(total)}} minutes left of the {{r['budget_seconds'] // 60}}-minute agent budget")
if total > slot + 1:
    print(f"this time slot ends in {{int(slot)}} minutes; the session is then interrupted and "
          "resumed later in a new slot with your files and conversation intact")
PY
"""
# Inserted into the workspace's check_candidate after run_id is set: tar the sources before
# the check, so the host can keep the exact tree of every passing check.
SNAPSHOT_LINE = (
    "mkdir -p .harness/pending && find . \\( -path ./.git -o -path ./reports -o -path ./.harness "
    "-o -path ./.venv -o -path ./.cache -o -path ./.torch_extensions -o -path ./reference "
    "-o -path ./megakernel_skills -o -path ./primbench \\) -prune -o -type f -size -50M "
    "! -name '*.so' ! -name '*.o' -print0 | tar --null -czf \".harness/pending/$run_id.tgz\" "
    "-T - 2>/dev/null || true\n")


# ---------------------------------------------------------------- build-image

def _codex_vendor() -> Path:
    """The directory of the host codex's native binary and its helpers."""
    codex = shutil.which("codex")
    if not codex:
        raise SystemExit("codex is not on PATH")
    real = Path(codex).resolve()
    if real.suffix == ".js":  # the npm wrapper: the native build is a vendored package
        found = sorted(real.parents[1].glob(
            "node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl"))
        if not found:
            raise SystemExit(f"no native codex next to {real}")
        return found[0]
    return real.parents[1] if real.parent.name == "bin" else real.parent


def _opencode_binary() -> Path:
    """The host opencode's native binary (npm installs it as bin/opencode.exe)."""
    opencode = shutil.which("opencode")
    if not opencode:
        raise SystemExit("opencode is not on PATH; install it with "
                         "`npm install -g --prefix ~/.local opencode-ai`")
    real = Path(opencode).resolve()
    with real.open("rb") as binary:
        native = binary.read(4) == b"\x7fELF"
    if not native:
        found = sorted(real.parents[2].glob("opencode-linux-x64/bin/opencode")) + sorted(
            real.parents[1].glob("node_modules/opencode-linux-x64/bin/opencode"))
        if not found:
            raise SystemExit(f"no native opencode next to {real}")
        real = found[0]
    return real


def stage_context(context: Path, platform: str = "cuda") -> str:
    """Copy everything the image build needs into *context*; return the image tag."""
    claude = shutil.which("claude")
    if not claude:
        raise SystemExit("claude is not on PATH")
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    if platform == "neuron":
        # Claude Code only: the Neuron runs use no Codex or opencode.
        shutil.copy2(NEURON_DOCKER_DIR / "agent.Dockerfile", context / "agent.Dockerfile")
        shutil.copy2(DOCKER_DIR / "egress_proxy.py", context / "egress_proxy.py")
        shutil.copy2(Path(claude).resolve(), context / "claude")
        return _tag(context, NEURON_IMAGE_REPOSITORY)
    for name in ("agent.Dockerfile", "requirements.lock", "egress_proxy.py"):
        shutil.copy2(DOCKER_DIR / name, context / name)
    shutil.copy2(Path(claude).resolve(), context / "claude")
    shutil.copytree(_codex_vendor(), context / "codex-vendor", symlinks=True)
    shutil.copy2(_opencode_binary(), context / "opencode")
    return _tag(context, IMAGE_REPOSITORY)


def _tag(context: Path, repository: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in context.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(context)).encode())
        digest.update(path.read_bytes())
    tag = f"{repository}:{digest.hexdigest()[:12]}"
    (context / "IMAGE").write_text(tag + "\n")
    return tag


def build_image(args: argparse.Namespace) -> int:
    context = args.context.resolve()
    tag = stage_context(context, args.platform)
    if args.platform == "neuron":
        # Built here with Docker (no Slurm); the check runs on the chip.
        print(f"image: {tag}\ncontext: {context}", flush=True)
        exists = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
        if exists.returncode != 0:
            subprocess.run(["docker", "build", "--build-arg", f"BASE_IMAGE={args.base_image}",
                            "-f", "agent.Dockerfile", "-t", tag, "."], check=True, cwd=context)
        subprocess.run(["docker", "run", "--rm", "--device=/dev/neuron0", tag, "bash", "-c",
                        "neuron-ls | head -8; python -c 'import torch, torch_neuronx; "
                        "print(torch.__version__, torch_neuronx.__version__)'; claude --version"],
                       check=True)
        print(f"built {tag}")
        return 0
    script = context / "build.sbatch"
    script.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(context))}\n"
        f"if podman image exists {tag}; then echo 'image exists: {tag}'; exit 0; fi\n"
        f"podman build --pull=missing -f agent.Dockerfile -t {tag} .\n"
        f"podman run --rm {tag} bash -c 'nvidia-smi -L; python -c \"import torch; "
        "print(torch.__version__, torch.cuda.is_available())\"; claude --version; "
        "codex --version; opencode --version; ncu --version | tail -1'\n"
        f"echo 'built {tag}'\n")
    script.chmod(0o755)
    command = ["sbatch", "--parsable", f"--partition={args.partition}", "--gres=gpu:1",
               f"--time={args.time}", "--job-name=megabench-agent-image",
               f"--output={context}/build-%j.out", str(script)]
    print(f"image: {tag}\ncontext: {context}")
    if not args.submit:
        print("submit the build (run slurm-windows first to pick partition and time):\n  "
              + shlex.join(command))
        return 0
    job = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    print(f"build job {job.split(';')[0]}; log: {context}/build-{job.split(';')[0]}.out")
    return 0


# ------------------------------------------------------------------- prepare

def claude_settings(workspace: Path, web_tools: bool = False) -> tuple[dict, dict]:
    """(~/.claude/settings.json, ~/.claude.json) for an unattended, trusted workspace."""
    settings = {
        "permissions": {"deny": [] if web_tools else list(DENIED_TOOLS),
                        "defaultMode": "bypassPermissions"},
        "skipDangerousModePermissionPrompt": True,
        "autoUpdates": False,
    }
    state = {
        "hasCompletedOnboarding": True,
        "bypassPermissionsModeAccepted": True,
        "projects": {str(workspace): {"hasTrustDialogAccepted": True,
                                      "hasCompletedProjectOnboarding": True}},
    }
    return settings, state


def codex_config(workspace: Path, model: str | None, effort: str | None) -> str:
    lines = []
    if model:
        lines.append(f"model = {json.dumps(model)}")
    if effort:
        lines.append(f"model_reasoning_effort = {json.dumps(effort)}")
    lines += ['approval_policy = "never"', 'sandbox_mode = "danger-full-access"',
              'web_search = "disabled"', "check_for_update_on_startup = false", "",
              f"[projects.{json.dumps(str(workspace))}]", 'trust_level = "trusted"', ""]
    return "\n".join(lines)


def opencode_config(provider: str, model: str, effort: str | None) -> dict:
    """~/.config/opencode/opencode.json: one OpenAI-compatible provider, no web tools."""
    spec = OPENCODE_PROVIDERS[provider]
    entry = {"name": model, "tool_call": True, "reasoning": True,
             "limit": {"context": spec["context"], "output": spec["output"]}}
    if model in spec["models"]:
        entry["cost"] = dict(spec["models"][model])
    if effort:
        entry["options"] = {"reasoningEffort": effort}
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{provider}/{model}",
        "autoupdate": False,
        "share": "disabled",
        # webfetch would be stopped by the proxy anyway; websearch runs past it.
        "permission": {"edit": "allow", "bash": "allow", "external_directory": "allow",
                       "webfetch": "deny", "websearch": "deny"},
        "provider": {provider: {
            "npm": "@ai-sdk/openai-compatible", "name": spec["name"],
            "options": {"baseURL": spec["base_url"], "apiKey": "{env:" + spec["env"] + "}"},
            "models": {model: entry}}},
    }


def round_seconds(args: argparse.Namespace) -> int:
    return (args.agent_minutes + args.grade_minutes + args.setup_minutes) * 60


def slurm_time(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"


def prepare(args: argparse.Namespace) -> int:
    run = args.run_dir.resolve()
    if run.exists():
        raise SystemExit(f"{run} exists; use a new directory for each run")
    if CREDENTIAL_AGENT[args.credential_kind] != args.agent:
        raise SystemExit(f"--credential-kind {args.credential_kind} is for {CREDENTIAL_AGENT[args.credential_kind]}")
    credential = args.credential_file.resolve()
    if not credential.is_file() or credential.stat().st_mode & 0o077:
        raise SystemExit("the credential file must exist and be readable only by its owner (chmod 600)")
    if credential.is_relative_to(run):
        raise SystemExit("keep the credential file outside the run directory")
    case = select_cases("all", [args.case])[0]
    if not case.ready:
        raise SystemExit(f"{args.case} is not ready")
    platform = "neuron" if case.backend == "neuron" else "cuda"
    runtime = args.runtime or ("docker" if platform == "neuron" else "podman")
    cores = args.neuron_cores or (f"0-{case.gpus - 1}" if case.gpus > 1 else "0")
    if platform == "neuron" and args.agent != "claude":
        raise SystemExit("the Trainium image carries Claude Code only")
    if platform == "neuron" and args.network != "open":
        raise SystemExit("Trainium runs use --network open (the proxy mode needs Podman's setup)")
    if args.partition in ("main", "priority", "background"):
        limit = {"main": 7200, "background": 7200, "priority": 1800}[args.partition]
        if round_seconds(args) > limit:
            raise SystemExit(f"a round needs {slurm_time(round_seconds(args))}, over the "
                             f"{args.partition} limit; lower --agent-minutes or --grade-minutes")
    provider = CREDENTIAL_PROVIDER.get(args.credential_kind)
    if args.agent == "opencode":
        if args.mode != "headless":
            raise SystemExit("opencode runs headless only (`opencode run`)")
        spec = OPENCODE_PROVIDERS[provider]
        args.model = args.model or spec["default_model"]
        args.effort = args.effort or spec["default_effort"]
        if args.effort == "none":
            args.effort = None
    model = args.model
    method = args.method or f"container-{args.agent}" + (f"-{model}" if model else "")

    if args.vibesys_plugin and not args.bare:
        raise SystemExit("--vibesys-plugin needs --bare (the workspace becomes a VibeSys task)")
    if args.vibesys_plugin and args.agent != "claude":
        raise SystemExit("--vibesys-plugin runs on Claude Code only")
    venv = (REPO / ".venv").absolute()
    build_trusted(args.ref, run / "trusted", venv.resolve())
    # A bare workspace sits in RUN/work, which the agent container mounts whole: VibeSys puts
    # its worktrees and phase maps beside the repository.
    workspace = run / "work" / args.case if args.bare else run / args.case
    command = [str(run / "trusted/.venv/bin/python"), "-m",
               "megabench.integrations.make_agent_workspace", "--case", args.case,
               "--output", str(workspace), "--agent", "plain", "--backend", "cuda",
               "--benchmark-root", str(run / "trusted")]
    if args.regimes:
        command += ["--regimes", str(args.regimes.resolve())]
        # Grading reads this copy: the workspace's regimes.json is the agent's to edit.
        (run / "state").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.regimes, run / "state/regimes.json")
    subprocess.run(command, check=True, cwd=run / "trusted",
                   env=os.environ | {"PYTHONPATH": str(run / "trusted")})
    if args.bare:
        if args.overlay or args.remove or args.regimes:
            raise SystemExit("--bare takes no --overlay, --remove or --regimes")
        args.snapshot_checks = False
        bare_workspace(workspace, case, run / "trusted", cores)
    else:
        customize_workspace(workspace, run / "control", args.overlay, args.snapshot_checks,
                            args.remove)
    # In the container the trusted snapshot's interpreter is the image's venv.
    (run / "trusted/.venv").unlink()
    (run / "trusted/.venv").symlink_to(IMAGE_VENV)
    agent_trusted = make_agent_trusted(run, args.trusted_hide)

    home = run / "home"
    if args.agent == "claude":
        settings, state = claude_settings(workspace, args.web_tools)
        (home / ".claude").mkdir(parents=True)
        (home / ".claude/settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        (home / ".claude.json").write_text(json.dumps(state, indent=2) + "\n")
        if args.vibesys_plugin:
            (home / ".config/vibesys").mkdir(parents=True)
            (home / ".config/vibesys/cli.toml").write_text(
                VIBESYS_CLI_TOML_NEURON if platform == "neuron" else VIBESYS_CLI_TOML.format(gpus=case.gpus))
    elif args.agent == "codex":
        (home / ".codex").mkdir(parents=True)
        (home / ".codex/config.toml").write_text(codex_config(workspace, model, args.effort))
    else:
        (home / ".config/opencode").mkdir(parents=True)
        (home / ".config/opencode/opencode.json").write_text(
            json.dumps(opencode_config(provider, model, args.effort), indent=2) + "\n")
    for name in ("state", "logs", "results", "grade", "control"):
        (run / name).mkdir(parents=True, exist_ok=True)

    budget = args.budget_minutes or args.agent_minutes * args.rounds
    if args.vibesys_plugin:
        # The lead's own budget leaves the end of the slot for its last integration and report.
        lead = args.vibesys_budget_minutes or max(1, budget - 15)
        args.goal = VIBESYS_GOAL.format(case=args.case, budget=lead)
        kickoff = VIBESYS_KICKOFF.format(case=args.case, budget=lead, fanout=args.vibesys_fanout)
    else:
        kickoff = args.kickoff_file.read_text() if args.kickoff_file else KICKOFF
    goal = " ".join(args.goal.split())  # one line: the TUI submits on Enter
    (run / "control/goal.txt").write_text(goal + "\n")
    (run / "control/kickoff.txt").write_text(" ".join(kickoff.format(minutes=budget).split()) + "\n")
    (run / "control/resume.txt").write_text(RESUME + "\n")
    (run / "control/goal_check.txt").write_text(GOAL_CHECK.format(goal=goal) + "\n")
    shutil.copy2(Path(__file__).with_name("container_entry.sh"), run / "control/entry.sh")
    shutil.copy2(DOCKER_DIR / "egress_proxy.py", run / "control/egress_proxy.py")
    shutil.copy2(Path(__file__).with_name("container_harvest.py"), run / "control/harvest.py")
    (run / "control/entry.sh").chmod(0o755)

    provider_hosts = [urlsplit(OPENCODE_PROVIDERS[provider]["base_url"]).hostname] if provider else []
    allow = list(dict.fromkeys([*DEFAULT_ALLOW[args.agent], *provider_hosts, *args.allow_host]))
    config = {
        "case": args.case, "agent": args.agent, "provider": provider, "model": model,
        "effort": args.effort,
        "mode": args.mode, "goal_style": args.goal_style, "image": args.image,
        "ref": args.ref,
        "ref_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", args.ref],
                                     capture_output=True, text=True).stdout.strip(),
        "gpus": case.gpus, "partition": args.partition, "rounds": args.rounds,
        "agent_minutes": args.agent_minutes, "grade_minutes": args.grade_minutes,
        "idle_minutes": args.idle_minutes, "allow_hosts": allow, "method": method,
        "session_id": f"{method}-{args.case}-{run.name}", "credential_kind": args.credential_kind,
        "credential_file": str(credential), "grade_reps": args.grade_reps,
        "grade_timeout": args.grade_timeout,
        "regimes": str(args.regimes.resolve()) if args.regimes else None,
        "budget_minutes": budget, "overlays": [str(o.resolve()) for o in args.overlay],
        "removed": list(args.remove), "trusted_hidden": list(args.trusted_hide),
        "kickoff": (run / "control/kickoff.txt").read_text().strip(),
        "goal": goal, "snapshot_checks": args.snapshot_checks,
        "bare": args.bare, "network": args.network, "web_tools": args.web_tools,
        "vibesys_plugin": args.vibesys_plugin, "vibesys_cli_version": args.vibesys_cli_version,
        "result_branch": f"vibesys/result/{args.case}" if args.vibesys_plugin else None,
        "platform": platform, "runtime": runtime, "neuron_cores": cores if platform == "neuron" else None,
    }
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = {
        "RUN": run, "CASE": args.case, "WS": workspace, "TRUSTED": run / "trusted",
        "IMAGE": args.image, "AGENT": args.agent, "MODEL": model or "",
        "EFFORT": args.effort or "", "MODE": args.mode, "GOAL_STYLE": args.goal_style,
        "AGENT_SECONDS": args.agent_minutes * 60, "IDLE_SECONDS": args.idle_minutes * 60,
        "BUDGET_SECONDS": budget * 60,
        "ALLOW_HOSTS": " ".join(allow), "CREDENTIAL_KIND": args.credential_kind,
        "CREDENTIAL_ENV": CREDENTIAL_ENV[args.credential_kind] or "",
        "CREDENTIAL_FILE": credential, "METHOD": method, "SESSION_ID": config["session_id"],
        "GRADE_REPS": args.grade_reps, "GRADE_TIMEOUT": args.grade_timeout,
        "AGENT_TRUSTED": agent_trusted or "", "SHM_SIZE": args.shm_size, "REGIMES": "1" if args.regimes else "", "FORBID": " ".join(map(str, forbidden_paths(run, credential))),
        # The agent container mounts WORK read-write (the workspace, or its parent when bare).
        "WORK": workspace.parent if args.bare else workspace,
        "NET_MODE": args.network, "VIBESYS_PLUGIN_SOURCE": args.vibesys_plugin or "",
        "VIBESYS_CLI_VERSION": args.vibesys_cli_version,
        "RESULT_BRANCH": config["result_branch"] or "",
        "PLATFORM": platform, "RUNTIME": runtime,
        "NEURON_CORES": cores if platform == "neuron" else "",
        "DOMAIN_PLUGIN": "vibesys-kernels-trainium" if platform == "neuron" else "vibesys-kernels-cuda",
    }
    (run / "run.env").write_text("".join(f"{k}={shlex.quote(str(v))}\n" for k, v in env.items()))
    shutil.copy2(Path(__file__).with_name("container_round.sh"), run / "round.sh")
    (run / "round.sh").chmod(0o755)
    print(f"prepared {run}\n"
          f"  workspace: {workspace}\n  image: {args.image}\n"
          f"  each round: {case.gpus} {'NeuronCore' if platform == 'neuron' else 'GPU'}(s), "
          f"--time={slurm_time(round_seconds(args))} "
          f"({args.agent_minutes} min agent, {args.grade_minutes} min grading)\n"
          "next:\n"
          f"  {sys.executable} -m megabench.integrations.container_run smoke --run-dir {run}\n"
          f"  {sys.executable} -m megabench.integrations.container_run submit --run-dir {run}")
    return 0


def bare_workspace(workspace: Path, case, trusted: Path, cores: str = "0") -> None:
    """Strip the stock workspace to the case's contract and make it a VibeSys task.

    Keeps TASK.md's contract paragraphs (not its tools), case.json, the submission stub and
    the reference; writes `.vibesys/tasks/<case>/`; and starts the git history afresh, so no
    earlier commit holds a removed tool or skill.
    """
    for entry in list(workspace.iterdir()):
        if entry.name in BARE_KEEP:
            continue
        shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
    task_md = workspace / "TASK.md"
    contract = task_md.read_text().split("\nUse `./check_candidate`")[0].rstrip()
    neuron = case.backend == "neuron"
    tail = (BARE_TASK_TAIL_NEURON.format(case=case.id, cores=case.gpus, core_range=cores) if neuron
            else BARE_TASK_TAIL.format(case=case.id, gpus=case.gpus))
    task_md.write_text(contract + "\n\n" + tail)
    (workspace / ".gitignore").write_text(BARE_GITIGNORE)
    reference = sorted((workspace / "reference/megabench/tasks").glob("*.py"))
    readme = (workspace / "reference/README.md").read_text()
    module = next((p.stem for p in reference if f"megabench/tasks/{p.stem}.py" in readme), None)
    if module is None:
        raise SystemExit(f"no reference module named in {workspace}/reference/README.md")
    task = workspace / ".vibesys/tasks" / case.id
    task.mkdir(parents=True)
    limits = SLOW_CASE_LIMITS.get(case.id, {})
    (task / "evaluate.py").write_text(BARE_EVALUATOR.format(
        trusted=str(trusted), case=case.id, bench_override=limits.get("benchmark_settings", {})))
    python = str(trusted / ".venv/bin/python")
    command = lambda mode: json.dumps(  # noqa: E731
        [python, f".vibesys/tasks/{case.id}/evaluate.py", mode, "--submission", "submission.py"])
    (task / "vibesys.input.toml").write_text(
        'version = "0.1"\n'
        + ('backends = ["trainium"]\n\n[agent]\ndomain = "kernel-writing"\n\n' if neuron else
           'backends = ["cuda"]\n\n[agent]\ndomain = "kernel-writing"\n\n'
           '[environment]\nneeds = ["gpu"]\n\n') +
        f"[accuracy]\ncommand = {command('accuracy')}\n"
        f"timeout_seconds = {limits.get('accuracy_timeout', 1800)}\n\n"
        f"[benchmark]\ncommand = {command('benchmark')}\n"
        f"timeout_seconds = {limits.get('benchmark_timeout', 2400)}\n\n"
        '[benchmark.result]\njson_argument = "--output-json"\nmetric = "kernel_latency_ms"\n')
    # vibesys-cli 0.3.0 reads no `samples` (a measure takes 5, each a fresh evaluate process)
    # and takes the noise floor of a single objective only from [pareto].
    (task / "objectives.toml").write_text(
        "[[objective]]\n"
        'name = "kernel_latency_ms"\ndirection = "min"\nunit = "ms"\n\n'
        "[pareto]\nrelative_noise = 0.01\n")
    (task / "OBJECTIVE.md").write_text(
        BARE_OBJECTIVE_NEURON.format(case=case.id, model=case.model, phase=case.phase,
                                     module=module, launches=case.max_gpu_launches,
                                     cores=case.gpus) if neuron else
        BARE_OBJECTIVE.format(case=case.id, model=case.model, phase=case.phase, module=module,
                              launches=case.max_gpu_launches, gpus=case.gpus))
    shutil.rmtree(workspace / ".git", ignore_errors=True)
    for argv in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.name=MegaBench", "-c", "user.email=megabench@example.invalid",
                  "commit", "-q", "-m", f"{case.id}: task and reference"]):
        subprocess.run(["git", "-C", str(workspace), *argv], check=True)


def make_agent_trusted(run: Path, hide: list[str]) -> Path | None:
    """The trusted snapshot the agent container sees: RUN/trusted minus `hide`.

    The agent container mounts it at RUN/trusted, so the workspace tools keep working; grading
    and rescoring mount the full RUN/trusted. With nothing hidden the agent sees RUN/trusted.
    """
    if not hide:
        return None
    full, agent = run / "trusted", run / "trusted-agent"
    shutil.copytree(full, agent, symlinks=True)
    for rel in hide:
        target = (agent / rel)
        if not target.resolve().is_relative_to(agent.resolve()) or target.resolve() == agent.resolve():
            raise SystemExit(f"--trusted-hide {rel}: not a path inside the trusted snapshot")
        if rel.strip("/") in ("megabench", "megabench/cases.py", "megabench/harness", "megabench/tasks", ".venv"):
            raise SystemExit(f"--trusted-hide {rel}: the evaluator needs it")
        if not (target.exists() or target.is_symlink()):
            raise SystemExit(f"--trusted-hide {rel}: no such path in the trusted snapshot")
        shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
    return agent


def customize_workspace(workspace: Path, control: Path, overlays: list[Path], snapshots: bool,
                        remove: list[str] = ()) -> None:
    """Arm removals and overlays, ./time_left and the snapshotting ./check_candidate; one git commit."""
    for rel in remove:
        target = (workspace / rel).resolve()
        if not target.is_relative_to(workspace.resolve()) or target == workspace.resolve():
            raise SystemExit(f"--remove {rel}: not a path inside the workspace")
        if rel.strip("/") in ("check_candidate", "submission.py", "TASK.md", "case.json", "reference"):
            raise SystemExit(f"--remove {rel}: the workspace needs it")
        if not (target.exists() or target.is_symlink()):
            raise SystemExit(f"--remove {rel}: no such path in the stock workspace")
        shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
    for overlay in overlays:
        shutil.copytree(overlay, workspace, dirs_exist_ok=True, symlinks=True)
    tool = workspace / "time_left"
    tool.write_text(TIME_LEFT.format(control=control))
    tool.chmod(0o755)
    if snapshots:
        check = workspace / "check_candidate"
        lines = check.read_text().splitlines(keepends=True)
        at = next(i for i, line in enumerate(lines) if line.startswith("run_id="))
        lines.insert(at + 1, SNAPSHOT_LINE)
        check.write_text("".join(lines))
    with (workspace / ".gitignore").open("a") as out:
        out.write(".harness/\n")
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=MegaBench",
                    "-c", "user.email=megabench@example.invalid", "commit", "-q", "-m",
                    "container_run: overlays and harness tools"], check=True)


def forbidden_paths(run: Path, credential: Path) -> list[Path]:
    """Host paths that must not exist in the agent container (the smoke check)."""
    home = Path.home()
    # Files, not directories: bind mounts create the parents of the run directory (which
    # may sit inside the repository) as empty directories in the container.
    paths = [REPO / "megabench/cases.py", REPO / "megabench/tasks", REPO / ".git",
             REPO / "README.md", REPO / "reproductions", REPO / "submission.py",
             home / ".claude", home / ".claude.json", home / ".codex", home / ".ssh",
             home / ".config/opencode/opencode.json", home / ".local/share/opencode/opencode.db",
             home / ".bashrc",
             run / "state", run / "results", run / "grade", run / "config.json", credential,
             Path("/scratch/hf/hub")]
    return list(dict.fromkeys(paths))


# ------------------------------------------------------------- smoke, submit

def _load(run: Path) -> dict:
    path = run / "config.json"
    if not path.is_file():
        raise SystemExit(f"{run} is not a prepared run (no config.json)")
    return json.loads(path.read_text())


def _scheduler(args: argparse.Namespace) -> str:
    return args.scheduler or ("slurm" if shutil.which("sbatch") else "local")


def _local(run: Path, argv: list[str], seconds: int) -> str:
    """Run round.sh here, in the foreground, once the accelerator lock is free; return the job id.

    Stands in for a Slurm job: a unique SLURM_JOB_ID names the job's containers and logs, the
    lock keeps two rounds off one chip, and `timeout` is the job's time limit.
    """
    job = f"L{time.strftime('%Y%m%d%H%M%S')}{os.getpid() % 1000:03d}"
    log = run / "logs" / f"slurm-{job}.out"
    command = ["flock", str(LOCAL_LOCK), "timeout", "--signal=TERM", "--kill-after=120",
               str(seconds), str(run / "round.sh"), str(run), *argv]
    print(f"job {job}: {' '.join(argv)} (log {log}); waiting for {LOCAL_LOCK}", flush=True)
    with log.open("ab") as out:
        rc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT,
                            env=os.environ | {"SLURM_JOB_ID": job}).returncode
    print(f"job {job} exited {rc}", flush=True)
    return job


def _sbatch(run: Path, config: dict, partition: str, time: str, name: str,
            argv: list[str], dependency: str | None = None) -> str:
    command = ["sbatch", "--parsable", f"--partition={partition}",
               f"--gres=gpu:{config['gpus']}", f"--time={time}", f"--job-name={name}",
               f"--output={run}/logs/slurm-%j.out"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command += [str(run / "round.sh"), str(run), *argv]
    out = subprocess.run(command, check=True, capture_output=True, text=True)
    sys.stderr.write(out.stderr)
    return out.stdout.strip().split(";")[0]


def smoke(args: argparse.Namespace) -> int:
    run = args.run_dir.resolve()
    config = _load(run)
    if _scheduler(args) == "local":
        job = _local(run, ["smoke"], 1800)
        print(f"smoke job {job}: read {run}/logs/smoke-{job}.txt (the last line is SMOKE PASS or SMOKE FAIL)")
        return 0
    job = _sbatch(run, config, args.partition, args.time, f"mb-smoke-{config['case'][:24]}",
                  ["smoke"])
    print(f"smoke job {job}: when it ends, read {run}/logs/smoke-{job}.txt "
          f"(the last line is SMOKE PASS or SMOKE FAIL)")
    return 0


def submit(args: argparse.Namespace) -> int:
    run = args.run_dir.resolve()
    config = _load(run)
    rounds = args.rounds or config["rounds"]
    seconds = (config["agent_minutes"] + config["grade_minutes"]) * 60 + args.setup_minutes * 60
    partition = args.partition or config["partition"]
    done = sorted((run / "state").glob("round-*.started"))
    first = len(done) + 1
    if _scheduler(args) == "local":
        # One round after another, here; each waits for the accelerator lock.
        jobs = [_local(run, ["round", str(number)], seconds) for number in range(first, first + rounds)]
        (run / "logs/jobs.txt").open("a").write(
            "".join(f"round {first + i} job {j}\n" for i, j in enumerate(jobs)))
        print(f"ran rounds {first}..{first + rounds - 1}; results in {run}/results")
        return 0
    previous = args.after
    jobs = []
    for number in range(first, first + rounds):
        dependency = f"afterany:{previous}" if previous else None
        previous = _sbatch(run, config, partition, slurm_time(seconds),
                           f"mb-{config['agent']}-{config['case'][:24]}-r{number}",
                           ["round", str(number)], dependency)
        jobs.append(previous)
    (run / "logs/jobs.txt").open("a").write(
        "".join(f"round {first + i} job {j}\n" for i, j in enumerate(jobs)))
    print("submitted rounds " + ", ".join(f"{first + i}: job {j}" for i, j in enumerate(jobs)))
    print(f"watch: squeue -j {','.join(jobs)}; logs in {run}/logs; results in {run}/results")
    if config["mode"] == "tmux":
        print("watch the agent's terminal while a round runs:\n"
              f"  srun --jobid=JOB --overlap --pty podman exec -it mb-agent-JOB "
              "tmux attach -r -t agent")
    return 0


def rescore(args: argparse.Namespace) -> int:
    run = args.run_dir.resolve()
    config = _load(run)
    if _scheduler(args) == "local":
        job = _local(run, ["rescore"], int(args.time.split(":")[0]) * 3600
                     + int(args.time.split(":")[1]) * 60)
        print(f"rescore job {job}; then: container_run report --run-dir {run}")
        return 0
    job = _sbatch(run, config, args.partition or config["partition"], args.time,
                  f"mb-rescore-{config['case'][:24]}", ["rescore"],
                  f"afterany:{args.after}" if args.after else None)
    print(f"rescore job {job}; then: container_run report --run-dir {run}")
    return 0


def _events(path: Path):
    for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
        try:
            yield json.loads(line)
        except ValueError:
            yield None


def _user_text(event: dict) -> str:
    content = (event.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content or []
                    if isinstance(part, dict) and part.get("type") == "text")


def round_summary(run: Path, number: int) -> dict:
    """One round from its logs: time worked, how it ended, goal re-prompts, cost, grade."""
    logs, state = run / "logs", run / "state"
    elapsed = state / f"round-{number}.elapsed"
    seconds, rc = (int(x) for x in elapsed.read_text().split()) if elapsed.exists() else (None, None)
    started = (state / f"round-{number}.started").stat().st_mtime
    times = [float(t) for t in (logs / f"agent-round-{number}.times").read_text().split()] \
        if (logs / f"agent-round-{number}.times").exists() else []
    out = {"round": number, "agent_seconds": seconds, "exit": rc,
           "ended": {None: "unknown", 124: "time slot over"}.get(rc, "agent ended the session"),
           "goal_reprompts": 0, "tool_calls": 0, "cost_usd": None, "last_tool_call_s": None}
    for index, event in enumerate(_events(logs / f"agent-round-{number}.out")):
        if not event:
            continue
        if event.get("type") == "user" and _user_text(event).startswith("Stop hook feedback"):
            out["goal_reprompts"] += 1
        if event.get("type") == "assistant":
            calls = [c for c in (event.get("message") or {}).get("content") or []
                     if isinstance(c, dict) and c.get("type") == "tool_use"]
            if calls:
                out["tool_calls"] += len(calls)
                if index < len(times):
                    out["last_tool_call_s"] = round(times[index] - started, 1)
        if event.get("type") == "result":
            out["cost_usd"] = event.get("total_cost_usd")
        # opencode (`opencode run --format json`): one tool_use event per call, the cost on
        # each step_finish, and the harness's own goal re-prompts.
        if event.get("type") == "harness_goal_reprompt":
            out["goal_reprompts"] += 1
        if event.get("type") == "tool_use":
            out["tool_calls"] += 1
            if index < len(times):
                out["last_tool_call_s"] = round(times[index] - started, 1)
        if event.get("type") == "step_finish":
            cost = (event.get("part") or {}).get("cost")
            if isinstance(cost, (int, float)):
                out["cost_usd"] = (out["cost_usd"] or 0) + cost
    grades = sorted((run / "results").glob(f"round-{number}-*.jsonl"))
    record = None
    for line in grades[-1].read_text().splitlines() if grades else []:
        if line.strip():
            try:
                record = json.loads(line)
            except ValueError:
                pass
    out["grade"] = None if record is None else {
        "status": record.get("status"),
        "p50_ms": (record.get("candidate_timing") or {}).get("cuda_event_p50_ms")}
    return out


def report(args: argparse.Namespace) -> int:
    """RUN/result.json: rounds, checks, snapshots and the rescored result L."""
    import statistics
    from . import container_harvest as harvest

    run = args.run_dir.resolve()
    config = _load(run)
    numbers = sorted(int(p.name.split("-")[1].split(".")[0])
                     for p in (run / "state").glob("round-*.started"))
    rounds = [round_summary(run, n) for n in numbers]
    worked = sum(r["agent_seconds"] or 0 for r in rounds[:-1]) + (
        (rounds[-1]["last_tool_call_s"] or 0) if rounds else 0)
    checks = [json.loads(l) for l in (run / "logs/checks.jsonl").read_text().splitlines()] \
        if (run / "logs/checks.jsonl").exists() else []
    tried, chosen = [], None
    for directory in sorted((run / "rescore").glob("*")) if (run / "rescore").exists() else []:
        records = [harvest._record(p) for p in sorted(directory.glob("check-*.jsonl"))]
        review = json.loads((directory / "review.json").read_text()) \
            if (directory / "review.json").exists() else ["not reviewed"]
        p50s = [harvest.p50(r) for r in records if r]
        ok = (not review and len(records) == int(args.reps) and all(
            r and r.get("status") in harvest.PASSING and harvest.launches_ok(r) for r in records))
        row = {"snapshot": directory.name, "review": review, "statuses":
               [r.get("status") if r else None for r in records], "p50s_ms": p50s,
               "L_ms": statistics.median(p50s) if ok else None, "valid": ok}
        tried.append(row)
        if ok and chosen is None:
            chosen = row
    index = run / "snapshots/index.jsonl"
    snaps = [json.loads(l) for l in index.read_text().splitlines()] if index.exists() else []
    # Agent minutes of each passing check (time between rounds does not count).
    spans, offset = [], 0.0
    for r in rounds:
        start = (run / "state" / f"round-{r['round']}.started").stat().st_mtime
        spans.append((start, start + (r["agent_seconds"] or 0), offset))
        offset += r["agent_seconds"] or 0
    trajectory, best = [], None
    for snap in sorted(snaps, key=lambda s: s["report_time"]):
        at = next(((snap["report_time"] - a + off) / 60 for a, b, off in spans
                   if a <= snap["report_time"] <= b + 1800), None)
        if snap.get("agent_p50_ms") and (best is None or snap["agent_p50_ms"] < best):
            best = snap["agent_p50_ms"]
            trajectory.append({"agent_minute": None if at is None else round(at, 1),
                               "best_agent_p50_ms": best})
    result = {
        "schema": "container-run/result/v1", "run": run.name, "case": config["case"],
        "goal_style": config["goal_style"], "goal": config.get("goal"),
        "overlays": config.get("overlays", []), "budget_minutes": config.get("budget_minutes"),
        "model": config["model"], "image": config["image"], "ref_commit": config["ref_commit"],
        "rounds": rounds, "active_minutes": round(worked / 60, 1),
        # Ended on its own with more than 2 minutes of the agent budget left (the slot floor):
        # a session that stops when ./time_left reaches 0 used its whole budget.
        "ended_early": bool(rounds) and rounds[-1]["exit"] not in (124, None)
        and (run / "state/finished").exists()
        and (config.get("budget_minutes") or 0) * 60 - sum(r["agent_seconds"] or 0 for r in rounds) > 120,
        "goal_reprompts": sum(r["goal_reprompts"] for r in rounds),
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in rounds), 2),
        "checks": {"n": len(checks), "passing": len(snaps)},
        "best_agent_p50_ms": min((s["agent_p50_ms"] for s in snaps if s.get("agent_p50_ms")), default=None),
        "first_pass_minutes": trajectory[0]["agent_minute"] if trajectory else None,
        "best_so_far": trajectory, "rescore": tried,
        "L_ms": chosen["L_ms"] if chosen else None,
    }
    # Exclusive-create: a later report (say, after another rescore) gets a new file.
    path = next(run / name for name in
                ["result.json"] + [f"result-{n}.json" for n in range(2, 1000)]
                if not (run / name).exists())
    with path.open("x") as out:
        out.write(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}")
    print(json.dumps({k: result[k] for k in ("run", "active_minutes", "ended_early", "goal_reprompts",
                                             "cost_usd", "checks", "best_agent_p50_ms", "L_ms")}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    image = sub.add_parser("build-image", help="stage the image build context and build it")
    image.add_argument("--context", type=Path,
                       default=Path(os.environ.get("HOME", "~")).expanduser()
                       / ".cache/megabench-agent-image")
    image.add_argument("--partition", default="main")
    image.add_argument("--time", default="01:00:00")
    image.add_argument("--submit", action="store_true", help="submit the build job")
    image.add_argument("--platform", choices=("cuda", "neuron"), default="cuda",
                       help="neuron: build megabench/docker/neuron/agent.Dockerfile here with Docker")
    image.add_argument("--base-image", default="megabench-neuron:native",
                       help="the Neuron image the neuron agent image builds on")

    prep = sub.add_parser("prepare", help="make a new run directory for one case")
    prep.add_argument("--agent", choices=("claude", "codex", "opencode"), required=True)
    prep.add_argument("--case", required=True)
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--image", required=True, help="tag printed by build-image")
    prep.add_argument("--credential-kind", choices=sorted(CREDENTIAL_ENV), required=True,
                      help="claude-oauth-token (`claude setup-token`) or anthropic-api-key for "
                           "Claude; openai-api-key or codex-auth-json (a ChatGPT login's "
                           "auth.json) for Codex; coralbricks-api-key for opencode")
    prep.add_argument("--credential-file", type=Path, required=True,
                      help="read at the start of each round; mode 600, outside the run")
    prep.add_argument("--model", help="model name passed to the CLI (default: the CLI's; for "
                                       "opencode on Coral Bricks, deepseek-v4.1-flash)")
    prep.add_argument("--effort", help="Codex model_reasoning_effort / Claude --effort / opencode "
                                       "reasoningEffort (Coral Bricks: low, high (default), max, "
                                       "none)")
    prep.add_argument("--mode", choices=("headless", "tmux"),
                      help="headless: `claude -p` / `codex exec`; tmux: the interactive TUI in "
                           "tmux, typed into like a user (default: headless for claude, tmux "
                           "for codex; opencode is headless only)")
    prep.add_argument("--goal", default=DEFAULT_GOAL, help="the /goal condition (one line)")
    prep.add_argument("--goal-style", choices=("command", "prompt"), default="command",
                      help="command: send `/goal <goal>` (opencode, which has no /goal: the "
                           "harness resumes the session with GOAL_CHECK whenever it stops, until "
                           "the agent answers GOAL MET); prompt: send the goal as plain text "
                           "(a run without the goal feature, for comparison)")
    prep.add_argument("--rounds", type=int, default=1)
    prep.add_argument("--agent-minutes", type=int, default=90)
    prep.add_argument("--grade-minutes", type=int, default=20)
    prep.add_argument("--setup-minutes", type=int, default=5)
    prep.add_argument("--idle-minutes", type=int, default=15,
                      help="tmux mode: end the round when the screen has not changed this long")
    prep.add_argument("--partition", default="main")
    prep.add_argument("--allow-host", action="append", default=[],
                      help="extra host the proxy lets through (repeat; .domain for a domain)")
    prep.add_argument("--ref", default="HEAD", help="git ref of the trusted megabench snapshot")
    prep.add_argument("--regimes", type=Path)
    prep.add_argument("--method", help="method name in the results (default: container-AGENT-MODEL)")
    prep.add_argument("--grade-reps", type=int, default=20)
    prep.add_argument("--grade-timeout", type=int, default=900)
    prep.add_argument("--shm-size", default="32g")
    prep.add_argument("--budget-minutes", type=int,
                      help="agent minutes over all rounds (default: rounds x agent minutes); "
                           "./time_left reports what is left")
    prep.add_argument("--overlay", type=Path, action="append", default=[],
                      help="directory copied into the workspace (repeat; in order)")
    prep.add_argument("--remove", action="append", default=[], metavar="PATH",
                      help="workspace path (relative) deleted before the overlays, e.g. an arm without "
                           "a stock tool or skill (repeat)")
    prep.add_argument("--trusted-hide", action="append", default=[], metavar="PATH",
                      help="path in the trusted snapshot (relative, e.g. megabench/skills) that the agent "
                           "container does not see; grading still uses the full snapshot (repeat)")
    prep.add_argument("--kickoff-file", type=Path,
                      help="the message after /goal ({minutes} = the budget); default KICKOFF")
    prep.add_argument("--no-snapshot-checks", dest="snapshot_checks", action="store_false",
                      help="do not keep the source tree of every passing ./check_candidate")
    prep.add_argument("--bare", action="store_true",
                      help="workspace = the case's contract and reference only, as a VibeSys task "
                           "(no tools or skills of this repository); grading reads the result "
                           "branch when --vibesys-plugin is set")
    prep.add_argument("--network", choices=("proxy", "open"), default="proxy",
                      help="proxy: no network but the allowlist proxy (default); open: the agent "
                           "container has its own full network (pasta)")
    prep.add_argument("--web-tools", action="store_true",
                      help="Claude: allow WebSearch and WebFetch (denied by default)")
    prep.add_argument("--vibesys-plugin", metavar="SOURCE",
                      help="install the VibeSys Claude Code plugins (vibesys, vibesys-kernels-cuda) "
                           "from this marketplace source in the container, and start the lead with "
                           "/vibesys:optimize in megakernel divide-and-conquer mode; e.g. "
                           "https://github.com/uw-syfi/vibesys.git#<branch>")
    prep.add_argument("--vibesys-cli-version", default="0.3.0",
                      help="vibesys-cli release installed in the container")
    prep.add_argument("--vibesys-budget-minutes", type=int,
                      help="the lead's --budget time (default: the agent budget minus 15)")
    prep.add_argument("--vibesys-fanout", type=int, default=3)
    prep.add_argument("--runtime", choices=("podman", "docker"),
                      help="container runtime (default: docker for Trainium cases, else podman)")
    prep.add_argument("--neuron-cores",
                      help="NEURON_RT_VISIBLE_CORES of a Trainium run (default: 0 to gpus-1)")

    sm = sub.add_parser("smoke", help="check the container's view and network in a short job")
    sm.add_argument("--run-dir", type=Path, required=True)
    sm.add_argument("--scheduler", choices=("slurm", "local"),
                       help="local: run here under a lock (default when sbatch is missing)")
    sm.add_argument("--partition", default="priority")
    sm.add_argument("--time", default="00:15:00")

    sb = sub.add_parser("submit", help="submit the rounds as a chain of Slurm jobs")
    sb.add_argument("--run-dir", type=Path, required=True)
    sb.add_argument("--scheduler", choices=("slurm", "local"),
                       help="local: run here under a lock (default when sbatch is missing)")
    sb.add_argument("--rounds", type=int, help="rounds to add (default: the prepared count)")
    sb.add_argument("--partition", help="default: the prepared partition")
    sb.add_argument("--setup-minutes", type=int, default=5)
    sb.add_argument("--after", help="job id the first round waits for (afterany)")

    rs = sub.add_parser("rescore", help="re-check the best passing snapshots (a GPU job)")
    rs.add_argument("--run-dir", type=Path, required=True)
    rs.add_argument("--scheduler", choices=("slurm", "local"),
                       help="local: run here under a lock (default when sbatch is missing)")
    rs.add_argument("--partition")
    rs.add_argument("--time", default="01:30:00")
    rs.add_argument("--after", help="job id to wait for (afterany), e.g. the last round")

    rp = sub.add_parser("report", help="write RUN/result.json from the logs and the rescore")
    rp.add_argument("--run-dir", type=Path, required=True)
    rp.add_argument("--reps", type=int, default=3, help="rescore checks per snapshot")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare" and args.mode is None:
        args.mode = "tmux" if args.agent == "codex" else "headless"
    return {"build-image": build_image, "prepare": prepare, "smoke": smoke,
            "submit": submit, "rescore": rescore, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
