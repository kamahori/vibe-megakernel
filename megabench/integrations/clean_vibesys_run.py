"""Prepare an isolated VibeSys run for one MegaBench case.

The agent must see the task and nothing from earlier work. This script builds:

- ``trusted/``: ``megabench/`` from one git ref, without past-run docs, probes,
  submissions and examples, plus a ``.venv`` link to the benchmark venv;
- ``<case>/``: a fresh workspace built from ``trusted/``;
- ``home/``: an empty ``HOME`` for the agent CLI, with web tools denied, so no
  user config, memory, skills, plugins or transcripts are visible;
- ``launch.sh``: runs on the host, not in a Slurm job. It starts
  ``gpu_broker`` and then VibeSys with ``env -i`` (only the listed variables
  pass), the agent credential read at launch (an API key from
  ``--api-key-file``, or a ``claude setup-token`` subscription token from
  ``--oauth-token-file``), and explicit
  skills, so preset skills outside the list do not load.

VibeSys runs in a namespace without GPU devices. Each GPU use is a Slurm job:
the accuracy and benchmark gates run under ``srun``, and the agent runs GPU
commands through ``./gpu_run``, which ``gpu_broker`` starts as a job in the
agent's own sandbox (``gpu_job.py``).

It then runs a preflight inside the VibeSys sandbox: forbidden paths (also the
GPU devices and the Slurm credential socket) must be absent and the workspace
and trusted root must be present. Start ``launch.sh`` only after the preflight
passes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from ..cases import select_cases

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS = ("kernel-correctness", "kernel-ncu-analysis", "kernel-performance")
# Generic guidance stays; docs that report past runs or designs do not.
KEEP_DOCS = ("AGENT_TASK.md", "MODEL_STEP_TASKS.md", "REFERENCE_PRECISION.md",
             "SESSION_PROTOCOL.md")
DROP_DIRS = ("probes", "submissions", "examples", "experiments", "runs", "archive")
DENIED_TOOLS = ("WebFetch", "WebSearch")
# VibeSys runs here: the host file system, but a /dev without GPU devices.
NO_GPU = ("bwrap", "--dev-bind", "/", "/", "--dev", "/dev", "--unshare-user",
          "--die-with-parent", "--")
# Without these the sandbox can neither use a GPU nor submit a Slurm job.
NO_GPU_PATHS = ("/dev/nvidiactl", "/dev/nvidia0", "/run/munge")


def build_trusted(ref: str, trusted: Path, venv: Path) -> None:
    trusted.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "megabench.tar"
        subprocess.run(["git", "-C", str(REPO), "archive", "-o", str(archive), ref, "megabench"],
                       check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(trusted, filter="data")
    root = trusted / "megabench"
    for name in DROP_DIRS:
        shutil.rmtree(root / name, ignore_errors=True)
    for doc in (root / "docs").iterdir():
        if doc.name not in KEEP_DOCS:
            doc.unlink() if doc.is_file() else shutil.rmtree(doc)
    (trusted / ".venv").symlink_to(venv)


# Variables that carry the agent credential; GPU jobs and gates drop both.
CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")


def credential(args) -> tuple[Path, str]:
    """The credential file and the variable the Claude CLI reads it from."""
    if args.oauth_token_file:
        return args.oauth_token_file, "CLAUDE_CODE_OAUTH_TOKEN"
    return args.api_key_file, "ANTHROPIC_API_KEY"


def agent_env(args, run: Path, claude: Path, key: str) -> dict[str, str]:
    """The complete environment VibeSys and the agent see; nothing else passes."""
    project = run / args.case
    return {
        "HOME": str(run / "home"), "USER": os.environ.get("USER", "agent"),
        "LOGNAME": os.environ.get("USER", "agent"), "LANG": "C.UTF-8", "TERM": "dumb",
        "PATH": ":".join((str(args.venv.resolve() / "bin"), str(claude.parent), str(args.cuda / "bin"),
                          "/usr/local/bin", "/usr/bin", "/bin")),
        credential(args)[1]: key, "TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp",
        "NCU_PATH": str(args.cuda / "bin/ncu"),
        "TORCH_EXTENSIONS_DIR": str(project / ".torch_extensions"),
        "TRITON_CACHE_DIR": str(project / ".cache/triton"),
        "VIBESYS_STATE_HOME": str(run / "state"),
        "VIBESYS_AGENT_SANDBOX_ALLOW": ":".join(
            (str(run / "trusted"), str(args.venv.resolve()), str(args.cuda), str(run / "gpu"))),
    }


def srun_command(args, gpus: int) -> list[str]:
    return ["srun", f"--partition={args.gpu_partition}", f"--gres=gpu:{gpus}", "--nodes=1",
            "--ntasks=1", f"--cpus-per-task={args.gpu_cpus}", f"--mem={args.gpu_mem}",
            f"--time={args.gpu_time}", "--kill-on-bad-exit=1"]


def write_launch(args, run: Path, env: dict[str, str], srun: list[str]) -> None:
    project = run / args.case
    vibesys_args = ["--project", str(project), "--config", str(project / "agent.toml"),
                    "--outer-loop", "agent", "--inner-loop", "multi-agent",
                    "--max-rounds", str(args.rounds), "--profiler", "ncu", "--local",
                    "--backend", "cuda", "--headless", "--task", args.case]
    for skill in args.skills:
        vibesys_args += ["--skills-dir", str(args.vibesys_skills / skill)]
    vibesys_args += ["--extra-skills", str(run / "trusted/megabench/skills")]
    fixed = {k: v for k, v in env.items() if k not in CREDENTIAL_ENV}
    key_file, key_env = credential(args)
    passed = " ".join(f"{k}={shlex.quote(v)}" for k, v in fixed.items())
    q = lambda path: shlex.quote(str(path))
    broker = [str(args.venv / "bin/python"), "-m", "megabench.integrations.gpu_broker", "serve",
              "--socket", str(run / "gpu/broker.sock"), "--requests", str(run / "state/gpu"),
              "--spec", str(run / "gpu-spec.json"), "--job-python", str(args.vibesys.parent / "python"),
              "--job-script", str(run / "trusted/megabench/integrations/gpu_job.py"),
              "--max-jobs", str(args.max_gpu_jobs), "--", *srun]
    timeout = args.timeout_seconds or args.rounds * 3600
    (run / "launch.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"key=$(<{q(key_file)})\n"
        f"mkdir -p {q(env['TORCH_EXTENSIONS_DIR'])} "
        f"{q(env['TRITON_CACHE_DIR'])} {q(env['VIBESYS_STATE_HOME'])}\n"
        f"cd {q(project)}\n"
        f"env -i {passed} PYTHONPATH={q(run / 'trusted')} "
        + " ".join(map(shlex.quote, broker)) + f" >> {q(run / 'gpu-broker.log')} 2>&1 &\n"
        "broker=$!\ntrap 'kill $broker 2>/dev/null || true' EXIT\n"
        f"for _ in $(seq 100); do [ -S {q(run / 'gpu/broker.sock')} ] && break; sleep 0.1; done\n"
        f"env -i {passed} {key_env}=\"$key\" \\\n  "
        + " ".join(NO_GPU) + " timeout --signal=TERM --kill-after=60s "
        f"{timeout}s {q(args.vibesys)} "
        + " ".join(shlex.quote(a) for a in vibesys_args) + "\n")
    (run / "launch.sh").chmod(0o700)


def preflight(args, run: Path, env: dict[str, str], claude: Path) -> bool:
    home = Path(os.environ.get("HOME", "~")).expanduser()
    # Mounting the venv creates its parent directories, so test for repo files.
    repos = {REPO, args.venv.resolve().parent}
    forbid = [str(home / ".claude"), str(home / ".claude.json"), *map(str, args.forbid)]
    forbid += [str(repo / name) for repo in sorted(repos) for name in ("megabench", "README.md")]
    forbid += [str(run / "trusted/megabench" / name) for name in DROP_DIRS]
    forbid += [*NO_GPU_PATHS, str(run / "state"), str(credential(args)[0].resolve())]
    require = [str(run / args.case / "TASK.md"), str(run / "trusted/megabench/cases.py"),
               str(args.venv.resolve() / "bin/python"), str(run / "gpu")]
    python = args.vibesys.parent / "python"
    # The same sandbox that gpu_job.py builds in each GPU job, seen from VibeSys's namespace.
    out = subprocess.run([*NO_GPU, str(python), str(run / "trusted/megabench/integrations/gpu_job.py"),
                          "probe", str(run / "gpu-spec.json"), *forbid, *require],
                         capture_output=True, text=True, check=False, env=env)
    if out.returncode:
        print(out.stderr[-3000:], file=sys.stderr)
        return False
    seen = json.loads(out.stdout.splitlines()[-1])["seen"]
    bad = [p for p in forbid if seen.get(p) != "absent"]
    bad += [p for p in require if seen.get(p) != "present"]
    for path in forbid:
        print(f"{'ok  ' if seen.get(path) == 'absent' else 'FAIL'} absent  {path}")
    for path in require:
        print(f"{'ok  ' if seen.get(path) == 'present' else 'FAIL'} present {path}")
    return not bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="new directory, outside any directory of earlier work")
    secret = parser.add_mutually_exclusive_group(required=True)
    secret.add_argument("--api-key-file", type=Path,
                        help="file with the API key for the agent account; read at launch")
    secret.add_argument("--oauth-token-file", type=Path,
                        help="file with a `claude setup-token` token for the agent's "
                             "subscription account; read at launch")
    parser.add_argument("--vibesys", type=Path, required=True, help="vibesys executable")
    parser.add_argument("--vibesys-skills", type=Path, required=True,
                        help="VibeSys preset skills root (resources/skills)")
    parser.add_argument("--skills", nargs="+", default=list(DEFAULT_SKILLS))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--regimes", type=Path)
    parser.add_argument("--ref", default="HEAD", help="git ref for the trusted snapshot")
    parser.add_argument("--venv", type=Path, default=REPO / ".venv")
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--timeout-seconds", type=int,
                        help="limit for the whole VibeSys run (default: one hour per round)")
    parser.add_argument("--gpu-partition", default="main")
    parser.add_argument("--gpu-time", default="00:30:00", help="time limit of each GPU job")
    parser.add_argument("--gpu-cpus", type=int, default=8)
    parser.add_argument("--gpu-mem", default="64G")
    parser.add_argument("--max-gpu-jobs", type=int, default=1,
                        help="GPU jobs from the agent queued or running at a time")
    parser.add_argument("--forbid", type=Path, action="append", default=[],
                        help="extra path that must be absent in the sandbox (repeat)")
    args = parser.parse_args()

    run = args.run_dir.resolve()
    if run.exists():
        parser.error(f"{run} exists; use a new directory for each run")
    key_file = credential(args)[0].resolve()
    if not key_file.is_file() or key_file.stat().st_mode & 0o077:
        parser.error("the credential file must be readable only by its owner (chmod 600)")
    if key_file.is_relative_to(run):
        parser.error("keep the key file outside the run directory")
    claude = shutil.which("claude")
    if not claude:
        parser.error("claude is not on PATH")
    for skill in args.skills:
        if not (args.vibesys_skills / skill / "SKILL.md").is_file():
            parser.error(f"no skill {skill} in {args.vibesys_skills}")

    args.venv = args.venv.absolute()
    build_trusted(args.ref, run / "trusted", args.venv.resolve())
    gpus = select_cases("all", [args.case])[0].gpus
    srun = srun_command(args, gpus)
    # Gates keep cached baseline timings in state/, which the agent cannot see;
    # the shared HOME would let the agent edit them.
    gate = ["env", *(f"-u{name}" for name in CREDENTIAL_ENV),
            f"MEGABENCH_BASELINE_CACHE={run / 'state/baseline-cache'}", *srun, "--job-name=vs-gate"]
    command = [str(run / "trusted/.venv/bin/python"), "-m",
               "megabench.integrations.make_agent_workspace", "--case", args.case,
               "--output", str(run / args.case), "--agent", "vibesys",
               "--benchmark-root", str(run / "trusted"),
               "--gpu-socket", str(run / "gpu/broker.sock"), "--gate-launcher", shlex.join(gate)]
    if args.regimes:
        command += ["--regimes", str(args.regimes.resolve())]
    subprocess.run(command, check=True, cwd=run / "trusted",
                   env=os.environ | {"PYTHONPATH": str(run / "trusted")})
    settings = run / "home/.claude/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"permissions": {"deny": list(DENIED_TOOLS)}}, indent=2))
    (run / "state").mkdir()
    (run / "gpu").mkdir(mode=0o700)
    env = agent_env(args, run, Path(claude), key="")
    # The sandbox of each GPU job; it holds no key.
    (run / "gpu-spec.json").write_text(json.dumps(
        {"env": {k: v for k, v in env.items() if k not in CREDENTIAL_ENV},
         "claude": str(claude), "project": str(run / args.case)}))
    write_launch(args, run, env, srun)
    if not preflight(args, run, env, Path(claude)):
        print("preflight failed; do not start the run", file=sys.stderr)
        return 1
    print("preflight passed. Start the run on this host (not in a Slurm job):\n"
          f"  setsid nohup {run}/launch.sh > {run}/vibesys.log 2>&1 < /dev/null &")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
