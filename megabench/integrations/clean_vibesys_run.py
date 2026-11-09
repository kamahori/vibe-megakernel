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
import re
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
                    "--max-rounds", str(args.rounds),
                    # Each approved candidate gets the official gates, so a failed final round
                    # cannot lose the gains of earlier rounds (only official rounds can win).
                    "--official-eval-every", "1", "--profiler", "ncu", "--local",
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
              "--max-jobs", str(args.max_gpu_jobs),
              "--hold-s", f"{getattr(args, 'gpu_hold_s', 0):g}", "--", *srun]
    timeout = args.timeout_seconds or args.rounds * 7200
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
        f"env -i {passed} {key_env}=\"$key\" VIBESYS_MEASUREMENT_DIR=reports "
        + ("VIBESYS_MICROBENCH_PHASE=1 " if getattr(args, "microbench_phase", False) else "")
        + (f"VIBESYS_MICROBENCH_MINUTES={args.microbench_minutes:g} "
           if getattr(args, "microbench_minutes", None) else "")
        + (f"VIBESYS_MICROBENCH_FULL_ROUNDS={full_rounds} "
           if (getattr(args, "microbench_phase", False) or getattr(args, "phase_agents", False))
           and (full_rounds := getattr(args, "microbench_full_rounds", None)) is not None
           and full_rounds >= 0 else "")
        + ("VIBESYS_PHASE_AGENTS=1 " if getattr(args, "phase_agents", False) else "")
        + f"VIBESYS_GPU_WAIT_FILE={q(run / 'gpu/queue_wait_s')} "
        + "\\\n  "
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


SEED_SKIP = (".build/", "progress/microbench/.build-cache/", "variants/")
ROUND_FILE = re.compile(r"(^|/)round-\d+")


def seed_workspace(workspace: Path, source: Path, ref: str) -> list[str]:
    """Copy the candidate and the progress files of commit *ref* of the workspace *source*.

    The new workspace keeps every file that the harness installed (TASK.md, tools, skills),
    except submission.py, which comes from the seed. Files under progress/ come from the seed.
    A progress file that belongs to one round of the earlier run (plans, microbench requests
    and answers, round notes) goes to progress/prior/, so that the new run's rounds do not
    find files of the same name. Build outputs are not copied.
    """
    names = subprocess.run(["git", "-C", str(source), "ls-tree", "-r", "-z", "--name-only", ref],
                           capture_output=True, check=True).stdout.decode().split("\0")
    names = list(filter(None, names))
    # Skill folders and agent-tool folders are installed by the harness and VibeSys.
    skills = {n.split("/", 1)[0] for n in names if n.count("/") == 1 and n.endswith("/SKILL.md")}
    copied = []
    for name in names:
        top = name.split("/", 1)[0]
        if name.startswith(SEED_SKIP) or ("/" in name and (top.startswith(".") or top in skills)):
            continue
        installed = (workspace / name).exists()
        if installed and name != "submission.py" and not name.startswith("progress/"):
            continue
        target = name
        if name.startswith("progress/") and ROUND_FILE.search(name):
            target = "progress/prior/" + name.removeprefix("progress/")
        data = subprocess.run(["git", "-C", str(source), "show", f"{ref}:{name}"],
                              capture_output=True, check=True).stdout
        path = workspace / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        copied.append(target)
    commit_workspace(workspace, f"Seed from {source.name} at {ref}")
    return copied


def commit_workspace(workspace: Path, message: str) -> None:
    """Commit what setup added: VibeSys starts only from a clean repository."""
    if not subprocess.run(["git", "status", "--porcelain"], cwd=workspace, capture_output=True,
                          text=True, check=True).stdout.strip():
        return
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(["git", "-c", "user.name=MegaBench", "-c", "user.email=megabench@example.invalid",
                    "commit", "-q", "-m", message], cwd=workspace, check=True)


def cache_rows(args, run: Path) -> int:
    """Copy the rows that earlier runs of the case measured into the new workspace."""
    from ..primdb import cache

    rows = cache.harvest(run.parent, args.case, exclude=(run.name,),
                         store=getattr(args, "microbench_cache_store", None))
    n = cache.write(rows, run / args.case / "progress/microbench/primdb_cache.jsonl")
    if getattr(args, "microbench_cache_store", None):
        cache.write(rows, args.microbench_cache_store)
    print(f"microbench cache: {n} measured rows of {args.case} from earlier runs")
    return n


def rounds_done(workspace: Path, ref: str) -> int | None:
    """K when *ref* of *workspace* is the commit that closes round K, else None."""
    out = subprocess.run(["git", "log", "-1", "--format=%s", ref], cwd=workspace,
                         capture_output=True, text=True)
    m = re.fullmatch(r"multi: close round (\d+)", out.stdout.strip())
    return int(m.group(1)) if out.returncode == 0 and m else None


def build_parser() -> argparse.ArgumentParser:
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
                        help="limit for the whole VibeSys run (default: two hours per round)")
    parser.add_argument("--gpu-partition", default="main")
    parser.add_argument("--gpu-time", default="00:20:00", help="time limit of each GPU job (default: %(default)s)")
    parser.add_argument("--gpu-cpus", type=int, default=8)
    parser.add_argument("--gpu-mem", default="64G")
    parser.add_argument("--gpu-hold-s", type=float, default=120,
                        help="the GPU broker keeps an allocation this many idle seconds after "
                             "a command, for the next one (0: one Slurm job per command)")
    parser.add_argument("--max-gpu-jobs", type=int, default=8,
                        help="GPU jobs from the agent queued or running at a time")
    parser.add_argument("--microbench-phase", action=argparse.BooleanOptionalAction, default=True,
                        help="each round plan is a draft: VibeSys runs the microbench phase on it "
                             "and the orchestrator revises the plan before the implementer starts "
                             "(default: on)")
    parser.add_argument("--microbench-minutes", type=float, default=30,
                        help="time limit of each pre-plan microbench phase; the agents get a "
                             "deadline and leave what they cannot measure open, and VibeSys "
                             "stops the phase a grace period later (default: %(default)s; "
                             "0: no limit)")
    parser.add_argument("--microbench-full-rounds", type=int,
                        help="rounds that run the full microbench phase; a later round runs it "
                             "only for open questions that the kernel timeline cannot settle "
                             "(-1: every round runs the full phase; default: 1, or 0 with "
                             "--seed-from, whose seed already has its opening rounds)")
    parser.add_argument("--phase-agents", action="store_true",
                        help="rounds after the full microbench rounds are phase rounds: the "
                             "implementer coordinates one phase subagent per target phase "
                             "(needs a round count: an unset or negative "
                             "--microbench-full-rounds means 2)")
    parser.add_argument("--seed-from", type=Path,
                        help="workspace of an earlier run: start from its candidate and progress "
                             "files at --seed-ref instead of an empty candidate")
    parser.add_argument("--seed-ref", default="HEAD",
                        help="commit of --seed-from to start from (default: %(default)s)")
    parser.add_argument("--microbench-cache", dest="microbench_cache", action="store_true",
                        default=False,
                        help="start with the measured rows of earlier runs of the case (numbers "
                             "only, no notes). Off by default: a run measures what it needs")
    parser.add_argument("--restart", action="store_true",
                        help="continue the run of --seed-from after the round that --seed-ref "
                             "closes ('multi: close round K'): --rounds and "
                             "--microbench-full-rounds count the whole run, and the K rounds "
                             "already done are subtracted")
    parser.add_argument("--no-microbench-cache", dest="microbench_cache", action="store_false",
                        help="start without cached rows (the default)")
    parser.add_argument("--microbench-cache-store", type=Path,
                        help="file that keeps cached rows after run directories are deleted")
    parser.add_argument("--forbid", type=Path, action="append", default=[],
                        help="extra path that must be absent in the sandbox (repeat)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.restart:
        if not args.seed_from:
            parser.error("--restart needs --seed-from")
        done = rounds_done(args.seed_from, args.seed_ref)
        if done is None:
            parser.error("--restart: --seed-ref is not a 'multi: close round K' commit")
        if args.rounds <= done:
            parser.error(f"--restart: the seed already did {done} of the {args.rounds} rounds")
        full = 1 if args.microbench_full_rounds is None else args.microbench_full_rounds
        args.rounds -= done
        args.microbench_full_rounds = max(full - done, 0) if full >= 0 else full
        print(f"restart after round {done}: {args.rounds} rounds remain, "
              f"{args.microbench_full_rounds} of them full microbench rounds")
    if args.microbench_full_rounds is None:
        args.microbench_full_rounds = 0 if args.seed_from else 1
    if args.phase_agents and args.microbench_full_rounds < 0:
        args.microbench_full_rounds = 2  # -1 means every round is a full round: no phase round
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
            f"MEGABENCH_BASELINE_CACHE={run / 'state/baseline-cache'}",
            f"MEGABENCH_OFFICIAL_TIMING_HISTORY={run / 'state/official-timing.jsonl'}",
            *srun, "--job-name=vs-gate"]
    command = [str(run / "trusted/.venv/bin/python"), "-m",
               "megabench.integrations.make_agent_workspace", "--case", args.case,
               "--output", str(run / args.case), "--agent", "vibesys",
               "--benchmark-root", str(run / "trusted"),
               "--gpu-socket", str(run / "gpu/broker.sock"), "--gate-launcher", shlex.join(gate)]
    if args.regimes:
        command += ["--regimes", str(args.regimes.resolve())]
    if args.phase_agents:
        command += ["--phase-agents"]
    subprocess.run(command, check=True, cwd=run / "trusted",
                   env=os.environ | {"PYTHONPATH": str(run / "trusted")})
    if args.seed_from:
        copied = seed_workspace(run / args.case, args.seed_from.resolve(), args.seed_ref)
        print(f"seed: {len(copied)} files from {args.seed_from} at {args.seed_ref}")
    if args.microbench_cache:
        cache_rows(args, run)
        commit_workspace(run / args.case, "Microbench rows of earlier runs of the case")
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
