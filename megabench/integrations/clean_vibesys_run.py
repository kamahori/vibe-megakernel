"""Prepare an isolated VibeSys run for one MegaBench case.

The agent must see the task and nothing from earlier work. This script builds:

- ``trusted/``: ``megabench/`` from one git ref, without past-run docs, probes,
  submissions and examples, plus a ``.venv`` link to the benchmark venv;
- ``<case>/``: a fresh workspace built from ``trusted/``;
- ``home/``: an empty ``HOME`` for the agent CLI, with web tools denied, so no
  user config, memory, skills, plugins or transcripts are visible;
- ``launch.sh`` and ``job.sbatch``: start VibeSys with ``env -i`` (only the
  listed variables pass), the API key read from ``--api-key-file`` at launch,
  and explicit skills, so preset skills outside the list do not load.

It then runs a preflight inside the VibeSys sandbox: forbidden paths must be
absent and the workspace and trusted root must be present. Submit
``job.sbatch`` only after the preflight passes.
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

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS = ("kernel-correctness", "kernel-ncu-analysis", "kernel-performance")
# Generic guidance stays; docs that report past runs or designs do not.
KEEP_DOCS = ("AGENT_TASK.md", "MODEL_STEP_TASKS.md", "REFERENCE_PRECISION.md",
             "SESSION_PROTOCOL.md")
DROP_DIRS = ("probes", "submissions", "examples", "experiments", "runs", "archive")
DENIED_TOOLS = ("WebFetch", "WebSearch")

PREFLIGHT = r"""
import json, os, subprocess, sys
from pathlib import Path
from vs_agent.host_resource_declarations import declare_agent_host_resources
from vs_sandbox.api import build_host_sandbox
spec = json.load(open(sys.argv[1]))
env = spec["env"]
res = declare_agent_host_resources(env, binary_path=spec["claude"], provider="claude")
sb = build_host_sandbox(spec["project"], env=env, resources=res, log=lambda _: None,
                        require_enforcement=True)
lines = [f'for p in "$@"; do if [ -e "$p" ]; then echo "present $p"; else echo "absent $p"; fi; done']
paths = spec["forbid"] + spec["require"]
out = subprocess.run(sb.wrap(["/bin/bash", "-c", lines[0], "probe", *paths]),
                     capture_output=True, text=True, timeout=120, env=env)
seen = dict(line.split(" ", 1)[::-1] for line in out.stdout.split("\n") if line)
print(json.dumps({"seen": seen, "stderr": out.stderr[-2000:]}))
"""


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


def agent_env(args, run: Path, claude: Path, key: str) -> dict[str, str]:
    """The complete environment VibeSys and the agent see; nothing else passes."""
    project = run / args.case
    return {
        "HOME": str(run / "home"), "USER": os.environ.get("USER", "agent"),
        "LOGNAME": os.environ.get("USER", "agent"), "LANG": "C.UTF-8", "TERM": "dumb",
        "PATH": ":".join((str(args.venv.resolve() / "bin"), str(claude.parent), str(args.cuda / "bin"),
                          "/usr/local/bin", "/usr/bin", "/bin")),
        "ANTHROPIC_API_KEY": key, "TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp",
        "NCU_PATH": str(args.cuda / "bin/ncu"),
        "TORCH_EXTENSIONS_DIR": str(project / ".torch_extensions"),
        "TRITON_CACHE_DIR": str(project / ".cache/triton"),
        "VIBESYS_STATE_HOME": str(run / "state"),
        "VIBESYS_AGENT_SANDBOX_ALLOW": ":".join(
            (str(run / "trusted"), str(args.venv.resolve()), str(args.cuda))),
    }


def write_launch(args, run: Path, env: dict[str, str]) -> None:
    project = run / args.case
    vibesys_args = ["--project", str(project), "--config", str(project / "agent.toml"),
                    "--outer-loop", "agent", "--inner-loop", "multi-agent",
                    "--max-rounds", str(args.rounds), "--profiler", "ncu", "--local",
                    "--headless", "--task", args.case]
    for skill in args.skills:
        vibesys_args += ["--skills-dir", str(args.vibesys_skills / skill)]
    vibesys_args += ["--extra-skills", str(run / "trusted/megabench/skills")]
    fixed = {k: v for k, v in env.items() if k != "ANTHROPIC_API_KEY"}
    passed = " ".join(f"{k}={shlex.quote(v)}" for k, v in fixed.items())
    (run / "launch.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"key=$(<{shlex.quote(str(args.api_key_file))})\n"
        f"mkdir -p {shlex.quote(env['TORCH_EXTENSIONS_DIR'])} "
        f"{shlex.quote(env['TRITON_CACHE_DIR'])} {shlex.quote(env['VIBESYS_STATE_HOME'])}\n"
        f"cd {shlex.quote(str(project))}\n"
        f"exec env -i {passed} CUDA_VISIBLE_DEVICES=\"${{CUDA_VISIBLE_DEVICES:-}}\" "
        "ANTHROPIC_API_KEY=\"$key\" \\\n  timeout --signal=TERM --kill-after=60s "
        f"{args.timeout_seconds}s {shlex.quote(str(args.vibesys))} "
        + " ".join(shlex.quote(a) for a in vibesys_args) + "\n")
    (run / "launch.sh").chmod(0o700)
    (run / "job.sbatch").write_text(
        "#!/bin/bash\n"
        f"#SBATCH --partition={args.partition}\n#SBATCH --gres=gpu:1\n#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=128G\n"
        f"#SBATCH --time={args.time}\n#SBATCH --chdir={run}\n"
        f"#SBATCH --output={run}/job-%j.log\n"
        f"exec {run}/launch.sh\n")


def preflight(args, run: Path, env: dict[str, str], claude: Path) -> bool:
    home = Path(os.environ.get("HOME", "~")).expanduser()
    # Mounting the venv creates its parent directories, so test for repo files.
    repos = {REPO, args.venv.resolve().parent}
    forbid = [str(home / ".claude"), str(home / ".claude.json"), *map(str, args.forbid)]
    forbid += [str(repo / name) for repo in sorted(repos) for name in ("megabench", "README.md")]
    forbid += [str(run / "trusted/megabench" / name) for name in DROP_DIRS]
    require = [str(run / args.case / "TASK.md"), str(run / "trusted/megabench/cases.py"),
               str(args.venv / "bin/python")]
    spec = run / "state/preflight.json"
    spec.write_text(json.dumps({"env": env | {"ANTHROPIC_API_KEY": "unset-in-preflight"},
                                "claude": str(claude), "project": str(run / args.case),
                                "forbid": forbid, "require": require}))
    python = args.vibesys.parent / "python"
    out = subprocess.run([str(python), "-c", PREFLIGHT, str(spec)], capture_output=True,
                         text=True, check=False)
    spec.unlink()
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
    parser.add_argument("--api-key-file", type=Path, required=True,
                        help="file with the API key for the agent account; read at launch")
    parser.add_argument("--vibesys", type=Path, required=True, help="vibesys executable")
    parser.add_argument("--vibesys-skills", type=Path, required=True,
                        help="VibeSys preset skills root (resources/skills)")
    parser.add_argument("--skills", nargs="+", default=list(DEFAULT_SKILLS))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--regimes", type=Path)
    parser.add_argument("--ref", default="HEAD", help="git ref for the trusted snapshot")
    parser.add_argument("--venv", type=Path, default=REPO / ".venv")
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--timeout-seconds", type=int, default=3300)
    parser.add_argument("--partition", default="main")
    parser.add_argument("--time", default="01:00:00")
    parser.add_argument("--forbid", type=Path, action="append", default=[],
                        help="extra path that must be absent in the sandbox (repeat)")
    args = parser.parse_args()

    run = args.run_dir.resolve()
    if run.exists():
        parser.error(f"{run} exists; use a new directory for each run")
    key_file = args.api_key_file.resolve()
    if not key_file.is_file() or key_file.stat().st_mode & 0o077:
        parser.error("--api-key-file must be a file readable only by its owner (chmod 600)")
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
    command = [str(run / "trusted/.venv/bin/python"), "-m",
               "megabench.integrations.make_agent_workspace", "--case", args.case,
               "--output", str(run / args.case), "--agent", "vibesys",
               "--benchmark-root", str(run / "trusted")]
    if args.regimes:
        command += ["--regimes", str(args.regimes.resolve())]
    subprocess.run(command, check=True, cwd=run / "trusted",
                   env=os.environ | {"PYTHONPATH": str(run / "trusted")})
    settings = run / "home/.claude/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"permissions": {"deny": list(DENIED_TOOLS)}}, indent=2))
    (run / "state").mkdir()
    env = agent_env(args, run, Path(claude), key="")
    write_launch(args, run, env)
    if not preflight(args, run, env, Path(claude)):
        print("preflight failed; do not submit", file=sys.stderr)
        return 1
    print(f"preflight passed. Submit from a neutral directory: cd / && sbatch {run}/job.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
