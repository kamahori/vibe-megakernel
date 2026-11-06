"""Run one agent command in the agent's own VibeSys sandbox.

``gpu_broker`` starts this file in each Slurm job, with the VibeSys Python
(it imports ``vs_agent`` and ``vs_sandbox``). Slurm starts the job outside the
agent sandbox, so this file builds the same sandbox again before it runs the
agent's command: the GPU is the only thing the command gets in addition.

``probe`` builds the same sandbox and prints, for each path, whether the
sandbox can see it. ``clean_vibesys_run`` uses it as the preflight.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from vs_agent.host_resource_declarations import declare_agent_host_resources
from vs_sandbox.api import build_host_sandbox

# Runs the command in the caller's directory: the sandbox starts in the project root.
_IN_CWD = 'cd -- "$1" && shift && exec "$@"'


def sandbox(spec: dict):
    env = spec["env"]
    resources = declare_agent_host_resources(env, binary_path=spec["claude"], provider="claude")
    return build_host_sandbox(spec["project"], env=env, resources=resources,
                              log=lambda _: None, require_enforcement=True)


def probe(spec: dict, paths: list[str]) -> int:
    script = 'for p in "$@"; do if [ -e "$p" ]; then echo "present $p"; else echo "absent $p"; fi; done'
    out = subprocess.run(sandbox(spec).wrap(["/bin/bash", "-c", script, "probe", *paths]),
                         capture_output=True, text=True, timeout=120, env=spec["env"])
    seen = dict(line.split(" ", 1)[::-1] for line in out.stdout.split("\n") if line)
    print(json.dumps({"seen": seen, "stderr": out.stderr[-2000:]}))
    return 0


def run(spec: dict, request: dict) -> int:
    project = Path(spec["project"])
    cwd = Path(request["cwd"])
    if cwd != project and project not in cwd.parents:
        cwd = project  # the job sandbox holds only the project
    env = dict(request["env"])
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if name in os.environ:
            env[name] = os.environ[name]
        else:
            env.pop(name, None)
    argv = sandbox(spec).wrap(["/bin/bash", "-c", _IN_CWD, "gpu_run", str(cwd), *request["argv"]])
    return subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, check=False).returncode


def main(argv: list[str]) -> int:
    mode, spec_path, *rest = argv
    spec = json.loads(Path(spec_path).read_text())
    if mode == "probe":
        return probe(spec, rest)
    if mode == "run":
        return run(spec, json.loads(Path(rest[0]).read_text()))
    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
