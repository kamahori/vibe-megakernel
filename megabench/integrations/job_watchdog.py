"""Stop a GPU job that hangs, before Slurm stops it at its time limit plus overtime.

``gpu_job.run`` starts the agent's command through ``run_watched``. Two rules:

* Idle rule: the harness (``megabench.heartbeat.beat``) writes a heartbeat file
  after each finished candidate call. When the file was written at least once
  and then stays unchanged for longer than the idle limit (or the longer
  ``allow`` of the last beat), the command is stopped. A command that never
  beats (an agent script without the harness) is not under the idle rule.
* Time rule: every command is stopped a short time before the job's Slurm time
  limit, so a hang never runs into the Slurm overtime.

A stopped command gets exit code ``HANG_EXIT`` and one ``gpu_run: HANG`` line on
stderr that says which rule stopped it and after which step.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HANG_EXIT = 124
IDLE_S = float(os.environ.get("MEGABENCH_HANG_S", 240))
LIMIT_MARGIN_S = 30.0
POLL_S = 2.0


def _beat(path: Path) -> tuple[float, dict] | None:
    try:
        mtime = path.stat().st_mtime
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return mtime, data if isinstance(data, dict) else {}


def _stop(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_watched(argv: list[str], env: dict, heartbeat: Path, *, idle_s: float = IDLE_S,
                limit_s: float | None = None, poll_s: float = POLL_S,
                stderr=None) -> int:
    stderr = stderr or sys.stderr
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.unlink(missing_ok=True)
    env = dict(env, MEGABENCH_HEARTBEAT=str(heartbeat))
    start = time.monotonic()
    deadline = start + limit_s - LIMIT_MARGIN_S if limit_s else None
    proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        while True:
            try:
                return proc.wait(timeout=poll_s)
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            reason = None
            if deadline is not None and now >= deadline:
                reason = (f"the job reached its time limit ({limit_s:.0f}s) less "
                          f"{LIMIT_MARGIN_S:.0f}s")
            seen = _beat(heartbeat)
            if reason is None and seen is not None:
                mtime, data = seen
                quiet = time.time() - mtime
                allowed = max(idle_s, float(data.get("allow") or 0))
                if quiet > allowed:
                    reason = (f"no progress for {quiet:.0f}s (limit {allowed:.0f}s) after "
                              f"'{data.get('label', '?')}'")
            if reason is not None:
                _stop(proc)
                print(f"gpu_run: HANG: {reason}; job stopped. The candidate or the command "
                      f"did not finish; find the step that does not end.", file=stderr,
                      flush=True)
                return HANG_EXIT
    finally:
        if proc.poll() is None:
            _stop(proc)
        heartbeat.unlink(missing_ok=True)
        for leftover in heartbeat.parent.glob(f"{heartbeat.name}.*"):
            leftover.unlink(missing_ok=True)
        try:
            heartbeat.parent.rmdir()
        except OSError:
            pass
