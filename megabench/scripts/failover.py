#!/usr/bin/env python3
"""Move a VibeSys run to another account when its account reaches a usage limit.

  setsid nohup python3 megabench/scripts/failover.py RUN_NAME CASE REF TOTAL_ROUNDS \
      --seed-from WS --seed-ref COMMIT [--near-min 30] >> $VS_LOOP_HOME/runs/failover.log 2>&1 &

Every two minutes the script reads `cswap list --json` (usage only; it never reads or
prints a credential). When the run's account is at a limit (5-hour or 7-day usage at or
above --limit-pct, or a usage-limit line in vibesys.log) and that limit does not reset
within --near-min minutes, the script stops the run, cancels its GPU jobs and prepares
a new run with prep.sh (next to this script) --restart from the last "multi: close round K" commit of
the stopped run, on the account with the most capacity. TOTAL_ROUNDS counts the whole
chain, so a restart runs only the rounds that remain. A limit that resets soon is waited
out. Never used: the account of the main Claude session (cswap's active account) and the
accounts listed in VS_NEVER_ACCOUNTS (comma-separated e-mail addresses).

Token files are VS_TOKEN_DIR/vibesys_oauth_token_<name> (default ~/.secrets), where <name>
is the account's e-mail name without a "-vibesys" suffix; the script uses them by path only.
Runs live in $VS_LOOP_HOME/runs (see env.sh).

--seed-from/--seed-ref/TOTAL_ROUNDS describe how the current run was prepared (used again
when the run stops before it closes a round).

A run that ends for another reason (its time limit, a crash) after it closed round K but
before round TOTAL_ROUNDS continues the same way from round K, on the same account when it
has capacity, at most --max-continues times.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("VS_REPO") or HERE.parents[1])
B = Path(os.environ.get("VS_LOOP_HOME") or REPO.parent)
SECRETS = Path(os.environ.get("VS_TOKEN_DIR") or Path.home() / ".secrets")
NEVER = {a.strip() for a in os.environ.get("VS_NEVER_ACCOUNTS", "").split(",") if a.strip()}
LIMIT_LINE = re.compile(r"usage limit reached|hit your limit|rate_limit_error|limit reached", re.I)
SECRET_WORDS = re.compile(r"token|secret|oauth|key", re.I)


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


def usage() -> tuple[int | None, list[dict]]:
    out = subprocess.run(["cswap", "list", "--json"], capture_output=True, text=True, timeout=120)
    data = json.loads(out.stdout)
    rows = []
    for a in data.get("accounts", []):
        u = a.get("usage") or a.get("lastGoodUsage") or {}
        def window(name):
            w = u.get(name) or {}
            reset = w.get("resetsAt")
            mins = ((datetime.fromisoformat(reset) - datetime.now(timezone.utc)).total_seconds() / 60
                    if reset else None)
            return w.get("pct"), mins
        p5, r5 = window("fiveHour")
        p7, r7 = window("sevenDay")
        rows.append({"number": a.get("number"), "email": a.get("email", ""), "p5": p5, "r5": r5,
                     "p7": p7, "r7": r7, "ok": a.get("usage") is not None})
    return data.get("activeAccountNumber"), rows


def token_file(email: str) -> Path | None:
    name = email.split("@")[0].removesuffix("-vibesys")
    path = SECRETS / f"vibesys_oauth_token_{name}"
    return path if path.is_file() else None


def account_of(token: Path, rows: list[dict]) -> dict | None:
    return next((r for r in rows if token_file(r["email"]) == token), None)


def blocked(acc: dict, limit: float, near: float, log_hit: bool) -> tuple[bool, float | None]:
    """(at a limit, minutes until it lifts)."""
    hits = [(p, r) for p, r in ((acc["p5"], acc["r5"]), (acc["p7"], acc["r7"]))
            if p is not None and p >= limit]
    if not hits and log_hit:
        # A log line alone can be agent text: it counts only near a measured limit.
        hits = [(p, r) for p, r in ((acc["p5"], acc["r5"]), (acc["p7"], acc["r7"]))
                if p is not None and p >= 90]
    if not hits:
        return False, None
    return True, max((r or 0) for _, r in hits)


def choose(rows: list[dict], active: int | None, exclude: set[str]) -> dict | None:
    pool = [r for r in rows if r["ok"] and r["number"] != active and r["email"] not in NEVER
            and r["email"] not in exclude and token_file(r["email"]) is not None
            and (r["p5"] or 0) < 80 and (r["p7"] or 0) < 90]
    return min(pool, key=lambda r: (max(r["p5"] or 0, r["p7"] or 0), r["p7"] or 0), default=None)


def running(run: Path) -> bool:
    # Match the launch script process only (its own command line, or the shell that runs
    # it), not other commands that merely mention its path.
    pattern = rf"^(\S*/)?(ba)?sh {re.escape(str(run))}/launch\.sh$|^{re.escape(str(run))}/launch\.sh$"
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def stop(run: Path) -> None:
    out = subprocess.run(["ps", "-eo", "pgid=,args="], capture_output=True, text=True).stdout
    groups = {int(l.split()[0]) for l in out.splitlines() if f"runs/{run.name}/" in l and "failover.py" not in l}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for g in groups:
            try:
                os.killpg(g, sig)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(10)
    tags = set(re.findall(r"vs-gpu-[0-9a-f]+-", (run / "gpu-broker.log").read_text())
               if (run / "gpu-broker.log").exists() else [])
    jobs = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %j"],
                          capture_output=True, text=True).stdout.splitlines()
    ids = [j.split()[0] for j in jobs if any(t in j for t in tags)]
    if ids:
        subprocess.run(["scancel", *ids])
    log(f"stopped {run.name}; cancelled {len(ids)} GPU jobs")


def last_close(ws: Path) -> tuple[str, int] | None:
    out = subprocess.run(["git", "log", "--format=%H %s"], cwd=ws, capture_output=True,
                         text=True).stdout
    for line in out.splitlines():
        sha, _, subject = line.partition(" ")
        m = re.fullmatch(r"multi: close round (\d+)", subject)
        if m:
            return sha, int(m.group(1))
    return None


def prep_and_launch(name: str, case: str, ref: str, token: Path, rounds: int, seed_from: Path,
                    seed_ref: str, extra: str) -> Path:
    if not str(seed_from):
        # A cold run that closed no round starts cold again: nothing to restart from.
        extra = " ".join(a for a in extra.split() if a != "--restart")
    env = os.environ | {"SEED_FROM": str(seed_from), "SEED_REF": seed_ref, "EXTRA_ARGS": extra}
    out = subprocess.run(["bash", str(HERE / "prep.sh"), name, case, ref, str(token),
                          str(rounds)], env=env, capture_output=True, text=True)
    for line in (out.stdout + out.stderr).splitlines():
        if re.search(r"restart|seed:|preflight|rror|FAIL", line) and not SECRET_WORDS.search(line):
            log(f"prep: {line.strip()}")
    run = B / "runs" / name
    if out.returncode != 0 or "preflight passed" not in out.stdout:
        raise RuntimeError(f"prep of {name} failed")
    subprocess.Popen(["setsid", "nohup", str(run / "launch.sh")], stdout=open(run / "vibesys.log", "w"),
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log(f"launched {name} ({rounds} rounds as a whole-run count, token file {token.name})")
    return run


def next_name(name: str) -> str:
    m = re.fullmatch(r"(.*-s)(\d+)", name)
    base, n = (m.group(1), int(m.group(2))) if m else (name + "-s", 1)
    while (B / "runs" / f"{base}{n + 1}").exists():
        n += 1
    return f"{base}{n + 1}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run"); p.add_argument("case"); p.add_argument("ref")
    p.add_argument("total_rounds", type=int)
    p.add_argument("--seed-from", default="", help="empty: the run started cold")
    p.add_argument("--seed-ref", default="")
    p.add_argument("--extra", default="--phase-agents --restart")
    p.add_argument("--near-min", type=float, default=30.0)
    p.add_argument("--limit-pct", type=float, default=98.0)
    p.add_argument("--poll-s", type=float, default=120.0)
    p.add_argument("--max-continues", type=int, default=2,
                   help="continue a run that ended early (not at a limit) at most this often")
    p.add_argument("--dry-run", action="store_true", help="report the decision once and exit")
    a = p.parse_args()
    run = B / "runs" / a.run
    ws = run / a.case
    token = Path(re.search(r"--oauth-token-file\s+(\S+)", (run / "prep.args").read_text()).group(1)) \
        if (run / "prep.args").exists() else None
    if token is None:
        launch = (run / "launch.sh").read_text()
        m = re.search(r"([^\s'\"<(=]*/vibesys_oauth_token\w*)", launch)
        token = Path(m.group(1)) if m else None
    if token is None:
        log("cannot tell which token file the run uses; stop")
        return 2
    seen = (run / "vibesys.log").stat().st_size if (run / "vibesys.log").exists() else 0
    continues = 0
    log(f"watching {a.run} (token file {token.name}); total rounds {a.total_rounds}")
    while True:
        active, rows = usage()
        acc = account_of(token, rows)
        text = ""
        if (run / "vibesys.log").exists():
            with open(run / "vibesys.log", errors="replace") as f:
                f.seek(seen); text = f.read(); seen = f.tell()
        log_hit = any(LIMIT_LINE.search(l) for l in text.splitlines()
                      if not l.lstrip().startswith(("[Implementer", "[Orchestrator", "[Profiler",
                                                     "[Judge", "[Phase", "[Microbench")))
        hit, lifts = blocked(acc, a.limit_pct, a.near_min, log_hit) if acc else (False, None)
        ended = not running(run)
        if a.dry_run:
            pick = choose(rows, active, {acc["email"]} if acc else set())
            log(f"dry run: account {acc and acc['email']} 5h {acc and acc['p5']}% 7d "
                f"{acc and acc['p7']}% at-limit={hit} lifts-in={lifts and round(lifts)} min "
                f"running={not ended}; next account would be {pick and pick['email']}")
            return 0
        if ended and not hit:
            closes = last_close(ws)
            done = closes[1] if closes else 0
            if closes and done < a.total_rounds and continues < a.max_continues:
                # Ended early (time limit or crash): continue the chain from round K.
                log(f"{a.run} ended after round {done} of {a.total_rounds}: continue from it")
                stop(run)  # leftovers: processes of the run, its GPU jobs
                if acc is None or max(acc["p5"] or 0, acc["p7"] or 0) >= a.limit_pct:
                    pick = choose(rows, active, {acc["email"]} if acc else set())
                    if pick is None:
                        log("no account has capacity to continue: wait")
                        time.sleep(a.poll_s)
                        continue
                    token = token_file(pick["email"])
                continues += 1
                name = next_name(a.run)
                run = prep_and_launch(name, a.case, a.ref, token, a.total_rounds, ws,
                                      closes[0], a.extra)
                a.run, ws, seen = name, run / a.case, 0
                a.seed_from, a.seed_ref = str(ws), closes[0]
                continue
            log(f"{a.run} ended (last close: round {done}); watcher stops")
            return 0
        if hit and lifts is not None and lifts <= a.near_min:
            log(f"{acc['email']} at a limit that lifts in {lifts:.0f} min: wait")
        elif hit:
            pick = choose(rows, active, {acc["email"]})
            if pick is None:
                log("at a limit and no other account has capacity: wait")
            else:
                log(f"{acc['email']} at a limit (5h {acc['p5']}%, 7d {acc['p7']}%, lifts in "
                    f"{lifts:.0f} min): move to {pick['email']}")
                if not ended:
                    stop(run)
                closes = last_close(ws)
                name = next_name(a.run)
                if closes:
                    # Continue the stopped run: the whole chain's round count, less its closed
                    # rounds (prep.sh --restart subtracts them).
                    seed_from, seed_ref = str(ws), closes[0]
                    rounds = a.total_rounds
                else:
                    # No round closed: prepare it again the way it was prepared.
                    seed_from, seed_ref, rounds = a.seed_from, a.seed_ref, a.total_rounds
                token = token_file(pick["email"])
                run = prep_and_launch(name, a.case, a.ref, token, rounds, seed_from, seed_ref,
                                      a.extra)
                a.run, ws, seen = name, run / a.case, 0
                a.seed_from, a.seed_ref, a.total_rounds = seed_from, seed_ref, rounds
                continue
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
