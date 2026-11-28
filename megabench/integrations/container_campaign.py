"""A serial campaign of container_run runs: one Slurm job at a time, each submitting the next.

    container_campaign.py add QUEUE RUN...     append prepared runs to the queue (a text file
                                               of run directories) and point them at it
    container_campaign.py next QUEUE           keep up to QUEUE.max-active runs (default 1) with a
                                               queued or running job (each round runs this on exit)
    container_campaign.py status QUEUE         each run's rounds, agent time and state

The next round is the first unfinished run's next round: a run goes on, one round at a time,
until its agent budget is spent or the agent ends its session, and then the queue moves on.
Rounds stopped by a usage limit of the agent's account (RUN/state/round-N.limited, from
container_round_apptainer.sh) count only their time until the refusal; the next round is
submitted with sbatch --begin at the moment the limit lifts, so no node is held while waiting.
Touch QUEUE.stop to stop submitting.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Wait this long after a limit's reset time before starting again.
RESET_MARGIN = 180
# A run whose setup failed this often, or that was refused this many rounds in a row with under
# 10 minutes of work each, is skipped: something is wrong that waiting does not fix.
MAX_SETUP_FAILURES = 3
MAX_IDLE_LIMITED = 5


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def runs(queue: Path) -> list[Path]:
    return [Path(line.strip()) for line in queue.read_text().splitlines() if line.strip()]


def env(run: Path) -> dict[str, str]:
    values = {}
    for line in (run / "run.env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value.strip("'\"")
    return values


def login_expired(run: Path, n: int) -> bool:
    """Whether round n stopped at once because the login it was given had expired (container_entry.sh):
    not a fault of the run, so it does not count towards MAX_SETUP_FAILURES."""
    err = run / "logs" / f"agent-round-{n}.err"
    text = err.read_text(errors="replace") if err.is_file() else ""
    return "access token has expired" in text or "OAuth token revoked" in text


def host_login_expires(run: Path) -> int:
    """When the access token of the run's credentials file expires (0: unknown)."""
    try:
        oauth = json.loads(Path(env(run)["CREDENTIAL_FILE"]).read_text())["claudeAiOauth"]
        return int(oauth["expiresAt"] / 1000)
    except (KeyError, OSError, ValueError, TypeError):
        return 0


def rounds(run: Path) -> list[dict]:
    out = []
    for started in sorted((run / "state").glob("round-*.started"),
                          key=lambda p: int(p.name.split("-")[1].split(".")[0])):
        n = int(started.name.split("-")[1].split(".")[0])
        state = run / "state"
        elapsed = state / f"round-{n}.elapsed"
        limited = state / f"round-{n}.limited"
        seconds, rc = (elapsed.read_text().split() + ["", ""])[:2] if elapsed.is_file() else ("", "")
        out.append({
            "n": n, "seconds": int(seconds) if seconds else None, "rc": rc,
            "limited": limited.read_text().split() if limited.is_file() else None,
            "setup_failed": (state / f"round-{n}.setup-failed").exists()
                            and not login_expired(run, n)})
    return out


def standing(run: Path) -> tuple[bool, str]:
    """(done, why) for a run."""
    if not (run / "config.json").is_file():
        return True, "not prepared"
    rs = rounds(run)
    used = sum(r["seconds"] or 0 for r in rs)
    budget = int(env(run).get("BUDGET_SECONDS", "0"))
    finished = run / "state/finished"
    if finished.exists():
        return True, f"finished: {finished.read_text().strip()}"
    # The last 15 minutes are not resumed: the agent's goal loop stops then (container_entry.sh),
    # and a round for them would only resume the session to end it again.
    if budget - used <= 900:
        return True, f"budget spent ({used // 60} of {budget // 60} min)"
    if sum(r["setup_failed"] for r in rs) >= MAX_SETUP_FAILURES:
        return True, f"setup failed {MAX_SETUP_FAILURES} times; skipped"
    tail = 0
    for r in reversed(rs):
        if r["limited"] and (r["seconds"] or 0) < 600:
            tail += 1
        else:
            break
    if tail >= MAX_IDLE_LIMITED:
        return True, f"refused by a usage limit {tail} rounds in a row; skipped"
    return False, f"{len(rs)} round(s), {used // 60} of {budget // 60} min used"


def our_jobs(exclude: str | None) -> dict[Path, str]:
    """The campaign's queued or running round jobs, by run directory (from the job's command,
    RUN/round.sh RUN round N), leaving out the job `exclude` (the one that is ending)."""
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%o"], check=True,
                         capture_output=True, text=True).stdout
    jobs = {}
    for line in out.splitlines():
        job, name, state, command = (line.split("|", 3) + [""] * 4)[:4]
        words = command.split()
        if job == exclude or len(words) < 3 or words[2] != "round":
            continue
        jobs[Path(words[1]).resolve()] = f"{job} {name} {state}"
    return jobs


def max_active(queue: Path) -> int:
    """How many runs of the queue may have a job at once: QUEUE.max-active (default 1)."""
    path = Path(f"{queue}.max-active")
    return int(path.read_text().strip()) if path.is_file() else 1


def lifts_at(queue: Path) -> int:
    """When the newest usage limit hit by any run of the queue lifts (0: none pending). Only each
    run's latest round counts: a round that started after a refusal and was not refused shows
    that the limit was lifted (or the login changed to an account with usage left)."""
    latest = 0
    for run in runs(queue):
        if (run / "state").is_dir():
            rs = rounds(run)
            if rs and rs[-1]["limited"]:
                latest = max(latest, int(rs[-1]["limited"][0]))
    return latest + RESET_MARGIN if latest + RESET_MARGIN > time.time() else 0


def add(args: argparse.Namespace) -> int:
    queue = args.queue.resolve()
    present = runs(queue) if queue.exists() else []
    with queue.open("a") as out:
        for run in args.runs:
            run = run.resolve()
            if not (run / "run.env").is_file():
                raise SystemExit(f"{run} is not a prepared run")
            lines = [line for line in (run / "run.env").read_text().splitlines()
                     if not line.startswith(("CAMPAIGN=", "CAMPAIGN_DRIVER="))]
            lines += [f"CAMPAIGN={queue}", f"CAMPAIGN_DRIVER={Path(__file__).resolve()}"]
            (run / "run.env").write_text("\n".join(lines) + "\n")
            if run not in present:
                out.write(f"{run}\n")
                present.append(run)
    log(f"queue {queue}: {len(present)} run(s)")
    return 0


def next_round(args: argparse.Namespace) -> int:
    """Submit one round for each open run without a job, while fewer than max_active runs have
    one. Every round job runs this as it ends, so the queue keeps max_active runs going."""
    queue = args.queue.resolve()
    with open(f"{queue}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if Path(f"{queue}.stop").exists():
            log(f"{queue}.stop exists; not submitting")
            return 0
        # A job of a run that is done is ending (several rounds can end at once): not busy.
        busy = {run: job for run, job in our_jobs(args.from_job).items() if not standing(run)[0]}
        limit = max_active(queue)
        rc, submitted, left = 0, 0, 0
        for run in runs(queue):
            done, why = standing(run)
            if done:
                continue
            left += 1
            if run.resolve() in busy or len(busy) >= limit:
                continue
            begin = lifts_at(queue)
            expires = host_login_expires(run)
            if expires and expires < time.time() + 600:
                # Nothing on the cluster can renew the login (only a claude session of the user
                # can): try again in an hour rather than start a round that cannot log in.
                begin = max(begin, int(time.time()) + 3600)
                log(f"the login expired at {time.strftime('%H:%M', time.localtime(expires))}; "
                    f"run claude on the login node to renew it")
            command = [sys.executable, "-m", "megabench.integrations.container_run", "submit",
                       "--run-dir", str(run), "--rounds", "1"]
            if begin:
                command += ["--begin", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(begin))]
            out = subprocess.run(command, capture_output=True, text=True, cwd=HERE.parents[1])
            when = f" at {time.strftime('%Y-%m-%d %H:%M', time.localtime(begin))} (usage limit)" if begin else ""
            errors = " ".join(l for l in out.stderr.splitlines() if not l.startswith("sbatch: "))
            log(f"{run.name} ({why}): submitted{when}: {out.stdout.strip()} {errors}".rstrip())
            if out.returncode:
                rc = out.returncode
                continue
            busy[run.resolve()] = "submitted"
            submitted += 1
        if not left:
            log("every run of the queue is done")
        elif not submitted:
            log(f"{len(busy)} of at most {limit} runs have a job; nothing to submit")
        return rc


def status(args: argparse.Namespace) -> int:
    queue = args.queue.resolve()
    for run in runs(queue):
        done, why = standing(run)
        detail = []
        for r in rounds(run) if (run / "state").is_dir() else []:
            mark = "L" if r["limited"] else ("S" if r["setup_failed"] else "")
            detail.append(f"r{r['n']}:{(r['seconds'] or 0) // 60}m{mark}")
        print(f"{'done ' if done else 'open '} {run.name:45} {why}  {' '.join(detail)}")
    lift = lifts_at(queue)
    if lift:
        print(f"waiting for a usage limit to lift at {time.strftime('%Y-%m-%d %H:%M', time.localtime(lift))}")
    print(f"at most {max_active(queue)} runs at once")
    for run, job in our_jobs(None).items():
        print(f"job {job}  {run.name}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("add")
    a.add_argument("queue", type=Path)
    a.add_argument("runs", type=Path, nargs="+")
    n = sub.add_parser("next")
    n.add_argument("queue", type=Path)
    n.add_argument("--from-job", help="the job running this (not counted as queued)")
    s = sub.add_parser("status")
    s.add_argument("queue", type=Path)
    args = parser.parse_args(argv)
    return {"add": add, "next": next_round, "status": status}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
