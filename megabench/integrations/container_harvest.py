"""Host-side bookkeeping of a container_run run (stdlib only; copied to RUN/control/).

Nothing here executes what the agent wrote: it reads JSON reports, moves tarballs and
writes JSON. The tarballs are extracted later in a container without network.

    harvest.py harvest RUN       move passing checks' source tarballs to RUN/snapshots/
    harvest.py candidates RUN    print up to 3 snapshot tarballs to rescore, best first
    harvest.py review RUN DIR    static source review of an extracted snapshot (exit 1: fail)
    harvest.py budget RUN N      print this round's agent seconds (0: the budget is spent)
                                 and write RUN/control/round.json for ./time_left

The workspace's ./check_candidate tars the sources to WS/.harness/pending/<id>.tgz before
the check and writes WS/reports/check-<id>.jsonl after it. A check passes when its record's
status is ok or ok_provisional; its tarball becomes RUN/snapshots/NNN-<id>.tgz and a line in
RUN/snapshots/index.jsonl with the agent-measured CUDA-event p50.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

PASSING = ("ok", "ok_provisional")
# A check whose report never appears within this long is logged as incomplete.
STALE_SECONDS = 3 * 3600
# Static review: what a submission's sources must not do (rescore.py/review.py of
# scaffold-search): load tensors from files, or reach the reference or the oracle.
FORBIDDEN = (
    (re.compile(r"\btorch\.load\s*\("), "torch.load"),
    (re.compile(r"\bnp\.load\s*\(|\bnumpy\.load\s*\("), "numpy.load"),
    (re.compile(r"\bsafetensors\b"), "safetensors"),
    (re.compile(r"\bpickle\b"), "pickle"),
    (re.compile(r"\.fromfile\s*\("), "fromfile"),
    (re.compile(r"^\s*(from|import)\s+megabench\b", re.M), "imports megabench"),
    (re.compile(r"^\s*(from|import)\s+reference\b", re.M), "imports reference"),
    (re.compile(r"\boracle\b", re.I), "mentions an oracle"),
)
SOURCE_SUFFIXES = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".c", ".sh")


def _env(run: Path) -> dict:
    out = {}
    for line in (run / "run.env").read_text().splitlines():
        key, _, value = line.partition("=")
        out[key] = value.strip("'")
    return out


def _record(path: Path) -> dict | None:
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def p50(record: dict) -> float | None:
    timing = record.get("candidate_timing") or {}
    return timing.get("cuda_event_p50_ms")


def launches_ok(record: dict) -> bool:
    audit = record.get("launch_audit") or {}
    ranks = audit.get("per_rank") or [audit]
    return audit.get("status") == "within_budget" and all(
        1 <= r.get("gpu_kernel_count", 0) <= r.get("max_gpu_launches", 1) for r in ranks)


def harvest(run: Path) -> int:
    ws = Path(_env(run)["WS"])
    pending, reports = ws / ".harness/pending", ws / "reports"
    snaps = run / "snapshots"
    snaps.mkdir(exist_ok=True)
    index, checks = snaps / "index.jsonl", run / "logs/checks.jsonl"
    seen = {json.loads(line)["id"] for line in checks.read_text().splitlines()} if checks.exists() else set()
    moved = 0
    for tarball in sorted(pending.glob("*.tgz")) if pending.is_dir() else []:
        check_id = tarball.name[:-4]
        if check_id in seen:
            continue
        record = _record(reports / f"check-{check_id}.jsonl")
        if record is None:
            if time.time() - tarball.stat().st_mtime < STALE_SECONDS:
                continue
            status = "incomplete"
        else:
            status = record.get("status", "unknown")
        row = {"id": check_id, "status": status, "agent_p50_ms": p50(record or {}),
               "report_time": (reports / f"check-{check_id}.jsonl").stat().st_mtime
               if record else None, "tar_time": tarball.stat().st_mtime,
               "harvested": time.time()}
        if status in PASSING:
            number = sum(1 for _ in snaps.glob("*.tgz")) + 1
            target = snaps / f"{number:03d}-{check_id}.tgz"
            shutil.copyfile(tarball, target)
            row["snapshot"] = target.name
            row["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            shutil.copyfile(reports / f"check-{check_id}.jsonl", snaps / f"{target.stem}.jsonl")
            with index.open("a") as out:
                out.write(json.dumps(row) + "\n")
            moved += 1
        with checks.open("a") as out:
            out.write(json.dumps(row) + "\n")
        seen.add(check_id)
    return moved


def candidates(run: Path, limit: int = 3) -> list[Path]:
    index = run / "snapshots/index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []
    rows = [r for r in rows if r.get("agent_p50_ms")]
    rows.sort(key=lambda r: r["agent_p50_ms"])
    out, digests = [], set()
    for row in rows:
        if row["sha256"] in digests:
            continue
        digests.add(row["sha256"])
        out.append(run / "snapshots" / row["snapshot"])
        if len(out) == limit:
            break
    return out


def review(directory: Path) -> list[str]:
    problems = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            problems.append(f"{path.relative_to(directory)}: symlink")
            continue
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        for pattern, what in FORBIDDEN:
            if pattern.search(text):
                problems.append(f"{path.relative_to(directory)}: {what}")
    return problems


def budget(run: Path, number: int) -> int:
    env = _env(run)
    total = int(env["BUDGET_SECONDS"])
    slot = int(env["AGENT_SECONDS"])
    used = sum(int(p.read_text().split()[0]) for p in (run / "state").glob("round-*.elapsed"))
    left = max(0, total - used)
    seconds = min(slot, left)
    now = int(time.time())
    (run / "control/round.json").write_text(json.dumps({
        "round": number, "start": now, "slot_seconds": seconds, "budget_seconds": total,
        "used_seconds_before": used, "last_slot": left <= slot}) + "\n")
    return seconds if seconds >= 120 else 0


def main(argv: list[str]) -> int:
    action, run = argv[0], Path(argv[1]).resolve()
    if action == "harvest":
        print(harvest(run))
    elif action == "candidates":
        print("\n".join(map(str, candidates(run))))
    elif action == "review":
        problems = review(Path(argv[2]))
        print(json.dumps(problems))
        return 1 if problems else 0
    elif action == "budget":
        print(budget(run, int(argv[2])))
    else:
        raise SystemExit(f"unknown action {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
