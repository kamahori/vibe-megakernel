"""Scorecard of finished (or stopped) VibeSys runs: one row per run, read from its artifacts.

usage: python -m megabench.integrations.scorecard RUN_DIR... [--targets 8.0,7.5] [--json OUT]
       [--markdown OUT] [--tolerance-pct 1]

A run directory is what clean_vibesys_run made: <run>/<case>/ (the agent workspace, a git
repository), <run>/state/ (VibeSys logs), <run>/gpu-broker.log. Every input is optional; a
metric whose input is missing is None.

Definitions
- check: one result line of reports/*check*.jsonl with correctness "pass" and a launch audit in
  budget. Its time is the fast-cluster p50 of the stability file when one exists, else the
  cuda_event p50. A root check timed the workspace candidate; a variant check timed a copy
  in another directory (variants/...). submission_sha256 hashes submission.py only, not the
  kernel sources, so it does not name a code tree.
- kept tree of round N: the blob ids of the candidate files of the workspace at the commit
  "multi: close round N". When it is the same as at the previous close, kept_ms carries over.
  When it changed, kept_ms is the lowest root check of that round (slow modes only add time)
  whose submission_sha256 is the hash of submission.py at the close, and, when there are
  such checks, timed after the last commit of the round with a different candidate tree (the
  attempts before it timed other code). With no such check it is "-" (kept_unmeasured).
- kept_slower: a round whose kept tree changed and whose kept_ms is more than the tolerance
  above the previous kept_ms.
- rejected_faster: a passing check (root or variant) of a round, faster by more than the
  tolerance than the kept_ms at that close, for a round whose kept_ms is above it. It names a
  measured build that the round did not keep (a judge or merge decision to look at).
- minutes by agent: VibeSys phase_started/phase_finished pairs. An implementer phase that the
  next phase of the same round hands back to the orchestrator is the microbench phase (the
  plan is revised after it); the others are implementer work. Older runs ran the microbench
  work inside the implementer turn; their min_microbench is None.
- GPU minutes: the broker's "exit RC after Ns" lines; failed = nonzero exit; reruns = jobs whose
  command has --rerun-of.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CLOSE = re.compile(r"^multi: close round (\d+)$")
BROKER = re.compile(r"^(\S+) (vs-gpu-\S+) (start|exit) ?(.*)$")
EXIT = re.compile(r"(-?\d+) after ([\d.]+)s")


def ts(text: str) -> datetime:
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t if t.tzinfo else t.astimezone()  # broker stamps are local time


def workspace_of(run: Path) -> Path | None:
    for child in sorted(run.iterdir()):
        if (child / ".git").exists() and (child / "TASK.md").exists():
            return child
    return None


def git(ws: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *args], capture_output=True, text=True,
                          check=True).stdout


NOT_CANDIDATE = {"progress", "reports", "roadmap", "variants", "dev"}


def tree_at(ws: Path, commit: str) -> str:
    """Digest of the candidate files at *commit*: every tracked file except agent notes,
    measurements, variant copies and dot folders (VibeSys state, build outputs)."""
    listing = [line for line in git(ws, "ls-tree", "-r", commit).splitlines()
               if (top := line.split("\t", 1)[1].split("/", 1)[0]) not in NOT_CANDIDATE
               and not top.startswith(".")]
    return hashlib.sha256("\n".join(listing).encode()).hexdigest()


def sub_sha_at(ws: Path, commit: str) -> str | None:
    try:
        data = subprocess.run(["git", "-C", str(ws), "show", f"{commit}:submission.py"],
                              capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(data).hexdigest()


def commits(ws: Path) -> list[tuple[str, datetime, str]]:
    out = git(ws, "log", "--reverse", "--format=%H%x09%aI%x09%s")
    rows = []
    for line in out.splitlines():
        sha, when, subject = line.split("\t", 2)
        rows.append((sha, ts(when), subject))
    return rows


def checks(ws: Path) -> list[dict]:
    rows = []
    for path in sorted((ws / "reports").glob("*check*.jsonl")):
        stab = path.with_suffix(".stability.json")
        stability = json.loads(stab.read_text()) if stab.exists() else {}
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            timing = d.get("candidate_timing") or {}
            ok = ((d.get("correctness") or {}).get("status") == "pass"
                  and (d.get("launch_audit") or {}).get("status", "within_budget") == "within_budget")
            ms = stability.get("fast_p50_ms") or timing.get("cuda_event_p50_ms")
            sub = Path(d.get("submission") or "")
            # A run directory can move after the run, so compare names, not full paths.
            root = sub.parent.name == ws.name and "variants" not in sub.parts
            rows.append({"file": path.name, "root": root, "sub_sha": d.get("submission_sha256"),
                         "dir": str(sub.parent.relative_to(ws)) if sub.is_relative_to(ws) else str(sub.parent),
                         "ok": ok and ms is not None,
                         "ms": ms, "verdict": stability.get("verdict"),
                         "utc": ts(d["timestamp_utc"]) if d.get("timestamp_utc") else None})
    return rows


def events(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("state/projects/*/runs/*/logs/core-events.jsonl")):
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    rows.sort(key=lambda d: d["timestamp"])
    return rows


def usage(run: Path) -> dict:
    cost, out_tokens = 0.0, 0
    for path in run.glob("state/projects/*/runs/*/logs/usage.jsonl"):
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cost += d.get("total_cost_usd") or 0.0
            out_tokens += d.get("output_tokens") or 0
    return {"cost_usd": round(cost, 2), "output_tokens": out_tokens}


def gpu(run: Path) -> dict:
    log = run / "gpu-broker.log"
    if not log.exists():
        return {}
    starts, jobs, secs, failed, reruns, cancelled = {}, 0, 0.0, 0.0, 0, 0
    for line in log.read_text(errors="replace").splitlines():
        m = BROKER.match(line)
        if not m:
            if "client gone; cancel" in line:
                cancelled += 1
            continue
        _, job, kind, rest = m.groups()
        if kind == "start":
            starts[job] = rest
            continue
        e = EXIT.search(rest)
        if not e:
            continue
        jobs += 1
        s = float(e.group(2))
        secs += s
        if int(e.group(1)) != 0:
            failed += s
        if "--rerun-of" in starts.get(job, ""):
            reruns += 1
    return {"gpu_jobs": jobs, "gpu_min": round(secs / 60, 1), "gpu_failed_min": round(failed / 60, 1),
            "gpu_rerun_jobs": reruns, "gpu_cancelled": cancelled}


def phases(evts: list[dict]) -> list[dict]:
    """Phase intervals in order: kind, start, end, round index (1-based, by round_finished)."""
    open_, out, rnd = {}, [], 1
    for d in evts:
        t = d["type"]
        if t == "phase_started":
            open_[d.get("execution_id")] = (d["data"].get("phase") or d.get("agent_kind"), ts(d["timestamp"]))
        elif t == "phase_finished" and d.get("execution_id") in open_:
            kind, start = open_.pop(d["execution_id"])
            out.append({"kind": kind, "start": start, "end": ts(d["timestamp"]), "round": rnd})
        elif t == "round_finished":
            rnd += 1
    for i, p in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        if p["kind"] == "implementer" and nxt and nxt["round"] == p["round"] and nxt["kind"] == "orchestrator":
            p["kind"] = "microbench"
    return out


def score_run(run: Path, targets: list[float], tol: float) -> dict:
    run = run.resolve()
    row: dict = {"run": run.name}
    ws = workspace_of(run)
    row["case"] = ws.name if ws else None
    evts = events(run)
    rounds_done = [d for d in evts if d["type"] == "round_finished"]
    started = next((ts(d["timestamp"]) for d in evts if d["type"] == "run_started"), None)
    ended = ts(evts[-1]["timestamp"]) if evts else None
    row["rounds_closed"] = len(rounds_done)
    row["wall_min"] = round((ended - started).total_seconds() / 60, 1) if started and ended else None
    row["retries"] = sum(max(0, (d["data"].get("attempts") or 1) - 1) for d in rounds_done)
    row["verdicts"] = "".join({"pass": "P", "fail": "F", "skipped": "S"}.get(d["data"].get("judge_verdict"), "?")
                              for d in rounds_done)

    mins: Counter = Counter()
    for p in phases(evts):
        mins[p["kind"]] += (p["end"] - p["start"]).total_seconds() / 60
    for kind in ("orchestrator", "profiler", "microbench", "implementer", "judge"):
        row[f"min_{kind}"] = round(mins.get(kind, 0.0), 1)
    if not mins.get("microbench") and ws and (ws / "progress/microbench/answers").is_dir():
        row["min_microbench"] = None  # the microbench work ran inside the implementer turn

    row.update(usage(run))
    row.update(gpu(run))

    cks = checks(ws) if ws else []
    good = [c for c in cks if c["ok"] and c["utc"]]
    root = [c for c in good if c["root"]]
    row["checks"] = len(cks)
    row["checks_variant"] = sum(not c["root"] for c in cks)
    row["checks_failed"] = sum(not c["ok"] for c in cks)
    row["checks_unstable"] = sum(c["verdict"] == "unstable" for c in cks)
    row["best_check_ms"] = round(min(c["ms"] for c in root), 3) if root else None
    variants = [c for c in good if not c["root"]]
    row["best_variant_ms"] = round(min(c["ms"] for c in variants), 3) if variants else None

    official = run / "state" / "official-timing.jsonl"
    vals = []
    if official.exists():
        for line in official.read_text().splitlines():
            try:
                vals.append(json.loads(line)["fast_p50_ms"])
            except (json.JSONDecodeError, KeyError):
                pass
    row["best_official_ms"] = round(min(vals), 3) if vals else None

    kept: list[dict] = []
    seed_tree = None
    row["seeded"] = False
    history: list[tuple[datetime, str]] = []  # (commit time, candidate tree) of every commit
    if ws:
        for sha, when, subject in commits(ws):
            tree = tree_at(ws, sha)
            history.append((when, tree))
            if subject.startswith("Seed from "):
                row["seeded"], seed_tree = True, tree
            m = CLOSE.match(subject)
            if m:
                kept.append({"round": int(m.group(1)), "utc": when, "tree": tree,
                             "sub_sha": sub_sha_at(ws, sha)})
    trail, prev_tree, prev_ms = [], seed_tree, None
    kept_slower, rejected_faster, unmeasured = [], [], 0
    window_start = started
    for k in kept:
        inside = [c for c in good if window_start and window_start <= c["utc"] <= k["utc"]]
        if k["tree"] == prev_tree:
            ms = prev_ms
        else:
            # Checks of the kept tree: same submission.py, timed after the last commit of the
            # round whose candidate differs from the kept one (other attempts end there).
            same = [c for c in inside if c["root"] and c["sub_sha"] == k["sub_sha"]]
            other = [t for t, tree in history if window_start <= t < k["utc"] and tree != k["tree"]]
            late = [c for c in same if not other or c["utc"] >= max(other)]
            mine = [c["ms"] for c in (late or same)]
            ms = min(mine) if mine else None
            unmeasured += ms is None
            if ms and prev_ms and ms > prev_ms * (1 + tol):
                kept_slower.append(f"r{k['round']} {prev_ms:.3f}->{ms:.3f}")
        trail.append(f"{ms:.3f}" if ms else "-")
        if ms:
            for c in inside:
                if c["ms"] < ms * (1 - tol):
                    rejected_faster.append(f"r{k['round']} {c['dir']} {c['ms']:.3f}<{ms:.3f}")
        window_start, prev_tree = k["utc"], k["tree"]
        prev_ms = ms if ms else prev_ms
    row["kept_trail_ms"] = " ".join(trail)
    vals = [float(t) for t in trail if t != "-"]
    row["best_kept_ms"] = min(vals) if vals else None
    row["kept_unmeasured"] = unmeasured
    row["kept_slower"] = kept_slower
    row["rejected_faster"] = rejected_faster

    # Minutes from the run start to the first passing check at or under each target.
    def minutes_to(limit: float) -> float | None:
        hit = [c["utc"] for c in root if c["ms"] <= limit]
        return round((min(hit) - started).total_seconds() / 60, 1) if hit and started else None

    for target in targets:
        row[f"min_to_{target:g}"] = minutes_to(target)
    row["min_to_best"] = minutes_to(min(c["ms"] for c in root)) if root else None
    return row


COLUMNS = ["run", "case", "seeded", "rounds_closed", "verdicts", "wall_min", "best_kept_ms",
           "best_check_ms", "best_variant_ms", "best_official_ms", "kept_trail_ms", "min_to_best", "min_orchestrator",
           "min_profiler", "min_microbench", "min_implementer", "min_judge", "retries", "gpu_jobs",
           "gpu_min", "gpu_failed_min", "gpu_rerun_jobs", "checks", "checks_variant", "checks_failed",
           "checks_unstable", "cost_usd", "kept_unmeasured", "kept_slower", "rejected_faster"]


def markdown(rows: list[dict], targets: list[float]) -> str:
    cols = COLUMNS[:12] + [f"min_to_{t:g}" for t in targets] + COLUMNS[12:]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, list):
                v = "; ".join(v) if v else ""
            cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--targets", default="", help="comma-separated ms targets for minutes-to-target")
    ap.add_argument("--tolerance-pct", type=float, default=1.0)
    ap.add_argument("--json", type=Path, help="write one JSON row per run here")
    ap.add_argument("--markdown", type=Path, help="write a markdown table here")
    args = ap.parse_args(argv)
    targets = [float(t) for t in args.targets.split(",") if t.strip()]
    rows = []
    for run in args.runs:
        if not run.is_dir():
            continue
        try:
            rows.append(score_run(run, targets, args.tolerance_pct / 100))
        except Exception as exc:  # one broken run must not stop the table
            rows.append({"run": run.name, "error": f"{type(exc).__name__}: {exc}"})
    if args.json:
        args.json.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    table = markdown(rows, targets)
    if args.markdown:
        args.markdown.write_text(table)
    print(table, end="")
    for r in rows:
        if "error" in r:
            print(f"{r['run']}: {r['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
