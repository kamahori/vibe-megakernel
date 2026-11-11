"""Measured rows of earlier runs of one case, for a new run to look up.

  python -m megabench.primdb.cache --runs-root DIR --case CASE -o FILE
         [--exclude RUN ...] [--store FILE]

Every run of a case measures its own primitives into
``<run>/<case>/progress/microbench/primdb_local.jsonl``. This module collects the
``measure`` rows of all those files (and of an optional store file that keeps
them after run directories are deleted), keeps one row per measured
configuration, and writes them to FILE. A new run's ``./primdb`` reads FILE as
origin ``cache``: its agents answer a question from a matched cached row and
measure only what no row covers.

Only measured data is kept. Plans, notes (also a row's own ``note`` and other free-text
fields), answers, code and paths into other
run directories are not: a source path keeps its file name and the row says
which run and round measured it (``source.cache_from``, ``source.cache_round``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import validate

LOCAL = Path("progress/microbench/primdb_local.jsonl")
KEEP = ("id", "kind", "family", "primitive", "params", "cond", "metrics", "n_reps", "cv",
        "verified", "tuned")
# Free text that carries an earlier run's conclusions, not measured data. Dropped at
# every level of a row, so a cached row gives a new run numbers and conditions only.
TEXT_KEYS = ("note", "notes", "desc", "description", "comment", "conclusion", "why")
# Conditions that vary between identical measurements, not between configurations.
VOLATILE_COND = ("sm_clock_mhz", "clock_mhz", "job", "host", "date")


def config_key(row: dict) -> str:
    cond = {k: v for k, v in (row.get("cond") or {}).items() if k not in VOLATILE_COND}
    return json.dumps([row.get("family"), row.get("primitive"), row.get("params") or {}, cond],
                      sort_keys=True, default=str)


def _quality(row: dict, order: int) -> tuple:
    """Higher is better: verified, lower cv, more repetitions, newer."""
    cv = row.get("cv")
    cv = float(cv) if isinstance(cv, (int, float)) else float("inf")
    reps = row.get("n_reps") if isinstance(row.get("n_reps"), int) else 0
    return (bool(row.get("verified")), -cv, reps, order)


def _no_text(value):
    if isinstance(value, dict):
        return {k: _no_text(v) for k, v in value.items() if k not in TEXT_KEYS}
    return value


def _strip(row: dict, run: str) -> dict:
    out = {k: _no_text(row[k]) for k in KEEP if k in row}
    source = row.get("source") or {}
    kept = {k: Path(str(v)).name for k, v in source.items() if k in ("file", "bench") and v}
    kept |= {k: v for k, v in source.items() if k in ("timer",)}
    kept["cache_from"] = source.get("cache_from", run)
    if source.get("cache_round") or source.get("round"):
        kept["cache_round"] = source.get("cache_round") or source.get("round")
    out["source"] = kept
    return out


def _rows(path: Path, run: str):
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
            if row.get("kind", "measure") != "measure":
                continue
            validate(row)
        except (json.JSONDecodeError, ValueError):
            continue
        if "id" in row:
            yield _strip(row, run)


def harvest(runs_root: Path, case: str, exclude: tuple[str, ...] = (),
            store: Path | None = None) -> list[dict]:
    """One row per measured configuration of *case*, from every run under *runs_root*.

    Runs are read oldest first, so a tie in quality keeps the newer row.
    """
    sources: list[tuple[Path, str]] = []
    if store is not None:
        sources.append((store, "store"))
    runs = sorted((p for p in runs_root.glob(f"*/{case}/{LOCAL}") if p.is_file()),
                  key=lambda p: p.stat().st_mtime)
    for path in runs:
        run = path.parents[len(LOCAL.parts)].name
        if run not in exclude:
            sources.append((path, run))
    best: dict[str, tuple[tuple, dict]] = {}
    order = 0
    for path, run in sources:
        for row in _rows(path, run):
            order += 1
            key = config_key(row)
            score = _quality(row, order)
            if key not in best or score > best[key][0]:
                best[key] = (score, row)
    return [row for _, row in best.values()]


def write(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="primdb.cache", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--exclude", action="append", default=[], help="run name to skip (repeat)")
    p.add_argument("--store", type=Path,
                   help="rows kept across runs: read first, then rewritten with the harvest")
    args = p.parse_args(argv)
    rows = harvest(args.runs_root, args.case, tuple(args.exclude), args.store)
    n = write(rows, args.output)
    if args.store is not None:
        write(rows, args.store)
    print(f"cached {n} measured rows of {args.case} in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
