"""Query measured B200 primitive costs.

  primdb list [family]                       families, primitives, parameter ranges, metrics
  primdb q WHAT [k=v ...] [--near k=v ...]   rows of a family or primitive (prefix ending in '.')
  primdb compare gemm M=.. N=.. K=.. [w_dtype=..] [l2=..]
                                             fastest measured engine per tile family for a GEMM
  primdb explain ID                          one row in full, with how it was measured
  primdb add [FILE]                          normalize bench JSON lines into the local file

Filters: k=v, k=a,b,c (any of), k=lo..hi (inclusive). Rows from the local
file (the workspace's own measurements) show origin=local.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import (SEED, append, compare_gemm, load, local_path, normalize, nearest,
               parse_filter, select)

def _filters(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"filter {pair!r} is not k=v")
        out[name] = parse_filter(value)
    return out


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _table(rows: list[dict], metrics: list[str] | None, out) -> None:
    if not rows:
        print("(no rows)", file=out)
        return
    pkeys = sorted({k for r in rows for k in r.get("params", {})})
    ckeys = sorted({k for r in rows for k in r.get("cond", {})})
    mkeys = metrics or sorted({k for r in rows for k in r.get("metrics", {})})
    head = ["id", "primitive", *pkeys, *ckeys, *mkeys, "verified", "origin"]
    body = [[r["id"], r["primitive"], *(_fmt(r.get("params", {}).get(k, "")) for k in pkeys),
             *(_fmt(r.get("cond", {}).get(k, "")) for k in ckeys),
             *(_fmt(r.get("metrics", {}).get(k, "")) for k in mkeys),
             _fmt(r.get("verified", "")), r.get("origin", "")] for r in rows]
    width = [max(len(str(c)) for c in col) for col in zip(head, *body)]
    for line in [head, *body]:
        print("  ".join(str(c).ljust(w) for c, w in zip(line, width)).rstrip(), file=out)


def cmd_list(args, rows, out) -> int:
    groups: dict = defaultdict(list)
    for r in select(rows, args.family):
        groups[(r["family"], r["primitive"], r.get("kind", "measure"))].append(r)
    for (family, primitive, kind), group in sorted(groups.items()):
        print(f"{family} / {primitive} [{kind}] rows={len(group)}", file=out)
        params: dict = defaultdict(set)
        for r in group:
            for k, v in r.get("params", {}).items():
                params[k].add(v)
        for k, values in sorted(params.items()):
            nums = sorted(v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
            if len(nums) == len(values) and len(values) > 6:
                print(f"    {k}: {_fmt(nums[0])}..{_fmt(nums[-1])} ({len(values)} values)", file=out)
            else:
                print(f"    {k}: {', '.join(sorted(map(str, values)))}", file=out)
        metrics = sorted({k for r in group for k in r.get("metrics", {})})
        if metrics:
            print(f"    metrics: {', '.join(metrics)}", file=out)
    return 0


def cmd_q(args, rows, out) -> int:
    found = select(rows, args.what, _filters(args.filters))
    if args.near:
        target = {}
        for pair in args.near:
            name, _, value = pair.partition("=")
            target[name] = float(value)
        found = nearest(found, target, args.limit)
    found = found[: args.limit]
    if args.json:
        for r in found:
            print(json.dumps(r, sort_keys=True), file=out)
    else:
        _table(found, args.metrics.split(",") if args.metrics else None, out)
    return 0


def cmd_compare(args, rows, out) -> int:
    if args.what != "gemm":
        raise SystemExit("only `compare gemm` exists")
    filters = _filters(args.filters)
    shape = {k: filters.pop(k) for k in ("M", "N", "K") if k in filters}
    want = {k: next(iter(v[1])) for k, v in shape.items() if v[0] == "set" and len(v[1]) == 1}
    if len(want) != len(shape):
        raise SystemExit("compare takes single values for M, N and K")
    notes, ranked = compare_gemm(rows, want, filters)
    for note in notes:
        print(note, file=out)
    if not ranked:
        print("(no GEMM rows match; drop a filter or measure the point with primbench)", file=out)
        return 1
    _table(ranked, ["us", "tflops", "hbm_gbps"], out)
    print("\ngemm: standalone kernel launch; gemm_phase: one phase inside a persistent "
          "148-CTA kernel (includes its grid barrier). Compare within a family.", file=out)
    return 0


def cmd_explain(args, rows, out) -> int:
    hits = [r for r in rows if r["id"] == args.id]
    if not hits:
        print(f"no row {args.id}", file=out)
        return 1
    for r in hits:
        print(json.dumps(r, indent=1, sort_keys=True), file=out)
    return 0


def cmd_add(args, rows, out) -> int:
    src = open(args.file) if args.file else sys.stdin
    new = []
    with src:
        for line in src:
            line = line.strip()
            if line.startswith("{"):
                new.append(normalize(json.loads(line), **{"s.file": args.file or "stdin"}))
    n = append(new, Path(args.local) if args.local else None)
    print(f"appended {n} rows to {args.local or local_path()}", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="primdb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=Path, default=SEED, help=argparse.SUPPRESS)
    p.add_argument("--local", help="local rows file (default $PRIMDB_LOCAL or primdb_local.jsonl)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.add_argument("family", nargs="?")
    for name in ("q", "compare"):
        s = sub.add_parser(name)
        s.add_argument("what")
        s.add_argument("filters", nargs="*")
        s.add_argument("--near", nargs="*", help="rank by log distance to these numeric params")
        s.add_argument("--metrics", help="comma list of metrics to show")
        s.add_argument("--limit", type=int, default=200)
        s.add_argument("--json", action="store_true")
    s = sub.add_parser("explain")
    s.add_argument("id")
    s = sub.add_parser("add")
    s.add_argument("file", nargs="?")
    return p


def main(argv: list[str] | None = None, out=None) -> int:
    args = build_parser().parse_args(argv)
    out = out or sys.stdout
    rows = load(args.seed, local=Path(args.local) if args.local else None)
    return {"list": cmd_list, "q": cmd_q, "compare": cmd_compare,
            "explain": cmd_explain, "add": cmd_add}[args.cmd](args, rows, out)


if __name__ == "__main__":
    raise SystemExit(main())
