"""Measured B200 primitive costs, for sizing kernel work to hardware shapes.

Rows live in JSONL files: the trusted seed next to this module and an
optional local file (primdb_local.jsonl or $PRIMDB_LOCAL) that
workspace benchmarks append to. See schema.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Iterator

SEED = Path(__file__).with_name("seed.jsonl")
LOCAL_ENV = "PRIMDB_LOCAL"
KINDS = ("measure", "model", "fact")
GEMM_FAMILIES = ("gemm", "gemm_phase")


def row_id(row: dict) -> str:
    """Stable id from what was measured and where, not from the values."""
    key = json.dumps([row.get("primitive"), row.get("params", {}), row.get("cond", {}),
                      row.get("source", {}).get("file"), row.get("source", {}).get("job")],
                     sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def normalize(flat: dict, **defaults) -> dict:
    """Build a schema row from a flat bench line.

    Benches print `p.<name>` for parameters, `m.<name>` for metrics and
    `c.<name>` for conditions; other keys stay top level.
    """
    row: dict = {"kind": "measure", "params": {}, "metrics": {}, "cond": {}, "source": {}}
    for key, value in (defaults | flat).items():
        prefix, _, name = key.partition(".")
        target = {"p": "params", "m": "metrics", "c": "cond", "s": "source"}.get(prefix)
        if target and name:
            row[target][name] = value
        elif isinstance(value, dict) and key in row:
            row[key] |= value
        else:
            row[key] = value
    validate(row)
    row.setdefault("id", row_id(row))
    return row


def validate(row: dict) -> None:
    for key in ("family", "primitive"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"row needs a non-empty {key!r}: {row}")
    if row.get("kind", "measure") not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}: {row}")
    if row.get("kind", "measure") == "measure" and not row.get("metrics"):
        raise ValueError(f"a measure row needs metrics: {row}")
    for name, value in row.get("metrics", {}).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric {name} is not finite: {row}")


def read(path: Path, origin: str) -> Iterator[dict]:
    if not path.is_file():
        return
    with path.open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
                validate(row)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"{path}:{n}: {e}") from None
            row.setdefault("id", row_id(row))
            row["origin"] = origin
            yield row


def local_path() -> Path:
    return Path(os.environ.get(LOCAL_ENV, "primdb_local.jsonl"))


def load(seed: Path = SEED, local: Path | None = None) -> list[dict]:
    if not seed.is_file():
        raise FileNotFoundError(f"primdb seed {seed} is missing; a snapshot of the repository "
                                "must include it (megabench/.gitignore ignores *.jsonl)")
    rows = list(read(seed, "seed"))
    rows += read(local if local is not None else local_path(), "local")
    return rows


def append(rows: Iterable[dict], path: Path | None = None) -> int:
    path = path or local_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a") as f:
        for row in rows:
            validate(row)
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


def _num(value):
    if isinstance(value, bool):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def parse_filter(text: str):
    """`a..b` is an inclusive numeric range, `a,b,c` a set, anything else one value."""
    if ".." in text:
        lo, hi = text.split("..", 1)
        return ("range", _num(lo) if lo else -math.inf, _num(hi) if hi else math.inf)
    return ("set", {_num(v) for v in text.split(",")})


def field(row: dict, name: str):
    for part in ("params", "cond", "metrics"):
        if name in row.get(part, {}):
            return row[part][name]
    return row.get(name)


def matches(row: dict, filters: dict) -> bool:
    for name, (op, *arg) in filters.items():
        value = field(row, name)
        if value is None:
            return False
        value = _num(value)
        if op == "range":
            if not isinstance(value, float) or not arg[0] <= value <= arg[1]:
                return False
        elif value not in arg[0] and str(value) not in {str(a) for a in arg[0]}:
            return False
    return True


def select(rows: Iterable[dict], what: str | None = None, filters: dict | None = None) -> list[dict]:
    """`what` matches a family, a primitive, or a primitive prefix ending in '.'."""
    out = []
    for row in rows:
        if what and what not in (row["family"], row["primitive"]) and not (
                what.endswith(".") and row["primitive"].startswith(what)):
            continue
        if filters and not matches(row, filters):
            continue
        out.append(row)
    return out


def nearest(rows: list[dict], target: dict[str, float], k: int = 5) -> list[dict]:
    """Rows closest to `target` in log space over its numeric parameters."""
    def distance(row: dict) -> float:
        d = 0.0
        for name, want in target.items():
            have = _num(field(row, name))
            if not isinstance(have, float) or have <= 0 or want <= 0:
                return math.inf
            d += math.log2(have / want) ** 2
        return d
    scored = sorted(((distance(r), i) for i, r in enumerate(rows)))
    return [rows[i] for d, i in scored[:k] if math.isfinite(d)]


def compare_gemm(rows: list[dict], want: dict[str, float], filters: dict | None = None
                 ) -> tuple[list[str], list[dict]]:
    """Fastest measured row per (family, engine, L2 state) at the nearest measured GEMM shape.

    Returns notes on shapes that differ from `want` and the rows, ranked within each family.
    """
    measured = [r for r in rows if r["family"] in GEMM_FAMILIES and r.get("kind", "measure") == "measure"]
    notes, pool = [], []
    for family in GEMM_FAMILIES:
        candidates = select(measured, family, filters)
        best = nearest(candidates, want, 1) if want else candidates[:1]
        if not best:
            continue
        got = {k: best[0]["params"][k] for k in want}
        pool += [r for r in candidates if all(r["params"].get(k) == v for k, v in got.items())]
        if any(float(got[k]) != want[k] for k in want):
            notes.append(f"{family}: nearest measured shape {got} (asked {want})")
    by_engine: dict = {}
    for r in pool:
        key = (r["family"], r["params"].get("engine"), r["params"].get("M"),
               r["params"].get("N"), r["params"].get("K"), r["cond"].get("l2"))
        if key not in by_engine or r["metrics"]["us"] < by_engine[key]["metrics"]["us"]:
            by_engine[key] = r
    ranked = sorted(by_engine.values(),
                    key=lambda r: (r["family"], r["cond"].get("l2", ""), r["metrics"]["us"]))
    return notes, ranked
