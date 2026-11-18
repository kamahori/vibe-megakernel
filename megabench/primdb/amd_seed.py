"""Build the AMD primdb seed from megabench/primbench/amd/run_amd.sh output directories.

  python -m megabench.primdb.amd_seed OUT_DIR [OUT_DIR ...] -o megabench/primdb/seed_amd.jsonl
  python -m megabench.primdb.amd_seed --skill-tables     # rewrite the CDNA skills' measured tables

Each OUT_DIR holds one GPU's `*.jsonl` probe lines; every row keeps `cond.arch` and `cond.cus`,
so one seed can hold several AMD GPUs. rocm workspaces query this seed instead of the B200 one.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from . import normalize, read, row_id, validate

SEED_AMD = Path(__file__).with_name("seed_amd.jsonl")
SKILLS = Path(__file__).resolve().parents[1] / "skills"
ARCHES = (("gfx942", "MI300X"), ("gfx90a", "MI210"))
BW_TB_S = {"gfx942": 5.3, "gfx90a": 1.6, "gfx950": 8.0}

PROBES = ("sync_amd", "mem_amd", "mfma_amd", "gemv_amd")


def rows_from(out: Path) -> list[dict]:
    rows = []
    for name in PROBES:
        path = out / f"{name}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = normalize(json.loads(line), **{"s.bench": f"megabench/primbench/amd/{name}.hip",
                                                 "s.file": f"{out.name}/{path.name}"})
            row["source"]["bench"] = f"megabench/primbench/amd/{name}.hip"
            row["id"] = row_id(row)
            validate(row)
            rows.append(row)
    return rows


def _fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def _arches(rows):
    present = {r["cond"].get("arch") for r in rows}
    return [(a, n) for a, n in ARCHES if a in present]


def table_gbar(rows):
    arches = _arches(rows)
    sel = defaultdict(dict)
    for r in rows:
        if not r["primitive"].startswith("grid_barrier.") or r["params"].get("threads") != 256:
            continue
        if r["params"]["blocks"] != r["cond"]["cus"]:
            continue
        sel[r["primitive"][13:]][(r["cond"]["arch"], r["params"]["dirty_bytes_per_block"])] = r
    dirt = (0, 1024, 16384)
    head = "| protocol (1 block/CU, 256 threads) | " + " | ".join(
        f"{n} us, {d // 1024} KiB dirty" for a, n in arches for d in dirt) + " | stale reads |"
    lines = [head, "|---" * (1 + len(arches) * len(dirt) + 1) + "|"]
    for proto, by in sel.items():
        stale = sum(r["metrics"]["stale_reads"] for r in by.values())
        cells = [_fmt(by.get((a, d), {}).get("metrics", {}).get("us"), 2) for a, _ in arches for d in dirt]
        lines.append(f"| `{proto}` | " + " | ".join(cells) + f" | {'**' + str(stale) + '**' if stale else 0} |")
    fences = defaultdict(dict)
    for r in rows:
        if r["primitive"].startswith("fence."):
            fences[r["primitive"][6:]][(r["cond"]["arch"], r["params"]["dirty_bytes_per_block"])] = r["metrics"]["ns"]
    db = sorted({d for f in fences.values() for _, d in f})
    lines += ["", "| one fence, all blocks fencing at once | " + " | ".join(
        f"{n} ns, {d // 1024 if d >= 1024 else d}{' KiB' if d >= 1024 else ' B'} dirty" for a, n in arches for d in db) + " |",
              "|---" * (1 + len(arches) * len(db)) + "|"]
    for f, by in fences.items():
        lines.append(f"| `{f}` | " + " | ".join(_fmt(by.get((a, d)), 0) for a, _ in arches for d in db) + " |")
    return "\n".join(lines)


def table_mem(rows):
    lines = ["| arch | load | width B | loads in flight | blocks | threads | level | GB/s |",
             "|---|---|---|---|---|---|---|---|"]
    for a, n in _arches(rows):
        for r in rows:
            if r["family"] != "mem" or r["cond"]["arch"] != a:
                continue
            p, c = r["params"], r["cond"]
            blocks = p["blocks"] / c["cus"]
            bl = f"{blocks:g} x CUs" if blocks >= 1 else f"1/{round(1 / blocks)} of CUs"
            nt = "nontemporal" if r["primitive"].endswith("nontemporal") else "plain"
            lines.append(f"| {n} | {nt} | {p['width_bytes']} | {p['loads_in_flight']} | {bl} | {p['threads']} | "
                         f"{c['level']} | {r['metrics']['gbps']:.0f} |")
    return "\n".join(lines)


def table_mfma(rows):
    arches = _arches(rows)
    by = defaultdict(dict)
    for r in rows:
        if r["family"] == "mfma":
            by[r["primitive"]][(r["cond"]["arch"], r["params"]["chains"])] = r
    lines = ["| instruction | " + " | ".join(f"{n} latency cyc | {n} issue cyc | {n} TFLOP/s" for a, n in arches) + " |",
             "|---" * (1 + 3 * len(arches)) + "|"]
    for prim, d in by.items():
        cells = []
        for a, _ in arches:
            lat, thr = d.get((a, 1)), d.get((a, 4))
            cells += [_fmt(lat and lat["metrics"].get("latency_cyc"), 0), _fmt(thr and thr["metrics"].get("issue_cyc"), 0),
                      _fmt(thr and thr["metrics"].get("tflops_device"), 0)]
        lines.append(f"| `{prim}` | " + " | ".join(cells) + " |")
    for r in rows:
        if r["primitive"].startswith("wave_sum."):
            lines.append(f"\n`{r['primitive']}` on {dict(ARCHES).get(r['cond']['arch'])}: {r['metrics']['cyc']:.0f} cycles")
    return "\n".join(lines)


def _best(rows, arch, n, k, m):
    best = {}
    for r in rows:
        p = r["params"]
        if r["family"] != "gemv" or r["cond"]["arch"] != arch or (p["N"], p["K"], p.get("M", 1)) != (n, k, m):
            continue
        e = r["primitive"].split(".")[-1]
        if r.get("verified") and (e not in best or r["metrics"]["us_single"] < best[e][0]):
            best[e] = (r["metrics"]["us_single"], p["threads"])
    return best


def table_gemv(rows, arch="gfx942"):
    shapes = sorted({(r["params"]["N"], r["params"]["K"]) for r in rows if r["family"] == "gemv"
                     and r["cond"]["arch"] == arch}, key=lambda s: s[0] * s[1])
    engines = ("valu_r1", "valu_r4", "mfma16", "mfma4")
    lines = ["| N x K (bf16) | HBM floor us | " + " | ".join(engines) + " |", "|---" * (2 + len(engines)) + "|"]
    for n, k in shapes:
        b = _best(rows, arch, n, k, 1)
        if not b:
            continue
        floor = n * k * 2 / (BW_TB_S[arch] * 1e12) * 1e6
        lo = min(v[0] for v in b.values())
        cells = [("**" if b[e][0] == lo else "") + f"{b[e][0]:.1f} ({b[e][1]})" + ("**" if b[e][0] == lo else "")
                 if e in b else "-" for e in engines]
        lines.append(f"| {n} x {k} | {floor:.1f} | " + " | ".join(cells) + " |")
    lines.append("\nus = one cold GEMV, in-kernel span (no launch overhead); (threads per block); bold = fastest.")
    return "\n".join(lines)


def table_smallm(rows, arch="gfx942"):
    ms = sorted({r["params"].get("M", 1) for r in rows if r["family"] == "gemv" and r["cond"]["arch"] == arch})
    shapes = sorted({(r["params"]["N"], r["params"]["K"]) for r in rows if r["family"] == "gemv"
                     and r["cond"]["arch"] == arch and r["params"].get("M", 1) > 1})
    lines = ["| N x K | M | valu_r1 us | valu_r4 us | mfma16 us |", "|---|---|---|---|---|"]
    for n, k in shapes:
        for m in ms:
            b = _best(rows, arch, n, k, m)
            cells = [_fmt(b[e][0]) if e in b else "-" for e in ("valu_r1", "valu_r4", "mfma16")]
            lines.append(f"| {n} x {k} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def skill_tables(seed: Path = SEED_AMD, skills: Path = SKILLS) -> None:
    """Rewrite the measured tables between <!-- NAME --> and <!-- /NAME --> markers."""
    rows = list(read(seed, "seed"))
    tables = {"MI300X_GBAR_TABLE": table_gbar(rows), "MI300X_MEM_TABLE": table_mem(rows),
              "MI300X_MFMA_TABLE": table_mfma(rows), "MI300X_GEMV_TABLE": table_gemv(rows),
              "MI300X_SMALLM_TABLE": table_smallm(rows)}
    for skill in ("cdna-sync-and-memory", "cdna-matrix-core-design"):
        path = skills / skill / "SKILL.md"
        text = path.read_text()
        for name, table in tables.items():
            text = re.sub(rf"<!-- {name} -->(.*?<!-- /{name} -->)?",
                          lambda _m: f"<!-- {name} -->\n{table}\n<!-- /{name} -->", text, flags=re.S)
        path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--skill-tables", action="store_true", help="rewrite the CDNA skills' tables from the seed")
    a = p.parse_args(argv)
    if a.skill_tables:
        skill_tables(a.output or SEED_AMD)
        return 0
    if not a.dirs or not a.output:
        p.error("give OUT_DIR... and -o, or --skill-tables")
    rows = [r for d in a.dirs for r in rows_from(d)]
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate row ids: run each GPU's probes into its own directory")
    a.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
