"""Build the AMD primdb seed from megabench/primbench/amd/run_amd.sh output directories.

  python -m megabench.primdb.amd_seed OUT_DIR [OUT_DIR ...] -o megabench/primdb/seed_amd.jsonl
  python -m megabench.primdb.amd_seed --skill-tables     # rewrite the CDNA skills' measured tables

Each OUT_DIR holds one GPU's `*.jsonl` probe lines (a `gemv_amd --msweep` directory holds
`gemv_msweep.jsonl`; its rows that repeat a full-probe measurement on the same arch are dropped); every row keeps `cond.arch` and `cond.cus`,
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

PROBES = ("sync_amd", "mem_amd", "mfma_amd", "gemv_amd", "gemv_msweep")
GEMV_ENGINES = ("valu_r1", "valu_r4", "mfma16", "mfma_lds_sk", "mfma_lds_db")


def _keep(name: str, flat: dict) -> bool:
    return name != "gemv_msweep" or flat.get("primitive", "").split(".")[-1] in GEMV_ENGINES


def _measured(r: dict) -> tuple:
    p = dict(r["params"])
    p.setdefault("launch_bounds", 1024)
    for k in ("split_k", "k_chunk"):
        p.pop(k, None)
    return r["primitive"], r["cond"].get("arch"), json.dumps(p, sort_keys=True)


def drop_repeats(rows: list[dict]) -> list[dict]:
    """Drop gemv_msweep rows that repeat a measurement a full probe already made on that arch."""
    probe = {_measured(r) for r in rows if not r["source"]["file"].endswith("gemv_msweep.jsonl")}
    return [r for r in rows if not r["source"]["file"].endswith("gemv_msweep.jsonl") or _measured(r) not in probe]


def rows_from(out: Path) -> list[dict]:
    rows = []
    for name in PROBES:
        path = out / f"{name}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            flat = json.loads(line)
            if not _keep(name, flat):
                continue
            bench = "gemv_amd" if name == "gemv_msweep" else name
            row = normalize(flat, **{"s.bench": f"megabench/primbench/amd/{bench}.hip",
                                                 "s.file": f"{out.name}/{path.name}"})
            row["source"]["bench"] = f"megabench/primbench/amd/{bench}.hip"
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
    lines = ["| N x K | M | " + " | ".join(f"{e} us" for e in GEMV_ENGINES) + " |", "|---" * (2 + len(GEMV_ENGINES)) + "|"]
    for n, k in shapes:
        for m in ms:
            b = _best(rows, arch, n, k, m)
            if not b:
                continue
            lo = min(v[0] for v in b.values())
            cells = [("**" if b[e][0] == lo else "") + _fmt(b[e][0]) + ("**" if b[e][0] == lo else "") if e in b else "-"
                     for e in GEMV_ENGINES]
            lines.append(f"| {n} x {k} | {m} | " + " | ".join(cells) + " |")
    lines.append("\nus = one cold GEMV, best thread count; bold = fastest. `mfma_lds_*`: x staged in LDS per k-chunk, "
                 "16x16x16 MFMA, split-K with fp32 atomics; `_db` double-buffers the chunk (64-k chunks at M = 128).")
    return "\n".join(lines)


def task_facts(rows, arch: str) -> str:
    """A short TASK.md section of the decisive measured costs for one AMD arch ('' if no rows)."""
    rows = [r for r in rows if r["cond"].get("arch") == arch]
    if not rows:
        return ""
    cus = rows[0]["cond"]["cus"]
    name = dict(ARCHES).get(arch, arch)

    def us(proto, dirty=0):
        for r in rows:
            p = r["params"]
            if (r["primitive"] == "grid_barrier." + proto and p["blocks"] == cus and p["threads"] == 256
                    and p["dirty_bytes_per_block"] == dirty):
                return r["metrics"]["us"]
        return None

    def f(v, nd=1):
        return "?" if v is None else f"{v:.{nd}f}"

    lines = [f"\n## Measured costs on this GPU ({name}, {arch}; from `./primdb`, read before you design)\n"]
    lines.append(
        f"- Grid barrier, {cus} blocks x 256 threads, per barrier: release add + relaxed poll + one acquire "
        f"fence {f(us('release_add_relaxed_poll_acquire_fence'))} us ({f(us('release_add_relaxed_poll_acquire_fence', 16384))} us "
        f"with 16 KiB written per block); acquire load in the poll {f(us('release_add_acquire_poll'))} us; "
        f"`__threadfence()` on every thread {f(us('every_thread_threadfence'))} us; "
        f"`cooperative_groups` grid sync {f(us('cooperative_groups_grid_sync'))} us; "
        f"cross-block data through relaxed agent-scope (sc1) loads/stores with a fence-free barrier "
        f"{f(us('coherent_data_no_fence'))} us; per-XCD two-level counters {f(us('per_xcd_two_level_release_acquire'))} us. "
        f"Relaxed atomics alone or workgroup-scope fences read stale data across XCDs.")
    fences = {r["primitive"]: r["metrics"]["ns"] for r in rows
              if r["primitive"].startswith("fence.") and r["params"]["dirty_bytes_per_block"] == 0}
    if fences:
        lines.append(f"- One agent-scope fence by every block at once: release {f(fences.get('fence.release.agent'), 0)} ns, "
                     f"acquire {f(fences.get('fence.acquire.agent'), 0)} ns (workgroup scope "
                     f"{f(fences.get('fence.release.workgroup'), 0)} ns).")
    flags = [r for r in rows if r["primitive"] == "flag_handoff.relaxed"]
    if flags:
        same = min((r["metrics"]["ns"] for r in flags if r["params"]["same_xcd"]), default=None)
        cross = min((r["metrics"]["ns"] for r in flags if not r["params"]["same_xcd"]), default=None)
        lines.append(f"- Flag hand-off between two blocks, one way: {f(same, 0)} ns same XCD, {f(cross, 0)} ns across XCDs.")
    mem = {(r["primitive"], r["params"]["blocks"], r["params"]["threads"], r["params"]["loads_in_flight"],
            r["params"]["width_bytes"], r["cond"]["level"]): r["metrics"]["gbps"] for r in rows if r["family"] == "mem"}
    full = mem.get(("mem.stream_read", cus, 256, 4, 16, "hbm"))
    nt = max((v for k, v in mem.items() if k[0].endswith("nontemporal") and k[5] == "hbm"), default=None)
    eighth = mem.get(("mem.stream_read", cus // 8, 1024, 4, 16, "hbm"))
    lines.append(f"- HBM streaming: {f(full and full / 1000, 2)} TB/s (16 B loads, 1 block/CU x 256 threads), "
                 f"{f(nt and nt / 1000, 2)} TB/s nontemporal; with 1/8 of the CUs {f(eighth and eighth / 1000, 2)} TB/s.")
    g = []
    for n, k in ((4096, 1024), (1024, 3072)):
        b = _best(rows, arch, n, k, 1)
        if b:
            g.append(f"{n}x{k}: " + ", ".join(f"{e} {b[e][0]:.1f}" for e in ("valu_r1", "valu_r4", "mfma4", "mfma16") if e in b) + " us")
    if g:
        lines.append("- Batch-1 bf16 GEMV, cold, best thread count: " + "; ".join(g) + ".")
    for n, k in ((6144, 4096), (1536, 2048)):
        sm = [(m, _best(rows, arch, n, k, m)) for m in (1, 4, 8, 16, 32, 128)]
        sm = [(m, b) for m, b in sm if any(e.startswith("mfma_lds") for e in b) and ("valu_r1" in b or "valu_r4" in b)]
        if sm:
            lines.append(f"- M tokens, {n}x{k} (VALU vs MFMA with x in LDS + split-K): " + "; ".join(
                f"M={m} {min(b[e][0] for e in ('valu_r1', 'valu_r4') if e in b):.1f} vs "
                f"{min(b[e][0] for e in b if e.startswith('mfma_lds')):.1f} us" for m, b in sm) + ".")
    ws = {r["primitive"]: r["metrics"]["cyc"] for r in rows if r["primitive"].startswith("wave_sum.")}
    if ws:
        lines.append(f"- 64-lane float sum: DPP + readlane {f(ws.get('wave_sum.dpp_readlane'), 0)} cycles, "
                     f"`__shfl_xor` {f(ws.get('wave_sum.shfl_xor'), 0)} cycles.")
    lines.append("- Each line is an isolated measurement; confirm the choice in your full kernel. Options, rules and "
                 "the reasons behind them: `megakernel_skills/cdna-sync-and-memory` and "
                 "`megakernel_skills/cdna-matrix-core-design`.\n")
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
    rows = drop_repeats([r for d in a.dirs for r in rows_from(d)])
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate row ids: run each GPU's probes into its own directory")
    a.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
