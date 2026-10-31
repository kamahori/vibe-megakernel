"""Engine map: for each matrix-multiply phase, the legal instruction mappings and the measured rows.

For one phase (M token rows, N output columns, K reduction, batch, data type) it lists every
engine and instruction shape that could compute it: tcgen05 (cta_group 1 and 2, tokens as the
MMA M or, swap-AB, as the MMA N), warp-level mma.sync, and SIMT FMA. For each mapping:

  useful   fraction of the issued FMAs that are real work after padding the token rows
  rate     the best measured FMA per clock per SM for that instruction shape (row id)
  est_us   padded FMAs / (rate x 148 SMs x SM clock): a compute-only lower bound, no data movement

Then the bytes the weight (B operand) moves per call and the HBM time at 8 TB/s, the dequant
time when the weight comes from an integer tensor, and the measured GEMM or attention rows
nearest to the shape. It ranks nothing and recommends nothing: the rows are the evidence.

  python -m megabench.primdb.engine_map M=50 N=4096 K=1024 [batch=1] [dtype=bfloat16] [weight=int8]
  python -m megabench.primdb.engine_map --case ID     every traced phase of a case (case_phases.json)
"""

from __future__ import annotations

import argparse
import statistics
import sys
from math import ceil

from . import compare_gemm, load, nearest, select
from .phases import load as load_phases

SMS = 148
HBM_BYTES_PER_US = 8e6
BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1, "uint8": 1, "float8_e4m3fn": 1}
KIND = {"bfloat16": "f16", "float16": "f16", "float32": "tf32"}
MMA_SYNC = {"bfloat16": "m16n8k16", "float16": "m16n8k16", "float32": "m16n8k8"}
SRC = {"int8": "s8"}


def _pad(n: int, step: int) -> int:
    return ceil(n / step) * step


def _clock(rows: list[dict]) -> float:
    clocks = [r["cond"]["sm_clock_mhz"] for r in rows
              if isinstance(r.get("cond", {}).get("sm_clock_mhz"), (int, float))]
    return statistics.median(clocks) if clocks else 1900.0


def _best(rows: list[dict], metric: str) -> dict | None:
    found = [r for r in rows if isinstance(r.get("metrics", {}).get(metric), (int, float))]
    return max(found, key=lambda r: r["metrics"][metric]) if found else None


def mappings(rows: list[dict], m: int, n: int, dtype: str) -> list[dict]:
    """Every legal instruction mapping for M token rows against N output columns."""
    out = []
    kind = KIND.get(dtype)
    tc = select(rows, "tcgen05.mma", {"kind": ("set", {kind}), "a_src": ("set", {"smem"})}) if kind else []
    for cta_group, sizes in ((1, (64, 128)), (2, (128, 256))):
        for mi in sizes:
            same = [r for r in tc if r["params"].get("cta_group") == cta_group and r["params"].get("M") == mi]
            # Tokens as the MMA M; the output columns fill the MMA N.
            ni = min(256, _pad(n, 16))
            out.append(_mapping("tcgen05", f"cta_group {cta_group}, tokens as MMA M", f"M{mi} N{ni}",
                                _pad(m, mi), m, _near_n(same, ni), "fma_per_clk_sm"))
            # Swap-AB: weight rows as the MMA M, tokens as the MMA N.
            step = 8 if mi == 64 else 16
            ni = min(256, _pad(m, step))
            out.append(_mapping("tcgen05", f"cta_group {cta_group}, swap-AB (tokens as MMA N)",
                                f"M{mi} N{ni}", _pad(m, ni), m, _near_n(same, ni), "fma_per_clk_sm"))
    shape = MMA_SYNC.get(dtype)
    if shape:
        ms = select(rows, "mma.sync", {"shape": ("set", {shape})})
        ms = [r for r in ms if r.get("kind", "measure") == "measure"]
        best = _best(ms, "fma_per_clk_sm")
        out.append(_mapping("mma.sync", "tokens as MMA M", shape, _pad(m, 16), m, best, "fma_per_clk_sm"))
        out.append(_mapping("mma.sync", "swap-AB (tokens as MMA N)", shape, _pad(m, 8), m, best,
                            "fma_per_clk_sm"))
    ffma = _best(select(rows, "simt.ffma"), "flop_per_clk_sm")
    out.append(_mapping("SIMT", "FFMA, fp32 accumulate", "-", m, m, ffma, "flop_per_clk_sm", 0.5))
    return out


def _near_n(rows: list[dict], n: int) -> dict | None:
    if not rows:
        return None
    closest = nearest(rows, {"N": n}, 1)
    if not closest:
        return None
    return _best([r for r in rows if r["params"].get("N") == closest[0]["params"]["N"]], "fma_per_clk_sm")


def _mapping(engine, orientation, instr, padded, m, row, metric, scale=1.0) -> dict:
    rate = row["metrics"][metric] * scale if row else None
    return {"engine": engine, "orientation": orientation, "instr": instr, "padded_m": padded,
            "useful": m / padded, "rate": rate, "row": row["id"] if row else None,
            "row_n": row["params"].get("N") if row else None}


def section(rows: list[dict], phase: dict, label: str = "") -> str:
    m, n, k = phase["M"], phase["N"], phase["K"]
    batch, dtype = phase.get("batch", 1), phase.get("dtype", "bfloat16")
    source = phase.get("weight_source")
    clock = _clock(rows)
    lines = []
    count = f", {phase['count']} per step" if phase.get("count") else ""
    kind = "batched (attention-like)" if batch > 1 else "weight GEMM"
    lines.append(f"### {label}{phase.get('op', 'mm')} M={m} N={n} K={k} batch={batch} {dtype}"
                 f"{', weight from ' + source if source else ''}{count}")
    lines.append("")
    wbytes = n * k * batch * BYTES.get(source or dtype, 2)
    lines.append(f"- {kind}: {m * n * k * batch / 1e6:.3g} M FMAs; B operand {wbytes / 1e6:.3g} MB "
                 f"per call, {wbytes / HBM_BYTES_PER_US:.3g} us at 8 TB/s")
    if source:
        src = SRC.get(source)
        deq = [r for r in select(rows, "dequant") if not src or r["params"].get("src") == src]
        best = _best(deq, "elems_per_clk_sm")
        if best:
            us = n * k * batch / (best["metrics"]["elems_per_clk_sm"] * SMS * clock)
            lines.append(f"- dequant of the weight: {us:.3g} us at the best measured rate "
                         f"({best['primitive']}, row {best['id']})")
        else:
            lines.append(f"- dequant of the weight: no row for source {source}; measure it")
    lines.append("")
    lines.append("| engine | mapping | instr | M padded | useful | FMA/clk/SM | row | est_us (compute only) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for mp in mappings(rows, m, n, dtype):
        if mp["rate"]:
            est = mp["padded_m"] * n * k * batch / (mp["rate"] * SMS * clock)
            rate, est_s = f"{mp['rate']:.4g}", f"{est:.3g}"
            row = mp["row"] + (f" (N{mp['row_n']})" if mp["engine"] == "tcgen05" else "")
        else:
            rate, est_s, row = "no row", "-", "measure"
        lines.append(f"| {mp['engine']} | {mp['orientation']} | {mp['instr']} | {mp['padded_m']} | "
                     f"{mp['useful']:.2f} | {rate} | {row} | {est_s} |")
    lines.append("")
    if batch > 1:
        attn = [r for r in select(rows, "attn_phase") if r.get("kind", "measure") == "measure"]
        near = nearest(attn, {"kv": max(n, k)}, 3)
        qlens = sorted({r["params"].get("q_len") for r in attn if r["params"].get("q_len") is not None})
        lines.append(f"Measured attention phases nearest kv={max(n, k)} (q_len measured: {qlens}; "
                     f"this phase has {m} query rows per batch entry). If several batch entries "
                     "share one K/V operand, their query rows can stack into one MMA.")
        for r in near:
            lines.append(f"- {r['id']} {r['primitive']} engine={r['params'].get('engine')} "
                         f"kv={r['params'].get('kv')} q_len={r['params'].get('q_len')} "
                         f"batch={r['params'].get('batch')} us_attend={r['metrics'].get('us_attend')}")
    else:
        w = "s8" if source == "int8" else ("bf16" if dtype == "bfloat16" else None)
        filters = {"w_dtype": ("set", {w})} if w else None
        notes, ranked = compare_gemm(rows, {"M": float(m), "N": float(n), "K": float(k)}, filters)
        lines.append("Measured GEMMs nearest this shape (gemm: standalone launch; gemm_phase: one "
                     "phase in a persistent 148-CTA kernel, with its grid barrier):")
        lines += [f"- {note}" for note in notes]
        if not ranked:
            lines.append("- no measured GEMM rows for this data type; measure the phase")
        for r in ranked:
            p = r["params"]
            lines.append(f"- {r['id']} {r['family']} engine={p.get('engine')} M={p.get('M')} "
                         f"N={p.get('N')} K={p.get('K')} l2={r['cond'].get('l2')} "
                         f"us={r['metrics']['us']:.4g}")
    lines.append("")
    return "\n".join(lines)


def case_map(rows: list[dict], case_id: str, phases: list[dict]) -> str:
    def traffic(p: dict) -> float:
        return p["N"] * p["K"] * p.get("batch", 1) * BYTES.get(p.get("weight_source") or p["dtype"], 2) \
            * p.get("count", 1)
    ordered = sorted(phases, key=traffic, reverse=True)
    head = [f"# Engine map: {case_id}", "",
            "Generated from `./primdb` rows and the matrix multiplies of the PyTorch reference "
            "(traced once, one step). The phases are in order of B-operand bytes per step. "
            "est_us is compute only, at 148 SMs and the median measured SM clock "
            f"({_clock(rows):.0f} MHz); it ignores data movement, so compare it with the bytes "
            "line. A tcgen05 rate comes from the measured N nearest the mapping's N.", "",
            "`./engine_map M=.. N=.. K=..` prints one phase; `./primdb explain <row>` shows how a "
            "row was measured.", ""]
    return "\n".join(head) + "\n" + "\n".join(
        section(rows, p, f"E{i}: ") for i, p in enumerate(ordered, 1))


def main(argv: list[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    p = argparse.ArgumentParser(prog="engine_map", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shape", nargs="*", help="M=.. N=.. K=.. [batch=..] [dtype=..] [weight=..]")
    p.add_argument("--case", help="map every traced phase of this case")
    args = p.parse_args(argv)
    rows = load()
    if args.case:
        phases = load_phases().get(args.case)
        if not phases:
            print(f"no traced phases for {args.case}; give one phase as M=.. N=.. K=..", file=out)
            return 1
        print(case_map(rows, args.case, phases), file=out)
        return 0
    kv = dict(pair.split("=", 1) for pair in args.shape)
    try:
        phase = {"M": int(kv["M"]), "N": int(kv["N"]), "K": int(kv["K"]),
                 "batch": int(kv.get("batch", 1)), "dtype": kv.get("dtype", "bfloat16"),
                 "weight_source": kv.get("weight")}
    except KeyError as e:
        raise SystemExit(f"missing {e}; give M=.. N=.. K=..") from None
    print(section(rows, phase), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
