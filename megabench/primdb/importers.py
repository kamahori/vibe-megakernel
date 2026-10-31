"""Convert existing B200 microbenchmark results into primdb seed rows.

Usage:
  python -m megabench.primdb.importers --tcgen05 DIR --probes DIR --results DIR -o seed.jsonl

--tcgen05 is the mma_bench / gemm_ab directory, --probes the directory with
xover/ and mk_core/{bar,occfuse}/, --results the dated results directory,
--primbench an output directory of megabench/primbench/run_seed.sh.
A missing source is skipped with a note on stderr.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path

from . import composed, normalize

# xover shapes are Y[T][N] += X[T][K] W[N][K]^T with bf16 W.
_XOVER_SHAPES = {"GU": (6144, 1024), "D": (1024, 3072)}
_XOVER_ENG = {-1: "oracle_load_barrier_only", 0: "ffma", 1: "tcgen05_swapab",
              2: "ffma_both_compiled", 3: "tcgen05_swapab_both_compiled"}
# xover2 arms: KU = weight rows per unit, XTMA = X operand by TMA, RING = weight ring
# slots (0: as many as fit), CG = cta_group, oracle = loads and barrier only.
_XOVER2_SHAPES = {"QKV": (4096, 1024), "O": (1024, 2048), "GU": (6144, 1024),
                  "D": (1024, 3072), "LM": (151936, 1024)}
_XOVER2_ARMS = {"a0": (128, 0, 2, 1, 0), "o0": (128, 0, 2, 1, 1), "a1": (64, 0, 4, 1, 0),
                "a2": (128, 1, 2, 1, 0), "a3": (128, 1, 0, 1, 0), "a4": (64, 1, 0, 1, 0),
                "a5": (128, 1, 0, 2, 0), "o3": (128, 1, 0, 1, 1)}
_BAR_ARMS = {0: "red.release+relaxed_poll+fence.acquire", 1: "red.release+ld.acquire_poll",
             2: "atom.acq_rel+flag", 3: "two_level_8_groups", 4: "per_cta_words+cta0_gather",
             5: "cg_grid_sync", 6: "atom.acq_rel+flag+warp0_any_sync_poll"}


def _cv(xs: list[float]) -> float | None:
    return statistics.pstdev(xs) / statistics.mean(xs) if len(xs) > 1 and statistics.mean(xs) else None


def _f(v: str):
    try:
        return int(v)
    except ValueError:
        return float(v)


def tcgen05_issuers(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(normalize({
                "family": "tc_mma", "primitive": "tcgen05.mma.issuers",
                "p.kind": "f16", "p.dtype": "bf16", "p.cta_group": 1, "p.K": 16,
                "p.M": int(r["M"]), "p.N": int(r["N"]),
                "p.a_src": "tmem" if r["atmem"] == "1" else "smem", "p.issuers": int(r["issuers"]),
                "m.cyc_per_mma": _f(r["sm_cyc_per_mma"]),
                "m.smem_model_cyc": _f(r["smem_model_cyc"]), "m.math_model_cyc": _f(r["math_cyc"]),
                "c.ctas": 1,
                "note": "MMAs issued from 1/2/4 threads in parallel; smem_model_cyc is the "
                        "operand-read time at the SM smem rate, math_model_cyc the tensor "
                        "pipe time; the larger one predicts cyc_per_mma",
            }, **{"s.bench": "tcgen05_rule/mma_bench.cu issuers", "s.file": str(path),
                  "s.timer": "clock64"}))
    return rows


def gemm_ab(path: Path, l2: str) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            swap = r["swap"] == "1"
            rows.append(normalize({
                "family": "gemm", "primitive": "gemm.cutlass_tcgen05",
                "p.M": int(r["m"]), "p.N": int(r["n"]), "p.K": int(r["k"]),
                "p.w_dtype": "bf16", "p.x_dtype": "bf16",
                "p.engine": "tcgen05_swapab" if swap else "tcgen05",
                "p.config": r["config"], "p.tile_m": int(r["tm"]), "p.tile_n": int(r["tn"]),
                "p.sched": r["mode"],
                "m.us": _f(r["us"]), "m.tflops": _f(r["tflops"]), "m.hbm_gbps": _f(r["hbm_gbps"]),
                "m.max_abs_err": _f(r["max_abs_err"]), "c.l2": l2,
                "verified": True,
                "note": "Y[M][N] = X[M][K] W[N][K]^T, M = tokens; swap-AB puts W rows in the MMA M "
                        "slot (tile_m) and tokens in N (tile_n). DP = data parallel, SK2 = split-K 2. "
                        + ("L2 warm: same buffers every rep" if l2 == "warm"
                           else "L2 cold: 512 MB of rotating copies"),
            }, **{"s.bench": "tcgen05_rule/gemm_ab (CUTLASS)", "s.file": str(path),
                  "s.timer": "cudaEvent"}))
    return rows


def _json_lines(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                yield json.loads(line)


def xover(path: Path) -> list[dict]:
    rows = []
    for r in _json_lines(path):
        if "eng" not in r:
            continue
        n, k = _XOVER_SHAPES[r["shape"]]
        rows.append(normalize({
            "family": "gemm_phase", "primitive": "gemm_phase.persistent",
            "p.M": r["T"], "p.N": n, "p.K": k, "p.w_dtype": "bf16", "p.x_dtype": "bf16",
            "p.engine": _XOVER_ENG[r["eng"]],
            "m.us": r["us_phase"], "m.regs": r["regs"], "m.spill": r["spill"],
            "c.sm_clock_mhz": r["mhz"], "c.ctas": 148, "c.threads": 512, "c.l2": "cold",
            "n_reps": len(r["us_reps"]), "cv": _cv(r["us_reps"]),
            "verified": r["pass"],
            "note": "one GEMM phase inside a persistent cooperative kernel (148 CTAs x 512, "
                    "1 CTA/SM), us per phase including one grid barrier; fp32 Y via red.add. "
                    "oracle = same loads and barrier, no math: the floor for this skeleton",
        }, **{"s.bench": "probe/xover/xover.cu", "s.file": str(path), "s.job": "9521",
              "s.timer": "cudaEvent"}))
    return rows


def xover2(path: Path) -> list[dict]:
    rows = []
    for r in _json_lines(path):
        if "us_phase" not in r:
            continue
        n, k = _XOVER2_SHAPES[r["shape"]]
        ku, xtma, ring, cg, oracle = _XOVER2_ARMS[r["bin"]]
        rows.append(normalize({
            "family": "gemm_phase", "primitive": "gemm_phase.persistent",
            "p.M": r["T"], "p.N": n, "p.K": k, "p.w_dtype": "bf16", "p.x_dtype": "bf16",
            "p.engine": "oracle_load_barrier_only" if oracle else "tcgen05_swapab",
            "p.cta_group": cg, "p.unit_rows": ku, "p.x_ingress": "tma" if xtma else "ld",
            "p.ring": r.get("ring", ring),
            "m.us": r["us_phase"], "m.regs": r["regs"], "m.spill": r["spill"],
            "c.sm_clock_mhz": r["mhz"], "c.ctas": 148, "c.threads": 512, "c.l2": "cold",
            "n_reps": len(r["us_reps"]), "cv": _cv(r["us_reps"]), "verified": r["pass"],
            "note": "one GEMM phase inside a persistent cooperative kernel (148 CTAs x 512, "
                    "1 CTA/SM), us per phase including one grid barrier; weights stream "
                    "through a TMA ring of unit_rows x K tiles, tokens are the MMA N (padded "
                    "to 16, 32 for cta_group 2); oracle = same loads and barrier, no math",
        }, **{"s.bench": "probe/mk_core/xover2/xover2.cu", "s.file": str(path), "s.job": "9566",
              "s.timer": "cudaEvent"}))
    return rows


def _attn_engine(defines: list[str]) -> str:
    d = set(defines)
    if "SKV_PROF_NOCOMP=1" in d:
        return "load_only"
    if "SKV_TC=1" in d:
        return "tcgen05_nomath" if "SKV_TC_NOMATH=1" in d else "tcgen05"
    return "ffma" if "SKV_TMA_MMA=0" in d else "mma_sync"


def attention(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for r in _json_lines(path):
            if r.get("kind") == "check" or "A_med" not in r:
                continue
            point = r["point"]
            batch, _, rest = point[1:].partition("kv")
            kv, _, q = rest.partition("q")
            rows.append(normalize({
                "family": "attn_phase", "primitive": "attention.splitkv_decode",
                "p.batch": int(batch), "p.kv": int(kv), "p.q_len": int(q or 1),
                "p.engine": _attn_engine(r["defines"]), "p.build": r["build"],
                "p.defines": ",".join(r["defines"]), "p.kv_ingress": r["engine"],
                "p.s_max": r["s_max"], "p.combine": r.get("comb"),
                "m.us_attend": r["A_med"], "m.us_attend_p10": r["A_p10"],
                "m.us_attend_p90": r["A_p90"], "m.us_combine": r["C_med"],
                "m.cta_work_us_med": r["W_med"], "m.cta_work_us_max": r["W_max"],
                "m.cta_imbalance_us": r["imb_med"], "m.kv_bytes_layer": r["kv_bytes_layer"],
                "c.ctas": r["n_cta"], "c.dyn_smem": r["dyn_smem"], "c.l2": "cold",
                "n_reps": r["launches"], "verified": None,
                "note": "decode attention phase of a persistent kernel, split over KV with a "
                        "merge; us per layer including the grid barrier (us_attend), the merge "
                        "(us_combine), per-CTA work time and its spread. engine names the QK/PV "
                        "math; load_only streams the same KV with no math (the floor). Numerics "
                        "were checked in separate runs of the same builds",
            }, **{"s.bench": "probe/mk_core/attn_ub/a1ub.py", "s.file": str(path),
                  "s.job": str(r.get("job", "")), "s.timer": "globaltimer"}))
    return rows


def grid_barrier(path: Path) -> list[dict]:
    rows = []
    for r in _json_lines(path):
        if "us_per_bar" not in r:
            continue
        rows.append(normalize({
            "family": "sync", "primitive": "sync.grid_barrier",
            "p.arm": _BAR_ARMS[r["bar"]], "p.ctas": r["G"], "p.skew_ns": r["skew_ns"],
            "m.us_per_bar": r["us_per_bar"], "m.regs": r["regs"],
            "c.sm_clock_mhz": r["mhz"], "c.threads": 512,
            "n_reps": len(r["us_reps"]), "cv": _cv(r["us_reps"]), "verified": True,
            "note": "persistent cooperative launch, loop of back-to-back barriers; skew_ns adds "
                    "a per-CTA spin before each arrive",
        }, **{"s.bench": "probe/mk_core/bar/bar.cu", "s.file": str(path), "s.job": "9579",
              "s.timer": "cudaEvent"}))
    return rows


def clc(paths: list[Path]) -> list[dict]:
    groups: dict = {}
    for path in paths:
        for r in _json_lines(path):
            groups.setdefault((str(path), r["arm"], r["C"]), []).append(r)
    rows = []
    for (path, arm, chunk), reps in sorted(groups.items()):
        bw = [r["bw_gbs"] for r in reps]
        ovh = [r["ovh_ns"] for r in reps if r.get("ovh_ns", -1) >= 0]
        first = reps[0]
        metrics = {"m.bw_gbs": statistics.median(bw),
                   "m.makespan_ns": statistics.median(r["makespan_ns"] for r in reps),
                   "m.tail_ns": statistics.median(r["tail_ns"] for r in reps)}
        if ovh:
            metrics["m.overhead_ns_per_chunk"] = statistics.median(ovh)
        rows.append(normalize({
            "family": "sched", "primitive": "sched.work_distribution",
            "p.arm": arm, "p.chunk_bytes": chunk, "p.nchunks": first["nchunks"],
            "p.grid": first["grid"], **metrics,
            "c.threads": first["threads"], "c.smem": first["smem"],
            "n_reps": len(reps), "cv": _cv(bw),
            "note": "streaming read split into chunks; arms: static split, guided, atomic queue "
                    "(queue_pf = with prefetch of the next index), hw = one CTA per chunk, "
                    "clc = cluster launch control try_cancel (clc_np without prefetch)",
        }, **{"s.bench": "probe/mk_core/clc", "s.file": path, "s.job": Path(path).stem.split("_")[-1],
              "s.timer": "globaltimer"}))
    return rows


def occfuse_models(path: Path) -> list[dict]:
    def walk(node, name):
        if isinstance(node, dict) and "formula" in node:
            yield name, node
        elif isinstance(node, dict):
            for key, child in node.items():
                yield from walk(child, key if key != name else name)

    rows = []
    for bench, tree in json.loads(path.read_text()).items():
        for name, model in walk(tree, bench):
            rows.append(normalize({
                "kind": "model", "family": "model", "primitive": f"model.{bench}.{name}",
                "model": model,
                "note": "fitted cost term for a persistent 148-CTA kernel; see model.formula "
                        "for the variables and model.range for where the fit holds",
            }, **{"s.bench": f"probe/mk_core/occfuse/{bench}", "s.file": str(path),
                  "s.job": ",".join(map(str, model.get("jobs", []))), "s.timer": "globaltimer"}))
    return rows


# xover5 arms: bin -> (X ingress, W ingress, ring slots; 0 = as many as fit). KU 128, cta_group 1.
# X "ld" = all threads LDG then STS into the swizzled slot; "cp_async" = cp.async.cg 16 B by all
# threads (wait_all + fence.proxy.async before the slot is published); "tma" = TMA box from a
# padded X map. W "cp_async" = the weight ring filled by all threads with cp.async.cg 16 B in the
# TMA box layout (completion by cp.async.mbarrier.arrive.noinc, count = threads).
_XOVER5_ARMS = {"a0": ("ld", "tma", 2), "c0": ("cp_async", "tma", 2), "x3": ("ld", "tma", 0),
                "c3": ("cp_async", "tma", 0), "a3": ("tma", "tma", 0), "w3": ("tma", "cp_async", 0),
                "wc3": ("cp_async", "cp_async", 0)}
_XOVER5_SHAPES = {"QKV": (4096, 1024), "O": (1024, 2048), "GU": (6144, 1024),
                  "D": (1024, 3072), "LM": (151936, 1024)}
_XOVER5_CHECK = re.compile(r"(PASS|FAIL|SKIP) (\S+) (\S+) T=(\d+) max\|d\|=\S+ bad=\d+(?: \(4096-row sample\))?\n")


def xover5(path: Path) -> list[dict]:
    """TMA vs cp.async ingress arms; one row per (bin, shape, T) with the two rounds pooled."""
    text = path.read_text()
    checks: dict = {}   # stderr check lines can land inside a block-buffered JSON line: cut them out
    for m in _XOVER5_CHECK.finditer(text):
        checks.setdefault((m.group(2), m.group(3), int(m.group(4))), []).append(m.group(1))
    text = _XOVER5_CHECK.sub("", text)
    job = path.stem.rsplit("-", 1)[-1]
    groups: dict = {}
    floors: dict = {}
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        key = (r["bin"], r["shape"], r["T"])
        if r.get("floor"):
            floors[key] = r
        else:
            groups.setdefault(key, []).append(r)
    rows = []
    for (b, shape, t), reps in sorted(groups.items()):
        xin, win, ring = _XOVER5_ARMS[b]
        n, k = _XOVER5_SHAPES[shape]
        samples = [x for r in reps for x in r["us_reps"]]
        flags = checks.get((b, shape, t), [])
        passed = [r["pass"] for r in reps if r.get("pass") is not None]
        verified = (all(passed) and "FAIL" not in flags) if (passed or flags) else None
        if flags and "FAIL" not in flags and not passed:
            verified = True
        first = reps[0]
        fl = floors.get((b, shape, t), {})
        rows.append(normalize({
            "family": "gemm_phase", "primitive": "gemm_phase.persistent",
            "p.M": t, "p.N": n, "p.K": k, "p.w_dtype": "bf16", "p.x_dtype": "bf16",
            "p.engine": "tcgen05_swapab", "p.cta_group": 1, "p.unit_rows": 128,
            "p.x_ingress": xin, "p.w_ingress": win, "p.ring": first["ring"], "p.arm": b,
            "m.us": statistics.median(r["us_phase"] for r in reps),
            "m.us_min_round": min(r["us_phase"] for r in reps),
            "m.us_max_round": max(r["us_phase"] for r in reps),
            "m.regs": first["regs"], "m.spill": first["spill"],
            "m.units_per_cta_max": first["units_per_cta_max"],
            **({"m.us_weight_stream_floor": fl["us_weight_stream"]} if fl else {}),
            "c.sm_clock_mhz": statistics.median(r["mhz"] for r in reps), "c.ctas": 148,
            "c.threads": 512, "c.l2": "cold",
            "n_reps": len(samples), "cv": _cv(samples), "verified": verified,
            "note": "one GEMM phase in a persistent cooperative kernel (148 CTAs x 512, 1 CTA/SM), "
                    "us per phase incl. one grid barrier, same skeleton as xover2 (weights cold from "
                    "a 2 GiB pool, tokens = MMA N). x_ingress / w_ingress: tma, cp_async (LDGSTS "
                    "by all threads) or ld (LDG + STS, X only). ring 0-requested = as many slots as "
                    "fit. us = median of 2 rounds x 5 reps in this job; us_weight_stream_floor is "
                    "the bytes / 148-SM stream time of the weights alone. verified = PASS line from "
                    "the in-bench check (run on T 1, 8, 64, 256 only), null otherwise",
        }, **{"s.bench": "probe/mk_core/xover5/xover5.cu", "s.file": str(path), "s.job": job,
              "s.timer": "cudaEvent"}))
    return rows


def b7_splitk(files: list[Path]) -> list[dict]:
    """Split-K price inside the persistent skeleton: output tile (KS 1), red.add and workspace fix-up pass."""
    groups: dict = {}
    for path in files:
        for r in _json_lines(path):
            if r.get("bench") != "b7_splitk" or r.get("orc") or "us_Tmean" not in r:
                continue
            key = (r["arm"], r["shape"], r["T"], r["KS"], r["pf"], str(r["job"]))
            groups.setdefault(key, []).append((path, r))
    rows = []
    for (arm, shape, t, ks, pf, job), items in sorted(groups.items()):
        rs = [r for _, r in items]
        samples = [x for r in rs for x in r["us_reps"]]
        if ks == 1:
            part, red = "output_tile", "none (plain fp32 store)"
        elif arm == "redadd":
            part, red = "split_k", "red.global.add.f32 into Y"
        else:
            part, red = "split_k", "workspace partials + second barrier + reduce pass by all CTAs"
        first = rs[0]
        rows.append(normalize({
            "family": "gemm_phase", "primitive": "gemm_phase.persistent",
            "p.M": t, "p.N": first["N"], "p.K": first["K"], "p.w_dtype": "bf16", "p.x_dtype": "bf16",
            "p.engine": "tcgen05_swapab", "p.cta_group": 1, "p.tile_n": first["TT"],
            "p.partition": part, "p.ks": ks, "p.reduction": red, "p.prefetch": "window" if pf else "none",
            "p.ring": first["ring"],
            "m.us": statistics.median(r["us_Tmean"] for r in rs),
            "m.us_min": min(r["us_Tmean"] for r in rs), "m.us_max": max(r["us_Tmean"] for r in rs),
            "m.busy_ctas": first["busy_ctas"], "m.tiles": first["tiles"],
            "m.max_steps": first["max_steps"], "m.tiles_max": first["tiles_max"],
            "m.regs": first["regs"],
            "c.sm_clock_mhz": statistics.median(r["mhz"] for r in rs), "c.ctas": first["grid"],
            "c.l2": "cold", "n_reps": len(samples), "cv": _cv([r["us_Tmean"] for r in rs]),
            "verified": all(r["pass"] == 1 for r in rs),
            "note": "output-tile vs split-K for a whole GEMM phase in the occfuse skeleton (148 CTAs, "
                    "tile order (row block, token tile, ks), CTA owns a contiguous tile range, ring of "
                    "A+B slots, TMEM accumulators). KS = k splits, so the partition is output_tile at "
                    "KS 1 and split_k above (equal k slices aligned to tiles; busy_ctas = CTAs with "
                    "work). us = telescoping mean phase time over 112 phases incl. the barrier, median "
                    "over passes. The partition table: us by (partition, ks). Stream-K (CTAs take "
                    "unaligned ranges of the flattened k space) is not in this skeleton; see "
                    "gemm.partition rows",
        }, **{"s.bench": "probe/mk_core/occfuse/b7_splitk/b7_splitk.cu",
              "s.file": str(items[0][0].parent / f"*_{job}.jsonl"), "s.job": job,
              "s.timer": "globaltimer"}))
    return rows


_B8_LINE = re.compile(r"^(\w+)\s+(\d+)\s+(\w+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+\(\s*([+-]\d+)%\)\s+\|\s+"
                      r"([\d.]+)\+\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def b8_terms(analyze: Path, results: Path) -> list[dict]:
    """b8_compose term-sum check: predicted vs measured phase time per (design, T, phase)."""
    import subprocess
    try:
        out = subprocess.run([sys.executable, "-W", "ignore", str(analyze), str(results), "terms"],
                             capture_output=True, text=True, timeout=300,
                             env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    except (OSError, subprocess.SubprocessError) as e:
        print(f"skip: b8 terms ({e})", file=sys.stderr)
        return []
    rows = []
    for line in out.stdout.splitlines():
        m = _B8_LINE.match(line.strip())
        if not m:
            continue
        design, t, phase, pred, meas, _, stream, bpath, chain, piece, fin = m.groups()
        pred, meas = float(pred), float(meas)
        rows.append(normalize({
            "family": "model_check", "primitive": "model_check.b8_compose_terms",
            "p.design": design, "p.M": int(t), "p.phase": phase,
            "m.pred_us": pred, "m.meas_us": meas, "m.residual": round(meas - pred, 4),
            "m.rel_err": round(pred / meas - 1, 4),
            "m.term_stream_us": float(stream), "m.term_bpath_us": float(bpath),
            "m.term_chain_us": float(chain), "m.term_piece_us": float(piece),
            "m.term_finalizer_us": float(fin),
            "verified": None,
            "note": "pred = t_bar 1.2 + max(stream + B path, MMA chain 0.525 us x max steps) + piece "
                    "(red.add / plain tile, b7 fit) + finalizer; every coefficient from "
                    "occfuse/params.json (b0 b1 b2 b3 b4 b6 b7), no refit; residual = meas - pred. "
                    "Computed by occfuse/b8_compose/analyze.py `terms` on the job 14306 stamps. "
                    "model rows used: model.b0_calib.t_bar, model.b1_stream.t_stream, "
                    "model.b3_step.t_chain, model.b7_splitk.redadd_T, model.b6_fin.*",
            "model": {"formula": "t_bar + max(stream + bpath, chain) + piece + fin",
                      "rows": ["model.b0_calib.t_bar", "model.b1_stream.t_stream",
                               "model.b2_prefetch.hidden", "model.b3_step.t_chain",
                               "model.b4_bpath.*", "model.b6_fin.*", "model.b7_splitk.redadd_T"]},
        }, **{"s.bench": "probe/mk_core/occfuse/b8_compose/analyze.py terms",
              "s.file": str(results), "s.job": "14306", "s.timer": "globaltimer"}))
    return rows


def primbench(out: Path) -> list[dict]:
    """Rows printed by the primbench families (already in p./m./c. form)."""
    job = out.name
    rows = []
    for path in sorted(out.glob("*.jsonl")):
        for line in _json_lines(path):
            rows.append(normalize(line, **{"s.bench": f"megabench/primbench/{ {'tc_overhead': 'tc_mma'}.get(path.stem, path.stem) }.cu",
                                           "s.file": str(path), "s.job": job, "s.timer": "clock64"}))
    return rows


def facts() -> list[dict]:
    return [normalize({
        "kind": "fact", "family": "tc_mma", "primitive": "wgmma",
        "p.arch": "sm_100", "available": False,
        "note": "wgmma.mma_async is sm_90a only and does not assemble for sm_100/sm_100a; "
                "the Blackwell tensor-core path is tcgen05.mma (or warp-level mma.sync)",
    }, **{"s.bench": "PTX ISA target table"}), normalize({
        "kind": "fact", "family": "mma_sync", "primitive": "mma.sync",
        "p.arch": "sm_100a", "p.dtype": "e4m3,e5m2", "p.shape": "m16n8k32", "native": False,
        "note": "fp8 mma.sync is emulated on sm_100a: ptxas emits F2FP (fp8 to f16), two "
                "HMMA.16816.F32 and FADDs per instruction, and an all-ones check returned "
                "wrong sums; native fp8 tensor-core math is tcgen05.mma kind::f8f6f4",
    }, **{"s.bench": "primbench/mma_sync.cu SASS (cuobjdump)"})]


def build(tcgen05: Path | None, probes: Path | None, results: Path | None,
          bench_out: Path | None = None) -> list[dict]:
    rows: list[dict] = list(facts())
    sources = []
    if tcgen05:
        # micro.csv is superseded by primbench/tc_mma.cu (same kernel, all kinds; matched within 0.01%)
        sources += [(tcgen05_issuers, tcgen05 / "issuers.csv"),
                    (lambda p: gemm_ab(p, "cold"), tcgen05 / "full.csv"),
                    (lambda p: gemm_ab(p, "cold"), tcgen05 / "quick.csv"),
                    (lambda p: gemm_ab(p, "warm"), tcgen05 / "warm.csv")]
    if probes:
        sources += [(xover, probes / "xover/logs/slurm-9521.log"),
                    (xover2, probes / "mk_core/xover2/logs/slurm-9566.log"),
                    (grid_barrier, probes / "mk_core/bar/logs/slurm-9579.log"),
                    (occfuse_models, probes / "mk_core/occfuse/params.json"),
                    (xover5, probes / "mk_core/xover5/logs/slurm-10272.log"),
                    (xover5, probes / "mk_core/xover5/logs/slurm-10273.log")]
    for fn, path in sources:
        if path.is_file():
            rows += fn(path)
        else:
            print(f"skip: {path} not found", file=sys.stderr)
    if results:
        paths = sorted(Path(p) for p in glob.glob(str(results / "*-clc/clc_*.jsonl")))
        rows += clc(paths)
        rows += attention(sorted(Path(p) for p in glob.glob(str(results / "*-attn-ub/*.jsonl"))
                                 + glob.glob(str(results / "*-mk-tc/a1_*.jsonl"))))
        b7 = sorted(Path(p) for p in glob.glob(str(results / "*-occfuse/b7_splitk/p[12]_*.jsonl")))
        if b7:
            rows += b7_splitk(b7)
        analyze = probes / "mk_core/occfuse/b8_compose/analyze.py" if probes else None
        b8 = next(iter(glob.glob(str(results / "*-occfuse/b8_compose"))), None)
        if analyze and analyze.is_file() and b8:
            rows += b8_terms(analyze, Path(b8))
    if bench_out:
        rows += primbench(bench_out)
    rows += composed.model_check(rows)
    seen: dict = {}
    for r in rows:
        seen[r["id"]] = r  # a later file with the same point replaces an earlier one
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="primdb-import", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tcgen05", type=Path)
    p.add_argument("--probes", type=Path)
    p.add_argument("--results", type=Path)
    p.add_argument("--primbench", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    a = p.parse_args(argv)
    rows = build(a.tcgen05, a.probes, a.results, a.primbench)
    with a.output.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    meta = {"rows": len(rows), "families": sorted({r["family"] for r in rows}),
            "sources": sorted({r["source"].get("file", r["source"].get("bench", "")) for r in rows})}
    if a.primbench:
        for name in ("gpu.txt", "nvcc.txt"):
            if (a.primbench / name).is_file():
                meta[name.removesuffix(".txt")] = (a.primbench / name).read_text().strip()
        meta["primbench_job"] = a.primbench.name
    a.output.with_name("meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {len(rows)} rows to {a.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
