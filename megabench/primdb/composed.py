"""Composed-cost check: predict measured GEMM-phase points from primitive rows, report the residual.

A GEMM phase of the persistent 148-CTA kernel (xover2 / xover5, tcgen05 swap-AB, weights as MMA M,
tokens as N) is predicted from rows that are already in the database, no refit:

  pred = t_bar + max(stream, mma) + epilogue
  t_bar    grid barrier (model.b0_calib.t_bar at 148 CTAs)
  stream   t0(R) + S / min(r_sm(R), BW_agg(R) / active CTAs)   model.b1_stream.t_stream, S = weight
           bytes of the busiest CTA, R = ring slots (capped at the largest fitted R)
  mma      units x 8 K16 MMAs x cyc_per_mma / SM clock, cyc_per_mma from the tcgen05.mma row at
           M=128, N = tokens padded to 16, bf16, A and B in smem, one issuer (the best naccum row)
  epilogue c_red x row blocks per CTA x tokens/128   model.b7_splitk.redadd_T (red.add per 128x128 tile)

Rows: family `model_check`, `m.pred_us`, `m.meas_us`, `m.residual` = meas - pred, `m.rel_err` =
pred / meas - 1, the term values, and in `model.rows` the primitive names, in `params.meas_row` the id
of the measured row. Without the b0/b1/b7 model rows nothing is emitted.
"""

from __future__ import annotations

import sys
from math import ceil

from . import normalize

CTAS = 148
UNIT_BYTES = 128 * 128 * 2     # one weight unit: 128 rows x 128 k, bf16
MMAS_PER_UNIT = 8              # K = 128 in K16 steps


def _model(rows: list[dict], primitive: str):
    for r in rows:
        if r.get("kind") == "model" and r["primitive"] == primitive:
            return r
    return None


def _mma_cycles(rows: list[dict], n_tok: int):
    """(cyc_per_mma, row id) of the tcgen05.mma row at M=128, bf16, smem A, 1 issuer, N = n_tok."""
    best = None
    for r in rows:
        p = r["params"]
        if (r["primitive"] == "tcgen05.mma" and p.get("M") == 128 and p.get("N") == n_tok
                and p.get("dtype") == "bf16" and p.get("a_src") == "smem" and p.get("issuers") == 1
                and p.get("K") == 16):
            c = r["metrics"].get("cyc_per_mma")
            if c is not None and (best is None or c < best[0]):
                best = (c, r["id"])
    return best


def _interp(xs: list, ys: list, x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
        if x <= b:
            return ya + (yb - ya) * (x - a) / (b - a)
    return ys[-1]


def model_check(rows: list[dict]) -> list[dict]:
    tbar = _model(rows, "model.b0_calib.t_bar")
    strm = _model(rows, "model.b1_stream.t_stream")
    red = _model(rows, "model.b7_splitk.redadd_T")
    if not (tbar and strm and red):
        print("skip: composed-cost check needs model.b0_calib.t_bar, b1_stream.t_stream, "
              "b7_splitk.redadd_T rows", file=sys.stderr)
        return []
    tb = _interp(tbar["model"]["coeffs"]["N"], tbar["model"]["coeffs"]["t_bar_us"], CTAS)
    sc = strm["model"]["coeffs"]
    c_red = red["model"]["coeffs"]["values"][2]
    out = []
    n_noma = 0
    for r in rows:
        p = r["params"]
        if not (r["family"] == "gemm_phase" and r["kind"] == "measure"
                and p.get("engine") == "tcgen05_swapab" and p.get("cta_group") == 1
                and p.get("unit_rows") == 128 and p.get("x_ingress") == "tma"
                and p.get("w_ingress", "tma") == "tma" and "partition" not in p):
            continue
        t, n, k = p["M"], p["N"], p["K"]
        units = (n // 128) * (k // 128)
        upc = ceil(units / CTAS)
        active = min(CTAS, units)
        ring = int(p.get("ring") or 0) or 7
        rs = sc["R"]
        i = min(range(len(rs)), key=lambda j: abs(rs[j] - min(ring, rs[-1])))
        rate = min(sc["r_sm_GBps"][i], sc["BW_agg_GBps"][i] / active) * 1e3   # bytes / us
        stream = sc["t0_us"][i] + upc * UNIT_BYTES / rate
        mhz = r["cond"].get("sm_clock_mhz") or 1850.0
        npad = max(16, -(-t // 16) * 16)
        mma_row = _mma_cycles(rows, npad)
        mma = upc * MMAS_PER_UNIT * mma_row[0] / mhz if mma_row else 0.0
        n_noma += mma_row is None
        nrb = ceil(upc / (k // 128)) + 1
        epi = c_red * nrb * max(t, 16) / 128.0
        pred = tb + max(stream, mma) + epi
        meas = r["metrics"]["us"]
        used = ["model.b0_calib.t_bar", "model.b1_stream.t_stream", "model.b7_splitk.redadd_T"]
        ids = [tbar["id"], strm["id"], red["id"]]
        if mma_row:
            used.append(f"tcgen05.mma M=128 N={npad} K=16 bf16 smem ({mma_row[0]} cyc)")
            ids.append(mma_row[1])
        out.append(normalize({
            "family": "model_check", "primitive": "model_check.gemm_phase",
            "p.M": t, "p.N": n, "p.K": k, "p.meas_row": r["id"], "p.meas_source": r["source"].get("bench"),
            "p.x_ingress": "tma", "p.ring": p.get("ring"),
            "m.pred_us": round(pred, 4), "m.meas_us": meas, "m.residual": round(meas - pred, 4),
            "m.rel_err": round(pred / meas - 1, 4),
            "m.term_barrier_us": round(tb, 4), "m.term_stream_us": round(stream, 4),
            "m.term_mma_us": round(mma, 4), "m.term_epilogue_us": round(epi, 4),
            "m.units_per_cta": upc,
            "c.sm_clock_mhz": mhz, "c.l2": "cold", "verified": None,
            "model": {"formula": "t_bar + max(t0(R) + S/min(r_sm, BW/active), units*8*cyc_mma/MHz) "
                                 "+ c_red * (ceil(units_per_cta / (K/128)) + 1) * max(T,16)/128",
                      "rows": used, "row_ids": ids,
                      "coeffs": {"t_bar_us": round(tb, 4), "c_red_us": c_red, "ring_fit_R": rs[i]}},
            "note": "predicted GEMM-phase time from primitive rows, no refit: barrier + max(weight "
                    "stream of the busiest CTA, MMA issue time) + red.add epilogue; residual = "
                    "meas - pred (positive: the model under-predicts). Excludes producer / consumer "
                    "handoff, X ingress and TMEM load latency, so a roughly constant positive "
                    "residual is the unmodelled per-phase chain",
        }, **{"s.bench": "megabench/primdb/composed.py", "s.file": r["source"].get("file", ""),
              "s.job": r["source"].get("job", ""), "s.timer": "derived"}))
    if n_noma:
        print(f"note: {n_noma} model_check rows without a tcgen05.mma row (MMA term 0)", file=sys.stderr)
    return out
