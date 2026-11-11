"""Derive a weighted regime set for a case from a production request trace (CPU only).

Replays an Azure-LLM-inference-style request trace through a simple continuous-batching
scheduler, records every step (decode batch, KV lengths, prefill tokens, duration), buckets
the decode steps by batch size and KV length, and writes a regime file for the case.
The public dataset is "Azure LLM inference traces 2024" (github.com/Azure/AzurePublicDataset).

    python -m megabench.integrations.trace_regimes --case ID --trace code --window median \
        --trace-dir DIR --max-seqs 64 --max-batched-tokens 8192 --chunked-prefill \
        --step-ms 1 --out regimes.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BATCHED_FAMILIES = frozenset({"dense_step", "quant_step"})  # families that accept batch > 1 and ragged contexts
HEADER = [b"TIMESTAMP", b"ContextTokens", b"GeneratedTokens"]


class UsageError(Exception):
    """Bad or missing input; the CLI exits with status 2."""


# --------------------------------------------------------------------------- trace loading

@dataclass
class Trace:
    t: np.ndarray    # float64 seconds since the first request
    inp: np.ndarray  # int32 prompt tokens
    out: np.ndarray  # int32 generated tokens


@dataclass
class Window:
    name: str
    t: np.ndarray    # seconds from window start
    inp: np.ndarray
    out: np.ndarray
    duration_s: float

    def __len__(self) -> int:
        return len(self.t)


def _parse_ts(col: list[bytes]) -> np.ndarray:
    a = np.array(col, dtype="S")
    if a.size and not (np.char.str_len(a) == 32).all():  # stamps without fractional digits etc.
        a = np.array([c.replace(b"+00:00", b"") for c in col], dtype="S")
        return a.astype("datetime64[us]")
    return a.astype("S26").astype("datetime64[us]")


def parse_csv(path: Path, block: int = 2_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream ``TIMESTAMP,ContextTokens,GeneratedTokens`` into (datetime64[us], int32, int32)."""
    ts_l, in_l, out_l = [], [], []
    with open(path, "rb") as f:
        header = f.readline().strip().split(b",")
        if header != HEADER:
            raise UsageError(f"{path}: expected header {b','.join(HEADER).decode()}, got {header}")
        while True:
            lines = list(itertools.islice(f, block))
            if not lines:
                break
            parts = [ln.rstrip().split(b",") for ln in lines if ln.strip()]
            ts_l.append(_parse_ts([p[0] for p in parts]))
            in_l.append(np.array([p[1] for p in parts], dtype="S").astype(np.int32))
            out_l.append(np.array([p[2] for p in parts], dtype="S").astype(np.int32))
    return np.concatenate(ts_l), np.concatenate(in_l), np.concatenate(out_l)


def load_trace(csv: Path, cache_dirs: list[Path]) -> Trace:
    """Load a 1-week CSV; the first cache dir that is writable keeps an .npz of the arrays."""
    name = f"_cache_{csv.stem}.npz"
    for d in cache_dirs:
        if (d / name).exists():
            z = np.load(d / name)
            return Trace(z["t"], z["inp"], z["out"])
    ts, inp, out = parse_csv(csv)
    order = np.argsort(ts, kind="stable")
    ts, inp, out = ts[order], inp[order], out[order]
    t = (ts - ts[0]).astype("timedelta64[us]").astype(np.int64) / 1e6
    for d in cache_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / name, t=t, inp=inp, out=out)
            break
        except OSError:
            continue
    return Trace(t, inp, out)


def minute_counts(t: np.ndarray) -> np.ndarray:
    return np.bincount((t // 60).astype(np.int64), minlength=int(t[-1] // 60) + 1)


def pick_windows(counts: np.ndarray, win_min: int) -> dict[str, int]:
    """Start minute of the peak / median / trough windows by rolling request rate.

    Windows with a zero-request minute (trace gaps) are excluded; picks do not overlap.
    """
    n = len(counts)
    if n < win_min:
        raise UsageError(f"trace is {n} minutes long, shorter than --window-minutes {win_min}")
    cs = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    roll = (cs[win_min:] - cs[:-win_min]) / (win_min * 60.0)
    zc = np.concatenate([[0], np.cumsum(counts == 0)])
    valid = (zc[win_min:] - zc[:-win_min]) == 0
    if not valid.any():
        raise UsageError("no gap-free window in the trace")
    taken = np.zeros(len(roll), dtype=bool)

    def pick(score: np.ndarray) -> int:
        ok = valid & ~taken
        if not ok.any():
            raise UsageError("no non-overlapping window left")
        s = int(np.argmin(np.where(ok, score, np.inf)))
        taken[max(0, s - win_min + 1): s + win_min] = True
        return s

    med = float(np.median(roll[valid]))
    out = {"peak": pick(-roll)}
    out["median"] = pick(np.abs(roll - med))
    out["trough"] = pick(roll)
    return out


def extract_window(tr: Trace, start_min: int, name: str, win_min: int) -> Window:
    t0, t1 = start_min * 60.0, (start_min + win_min) * 60.0
    lo, hi = np.searchsorted(tr.t, [t0, t1], side="left")
    return Window(name, tr.t[lo:hi] - t0, tr.inp[lo:hi].copy(), tr.out[lo:hi].copy(), win_min * 60.0)


def window_stats(w: Window) -> dict:
    def q(x):
        return {"p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
                "p99": float(np.percentile(x, 99)), "max": int(x.max())}
    return {"requests": len(w), "duration_s": w.duration_s, "rps_mean": len(w) / w.duration_s,
            "in_tokens": q(w.inp), "out_tokens": q(w.out)}


def load_window_file(path: Path, win_min: int) -> Window:
    """JSONL rows ``{"t": seconds, "in": prompt tokens, "out": generated tokens}``."""
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    if not rows:
        raise UsageError(f"{path} has no requests")
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    order = np.argsort(t, kind="stable")
    return Window(path.stem, t[order], np.array([r["in"] for r in rows], dtype=np.int32)[order],
                  np.array([r["out"] for r in rows], dtype=np.int32)[order], win_min * 60.0)


# --------------------------------------------------------------------------- replay

@dataclass
class Sched:
    max_seqs: int
    max_batched_tokens: int
    chunked_prefill: bool
    max_context: int
    load_scale: float = 1.0
    max_steps: int = 5_000_000
    max_sim_seconds: float = 7200.0


@dataclass
class StepLog:
    d: np.ndarray        # decode sequences per step
    kvmax: np.ndarray    # max KV length among them (0 if none)
    kvmean: np.ndarray
    pf: np.ndarray       # prefill tokens in the step
    dur: np.ndarray      # seconds
    idle_s: float
    truncated: str
    counts: dict
    contexts: dict       # step index -> KV lengths (only for requested steps)


def clip_lengths(inp: np.ndarray, out: np.ndarray, max_context: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Clamp to >= 1 and shrink the prompt first so prompt + output <= max_context."""
    i0, o0 = np.maximum(inp.astype(np.int64), 1), np.maximum(out.astype(np.int64), 1)
    o = np.minimum(o0, max_context - 1)
    i = np.minimum(i0, max_context - o)
    return i, o, int(((i != i0) | (o != o0)).sum())


def replay(w: Window, s: Sched, step_time, want: frozenset = frozenset()) -> StepLog:
    """Discrete-event continuous batching. ``step_time(d, kvmax, pf) -> seconds``.

    A sequence's KV length when a decode step runs is prompt + tokens generated so far - 1
    (the newest token is fed this step), so the first decode step sees KV == prompt.
    """
    arr = w.t / s.load_scale
    P, O, nclip = clip_lengths(w.inp, w.out, s.max_context)
    if not s.chunked_prefill:
        P = np.minimum(P, s.max_batched_tokens)  # a prompt must fit one step
    n, i, now, idle = len(arr), 0, 0.0, 0.0
    waiting: deque[int] = deque()
    partial: list[list[int]] = []                  # [request, prompt tokens left] mid-prefill
    kv, rem = np.empty(0, np.int64), np.empty(0, np.int64)   # decode sequences
    D, KM, KA, PF, DUR, ctxs = [], [], [], [], [], {}
    truncated, done = "", 0
    while i < n or waiting or partial or len(kv):
        if not (waiting or partial or len(kv)):
            idle += max(0.0, arr[i] - now)
            now = max(now, arr[i])
        while i < n and arr[i] <= now:
            waiting.append(i)
            i += 1
        d = len(kv)
        budget = max(s.max_batched_tokens - d, 0)
        pf, nrun, joined = 0, d + len(partial), []
        still = []
        for r, left in partial:                    # continue chunked prompts first
            take = min(left, budget)
            budget, pf = budget - take, pf + take
            if take == left:
                joined.append(r)
            else:
                still.append([r, left - take])
        partial = still
        while waiting and nrun < s.max_seqs:
            r, p = waiting[0], int(P[waiting[0]])
            take = min(p, budget) if s.chunked_prefill else (p if p <= budget else 0)
            if take <= 0:
                break
            waiting.popleft()
            budget, pf, nrun = budget - take, pf + take, nrun + 1
            if take == p:
                joined.append(r)
            else:
                partial.append([r, p - take])
        step = len(D)
        D.append(d)
        KM.append(int(kv.max()) if d else 0)
        KA.append(float(kv.mean()) if d else 0.0)
        PF.append(pf)
        if step in want:
            ctxs[step] = kv.copy()
        dt = step_time(d, KM[-1], pf)
        DUR.append(dt)
        now += dt
        kv, rem = kv + 1, rem - 1
        keep = rem > 0
        done += int((~keep).sum())
        kv, rem = kv[keep], rem[keep]
        for r in joined:                           # first token comes from the prefill step
            if O[r] > 1:
                kv, rem = np.append(kv, P[r]), np.append(rem, O[r] - 1)
            else:
                done += 1
        if len(D) >= s.max_steps:
            truncated = f"--max-steps {s.max_steps}"
            break
        if now >= s.max_sim_seconds:
            truncated = f"--max-sim-seconds {s.max_sim_seconds}"
            break
    counts = {"requests": n, "completed": done, "clipped_requests": nclip,
              "steps": len(D), "sim_seconds": now}
    return StepLog(np.array(D), np.array(KM), np.array(KA), np.array(PF), np.array(DUR),
                   idle, truncated, counts, ctxs)


def make_step_time(step_ms: float | None, timings: dict | None):
    """Constant step time, or the nearest measured regime (log batch, log KV) for decode steps."""
    if timings is None:
        return lambda d, kvmax, pf: step_ms / 1e3
    rows = []
    for r in timings["regimes"]:
        ms = r.get("candidate_cuda_event_p50_ms")
        if ms is None:
            continue
        ctx = r.get("contexts") or []
        rows.append((np.log2(r["params"].get("batch", len(ctx) or 1)),
                     np.log2(max(r["params"].get("context", max(ctx, default=1)), 1)), ms / 1e3))
    if not rows:
        raise UsageError("--step-timings has no regimes with candidate_cuda_event_p50_ms")
    arr, cache = np.array(rows), {}

    def f(d, kvmax, pf):
        if pf or d == 0:
            return step_ms / 1e3
        key = (d, kvmax)
        if key not in cache:
            cache[key] = float(arr[np.argmin(np.abs(arr[:, 0] - np.log2(d)) +
                                             np.abs(arr[:, 1] - np.log2(max(kvmax, 1)))), 2])
        return cache[key]
    return f


# --------------------------------------------------------------------------- bucketing

def pow2_edges(top: int) -> list[int]:
    e = [1]
    while e[-1] < top:
        e.append(min(e[-1] * 2, top))
    return e


def bucket_of(x: np.ndarray, edges: list[int]) -> np.ndarray:
    """Index of the first edge >= x; buckets are (previous edge, edge], the first starts at 1."""
    return np.minimum(np.searchsorted(edges, x, side="left"), len(edges) - 1)


def _range(edges: list[int], k: int) -> list[int]:
    return [1 if k == 0 else edges[k - 1] + 1, edges[k]]


def _label(prefix: str, lohi: list[int]) -> str:
    return f"{prefix}{lohi[0]}" if lohi[0] == lohi[1] else f"{prefix}{lohi[0]}-{lohi[1]}"


def analyze(log: StepLog, batch_edges: list[int], kv_edges: list[int]) -> tuple[dict, dict]:
    """Histogram report plus the decode-only buckets (key -> step indices) for regime picking."""
    total = float(log.dur.sum())
    dec = np.flatnonzero((log.d > 0) & (log.pf == 0))
    bb, kb = bucket_of(log.d, batch_edges), bucket_of(log.kvmax, kv_edges)
    buckets, rows = {}, []
    for key in sorted(set(zip(bb[dec].tolist(), kb[dec].tolist()))):
        idx = dec[(bb[dec] == key[0]) & (kb[dec] == key[1])]
        buckets[key] = idx
        km = log.kvmax[idx]
        rows.append({"kind": "decode", "batch": _range(batch_edges, key[0]), "kv": _range(kv_edges, key[1]),
                     "steps": len(idx), "time_share": float(log.dur[idx].sum() / total),
                     "batch_mean": float(log.d[idx].mean()), "kv_max_mean": float(km.mean()),
                     "kv_mean_mean": float(log.kvmean[idx].mean()),
                     "kv_max_quantiles": {f"p{q}": float(np.percentile(km, q)) for q in (50, 90, 99)}})
    for kind, m in (("prefill_only", (log.pf > 0) & (log.d == 0)), ("mixed", (log.pf > 0) & (log.d > 0))):
        rows.append({"kind": kind, "steps": int(m.sum()), "prefill_tokens": int(log.pf[m].sum()),
                     "time_share": float(log.dur[m].sum() / total)})
    return {"total_busy_s": total, "idle_s": log.idle_s, "buckets": rows}, buckets


def pick_regimes(log: StepLog, buckets: dict, batch_edges, kv_edges, family: str,
                 max_regimes: int | None = None) -> tuple[list[dict], dict]:
    """One candidate per non-empty decode-only bucket the family can express.

    Returns (candidates, info); each candidate carries its representative step index.
    """
    batched = family in BATCHED_FAMILIES
    groups: dict = {}
    for (bk, kk), idx in buckets.items():
        if batched:
            groups[(bk, kk)] = idx
        else:
            one = idx[log.d[idx] == 1]
            if len(one):
                groups[(0, kk)] = np.concatenate([groups.get((0, kk), np.empty(0, int)), one])
    total = float(log.dur.sum())
    cands = []
    for (bk, kk), idx in groups.items():
        order = idx[np.argsort(log.kvmax[idx], kind="stable")]
        cands.append({"step": int(order[len(order) // 2]), "bk": bk, "kk": kk,
                      "time": float(log.dur[idx].sum()), "steps": len(idx)})
    scorable = sum(c["time"] for c in cands)
    info = {"scorable_time_share": scorable / total, "merged": {}}
    if not cands:
        return [], info
    while max_regimes and len(cands) > max_regimes:   # merge the lightest bucket into its nearest
        cands.sort(key=lambda c: c["time"])
        a, rest = cands[0], cands[1:]
        near = min(rest, key=lambda c: abs(c["bk"] - a["bk"]) + abs(c["kk"] - a["kk"]))
        near["time"] += a["time"]
        near["steps"] += a["steps"]
        near.setdefault("absorbed", []).append(a)
        cands = rest
    for c in cands:
        c["weight"] = c["time"] / scorable
        c["batch_range"] = _range(batch_edges, c["bk"]) if batched else [1, 1]
        c["kv_range"] = _range(kv_edges, c["kk"])
    cands.sort(key=lambda c: (c["bk"], c["kk"]))
    return cands, info


def regime_entries(log: StepLog, cands: list[dict], family: str, contexts: dict) -> tuple[list[dict], dict]:
    batched, merged = family in BATCHED_FAMILIES, {}
    out = []
    for c in cands:
        rid = _label("b", c["batch_range"]) + "-" + _label("kv", c["kv_range"])
        if batched:
            ctx = [int(x) for x in contexts[c["step"]]]
            out.append({"id": rid, "weight": c["weight"],
                        "params": {"batch": len(ctx), "context": max(ctx)}, "contexts": ctx})
        else:
            out.append({"id": rid, "weight": c["weight"], "params": {"context": int(log.kvmax[c["step"]])}})
        if c.get("absorbed"):
            merged[rid] = [_label("b", a["batch_range"]) + "-" + _label("kv", a["kv_range"]) for a in c["absorbed"]]
    return out, merged


# --------------------------------------------------------------------------- CLI

def _edges(text: str | None, top: int) -> list[int]:
    e = sorted({int(x) for x in text.split(",")}) if text else pow2_edges(top)
    return e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trace_regimes", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, help="case id; the regime set is shaped for its family")
    p.add_argument("--trace-dir", help="dir with AzureLLMInferenceTrace_<NAME>_1week.csv (or env MEGABENCH_TRACE_DIR)")
    p.add_argument("--trace", help="trace name, e.g. code or conv")
    p.add_argument("--window", default="median", help="peak | median | trough | <start minute>")
    p.add_argument("--window-minutes", type=int, default=20)
    p.add_argument("--window-file", help="JSONL rows {t, in, out} instead of --trace")
    p.add_argument("--load-scale", type=float, default=1.0, help="arrivals at t / load-scale")
    p.add_argument("--max-seqs", type=int, required=True)
    p.add_argument("--max-batched-tokens", type=int, required=True)
    p.add_argument("--chunked-prefill", action="store_true")
    p.add_argument("--max-context", type=int, required=True, help="serving context limit; clip prompt + output (prompt shrinks first)")
    p.add_argument("--step-ms", type=float, help="constant step time in ms")
    p.add_argument("--step-timings", help="regime-evaluation result JSON used for decode steps")
    p.add_argument("--max-steps", type=int, default=5_000_000)
    p.add_argument("--max-sim-seconds", type=float, default=7200.0)
    p.add_argument("--batch-edges", help="comma-separated upper bounds (default powers of two to max-seqs)")
    p.add_argument("--kv-edges", help="comma-separated upper bounds (default powers of two to max-context)")
    p.add_argument("--max-regimes", type=int, help="merge the lightest buckets into neighbors down to N")
    p.add_argument("--out", required=True)
    return p


def _get_window(a) -> tuple[Window, dict]:
    if a.window_file:
        w = load_window_file(Path(a.window_file), a.window_minutes)
        return w, {"window_file": a.window_file}
    d = a.trace_dir or os.environ.get("MEGABENCH_TRACE_DIR")
    if not d or not a.trace:
        raise UsageError(
            "no trace data source. Pass --trace NAME with --trace-dir DIR (or env MEGABENCH_TRACE_DIR), "
            "or --window-file PATH. The data is the public Azure LLM inference traces 2024 "
            "(github.com/Azure/AzurePublicDataset): download AzureLLMInferenceTrace_<NAME>_1week.csv "
            "and point MEGABENCH_TRACE_DIR at its directory.")
    csv = Path(d) / f"AzureLLMInferenceTrace_{a.trace}_1week.csv"
    if not csv.exists():
        raise UsageError(f"{csv} not found")
    tr = load_trace(csv, [csv.parent, Path(a.out).resolve().parent / "trace_cache"])
    if a.window in ("peak", "median", "trough"):
        start = pick_windows(minute_counts(tr.t), a.window_minutes)[a.window]
    else:
        start = int(a.window)
    w = extract_window(tr, start, f"{a.trace}_{a.window}", a.window_minutes)
    if not len(w):
        raise UsageError("selected window has no requests")
    return w, {"trace": a.trace, "csv": str(csv), "window": a.window, "start_minute": start,
               "window_minutes": a.window_minutes, "window_stats": window_stats(w)}


def run(a) -> dict:
    from ..cases import select_cases
    try:
        case = select_cases("all", [a.case])[0]
    except ValueError as e:
        raise UsageError(str(e))
    if a.step_ms is None and not a.step_timings:
        raise UsageError(
            "the replay needs a step-time model; none is assumed. Pass --step-ms X (constant) or "
            "--step-timings FILE, a regime-evaluation result JSON measured on the target GPU "
            "(per-regime candidate_cuda_event_p50_ms); --step-ms is then used for prefill steps.")
    timings = None
    if a.step_timings:
        if a.step_ms is None:
            raise UsageError("--step-timings covers decode steps only; also pass --step-ms for prefill/mixed steps")
        timings = json.load(open(a.step_timings))
    if a.max_seqs < 1 or a.max_batched_tokens < 1 or a.max_context < 2 or a.load_scale <= 0:
        raise UsageError("--max-seqs, --max-batched-tokens, --max-context (>= 2) and --load-scale must be positive")
    w, wsrc = _get_window(a)
    s = Sched(a.max_seqs, a.max_batched_tokens, a.chunked_prefill, a.max_context, a.load_scale,
              a.max_steps, a.max_sim_seconds)
    st = make_step_time(a.step_ms, timings)
    log = replay(w, s, st)
    if not len(log.d):
        raise UsageError("replay produced no steps")
    batch_edges, kv_edges = _edges(a.batch_edges, a.max_seqs), _edges(a.kv_edges, a.max_context)
    if batch_edges[-1] < log.d.max():
        batch_edges.append(int(log.d.max()))
    if kv_edges[-1] < log.kvmax.max():
        kv_edges.append(int(log.kvmax.max()))
    hist, buckets = analyze(log, batch_edges, kv_edges)
    cands, info = pick_regimes(log, buckets, batch_edges, kv_edges, case.family, a.max_regimes)
    ctxs = {}
    if case.family in BATCHED_FAMILIES and cands:   # second deterministic pass to fetch contexts
        ctxs = replay(w, s, st, frozenset(c["step"] for c in cands)).contexts
    regimes, merged = regime_entries(log, cands, case.family, ctxs)
    total = float(log.dur.sum())
    nonscor = 1.0 - info["scorable_time_share"]
    source = {"tool": "megabench.integrations.trace_regimes", "case": case.id, "family": case.family,
              **wsrc, "scheduler": {"max_seqs": a.max_seqs, "max_batched_tokens": a.max_batched_tokens,
                                    "chunked_prefill": a.chunked_prefill, "max_context": a.max_context,
                                    "load_scale": a.load_scale, "max_steps": a.max_steps,
                                    "max_sim_seconds": a.max_sim_seconds},
              "step_time_model": {"step_ms": a.step_ms, "step_timings": a.step_timings},
              "counts": log.counts, "truncated": log.truncated or None, "merged_buckets": merged,
              "edges": {"batch": batch_edges, "kv": kv_edges}, "busy_s": total}
    return {"version": 1, "source": source, "histogram": hist, "regimes": regimes,
            "not_scorable": {"time_share": nonscor,
                             "reason": "prefill/mixed steps" + ("" if case.family in BATCHED_FAMILIES
                                                               else " and decode batches > 1")}}


def print_table(res: dict) -> None:
    src = res["source"]
    print(f"{src.get('trace', src.get('window_file'))} {src.get('window', '')}: {src['counts']}"
          + (f" TRUNCATED by {src['truncated']}" if src["truncated"] else ""))
    print(f"{'bucket':<28}{'steps':>9}{'time%':>8}{'kv p50/p90/p99 (max)':>26}")
    for r in res["histogram"]["buckets"]:
        if r["kind"] == "decode":
            q = r["kv_max_quantiles"]
            print(f"{'decode b%d-%d kv%d-%d' % (*r['batch'], *r['kv']):<28}{r['steps']:>9}"
                  f"{100 * r['time_share']:>7.2f}%{q['p50']:>9.0f}{q['p90']:>8.0f}{q['p99']:>8.0f}")
        else:
            print(f"{r['kind']:<28}{r['steps']:>9}{100 * r['time_share']:>7.2f}%")
    print(f"\nregimes for {src['case']} (weights over scorable buckets; "
          f"not scorable {100 * res['not_scorable']['time_share']:.1f}% of time)")
    for r in res["regimes"]:
        print(f"  {r['id']:<22} weight {r['weight']:.4f}  params {r['params']}")


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        res = run(a)
    except UsageError as e:
        print(f"trace_regimes: {e}", file=sys.stderr)
        raise SystemExit(2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print_table(res)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
