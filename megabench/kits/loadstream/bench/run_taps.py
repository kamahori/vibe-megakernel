"""Tap cost per slot on a resident tile (bench/taps.cu).

    python -m bench.run_taps [--rows 16 --k 1024] [--output taps.jsonl]

(from the kit root). For each (RB, KS) of gemv_ksplit, and for gemv_mma (rb 0), prints ns per call and the implied
per-SM consumption rate in KB/us; the stream delivers ~46-54 KB/us per SM,
so a tap must beat that with margin (it also pays waits and reductions).
Without --rows/--k it sweeps slot shapes typical of decode (32 KB slots).
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch

from loadstream.build import compile_shared

HERE = Path(__file__).resolve().parent
SHAPES = [(16, 1024), (7, 1024), (14, 1024), (8, 2048), (16, 2048), (5, 3072), (4, 4096), (2, 8192), (32, 512)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--output")
    args = ap.parse_args()
    lib = compile_shared([HERE / "taps.cu"], name="taps_bench")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    sink = torch.zeros(sms, device="cuda")
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    out = open(args.output, "x") if args.output else None
    shapes = [(args.rows, args.k)] if args.rows else SHAPES
    for rows, k in shapes:
        for rb in (0, 1, 2, 4, 8):
            for ks in (1, 2, 4, 8):
                if k // 8 // ks < 16:  # fewer than half a vector per lane
                    continue
                if rb == 0 and rows * (2 * k + 16) > 64 * 1024:
                    continue
                run = lambda: lib.bench_tap(rb, ks, rows, k, args.iters, ctypes.c_void_p(sink.data_ptr()), sms, stream)
                assert run() == 0
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                assert run() == 0
                end.record()
                end.synchronize()
                ns = start.elapsed_time(end) * 1e6 / args.iters
                row = {"bench": "tap", "tap": "mma" if rb == 0 else "ksplit", "rows": rows, "k": k, "rb": rb, "ks": ks, "ns_per_slot": round(ns, 1),
                       "kb_per_us": round(rows * k * 2 / ns, 1)}
                print(json.dumps(row), flush=True)
                if out:
                    out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
