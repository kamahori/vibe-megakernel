"""Phase-1 calibration: streaming ceiling, event and barrier latency, mma.sync rate.

    python -m bench.run_micro [--gb 2] [--output results.jsonl]

(from the kit root). Prints one JSON object per measurement.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch

from loadstream.build import compile_shared

HERE = Path(__file__).resolve().parent


def timed(fn, reps: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gb", type=float, default=2.0)
    ap.add_argument("--output")
    args = ap.parse_args()
    lib = compile_shared([HERE / "micro.cu"], name="micro")
    dev = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    nbytes = int(args.gb * 2 ** 30) // (64 * 1024 * sms) * (64 * 1024 * sms)
    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    buf.random_(0, 255)
    sink = torch.zeros(4, device="cuda")
    out = open(args.output, "x") if args.output else None

    def emit(row: dict) -> None:
        row["device"] = torch.cuda.get_device_name(dev)
        print(json.dumps(row), flush=True)
        if out:
            out.write(json.dumps(row) + "\n")

    P = lambda t: ctypes.c_void_p(t.data_ptr())
    for slots, kb in [(2, 16), (4, 16), (8, 16), (12, 16), (2, 32), (4, 32), (6, 32), (2, 64), (3, 64),
                      (4, 8), (8, 8), (16, 8), (8, 4), (16, 4), (32, 4)]:
        for grid in (sms, 2 * sms) if (slots * kb <= 96) else (sms,):
            def run():
                assert lib.micro_stream_ring(slots, kb, P(buf), ctypes.c_size_t(nbytes), P(sink), grid, stream()) == 0
            ms = timed(run)
            emit({"bench": "stream_ring", "slots": slots, "slot_kb": kb, "in_flight_kb": slots * kb,
                  "ctas": grid, "tb_s": nbytes / ms / 1e9, "pct_of_8tbs": nbytes / ms / 1e9 / 8 * 100})
    for grid in (sms, 2 * sms, 4 * sms):
        ms = timed(lambda: lib.micro_stream_ldg(P(buf), ctypes.c_size_t(nbytes), P(sink), grid, stream()))
        emit({"bench": "stream_ldg", "ctas": grid, "tb_s": nbytes / ms / 1e9, "pct_of_8tbs": nbytes / ms / 1e9 / 8 * 100})
    ms = timed(lambda: buf[: nbytes // 2].copy_(buf[nbytes // 2:]))
    emit({"bench": "torch_copy", "tb_s": nbytes / ms / 1e9, "note": "read+write bytes"})

    iters = 2000
    ns = torch.zeros(1, dtype=torch.int64, device="cuda")
    ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
    assert lib.micro_event_pingpong(P(ctr), iters, P(ns), sms, stream()) == 0
    torch.cuda.synchronize()
    emit({"bench": "event_pingpong", "one_way_us": ns.item() / iters / 2 / 1e3})
    for threads in (32, 288, 512):
        ctr = torch.zeros(1, dtype=torch.int32, device="cuda")
        assert lib.micro_grid_barriers(P(ctr), iters, P(ns), sms, threads, stream()) == 0
        torch.cuda.synchronize()
        emit({"bench": "grid_barrier", "threads": threads, "us_per_barrier": ns.item() / iters / 1e3})

    for blocks_per_sm, threads in ((1, 128), (1, 256), (2, 256), (4, 256)):
        it = 20000
        ms = timed(lambda: lib.micro_mma_rate(it, P(sink), sms * blocks_per_sm, threads, stream()), reps=5)
        flops = sms * blocks_per_sm * (threads // 32) * it * 4 * 2 * 16 * 8 * 16
        emit({"bench": "mma_sync_bf16", "warps_per_sm": blocks_per_sm * threads // 32, "tflops": flops / ms / 1e9})
    if out:
        out.close()


if __name__ == "__main__":
    main()
