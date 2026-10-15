"""Build, verify, time and profile the mlp_chain worked example.

    python -m examples.mlp_chain.run [--layers 8 --hidden 2048 --inter 8192] [--slots 3 --kc 2048] [--profile]

(run from the kit root, the directory containing ``loadstream/``). Prints
max error vs a PyTorch reference (FP32 math, BF16 matmul inputs as in the
taps), the median time, the HBM floor at
8 TB/s and the fraction of it reached; ``--profile`` rebuilds with
LS_PROFILE and prints the stamp timeline.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import torch

from loadstream.build import compile_shared
from loadstream.timeline import StampBuffer, report

HERE = Path(__file__).resolve().parent
TAGS = {1: "cons:start", 2: "cons:wait_x", 3: "cons:got_x", 4: "cons:norm", 5: "cons:gu_taps",
        6: "cons:gu_done", 7: "cons:wait_h", 8: "cons:got_h", 9: "cons:h_loaded", 10: "cons:dn_taps",
        11: "cons:down_done", 12: "prod:layer", 13: "prod:done", 14: "cons:end"}


class MlpChain:
    def __init__(self, layers: int, hidden: int, inter: int, *, slots: int = 3, kc: int = 2048,
                 profile: bool = False, device: str = "cuda"):
        self.L, self.H, self.I = layers, hidden, inter
        assert hidden % 16 == 0 and inter % 16 == 0 and kc % 16 == 0
        defines = {"LS_WATCHDOG": None, "MLP_SLOTS": slots, "MLP_KC": kc}
        defines |= {"LS_PROFILE": None} if profile else {}
        self.lib = compile_shared([HERE / "mlp_chain.cu"], defines=defines, name="mlp_chain")
        self.lib.mlp_chain_launch.restype = ctypes.c_int
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count
        self.xs = torch.empty(layers, hidden, device=device)
        self.hs = torch.empty(layers, inter, device=device)
        self.ev = torch.zeros(2, layers, dtype=torch.int32, device=device)  # never reset; targets use epoch
        self.epoch = 0
        self.stamps = StampBuffer(self.grid, 12 * layers + 16, device) if profile else None

    def __call__(self, x, norm, wg, wu, wd) -> torch.Tensor:
        self.epoch += 1
        if self.stamps is not None:
            self.stamps.clear()
        p = lambda t: ctypes.c_void_p(t.data_ptr())
        status = self.lib.mlp_chain_launch(
            p(x), p(norm), p(wg), p(wu), p(wd), p(self.xs), p(self.hs), p(self.ev[0]), p(self.ev[1]),
            self.L, self.H, self.I, ctypes.c_uint32(self.epoch),
            ctypes.c_void_p(self.stamps.ptr if self.stamps else 0),
            ctypes.c_int(self.stamps.capacity if self.stamps else 0), self.grid,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if status:
            raise RuntimeError(f"launch failed: cudaError {status}")
        return self.xs[-1]


def reference(x, norm, wg, wu, wd):
    x = x.clone()
    for l in range(norm.shape[0]):
        # Same rounding points as the kernel: BF16 matmul inputs, 1/rms applied
        # after the gate/up products.
        inv = torch.rsqrt(x.square().mean() + 1e-6)
        n = (x * norm[l].float()).bfloat16().float()
        h = (torch.nn.functional.silu((wg[l].float() @ n) * inv) * ((wu[l].float() @ n) * inv)).bfloat16().float()
        x = x + wd[l].float() @ h
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--kc", type=int, default=2048, help="columns per K-chunk slot (slot = 16 x (2 kc + 16) B)")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    L, H, I = args.layers, args.hidden, args.inter
    dev = "cuda"
    x = torch.randn(H, device=dev)
    norm = (1 + 0.1 * torch.randn(L, H, device=dev)).bfloat16()
    wg = (torch.randn(L, I, H, device=dev) / H ** 0.5).bfloat16()
    wu = (torch.randn(L, I, H, device=dev) / H ** 0.5).bfloat16()
    wd = (torch.randn(L, H, I, device=dev) / I ** 0.5).bfloat16()
    model = MlpChain(L, H, I, slots=args.slots, kc=args.kc, profile=args.profile)
    out = model(x, norm, wg, wu, wd).clone()
    ref = reference(x, norm, wg, wu, wd)
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    again = model(x, norm, wg, wu, wd)
    print(f"max rel error {err:.3e}; deterministic {torch.equal(out, again)}")
    # BF16 intermediates: a different FP32 summation order flips the rounding
    # of single h elements (2^-8 relative), so expect errors of ~1e-2.
    if err > 2e-2:
        raise SystemExit("mismatch")

    weight_bytes = (wg.numel() + wu.numel() + wd.numel()) * 2
    floor_ms = weight_bytes / 8e12 * 1e3
    if args.profile:
        model(x, norm, wg, wu, wd)
        torch.cuda.synchronize()
        print(report(model.stamps.read(), TAGS, bytes_total=weight_bytes))
        return
    for _ in range(5):
        model(x, norm, wg, wu, wd)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.reps):
        start.record()
        model(x, norm, wg, wu, wd)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    t = sorted(times)[len(times) // 2]
    print(f"L={L} H={H} I={I} ring {args.slots}x16x{args.kc}: {weight_bytes / 1e6:.1f} MB, median {t * 1e3:.1f} us, floor {floor_ms * 1e3:.1f} us "
          f"-> {100 * floor_ms / t:.1f}% of 8 TB/s SOL ({weight_bytes / t / 1e9:.2f} TB/s)")


if __name__ == "__main__":
    main()
