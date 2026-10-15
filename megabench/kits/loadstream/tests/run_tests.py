"""GPU unit tests for the ring and taps: python -m tests.run_tests (from the kit root)."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch

from loadstream.build import compile_shared

HERE = Path(__file__).resolve().parent
P = lambda t: ctypes.c_void_p(t.data_ptr() if t is not None else 0)


def main() -> int:
    lib = compile_shared([HERE / "taps.cu"], defines={"LS_WATCHDOG": None}, name="taps")
    torch.manual_seed(0)
    dev = "cuda"
    failures = 0

    def check(name: str, got: torch.Tensor, want: torch.Tensor, tol: float = 1e-4) -> None:
        nonlocal failures
        err = ((got - want).abs().max() / want.abs().max().clamp_min(1e-6)).item()
        ok = err < tol
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: rel err {err:.2e}")

    for rows, k, grid in ((1000, 2048, 148), (77, 4096, 13), (512, 8192, 148)):
        x = torch.randn(k, device=dev)
        wf = torch.randn(rows, k, device=dev)
        # BF16
        w = wf.bfloat16()
        for pad in (0, 16):
            row_bytes = 2 * k
            tile = max(1, (32 * 1024) // (row_bytes + pad))
            y = torch.empty(rows, device=dev)
            assert lib.test_gemv(0, P(w), rows, k, row_bytes, P(x), P(None), P(y), tile, pad, grid) == 0
            torch.cuda.synchronize()
            check(f"gemv bf16 {rows}x{k} pad{pad}", y, w.float() @ x)
        # FP8 e4m3
        w8 = (wf * 4).to(torch.float8_e4m3fn)
        y = torch.empty(rows, device=dev)
        tile = max(1, (32 * 1024) // k)
        assert lib.test_gemv(1, P(w8), rows, k, k, P(x), P(None), P(y), tile, 0, grid) == 0
        torch.cuda.synchronize()
        check(f"gemv fp8 {rows}x{k}", y, w8.float() @ x)
        # int8 with per-row scale
        q = torch.randint(-127, 128, (rows, k), device=dev, dtype=torch.int8)
        scale = torch.rand(rows, device=dev) * 0.01
        assert lib.test_gemv(2, P(q), rows, k, k, P(x), P(scale), P(y), tile, 0, grid) == 0
        torch.cuda.synchronize()
        check(f"gemv int8 {rows}x{k}", y, (q.float() @ x) * scale)
        # int4, gemma packing: low nibble = even column, value = nibble - 8
        q4 = torch.randint(-7, 8, (rows, k), device=dev)
        packed = ((q4[:, 0::2] + 8) | ((q4[:, 1::2] + 8) << 4)).to(torch.uint8)
        tile = max(1, (32 * 1024) // (k // 2))
        assert lib.test_gemv(3, P(packed), rows, k, k // 2, P(x), P(scale), P(y), tile, 0, grid) == 0
        torch.cuda.synchronize()
        check(f"gemv int4 {rows}x{k}", y, (q4.float() @ x) * scale)

    for rows, k, grid in ((1000, 1024, 148), (77, 4096, 13), (3072, 2048, 148), (500, 8192, 148)):
        w = torch.randn(rows, k, device=dev).bfloat16()
        x = torch.randn(k, device=dev).bfloat16()
        for ks in (1, 2, 4):
            y = torch.empty(rows, device=dev)
            assert lib.test_gemv_mma(P(w), rows, k, P(x), P(y), ks, grid) == 0
            torch.cuda.synchronize()
            check(f"gemv_mma {rows}x{k} ks{ks}", y, w.float() @ x.float())

    for rows, k, tokens, grid in ((1024, 1024, 5, 148), (4096, 1024, 50, 148), (512, 4096, 16, 7)):
        w = torch.randn(rows, k, device=dev).bfloat16()
        x = torch.randn(tokens, k, device=dev).bfloat16()
        for pad in (16,):
            y = torch.empty(tokens, rows, device=dev)
            status = lib.test_mma(P(w), rows, k, P(x), tokens, P(y), pad, grid)
            assert status == 0, status
            torch.cuda.synchronize()
            check(f"mma {tokens}x{rows}x{k} pad{pad}", y, x.float() @ w.float().T, 1e-3)
    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
