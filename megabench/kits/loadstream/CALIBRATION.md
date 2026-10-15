# B200 calibration (measured)

`python -m bench.run_micro` and `python -m bench.run_taps` on one NVIDIA B200 (148 SMs, CUDA 13.1), shared
host, 2026-10-14. Use these numbers, not datasheet ones, in your ledger.
Re-run the benchmark if a number matters to a decision.

## Streaming ceiling

A 2 GB buffer is banded over the CTAs; each CTA streams its band through a
`WeightRing` in whole-slot `tile_contiguous` copies (1 producer warp, 4
consumer warps that only touch one word per slot), so this is the best a
ring of that shape can do.

| ring (slots × slot) | in flight / SM | 148 CTAs | 296 CTAs |
|---|---|---|---|
| 2 × 64 KB | 128 KB | **7.07 TB/s (88%)** | |
| 3 × 64 KB | 192 KB | 6.97 TB/s (87%) | |
| 2 × 32 KB | 64 KB | 6.72 TB/s (84%) | 6.86 TB/s (86%) |
| 4 × 32 KB | 128 KB | 6.72 TB/s (84%) | |
| 6 × 32 KB | 192 KB | 6.48 TB/s (81%) | |
| 4 × 16 KB | 64 KB | 5.50 TB/s (69%) | 6.40 TB/s (80%) |
| 12 × 16 KB | 192 KB | 5.55 TB/s (69%) | |
| 8 × 8 KB | 64 KB | 2.80 TB/s (35%) | 5.48 TB/s (68%) |
| 16 × 4 KB | 64 KB | 4.87 TB/s (61%) | |

- **Slot size matters more than depth.** With one CTA per SM, use slots of
  ≥ 32 KB (64 KB best); 8–16 KB slots lose 30–65% no matter how many are in
  flight. Copies of ≥ 32 KB per `begin()` are the goal; many small
  `copy_rows` pieces inside one slot are fine as long as the slot is big.
- Plain vectorized loads (`ld.global.nc.L1::no_allocate.v4`, 16 B/lane, no
  ring): 6.88 / 6.93 / 6.97 TB/s at 148 / 296 / 592 CTAs (86–87%).
- `torch` device copy: 6.47 TB/s counting read + write.
- **Plan with ~6.8 TB/s** (`Ledger(tb_s=6.8)`): 85% of the 8 TB/s SOL is the
  practical ceiling, so 50% of SOL means ~59% of the achievable stream.

## Synchronization

| primitive | cost |
|---|---|
| event hop (`signal` → `wait_geq` in another CTA), one way | 0.73 µs (0.92 µs with a `__threadfence` before the release) |
| grid barrier (148 CTAs, 32–512 threads) | 1.23–1.24 µs |

`red.release.gpu` after a CTA/warp barrier is cumulative: it publishes every
write the barrier ordered before it, so `ls::signal` needs no
`__threadfence` (that fence alone cost ~0.2 µs per hop).

An event hop is not much cheaper than a barrier; events win because the
producer keeps streaming behind them and a reader waits for a slice, not for
the whole grid.

**Do not poll data flags with every thread** (LL-style flag-in-data
exchange): tried for the h/x hand-offs in `mlp_chain`, it hammered L2 and
was ~2× slower than one thread polling a counter.

## Tensor cores

`mma.sync.m16n8k16` BF16 → FP32 from registers, all SMs:

| warps / SM | TFLOPS |
|---|---|
| 4 | 522 |
| 8 | 549 |
| 16 | 553 |
| 32 | 555 |

That is ~24% of the 2250 TFLOPS dense BF16 peak used by `megabench sol`.
mma.sync is enough wherever the case is HBM-bound (decode, verify, pi0.5
steps: tens of tokens). A case whose floor is tensor-core time
(Waypoint, 512 tokens per pass) **cannot** reach 50% of SOL with mma.sync;
it needs `tcgen05.mma` (UMMA, operands in smem descriptors, accumulator in
tensor memory), e.g. via CUTLASS/CuTe sm100 primitives.

## Shared-memory bandwidth (derived)

Streaming at 54 KB/µs per SM, the ring is written once by TMA and read once
by the taps (~108 KB/µs). An FP32 activation read per BF16 weight element
adds 2× the weight bytes again, more for FP8/INT4 (4×/8×), against roughly
240 KB/µs of smem bandwidth per SM. Reuse each loaded x vector for several
rows (`lane_dot_rows<Fmt, RB>`, `gemv_*<..., RB>`), or use `mma_rows16`
(x lives in registers as B fragments) when the weights are BF16.

## Tap throughput (`python -m bench.run_taps`)

One CTA per SM consumes a resident 16-row slot repeatedly (8 consumer
warps, no streaming), so this is the per-slot compute cost of a tap.
Compare with the stream: one SM receives **~46–54 KB/µs**; a tap must clearly
beat that or it, not HBM, sets the pace.

| slot (rows × k, BF16) | best CUDA-core tap (`gemv_ksplit`, RB/KS) | tensor-core `gemv_mma` (KS 8) |
|---|---|---|
| 16 × 1024 (32 KB) | 491 ns, 67 KB/µs (4/2) | **340 ns, 96 KB/µs** |
| 16 × 2048 (64 KB) | 741 ns, 88 KB/µs (4/2) | |
| 14 × 1024 | 489 ns, 59 KB/µs | 342 ns, 84 KB/µs |
| 8 × 2048 | 466 ns, 70 KB/µs (8/8) | 537 ns, 61 KB/µs |
| 4 × 4096 | 487 ns, 67 KB/µs (4/8) | 924 ns, 36 KB/µs |
| 2 × 8192 | 556 ns, 59 KB/µs (2/8) | 1753 ns, 19 KB/µs |
| 32 × 512 | 490 ns, 67 KB/µs (4/1) | 350 ns, 94 KB/µs (KS 4) |

- Taps are only 1.2–1.9× faster than the stream, and each call costs
  ~340–500 ns almost regardless of size (latency bound: few warps, long
  dependent chains). Give each tap call **≥ 32 KB** of work.
- `gemv_mma` (x as the single B column of `mma.m16n8k16`) wants slots of
  **16 rows × a K-chunk** with pitch `2*KC + 16` (conflict-free `ldmatrix`);
  for fewer than ~8 rows of long K use `gemv_ksplit`.
- Going from 4 to 8 independent MMA accumulator chains did not change the
  340 ns, so the limit is not the MMA dependency chain.

## Worked example: `python -m examples.mlp_chain.run`

L residual MLP blocks, BF16, `gemv_mma` taps, one event per op, no grid
barriers. All runs pass the PyTorch check. % of the 8 TB/s SOL:

| ring (slots × 16 rows × KC) | L8 H2048 I8192 (805 MB) | L28 H1024 I3072 (529 MB, Qwen3-0.6B MLPs) |
|---|---|---|
| 3 × 16 × 2048 (default, 64 KB) | 57.2% | **27.3%** |
| 6 × 16 × 1024 (32 KB) | **57.4%** | 25.5% |
| 5 × 16 × 1024 | 52.8% | 25.7% |
| 4 × 16 × 1024 | 49.7% | 26.1% |
| 6 × 16 × 512 (16 KB) | 31.7% | 23.1% |
| CUDA-core `gemv_ksplit`, 5 × 32 KB | 54.9% | 20.1% |

The small geometry is the warning: per layer (2.4 µs of bytes at 8 TB/s,
~8.6 µs measured) the timeline is

| interval (per layer) | µs |
|---|---|
| got x → norm done (load x, RMS) | 0.85 |
| norm → first gate/up tap | 1.9 |
| gate/up taps | 1.6 |
| wait h (event hop + skew) | 1.4 |
| load h → first down tap | 1.5 |
| down taps | 1.4 |
| wait x (event hop + skew) | 0.9 |

Bandwidth is not the limit there: two dependent hand-offs per layer, each
costing a hop plus a re-stage of the activation plus a tap ramp, are. A
dense decoder has ~5 such points per layer, so the design must minimize
**dependency points on the critical path** (fuse ops, recompute cheap
things per CTA, overlap the activation load with the previous op's tail),
not only keep the ring full.
