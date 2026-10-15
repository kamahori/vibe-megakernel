---
name: load-centric-megakernel
description: Design, build and tune a single-launch decode megakernel as a byte stream per SM (weights run ahead through a TMA ring; activations gate only the compute) using the loadstream kit. Use for any MegaBench case scored against an HBM speed-of-light floor, before writing the first kernel line and whenever the kernel is far (>2x) from `./sol_info`.
---

# Load-centric megakernels

Goal: reach **at least 50% of the `./sol_info` floor** with a correct,
single-launch kernel. Low-batch decode is set by **HBM bytes**, not FLOPs. A
megakernel loses time in three places, and this skill is about removing them:

1. **Bubbles**: weights for op N+1 are requested only after op N finishes, though
   weights never depend on the input. Every grid barrier (≈1.23 µs on B200) with
   nothing streaming behind it costs ~70 KB of lost streaming per SM.
2. **Too few bytes in flight**: B200 needs ≈ 8 TB/s × 0.8 µs ≈ 6 MB in flight,
   i.e. **≥ 40 KB per SM at all times** (Little's law).
3. **Imbalance**: an op split over 32 heads uses 32 of 148 SMs; a matrix split by
   tiles instead of bytes leaves SMs idle at the tail.

The fix is a **byte stream per SM with compute attached**:

- **Stream**: each SM owns a static, byte-balanced band of every weight matrix
  (and KV block). One **producer warp** issues `cp.async.bulk` copies of those
  bands, in program order, into a shared-memory **ring** of a few **big** slots
  (≥ 32 KB each; 2–4 slots, 128–192 KB total; slots of 8–16 KB lose 30–65%).
  It waits only for free slots, never for activations, so it runs ahead across
  op and layer boundaries.
- **Taps**: consumer warps drain the ring in the same order and compute
  (GEMV, dequant-GEMV, small-M MMA, norms). Only taps wait for activations.
- **Events**: monotonically increasing ready counters per activation replace
  grid barriers. A tap waits for exactly the rows it needs.
- **Ledger**: every phase has a byte count, so expected time per phase is
  known before measuring.

## The kit (`loadstream_kit/` in the workspace)

| File | What it gives you |
|---|---|
| `include/loadstream/ring.cuh` | `WeightRing`, `Producer` (`begin/copy/copy_rows/commit`, `tile_contiguous`), `Consumer` (`wait/release`) |
| `include/loadstream/event.cuh` | `wait_geq`, `warp_wait_geq`, `warp_signal`, `signal`, `grid_barrier`, watchdog (`-DLS_WATCHDOG`) |
| `include/loadstream/tap.cuh` | GEMV taps for BF16/FP8E4M3/I8/U4Off8 weights, all deterministic and sync-free: `gemv_rows` (rows over warps), `gemv_ksplit` + `reduce_ksplit` (K over warps, partials accumulated over a whole band, one sync per op), `lane_dot_rows<Fmt, RB>` (RB rows share each x load); `gemv_mma_acc` + `gemv_mma_store` (batch-1 BF16 GEMV on tensor cores, x as the B column, over 16-row × K-chunk slots: the fastest tap for ≥ 14 rows); `gemv_tile` (self-contained tile, 2 syncs: tests only); `mma_rows16<NT>` small-M BF16 tensor-core tap over K-chunks; `warp_sum/max/argmax` |
| `include/loadstream/stamp.cuh` | `ls::stamp(stamps, tag, value)`: globaltimer records, compiled out unless `-DLS_PROFILE` |
| `include/loadstream/plan.cuh` | `band`, `band_rows`, `rows_per_slot` |
| `loadstream/build.py` | `compile_shared([...cu], defines=...)`: nvcc → cached `.so` via ctypes, with the kit include path |
| `loadstream/plan.py` | `Ledger`: bytes per phase, band imbalance, sync overhead, Little's-law ring size, *before* coding |
| `loadstream/timeline.py` | `StampBuffer`, `report`, `chrome_trace`: where the time went, per interval, per CTA |
| `examples/mlp_chain/` | **Start here.** A complete, verified kernel: L residual MLP blocks, producer warp + 8 consumer warps, events instead of barriers, stamps, PyTorch check, SOL report |
| `bench/` | B200 calibration microbenchmarks: `run_micro` (streaming ceiling, event latency, barrier cost, mma.sync rate), `run_taps` (per-slot cost of each tap) |
| `CALIBRATION.md` | **Measured** B200 numbers (ring shapes, event/barrier cost, mma.sync rate, the example's % of SOL); use them in your ledger |

Copy the kit headers into your build (or point `-I` at `loadstream_kit/include`),
copy `examples/mlp_chain/mlp_chain.cu` as your skeleton, and grow it op by op.
The grader imports only `submission.py` and the files it loads, so make
`build()` compile from paths relative to `submission.py`. The Python helpers
import as `sys.path.insert(0, str(Path(__file__).parent / "loadstream_kit"))`
then `from loadstream.build import compile_shared`. Run the kit's own checks
once in a GPU job (`cd loadstream_kit && python -m tests.run_tests && python -m
examples.mlp_chain.run --profile`) to see a passing kernel and a timeline.

## Procedure

1. **Ledger first.** Read `./sol_info --json`. With `loadstream.plan.Ledger`,
   list every matrix (rows, row bytes), KV reads, the LM head, and every
   dependency point on the critical path. Use the measured streaming ceiling
   from `CALIBRATION.md` (~6.8 TB/s, not 8) for the time estimate. Write the
   per-phase budget into your notes: e.g. "Qwen3-0.6B layer: 39 MB → 5.6 µs;
   lm_head 311 MB → 44 µs".
2. **Skeleton.** One cooperative persistent kernel (`cudaLaunchCooperativeKernel`),
   grid = number of SMs, 1 producer warp + 8 consumer warps, ring of big slots
   as in `mlp_chain` (see `CALIBRATION.md` for the measured shapes), all
   weights row-banded over **all** SMs with `ls::band`. Fuse matrices that
   share an input into one tile sequence (Q|K|V, gate|up), with a band's gate
   rows and up rows in the same slot.
   **Taps must never sync per slot.** Per op: wait for the input event once,
   stage x in smem, run the tap on every slot of the band with partials in
   smem (BF16: `gemv_mma_acc` over 16-row × KC slots, accumulators kept
   across K-chunks; quantized or < 8 rows: `gemv_ksplit`), then ONE named
   barrier, the reduction (fold in silu·up / residual / norm statistics), ONE
   named barrier and ONE `signal`. `mlp_chain` does exactly this; an earlier
   version with per-slot syncs and per-slot signals was several times slower.
3. **Dependencies as events.** One counter per (layer, activation). A writer adds
   the number of rows it wrote (`signal` after a named barrier, or
   `warp_signal`); a reader waits for `epoch * rows_per_call`. Pass `epoch` (the
   1-based call index) as a kernel argument from the host; never reset counters.
   Recompute cheap things redundantly per CTA (RMSNorm of a 1–4 K vector, RoPE
   of one token, the MoE router top-k) instead of adding an event.
4. **Correct first, at full geometry**, with `-DLS_WATCHDOG` on. Compare every
   intermediate against the reference on a fixed seed before tuning. Keep the
   graded build free of `LS_PROFILE`.
5. **Measure with stamps, not guesses.** Stamp before/after each wait and at the end
   of each op (consumer warp 0 lane 0; producer lane 0), name tags `role:name`,
   run `timeline.report`. Read it as follows:
   - big `wait_x -> got_x`: **bubble**. Deepen the ring, move the producer further
     ahead, split the event finer (wait for the rows you need, not the whole
     vector), or shorten the producer of that activation.
   - op interval ≫ its ledger time while waits are small: **under-streaming or compute
     bound**. Check bytes in flight, slot size, consumer instruction count (dequant!),
     smem bank conflicts.
   - large CTA end skew: **imbalance**. Re-band by bytes; give the tail ops (LM
     head, attention) to all SMs.
   **Too few rows for all SMs** (e.g. 1024 output rows in 16-row MMA groups = 64
   groups for 148 SMs, or a 4096-wide down projection at batch 50): split K across
   CTAs instead of leaving SMs idle. CTA (g, s) streams rows of group g, K-slice s,
   writes its FP32 partial to `part[g][s]`, and `signal`s `cnt[g]`. Whoever consumes
   rows of group g waits for `cnt[g]` to reach `epoch * slices` and sums the slices
   in fixed `s` order (deterministic, no float atomics). Fold the residual add,
   norm statistics or activation into that reduction.
6. **Iterate on the largest gap only**, re-checking correctness each time. Stop when
   ≥ 50% of SOL is reached and stable across 3 trials, then polish.

## What earlier agent sessions got wrong (audit of ~150 rounds on these cases)

None of them lost rounds to hangs or wrong results. They lost by building the
pieces separately on a **barrier-centric skeleton**: each grid barrier (1.2–1.5 µs
release + ~0.5 µs skew + a ramp afterwards) left HBM idle 40–60% of every layer.
What did and did not work:

- **Lost**: `cp.async.bulk.prefetch.L2` of the next op's weights (raises L2 hit rate,
  contends with latency-critical loads, no speedup); register prefetch across a
  barrier (spills at the register cap); bulk copies left in flight across
  `grid.sync` (the barrier's fence waits for them); custom/cluster grid barriers
  (no faster than cooperative groups); swapping one barrier for one "handoff"
  flag that still waits for every producer (same cost, skew returns).
- **Won**: a steady-depth shared-memory TMA ring that is **decoupled from all
  synchronization** (only the producer warp touches it), and dataflow counters
  that let a tap start when *its* inputs are ready. No session applied the ring to
  every op of the step; do that from the first version.
- An event hop costs about as much as a barrier hop (~1 µs). Events win because the
  stream keeps flowing behind them and because they wait for a slice, not because
  a hop is cheaper. Never let the producer stop.
- Don't over-constrain: grading uses the reference's tolerances (and an oracle band
  for logits), so bit-exactness with the PyTorch op order is not required. Prefer
  determinism (fixed reduction order) over exactness.

## Rules that prevent lost sessions

- Every consumer warp must `wait()` and `release()` **every** slot, in the same
  order the producer filled them, even slots it does not compute on. The producer
  and consumers must iterate the identical tile sequence: write one schedule and
  derive both loops from it (as `mlp_chain` does with shared `band`/`rows_per_slot`).
- `begin(total)` must announce exactly the bytes the following copies deliver.
  Copies need 16-byte aligned source/destination and sizes that are multiples of 16.
- Data written by other CTAs during this launch: read with `ld_cg` / `__ldcg` or
  `ld.acquire`, **never** `__ldg`, `const __restrict__`, or `ld.global.nc`.
  Weights and inputs are read-only and may use any path.
- Publish after writing: writers → `__syncwarp`/named barrier → one thread
  `red.release.gpu.add` (`ls::signal`; release is cumulative over the barrier,
  so no `__threadfence`, which costs ~0.2 µs per hop). Wait with one thread
  polling `ld.acquire` (`ls::wait_geq`), then a warp/named barrier. Never
  have every thread poll flags inside the data (LL style): measured 2× slower.
- All CTAs must be co-resident: grid ≤ SMs × occupancy, cooperative launch, and
  the smem opt-in (`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)`).
- Use named barriers (`ls::named_sync(id, NW*32)`) among consumer warps; never
  `__syncthreads()` after the roles split (the producer is elsewhere).
- Keep results deterministic: fixed work assignment and reduction order
  (all kit taps do this). Float atomics for split-K make outputs vary run to run.
- A hang is a bug with a location: rebuild with `-DLS_WATCHDOG`; the trap message
  names the CTA, the event id, its value and its target.
- No hidden extra launches: no `torch.zeros`, `.fill_`, `.copy_`, or cuBLAS calls in
  `run`. Allocate outputs with `torch.empty`; counters persist across calls.

## Case playbook

**Dense decode (Qwen3-0.6B, Llama-style).** Per layer: rmsnorm → QKV (fused) →
q/k norm + RoPE + KV append + attention → O → rmsnorm → gate|up → down.
Attention over 128 cached positions is tiny (KV ≈ 0.5 MB per layer): give one
(or a few) KV head(s) per CTA, but **all** CTAs keep streaming the next weights
meanwhile. The LM head (151936 × 1024 BF16 = 311 MB, a quarter of the bytes) must
be banded over all SMs; fuse the argmax into its tap (per-CTA best (value, index),
then the last CTA to arrive on a counter picks the winner, ties → lowest index).

**MoE decode (Qwen3-30B-A3B).** Expert addresses are known only after routing,
the one place the producer must wait. Compute the router logits/top-k redundantly
in every CTA (2048 × 128 router: 0.5 MB from L2) as soon as the post-attention
hidden is ready, so every producer learns the experts with one event wait. Band
the 8 selected experts' rows over all SMs by bytes. While waiting for routing, the
producer can issue `ls::prefetch_l2` for the next layer's attention weights.

**Quantized (Gemma-3 W8/W4).** Bytes shrink 2–4×, so the taps must dequantize at
4–8 bf16-equivalent elements per byte without becoming ALU bound: keep x in FP32
smem, use `ls::I8` / `ls::U4Off8`, apply the per-row scale once per row, and use
all 8 consumer warps. If timeline shows tap time > stream time, vectorize the
nibble extraction (e.g. `__byte_perm`, `lop3`) and move to packed `__hfma2`.

**Speculative verify (k+1 = 5 tokens).** Weights are read once for 5 tokens: stage
the 5 activation rows in smem as BF16 and use `mma_rows16<1>` (tokens padded to 8)
over 16-row weight groups. A 16-row group of a 4096-wide matrix is 128 KB, so
slots hold a **K-chunk** of a block of 16-row groups (one group per warp);
each warp keeps its accumulators across the chunks (`tests/taps.cu`
`mma_test` is the pattern: `copy_rows` with `dst_pitch = 2*KC + 16`, pass
`x + 2*k0` to `mma_rows16`).

**CSM-1B (backbone + 31 dependent depth passes).** The depth decoder (~226 MB)
is re-streamed by each of 31 passes; the floor counts this. Each pass is short
(~30 µs), so per-pass dependency points dominate: 4 layers × ~5 events. Keep the
producer streaming the next pass's weights during argmax/embedding, fuse
positions 0 and 1 into one 2-token pass, and consider keeping part of the depth
weights L2-resident across passes (`policy_evict_last` on those copies,
`evict_first` on the rest; B200 L2 ≈ 126 MB).

**pi0.5 (10 dependent Euler steps, 50 action tokens).** All 10 time conditions
are known up front: compute every step's adaRMS modulation in one pass at the
start (reads the 466 MB of FP32 conditioning weights once). Each step streams the
~0.62 GB of expert weights once; 50 tokens need tensor cores: `mma_rows16<7>`
(56 token columns) with activations staged in smem as BF16 per K-chunk.

**Waypoint (512 tokens per pass, tensor-core bound).** Not load-bound: the floor
is BF16 tensor-core throughput. Use the ring only for operand staging; the work
is GEMM tiling with high MMA utilisation. mma.sync tops out at ~550 TFLOPS on
B200 (24% of the 2250 TFLOPS the floor assumes), so 50% of SOL **requires
`tcgen05.mma` (UMMA)** with smem descriptors and tensor-memory accumulators,
e.g. via CUTLASS/CuTe sm100 building blocks inside the persistent kernel.
Measure your MMA rate in isolation before integrating.

## Numbers to keep in mind (B200, 148 SMs; see CALIBRATION.md for measured values)

- 8 TB/s ⇒ 54 KB/µs per SM. A 32 KB slot drains in ~0.6 µs. Achievable
  streaming ≈ 6.8–7.0 TB/s (85–88%).
- ≥ 64 KB in flight per SM in slots of ≥ 32 KB; 128 KB in flight hides a ~2 µs stall.
- Grid barrier 1.23 µs; an event hop 0.73 µs one way, plus ~0.5–1 µs of
  re-staging the activation and ramping the tap after it. Count dependency
  points per layer × ~2 µs: for Qwen3-0.6B (28 layers, ~5 points) that alone
  is ~280 µs against a ~150 µs floor. **Remove dependency points** (fuse ops,
  recompute per CTA, split events finer, overlap) before tuning anything else.
- A tap call costs ≥ ~340 ns even on a resident slot; the best taps process
  only 1.2–1.9× the per-SM stream rate (`CALIBRATION.md`, tap table). Taps,
  not HBM, are usually what keeps a kernel under 50%.
- Shared memory bandwidth is the hidden limit of CUDA-core taps: reading an
  FP32 x element per weight element costs 2× (BF16) to 8× (INT4) the weight
  bytes again. Reuse x across rows (`RB`) or use MMA.
- CUDA-core BF16 GEMV keeps up with HBM up to ~4–8 tokens; beyond that use MMA.
- Shared memory: 227 KB per CTA max; leave room for activations (x in FP32 = 4 B × hidden).
