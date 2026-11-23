---
name: cdna-sync-and-memory
description: Use when planning, writing or reviewing grid barriers, flags, fences, cross-workgroup data hand-off, weight streaming or cache residency in a persistent megakernel on AMD CDNA (MI300X gfx942, MI210 gfx90a, MI350X gfx950), to get the XCD and coherence rules right and to choose a protocol by measured cost.
---

# CDNA synchronization, coherence and memory

Use this skill with `megakernel-design` (sections 1, 2 and 4) on the rocm backend. It lists
what the hardware does, the design options, and the questions that decide among them. Rows
quoted below are in `./primdb` (they carry `cond.arch`; filter with `arch=gfx942` or
`arch=gfx90a`); the probes that produced them are in `primbench/amd/` and run on your GPU with
`bash primbench/amd/run_amd.sh OUT_DIR`. A row is a measurement in an empty kernel. Your kernel
has skew, other traffic and register pressure, so confirm a choice on the full candidate.

## The machine

| | MI300X (gfx942) | MI210 (gfx90a) | MI350X (gfx950, not measured here) |
|---|---|---|---|
| CUs | 304 = 8 XCDs x 38 | 104, one die | 256 = 8 XCDs x 32 |
| L2 | 4 MB per XCD, **not coherent across XCDs** | 8 MB, one L2 | 4 MB per XCD |
| Infinity Cache (MALL) | 256 MB, memory side, shared | none | 256 MB |
| HBM | 5.3 TB/s peak; 4.2 TB/s measured streaming, 4.9 nontemporal | 1.6 TB/s peak; 1.2-1.4 measured | 8 TB/s peak |
| LDS per CU | 64 KiB | 64 KiB | 160 KiB |
| wavefront | 64 lanes | 64 | 64 |

- Workgroups are dispatched round-robin over XCDs: block `b` runs on XCD `b % 8`
  (`./primdb q xcd.block_map`, `frac_round_robin` = 1.0 on MI300X). Blocks `b` and `b + 8`
  share an L2; blocks `b` and `b + 1` never do.
- A store that one XCD leaves in its L2 is invisible to the other seven until it is written
  back, and a load can hit a stale line that its own XCD's L2 (or its CU's L1) still holds. On
  MI210 there is one L2, so only L1 staleness exists.

## What the compiler emits (read the ISA, do not guess)

On gfx942, at agent (device) scope:

| source | ISA | effect |
|---|---|---|
| `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")`, release atomic | `buffer_wbl2 sc1` + `s_waitcnt` | writes back **every dirty line of this XCD's L2** |
| `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")`, acquire load | `buffer_inv sc1` | invalidates this XCD's L2 and the CU's L1 |
| `__threadfence()` | both of the above | release + acquire |
| relaxed `__hip_atomic_load/store(..., __HIP_MEMORY_SCOPE_AGENT)` | `global_load/store ... sc1` | goes past the non-coherent caches, no fence |
| `volatile` 128-bit load | `flat_load_dwordx4 sc0 sc1` | system scope, slower than agent scope |
| workgroup-scope fences | no cache maintenance | **do not publish data to other XCDs** |

Check a build with `llvm-objdump -d` and count `buffer_wbl2` and `buffer_inv` per barrier. A
fence that the codegen gate shows in a loop or on every thread is a cost you can remove.

## Correctness rules (each one broke an earlier run)

- Put `asm volatile("" ::: "memory")` in every spin loop. Clang deleted a side-effect-free spin
  and hoisted plain loads above a relaxed-atomic poll.
- Every block must be resident: launch at most the number of blocks the occupancy query allows
  (`hipOccupancyMaxActiveBlocksPerMultiprocessor` x CUs), or use `hipLaunchCooperativeKernel`,
  which refuses a grid that does not fit. 2 blocks per CU deadlocked when they were not
  co-resident. In HIP, the second argument of `__launch_bounds__` is the minimum waves per EU
  (SIMD), not blocks per CU.
- A readiness tag must never equal the initial fill of its buffer. Use generation + 1, and a
  64-bit or reset counter (a 32-bit monotonic counter wraps; a counter reused with a different
  grid size hangs).
- Do not run two persistent grids on one GPU at the same time (for example a check and a timeline
  in parallel). Each one can starve the other's blocks.
- Plain loads and stores with relaxed atomics or workgroup-scope fences are not a barrier for
  data. `grid_barrier.relaxed_no_fence` rows show hundreds of thousands of stale cross-XCD reads.
- The stale-read count in the probe detects staleness only while the reader still caches the
  line. On MI300X every read in `relaxed_no_fence` was stale at every dirty size; on MI210 it
  read 0 at 16 KB dirty because the writes evicted L1. Zero stale reads in a test is not a
  proof. Reason from the rules above.

## Option 1: fence-based barrier (plain data, fences in the barrier)

Thread 0 of each block, after `__syncthreads()`: atomic add with **release** (agent scope),
poll with **relaxed** loads, then **one acquire fence**; then `__syncthreads()`.

Measured (`./primdb q grid_barrier. arch=gfx942`):

<!-- MI300X_GBAR_TABLE -->
| protocol (1 block/CU, 256 threads) | MI300X us, 0 KiB dirty | MI300X us, 1 KiB dirty | MI300X us, 16 KiB dirty | MI210 us, 0 KiB dirty | MI210 us, 1 KiB dirty | MI210 us, 16 KiB dirty | stale reads |
|---|---|---|---|---|---|---|---|
| `relaxed_no_fence` | 3.88 | 3.92 | 4.61 | 0.62 | 0.69 | 1.66 | **4702386** |
| `threadfence` | 16.97 | 18.61 | 21.75 | 0.86 | 0.94 | 1.82 | 0 |
| `release_add_acquire_poll` | 13.26 | 14.83 | 20.64 | 0.84 | 0.91 | 1.80 | 0 |
| `release_add_relaxed_poll_acquire_fence` | 12.14 | 12.63 | 14.17 | 0.84 | 0.91 | 1.79 | 0 |
| `release_add_relaxed_poll_sleep_acquire_fence` | 12.08 | 12.68 | 14.43 | 0.87 | 0.93 | 1.83 | 0 |
| `every_thread_threadfence` | 46.77 | 51.62 | 61.26 | 0.91 | 0.98 | 1.85 | 0 |
| `cooperative_groups_grid_sync` | 46.56 | 42.33 | 45.81 | 0.81 | 0.88 | 1.69 | 0 |
| `coherent_data_no_fence` | 5.79 | 5.88 | 6.44 | 0.90 | 0.96 | 1.85 | 0 |
| `per_xcd_two_level_release_acquire` | 8.03 | 8.40 | 10.64 | 1.15 | 1.20 | 2.15 | 0 |

| one fence, all blocks fencing at once | MI300X ns, 0 B dirty | MI300X ns, 256 B dirty | MI300X ns, 4 KiB dirty | MI300X ns, 64 KiB dirty | MI210 ns, 0 B dirty | MI210 ns, 256 B dirty | MI210 ns, 4 KiB dirty | MI210 ns, 64 KiB dirty |
|---|---|---|---|---|---|---|---|---|
| `release.agent` | 2376 | 1765 | 2756 | 1608 | 26 | 23 | 24 | 25 |
| `acquire.agent` | 2375 | 1225 | 1179 | 474 | 90 | 90 | 90 | 93 |
| `seq_cst.agent` | 4829 | 4103 | 4665 | 2384 | 90 | 90 | 91 | 92 |
| `release.workgroup` | 19 | 19 | 19 | 22 | 24 | 24 | 24 | 25 |
<!-- /MI300X_GBAR_TABLE -->

What the MI300X rows say (they differ from MI210, where every correct protocol costs about
0.85 us and the choice barely matters):
- An agent-scope fence by every block at once costs 1-3 us on MI300X (`fence.*.agent` rows),
  against 25-90 ns on MI210. Fences, not the counter, are most of a fenced barrier: 12.1 us
  fenced against 3.9 us for the same counter without fences.
- `cooperative_groups::this_grid().sync()` costs about 45 us at 304 blocks, 4x a hand-written
  fenced barrier. Do not use it in a decode loop on MI300X.
- Coherent data (5.8 us) and per-XCD counters (8.0 us) are the cheapest correct options in the
  rows. Two blocks per CU (608) roughly triples every fenced protocol (33-55 us).

Questions:
- Is the poll an acquire load? Each poll then runs `buffer_inv`; earlier runs lost 2x end to end
  (12.45 -> 5.6 ms in one MoE kernel) from this alone.
- Do all threads fence (`__threadfence()` everywhere)? That is 256x the writebacks.
- How many dirty bytes sit in L2 at each barrier? The release writes back the whole XCD's dirty
  lines. In the barrier rows 16 KiB per block added 2 us (1 block/CU) to 20 us (2 blocks/CU);
  the standalone `fence.release.agent` rows are dominated by 304 blocks fencing at once, not by
  the dirty bytes.

## Option 2: coherent cross-block data, barrier without fences

Write every value that another block reads with relaxed agent-scope atomic stores (or
`global_store ... sc1`) and read it with relaxed agent-scope atomic loads; weights, KV cache and
data a block reads only from itself stay plain. The barrier is `s_waitcnt vmcnt(0)` +
`__syncthreads()` + a relaxed counter and poll (`grid_barrier.coherent_data_no_fence`).

Earlier MI300X runs: barrier 6.9 -> 1.9 us; kernels 3.49 -> 2.87 ms, 4.81 -> 3.26 ms,
8.2 -> 4.9 ms. It also lost once (+0.33 ms) when large phase outputs (gate/up, QKV) were all
written with sc1 stores.

Questions:
- Which buffers cross blocks, and how many bytes per step? sc1 stores write through; a
  high-volume output may cost more than one release per barrier. A hybrid (sc1 for small
  hand-offs, plain stores plus one release for large outputs) is an option.
- Are the coherent loads wide? 16-bit sc1 loads each got their own `s_waitcnt vmcnt(0)`; load 64
  or 128 bits, or use `__builtin_amdgcn_raw_buffer_load_b128(rsrc, off, 0, /*aux sc1*/ 16)`
  with a wave-uniform `rsrc` (a non-uniform one becomes a waterfall loop).
- Does any reader still use a plain load for a cross-block value? Grep the source for each buffer.

## Option 3: barrier topology

- One counter per XCD (`blockIdx.x % 8`, each on its own 128 B line), with the last arrival of
  an XCD bumping a top counter (`grid_barrier.per_xcd_two_level_release_acquire`). Earlier runs:
  7.8 vs 10.95 us flat; 16 or 32 groups that span XCDs were worse. A one-hop variant where 8 lanes
  poll all 8 per-XCD counters beat two levels (6.79 vs 9.42 us).
- All 304 blocks hitting one address serializes in one L2 channel. Tickets and flat 64-bit
  counters lost 1.3-1.4x end to end.
- One flag per 128 B line beat 64 B (~2%); packed flags lost badly.
- `s_sleep 1` in the poll: measured win, neutral and loss in different kernels. Measure it.

## How much of a barrier is protocol

- Flag hand-off between two blocks (`./primdb q flag_handoff.`): MI300X rows give about 0.52 us
  one way within an XCD and 0.58 us across XCDs with relaxed agent atomics, 0.86-0.90 us with
  release/acquire. Earlier runs measured 0.37 us on one XCD, 0.40-0.43 us across, and
  1.7-2.0 us while 302 other blocks streamed HBM.
- In earlier dense runs about 2 of 10.7 us per layer of "sync" was the hand-off and the rest was
  waiting for the slowest block. Read the per-block finish spread in `./timeline_candidate`
  before you change the protocol: skew is fixed by balancing work, not by a faster barrier.

## Weight streaming and residency

Measured streaming reads (`./primdb q mem arch=gfx942`):

<!-- MI300X_MEM_TABLE -->
| arch | load | width B | loads in flight | blocks | threads | level | GB/s |
|---|---|---|---|---|---|---|---|
| MI300X | plain | 4 | 1 | 4 x CUs | 256 | hbm | 2756 |
| MI300X | plain | 4 | 4 | 4 x CUs | 256 | hbm | 4230 |
| MI300X | plain | 8 | 1 | 4 x CUs | 256 | hbm | 4122 |
| MI300X | plain | 8 | 4 | 4 x CUs | 256 | hbm | 4225 |
| MI300X | plain | 16 | 1 | 4 x CUs | 256 | hbm | 4258 |
| MI300X | plain | 16 | 2 | 4 x CUs | 256 | hbm | 4220 |
| MI300X | plain | 16 | 4 | 4 x CUs | 256 | hbm | 4234 |
| MI300X | plain | 16 | 8 | 4 x CUs | 256 | hbm | 4244 |
| MI300X | nontemporal | 16 | 4 | 4 x CUs | 256 | hbm | 4889 |
| MI300X | plain | 16 | 1 | 1 x CUs | 256 | hbm | 2752 |
| MI300X | plain | 16 | 4 | 1 x CUs | 256 | hbm | 4211 |
| MI300X | plain | 16 | 1 | 1 x CUs | 512 | hbm | 4231 |
| MI300X | plain | 16 | 4 | 1 x CUs | 512 | hbm | 4201 |
| MI300X | plain | 16 | 1 | 1 x CUs | 1024 | hbm | 4208 |
| MI300X | plain | 16 | 4 | 1 x CUs | 1024 | hbm | 4064 |
| MI300X | plain | 16 | 4 | 2 x CUs | 512 | hbm | 4238 |
| MI300X | plain | 16 | 4 | 1/8 of CUs | 256 | hbm | 960 |
| MI300X | plain | 16 | 4 | 1/8 of CUs | 1024 | hbm | 2002 |
| MI300X | plain | 16 | 4 | 1/4 of CUs | 256 | hbm | 1867 |
| MI300X | plain | 16 | 4 | 1/4 of CUs | 1024 | hbm | 3477 |
| MI300X | plain | 16 | 4 | 1/2 of CUs | 256 | hbm | 3461 |
| MI300X | plain | 16 | 4 | 1/2 of CUs | 1024 | hbm | 4225 |
| MI300X | plain | 16 | 4 | 1 x CUs | 256 | mall | 5344 |
| MI300X | plain | 16 | 4 | 1 x CUs | 1024 | mall | 4807 |
| MI300X | nontemporal | 16 | 4 | 1 x CUs | 1024 | mall | 5437 |
| MI300X | plain | 16 | 4 | 1 x CUs | 256 | l2 | 9133 |
| MI300X | plain | 16 | 4 | 1 x CUs | 1024 | l2 | 14668 |
| MI300X | nontemporal | 16 | 4 | 1 x CUs | 1024 | l2 | 12538 |
| MI210 | plain | 4 | 1 | 4 x CUs | 256 | hbm | 904 |
| MI210 | plain | 4 | 4 | 4 x CUs | 256 | hbm | 1145 |
| MI210 | plain | 8 | 1 | 4 x CUs | 256 | hbm | 1243 |
| MI210 | plain | 8 | 4 | 4 x CUs | 256 | hbm | 1228 |
| MI210 | plain | 16 | 1 | 4 x CUs | 256 | hbm | 1289 |
| MI210 | plain | 16 | 2 | 4 x CUs | 256 | hbm | 1279 |
| MI210 | plain | 16 | 4 | 4 x CUs | 256 | hbm | 1236 |
| MI210 | plain | 16 | 8 | 4 x CUs | 256 | hbm | 1226 |
| MI210 | nontemporal | 16 | 4 | 4 x CUs | 256 | hbm | 1222 |
| MI210 | plain | 16 | 1 | 1 x CUs | 256 | hbm | 1042 |
| MI210 | plain | 16 | 4 | 1 x CUs | 256 | hbm | 1413 |
| MI210 | plain | 16 | 1 | 1 x CUs | 512 | hbm | 1426 |
| MI210 | plain | 16 | 4 | 1 x CUs | 512 | hbm | 1280 |
| MI210 | plain | 16 | 1 | 1 x CUs | 1024 | hbm | 1375 |
| MI210 | plain | 16 | 4 | 1 x CUs | 1024 | hbm | 1205 |
| MI210 | plain | 16 | 4 | 2 x CUs | 512 | hbm | 1194 |
| MI210 | plain | 16 | 4 | 1/8 of CUs | 256 | hbm | 286 |
| MI210 | plain | 16 | 4 | 1/8 of CUs | 1024 | hbm | 440 |
| MI210 | plain | 16 | 4 | 1/4 of CUs | 256 | hbm | 588 |
| MI210 | plain | 16 | 4 | 1/4 of CUs | 1024 | hbm | 861 |
| MI210 | plain | 16 | 4 | 1/2 of CUs | 256 | hbm | 1115 |
| MI210 | plain | 16 | 4 | 1/2 of CUs | 1024 | hbm | 1160 |
| MI210 | plain | 16 | 4 | 1 x CUs | 256 | mall | 1300 |
| MI210 | plain | 16 | 4 | 1 x CUs | 1024 | mall | 1207 |
| MI210 | nontemporal | 16 | 4 | 1 x CUs | 1024 | mall | 1224 |
| MI210 | plain | 16 | 4 | 1 x CUs | 256 | l2 | 1373 |
| MI210 | plain | 16 | 4 | 1 x CUs | 1024 | l2 | 1378 |
| MI210 | nontemporal | 16 | 4 | 1 x CUs | 1024 | l2 | 1840 |
<!-- /MI300X_MEM_TABLE -->

Questions:
- How many CUs have work in the phase, with how many waves each? On MI300X 1/8 of the CUs reach
  0.96 TB/s at 256 threads and 2.0 TB/s at 1024; 1/2 of the CUs at 1024 threads reach the full
  4.2 TB/s. With one block per CU, 256 threads and one 16 B load in flight reach only 2.75 TB/s;
  4 loads in flight (or 512 threads) reach 4.2.
- Loads in flight: more bytes per wave rarely helped inside a kernel. On CDNA, `vmcnt` retires in
  order per wave, so a large weight load issued first delays a small dependent read (a norm, a
  flag, a partial sum) issued after it. Issue small critical reads first, or give them to a wave
  with nothing in flight.
- Nontemporal loads (`__builtin_nontemporal_load` on an `ext_vector_type` element): 4.9 vs
  4.2 TB/s in the MI300X stream rows; won 2-15%
  for single-use streams in several runs (embedding, logits, expert weights) and lost when the
  weights would have stayed in the Infinity Cache across calls. Decide per buffer by its reuse.
- Does the reused set fit? A whole decode step's weights larger than 256 MB come from HBM every
  step; KV cache, activations and small hot weights can stay in the Infinity Cache or L2.
- Group blocks that share data (for example the K/V of one head) on one XCD: `b, b+8, b+16, ...`.
  This won about 2% once and was neutral elsewhere.
- Blocks idle during a latency-bound phase (attention) can prefetch their next weights into L2.
  This won 3-5% in several runs, and lost when the prefetch competed with the critical phase.

## Record the decision

Write down the protocol, the fences per barrier in the ISA, the measured barrier cost in the
full kernel (null-work build) and the skew. When an option fails, record the dirty bytes, grid and
block size, so a later round knows whether the conditions differ.
