---
name: blackwell-hardware-features
description: Use when a decode megakernel or memory-bound kernel on B200 (sm_100a) is limited by grid-wide sync latency, by streaming bandwidth or by short dependent phases, to put the relevant hardware features (acquire polls, clusters, 256-bit loads, L2 eviction hints, bulk L2 prefetch, TMA depth, nanosleep) into the candidate set and to decide among them with the measured costs below.
---

# Blackwell hardware features for b1 megakernels

Use this skill to widen the candidate set when the remaining gap is in sync, in streaming
efficiency, or in short dependent phases. It lists the features, the questions that decide
between them, and what was measured on one B200. It does not say which option wins on your
kernel. Most numbers come from isolated microbenchmarks, so treat each one as a
hypothesis about your kernel and confirm it in place with `megakernel-microbench` and
`megakernel-phase-profiling`. Rows are in primdb (`family` `sync` and `mem`, source file `mk-exp7/H2/...`).

Conditions for all numbers unless stated: B200, 148 SMs, SM clock 1965 MHz (it stays there
under memory streaming), CUDA 13.1, one CTA per SM, globaltimer (32 ns step) or cudaEvent.

Background reading in mit-han-lab KernelWiki (https://github.com/mit-han-lab/KernelWiki, page ids): `hw-pdl-gdc`,
`hw-mbarrier`, `hw-tma`, `hw-clc`, `technique-cache-policy`, `technique-vectorized-loads`,
`technique-persistent-kernels`, `pattern-memory-bound`, `pattern-low-sm-utilization`,
`kernel-nvfp4-gemv`. These pages give spellings and protocols, not B200 numbers.

## 1. Grid-wide sync

### Know the floor before you redesign the barrier
Measured release latency with 148 CTAs: one CTA arrives 4 µs late, the others are already
polling. Latency runs from the late CTA's arrive to the last CTA's exit.

| protocol | release latency (last / median CTA) | back-to-back period |
|---|---|---|
| `red.release.gpu` + `ld.acquire.gpu` poll (thread 0), bar.sync around | 1.09 / 1.02 µs | 1.30 µs |
| same, `__nanosleep(32)` in the poll | 1.06 / 1.02 µs | 1.29 µs |
| `red.release` + `ld.relaxed` poll + `fence.acq_rel.gpu` | 1.47 / 1.41 µs | 1.45 µs |
| 8 counters on separate lines, 8 lanes poll (acquire) | 1.06 / 1.02 µs | 1.59 µs |
| per-CTA flag lines, 148 threads poll all (acquire) | 1.09 / 1.02 µs | 1.59 µs |
| `atom.acq_rel` last arriver flips a flag | 1.82 / 1.76 µs | 1.96 µs |
| last arriver writes 148 private flag lines | 1.76–2.14 µs | 2.07–2.17 µs |
| cluster 2/4/8/16: `barrier.cluster` + leader counter | 1.86 / 1.70 / 1.70 / 1.66 µs | 1.75–1.79 µs |
| cluster 2/4/8/16: remote `mbarrier.arrive` gather + leader counter | 1.92 / 2.21 / 2.91 / 4.32 µs | 1.87–4.34 µs |
| lower bound: `red.relaxed` + `ld.relaxed`, no fences (unsafe) | 0.67 µs | 0.98 µs |

What explains the numbers:
- `fence.acq_rel.gpu` and `red.release` both compile to `MEMBAR.ALL.GPU`. One fence costs
  about 135 ns with nothing outstanding, and about 380 ns after a store, because it waits for
  the store's ack. `ld.acquire.gpu` compiles to a strong load plus `CCTL.IVALL` with no
  MEMBAR, so an acquire poll saves the second MEMBAR (about 0.4 µs per barrier).
- Contention on one counter is not the bottleneck. 148 `red`s to one address serialize in
  about 0.5 µs total, and splitting the counter or the flags gave nothing.
- A flag's one-way latency between SMs is bimodal: about 225 ns or about 420 ns, depending
  on the SM pair (likely same die or cross die).
- Clusters cost SMs. With one CTA per SM, the most co-resident CTAs are 148 for cluster 2,
  132 for 4, 120 for 8 and 112 for 16. The cluster hierarchy was slower than the flat counter
  at every size.
- Traffic on the arriving CTA has little effect when the GPU is otherwise idle. The flat
  barrier is +60 to +190 ns when the CTA has 16 KB of cold loads or stores outstanding.

Questions: Does your barrier poll with `ld.acquire`, or with a relaxed load plus a fence? Is
your in-place release latency close to 1.0 µs? If it is 2–4 µs, the protocol is not the
cost: look at skew and at concurrent bulk traffic (next section).

### Sync under concurrent bulk traffic
Setup: each of the 148 CTAs runs 8 barrier warps and 1 producer warp. The producer streams a
2 GiB buffer from HBM through a ring of S × 16 KB `cp.async.bulk` chunks. One CTA arrives 4 µs
late. Latency is shown as last / median exit.

| arm | S=0 (idle) | S=1 (2.7 TB/s) | S=2 (4.9 TB/s) | S=4 (6.4–6.7 TB/s) | S=8 (6.4–6.8 TB/s) |
|---|---|---|---|---|---|
| flat `red.release` + `ld.acquire`, late CTA rotates | 0.96 / 0.90 | 1.31 / 1.22 | 1.60 / 1.50 | 1.89 / 1.70 | 1.92 / 1.73 µs |
| same, late CTA always in CTAs 0–7, which have no producer | 0.93 / 0.90 | 0.96 / 0.90 | 0.96 / 0.93 | 1.02 / 0.93 | 1.02 / 0.93 µs |
| same, late CTA in CTAs 0–7, all CTAs produce | 0.93 / 0.90 | 1.31 / 1.25 | 1.54 / 1.47 | 1.76 / 1.63 | 1.79 / 1.66 µs |
| producer stops issuing while its CTA is in the barrier | 0.96 / 0.90 | 1.15 / 0.99 | 1.18 / 1.06 | 1.22 / 1.15 | 1.22 / 1.12 µs |
| cluster 2, remote mbarrier + leader | 1.60 | 3.55 | 4.03 | 4.80 | 4.99 µs |
| cluster 4 (132 CTAs), remote mbarrier + leader | 1.98 | 5.28 | 5.95 | 6.75 | 6.82 µs |
| lower bound `red.relaxed` + `ld.relaxed` (unsafe) | 0.58 / 0.51 | 0.64 / 0.51 | 0.64 / 0.51 | 0.67 / 0.54 | 0.67 / 0.54 µs |
| one-way flag ping-pong, CTA0↔74 and CTA0↔1 | 227–231 ns at every load level ||||| 

What this shows:
- The slowdown is in the release (`MEMBAR.ALL.GPU`) of the CTA that arrives, when that CTA's
  SM has bulk loads in flight. The fence appears to wait for the SM's outstanding TMA reads.
  - If the late CTA has no producer, the barrier stays at about 1.0 µs even at 6.1 TB/s.
  - The relaxed lower bound and a single flag hand-off are not slowed by load.
  - Neither the polling side nor the L2 round trip is the cause.
- The slowdown grows with the bytes in flight on the arriving SM: +0.35 µs with 16 KB and
  +0.95 µs with 64–128 KB.
- Stopping the producer while its CTA is in the barrier bounds the slowdown to about +0.25 µs.
  The stream rate in that row only reflects the 4 µs waits of the test.
- Cluster gathers are 3–4× slower under load. `barrier.cluster` could not be tested under
  load because it needs every thread, including the producer warp.
- PTX has no QoS or priority control for a sync line.

### In a real kernel (Qwen3-30B-A3B b1 MoE megakernel, 148 CTAs, 2026-11-18)
- Measured per block, the layer-end `red.release` takes 1.13–1.53 µs. Load adds only
  0.1–0.2 µs over the idle floor. Align timelines per block. Aligning by event index across
  blocks with different event counts made it look like 2–4 µs.
- Deleting every barrier (an incorrect run, used only as an upper bound) saved about 245 µs of
  1536 µs. Most of that is waiting, arrival skew and HBM contention, not release latency.
- Holding the next layer's TMA prefetch until two chunks before the barrier arrival (QD) saved
  34 µs (2.2%). It moved the next layer's traffic off the tail of the expert phase, and the
  last expert arrival came 1 µs earlier. Holding until full arrival saved 10 µs. Holding until
  after the next phase's first load lost 21–123 µs.
- A router barrier became per-copy tagged logits with 16 replicas, polled by the TMA producer.
  The producer computed the top-k itself and issued the expert loads directly. Together with a
  sorting-network top-k this saved about 58 µs (3.8%). Without the replicas, with consumers
  polling, or without the producer issuing the expert loads, it was neutral or slower.
- In this kernel a 30 ns poll backoff cost 35 µs relative to 300 ns. On a hot all-to-all flag
  the poll traffic slows the writers. Sweep the backoff in the real kernel; the idle-barrier
  result below does not transfer.
- Retried without a gain: evict_first on expert weights (+7 µs), fewer ring slots (+2 to
  +99 µs), and holding back the other warps' loads near a barrier (Qwen3-0.6B dense b1, +2.7%).

### Other sync features
- PDL (`griddepcontrol.launch_dependents` / `.wait`; `hw-pdl-gdc`) is a launch-boundary
  feature. At about 1.2 µs per boundary it is not below an acquire-polled grid barrier.
  Inside one persistent kernel, `griddepcontrol` has nothing to wait on.
- `__nanosleep(t)` in a poll loop rounds up. The measured sleep for a request of
  0/20/32/64/100/256/1000 ns was 38/32/64/127/128/512/1019 ns. A 32 ns backoff costs
  nothing measurable on an idle barrier. Do not use 256 or more.
- Retried without a gain: hierarchical (cluster) barriers, split counters, per-CTA flag
  all-gather, last-arriver fan-out. They pay an extra hop and do not shrink the MEMBAR or
  the L2 round trip that dominate.

## 2. Streaming bandwidth

### What the ceiling is and how many SMs reach it
- The cold HBM read ceiling is about 7.3–7.4 TB/s (the best point was 7.43 TB/s, `ld.global.v4`,
  96 CTAs × 1024 threads). TMA and 256-bit loads reach the same.
- Per-thread loads deliver about 6–7 GB/s per resident warp. This holds for unroll 2 to 16
  and for 128- or 256-bit loads. More bytes in flight per warp did not help: only more warps
  did. The per-SM rate is about 54 GB/s with 8 warps, 105–113 GB/s with 16 and 190–207 GB/s
  with 32.
- SMs needed for 95% of the ceiling (6.9 TB/s):
  - 48 SMs with 1024 threads and 256-bit loads; 64 with 1024 threads and 128-bit loads.
  - 96–128 SMs with 512 threads.
  - Not reachable with 256 threads, even at 148 SMs (6.0–6.3 TB/s).
- A TMA ring is limited by its depth (Little's law, about 0.7–0.8 µs latency). Primdb
  `tma.load2d.bw` reaches 158 GB/s for a single SM with 4 × 32 KB in flight. At 148 SMs the
  share is 49 GB/s per SM. Measured 1D ring, with one consumer `__syncthreads` per chunk (per-SM
  GB/s at 16–64 SMs, then total at 148 SMs):
  - 8 KB chunks: 21–26 GB/s per SM at any depth. The consumer handles about 3 chunks/µs, so
    this is a per-chunk overhead limit, not a bandwidth limit.
  - 16 KB × 2: 37–39 GB/s per SM, 4.9 TB/s total.
  - 16 KB × 4–8: 52–53 GB/s per SM, 6.8–7.0 TB/s total.
  - 32 KB × 2: 61–66 GB/s per SM, 6.8 TB/s total.
  - 32 KB × 4: 84–87 GB/s per SM, 7.0 TB/s total.
  - At 148 SMs, 64–128 KB in flight is enough. More (192 KB) loses 5–10%.
  - `.L2::cache_hint` evict_first on the TMA changes nothing (<1%).
  - A ring at about 41 GB/s per SM with ≥16 KB chunks is short of in-flight bytes, or it pays
    a per-chunk overhead.
- Questions: How many warps per SM issue the stream, and at about 6.5 GB/s per warp, can they
  reach your per-SM share? How many bytes per SM does your ring keep in flight? Does a phase
  run on fewer SMs than the 48–64 that saturate HBM?

### 256-bit loads (`ld.global.v8.b32`)
- They exist on sm_100a: `LDG.E.ENL2.256`, 32 B per thread. Per warp they were 2–6% faster
  than 128-bit loads, and they need about 25% fewer SMs to saturate HBM.
- The L2 eviction-priority qualifiers (`.L2::evict_first` / `.L2::evict_last`) are accepted
  only on 256-bit loads. ptxas rejects them on `.v4.u32`. For 128-bit loads use
  `.L2::cache_hint` with a `createpolicy` operand.

### L2 eviction hints: they protect other data, they do not raise stream bandwidth
- On a pure stream, every hint changed bandwidth by less than ±1%. The hints tested were
  `.nc`, `.L1::no_allocate`, `.L2::256B`, `.L2::cache_hint(evict_first)` and
  `.L2::evict_first` / `evict_last` on v8.
- They matter when something else must stay in L2. Test: per "layer", stream W MB of unique
  weights, then re-read an H MB hot set; 148 CTAs × 512 threads.

| W / H (MB) | weights loaded with | weight phase | hot-set phase | total |
|---|---|---|---|---|
| 96 / 32 | `ld.v4` | 17.9 µs | 15.8 µs (from HBM) | 541 µs |
| 96 / 32 | `.nc.L1::no_allocate.L2::evict_first.v8` | 17.5 µs | 6.3 µs (L2 hits) | 382 µs |
| 96 / 64 | `ld.v4` | 17.9 µs | 28.5 µs | 744 µs |
| 96 / 64 | `evict_first.v8` | 17.6 µs | 11.4 µs | 464 µs |
| 96 / 64 | `ld.v4`, accessPolicyWindow persisting on H | 18.0 µs | 17.4 µs | 566 µs |
| 24 / 8 | any | 5.8–6.0 µs | 2.5 µs | no effect |

- An evict_last hint on the hot set alone did nothing. The weights must carry evict_first or
  no_allocate.
- The persisting window can hold at most 79 MB of the 126 MB L2, and it helped only partly.
- In this loop, the 128-bit `.nc.L1::no_allocate` loads streamed 25% slower than plain
  `ld.v4`. They also protected the hot set. The 256-bit evict_first form was both the fastest
  and protective. Check codegen and bandwidth in place.
- Questions: Is there data that the next phase reads again (KV cache, activations, the next
  layer's small weights, counters)? Is weights-per-phase plus that data larger than about
  100 MB? If not, hints will not move the time.

### Bulk L2 prefetch (`cp.async.bulk.prefetch.L2.global [p], bytes`)
- One lane can cover a whole tile with no shared memory, no mbarrier and no registers.
- Measured with the primdb ingress harness (`ld.global.nc.v4`, U=2). Warp 0 lanes issue
  `cp.async.bulk.prefetch.L2` for the CTA's tile 2 or 8 iterations ahead. Total GB/s, cold:

| threads/CTA | CTAs | plain | + prefetch d2 | + prefetch d8 |
|---|---|---|---|---|
| 256 | 148 | 6071 | 6810 (+12%) | 6560 |
| 256 | 128 | 5504 | 6194 (+13%) | 6047 |
| 512 | 64 | 5530 | 5938 (+7%) | 5885 |
| 512 | 148 | 7290 | 7307 | 6682 (−8%) |
| 1024 | 32 | 5392 | 3912 (−27%) | 3863 |
| 1024 | 148 | 7279 | 7238 | 6572 (−10%) |

  - Prefetch helps only when there are too few warps to reach the per-SM share (256 threads
    at 128–148 SMs). Prefetching too far ahead (d8) or with enough warps hurts.
  - At 1024 threads on few SMs, the prefetch costs 27–35% (it competes with demand loads).
- Measured in place on the Qwen3-30B-A3B b1 MoE megakernel: an oracle prefetch cost +156 µs end to end, and a TMA
  ring with prefetch reached only 4.0–4.9 TB/s. Prefetch adds traffic and ordering
  pressure that a saturated stream does not need.
- Questions: Is the phase latency-bound with few warps (then prefetch can help) or already
  near 7 TB/s (then it hurts)? Does the prefetch land inside a sync window?

## 3. Short dependent phases

- L2 hit latency (strong load) is 262–355 cycles, mean 283 (near) and 316 (far), or about
  145–165 ns. A dependent relaxed `atom.add` takes 356 ns; an `atom.acq_rel` 750 ns.
- One-way flag hand-off between two SMs takes about 225 ns or about 420 ns. A per-tile
  readiness flag therefore costs at least about 0.25–0.45 µs, plus the consumer's load.
- The clock stays at 1965 MHz under load, with application clocks fixed. Only sustained
  tensor-core work lowers it (primdb `model.b0_calib.clock_mhz`: 1699 busy).
- elect.sync, setmaxnreg and CLC (`hw-clc`) were not measured here. They are relevant to
  warp-specialized and dynamically scheduled kernels, not to the latency floor.
- Questions: How many µs-scale serial events does one layer have, at about 0.25–0.45 µs per
  hop and about 1 µs per grid barrier? Can one hop become a local recompute?

## Beyond b1: large-batch attention and moderate-T MoE (measured 2026-11-18)

- Decode attention, bs 128/512, GQA 8 KV heads × hd 128, paged (page 64), bf16:
  - trtllm-gen already reaches 7.35–7.45 TB/s at ctx ≥ 2048. That equals a pure v8 stream
    of the same bytes, so no streaming rule can help there.
  - At ctx 512 a zero-compute stream would be 3–11% faster. That gap is launch, ramp and
    split-KV combine cost, not bandwidth.
  - flashinfer's default CUDA-core decode is compute-bound at G=4 (2.6–3.6 TB/s). Use the
    trtllm-gen backend.
- Qwen3-30B-A3B MoE layer (E128 top8 H2048 I768), measured against trtllm_bf16_moe:
  - A persistent kernel (TMA ring, mma.sync, fixed tagged hand-offs) was 9–20% faster at
    T ≤ 16. It was 3–5% faster at T = 32–64 with uniform routing, and 4% slower at T = 64 with
    skewed routing.
  - Stop using the persistent design once an expert gets about 24–32 tokens; use a grouped
    GEMM instead.
  - About 17–23 µs of fixed routing, scheduling and finish overhead remains per layer. It caps
    T = 1 at about 37% of the bandwidth floor.
- Rules that held in both:
  - 96–128 KB in flight per SM with chunks of at least 16 KB (32–64 KB is slightly better).
    160–192 KB loses 3–12%. A single large slot loses 17–27%.
  - Bulk L2 prefetch on a saturated stream hurts. More bytes in flight per warp does not help.
- Corrections:
  - The warp count is capped by the register file. With 16 or more compute warps,
    attention and MoE kernels spilled (+5–24%), and 12 warps × 256 B/thread was best. Size by
    bytes in flight per SM, not by warp count.
  - In single-op kernels, evict_first on read-once weights or KV gave 1–5%, by keeping small
    shared data (Q, partials, page tables, counters) in L2. In the b1 megakernel, evict_first
    on experts cost 7 µs. Measure it per kernel.
  - QD-style holding only helps when the next phase's traffic overlaps this phase's tail
    (−2.2% in the b1 megakernel, 0 to +5% inside one MoE layer). Barrier-to-hand-off
    replacement inside one MoE layer is worth only 1–2%.
- New rules:
  - Under skewed routing, balance CTAs by weighted compute rather than by bytes. A per-token
    weight about 500× the per-byte cost saved 4–13% at T = 16–64.
  - Never finish serially in the last-arriving CTA. That made a ~50 µs tail in one version.
  - In short phases, issue all loads before the dependent stores.

## What this suggests for b1 MoE and dense decode

- Make every poll an `ld.acquire` poll with no extra fence. Expect about 1.0 µs per barrier
  in isolation, not less. Sweep the `__nanosleep` backoff (30–300 ns) in the real kernel.
- Do not use clusters for grid sync. They lose SMs and add hops.
- Stream with 32 warps per SM, or with a TMA ring that keeps at least 64 KB in flight per SM.
  Use v8 loads with `.L2::evict_first` for weights read once, so that KV, activations and
  sync lines stay in L2.
- Keep bulk traffic and prefetches out of the sync windows. A barrier costs about 1.0 µs only if
  the arriving CTA's SM has no bulk loads in flight when it executes the release. Options:
  - Let the producer stop issuing (or drain the ring) before the CTA arrives (+0.25 µs
    instead of +0.95 µs). In the MoE kernel, holding it until two chunks before arrival
    saved 2.2% end to end, mostly by reducing HBM contention.
  - Keep the ring at 2 × 16 KB per SM near barriers.
  - Arrive from SMs that do not stream.
  - Replace barriers with per-consumer flags (one hop is about 230 ns even under load).
