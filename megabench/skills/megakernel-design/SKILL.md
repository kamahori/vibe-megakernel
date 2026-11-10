---
name: megakernel-design
description: Use when planning or reviewing an optimization of a megakernel, to list the design options for synchronization, overlap across phases, fusion, on-chip residency, batch-1 GEMV and GEMM, attention, sequential inner loops and work distribution, with the questions that decide among them.
---

# Megakernel design

Use this skill to choose what to try next. It lists design options and the questions that decide
between them. It does not tell you which option wins on your case. Answer the questions with
measurements: `megakernel-phase-profiling` finds where the time goes, and `megakernel-microbench`
tests one option in isolation. For tensor-core engines on Blackwell, also read
`blackwell-tensor-core-design`.

Each section ends with ideas that earlier runs retried many times without a gain. They are not
forbidden. Before you plan one, write down what in your case differs from the cases where it
failed.

## Start from the structure of one step

Before you pick an option, write down for one step of the kernel:

- The phases in order, and which phase needs which output of an earlier phase.
- The dependency points: every place where one CTA waits for data that other CTAs produce.
- For each phase: bytes read, FLOPs, how many CTAs have work, and the floor from `./sol_info`.
- Loops whose iterations depend on each other (decode steps, codebooks, flow or denoising
  steps), and what is the same in every iteration.

Questions: Does the step take much longer than its bytes need? Then the cost is in latency,
dependency points and idle CTAs, not in bandwidth. How much of the step would be left if all
math and all loads were free (the sync skeleton)?

## 1. Dependency points and barriers

### Remove a dependency point
- Merge two phases so that one CTA produces what it later consumes.
- Replace a grid-wide barrier with a readiness signal that only the real consumers wait for
  (a counter per group, a flag per tile, or data tagged with the step).
- Let each CTA recompute small shared results (a norm, a reduction, an argmax) instead of
  waiting for one producer.
- Questions: Which consumers really need all producers? How much redundant work does a CTA
  add to skip the wait? Does the new signal need a reset, and who resets it?

### Make a barrier cheaper
- Change the barrier protocol (cooperative grid sync, an atomic counter with acquire
  polling, per-CTA flags, a tree).
- Reduce skew: give the slowest CTA less work before the barrier.
- Questions: What does one barrier cost at your grid, and how much of it is skew rather than
  the protocol? How many barriers does a step have, and what share of the step is that?

Retried often without a gain: hierarchical barriers and barriers built from per-CTA flags. A
cheaper protocol helped in some cases and not in others, so measure your skew first.

## 2. Overlap across phases and weight streaming

### Stream weights through an on-chip ring
- Copy weights into a shared-memory ring with bulk or tensor copies while the CTA computes.
- Questions: How large is each copy, and how many bytes are in flight per SM? Can one
  multidimensional tile copy replace many small row copies? Does the ring depth fit the shared
  memory left by the other phases?

### Issue the next phase's loads before the barrier
- Start the first copies of the next phase before the barrier, so they land during the wait.
- Questions: How many tiles must be in flight to hide the wait? Do the early copies take
  bandwidth or shared memory from the current phase? Does the barrier release later while copies
  are in flight?

### Hoist loads
- Load the operands of the next unit (K/V rows, weights) into registers before the work that
  precedes them.
- Questions: Are the registers free at that point? Does the compiler keep the load early, or
  move it back? Check the SASS.

Retried often without a gain: prefetching later weights into L2 with prefetch instructions, and
issuing the whole next phase ahead instead of its first tiles.

## 3. Fusion and folding

- Fold a norm, a modulation (scale and shift), RoPE or a residual into the prologue or epilogue
  of the neighbouring GEMV or GEMM.
- Fuse attention with its output projection, so each head adds its slice of the result.
- Fuse gate/up with the activation, and consider fusing the down projection with a split
  reduction.
- Fuse an output head (vocabulary or codebook) with its argmax: each CTA writes a partial
  maximum in the epilogue.
- Questions: Which barrier does the fusion remove? Does it make the reduction order change, or
  need atomics? Does the fused phase still use all SMs? Does it raise register use?

Retried often without a gain: precomputing per-position constants such as RoPE tables for a
fixed position, and folding a residual into an epilogue where it adds a dependency.

## 4. On-chip residency

- L2 cache hints: keep data that a step reads many times (evict last), and let single-use
  streams leave first (evict first).
- An L2 persisting window set at launch.
- A slice of weights kept in shared memory across steps by the CTA that uses it.
- Intermediates kept in registers or tensor memory instead of global memory.
- Questions: How many times per step is each weight set read, and how large is it next to the
  L2 and the shared memory left after the rings? Did the hint take effect? Measure DRAM bytes,
  not only time. Which single-use stream pushes the reused set out? Does a resident slice take
  shared memory or registers from the pipeline of the current phase?

## 5. GEMV and GEMM at small batch

- Batch 1 (decode): warp per row, split over K, several rows per warp, and how the input vector
  is shared across rows.
- Mid M (tens to hundreds of tokens, for example an action chunk or the tokens of one frame):
  between GEMV and tiled GEMM.
- Engine: SIMT, warp-level MMA, or the asynchronous tensor-core path
  (`blackwell-tensor-core-design`; `./primdb` and `./engine_map` when the workspace has them).
- Dequantization: the instruction count per weight byte.
- Questions: Is the phase bound by bytes, by issue of the compute step, or by its prologue
  (norm, partial sums)? What fraction of a tensor-core tile is useful after padding? Does the
  engine need a launch geometry (CTAs per SM, tensor memory) that hurts other phases?

## 6. Attention at small batch

- Split each head over keys or KV chunks across more CTAs, then merge the partial softmax.
- Group query heads that share a KV head.
- Shapes that multimodal models add: bidirectional attention within a chunk, cross-attention
  to a cached prefix, rolling or dilated windows, a few hundred tokens per frame.
- Questions: How many CTAs work during attention, and how long do the others wait? What does
  the merge cost? Does the KV fit on chip for the whole step? Is attention bound by loads, by the
  softmax chain, or by the matrix products?

## 7. Sequential inner loops

Some steps contain a loop whose iterations depend on each other: the codebooks of a depth
decoder, the steps of a flow or diffusion sampler, the denoising passes of a world model.

- Hoist work that is the same in every iteration out of the loop (time or noise conditioning,
  modulation for all steps, constant projections).
- Keep the loop's state and its reused weights on chip across iterations (section 4).
- Shorten the critical path of one iteration: fewer dependency points, the argmax and the next
  embedding fused into the previous iteration's epilogue.
- Questions: What share of the step does the loop take? How many dependency points does one
  iteration have? Which weights does every iteration read again? What is the floor of one
  iteration, and how far is it from the measured time?

## 8. Work distribution and launch geometry

- Give CTAs roles (for example attention and its projections on some CTAs, weight streaming on
  others), so idle SMs keep working.
- Spread large heads and small phases over all SMs.
- CTAs per SM, threads per CTA, registers per thread, and clusters.
- Questions: How many SMs are busy in each phase? Does the slowest CTA always belong to the same
  phase? Does a cluster size still allow a cooperative launch with every CTA resident? Does a
  change in registers lower the CTAs per SM?

Retried often without a gain: clusters added to a design built around grid barriers, and larger
grids than the SMs can hold at once.

## 9. Numerics

- Check which changes move the result: a different reduction order, atomics, fused rounding,
  lower-precision intermediates, parts that the reference computes in FP32.
- Questions: Does the change stay inside the task's tolerance on all outputs? Is the result
  still deterministic run to run, and does the task require it? Near ties, can a greedy argmax
  pick a different index?

## Record the decision

In the plan, name the option you chose, the question that decided it, and the measurement that
answered it. When an option fails, write down the conditions, so a later round knows when a retry
is worth it.
