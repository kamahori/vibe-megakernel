---
name: blackwell-tensor-core-design
description: Use when planning or reviewing a megakernel phase that is matrix-multiply-shaped or attention-shaped on Blackwell (sm_100a), to consider tcgen05, TMEM, TMA and warp-specialized designs and to reshape the problem to fit them before choosing an engine by measurement.
---

# Blackwell tensor-core design

Use this skill to widen the design space. Blackwell has hardware that SIMT FMA and warp-level `mma.sync` do not use. Agents often skip it because it is harder to write. Do not skip it without evidence.

The rule: for each matrix-multiply-shaped or attention-shaped phase, build and measure a tcgen05 candidate before you dismiss it. Decide by measurement, per phase, with `megakernel-microbench`.

## What the Blackwell path offers

- tcgen05 MMA is issued asynchronously by a single thread. The other threads in the CTA are free to do other work.
- Accumulators live in Tensor Memory (TMEM), not in registers. This frees registers for other work and reduces spills.
- Operands come from shared memory through descriptors (and, for some forms, from TMEM). The descriptor encodes layout and swizzle.
- Completion is tracked with an mbarrier, using the tcgen05 commit operation.
- The epilogue reads accumulators from TMEM into registers with `tcgen05.ld`. Double-buffer the TMEM accumulators so the epilogue of one tile overlaps the MMA of the next.
- Instruction shape (M, N, K), data type, and `cta_group` (one CTA, or a pair of CTAs) are choices. Make them on purpose.

## Operand ingress with TMA

- TMA (`cp.async.bulk.tensor`) moves tiles from global to shared memory without per-thread load instructions.
- Use a multi-stage ring of shared-memory buffers. Each stage has a "full" mbarrier (expect-tx bytes, then complete-tx by the copy) and an "empty" mbarrier released by the consumer.
- Consider multicast across a cluster and L2 cache hints when several CTAs read the same data or when data is read once.
- For streamed weights, prefer bulk async copies over per-thread loads unless measurement says otherwise.
- Tensor maps (descriptors) are built on the host or at build time. Plan where they live and how the kernel reads them.

## Warp specialization

- Split the CTA into roles: a producer warp (TMA), an MMA-issuer warp (one elected thread), and epilogue or consumer warps.
- Connect roles with mbarrier hand-offs. Track the phase (parity) bit for every ring and every reuse.
- Use register reallocation between roles (`setmaxnreg`) to give consumer warps more registers and shrink the producer and issuer.
- In a megakernel, roles can persist across phases. A producer can prefetch the next phase weights while consumers finish the current one. Test whether this overlap helps.

## Reshape the problem to fit the tile

tcgen05 tile shapes are large. Small or skinny problems need work to fill them.

- Swap A and B (compute the transpose) when the row or token dimension is small, so the large dimension maps to the MMA M and the small one maps to N.
- Pad or group small dimensions to reach a legal N. Count the wasted work and compare it with the savings.
- Split K, or split along sequence or KV length, to create enough parallel tiles. Add a merge step, and weigh its cost.
- Fuse neighbor operations into the epilogue: scaling, activation, residual add, quantization, or the next reduction.
- Pre-arrange weight layouts and swizzles at build time so TMA and descriptors read them directly.
- Pick data types for operands and accumulators deliberately. Check which types the instruction accepts and what accuracy the task needs.
- Choose `cta_group`: one CTA is simpler and fits small tiles. A CTA pair gives a larger tile and shares operand traffic, but it needs paired scheduling and cluster launch.

## Constraints to plan for

- TMEM must be allocated (`tcgen05.alloc`), relinquished, and deallocated (`tcgen05.dealloc`) correctly. The column budget is shared across all phases that use it.
- A kernel uses one `cta_group` choice for all its tcgen05 instructions. Decide early.
- Initialize mbarriers once. Track phase and parity across reuse. A wrong parity causes hangs or silent races.
- Shared memory holds the rings and also the data of the other phases. Budget the total, including alignment for swizzled layouts.
- Use the right fences: tcgen05 fences around synchronization with the MMA and TMEM operations, and proxy fences when generic-proxy writes to shared memory must be seen by the async proxy (TMA or MMA descriptors).
- Register caps follow from the launch geometry (threads per CTA, CTAs per SM). Check them against the roles you want.
- Persistent kernels with grid-wide synchronization need all CTAs resident. TMEM and shared memory use affect residency.
- Check how TMEM and rings are handed over between phases, and what state must be reset.

## When tcgen05 may lose

- Very small phases, or phases bound by loading data, where tensor-core throughput is not the limit.
- Setup cost (TMEM allocation, descriptor setup, barrier init, pipeline fill) is a large share of the phase time.
- Padding waste is large because a dimension is too small for the tile.
- An extra merge pass or an extra layout step costs more than the MMA saves.
- Losing neighbor-phase overlap that a simpler design had.

A megakernel may use different engines for different phases, and different engines for different problem-size ranges. Decide per phase and per size range. Record when and why a tcgen05 design won or lost, with the conditions.

## Verify before integrating

- Write a numerics unit test of the tcgen05 path against a simple reference (a plain GPU or CPU matmul). Test the layouts and the swizzles with distinctive data, not only random data.
- Check the SASS with `cuobjdump -sass` for the expected tensor-core MMA instructions (opcodes of the form `UTC*MMA`; the middle letters vary with the data type) and for TMA loads (`UTMALDG`). If they are missing, the compiler or your code took another path.
- Check `-res-usage` (for example `cuobjdump -res-usage` or `--resource-usage`) for registers, shared memory, and spills.
- Then measure with `megakernel-microbench`, calibrated against the current phase.

## Reference sources

If available in your environment, consult:

- The CUDA PTX ISA documentation: tcgen05, `cp.async.bulk.tensor`, and mbarrier sections.
- CUTLASS and CuTe sm100 examples.
- DeepGEMM.
- FlashInfer and FlashAttention Blackwell kernels.

Do not assume a path. Search the environment first. If a mnemonic or operand form is uncertain, check it against the PTX documentation or an example rather than guessing.
