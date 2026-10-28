---
name: blackwell-tensor-core-design
description: Use when planning or reviewing a megakernel phase that is matrix-multiply-shaped or attention-shaped on Blackwell (sm_100a), to put tcgen05, TMEM, TMA, warp-specialized and work-scheduling designs into the candidate set and to decide among them by measurement.
---

# Blackwell tensor-core design

Use this skill to widen the design space. It lists options and the questions that decide
between them. It does not tell you which option wins. Treat every property below as a
hypothesis about your kernel on your device. Confirm it with a small experiment
(`megakernel-microbench`) before you build a design on it.

Agents often stay with SIMT FMA or warp-level `mma.sync` because the Blackwell path is
harder to write. Do not drop the Blackwell path without evidence from your own
measurements. Do not adopt it without evidence either.

## Learn the hardware from first principles

Before you design around a feature, measure the feature in isolation. Examples:

- Issue cost and latency of one MMA, for each instruction shape and data type you consider.
- Throughput of the MMA as tile size and pipeline depth change.
- Cost of moving accumulators from Tensor Memory (TMEM) to registers.
- Bandwidth and latency of TMA copies against per-thread loads, for your tile shapes,
  for cold and warm cache, and with other CTAs that compete for bandwidth.
- Cost of an mbarrier hand-off between warps, and of a fence between the generic and
  async proxies.
- Cost of each synchronization or scheduling primitive you plan to use.

Write each result in your notes with the conditions of the test. Use the results, not
general claims, to choose a design.

## Options to evaluate

Each item names an option and the questions that decide it.

### tcgen05 MMA with TMEM accumulators
- The MMA is issued by one thread and completes asynchronously. Accumulators can live in
  TMEM instead of registers.
- Questions: Does this lower register pressure or spills in your phase? Does it let other
  warps do useful work during the MMA? What is the setup cost (allocation, descriptors,
  barrier init, pipeline fill) relative to the phase time?
- Double-buffered TMEM accumulators can overlap the epilogue of one tile with the MMA of
  the next. Measure whether the overlap appears in your phase.

### Operand ingress: TMA, per-thread loads, or a mix
- TMA (`cp.async.bulk.tensor`) moves tiles with descriptors and mbarrier transaction
  counts. Per-thread loads give more control per element.
- Questions: Which one reaches more of the bandwidth floor for your tile shapes? How deep
  must the ring be to hide latency, and does that depth fit the shared-memory budget?
  Do multicast or cache hints change the result?

### Warp specialization
- Roles are possible: a producer for copies, an MMA issuer, and epilogue or consumer warps,
  joined by mbarriers. Register reallocation between roles (`setmaxnreg`) is possible.
- Questions: Does a role split shorten the phase, or only add hand-off cost? Which role is
  on the critical path? In a megakernel, can a role run ahead into the next phase, and
  does that help?

### Problem shape
- Small or skinny problems may not fill a tensor-core tile. Options:
  - swap the operands (compute the transpose) so that the large dimension maps to M;
  - pad or group small dimensions;
  - split K, or split along the sequence or KV length, with a merge step;
  - fuse neighbor operations into the epilogue;
  - arrange weight layouts and swizzles at build time.
- Questions: How much work does padding waste? What does the merge step cost? Which legal
  instruction shapes and data types fit your problem, according to the PTX documentation?

### CTA grouping
- An MMA can run on one CTA or on a pair of CTAs (`cta_group`).
- Questions: Does a pair reduce operand traffic enough to pay for paired scheduling and
  cluster launch? Read the PTX documentation for the rules on mixing `cta_group` values in
  one kernel, and design for those rules from the start.

## Work scheduling: static or dynamic

Decide how work units reach CTAs. Do not assume a static split is enough.

- Static: each CTA gets a fixed set of units from its index. It has no runtime cost, but
  it cannot react to uneven work or to late CTAs.
- Dynamic: CTAs take units from a queue at run time. Examples are an atomic counter in
  global memory, per-phase work lists, or hardware-assisted scheduling such as cluster
  launch control, if your toolchain supports it.
- Hybrid: a static first share, then dynamic stealing for the remaining units.

Questions:
- How uneven are the units? Use the phase timeline (`megakernel-phase-profiling`) to see
  the spread of per-CTA finish times.
- What does one dequeue cost, and how does it scale with the number of CTAs that compete?
  Measure it.
- Does dynamic order change cache reuse or the overlap with the next phase?
- Does the scheme stay correct across grid-wide synchronization and across phases?

Build both schemes in a microbenchmark of the phase. Keep the one that wins end to end.

## Constraints to check

Read the PTX documentation and confirm each rule with a small test:

- TMEM allocation, relinquish, and deallocation, and how the column budget is shared
  across phases.
- mbarrier initialization, phase and parity tracking across reuse, and what happens on
  a wrong parity.
- The fences that are necessary between MMA, TMEM, TMA, and generic memory operations.
- The shared-memory budget for rings and for the data of other phases, with alignment
  for swizzled layouts.
- Register limits for your launch geometry, and residency for any grid-wide
  synchronization.
- The state that one phase must reset or hand over to the next.

## Verify before integrating

- Write a numerics unit test of each new path against a simple reference. Use
  distinctive data so that layout and swizzle errors show.
- Read the SASS (`cuobjdump -sass`) and the resource use (`cuobjdump -res-usage`).
  Confirm that the instructions you expect are present and that no spills appear.
- Measure with `megakernel-microbench`, calibrated against the current phase.
- A megakernel can use different engines and schedules for different phases and for
  different problem-size ranges. Record when and why each choice won or lost, with the
  conditions, so later rounds can reuse the evidence.

## Reference sources

If available in your environment, consult the CUDA PTX ISA documentation (tcgen05,
`cp.async.bulk.tensor`, mbarrier), CUTLASS and CuTe sm100 examples, DeepGEMM, and
FlashInfer or FlashAttention Blackwell kernels. Search the environment first. Do not
assume a path. When an instruction form is uncertain, check the documentation or an
example instead of a guess.
