---
name: megakernel-phase-profiling
description: Use before choosing an optimization for a megakernel, and whenever it is unclear which phase of the kernel is the bottleneck.
---

# Megakernel phase profiling

A megakernel hides its structure from whole-kernel counters. NCU averages over the grid.
It cannot say which phase, or which CTA, sets the critical path.
Do not tune blindly. Split the kernel into phases, time each one inside the kernel,
and attack the phase with the most time or the most recoverable gap.

This skill builds on the workspace tools. It does not replace them:
`./timeline_candidate` with `timeline_probe.cuh`, `./profile_candidate` (NCU),
`./sol_info` and `./check_candidate`.

## 1. Split into phases

- Cut the kernel at its natural sync points: grid or group barriers, flag or mbarrier waits,
  role hand-offs between warps, and the stages of a loop-carried pipeline.
- Name each phase after what it does. You choose the split.
- Start coarse. Add finer splits only where they will change a decision.
- A phase that repeats (per layer, per iteration) keeps one name. Record every occurrence.

## 2. Pick the timer for the question

`%globaltimer` through `timeline_probe.cuh` and `./timeline_candidate`:
- Use it for per-block and per-group phases across the whole grid.
- Its clock is shared, so CTAs can be aligned. You get begin skew and imbalance.
- Its resolution is coarse. Do not read differences below one tick of the timer.
  Check the tick size on your device before trusting short phases.
- Sum many occurrences before judging a short phase.

`clock64` (SM-local cycles):
- Use it for fine intervals inside one CTA, or inside one warp role.
- Cycles from different SMs are not aligned. Never subtract stamps from two SMs.
- Let one lane stamp. Keep the stamps in registers or shared memory.
  Write them to global memory once, at kernel end, or into extra timeline events.
- Resolution is one cycle, but the stamp itself costs cycles and can reorder issue.
  Do not read differences near that cost.

For each sync point, separate three numbers:
- work time: the CTA is doing the phase;
- wait time: from arrive to release;
- imbalance: the last arriver compared with a typical arriver.
A long wait can mean that another CTA is late. Find which one, not just how long.

## 3. Do not disturb production (non-intrusive rule)

Profiling must not change what the production kernel does.

- Put all instrumentation behind one compile-time switch (for example `MEGABENCH_TIMELINE`).
- Switch off: the build must be identical to production. Gate check 1.
- Switch on: keep probes minimal.
  One lane stamps. Few stamps. No extra syncs.
  No new shared memory that changes occupancy.
- Switch on: resource use must not drift from production. Gate check 2.
  Compare registers, spills, stack, local memory, shared memory, occupancy and launch geometry.
- If the resources drift, the timeline does not show production. Do not trust it.
  Reduce the probes or move them.

## 4. Codegen gate

Run `scripts/codegen_gate.py` (next to this file). It needs only Python 3 and `cuobjdump`.
Inputs may be `.cubin`, `.o`, `.so`, or an executable with a fatbin.

```
# check 1: profiling OFF build equals production
python3 scripts/codegen_gate.py identical --production prod.cubin --candidate off.cubin [--kernel REGEX]

# check 2: profiling ON build does not grow resources
python3 scripts/codegen_gate.py resources --production prod.cubin --candidate on.cubin \
    [--kernel REGEX] [--allow reg,stack,smem,spill] [--json]
```

- `identical` compares normalized SASS per kernel and prints a diff excerpt on mismatch.
- `resources` prints REG, STACK, SHARED and LOCAL for both builds.
  Growth in spills (LOCAL), stack or shared memory fails.
- Register growth is reported as WARN. It fails when it lowers occupancy.
  Pass `--threads-per-cta` (and `--smem-per-cta` for dynamic shared memory) to enable that check.
  It also fails if it exceeds the launch's register cap (`--max-regs`).
  `--strict-regs` fails on any register growth.
- Use `--allow` only for a growth you can explain, and say so in your notes.
- `resources` accepts `--production-log` and `--candidate-log` (`ptxas -v` text) for spill detail.
- Exit codes: 0 pass, 1 mismatch, 2 usage or tool error.

Run the gate before you trust any timeline, and again before you submit.
Build the production, off and on variants from the same source and flags
except the profiling define.

## 5. Staged profiling

1. Coarse pass. All CTAs, every phase, `./timeline_candidate`.
   Read wall, slowest, mean, begin_skew and imbalance per phase.
2. Narrow. Keep stamps only for the problem phase and the problem CTA(s).
   Use the `is_writer` argument of `MB_TIMELINE_INIT`, or a compile-time CTA or phase mask.
3. Go deeper in that CTA. Add per-warp or per-role stamps with `clock64`.
4. Watch the cost of depth. If more stamps cause spills, register growth,
   or timing shifts that production would not show, back off.
   Use fewer stamps, or split the question across several builds.
5. Run the codegen gate after every change to the instrumentation.

## 6. Choose the target

- Reduce per phase across CTAs and across repeated occurrences.
- Find the critical path. A phase that is long but overlapped with others may not matter.
- Rank phases twice: by time, and by gap to a floor.
  - Floor from data: bytes over bandwidth, and FLOPs over peak. Use `./sol_info`.
  - Floor from a null-work variant: keep the phase's sync and memory movement and skip the math,
    or keep the math and skip the memory movement.
    The null variant is a measurement tool. It is not a candidate.
- Prefer the phase with the largest recoverable gap, not only the slowest one.
- Re-profile after each accepted change. The bottleneck moves.

## 7. Per-phase resource signals (supplement)

Collect these per phase. They are signals for forming hypotheses, not proof.
- Registers and live-register pressure.
  Use `nvdisasm -lrm count` or `--print-life-ranges`.
  Map the result to source with `-lineinfo`.
- Active threads, active warps and warp roles.
- Shared memory and TMEM or smem buffers in use.
- Units of work per CTA, to see balance.
- Bytes moved and FLOPs.
- NCU source-level counters from `./profile_candidate` with source correlation,
  attributed to the phase's code region.

Example hypotheses these can suggest: high live registers with spills; few active warps;
uneven work per CTA. Test each one with a measurement before acting on it.

## 8. Keep a per-phase table

Keep a short table in the workspace notes. Update it every round.

| phase | time | share | floor / gap | resource signals | hypothesis |
|-------|------|-------|-------------|------------------|------------|

Later rounds reuse it. Record what you tried and what moved.

## Companion skills

- `megakernel-microbench`: isolate the chosen phase in a small harness and iterate on it.
- `blackwell-tensor-core-design`: design options for matrix-shaped phases.

Confirm every accepted change with `./check_candidate` on the production build.
