---
name: megakernel-stage-linking
description: Use when a megakernel's phases should be developed, microbenchmarked or swapped one at a time — split it into stage packages with explicit contracts and link them back into one kernel, then verify the link costs nothing.
---

# Megakernel stage linking (PipeLink-style, AMD/HIP)

Idea from PipeLink (contract-guided linking of GPU kernel components): write each phase of the
megakernel as a separately compiled *stage package*, declare what each stage needs and provides,
check the contracts before linking, and link all stages into ONE kernel. Payoffs: a stage can be
rewritten, microbenchmarked and swapped without touching the others, and a wrong hand-off fails at
link time instead of corrupting results. The split itself is NOT a speedup. Gains only come from a
better stage that the split lets you build and swap in. Keep the split only if the linked kernel is
at least as fast as the one-unit build of the same sources.

## 1. Split along the hand-offs you already have
- Stage = one phase between hand-offs (grid barrier, tagged/flag wait, role hand-off).
- Entry point per stage: `extern "C" __device__ void stage_x(const Params&, Ctx&, Smem&, Regs&, int layer)`.
- Make every cross-stage value explicit: `Ctx` (geometry, ids), `Smem` (one `__shared__` struct owned by
  the driver kernel), `Regs` (prefetched weights, early K/V loads, residuals). Hidden state through
  globals or file-static variables defeats the contract check.
- A small driver ("interpreter") kernel sequences the stages. Shared helpers in the header must be
  `__forceinline__` or `inline` (a plain `__device__` function in a header is a duplicate symbol at link).

## 2. Declare and check contracts
`scripts/plk_link.py` reads `packages.json` (per package: src, defines, exports, calls, needs, provides,
consumes) and a pipeline file (`slots` + `sequence`, with a `repeat` block for the layer loop). It walks
the sequence twice through the loop (to catch loop-carried hand-offs such as prefetching the next
layer's weights), rejects a missing provider (`L-NEED`), a call to an unlinked export (`L-CALL`), then
compiles each package with `-fgpu-rdc` and links one shared library with device LTO.
Scope resources by block role when needed, for example `wwo@attn` and `wwo@other`.
Run `--check-only` first. See `example/` for a two-stage toy.

## 3. Measured seam costs on MI300X (dense Qwen3 megakernel, ROCm 7.2, gfx942)
- Separate TUs + LTO link produced 3.1x the instructions of the same sources in one TU (15.8k vs 5.0k),
  with the same VGPR count. A one-TU build through the same `-fgpu-rdc` pipeline did not grow. The
  growth comes from per-TU pre-link optimization. Watch the instruction cache (64 KB shared per 2 CUs).
- A swapped attention stage needed 704 B of scratch when linked, but 0 B in a one-TU build. Seam cost
  depends on the package.
- Opaque stages (noinline, as if binary-only without IR) put the cross-stage structs in scratch
  (720 B). Never ship stages without LTO IR inside a megakernel.
Runtime, same inputs, 3 processes x 60 interleaved rounds (noise floor 0.5%; outputs bitwise equal):
- split refactor compiled as one TU vs the original: +-0.5% (free).
- same one TU through the -fgpu-rdc pipeline: about +1.5%.
- separate TUs + LTO link: +1.6 to +2.4% (PipeLink reports <=0.2% on NVIDIA; on ROCm the seam is not free).
- opaque stages: 1.97-1.99x slower.
- swapped attention stage: +1.5% as one TU, but 1.82x slower when linked. The codegen gate's scratch
  growth (0 -> 704 B) predicted this before any GPU run.
So for every linked build:
1. `codegen_gate_amd.py resources --production onetu.so --candidate linked.so --kernel '^<kernel>$'`:
   scratch/spill growth fails.
2. Compare instruction counts (`identical` prints them) and time both builds interleaved on the same inputs.
3. If the linked build is slower, keep developing stages separately but ship the one-TU build
   (`#include` all stage files into one TU). That keeps the modularity without the seam cost.

## 4. Workflow with the other skills
1. `megakernel-phase-profiling`: pick the stage with the largest recoverable gap.
2. Write the replacement as a new package of the same slot (same contract). Microbenchmark it in
   isolation (`megakernel-microbench`): calibrate a baseline arm that runs the current package's
   code under the same launch geometry.
3. Swap it into the pipeline. The checker must pass. Then run the codegen gate, `./check_candidate`
   and the timeline. Keep the swap only if end-to-end time improves beyond the noise.
4. Log every swap (kept or killed) in dev/NOTES.md with the per-phase numbers.
