---
name: megakernel-microbench
description: Use when a phase of the megakernel has been identified as the losing phase and you need to test optimization hypotheses on it in isolation with faithful microbenchmarks, then port only the proven winners back into the kernel.
---

# Megakernel microbenchmarks

Use this skill after `megakernel-phase-profiling` has picked a phase. The goal is to learn what limits that one phase, test ideas cheaply, and move only evidence-backed changes into the kernel.

Do not tune the whole kernel blindly. Do not trust a number from a bench that does not match the real kernel.

## 1. Start from one phase and one question

- Take the phase from the phase profile. Write down its in-kernel time and its gap to the floor from `./sol_info`.
- Pick one question at a time. Example forms: "Is this phase limited by memory, math, or synchronization?" or "Would a different engine or layout remove the gap?"
- Leave other phases alone until this question is answered.
- If the gap to the floor is small, pick a different phase.

## 2. Write the hypothesis before you measure

Put these in your log before the first run:

- Hypothesis: what you think limits the phase, or what change will help, and why.
- Expected effect: how much time you expect to save, and how that changes the whole kernel time.
- Keep rule: what result makes you port the change.
- Kill rule: what result makes you drop it.
- Noise level: the smallest difference you will treat as real. Measure the run-to-run spread of the baseline arm first, then set this level above it.

Changing the rule after you see the data is how wrong ideas survive. If you must change it, say why in the log.

## 3. Make the bench faithful

A bench that is easy to write often measures a different problem. Match the real kernel on:

- Tile and problem shapes, data types, and data layouts, including padding and swizzles.
- Launch geometry: grid size, block size, shared memory size, registers per thread, and how many CTAs are resident per SM.
- Co-running work: how many CTAs run the phase at the same time, and what the other CTAs do.
- Cache state: cold or warm L2. Think about what the previous phase leaves in L2 and what the phase needs to read from HBM. Flush or prime the cache to match.
- Synchronization pattern: the same barriers, flags, and waits as in the kernel, not a cleaner version.
- Clocks and power state: run long enough to reach steady clocks.

### Calibrate before you trust it

- Add a baseline arm that is the current code of the phase.
- Compare its time with the phase time from `./timeline_candidate`. Remember that timeline probes add some cost.
- If they disagree, fix the bench. Do not run variants until the baseline arm reproduces the in-kernel time.
- Re-check calibration when you change the geometry or the cache setup.

## 4. Find the limiter with oracle variants

Build cut-down arms of the baseline to see which part of the phase sets its time. Useful cuts:

- Loads only: keep all memory movement, remove the math.
- Math only: keep the math on data already in shared memory or registers, remove the loads.
- No epilogue: remove the output work.
- Sync only: keep barriers and waits, remove the work.
- Null work: keep synchronization and memory movement, remove the compute.

Compare each cut with the baseline and with the `./sol_info` floor. The cut that stays close to the baseline is the limiter. If the sum of the cuts is far above the baseline, the parts overlap well. If the baseline is far above every cut, the cost is in how the parts interact, for example in dependencies or lack of overlap.

Oracle arms give wrong answers. That is fine. Mark them as oracle arms and never port them.

## 5. Check correctness for every arm

- Compare each real arm with the reference, or with the baseline arm output, before you time it.
- Use a tolerance that matches the task. Check for NaN and for uninitialized output.
- Re-run the check after any edit to the arm.
- Oracle arms are exempt, but label them.

## 6. Measurement hygiene

- Warm up before timing. Use many repetitions per sample.
- Interleave arms and rotate their order between rounds, so drift in clocks or temperature does not favor one arm.
- Report the median and the spread (for example min, median, max or interquartile range). Do not report one run.
- Repeat across separate process runs. Placement on SMs and clocks can change between runs.
- Treat a difference inside the noise level as no difference.
- Note anything odd: clock changes, other jobs on the GPU, different SM placement.
- Use device-side timing (clock or event timing) that matches how the timeline measures the phase.

## 7. Work in the implementer turn, not in rounds

- Run microbenchmarks inside your turn with the workspace GPU command path. Compiling and running a bench does not start a new round and does not call the trusted evaluator.
- Run many of them. Cheap, fast experiments are the point.
- Keep benchmarks out of validation recipes. Validation recipes are only for correctness and final checks.
- Keep bench code in a `dev/` directory, not in the submission.
- Keep a short log in `dev/`. One entry per experiment: hypothesis, arms, result with spread, decision.

## 8. Port winners behind a switch

- Add the winning variant to the megakernel behind a compile-time or launch-time switch. Keep the old path until the new one is proven.
- Run `./check_candidate` for correctness and scored timing.
- Re-run the phase timeline and compare the phase with the bench prediction.
- Check that neighbor phases did not get slower.
- Remove the old path only when the end-to-end result is better and stable.

## 9. When a bench win does not transfer

A bench win that does not improve the full kernel is rejected. Find out why and write it down. Common causes:

- The real kernel already overlapped this work with another phase.
- The variant competes with neighbor phases for bandwidth, shared memory, or issue slots.
- The variant changes what is left in L2 for the next phase.
- Register or shared memory use grows and lowers occupancy or causes spills in the full kernel.
- A different number of CTAs or a different tail shape in the real launch.
- The bench was not faithful in some way you did not list. Fix the bench, then re-test.

## 10. Record negative results

- Log every killed hypothesis with the evidence and the conditions (shapes, cache state, arm settings).
- State which conditions would change the answer, so later rounds know when a retry is worth it.
- Read the log before you start a new experiment. Do not repeat a measured dead end.

## Design candidates to test

For phases shaped like matrix multiply or attention, use `blackwell-tensor-core-design` to choose candidates. Treat its designs as hypotheses. Build them in the bench, calibrate against the baseline arm, and decide by measurement.
