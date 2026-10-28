---
name: megakernel-regimes
description: Use when the case lists regimes, or when deciding whether to specialize the megakernel for different batch sizes or KV lengths.
---

# Megakernel regimes

A regime is one shape the kernel must serve: a batch size, a set of KV lengths, a context. When the case lists regimes, each time in the score is a weighted sum over regimes, and the speedup is the ratio of the weighted baseline time to the weighted candidate time. The weight is the share of serving time the shape accounts for in a production trace. A regime with a large weight and a long time moves the score most.

One design is rarely best everywhere. This skill tells you how to find out where it is not, and what to do about it.

## 1. Why the best design can change between regimes

Treat each of these as a question to measure. Do not assume the answer.

- Which phase dominates the step. A phase that is small in one regime can be the largest in another.
- What bounds that phase. It may be memory traffic, math, synchronization, launch of work, or load imbalance. The bound can change with the shape.
- How the work maps to the machine. Some shapes give each SM plenty of independent work. Others leave SMs idle unless you split the work along another axis.
- Which data is reused. Weights are shared across the batch. Per-sequence data is not. The balance between them moves with the shape.
- Whether sequences are uniform. Ragged KV lengths can make a static split of work unbalanced.

If none of these changes between two regimes, they probably share a best design.

## 2. Know your regimes

- If the case lists regimes, read them first. Note each id, weight, batch and KV lengths.
- If you need to derive them from production traffic, use `./trace_regimes`. It replays a request trace through a batching scheduler and writes a weighted regime file.
- Choose the scheduler flags to match the serving setup you target. Examples are the sequence limit, the token budget per step, chunked prefill, the load level and the context limit.
- Record the flags you chose and why. A different setup gives different regimes.
- Run `./trace_regimes --help` for the data source and the step-time model. The replay needs a step-time model. Prefer measured step times over a constant.
- Check how much time falls outside what the case can express. Report it. Do not hide it.

## 3. Start with one implementation

- Build one correct kernel first. Do not specialize before you have a baseline for every regime.
- Time it on every regime, not only the first one. Keep the per-regime numbers.
- Profile every regime with the phase timeline from `megakernel-phase-profiling`. Use the same method for each.
- Put the results in one table: regime, weight, total time, time per phase, gap to the floor.

## 4. Find where the picture changes

Read the table across regimes. Look for:

- the losing phase changes;
- the limiter of the losing phase changes;
- one regime is far from its floor while the others are close;
- one phase is idle on the machine in some regimes and saturated in others.

A change in any of these is a candidate for a specialized design. No change means no reason to split.

## 5. Test variants per regime

- For each candidate regime, use `megakernel-microbench` on the losing phase.
- Write the hypothesis, the keep rule and the kill rule before you measure, as that skill says.
- Test the variant on its own regime and on the neighbors. A variant that wins in one regime can lose in another.
- Use `blackwell-tensor-core-design` when the question is which engine to use, or whether to schedule work statically or dynamically. The answer can differ per regime.
- Port a variant into the kernel only after it wins in the microbenchmark.

## 6. Split only when it pays

Every variant adds code, build time and dispatch cost. Split only when the measured gain is larger than that cost.

- Rank candidates by weight times gap. A large gap in a regime with a tiny weight is not worth a variant.
- Merge regimes that share a best design. One variant for several regimes is better than one per regime.
- Drop a variant that wins by less than your noise level.
- Do not tune a regime that the case does not list or that has no weight.

## 7. Dispatch inside the kernel interface

- Choose the variant per regime inside `build` and `run`.
- Keep one launch per call. Dispatch must not add a launch.
- Make the dispatch rule explicit and cheap. Base it on the shape the call receives.
- Keep every variant correct. Check each one against the reference for every regime it serves, with the case's own checks.
- Make sure a shape that matches no rule still runs a correct path.

## 8. Re-run the replay after large speedups

Step times decide how many requests wait and how many run together. When your kernel gets much faster, the batch sizes and KV lengths of the trace change.

- Take a regime evaluation result of your kernel.
- Run `./trace_regimes` again with `--step-timings` set to that result.
- Compare the new regimes with the old ones. If the weights moved, re-rank the work by weight times gap.
- Repeat only when the step times changed by a lot.

## 9. Record evidence

Keep one record per regime in your log:

- the weight and the shape;
- the phase table of the baseline;
- the variant you tried, the microbenchmark result, and the keep or kill decision;
- the end-to-end result of the change on every regime, not only the target one;
- the regimes you merged and why.

State the final split in your report: which regimes use which variant, and what you measured to justify it.
