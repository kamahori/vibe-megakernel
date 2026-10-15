# Plan: load-centric megakernels and agent tooling

Status: draft, 2026-10-14. Nothing in this plan has been measured yet. Hardware numbers are back-of-envelope estimates to be replaced by Phase 1 measurements.

## Thesis

Most MegaBench P0 cases are low-batch decode steps, where the time is set by HBM bytes rather than FLOPs. Mainstream kernel DSLs (Triton, CuTe-DSL) are organized around tiles and tensor-core utilization within one op. A decode megakernel loses time elsewhere:

1. **Bubbles between ops.** Weights for op N+1 are not requested until op N's outputs are ready, even though **weights do not depend on the input**. Grid barriers (a few µs each, several per layer) are expensive against a step that takes roughly 100–200 µs for Qwen3-0.6B.
2. **Too few bytes in flight.** By Little's law, B200 needs about 8 TB/s × ~0.7 µs ≈ 5–6 MB outstanding, or roughly 35–40 KB per SM across 148 SMs, at all times.
3. **Imbalance.** Work is split by tiles or FLOPs, but each SM's runtime is set by the bytes it streams.

The proposed programming model treats the forward pass as a **statically known byte stream per SM**, with compute attached:

- **Streams:** weight and KV segments, assigned to SMs balanced by bytes. A producer warp issues TMA loads in global order into a deep shared-memory ring. It waits only for free ring slots, never for activations.
- **Taps:** consumers (GEMV, dequant + GEMV, split-K reduce, norm) attached to stream segments. Only taps wait on activation dependencies.
- **Events:** fine-grained ready counters on activation slices replace grid barriers.
- **SOL accounting:** every segment declares its bytes, so predicted versus measured time is available per phase.

Known limits: above about batch 8, BF16 GEMV is limited by CUDA-core throughput, so taps need tensor cores. Low-bit dequant can be ALU-bound. MoE expert addresses are known only after routing. Prefill is compute-bound and out of scope.

## Main question

Should we build a DSL, or are skills files for agents enough? The answer depends on why agents fall short of SOL today:

| Failure class | Symptom | Remedy |
|---|---|---|
| Knowledge | The agent never tries cross-op prefetch, byte balancing or deep rings | Skills file |
| Execution | The agent tries the right structure but loses the session to deadlocks, races, mbarrier phase bugs or wrong results | Primitives library |
| Search | Kernels differ structurally across sessions, with high variance and no systematic schedule tuning | DSL or compiler with a searchable schedule |

Working hypothesis: execution dominates. The phases below test this before committing to a DSL.

## Prior art to reuse

| Work | Relevance |
|---|---|
| [QuACK](https://github.com/Dao-AILab/quack) | Gets single memory-bound ops to about 90% of HBM peak in CuTe-DSL using hierarchical and cluster (DSMEM) reductions. Reuse for taps, especially split-K GEMV reductions. It does not address bubbles between ops. |
| Hazy Megakernels (`reproductions/Megakernels`) | Instruction interpreter, shared-memory pages, early weight loads. Closest existing realization of the pattern. |
| Mirage MPK (`reproductions/mirage-mpk`) | Compiler-generated task graph with an in-kernel event scheduler. |
| AutoMegaKernel (`reproductions/AutoMegaKernel`) | Agents edit behind a statically checked schedule IR. Direct prior art for the DSL-versus-skills question. |
| ThunderKittens LCSF, CUTLASS warp specialization | Producer/consumer warp roles inside a kernel. |

## Phase 0: audit agent trajectories (CPU only, about 1 day)

Goal: assign each existing agent session to the failure classes above.

- Inputs: VibeSys, plain and KernelAgent campaign records for the P0 cases (`megabench/runs/`, the campaign directories, and GitHub issues #1–#4).
- For each session, record: final status (pass, `non_megakernel`, incorrect, timeout), speedup versus CUDA Graph, fraction of HBM SOL reached, the structure attempted (barriers or events, prefetch depth, partitioning), and where wall-clock and tool calls went (debugging hangs or numerics versus tuning).
- Output: `audits/<date>-failure-taxonomy.md` with per-case counts per class and representative excerpts.
- Decision: if knowledge dominates, jump to Phase 3 and ship skills only. Otherwise continue.

## Phase 1: gap decomposition microbenchmarks (1 GPU, Slurm)

Goal: find out where time goes relative to SOL, independent of agents.

Location: `sandbox/cute_dsl/loadstream/` (sibling of `sandbox/tirx/`, per AGENTS.md). Start with `dense-step-qwen3-06b-b1-s128` at full geometry.

1. **Streaming ceiling:** a persistent kernel that reads every weight byte of the step in the planned per-SM order, with no compute and no dependencies. Sweep ring depth, TMA box size and segment size. This gives the achievable bandwidth, likely 85–92% of spec.
2. **Plus dependencies:** add the real dependency graph with dummy taps, comparing (a) grid barriers and (b) event counters with producer run-ahead. The difference from step 1 is the cost of bubbles.
3. **Plus compute:** replace dummy taps with real GEMV, attention, norm and RoPE. The difference from step 2 is compute or dequant stalls.

Then repeat for `quant-step-gemma3-4b-w4-b1-s128` (dequant pressure) and `moe-step-qwen3-30b-a3b-b1-s128` (routing-dependent addresses).

Reference points: `megabench sol` floors, the TIRx Qwen3 kernel, and the Hazy and MPK reproductions where they run.

Outputs: a results JSONL (new path, exclusive-create) and a short report. Decision rule:

- Bubbles are the largest gap: the scheduler and event layer is the priority.
- Compute is the largest gap: invest in taps (QuACK-style reductions, tensor-core taps for batch above 1, dequant).
- The streaming ceiling itself is low: investigate TMA, L2 and partitioning before anything else.

## Phase 2: measurement tool

Build it whatever Phases 0 and 1 show, because both humans and agents need it.

- A device-side `globaltimer` stamp buffer (per SM, per phase start and end, bytes issued), written to a preallocated tensor so it adds no extra launches.
- A host-side report: per-phase achieved bandwidth, an SM-by-time timeline, and the gap split into the Phase 1 categories against `megabench sol`.
- Packaging: a Python module that agents can call from their workspace, plus a CLI. It must not count as a launch in the MegaBench audit, or must be disabled during `evaluate`.

## Phase 3: primitives library and skills file

Library (CuTe-DSL first, with a CUDA header variant if agents strongly prefer raw CUDA):

- `WeightStream`: TMA ring with a producer warp, configurable depth, and per-SM segment lists.
- `Event`: activation-slice ready counters with acquire/release semantics. Replaces grid sync.
- `partition_by_bytes(segments, num_sms)`: static byte-balanced assignment.
- Tap helpers: GEMV (BF16, FP8, MXFP4, int4 dequant), split-K reduce with cluster DSMEM, RMSNorm, RoPE.
- Each primitive gets a CPU-checkable spec and a GPU unit test, plus a deadlock watchdog (timeout plus a diagnostic dump) in debug builds.

Skills file (`skills/load-centric-megakernel/SKILL.md` or the VibeSys equivalent):

- The mental model: weights run ahead, activations gate taps, balance by bytes.
- Little's-law sizing for B200.
- How to read the Phase 2 report and which knob fixes which gap.
- One worked example: a Qwen3-0.6B layer built from the library.

## Phase 4: controlled agent evaluation

Run the P0 suite under the session protocol (a fresh session and workspace per case, see `megabench/docs/SESSION_PROTOCOL.md`) in three arms, each a separate `--method`:

| Arm | Provided to the agent |
|---|---|
| A: baseline | Current workspace |
| B: skills | A + skills file + measurement tool |
| C: skills + library | B + primitives library |

Measure pass rate, geomean speedup (only when every case passes), fraction of SOL, variance across repeated sessions, and session time spent debugging versus tuning. Use at least two sessions per case per arm. Arms B and C change what is being measured, so report them as methods alongside A, not as replacements.

## Phase 5 (conditional): DSL extraction

Only start if arm C kernels converge on a common structure and Phase 4 still shows search-type failures (high variance, untuned schedules).

- Extract the converged pattern into a declarative layer: streams, taps and events in Python, compiled to the Phase 3 primitives.
- Make the schedule (segment order, partitioning, ring depth, tap variants) an explicit, searchable object, statically checked for deadlock freedom and byte coverage as in AutoMegaKernel's IR.
- Keep an escape hatch: raw tap bodies in CuTe-DSL or CUDA.
- Evaluate as arm D with the Phase 4 protocol.

## Non-goals

- Prefill and large-batch compute-bound cases.
- Replacing the TIRx sandbox. It stays a reference point.
- Changing MegaBench scoring or tolerances.

## Risks

- Shared-memory pressure: deep rings compete with tap shared memory. Phase 1 should sweep this tradeoff explicitly.
- MoE: run-ahead stops at the router. Bubbles may remain unless non-expert weights for the next layer fill the gap.
- Event counters need correct memory ordering. Get this wrong and you get silent races, so the library must own it.
- Agents may ignore the library. Phase 0 and Phase 4 transcripts show whether they adopt it.
- CuTe-DSL version churn. Pin the version in the library.

## Immediate next steps

1. Locate all P0 campaign records and start the Phase 0 audit.
2. Implement Phase 1 step 1 (the streaming ceiling) for Qwen3-0.6B and run it through Slurm.
3. Review the Phase 0 and Phase 1 results together before committing to Phase 3 scope.
