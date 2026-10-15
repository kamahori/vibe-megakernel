# MPK vs. the Opus megakernel at the exact MegaBench decode step (Method 1)

Experiment: 2026-10-08, one NVIDIA B200 (148 SMs, SM clock 1965 MHz, no
throttle events), pinned Mirage MPK source `reproductions/mirage-mpk` at
`6ce3a6b` (the commit PR #10 used), CUDA 13.1, torch 2.13.0+cu130. Case
`dense-step-qwen3-06b-b1-s128`: Qwen3-0.6B, batch 1, MegaBench seeded
synthetic BF16 weights (seed 7301). Run directories (ignored by Git; numbers
below are copied from the combined `method1.md`):

- `megabench/sota/runs/20261008T211524Z-method1-rerun/`: our kernel, MPK
  CUTLASS-off slope, MPK `online_notoken`, Nsight Systems, Nsight Compute.
- `megabench/sota/runs/20261008T193044Z-method1/`: MPK CUTLASS-on slope, mbt 8,
  two-stop CUTLASS off, two separate builds, stock-demo cross-check.

## Question

MegaBench times **one decode step**: 128 cached KV positions, the new token at
position 128, attention over 129 positions, LM head and argmax included. MPK
reports a **per-token average over a whole generation** (total time divided by
prompt + generated tokens, with chunked prefill and a KV cache that grows to
about 1,100 positions). Those numbers cannot be compared. Method 1 measures one
MPK decode step at exactly the MegaBench point and compares it with our best
correct dense kernel (VibeSys/Opus 5.5 run B, `submission.py` sha `b9f00de3`).

## Result

| View | MPK | Opus run B | MPK / Opus |
| --- | ---: | ---: | ---: |
| Device step: MPK slope over positions 128–191, CUTLASS on, vs. Opus steady | **1.302 ms** [1.300, 1.304] | **0.470 ms** [0.4700, 0.4702] | **2.77x** |
| Same, MPK CUTLASS off | 1.302 ms [1.297, 1.307] | 0.470 ms | 2.77x |
| Device step at position 128 only: MPK ΔT(stop 130 − stop 129) | 1.297 ms [1.086, 1.507] | 0.470 ms | 2.76x |
| Call contract: MPK `online_notoken` call vs. Opus isolated | 1.417 ms [1.416, 1.417] | 0.476 ms [0.4756, 0.4760] | 2.98x |
| GPU kernel only (Nsight Systems): MPK worker kernel vs. Opus `qwen3_step` | 1.400 ms | 0.467 ms | 3.00x |

MPK CIs are Student-t 95% intervals over 5 rounds; ours are bootstrap 95% CIs of
the median. Our kernel passes the oracle-band accuracy grade. MPK's single
step **fails** the strict MegaBench tolerance (logits max abs error 0.34, K/V
writes up to 0.48; next token matches), as PR #10 also found, so MPK is timed
as it is.

**Correction to the first version of this write-up.** It reported ΔT =
1.342 ms [1.306, 1.376] and named CUTLASS off the best MPK configuration
(1.152 ms [1.120, 1.197]). Those CIs came from a bootstrap over pooled reps,
which ignores that reps in one round share that round's drift. The full
CUTLASS-off sweep shows the same slope as CUTLASS on, to within 0.001 ms. A single ΔT
moves by about ±0.15 ms between rounds; the slope divides that noise over 64
steps and is the headline estimate now.

| MPK offline configuration | Stop method | ΔT at position 128 [95% CI] | Per-round ΔT (ms) | Slope over stops 129–193 [95% CI] |
| --- | --- | ---: | --- | ---: |
| mbt 1, CUTLASS on (`main`) | one build, runtime stop | 1.297 [1.086, 1.507] | 1.440, 1.083, 1.150, 1.359, 1.450 | 1.302 [1.300, 1.304] |
| mbt 1, CUTLASS off (`main-nocutlass`) | one build, runtime stop | 1.288 [1.127, 1.450] | 1.121, 1.188, 1.407, 1.316, 1.411 | 1.302 [1.297, 1.307] |
| mbt 8 (the paper's setting), CUTLASS on | one build, runtime stop | 1.295 [1.222, 1.368] | 1.393, 1.307, 1.256, 1.249, 1.270 | — |
| mbt 1, CUTLASS off, two stops only | one build, runtime stop | 1.193 [0.815, 1.570] | 0.772, 1.139, 1.616, 1.149, 1.288 | — |
| mbt 1, CUTLASS on, two separate builds | one build per stop | 1.512 [1.473, 1.550] | 1.482, 1.486, 1.520, 1.511, 1.559 | — |

The per-round slopes agree to within 0.01 ms (CUTLASS on: 1.2989–1.3040;
off: 1.2951–1.3047). The two-build row is biased by about +0.21 ms (its CI
excludes the slope), which is why the method uses one build.

Cross-checks of the MPK step:

| Method | MPK time for one step |
| --- | ---: |
| Slope over stops {129, 130, 146, 162, 178, 193} (positions 128–191), `main` | 1.302 ms |
| ΔT(stop 130 − stop 129), `main` | 1.297 ms |
| Slope of MPK's **stock** `demo/qwen3/demo.py` (real weights, its own graph, artifact settings), msl 193 → 257 | 1.341 ms |

The stock demo, run exactly as `artifact_evaluation/B200/run_tgx.sh` runs the
Qwen3-0.6B batch-1 cell (msl 1088, 39-token prompt), reports **1.515 ms/token**
in the paper's metric on this machine.

## Profiles

### Nsight Systems (50 calls per side)

| Side | Kernel | Grid × block | Median GPU time |
| --- | --- | --- | ---: |
| MPK `online_notoken` | `prepare_kernel` | 128 × 128 | 2.2 µs |
| | `worker_kernel` (228 regs/thread, 219 KB dynamic smem) | 128 × 256 | 1399.6 µs |
| | `scheduler_kernel` | 20 × 128 | 1395.0 µs |
| | call span, prepare start → worker end | | 1406.2 µs |
| Opus | `qwen3_step` (cooperative) | 148 × 512 | 467.1 µs |

- The MPK worker kernel (1.400 ms) is about 0.1 ms longer than one in-kernel
  iteration (1.302 ms): each `online_notoken` launch also pays the persistent
  kernel's start-up and shutdown. The host-side call (1.417 ms) adds about
  10 µs to the 1.406 ms GPU span.
- Our kernel's GPU time (0.467 ms) is 3 µs below its steady launch-to-launch
  time (0.470 ms), so launch overhead is small on our side.

### Nsight Compute (SM clock not locked, `--clock-control none`, to match the timings)

| Metric | Opus `qwen3_step` (kernel replay, `--set full`) | MPK one call (range replay) |
| --- | ---: | ---: |
| Duration | 470 µs at 1.94 GHz | 1.79 ms range at 1.91 GHz |
| DRAM bytes read | 1.207 GB | ≈1.22 GB (from 8.9% of peak over the range) |
| DRAM throughput, % of peak | 33.6% (2.57 TB/s) | 8.9% over the range; ≈11% over the worker kernel's 1.40 ms |
| Compute (SM) throughput | 25.6% | 3.5% |
| Achieved occupancy | 25% (127 regs/thread → 16 warps/SM) | 11.7% |
| L1 / L2 hit rate | 5.5% / 8.3% | 63.9% / 19.8% |
| Top warp stalls (cycles per issued instruction) | barrier 6.8, long scoreboard 2.2, short scoreboard 1.8 (of 15.1) | not collected (range sections only) |

- Both kernels read the weights once (about 1.2 GB) and neither is bound by
  HBM bandwidth. Our kernel reaches a third of peak; MPK reaches about a
  ninth. MPK's extra time is latency between tasks (task dispatch through the
  scheduler kernel and event waits), not extra memory traffic.
- In our kernel, 45% of stall cycles are barrier waits (the grid-wide syncs
  between stages) and 74% of scheduler cycles have no eligible warp. That is
  the first place to look for our own headroom; the HBM floor for 1.2 GB is
  about 0.16 ms.
- MPK range replay measures the whole profiler range (fill kernels, prepare,
  worker and scheduler, and the gaps between them) as one result, so its
  percentages are diluted against the worker kernel alone. Per-kernel replay
  cannot profile MPK: the worker spins on events from the concurrently running
  scheduler.

Files (on `cayenne`, under the rerun directory):

| File | Use |
| --- | --- |
| `nsys/{mpk,opus}/{mpk,opus}.nsys-rep` | Nsight Systems GUI timeline |
| `nsys/{mpk,opus}/{mpk,opus}.sqlite` | SQLite export (`CUPTI_ACTIVITY_KIND_KERNEL` joined with `StringIds`) |
| `nsys/{mpk,opus}/*_cuda_gpu_trace.csv`, `nsys/{mpk,opus}.json` | per-kernel CSV and the summary used above |
| `ncu/opus/opus.ncu-rep`, `ncu/mpk/mpk-range.ncu-rep` | Nsight Compute GUI |
| `ncu/opus/opus_{raw,details}.csv`, `ncu/mpk/mpk-range_{raw,details}.csv` | every metric as CSV |

## Method

**Isolating one MPK step.** In MPK's offline scheduler
(`include/mirage/persistent_kernel/persistent_kernel.cuh`), a decode iteration
at step `s` processes the token at position `s` and attends to `s + 1`
positions (`:375-407`); the last prefill iteration produces the first generated
token (`:306-313`); a request stops when `step + 1 >= max_seq_length`
(`:319`), so the last processed position is `max_seq_length − 2`. Therefore
`T(stop = 130) − T(stop = 129)` is exactly the iteration at position 128 over
129 positions, for any prompt of 128 tokens or fewer. Launch, prefill, host
`printf`s, the final `prepare_next_batch`, and shutdown are identical in the
two runs and cancel. The worker uses a 128-token prompt (token ids passed
directly, no chat template) and `eos_token_id = −1`, and checks that every run
generated exactly `stop − 128` tokens. The slope over six stops from 129 to 193
is the mean iteration over positions 128–191 and is far less noisy.

**One build, runtime stop.** Differencing two separately compiled kernels
(`max_seq_length` 129 and 130) does not work: each compile carries its own
offset of 0.2–1.7 ms on a ~160 ms run (the two-build row above is biased by
+0.21 ms). The scheduler's stop test reads the host-side runtime field
`config.max_seq_length`, set only by `init_persistent_kernel`; the compiled
`MPK_MAX_SEQ_LENGTH` is just the token-buffer stride. The worker therefore
compiles one kernel at the largest stop and calls the launcher's `init_func`
again with each smaller stop (`OfflineMPK.set_stop`). Code, weights, KV cache,
and intermediate buffers are identical for every point; MPK's own small runtime
queues are reallocated (and the old ones leaked). This use of `init_func` is
not documented upstream; the stock-demo slope agreeing to 3% is the evidence
that it does not change the measured step.

**Timing.** Each MPK rep resets the request (`init_request_func`), then times
`pk()` with CUDA events after a device sync; in offline mode the calling stream
waits for the worker and scheduler streams. Five rounds of 30 reps per stop,
stops in a random order each round, one discarded run after each re-init,
NVML clocks logged per round. Each round gives one ΔT and one slope (from that
round's medians); the CI is a Student-t interval over the five rounds. Our
kernel is timed by the existing `megabench.sota` harness: `steady` = 50
back-to-back launches with device-side token feedback and no host sync,
`isolated` = sync + one launch, arms interleaved in shuffled rounds.

**What each view compares.** The device-step view compares MPK's in-kernel
iteration (including its per-iteration `prepare_next_batch` scheduling) with
our kernel's launch-to-launch time; it is the fairer view. The call-contract
view compares one MegaBench-style call; MPK's `online_notoken` mode needs three
kernel launches and a host synchronization per call because it does not join
its streams back to the caller.

## Caveats

- The slope is the mean step over positions 128–191, not position 128 alone.
  MPK's step grows with KV length (stock-demo slope 1.341 ms over positions
  192–255 and 1.637 ms over 256–1086, about 0.0007 ms per extra position), so its step at
  position 128 is about 0.02 ms below the slope. That moves the ratio by under
  2%. The exact-point ΔT agrees with the slope within its CI.
- MPK's greedy tokens vary between runs on these synthetic weights (near-tied
  logits); upstream issue #776 reports related nondeterminism on B200. A dense
  step's work does not depend on token values, so timing is unaffected.
- MPK uses 128 SMs as workers and the other 20 for its schedulers; the Opus
  kernel uses all 148 SMs.
- Our kernel is not bit-identical across runs (split-K `red.add` atomics);
  logits differ by at most about 5e-6 and K/V by one or two BF16 ulps.
- At this per-step level MPK (1.30 ms) is close to vLLM on this node
  (1.20–1.36 ms, `megabench/sota/runs/report-b1s128.md`, a different harness).
  This does not reproduce the paper's 1.4x claim over vLLM, but the paper's
  per-token metric also amortizes cheap chunked prefill.
- Results were measured on `main` at `7dfc6d4`; this branch is based on a later
  `main` (PR #13, oracle-band grading).

## Reproduce

```bash
# Environment (once): MPK checkout + isolated venv (MPK pins transformers 4.57.1)
git clone --recursive --branch mpk https://github.com/mirage-project/mirage reproductions/mirage-mpk
git -C reproductions/mirage-mpk checkout 6ce3a6b && git -C reproductions/mirage-mpk submodule update --init --recursive
python3.12 -m venv /raid/garv901/mpk/venv
/raid/garv901/mpk/venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
/raid/garv901/mpk/venv/bin/pip install -r reproductions/mirage-mpk/requirements.txt nvidia-ml-py pytest
(cd reproductions/mirage-mpk && /raid/garv901/mpk/venv/bin/pip install -e . --no-build-isolation --no-deps)

# Full study incl. Nsight Systems (about 13 minutes), Nsight Compute (about 2 minutes),
# and the stock-demo cross-check (about 4 minutes)
sbatch --time=00:30:00 megabench/sota/slurm/method1.sbatch --tag method1
sbatch --time=00:30:00 megabench/sota/slurm/method1.sbatch --out-dir "$PWD/megabench/sota/runs/<run>" --parts ncu
sbatch --time=00:20:00 megabench/sota/slurm/mpk_demo_check.sbatch "$PWD/megabench/sota/runs/<run>/demo-check"
.venv/bin/python -m megabench.sota.method1 report megabench/sota/runs/<run> [--from megabench/sota/runs/<earlier run>]
```

`setup.py` installs Rust with `curl | sh` when `cargo` is missing; install it
first under a cache directory (`CARGO_HOME`, `RUSTUP_HOME`) to avoid that.
`pip install -e .` with dependencies fails on the `tg4perfetto` git
requirement (`hatchling` missing); `--no-deps` after installing
`requirements.txt` avoids it.
