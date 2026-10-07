# MPK-assisted full-step baseline for MegaBench P0

**Later native trial:** The full upstream MPK Qwen3 builder was compiled and
run on the dense P0 geometry. Its outputs failed MegaBench's numerical check;
see `MPK_NATIVE_QWEN3_TRIAL.md` for the diagnostic timings and failure data.

Experiment: 2026-10-02. One NVIDIA B200 per case. MPK source:
`reproductions/mirage-mpk` at `6ce3a6b`; PyTorch `2.13.0+cu130`, CUDA 13.0.
The installed MPK package is in the ignored
`megabench/experiments/mpk-install.ehi7Pu/site` directory.

## Scope and result

This is a **full-task, MPK-assisted hybrid baseline**, with the GPU launch
limit explicitly relaxed. Each call consumes MegaBench's fresh input tensors
and produces every requested output. The candidate passes the unmodified
MegaBench `check_trials` full-output and input-immutability checks for all five
P0 cases, with three independent trials per case. The evaluator separately
audits launches with a limit of 100,000; it does **not** alter the canonical
one-launch limit or produce a passing canonical MegaBench grade.

PyTorch computes all decoder layers, attention, routing, quantized
dequantization, and speculative acceptance logic. MPK computes a BF16
vocabulary projection in 16 tasks: eight hidden-dimension chunks, each split
into high and low BF16 input parts. PyTorch accumulates the partial results in
FP32 and recomputes uncertain vocabulary rows with an FP32 dot product to
meet the strict logit tolerance. The fallback fraction below is from the
timed input; **69–86% of vocabulary rows were recomputed by PyTorch**. Thus
these numbers measure the complete hybrid call, not a native whole-model MPK
implementation or an isolated MPK scheduler. The result is useful as a
correct full-step integration point, but it is a weak measure of native MPK
performance.

| P0 case | Hybrid CUDA-event p50 (ms) | Latest selected VibeSys p50 (ms) | Hybrid / VibeSys | Hybrid GPU launches | FP32 vocabulary fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense Qwen3 0.6B | 15.455 | 1.946 | 7.94x | 2,008 | 86.0% |
| MoE Qwen3 30B-A3B | 53.451 | 28.717 | 1.86x | 7,503 | 69.1% |
| Gemma 3 4B W8A16 | 42.172 | 8.772 | 4.81x | 4,242 | 69.7% |
| Gemma 3 4B W4A16 | 60.159 | 3.993 | 15.07x | 5,908 | 69.7% |
| Llama 3.1 8B + EAGLE3 target | 57.628 | 12.740 | 4.52x | 3,010 | 69.3% |

Lower latency is better. The hybrid is slower in every case; its geometric
mean latency ratio is 5.46x. The VibeSys figures are the *selected* passing
full-task, one-launch grades from the completed 20-round continuation in
`megabench/experiments/2026-10-02/04-49-25-vibesys-ncu-p0-plus20/reports/campaign-summary.md`.
They were measured on B200 GPUs with a different benchmark run and three
timing repetitions, whereas the hybrid used five. They are a practical
full-task reference, not a controlled apples-to-apples measure of MPK's
native scheduler. Faster, nondefault passing VibeSys archives exist for MoE
and W4, and an earlier issue-7 W8 candidate measured 8.271 ms; the table
consistently uses the latest selected candidates.

## Correctness and measurement

All three trials in each case passed with zero mismatched output elements.
The largest absolute logit errors across those trials were 0.004248 (dense),
0.007888 (MoE), 0.008832 (W8), 0.009413 (W4), and 0.009531 (EAGLE). The
checker applies each case's elementwise `atol + rtol * abs(reference)` limit,
so a maximum absolute difference alone is not a pass/fail threshold. Token
IDs and K/V writes also matched. EAGLE passed the `eagle3_raw`, `accept_full`,
and `accept_one` scenarios, including accepted/committed counts, committed
tokens, cache length, and target features.

The timing uses one warmup and five CUDA-event repetitions on fresh synthetic
benchmark input after MPK compilation. It times the complete candidate call,
including the MPK request reset, input/weight copies to MPK's attached
buffers, its kernels, FP32 row refinement, and the PyTorch decoder. Compilation
and input generation are outside the timed region. Launch counts come from
the same candidate with the relaxed audit. MPK's reset is accessed through
`pk.init_func.__self__.init_request_func` because `PersistentKernel.__call__`
otherwise retains its request state; `attach_input` binds buffers when the
graph is built, so new input values are copied into those buffers each call.

Earlier attempts explain the precision choice. A direct MPK BF16 gate/up
projection passed an isolated tolerance check but accumulated enough error
across 28 dense layers to fail the full-task logits check. A simple BF16 MPK
vocabulary projection also failed. Splitting the vocabulary projection into
eight high/low BF16 chunks reduced error but still failed without FP32 row
refinement. An initial very conservative refinement passed while recomputing
99.95% of rows; the final heuristic reduced that to the fractions above.
An earlier EAGLE run had one wrong committed token despite tolerable logits;
refining rows near the largest logit made all three final scenarios pass.
These observations are from the saved diagnostic runs, not guarantees for
unseen input distributions. The row-selection heuristic is empirical and
has only been validated on the recorded trials.

## Artifacts and reproduction

Implementation: `megabench/probes/mpk_hybrid_baseline.py`. Evaluator:
`megabench/probes/mpk_relaxed_eval.py`. It imports the candidate and the
MegaBench checker but the candidate imports no oracle implementation.
Raw result JSON, Slurm logs, scripts, and compiled MPK launchers are in the
ignored `megabench/experiments/2026-10-02/12-00-00-mpk-full-baseline/`.
The final JSON files are `dense-final.json`, `moe-final.json`,
`gemma-w8-final.json`, `gemma-w4-final.json`, and `eagle-final-top.json`.

From the repository root, submit one case at a time or to separate available
GPUs, using a **new** result name so saved runs are never overwritten:

```bash
sbatch --time=00:20:00 \
  --export=ALL,MEGABENCH_CASE=dense-step-qwen3-06b-b1-s128,RESULT_NAME=dense-rerun-1,TRIALS=3 \
  megabench/experiments/2026-10-02/12-00-00-mpk-full-baseline/case-run.sbatch
```

The other case IDs are `moe-step-qwen3-30b-a3b-b1-s128`,
`quant-step-gemma3-4b-w8-b1-s128`, `quant-step-gemma3-4b-w4-b1-s128`, and
`spec-target-step-llama31-8b-k4`. The Slurm script sets the local MPK install,
cache, and CUDA paths. The native whole-model MPK adapter remains open work;
the earlier native compatibility investigation and component timings are in
`MPK_RELAXED_TRIAL.md`.
