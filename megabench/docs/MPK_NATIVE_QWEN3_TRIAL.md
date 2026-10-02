# Native MPK Qwen3 graph on the MegaBench dense step

Experiment: 2026-10-02, NVIDIA B200, pinned Mirage MPK source
`reproductions/mirage-mpk` at `6ce3a6b`. This is a follow-up to
`MPK_FULL_BASELINE.md`. It attempts the **whole Qwen3 model graph** through
MPK's upstream `Qwen3Builder`, rather than running decoder layers in PyTorch.

## Result

The native graph compiled and executed the embedding, all 28 Qwen3 decoder
layers, final normalization, vocabulary projection, and argmax. The three
fresh MegaBench synthetic-weight trials produced finite logits and matched
the next token, but **failed full-output numerical correctness**. This is a
diagnostic timing, **not a valid MegaBench baseline or a passing VibeSys
comparison**.

| Trial | Logit elements outside tolerance | Max logit absolute error | K-write elements outside tolerance | V-write elements outside tolerance | Next token |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 147,184 / 151,936 | 0.352842 | 24,232 / 28,672 | 24,188 / 28,672 | Match |
| 1 | 145,666 / 151,936 | 0.276937 | 23,586 / 28,672 | 23,591 / 28,672 | Match |
| 2 | 144,811 / 151,936 | 0.213933 | 23,326 / 28,672 | 23,364 / 28,672 | Match |

The K/V writes are BF16 and were checked with MegaBench's
`atol=0.002, bf16_rtol=0.008`; logits use `atol=rtol=0.002`. The maximum
K/V absolute error over the three trials was 0.445313. A one-layer trace had
close QKV and attention outputs (max absolute errors 0.0112 and 0.00054),
but already exceeded full logit and K/V tolerances. With the correct gate/up
packing, the one-layer final hidden vector's maximum absolute error was
0.0288, and the full 28-layer discrepancy was much larger. Accumulated BF16
rounding is a plausible cause; this experiment does not prove it is the only
cause. The builder uses BF16 intermediate buffers, whereas MegaBench's
reference accumulates decoder operations in FP32.

The first version of this adapter packed fused gate/up weights into 32 groups.
The generated SiLU CUDA task showed the actual grid required **48 groups** for
this geometry. Correcting that layout reduced the one-layer activation's mean
absolute error from about 0.48 to 0.0038. The final three-trial results above
use the corrected layout. A separate stream synchronization fix made sure the
worker/scheduler streams completed before reading outputs; earlier unsynced
results are invalid and excluded.

## Diagnostic timings

One warmup and five CUDA-event repetitions on one B200:

| Scope | CUDA-event p50 | Host p50 | Meaning |
| --- | ---: | ---: | --- |
| MPK graph launch and completion | 1.431 ms | 1.448 ms | Inputs already attached and populated; no weight/KV copy or repacking. |
| Complete MegaBench adapter call | 28.416 ms | 28.428 ms | Fresh input, weight repacking/copy, MPK graph, output reads. |

These times were recorded despite the failed correctness check to diagnose
where cost lies. The selected VibeSys dense result was 1.946 ms, passed all
outputs, and used one GPU launch. The 1.431 ms MPK graph time **must not be
reported as faster**, because its output is wrong and excludes input binding
work. The complete 28.416 ms adapter is likewise incorrect and includes
significant MegaBench fresh-weight setup. The pinned MPK runtime uses a
preparation kernel plus separate worker and scheduler kernels for this
one-pass call, not a literal one-launch megakernel.

## Adapter and reproducibility

`megabench/probes/mpk_native_qwen3_probe.py` constructs the upstream
`Qwen3Builder` from a `MirageModelConfig` matching MegaBench's Qwen3 0.6B
geometry. It uses MPK's `online_notoken` one-pass mode so the supplied
128-token KV cache can be consumed directly rather than processing a prompt
from token zero. The adapter exposes the builder's logits and argmax scratch
buffers, which that mode normally omits, and maps MegaBench weights into
MPK's fused QKV and gate/up layouts. It copies each trial's fresh weights and
KV into the attached buffers and synchronizes MPK's worker/scheduler streams
before reading the outputs. No decoder operation is delegated to PyTorch in
the timed MPK graph; PyTorch still performs input binding and output copying
in the complete adapter call.

The installed MPK Python package omitted the upstream `qwen3/` builder
directory, so the probe adds the pinned source model directory to the MPK
model package path. No MPK source or generated data was overwritten.

Raw JSON, Slurm logs, scripts, generated CUDA, task graph JSON, and compiled
launchers are in ignored
`megabench/experiments/2026-10-02/18-00-00-mpk-native-qwen3/`.
The final result is `full-28-final-diagnostic.json` (Slurm job 3349).
The corrected one-layer trace is `one-layer-layout-fixed.json` (job 3347).
Prior failed adapter investigations are preserved separately in that directory.

To rerun on Slurm, choose a new `RESULT_NAME` to preserve these artifacts:

```bash
sbatch --mem=32G --time=00:50:00 \
  --export=ALL,LAYERS=28,RESULT_NAME=full-28-rerun-1,TRIALS=3,REPS=5 \
  megabench/experiments/2026-10-02/18-00-00-mpk-native-qwen3/run.sbatch
```

A numerically valid native MPK baseline would require an FP32-accurate path
for the decoder and output contracts, or a MegaBench fixture/tolerance that
explicitly targets MPK's BF16 arithmetic. The latter would be a different
benchmark task and was not used here.
