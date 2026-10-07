# Non-P0 synthetic tier

These tasks use seeded synthetic weights at the catalog's full model geometry,
matching the P0 tier. They do not claim checkpoint accuracy. Immutable model
revisions, configuration hashes, source hashes, and sampled native checkpoint
tensor layouts are recorded in [`model_specs.json`](../tasks/model_specs.json).
The current scoring gate is `Case.ready` in [`cases.py`](../cases.py).
Kimi-K3 is deferred at the user's request; it remains visible and disabled.

| Case | Implementation and timed boundary | Validation and current gate |
| --- | --- | --- |
| GPT-OSS-20B | All 24 layers, native MXFP4 experts, YaRN, sinks, alternating sliding/full attention, logits and KV writes | Independent upstream decoder and native-format tests; full B200 geometry passed three seeds; enabled |
| Qwen3.5-0.8B | All 24 layers, 18 GatedDeltaNet and six GQA layers, convolution/recurrent/KV state | Independent upstream recurrence and three-step decoder rollout; full B200 geometry passed three seeds; enabled |
| Gemma 3 4B vision-conditioned decode | Synthetic full SigLIP encoder, projector and multimodal prefill build the fixture; all 34 text layers are timed | Upstream vision/projector/prefill checks and image dependence; full B200 geometry passed three seeds; enabled |
| Llama 3.1 8B + EAGLE3 full iteration | Actual draft head, proposal tree, full target verification, greedy acceptance, both KV commits/rollback and next-draft features | Upstream target ancestor-mask/features tests, K=2/4/8 and partial/full/reject acceptance; full B200 K=4 run passed; enabled |
| Gemma 3 27B TP2 | All 62 layers with vocabulary/head/intermediate shards and live TP reductions | Two-rank CPU outputs match the independent serial decoder; full two-GPU rerun queued; disabled pending that check |
| Qwen3-30B-A3B TP2/EP2 | All 48 layers, contiguous EP expert ownership, TP attention/FFN and distributed greedy token | Four-rank CPU outputs match the independent serial decoder; full four-GPU rerun queued; disabled pending that check |
| DeepSeek-V3.2 TP8 | All 61 layers, block FP8 weights/activation quantization, MLA, Hadamard FP8 indexer, grouped routing and shared expert | Independent expanded MLA, YaRN, routing and quantization tests; CPU TP and candidate harness pass; full eight-GPU check queued; disabled |
| GLM-5.2-FP8 TP8 | All 78 layers, block FP8 weights, MLA, scheduled full/shared indexers and MoE | Independent expanded MLA, FP8 and routing tests; CPU TP and candidate harness pass; full eight-GPU check queued; disabled |
| Kimi-K3 TP16 | All 93 layers, 69 KDA/24 gated MLA, attention residuals, BF16 latent/shared paths and native MXFP4 routed experts | Official pinned decoder/FLA oracle, eight-step independent KDA recurrence and 16-rank CPU protocol pass; deferred by request; disabled |

The full EAGLE iteration consumes target features entering layers 2, 16 and
29, matching the pinned EAGLE implementation. Its earlier full-GPU report
predates the correction to those feature tap locations; the corrected taps
are checked against upstream hidden states on CPU. A fresh full-GPU report
is still required before using the old timings for the corrected contract.

## Reference contracts

Conventional KV writes contain this step's writes, without mutating input
caches. Compressed MLA KV/position and DSA index state are replicated across
TP ranks. Vocabulary logits are rank-local; `next_token` is the global greedy
token. Kimi convolution and V-first FP32 recurrent state are head-sharded.
Reset fixtures clear both conventional/compressed and recurrent history.

Frontier FP8 payloads and FP32 block/activation scales remain separate runtime
inputs; the reference performs activation quantization and dequantization
inside the timed call. GPT-OSS uses its native input/output expert orientation;
Kimi uses native row-major MXFP4/E8M0 groups of 32. No dequantized expert weights
or precomputed timed decoder activations are supplied to candidates.

Distributed evaluation initializes one process per device and supplies live
TP/EP handles in `case['execution']` before calling `build`. Every trial uses
a shared fresh seed, validates all output keys/dtypes/devices and input
preservation, propagates rank-local failures, and reports the slowest rank
for each timed repetition. Worker timeouts terminate the owned process group.
The one-launch budget applies per rank and includes communication kernels.
Eager references use many launches and serve as correctness/timing baselines;
passing their correctness checks does not produce a fused-candidate score.

The new families do not yet have speed-of-light traffic models. Their SOL
output explicitly reports unavailable rather than borrowing a P0 formula.

## Reproduce the checks

Use the repository environment (PyTorch 2.13.0+cu130 and Transformers 5.17.0
for the recorded checks). CPU multi-process tests need permission to
create local process-group sockets and do not require CUDA:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s megabench/tests -t . -v
.venv/bin/python -m megabench.verify_storage --output /tmp/frontier-storage.json
```

`verify_storage` counts input tensors using meta shapes; it does not measure
GPU allocation or transient execution peaks. Current native input storage
per rank is 86,026,008,968 bytes for DeepSeek TP8, 94,925,899,592 for GLM TP8,
and 108,308,019,073 for Kimi TP16. Kimi TP8 needs 205,114,366,457 bytes before
temporary allocations, exceeding a 192 GB B200's capacity.
The full unsharded native inputs also total 1,560,403,229,833 bytes, exceeding
the eight GPUs' aggregate 1,538,126,774,272 bytes before runtime buffers.
Reducing replicated storage alone therefore cannot fit this model entirely
in GPU memory here.

Download the three official files at the exact URLs recorded under Kimi's
`source_files` in `model_specs.json`, then run:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m megabench.verify_kimi_primary \
  --model-source /path/modeling_kimi_linear.py \
  --config-source /path/configuration_kimi_k3.py \
  --fla-source /path/naive.py --output /tmp/kimi-primary.json
```

The script checks every source hash before extracting the official decoder
math. CPU adapters replace GPU convolution and dispatch with PyTorch and
FLA's pinned naive KDA recurrence. The original MLA, router, latent MoE,
SiTU and attention residual blocks execute unchanged at development geometry.
Seeds 17/18 cover continuing/reset state; maximum logit errors are
3.0398369e-6 and 1.2069941e-6 in the single-step report. The newer three-step
rollouts validate greedy IDs, routing sets, native MLA writes, both KDA state
types and cache lengths after each commit; the maximum logit error across
both trajectories is 6.3478947e-6. Frontier protocol checks also default to
three committed steps. Development-geometry TP2 rollouts pass for DeepSeek,
GLM and Kimi, and Kimi's 16-rank CPU rollout passes both seeded histories.
DeepSeek and GLM also match their upstream DSA indexer implementations with
native activation quantization and partial-context selection; selected IDs
and emitted FP8 key/scale payloads agree exactly. This checks actual sparse
selection rather than relying on the short full-model context, where every
cached token fits inside the default indexer top-k budget.

Run full CUDA work inside a scheduler allocation with the required cards.
These are commands for the batch script, not direct interactive GPU runs:

```bash
.venv/bin/python -m megabench.verify_non_p0 \
  --case gptoss-step-20b-b1-s128 --case hybrid-step-qwen35-08b-b1-s128 \
  --device cuda:0 --trials 3 --output /path/single-gpu.json
.venv/bin/python -m megabench.verify_non_p0 \
  --case vl-decode-step-gemma3-4b-b1-s128 \
  --case spec-full-iteration-llama31-8b-k4 \
  --device cuda:0 --trials 3 --output /path/vision-spec.json
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m megabench.tests.distributed_probe --case distributed-step-gemma3-27b-tp2 \
  --device cuda --full --trials 2 --output /path/gemma-tp
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m megabench.tests.distributed_probe --case distributed-step-qwen3-30b-a3b-tp2-ep2 \
  --device cuda --full --trials 2 --output /path/qwen-tp-ep
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m megabench.verify_frontier --case deepseek-v32-step \
  --device cuda --full --trials 2 --output /path/deepseek
```

Use `--case glm52-step` with eight ranks or `--case kimi-k3-step` with sixteen
ranks for the other full frontier checks. Omit `--full` and use `--device cpu`
for development protocol checks. A one-GPU allocation can separately run
`python -m megabench.verify_kimi_shard --output /path/kimi-shard.json`; its
report explicitly sets `full_distributed_decode_verified` to false.
Reports use exclusive creation to preserve previous experiment data.

## Recorded hardware evidence

The local campaign is
`experiments/2026-10-06/22-39-46-nonp0-readiness/` (ignored run artifacts).
B200 jobs 4141/4144 completed successfully on PyTorch 2.13/CUDA 13.0.
Three-seed eager CUDA-event medians were 84.949–85.555 ms for GPT-OSS,
13.299–13.326 ms for Qwen3.5 and 27.218–27.351 ms for conditioned Gemma.
The earlier EAGLE run measured 61.135–61.333 ms with raw/full/one acceptance;
see the feature-tap caveat above. These are synthetic eager reference timings,
not optimized serving baselines or megakernel speedups.

Initial full P2 jobs 4153/4154 rejected only FP32 accumulation roundoff
(maximum logit errors 2.15e-5/5.79e-5) because the new cases inherited zero
tolerances. Their explicit tolerances now match the MoE/Gemma P0 tier
(atol=rtol=0.003; BF16 rtol=0.008). Corrected reruns 4157/4158 and frontier
jobs 4160/4161 are queued; these pending runs are not passing evidence.
Kimi shard job 4163 was canceled after the user deferred that task.
Job 4168 queues a fresh full EAGLE run with the corrected feature taps.
