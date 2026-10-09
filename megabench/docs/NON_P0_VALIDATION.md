# Non-P0 synthetic tier

These tasks use seeded synthetic weights at the catalog's full model geometry,
matching the P0 tier. They do not claim checkpoint accuracy. Immutable model
revisions, configuration hashes, source hashes, and sampled native checkpoint
tensor layouts are recorded in [`model_specs.json`](../tasks/model_specs.json).
The current scoring gate is `Case.ready` in [`cases.py`](../cases.py).
Full Kimi-K3 decode is deferred at the user's request; it remains visible and
disabled. Two single-layer Kimi-K3 EP8 cases are enabled instead.

| Case | Implementation and timed boundary | Validation and current gate |
| --- | --- | --- |
| GPT-OSS-20B | All 24 layers, native MXFP4 experts, YaRN, sinks, alternating sliding/full attention, logits and KV writes | Independent upstream decoder and native-format tests; full B200 geometry passed three seeds; enabled |
| Qwen3.5-0.8B | All 24 layers, 18 GatedDeltaNet and six GQA layers, convolution/recurrent/KV state | Independent upstream recurrence and three-step decoder rollout; full B200 geometry passed three seeds; enabled |
| Gemma 3 4B vision-conditioned decode | Synthetic full SigLIP encoder, projector and multimodal prefill build the fixture; all 34 text layers are timed | Upstream vision/projector/prefill checks and image dependence; full B200 geometry passed three seeds; enabled |
| Llama 3.1 8B + EAGLE3 full iteration | Actual draft head, proposal tree, full target verification, greedy acceptance, both KV commits/rollback and next-draft features | Upstream target ancestor-mask/features tests, K=2/4/8 and partial/full/reject acceptance; full B200 K=4 run passed; enabled |
| Gemma 3 27B TP2 | All 62 layers with vocabulary/head/intermediate shards and live TP reductions | CPU and full two-B200 outputs match the independent serial decoder for both seeds; enabled |
| Qwen3-30B-A3B TP2/EP2 | All 48 layers, contiguous EP expert ownership, TP attention/FFN and distributed greedy token | CPU and full four-B200 logits, KV and global expert IDs match the independent serial decoder for both seeds; enabled |
| DeepSeek-V3.2 TP8 | All 61 layers, block FP8 weights/activation quantization, MLA, Hadamard FP8 indexer, grouped routing and shared expert | Independent expanded MLA, YaRN, routing and quantization tests; CPU TP and candidate harness pass; full eight-B200 three-step rollouts pass both seeds with exact replicated-state agreement; enabled |
| GLM-5.2-FP8 TP8 | All 78 layers, block FP8 weights, MLA, scheduled full/shared indexers and MoE | Independent expanded MLA, FP8 and routing tests; CPU TP and candidate harness pass; full eight-B200 three-step rollouts pass both seeds with exact replicated-state agreement; enabled |
| GLM-5.3-Flash TP4 | All 45 layers: mHC (4 streams, 20 Sinkhorn iterations), 34 KDA and 11 NoPE MLA/DSA layers, block FP8 weights, 288-expert MoE | Pinned transformers 5.17.0 `Glm5NextTextModel` decode matches every output for three seeds; CPU TP2/TP4 rollouts and candidate harness pass; full four-B200 three-step rollout passes with exact replicated-state agreement; enabled |
| Kimi-K3 layer EP8 (KDA layer 61, MLA layer 63) | One mid-block layer per attention variant: attention residuals, DP attention over 8 sequences per rank, EP8 MXFP4 experts with all-to-all dispatch/combine, latent norm/up and shared experts | Matches the whole-model Kimi reference's layer slice; CPU 2/8-rank all-to-all equals the serial 64-sequence batch; candidate harness passes; full eight-B200 EP8 outputs equal the serial run bit for bit for both seeds; enabled |
| Kimi-K3 TP16 | All 93 layers, 69 KDA/24 gated MLA, attention residuals, BF16 latent/shared paths and native MXFP4 routed experts | Official pinned decoder/FLA oracle, eight-step independent KDA recurrence and 16-rank CPU protocol pass; deferred by request; disabled |

The full EAGLE iteration consumes target features entering layers 2, 16 and
29, matching the pinned EAGLE implementation. The corrected taps agree with
upstream hidden states on CPU. A fresh full-B200 run also passed all three
seeded/full/one-token acceptance scenarios with these feature taps.

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
DeepSeek and GLM also pass eight-rank CPU rollouts with development dimensions
that retain full native 128-channel FP8 blocks in every TP slice.
Every rollout also checks exact agreement on raw bytes of the replicated
native state across ranks, including paired FP8 payloads and scales.
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

Use `--case glm52-step` with eight ranks, `--case glm53-flash-step` with four,
or `--case kimi-k3-step` with sixteen ranks for the other full frontier checks.
The Kimi layer cases use their own verifier, which compares the EP all-to-all
result with a serial single-rank run of all 64 sequences, at full geometry too:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m megabench.verify_kimi_layer --case kimi-k3-kda-layer-ep8 \
  --device cuda --full --trials 2 --output /path/kimi-kda-layer
```
 Omit `--full` and use `--device cpu`
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
The earlier EAGLE run measured 61.135–61.333 ms before correcting feature taps.
Fresh job 4168 passed the corrected contract and measured 60.987–61.440 ms
across the seeded/full/one-token acceptance scenarios.
These are synthetic eager reference timings,
not optimized serving baselines or megakernel speedups.

Initial full P2 jobs 4153/4154 rejected only FP32 accumulation roundoff
(maximum logit errors 2.15e-5/5.79e-5) because the new cases inherited zero
tolerances. Their explicit tolerances now match the MoE/Gemma P0 tier
(atol=rtol=0.003; BF16 rtol=0.008). Corrected full B200 jobs 4157/4158 passed
both seeds on every rank against the independent serial decoder, including
global expert IDs for Qwen. Worst-rank CUDA-event medians were
78.633–78.943 ms for Gemma TP2 and 61.536–75.282 ms for Qwen TP2/EP2.
Maximum logit errors were 2.15173e-5 and 5.93066e-5 respectively.
Non-oracle ranks peaked at 29.903 GB allocated for Gemma and 16.714 GB for
Qwen; rank zero also held the full serial oracle and peaked at 86.810 GB and
78.439 GB respectively. Profiler traces include collectives and 7,350
device events per Gemma rank and 6,766–6,926 per Qwen rank. These eager
references fail the one-launch budget, as expected; correctness readiness
does not certify a fused candidate.

Eight-B200 frontier jobs 4160/4161 completed successfully in 17:25 and 18:31.
Each rank passed both seeds and all three committed steps at contexts 128,
129 and 130, including deterministic outputs, finite values, input preservation,
runtime-token dependence and exact raw-byte agreement on replicated native
KV/index payloads and scales. The full reports retain the catalog geometry
and the timing context of 128. Independent mathematical/source oracles and
serial comparisons run at development geometry; the full GPU runs validate
execution and protocol rather than comparing against a full unsharded model.

Worst-rank eager CUDA-event medians were 490.730–663.316 ms for DeepSeek
and 435.918–572.605 ms for GLM. Native inputs occupied 86.026 GB and
94.926 GB per rank; measured allocated peaks were 86.579 GB and 95.526 GB,
with reserved peaks of 87.294 GB and 95.798 GB. Reference launch audits
recorded 63,541 and 57,579 device events per rank respectively, including
communication, so these eager baselines fail the one-launch candidate budget.
Input construction took 99.5–155.2 seconds per rank/seed and remains outside
the timed decoder step. All eight requested non-P0 tasks are now enabled;
Kimi remains deferred.

Kimi shard job 4163 was canceled after the user deferred that task.

### GLM-5.3-Flash and Kimi-K3 layers

Campaign `experiments/2026-10-08/16-36-37-glm53-flash-gpu-validation/`.
Four-B200 job 4784 passed the full GLM-5.3-Flash three-step rollout (contexts
128–130) on every rank in 12:18. Inputs occupy 80.34 GB per rank and build in
48 s; allocated peaks were 80.76 GB. The eager reference took 260.9 ms
(CUDA-event median) and recorded 42,158 device events per rank. Harness job
4785 ran the reference submission: correctness passed and the launch gate
failed, as expected, with a 310 GiB host peak.

At full depth the BF16 reference is far from the FP32 oracle: relative L2
errors are 0.79 for logits, 0.56 for MLA latents, 0.74 for index scores and
0.22–0.27 for KDA state. FP8 activation quantization amplifies BF16 rounding
differences layer by layer. Since the band is `noise_factor` times this error,
it is loose for this case; review candidate outputs rather than relying on
the band alone.

Campaign `experiments/2026-10-08/16-56-00-kimi-k3-layer-gpu-validation/`.
Eight-B200 job 4795 ran `verify_kimi_layer --full` for both layer cases with
two seeds. On every rank, the EP8 all-to-all outputs (residual prefix, state
writes and expert IDs) were bitwise equal to rank 0's serial run of all 64
sequences with all 896 experts. The reference was deterministic, preserved
its inputs and depended on the expert weights. Inputs occupy 3.29 GB (KDA) and
2.81 GB (MLA) per rank. Eager reference CUDA-event medians were 47.1–57.2 ms
and 46.0–59.4 ms. The reference submission passed harness correctness on all
ranks and failed the launch gate with 3,640–4,746 device events per rank.

An earlier job 4787 used batched projections. GPU BF16 GEMMs round
differently for 8 and 64 rows, so the EP and serial results differed. The
reference now computes dense projections one token at a time, as the whole-model
Kimi reference does. Expert GEMMs see the same rows in both layouts.
