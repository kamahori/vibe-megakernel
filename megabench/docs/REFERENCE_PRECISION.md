# Reference precision

MegaBench executes all implemented model references with BF16 serving
precision, including draft/verification and vision paths. Synthetic
unquantized weights were already BF16; references now preserve that dtype
instead of widening the entire model to FP32.

| Operation or tensor | Precision |
| --- | --- |
| Unquantized embeddings, projections and norm weights | BF16 |
| Projection outputs, nonlinear activations and residual streams | BF16 |
| Ordinary KV caches, convolution caches and draft features | BF16 |
| Normalization statistics and softmax reductions | FP32 internally; model outputs return to BF16 |
| Positional frequencies | Higher precision; rotation coefficients and outputs use activation precision |
| Qwen3/DeepSeek/GLM/Kimi routing and weighted expert accumulation | FP32 internally; residual updates return to BF16 |
| GPT-OSS routing/expert outputs | BF16, matching its upstream eager implementation |
| Hybrid/Kimi recurrent state, decay parameters and recurrence reductions | FP32 |
| Tensor-parallel collective sums | FP32 accumulation; BF16 inputs publish BF16 results |
| Public logits | FP32 containers holding widened BF16 LM-head results |

The eager text-attention oracle rounds scores and softmax probabilities to
BF16, while computing softmax reduction in FP32. Vision attention uses
PyTorch SDPA with BF16 inputs and outputs. MLA retains FP32 score/softmax
reductions and BF16 model/latent outputs. Normalization follows each model's
weighting order: Llama/Qwen cast normalized activations before weighting;
Gemma and GPT-OSS apply weights before the final cast. GLM's sparse
index weighting projection retains its upstream FP32 exception; DeepSeek
follows TileRT (BF16 head weights, FP32-gamma normalization, FP32 rotary and
SwiGLU products, FP8 activations only for q_a/kv_a/indexer wk, FP32 logits).

Quantized inputs retain their native representation: signed INT8, packed INT4,
native FP8 with block scales, or packed MXFP4 E2M1 values with E8M0 scales.
Scales keep their original dtype. W8/W4 and MXFP4 eager fallbacks unpack to
BF16 compute weights; native FP8 projections accumulate quantized inputs and
publish BF16 outputs. Existing quantized weight payloads remain unchanged.

This follows the mixed precision pattern in
[vLLM normalization](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py),
[Transformers Qwen3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py),
[Gemma3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py),
and [GPT-OSS](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py).
Fused serving kernels can have different internal rounding boundaries; the
eager reference defines output formats and the BF16 noise scale, not a
bitwise target.

## Multimodal and MegaMoE tasks

These tasks follow their upstream serving paths:

- **CSM-1B** uses the Llama text pattern. Llama-3 RoPE frequencies (scaling
  factor 32) are computed in FP32 on the host, as Transformers initializes
  them. Codebook logits are widened BF16 head results.
- **π0.5** follows LeRobot's `PI05Pytorch` with `dtype=bfloat16`. Expert
  projections and activations are BF16. The time MLP, adaRMSNorm modulation
  layers and action in/out projections keep FP32 weights and activations, as
  do the noise, velocity and returned actions. LeRobot's BF16 cast also rounds
  the rotary `inv_freq` buffers to BF16. The reference keeps FP32 frequencies,
  as openpi's JAX model and the other MegaBench tasks do;
  `verify_pi05_primary` reports the difference between the two variants.
- **Waypoint-1.5** keeps the noise-level MLP in FP32 (upstream
  `_keep_in_fp32_modules`), as well as its RoPE frequencies and attention
  scores/softmax. Projections, residuals, caches and the Euler update are BF16.
  The weightless RMSNorm uses FP32 epsilon.
- **MegaMoE** keeps FP8 E4M3 activations with UE8M0 (`uint8` exponent)
  per-32 scales and MXFP4 experts native. Gate/up and down GEMMs accumulate in
  FP32 and publish BF16. The routing weight is applied in FP32 before the
  per-32 FP8 requantization, and combine sums the top-k BF16 rows in FP32 in
  slot order with one BF16 rounding. The requantization is part of the oracle
  too. A small perturbation of the linear-1 output can therefore move an
  element to a neighboring FP8 value or flip a block's power-of-two scale, so
  the band is wider than for unquantized cases.

## DeepSeek-V4.1-Flash DSpark iteration

The reference follows DeepSeek's `inference/model.py` (pinned in
`model_specs.json`) operation by operation: BF16 activations; FP32 RMSNorm with
the weight applied in FP32 (`eps=1e-20`); per-32-channel FP8 activation
quantization with power-of-two scales before every FP8/FP4 GEMM, FP32
accumulation and a BF16 result; FP32 2:1 compressor projections and pooling;
BF16 indexer scores over FP4-dequantized queries and keys; FP8 window KV, FP4
compressed KV (E4M3 scale per 16) and FP4 index keys (UE8M0 per 32) stored as
the dequantized BF16 values the kernels write; FP32 attention scores with a
sink and BF16 probabilities; FP32 mHC coefficients and Sinkhorn; FP32 router,
LM head, Markov head and confidence head. Two choices make the result
independent of reduction order: cross-rank sums accumulate in FP32 (upstream
sums BF16 index scores and embeddings with NCCL), and top-k ties go to the
lower index. `tests/test_dsv41.py` runs the pinned upstream code with PyTorch
stand-ins for its kernels, decoding one token at a time, and requires the same
tokens, acceptance and drafts, with every float output inside the band below.
Batching six rows changes GEMM accumulation order, which can flip a BF16 ulp
that FP8 re-quantization then amplifies, so bit equality is not expected.

Synthetic weights scale each block's residual-stream projections (`wo_b`,
expert and shared `w2`) by `1/sqrt(2 * blocks)` beyond fan-in. With unit-scale
updates the random model is chaotic: FP8 scale, route and sparse-selection
flips on BF16-level noise grow by about 10% per layer, and the reference's
logits reach cosine 0.73 to the oracle. With the scaling, the reference
drifts 3% (first layer) to 9% (last layer) relative L2 from the oracle and its
logits about 8%. That floor also bounds what output grading can see: dropping
one layer's whole MoE moves the next layer's writes by about 9%, and dropping
one attention sublayer by about 4%, so such omissions in deep layers are below
the band and need source and trace review.

The `dspark` gate (`harness.correctness.dspark_metrics`) applies `_band`'s
relative-L2 rule (against the oracle or the reference, `noise_factor=2`) to
the logits, every layer's window write, every DSpark stage's window write and
each whole per-source write. DSpark outputs read the deepest backbone state, so
their reference error is floored at the last layer's (three times that for the
five confidences). Expert IDs per layer, sparse indices per source and DSpark
expert IDs per stage must keep mean recall 0.5, 0.7 and 0.5 of the reference's
selections; legitimate schedules kept at least 0.75, 0.89 and 0.87.

Verify rows and Markov steps can tie within that noise, and a different pick
changes the bonus token and every later draft. Before grading,
`tasks.dsv41.regrade` reruns the reference and oracle forced onto the
submission's picks when they differ. Each pick must then be the reference's
or the oracle's greedy token for its row, or within `noise_factor` times the
row's relative logit error (the reference's or the submission's) times the
row's peak logit. `accepted_count` and `committed_tokens` must match exactly.
Over 24 eight-GPU seeds, the reference, the FP32 oracle and an alternative
schedule (split-K accumulation, BF16-applied norm weights, reversed attention
order) passed every trial. Zeroing one layer's KV write, scaling logits, an
off-by-one acceptance, wrong draft tokens, an inverted indexer, a router that
ignores its scores, and a drafter that ignores the target hidden all failed.

## Grading band

BF16 rounding differences compound through the model. On the full synthetic
Qwen3-0.6B decode, the BF16 reference differs from FP32 math by 3.8-5.0%
relative L2 in logits (max absolute error about 0.14). A BF16 schedule that
differs only by exactly accumulated matmuls fails 71% of logit elements under
the former elementwise `rtol=0.002`. The real Qwen3-0.6B checkpoint shows
1.5-2.5% BF16-to-FP32 logit drift, so this is a property of BF16 execution,
amplified about twofold by the synthetic weights.

`harness/correctness.py` therefore grades against an FP32 oracle
(`tasks/oracle.py`). The oracle executes the same task reference under a
dispatch mode that widens BF16 compute, casts and factories to FP32, while
quantized payloads, scales, integer tensors and data movement keep their
native types. On every single-process family, its outputs match the former
FP32 references to FP32 precision, except where those rounded outputs or
convolution inputs to BF16.

For each floating output, with errors measured against the oracle:

- relative L2 error <= `noise_factor` x reference relative L2 error + `rtol`
  (`bf16_rtol` for BF16 outputs);
- max absolute error <= `noise_factor` x reference max absolute error +
  `atol` + `rtol` x max |oracle|.

Each integer element must equal the BF16 reference or the oracle. A greedy
token on an unsharded vocabulary may also be any token whose oracle logit is
within a margin of the oracle maximum. The margin is the larger of
`noise_factor` x the reference's RMS logit error and the max absolute error
bound above for the logits, so logits that pass the band cannot fail on their
greedy token. Batched logits are checked one row at a time.

Over four seeds on the full dense case, alternative BF16 schedules (SDPA
attention, FP32-weighted RMSNorm, exact matmul accumulation) had 0.84-1.22x
the reference's error. `noise_factor=2` accepts them. Gross errors such as
missing state, a corrupted layer or scaled logits fail. Subtle bugs below the
BF16 noise floor are not distinguishable: a 10x RMSNorm epsilon or a 1%
attention scale error measured 1.1-1.5x. Native FP8 frontier cases have
much wider bands, because small perturbations flip power-of-two scales,
sparse-index selections and routes.

Cases with `Case.gate = "tilert"` (the real-weight DeepSeek-V3.2 cases) use a
per-output gate instead of this band: cosine similarity and relative L2 against
either the oracle or the BF16 reference for logits and the first four layers'
cache writes, the near-tie rule for the greedy token, and reported (ungraded)
metrics for later-layer cache writes, sparse indices and expert IDs. See
[NON_P0_VALIDATION.md](NON_P0_VALIDATION.md#deepseek-v32-real-weights-and-tilert-style-gate).

## Validation and historical results

Submission output keys and public logit dtype are unchanged. Value grading
uses the oracle band above instead of elementwise tolerances around the
reference. The contract digest changes automatically because it includes the
reference sources. Existing Claude, Codex and MPK results measure the previous
FP32 execution contract and require fresh grading for comparisons against
these references. Historical experiments are preserved.

CPU checks cover operator dtypes, input/payload preservation, exact BF16 Qwen3
agreement with upstream Transformers, BF16 upstream whole-model checks for
other available architectures, and state/rollback behavior. Distributed
serial diagnostics reproduce each TP partial projection's BF16 rounding.
They compare widened logits at BF16 precision. FP8 MLA absorption/expansion is
checked before the next quantization boundary using a 5% relative RMS bound,
because the schedules have distinct BF16 rounding and FP8 latent quantization
boundaries.

Full-size P0 GPU checks are saved separately under
`experiments/2026-10-07/bf16-reference-validation/`. They cover repeatability,
finite results, BF16 KV writes and LM-head results, and input token/cache
preservation. These checks do not regrade historical candidate kernels.
