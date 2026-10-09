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
within `noise_factor` x the reference's RMS logit error of the oracle maximum.

Over four seeds on the full dense case, alternative BF16 schedules (SDPA
attention, FP32-weighted RMSNorm, exact matmul accumulation) had 0.84-1.22x
the reference's error. `noise_factor=2` accepts them. Gross errors such as
missing state, a corrupted layer or scaled logits fail. Subtle bugs below the
BF16 noise floor are not distinguishable: a 10x RMSNorm epsilon or a 1%
attention scale error measured 1.1-1.5x. Native FP8 frontier cases have
much wider bands, because small perturbations flip power-of-two scales,
sparse-index selections and routes.

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
