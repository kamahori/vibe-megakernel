# MegaBench model-step task list

This is the active whole-model benchmark deck. The exact case IDs and readiness
states are in [`cases.py`](cases.py). The P0
`dense-step-qwen3-06b-b1-s128` case is runnable with seeded synthetic BF16
weights; the other cells are cataloged as planned and cannot score yet.
`--suite core` selects only runnable cases. A model-step case exercises every layer in its stated model
phase, from token or modality input through final logits and state updates.
The timed boundary and launch policy must be fixed *per phase*: decode and
speculative verification/iteration have different contracts.

The priority order is intentional. Full dense and MoE decode establish the
central megakernel claim; quantization and speculation change the dataflow;
Gemma's local/global attention, vision-conditioned decode, and distributed
cases test whether the approach generalizes. Frontier-scale models are P3
because their checkpoint and communication requirements need a separate
multi-GPU harness.
Architecture-shaped random weights are useful for early correctness tests,
but the final checkpoint tier must use pinned public weights and tokenizer/
processor revisions. Shape-only success and checkpoint agreement are reported
separately.

## Task deck

| Priority | Task ID / phase | Model architecture | Complete timed work | Required cells and stressor |
| --- | --- | --- | --- | --- |
| P0 | `dense-step-qwen3-06b-b1-s128` / decode | **Qwen3-0.6B**: 28-layer dense GQA decoder, 16 Q/8 KV heads, hidden 1,024. | Token embedding, every decoder layer, Q/K norm, RoPE, attention, SwiGLU, final norm, LM head, greedy token, and every layer's KV append. | **Ready:** B=1, context 128, synthetic BF16 weights and prior KV. Future cells: B=8, context 4,096, ragged lengths and page-boundary writes. |
| P0 | `moe-step` / decode | **Qwen3-30B-A3B**: 48-layer GQA MoE decoder, 128 experts and top-8 routing. | Full attention and routed experts in every layer, expert reduction, LM head, and KV updates. | B=1 and 8; skewed and balanced routing; short and long context. Use the existing tiny MoE config for development and full geometry where memory allows. |
| P0 | `quant-step` / decode | **Gemma 3 4B IT text decoder**: 34 layers, five 1,024-token local-attention layers per global-attention layer, QK norm, and GeGLU; quantize the same pinned base weights to W8A16 and W4A16. | Full text decode with quantized linear projections and both local/global KV updates; dequantization is timed. Keep tied embedding/LM head BF16 in the first version. | Matched BF16, W8A16, and W4A16 input cells; contexts on both sides of the local-window boundary. Freeze packing, group size, scales, zero points, and exceptions. |
| P0 | `spec-target-step` / EAGLE3 verification | **Llama-3.1-8B-Instruct target** with its paired **EAGLE3 draft head** (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`). | Given EAGLE3-proposed token IDs, tree parents, and target state, run the full 32-layer target with the proposal attention mask; select the accepted path and correction/bonus token, update target KV, and expose target hidden features for the next draft. Proposal generation is outside this timed task. | K=2/4/8 proposed depth; first/middle/last rejection and full acceptance; B=1/8; linear and branched proposal trees. |
| P1 | `gptoss-step` / decode | **GPT-OSS-20B**: 24-layer MoE decoder, 32 experts/top-4, alternating 128-token sliding and full attention, with native MXFP4 expert weights. | Full decode including YaRN RoPE, attention sinks, clamped gated experts, LM head, and KV updates. | B=1/8; contexts across a sliding-window boundary; skewed/balanced routes. Use `GPT_OSS_TINY` for development, then checkpoint-accurate MXFP4. |
| P1 | `hybrid-step` / decode | **Qwen3.5-0.8B text path**: 24 layers, 18 linear-attention/DeltaNet and 6 full-attention layers. | Full text decoder, recurrent state updates, conventional attention, FFNs, final logits, and both state types. | B=1/8, short/long context; reset and continuing state. Vision input is outside this text-path task. |
| P1 | `vl-decode-step` / conditioned decode | **Gemma 3 4B IT**: 27-layer SigLIP vision tower and 34-layer text decoder with five local layers per global layer. Only the text decoder is in this timed step. | Complete image-conditioned text decode through LM head and local/global KV append, starting from a reference-generated multimodal prefill state. | B=1/4, one vs several images, varied image-token counts and context. The vision tower and prefill are outside timing; report this as conditioned decode, not full vision inference. |
| P2 | `spec-full-iteration` / EAGLE3 iteration | **Llama-3.1-8B-Instruct target + the same EAGLE3 draft head** and tokenizer as `spec-target-step`. The head consumes selected target hidden states; it is not a standalone small LLM. | From current target features, run EAGLE3 draft-head passes to build the proposal tree, run full target verification, select/commit tokens, and advance both draft and target KV state. | K=2/4/8; linear and branched trees; measured acceptance buckets and natural acceptance on pinned prompts. Report proposed, accepted, and committed tokens separately. |
| P2 | `distributed-step` / multi-GPU decode | Two separate text-decoder cells: **Gemma 3 27B IT**, 62 local/global-attention layers for TP, and **Qwen3-30B-A3B**, 48 MoE layers for TP+EP. | Full-model decode with TP collectives or MoE expert dispatch/combine inside the schedule. | TP=2/4 Gemma; TP+EP Qwen MoE; B=1/8. Enable after multi-process inputs, collective traces, and per-rank correctness work. |
| P3 | `deepseek-v32-step` / frontier decode | **DeepSeek-V3.2**: 61 layers, MLA with sparse attention indexing, 256 routed experts/top-8 plus a shared expert, block FP8 weights. | Full text decode from embedding through all attention/indexer and MoE layers to logits and compressed KV update. | Multi-GPU TP+EP; B=1/8, short/long context and variable selected KV blocks. Pin the FP8 checkpoint and memory fit before scoring. |
| P3 | `glm52-step` / frontier decode | **GLM-5.2-FP8**: 78-layer GLM MoE with DSA indexer, 256 routed experts/top-8 plus a shared expert. | Full text decode including index selection, sparse attention, expert routing, LM head, and KV update. | Multi-GPU TP+EP; B=1/8, indexer boundary cases and skewed routes. Pin the indexer pattern and FP8 checkpoint before scoring. |
| P3 | `kimi-k3-step` / frontier hybrid decode | **Kimi-K3 text path**: 93 layers (69 Kimi Delta Attention, 24 gated MLA), 896 routed experts/top-16, native quantized expert weights. | Full text decode including recurrent state, MLA KV, expert routing, attention residuals, and LM head. | Memory-audited multi-GPU placement; B=1/8, reset/continuing state and long context. Vision input is outside this text-path task. |

The first implementation milestone, full `dense-step` B=1 at context 128,
is active with real layer count and final projection. Its random-weight
result must remain separate from a future pinned-checkpoint result. Next
are `moe-step`, `quant-step`, and `spec-target-step`. Small development cells
should be labeled `dev` and excluded from the score.
`vl-decode-step` begins with an image-conditioned KV fixture built by the
trusted reference prefill. It measures the full decoder step but does not
claim fusion of the vision tower or prompt prefill. The P3 cases require a
separate memory-fit and multi-rank launch audit before they can be scored.

These profiles are fixed by the model authors' published configs:
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json),
[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json),
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json),
[Gemma 3 4B/27B](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py),
and the [GPT-OSS-20B model card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf).
The larger profiles come from
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json),
[GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/main/config.json),
and [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json).
The speculative pair is the
[Llama-3.1-8B-Instruct target](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
and the [official EAGLE3 draft checkpoint](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B),
as listed by the [EAGLE project](https://github.com/SafeAILab/EAGLE#eagle-3-models-on-hugging-face).
Start with greedy acceptance and the project's pinned draft-tree setup; a
sampling variant needs its own precisely specified acceptance distribution.
Pin immutable checkpoint and processor revisions when implementing each case;
the links above identify architecture, not benchmark artifact versions.

## Contract for promoting a task to runnable

1. **State the exact call boundary.** Inputs include token IDs, positions,
   model weights, page tables, current KV/recurrent state, and EAGLE3
   proposal/tree metadata or target hidden features for speculative cases;
   outputs include logits or committed IDs, updated state/pages, and lengths.
   Prior-step EAGLE3 features are state; precomputed target logits or
   activations that bypass work in the timed call are not allowed.
2. **Freeze semantics.** Pin model and checkpoint revisions, tokenizer and
   image processor, quantization packing/scales, RoPE, attention masks,
   tie-breaking/sampling seed, and state ownership. Seeded shape tests must
   change weights, tokens, routing, context length, and page placement.
3. **Check the whole trajectory.** Compare per-step logits with explicit
   tolerance, greedy/committed IDs exactly, routing choices where exposed,
   KV/recurrent state after each step, and multi-token rollouts. Speculative
   output must equal the selected non-speculative target decoding policy;
   test rejection and rollback, not just all-accept paths.
4. **Audit actual device work.** Profile GPU dispatches per timed phase and
   inspect source/traces for hidden graph replay, cached answers, host
   computation, and omitted model stages. A one-launch requirement applies
   to a **decode or target-verification candidate** only when the selected
   architecture/backend makes that a meaningful megakernel contract.
5. **Measure comparable baselines.** Record same-input eager, CUDA Graph,
   and a competent model-serving/optimized baseline where available, all on
   the same hardware and precision. Report cold compile/load time, warm
   device latency, p50/p95, memory peak, and correct tokens/s. For speculation
   report committed tokens/s versus acceptance length; for the vision case,
   report image-conditioned decode latency. Never combine checkpoint,
   synthetic, and precision-mismatched results into one speedup.

Keep separate score groups for single-GPU decode, speculative iterations,
image-conditioned decode, and distributed decode. A case with no oracle or
input generator may appear in `megabench list` as planned, but evaluation
must return `not_implemented` and it cannot contribute to a score.

## Why these boundaries

[Cohere's serving engine](https://github.com/cohere-ai/cohere-megakernel/blob/main/BUILD_AND_RUN.md)
puts a complete attention-plus-MoE **decode** pass in its persistent kernel,
while prefill has a separate path; its README also documents ragged batches
and paged KV. [Mirage MPK](https://github.com/mirage-project/mirage) and the
[Triton-distributed megakernel demo](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md)
motivate full-model and multi-GPU cells. The
[Gemma 3's architecture](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py)
provides a different local/global attention and KV schedule; its
image-conditioned decode cell measures the text path.
[EAGLE3](https://github.com/SafeAILab/EAGLE) supplies a
target-feature-conditioned draft head and proposal tree, while
[vLLM's speculative metrics](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/)
distinguish drafted, accepted, and committed output. The retired logits-only
microcase did not measure a full inference iteration.
