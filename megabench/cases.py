"""Whole-model inference challenges for agent-written megakernels.

Only cases with an implemented oracle are ready for evaluation. Planned cases
stay visible without entering a score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    model: str
    phase: str
    params: dict[str, int]
    category: str
    suite: str
    backend: str = "cuda"
    gpus: int = 1
    tp: int = 1
    ep: int = 1
    max_gpu_launches: int = 1
    atol: float = 0.0
    rtol: float = 0.0
    bf16_rtol: float = 0.008
    # Graded error may reach this multiple of the BF16 reference's own error
    # against the FP32 oracle (see docs/REFERENCE_PRECISION.md).
    noise_factor: float = 2.0
    ready: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.family or not self.model or not self.phase:
            raise ValueError(f"invalid case {self.id}")
        if any(v < 1 for v in self.params.values()):
            raise ValueError(f"invalid dimensions in {self.id}")
        if min(self.gpus, self.tp, self.ep, self.max_gpu_launches) < 1:
            raise ValueError(f"invalid execution geometry in {self.id}")
        if min(self.atol, self.rtol, self.bf16_rtol) < 0 or self.noise_factor < 1:
            raise ValueError(f"invalid tolerance in {self.id}")
        if self.suite not in ("p0", "p1", "p2", "p3"):
            raise ValueError(f"invalid priority in {self.id}")

    def to_dict(self) -> dict:
        return asdict(self)


# Families whose cases time one layer rather than a whole model step.
LAYER_FAMILIES = frozenset({"kimi_k3_layer", "megamoe_layer"})

DEEPSEEK_V32 = {"batch": 1, "context": 128, "layers": 61,
                "experts": 256, "topk": 8, "hidden": 7168, "q_heads": 128,
                "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_dim": 128,
                "qk_rope_dim": 64, "v_head_dim": 128, "intermediate": 2048,
                "dense_intermediate": 18432, "first_dense": 3, "shared_experts": 1,
                "vocab": 129280, "index_heads": 64, "index_dim": 128,
                "index_topk": 2048, "router_groups": 8, "router_top_groups": 4}
DEEPSEEK_NOTE = ("Synthetic FP8-weight MLA/DSA TP8 decode with TileRT 0.1.6 numerics "
                 "(BF16 caches and indexer); eight-B200 verification pending.")

CASES: tuple[Case, ...] = (
    Case("dense-step-qwen3-06b-b1-s128", "dense_step", "Qwen/Qwen3-0.6B",
         "decode", {"batch": 1, "context": 128, "layers": 28,
                    "hidden": 1024, "q_heads": 16, "kv_heads": 8,
                    "head_dim": 128, "intermediate": 3072, "vocab": 151936},
         "dense_llm", "p0", ready=True, atol=0.002, rtol=0.002,
         note="Full 28-layer synthetic-weight decode; checkpoint tier pending."),
    Case("moe-step-qwen3-30b-a3b-b1-s128", "moe_step",
         "Qwen/Qwen3-30B-A3B", "decode",
         {"batch": 1, "context": 128, "layers": 48, "hidden": 2048,
          "q_heads": 32, "kv_heads": 4, "head_dim": 128,
          "experts": 128, "topk": 8, "intermediate": 768, "vocab": 151936},
         "routed_moe", "p0", ready=True, atol=0.003, rtol=0.003,
         note="Full 48-layer synthetic-BF16 MoE decode; checkpoint tier pending."),
    Case("quant-step-gemma3-4b-w8-b1-s128", "quant_step",
         "google/gemma-3-4b-it", "decode",
         {"batch": 1, "context": 128, "layers": 34, "hidden": 2560,
          "q_heads": 8, "kv_heads": 4, "head_dim": 256,
          "intermediate": 10240, "vocab": 262144,
          "bits": 8, "local_window": 1024}, "quantized_llm", "p0",
         ready=True, atol=0.003, rtol=0.003,
         note="Full synthetic Gemma 3 W8A16 decode; checkpoint tier pending."),
    Case("quant-step-gemma3-4b-w4-b1-s128", "quant_step",
         "google/gemma-3-4b-it", "decode",
         {"batch": 1, "context": 128, "layers": 34, "hidden": 2560,
          "q_heads": 8, "kv_heads": 4, "head_dim": 256,
          "intermediate": 10240, "vocab": 262144,
          "bits": 4, "local_window": 1024}, "quantized_llm", "p0",
         ready=True, atol=0.003, rtol=0.003,
         note="Full synthetic Gemma 3 W4A16 decode; checkpoint tier pending."),
    Case("spec-target-step-llama31-8b-k4", "spec_target_step",
         "meta-llama/Llama-3.1-8B-Instruct + EAGLE3", "verify",
         {"batch": 1, "context": 128, "layers": 32, "draft_depth": 4,
          "hidden": 4096, "q_heads": 32, "kv_heads": 8,
          "head_dim": 128, "intermediate": 14336,
          "vocab": 128256, "draft_vocab": 32000},
         "speculative_decoding", "p0", ready=True, atol=0.003, rtol=0.003,
         note="Full synthetic Llama target verification; linear EAGLE3-head proposals."),
    Case("gptoss-step-20b-b1-s128", "gptoss_step", "openai/gpt-oss-20b",
         "decode", {"batch": 1, "context": 128, "layers": 24,
                    "experts": 32, "topk": 4, "sliding_window": 128,
                    "hidden": 2880, "q_heads": 64, "kv_heads": 8,
                    "head_dim": 64, "intermediate": 2880, "vocab": 201088},
         "routed_moe", "p1", ready=True, atol=0.003, rtol=0.003,
         note="Full synthetic native-MXFP4 decode; checkpoint tier pending."),
    Case("hybrid-step-qwen35-08b-b1-s128", "hybrid_step",
         "Qwen/Qwen3.5-0.8B", "decode",
         {"batch": 1, "context": 128, "layers": 24,
          "linear_layers": 18, "full_attention_layers": 6,
          "hidden": 1024, "intermediate": 3584, "vocab": 248320,
          "q_heads": 8, "kv_heads": 2, "head_dim": 256, "rotary_dim": 64,
          "full_attention_interval": 4, "linear_key_heads": 16,
          "linear_value_heads": 16, "linear_key_dim": 128,
          "linear_value_dim": 128, "conv_kernel": 4},
         "hybrid_llm", "p1", ready=True, atol=0.003, rtol=0.003,
         note="Full synthetic GatedDeltaNet/GQA decode, reset and continuing state."),
    Case("vl-decode-step-gemma3-4b-b1-s128", "vl_decode_step",
         "google/gemma-3-4b-it", "conditioned_decode",
         {"batch": 1, "context": 128, "layers": 34, "images": 1,
          "hidden": 2560, "intermediate": 10240, "q_heads": 8,
          "kv_heads": 4, "head_dim": 256, "vocab": 262144,
          "local_window": 1024, "image_tokens": 256, "image_size": 896,
          "patch_size": 14, "vision_hidden": 1152, "vision_layers": 27,
          "vision_intermediate": 4304, "vision_heads": 16},
         "vision_language", "p1", ready=True, atol=0.003, rtol=0.003,
         note="Synthetic SigLIP/projector/prefill fixture; timed BF16 conditioned decode."),
    Case("spec-full-iteration-llama31-8b-k4", "spec_full_iteration",
         "meta-llama/Llama-3.1-8B-Instruct + EAGLE3", "spec_iteration",
         {"batch": 1, "context": 128, "layers": 32, "draft_depth": 4,
          "hidden": 4096, "q_heads": 32, "kv_heads": 8, "head_dim": 128,
          "intermediate": 14336, "vocab": 128256, "draft_vocab": 32000,
          "tree_width": 1},
         "speculative_decoding", "p2", ready=True, atol=0.003, rtol=0.003,
         note="Full synthetic EAGLE3 draft, Llama verification, and both KV commits."),
    Case("distributed-step-gemma3-27b-tp2", "distributed_step",
         "google/gemma-3-27b-it", "decode",
         {"batch": 1, "context": 128, "layers": 62,
          "hidden": 5376, "intermediate": 21504, "q_heads": 32,
          "kv_heads": 16, "head_dim": 128, "vocab": 262144,
          "local_window": 1024, "query_pre_attn_scalar": 168},
         "tensor_parallel", "p2", ready=True, gpus=2, tp=2, atol=0.003, rtol=0.003,
         note="Full synthetic TP decode; verified on two B200 GPUs."),
    Case("distributed-step-qwen3-30b-a3b-tp2-ep2", "distributed_step",
         "Qwen/Qwen3-30B-A3B", "decode",
         {"batch": 1, "context": 128, "layers": 48,
          "experts": 128, "topk": 8, "hidden": 2048, "intermediate": 768,
          "q_heads": 32, "kv_heads": 4, "head_dim": 128, "vocab": 151936},
         "expert_parallel", "p2", ready=True, gpus=4, tp=2, ep=2, atol=0.003, rtol=0.003,
         note="Full synthetic TP/EP decode; verified on four B200 GPUs."),
    Case("deepseek-v32-step", "deepseek_v32_step",
         "deepseek-ai/DeepSeek-V3.2", "decode", DEEPSEEK_V32,
         "frontier_moe", "p3", ready=True, gpus=8, tp=8, atol=0.003, rtol=0.003,
         note=DEEPSEEK_NOTE),
    Case("glm52-step", "glm52_step", "zai-org/GLM-5.2-FP8", "decode",
         {"batch": 1, "context": 128, "layers": 78,
          "experts": 256, "topk": 8, "hidden": 6144, "q_heads": 64,
          "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_dim": 192,
          "qk_rope_dim": 64, "v_head_dim": 256, "intermediate": 2048,
          "dense_intermediate": 12288, "first_dense": 3, "shared_experts": 1,
          "vocab": 154880, "index_heads": 32, "index_dim": 128,
          "index_topk": 2048, "router_groups": 1, "router_top_groups": 1},
         "frontier_moe", "p3", ready=True, gpus=8, tp=8, atol=0.003, rtol=0.003,
         note="Native synthetic FP8 DSA TP8 decode; verified on eight B200 GPUs."),
    Case("glm53-flash-step", "glm53_flash_step", "zai-org/GLM-5.3-Flash", "decode",
         {"batch": 1, "context": 128, "layers": 45, "hidden": 4096,
          "hc_mult": 4, "hc_sinkhorn_iters": 20, "linear_heads": 64,
          "linear_head_dim": 128, "conv_kernel": 4, "q_heads": 64,
          "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_dim": 256,
          "v_head_dim": 256, "index_heads": 32, "index_dim": 128,
          "index_topk": 2048, "index_kpool": 4, "experts": 288, "topk": 8,
          "intermediate": 2048, "dense_intermediate": 12288, "first_dense": 3,
          "shared_experts": 1, "vocab": 154880},
         "frontier_hybrid", "p3", ready=True, gpus=4, tp=4, atol=0.003, rtol=0.003,
         note="Native synthetic FP8 mHC/KDA/NoPE-DSA TP4 text decode; verified on four B200 GPUs."),
    Case("kimi-k3-kda-layer-ep8", "kimi_k3_layer", "moonshotai/Kimi-K3",
         "decode", {"batch": 8, "context": 128, "layers": 93, "layer": 61,
                    "experts": 896, "topk": 16, "hidden": 7168,
                    "latent_hidden": 3584, "intermediate": 3072, "first_dense": 1,
                    "shared_experts": 2, "q_heads": 96, "head_dim": 128,
                    "q_lora_rank": 1536, "kv_lora_rank": 512,
                    "qk_nope_dim": 128, "qk_rope_dim": 64,
                    "v_head_dim": 128, "conv_kernel": 4, "attn_res_block_size": 12},
         "expert_parallel", "p3", ready=True, gpus=8, ep=8, atol=0.003, rtol=0.003,
         note="One KDA MoE layer: DP attention over 8x8 sequences, EP8 MXFP4 experts via all-to-all; verified on eight B200 GPUs."),
    Case("kimi-k3-mla-layer-ep8", "kimi_k3_layer", "moonshotai/Kimi-K3",
         "decode", {"batch": 8, "context": 128, "layers": 93, "layer": 63,
                    "experts": 896, "topk": 16, "hidden": 7168,
                    "latent_hidden": 3584, "intermediate": 3072, "first_dense": 1,
                    "shared_experts": 2, "q_heads": 96, "head_dim": 128,
                    "q_lora_rank": 1536, "kv_lora_rank": 512,
                    "qk_nope_dim": 128, "qk_rope_dim": 64,
                    "v_head_dim": 128, "conv_kernel": 4, "attn_res_block_size": 12},
         "expert_parallel", "p3", ready=True, gpus=8, ep=8, atol=0.003, rtol=0.003,
         note="One gated MLA MoE layer: DP attention over 8x8 sequences, EP8 MXFP4 experts via all-to-all; verified on eight B200 GPUs."),
    Case("kimi-k3-step", "kimi_k3_step", "moonshotai/Kimi-K3",
         "decode", {"batch": 1, "context": 128, "layers": 93,
                    "experts": 896, "topk": 16, "hidden": 7168,
                    "latent_hidden": 3584, "intermediate": 3072,
                    "dense_intermediate": 33792, "first_dense": 1,
                    "shared_experts": 2, "q_heads": 96, "head_dim": 128,
                    "q_lora_rank": 1536, "kv_lora_rank": 512,
                    "qk_nope_dim": 128, "qk_rope_dim": 64,
                    "v_head_dim": 128, "vocab": 163840,
                    "conv_kernel": 4, "attn_res_block_size": 12},
         "frontier_hybrid", "p3", gpus=16, tp=16, atol=0.003, rtol=0.003,
         note="Deferred by request; native synthetic KDA/MLA TP16 decode needs 16-GPU validation."),
    # Past index_topk=2048 cached tokens, DSA selection is sparse; these
    # contexts match TileRT's long-context decode regime.
    *(Case(f"deepseek-v32-step-ctx{context // 1024}k", "deepseek_v32_step",
           "deepseek-ai/DeepSeek-V3.2", "decode", DEEPSEEK_V32 | {"context": context},
           "frontier_moe", "p3", ready=True, gpus=8, tp=8, atol=0.003, rtol=0.003,
           note=DEEPSEEK_NOTE) for context in (4096, 32768)),
    Case("tts-frame-step-csm-1b-b1-s128", "tts_frame_step", "sesame/csm-1b", "audio_frame",
         {"batch": 1, "context": 128, "layers": 16, "hidden": 2048, "q_heads": 32,
          "kv_heads": 8, "head_dim": 64, "intermediate": 8192, "text_vocab": 128256,
          "codebooks": 32, "audio_vocab": 2051, "depth_layers": 4, "depth_hidden": 1024,
          "depth_q_heads": 8, "depth_kv_heads": 2, "depth_head_dim": 128,
          "depth_intermediate": 8192},
         "speech_generation", "p1", ready=True, atol=0.003, rtol=0.003,
         note="Synthetic CSM backbone decode plus 31 sequential depth-decoder codebooks; verified on B200."),
    Case("vla-action-step-pi05-b1", "vla_action_step", "lerobot/pi05_base", "action_chunk",
         {"batch": 1, "images": 3, "image_tokens": 256, "image_size": 224, "patch_size": 14,
          "text_tokens": 200, "vision_hidden": 1152, "vision_layers": 27,
          "vision_intermediate": 4304, "vision_heads": 16, "layers": 18, "hidden": 2048,
          "intermediate": 16384, "q_heads": 8, "kv_heads": 1, "head_dim": 256,
          "expert_hidden": 1024, "expert_intermediate": 4096, "horizon": 50,
          "action_dim": 32, "denoise_steps": 10},
         "vision_language_action", "p2", ready=True, atol=0.003, rtol=0.003,
         note="Synthetic PaliGemma prefix fixture; timed 10-step flow-matching action expert; verified on B200."),
    Case("world-frame-step-waypoint15-1b-f128", "world_frame_step", "Overworld/Waypoint-1.5-1B",
         "latent_frame",
         {"batch": 1, "frame_index": 128, "layers": 24, "hidden": 2048, "q_heads": 32,
          "kv_heads": 16, "head_dim": 64, "mlp_ratio": 4, "latent_channels": 32,
          "latent_height": 32, "latent_width": 64, "patch": 2, "tokens_per_frame": 512,
          "local_window": 16, "global_window": 128, "global_period": 4,
          "global_dilation": 8, "buttons": 256, "control_period": 3, "denoise_steps": 4},
         "world_model", "p2", ready=True, atol=0.003, rtol=0.003,
         note="Synthetic causal DiT frame: four denoise passes plus cache commit; verified on B200."),
    # DeepGEMM/FlashInfer MegaMoE geometry: the routed experts of one
    # DeepSeek-V4-Pro MoE layer, dispatch through combine, at large batch.
    *(Case(f"megamoe-layer-deepseek-v4-pro-ep8-t{tokens}", "megamoe_layer",
           "deepseek-ai/DeepSeek-V4-Pro", "moe_layer",
           {"tokens": tokens, "experts": 384, "topk": 6, "hidden": 7168,
            "intermediate": 3072, "act_block": 32, "swiglu_limit": 10},
           "expert_parallel", "p3", ready=True, gpus=8, ep=8, atol=0.003, rtol=0.003,
           note="FP8 x MXFP4 routed-expert layer over EP8 all-to-all; verified bitwise against the serial run on eight B200 GPUs.")
      for tokens in (512, 4096)),
)


def select_cases(suite: str = "core", ids: list[str] | None = None) -> list[Case]:
    by_id = {case.id: case for case in CASES}
    if ids:
        missing = sorted(set(ids) - by_id.keys())
        if missing:
            raise ValueError(f"unknown cases: {', '.join(missing)}")
        return [by_id[id] for id in ids]
    if suite == "all":
        return list(CASES)
    if suite == "core":
        return [case for case in CASES if case.ready]
    if suite == "planned":
        return [case for case in CASES if not case.ready]
    if suite in ("p0", "p1", "p2", "p3"):
        return [case for case in CASES if case.suite == suite]
    raise ValueError(f"unknown suite {suite}")
