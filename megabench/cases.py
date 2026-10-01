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
    ready: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.family or not self.model or not self.phase:
            raise ValueError(f"invalid case {self.id}")
        if any(v < 1 for v in self.params.values()):
            raise ValueError(f"invalid dimensions in {self.id}")
        if min(self.gpus, self.tp, self.ep, self.max_gpu_launches) < 1:
            raise ValueError(f"invalid execution geometry in {self.id}")
        if min(self.atol, self.rtol, self.bf16_rtol) < 0:
            raise ValueError(f"invalid tolerance in {self.id}")
        if self.suite not in ("p0", "p1", "p2", "p3"):
            raise ValueError(f"invalid priority in {self.id}")

    def to_dict(self) -> dict:
        return asdict(self)


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
                    "experts": 32, "topk": 4, "sliding_window": 128},
         "routed_moe", "p1", note="Native MXFP4 checkpoint oracle pending."),
    Case("hybrid-step-qwen35-08b-b1-s128", "hybrid_step",
         "Qwen/Qwen3.5-0.8B", "decode",
         {"batch": 1, "context": 128, "layers": 24,
          "linear_layers": 18, "full_attention_layers": 6},
         "hybrid_llm", "p1", note="DeltaNet and GQA state oracle pending."),
    Case("vl-decode-step-gemma3-4b-b1-s128", "vl_decode_step",
         "google/gemma-3-4b-it", "conditioned_decode",
         {"batch": 1, "context": 128, "layers": 34, "images": 1},
         "vision_language", "p1",
         note="Trusted image-conditioned KV fixture and decoder oracle pending."),
    Case("spec-full-iteration-llama31-8b-k4", "spec_full_iteration",
         "meta-llama/Llama-3.1-8B-Instruct + EAGLE3", "spec_iteration",
         {"batch": 1, "context": 128, "layers": 32, "draft_depth": 4},
         "speculative_decoding", "p2",
         note="EAGLE3 head, target verification, and state oracle pending."),
    Case("distributed-step-gemma3-27b-tp2", "distributed_step",
         "google/gemma-3-27b-it", "decode",
         {"batch": 1, "context": 128, "layers": 62},
         "tensor_parallel", "p2", gpus=2, tp=2,
         note="Multi-process TP protocol and collective audit pending."),
    Case("distributed-step-qwen3-30b-a3b-tp2-ep2", "distributed_step",
         "Qwen/Qwen3-30B-A3B", "decode",
         {"batch": 1, "context": 128, "layers": 48,
          "experts": 128, "topk": 8},
         "expert_parallel", "p2", gpus=4, tp=2, ep=2,
         note="Multi-process TP/EP protocol and collective audit pending."),
    Case("deepseek-v32-step", "deepseek_v32_step",
         "deepseek-ai/DeepSeek-V3.2", "decode",
         {"batch": 1, "context": 128, "layers": 61,
          "experts": 256, "topk": 8},
         "frontier_moe", "p3", gpus=8, tp=8,
         note="FP8 MLA/DSA oracle, checkpoint staging, and topology audit pending."),
    Case("glm52-step", "glm52_step", "zai-org/GLM-5.2-FP8", "decode",
         {"batch": 1, "context": 128, "layers": 78,
          "experts": 256, "topk": 8},
         "frontier_moe", "p3", gpus=8, tp=8,
         note="FP8 DSA oracle, checkpoint staging, and topology audit pending."),
    Case("kimi-k3-step", "kimi_k3_step", "moonshotai/Kimi-K3",
         "decode", {"batch": 1, "context": 128, "layers": 93,
                    "experts": 896, "topk": 16},
         "frontier_hybrid", "p3", gpus=16, tp=16,
         note="KDA/MLA oracle and memory-fit topology audit pending; GPU count provisional."),
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
