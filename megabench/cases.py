"""Backend-neutral challenge definitions for agent-written megakernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    params: dict[str, int]
    category: str
    suite: str = "core"
    backend: str = "cuda"
    gpus: int = 1
    tp: int = 1
    ep: int = 1
    max_gpu_launches: int = 1
    atol: float = 0.0
    rtol: float = 0.0
    ready: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.family or any(v < 1 for v in self.params.values()):
            raise ValueError(f"invalid case {self.id}")
        if min(self.gpus, self.tp, self.ep, self.max_gpu_launches) < 1:
            raise ValueError(f"invalid execution geometry in {self.id}")

    def to_dict(self) -> dict:
        return asdict(self)


CASES: tuple[Case, ...] = (
    Case("stencil-b4-n128-t4", "stencil", {"batch": 4, "width": 128, "steps": 4},
         "iterative_science", suite="smoke", atol=1e-5, rtol=1e-4),
    Case("stencil-b8-n512-t8", "stencil", {"batch": 8, "width": 512, "steps": 8},
         "iterative_science", atol=1e-5, rtol=1e-4),
    Case("decoder-b1-h64-s32", "decoder", {"batch": 1, "hidden": 64,
         "context": 32, "intermediate": 128}, "dense_llm", suite="smoke",
         atol=0.03, rtol=0.03),
    Case("decoder-b4-h64-s128", "decoder", {"batch": 4, "hidden": 64,
         "context": 128, "intermediate": 128}, "dense_llm",
         atol=0.03, rtol=0.03),
    Case("moe-b4-h64-e4-k2", "moe", {"batch": 4, "hidden": 64,
         "experts": 4, "intermediate": 128, "topk": 2}, "routed_moe",
         suite="smoke", atol=0.03, rtol=0.03),
    Case("moe-b16-h128-e8-k2", "moe", {"batch": 16, "hidden": 128,
         "experts": 8, "intermediate": 256, "topk": 2}, "routed_moe",
         atol=0.03, rtol=0.03),
    Case("quant-w8-b1-h64-i128", "quant_mlp", {"batch": 1, "hidden": 64,
         "intermediate": 128, "bits": 8}, "quantized_llm", suite="smoke",
         atol=0.03, rtol=0.03),
    Case("quant-w8-b8-h128-i256", "quant_mlp", {"batch": 8, "hidden": 128,
         "intermediate": 256, "bits": 8}, "quantized_llm",
         atol=0.03, rtol=0.03),
    Case("quant-w4-b1-h64-i128", "quant_mlp", {"batch": 1, "hidden": 64,
         "intermediate": 128, "bits": 4}, "quantized_llm",
         atol=0.03, rtol=0.03),
    Case("quant-w4-b8-h128-i256", "quant_mlp", {"batch": 8, "hidden": 128,
         "intermediate": 256, "bits": 4}, "quantized_llm",
         atol=0.03, rtol=0.03),
    Case("spec-b1-k2-v128", "spec_verify", {"batch": 1, "draft_len": 2,
         "vocab": 128}, "speculative_decoding", suite="smoke", note=
         "Draft+target verification and commit; not a full model decode."),
    Case("spec-b8-k4-v256", "spec_verify", {"batch": 8, "draft_len": 4,
         "vocab": 256}, "speculative_decoding", note=
         "Accepted draft prefix and target fallback/bonus are scored exactly."),
    Case("spec-b8-k8-v256", "spec_verify", {"batch": 8, "draft_len": 8,
         "vocab": 256}, "speculative_decoding"),
    Case("decoder-tp2-b1-h128-s128", "decoder", {"batch": 1,
         "hidden": 128, "context": 128, "intermediate": 256}, "dense_llm",
         suite="planned", gpus=2, tp=2, ready=False,
         note="Multi-GPU submission protocol and communication audit pending."),
    Case("moe-ep2-b16-h128-e8", "moe", {"batch": 16, "hidden": 128,
         "experts": 8, "intermediate": 256, "topk": 2}, "routed_moe",
         suite="planned", gpus=2, ep=2, ready=False,
         note="Expert-parallel placement and collective audit pending."),
)


def select_cases(suite: str = "smoke", ids: list[str] | None = None) -> list[Case]:
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
    if suite in ("smoke", "planned"):
        return [case for case in CASES if case.suite == suite]
    raise ValueError(f"unknown suite {suite}")
