"""Configuration-derived quantities kept outside the candidate implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 32
    layers: int = 2
    q_heads: int = 4
    kv_heads: int = 2
    head_dim: int = 8
    intermediate: int = 64
    vocab: int = 128
    max_seq: int = 16
    eps: float = 1e-6
    rope_theta: float = 1e6

    def __post_init__(self) -> None:
        if min(self.hidden, self.layers, self.q_heads, self.kv_heads,
               self.head_dim, self.intermediate, self.vocab, self.max_seq) <= 0:
            raise ValueError("all dimensions must be positive")
        if self.head_dim % 2 or self.q_heads % self.kv_heads:
            raise ValueError("head_dim must be even and q_heads divisible by kv_heads")


QWEN3_06B = ModelConfig(
    hidden=1024, layers=28, q_heads=16, kv_heads=8, head_dim=128,
    intermediate=3072, vocab=151936, max_seq=2048,
)


def weight_elements(cfg: ModelConfig) -> int:
    """Eq. 2: streamed projection and LM-head elements, not full embedding table."""
    h, d = cfg.hidden, cfg.head_dim
    return cfg.layers * (
        h * cfg.q_heads * d + 2 * h * cfg.kv_heads * d
        + cfg.q_heads * d * h + 3 * h * cfg.intermediate
    ) + cfg.vocab * h


def accessed_bytes(cfg: ModelConfig, batch: int, context: int, width: int = 2) -> int:
    """Forge Eq. 2 for bf16 weights and KV, with a gathered embedding row/lane."""
    if batch <= 0 or not 0 <= context <= cfg.max_seq or width <= 0:
        raise ValueError("invalid batch, context, or element width")
    return width * (
        weight_elements(cfg)
        + 2 * cfg.layers * cfg.kv_heads * cfg.head_dim * context * batch
        + batch * cfg.hidden
    )


def config_hash(cfg: ModelConfig, batch: int, context: int, width: int = 2) -> str:
    payload = {"model": asdict(cfg), "batch": batch, "context": context, "width": width}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
