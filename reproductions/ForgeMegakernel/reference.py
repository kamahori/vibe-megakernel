"""Independent eager Qwen3-style decode and float64 oracle for small cells."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .config import ModelConfig


@dataclass
class DecodeResult:
    logits: torch.Tensor  # [batch, vocab]
    keys: torch.Tensor  # [layer, batch, sequence, KV head, head dimension]
    values: torch.Tensor


def random_weights(cfg: ModelConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    h, qd, kd, inter = (
        cfg.hidden, cfg.q_heads * cfg.head_dim,
        cfg.kv_heads * cfg.head_dim, cfg.intermediate,
    )

    def matrix(*shape: int, fan_in: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) / math.sqrt(fan_in)).to(torch.bfloat16)

    def gain(*shape: int) -> torch.Tensor:
        return (1 + torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)

    return {
        "embed": matrix(cfg.vocab, h, fan_in=h),
        "ln1": gain(cfg.layers, h), "ln2": gain(cfg.layers, h),
        "qn": gain(cfg.layers, cfg.head_dim),
        "kn": gain(cfg.layers, cfg.head_dim),
        "wq": matrix(cfg.layers, qd, h, fan_in=h),
        "wk": matrix(cfg.layers, kd, h, fan_in=h),
        "wv": matrix(cfg.layers, kd, h, fan_in=h),
        "wo": matrix(cfg.layers, h, qd, fan_in=qd),
        "wg": matrix(cfg.layers, inter, h, fan_in=h),
        "wu": matrix(cfg.layers, inter, h, fan_in=h),
        "wd": matrix(cfg.layers, h, inter, fan_in=inter),
        "final_norm": gain(h),
    }


def empty_cache(cfg: ModelConfig, batch: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (cfg.layers, batch, cfg.max_seq, cfg.kv_heads, cfg.head_dim)
    return torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)


def _norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * gain


def _rope(x: torch.Tensor, position: int, theta: float) -> torch.Tensor:
    dim = x.shape[-1]
    inv = theta ** (-torch.arange(0, dim, 2, dtype=x.dtype) / dim)
    angle = position * inv
    cos = torch.cat((angle.cos(), angle.cos()))
    sin = torch.cat((angle.sin(), angle.sin()))
    half = dim // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def decode_reference(
    cfg: ModelConfig, weights: dict[str, torch.Tensor], tokens: torch.Tensor,
    positions: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype = torch.float64,
) -> DecodeResult:
    """Teacher-forced one-step decode; weights are upcast from identical bf16 bits."""
    if tokens.ndim != 1 or positions.shape != tokens.shape:
        raise ValueError("tokens and positions must be matching 1D tensors")
    if torch.any(positions < 0) or torch.any(positions >= cfg.max_seq):
        raise ValueError("positions must be inside the cache")
    w = {name: value.to(dtype) for name, value in weights.items()}
    keys, values = (tensor.to(dtype).clone() for tensor in cache)
    logits = []
    group_size = cfg.q_heads // cfg.kv_heads

    for lane in range(tokens.numel()):
        position = int(positions[lane])
        x = w["embed"][int(tokens[lane])].clone()
        for layer in range(cfg.layers):
            norm = _norm(x, w["ln1"][layer], cfg.eps)
            q = (w["wq"][layer] @ norm).reshape(cfg.q_heads, cfg.head_dim)
            k = (w["wk"][layer] @ norm).reshape(cfg.kv_heads, cfg.head_dim)
            v = (w["wv"][layer] @ norm).reshape(cfg.kv_heads, cfg.head_dim)
            q = _rope(_norm(q, w["qn"][layer], cfg.eps), position, cfg.rope_theta)
            k = _rope(_norm(k, w["kn"][layer], cfg.eps), position, cfg.rope_theta)
            keys[layer, lane, position] = k
            values[layer, lane, position] = v

            context = []
            for head in range(cfg.q_heads):
                kv_head = head // group_size
                kh = keys[layer, lane, : position + 1, kv_head]
                vh = values[layer, lane, : position + 1, kv_head]
                score = (kh @ q[head]) / math.sqrt(cfg.head_dim)
                context.append(torch.softmax(score, dim=0) @ vh)
            attention = torch.cat(context)
            x = x + w["wo"][layer] @ attention

            norm = _norm(x, w["ln2"][layer], cfg.eps)
            gate = w["wg"][layer] @ norm
            up = w["wu"][layer] @ norm
            x = x + w["wd"][layer] @ (torch.nn.functional.silu(gate) * up)
        logits.append(w["embed"] @ _norm(x, w["final_norm"], cfg.eps))
    return DecodeResult(torch.stack(logits), keys, values)
