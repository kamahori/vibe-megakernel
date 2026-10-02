"""Batch-vectorized functional candidate; this is not a GPU megakernel."""

from __future__ import annotations

import torch

from .config import ModelConfig
from .reference import DecodeResult


def _rms_batch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def decode_candidate(
    cfg: ModelConfig, weights: dict[str, torch.Tensor], tokens: torch.Tensor,
    positions: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor],
) -> DecodeResult:
    """One bf16-weight/fp32-arithmetic decode with per-lane ragged positions."""
    if tokens.ndim != 1 or positions.shape != tokens.shape:
        raise ValueError("tokens and positions must be matching 1D tensors")
    if torch.any(positions < 0) or torch.any(positions >= cfg.max_seq):
        raise ValueError("positions must be inside the cache")
    w = {name: tensor.float() for name, tensor in weights.items()}
    keys, values = (tensor.float().clone() for tensor in cache)
    batch = tokens.numel()
    x = w["embed"].index_select(0, tokens.long())
    half = cfg.head_dim // 2
    freqs = cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim)
    angles = positions.float().unsqueeze(-1) * freqs
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).unsqueeze(1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).unsqueeze(1)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        flipped = torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)
        return tensor * cos + flipped * sin

    for layer in range(cfg.layers):
        xn = _rms_batch(x, w["ln1"][layer], cfg.eps)
        q = (xn @ w["wq"][layer].T).reshape(batch, cfg.q_heads, cfg.head_dim)
        k = (xn @ w["wk"][layer].T).reshape(batch, cfg.kv_heads, cfg.head_dim)
        v = (xn @ w["wv"][layer].T).reshape(batch, cfg.kv_heads, cfg.head_dim)
        q = rotate(_rms_batch(q, w["qn"][layer], cfg.eps))
        k = rotate(_rms_batch(k, w["kn"][layer], cfg.eps))
        keys[layer, torch.arange(batch), positions] = k
        values[layer, torch.arange(batch), positions] = v

        heads = []
        for lane in range(batch):
            length = int(positions[lane]) + 1
            kv_index = torch.arange(cfg.q_heads) // (cfg.q_heads // cfg.kv_heads)
            kh = keys[layer, lane, :length, kv_index].permute(1, 0, 2)
            vh = values[layer, lane, :length, kv_index].permute(1, 0, 2)
            score = (kh * q[lane, :, None, :]).sum(-1) * cfg.head_dim ** -0.5
            probability = torch.softmax(score, dim=-1)
            heads.append((probability.unsqueeze(-1) * vh).sum(dim=1).reshape(-1))
        x = x + torch.stack(heads) @ w["wo"][layer].T

        xn = _rms_batch(x, w["ln2"][layer], cfg.eps)
        gate = xn @ w["wg"][layer].T
        up = xn @ w["wu"][layer].T
        x = x + (torch.nn.functional.silu(gate) * up) @ w["wd"][layer].T
    logits = _rms_batch(x, w["final_norm"], cfg.eps) @ w["embed"].T
    return DecodeResult(logits, keys, values)
