"""Qwen3 dense decoder used only by MegaBench's PyTorch oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Config:
    hidden: int = 1024
    layers: int = 28
    q_heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    inter: int = 3072
    vocab: int = 151936
    eps: float = 1e-6
    rope_theta: float = 1e6
    max_seq: int = 2048


def rope_tables(cfg: Config, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin[max_seq, head_dim] fp32, HF layout (freqs duplicated across halves)."""
    D = cfg.head_dim
    inv_freq = cfg.rope_theta ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    t = torch.arange(cfg.max_seq, dtype=torch.float64)
    f = torch.outer(t, inv_freq)  # [S, D/2]
    emb = torch.cat([f, f], dim=-1)  # [S, D]
    return emb.cos().float().to(device), emb.sin().float().to(device)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class RefDecoder:
    """Eager fp32 decoder over BF16 task weights, with an fp32 KV cache."""

    def __init__(self, cfg: Config, w: dict[str, torch.Tensor], device="cuda"):
        self.cfg = cfg
        self.w_bf16 = w
        self.cos, self.sin = rope_tables(cfg, device)
        L, S = cfg.layers, cfg.max_seq
        self.k_cache = torch.zeros(L, S, cfg.kv_heads, cfg.head_dim, device=device)
        self.v_cache = torch.zeros(L, S, cfg.kv_heads, cfg.head_dim, device=device)
        self.pos = 0

    def _get(self, name, idx=None):
        t = self.w_bf16[name] if idx is None else self.w_bf16[name][idx]
        return t.float()

    def step(self, token: int) -> torch.Tensor:
        """One decode step: returns logits[vocab] fp32; advances the KV cache."""
        cfg, g = self.cfg, self._get
        D, QH, KVH = cfg.head_dim, cfg.q_heads, cfg.kv_heads
        pos = self.pos
        x = g("embed", token).clone()  # [H]

        for l in range(cfg.layers):
            # --- attention block ---
            xn = _rms(x, g("ln1", l), cfg.eps)
            q = (g("wq", l) @ xn).view(QH, D)
            k = (g("wk", l) @ xn).view(KVH, D)
            v = (g("wv", l) @ xn).view(KVH, D)
            q = _rms(q, g("qn", l), cfg.eps)
            k = _rms(k, g("kn", l), cfg.eps)
            cos, sin = self.cos[pos], self.sin[pos]
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            self.k_cache[l, pos] = k
            self.v_cache[l, pos] = v

            ks = self.k_cache[l, : pos + 1]  # [t, KVH, D]
            vs = self.v_cache[l, : pos + 1]
            qh = q.view(KVH, QH // KVH, D)  # group query heads per kv head
            scores = torch.einsum("gqd,tgd->gqt", qh, ks) * D**-0.5
            p = torch.softmax(scores, dim=-1)
            o = torch.einsum("gqt,tgd->gqd", p, vs).reshape(QH * D)
            x = x + g("wo", l) @ o

            # --- MLP block ---
            xn = _rms(x, g("ln2", l), cfg.eps)
            gate = g("wg", l) @ xn
            up = g("wu", l) @ xn
            x = x + g("wd", l) @ (torch.nn.functional.silu(gate) * up)

        logits = g("embed") @ _rms(x, g("fnorm"), cfg.eps)
        self.pos += 1
        return logits
