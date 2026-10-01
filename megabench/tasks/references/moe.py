"""Qwen3 MoE decoder used only by MegaBench's PyTorch oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .qwen3 import _rms, _rot_half, rope_tables


@dataclass
class MoeConfig:
    hidden: int = 2048
    layers: int = 48
    q_heads: int = 32
    kv_heads: int = 4
    head_dim: int = 128
    experts: int = 128
    topk: int = 8
    moe_inter: int = 768
    vocab: int = 151936
    eps: float = 1e-6
    rope_theta: float = 1e6
    max_seq: int = 2048

    # aliases so rope_tables()/Config-shaped helpers work unchanged
    @property
    def inter(self):
        return self.moe_inter


class MoeRefDecoder:
    """fp32 eager decode over the shared bf16 weights (upcast per use)."""

    def __init__(self, cfg: MoeConfig, w: dict[str, torch.Tensor], device="cuda"):
        self.cfg, self.w = cfg, w
        self.cos, self.sin = rope_tables(cfg, device)
        L, S = cfg.layers, cfg.max_seq
        self.k_cache = torch.zeros(L, S, cfg.kv_heads, cfg.head_dim, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.pos = 0

    def step(self, token: int) -> torch.Tensor:
        cfg, w = self.cfg, self.w
        D, QH, KVH, I = cfg.head_dim, cfg.q_heads, cfg.kv_heads, cfg.moe_inter
        pos = self.pos
        x = w["embed"][token].float()

        for l in range(cfg.layers):
            xn = _rms(x, w["ln1"][l].float(), cfg.eps)
            q = _rms((w["wq"][l].float() @ xn).view(QH, D), w["qn"][l].float(), cfg.eps)
            k = _rms((w["wk"][l].float() @ xn).view(KVH, D), w["kn"][l].float(), cfg.eps)
            v = (w["wv"][l].float() @ xn).view(KVH, D)
            cos, sin = self.cos[pos], self.sin[pos]
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            self.k_cache[l, pos], self.v_cache[l, pos] = k, v

            ks, vs = self.k_cache[l, : pos + 1], self.v_cache[l, : pos + 1]
            qh = q.view(KVH, QH // KVH, D)
            p = torch.softmax(torch.einsum("gqd,tgd->gqt", qh, ks) * D**-0.5, dim=-1)
            o = torch.einsum("gqt,tgd->gqd", p, vs).reshape(QH * D)
            x = x + w["wo"][l].float() @ o

            # --- MoE block ---
            xn = _rms(x, w["ln2"][l].float(), cfg.eps)
            logits = w["wrt"][l].float() @ xn                      # [E]
            top_v, top_i = torch.topk(logits, cfg.topk)
            probs = torch.softmax(top_v, dim=-1)                   # == softmax-all + renorm
            acc = torch.zeros_like(x)
            for s in range(cfg.topk):
                e = int(top_i[s])
                gate = w["wg"][l, e].float() @ xn
                up = w["wu"][l, e].float() @ xn
                h = torch.nn.functional.silu(gate) * up * probs[s]
                acc = acc + w["wd"][l, e].float() @ h
            x = x + acc

        logits = w["lm_head"].float() @ _rms(x, w["fnorm"].float(), cfg.eps)
        self.pos += 1
        return logits
