"""PyTorch reference for a Qwen3-MoE-shaped decoder (batch-1 greedy decode).

Qwen3-30B-A3B geometry (verified against the HF config): hidden 2048, 48
layers, 32 Q / 4 KV heads, head_dim 128, 128 experts, top-8 routing, expert
FFN 768, RMSNorm eps 1e-6, RoPE theta 1e6, per-head QK-norm, vocab 151936.

Routing: softmax over ALL experts, take top-k, renormalise. That is identical
to softmax over just the top-k logits, which is what both this reference and
the megakernel compute (exp(x_i) / sum_{j in topk} exp(x_j)).

Weights are generated once in bf16, layer by layer, in the exact stacked
layout the megakernel consumes -- the full 30B expert stack is ~58 GB, so a
single fp32 temporary of that size would not fit.
"""

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


QWEN3_30B_A3B = MoeConfig()
# A small architecturally-identical config for fast correctness runs.
QWEN3_MOE_TINY = MoeConfig(
    hidden=1024, layers=4, q_heads=16, kv_heads=4, experts=16, topk=4,
    moe_inter=256, vocab=8192,
)  # fmt: skip


def make_moe_weights(cfg: MoeConfig, seed: int = 0, device="cuda") -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    H, L, I, D, E = cfg.hidden, cfg.layers, cfg.moe_inter, cfg.head_dim, cfg.experts
    QD, KD = cfg.q_heads * D, cfg.kv_heads * D

    def randw(*shape, fan_in):
        return (torch.randn(*shape, generator=g, device=device) * fan_in**-0.5).to(torch.bfloat16)

    def norm_w(*shape):
        return (1.0 + 0.1 * torch.randn(*shape, generator=g, device=device)).to(torch.bfloat16)

    def big(shape, fan_in):
        """Fill a huge [L, E, ...] tensor layer-by-layer to bound the fp32 temp."""
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        for l in range(shape[0]):
            out[l] = (
                torch.randn(*shape[1:], generator=g, device=device) * fan_in**-0.5
            ).to(torch.bfloat16)
        return out

    w = {
        "ln1": norm_w(L, H), "ln2": norm_w(L, H), "fnorm": norm_w(H),
        "wq": randw(L, QD, H, fan_in=H), "wk": randw(L, KD, H, fan_in=H),
        "wv": randw(L, KD, H, fan_in=H), "wo": randw(L, H, QD, fan_in=QD),
        "qn": norm_w(L, D), "kn": norm_w(L, D),
        "wrt": randw(L, E, H, fan_in=H),                      # router
        "wg": big((L, E, I, H), H), "wu": big((L, E, I, H), H),  # expert gate/up
        "wd": big((L, E, H, I), I),                              # expert down
        "embed": (torch.randn(cfg.vocab, H, generator=g, device=device) * 0.02).to(
            torch.bfloat16
        ),
    }  # fmt: skip
    torch.cuda.synchronize()
    return w


class MoeRefDecoder:
    """fp32 eager decode over the shared bf16 weights (upcast per use)."""

    def __init__(self, cfg: MoeConfig, w: dict[str, torch.Tensor], device="cuda"):
        self.cfg, self.w = cfg, w
        self.cos, self.sin = rope_tables(cfg, device)
        L, S = cfg.layers, cfg.max_seq
        self.k_cache = torch.zeros(L, S, cfg.kv_heads, cfg.head_dim, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.pos = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
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

        # Qwen3-30B-A3B uses an independent LM head. Older local experiments
        # supplied only a tied embedding, so retain that fallback for them.
        head = w.get("lm_head", w["embed"])
        logits = head.float() @ _rms(x, w["fnorm"].float(), cfg.eps)
        self.pos += 1
        return logits

    def generate(self, start_token: int, n_steps: int):
        tok, out, logits = start_token, [], None
        for _ in range(n_steps):
            logits = self.step(tok)
            tok = int(logits.argmax())
            out.append(tok)
        return out, logits


def active_bytes(cfg: MoeConfig) -> int:
    """DRAM bytes/token: attention + router + top-k experts + LM head, bf16."""
    H, I, D = cfg.hidden, cfg.moe_inter, cfg.head_dim
    QD, KD = cfg.q_heads * D, cfg.kv_heads * D
    attn = (QD + 2 * KD) * H + H * QD
    router = cfg.experts * H
    experts = cfg.topk * (2 * I * H + H * I)
    return 2 * (cfg.layers * (attn + router + experts) + cfg.vocab * H)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = QWEN3_MOE_TINY
    w = make_moe_weights(cfg)
    ref = MoeRefDecoder(cfg, w)
    toks, lg = ref.generate(1234, 4)
    print("tiny MoE tokens:", toks)
    print("active bytes/token (30B-A3B): %.2f GB" % (active_bytes(QWEN3_30B_A3B) / 1e9))
