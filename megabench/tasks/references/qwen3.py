"""PyTorch reference for a Qwen3-0.6B-shaped decoder (batch-1 greedy decode).

Architecture (matches Qwen3-0.6B exactly, random weights):
  hidden 1024, 28 layers, 16 Q heads / 8 KV heads (GQA), head_dim 128,
  intermediate 3072 (SwiGLU), RMSNorm eps 1e-6, per-head QK-norm,
  RoPE theta 1e6 (rotate-half), tied embedding / lm_head.

Weights are generated ONCE in bf16 in the exact stacked layout the TIRx
megakernel consumes; this reference upcasts the same bf16 bits to fp32,
so any mismatch with the kernel is accumulation order / precision, not data.
"""

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


def make_weights(cfg: Config, seed: int = 0, device="cuda") -> dict[str, torch.Tensor]:
    """Stacked bf16 weights in the layout the megakernel reads (GPU-side RNG:
    a 30B-class model would take minutes and 128 GB of host RAM on CPU)."""
    g = torch.Generator(device=device).manual_seed(seed)
    H, L, I, D = cfg.hidden, cfg.layers, cfg.inter, cfg.head_dim
    QD, KD = cfg.q_heads * D, cfg.kv_heads * D

    def randw(*shape, fan_in):
        w = torch.randn(*shape, generator=g, device=device) * (fan_in**-0.5)
        return w.to(torch.bfloat16)

    def randn_norm(*shape):
        w = 1.0 + 0.1 * torch.randn(*shape, generator=g, device=device)
        return w.to(torch.bfloat16)

    ret = {
        "ln1": randn_norm(L, H),
        "wq": randw(L, QD, H, fan_in=H),
        "wk": randw(L, KD, H, fan_in=H),
        "wv": randw(L, KD, H, fan_in=H),
        "qn": randn_norm(L, D),
        "kn": randn_norm(L, D),
        "wo": randw(L, H, QD, fan_in=QD),
        "ln2": randn_norm(L, H),
        "wg": randw(L, I, H, fan_in=H),
        "wu": randw(L, I, H, fan_in=H),
        "wd": randw(L, H, I, fan_in=I),
        "fnorm": randn_norm(H),
        "embed": (torch.randn(cfg.vocab, H, generator=g, device=device) * 0.02).to(
            torch.bfloat16
        ),
    }
    torch.cuda.synchronize()
    return ret


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
    """fp32 eager decoder over the shared bf16 weights, with fp32 KV cache.

    lazy_cast=True keeps weights in bf16 and upcasts per use — required for
    30B-class models where a persistent fp32 copy would not fit in memory.
    Numerics are identical either way (fp32 math on the same bf16 values).
    """

    def __init__(self, cfg: Config, w: dict[str, torch.Tensor], device="cuda", lazy_cast=False):
        self.cfg = cfg
        self.lazy = lazy_cast
        self.w_bf16 = w
        self.w32 = None if lazy_cast else {k: v.float() for k, v in w.items()}
        self.cos, self.sin = rope_tables(cfg, device)
        L, S = cfg.layers, cfg.max_seq
        self.k_cache = torch.zeros(L, S, cfg.kv_heads, cfg.head_dim, device=device)
        self.v_cache = torch.zeros(L, S, cfg.kv_heads, cfg.head_dim, device=device)
        self.pos = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.pos = 0

    def _get(self, name, idx=None):
        if self.lazy:
            t = self.w_bf16[name] if idx is None else self.w_bf16[name][idx]
            return t.float()
        return self.w32[name] if idx is None else self.w32[name][idx]

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

    def generate(self, start_token: int, n_steps: int) -> tuple[list[int], torch.Tensor]:
        """Greedy decode n_steps tokens. Returns (tokens, last_logits)."""
        tok = start_token
        out = []
        logits = None
        for _ in range(n_steps):
            logits = self.step(tok)
            tok = int(logits.argmax())
            out.append(tok)
        return out, logits


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config()
    w = make_weights(cfg)
    ref = RefDecoder(cfg, w)
    toks, logits = ref.generate(start_token=12345, n_steps=8)
    print("tokens:", toks)
    print("logits: mean %.4f std %.4f max %.4f" % (logits.mean(), logits.std(), logits.max()))
    assert not torch.isnan(logits).any()
    print("reference OK")
