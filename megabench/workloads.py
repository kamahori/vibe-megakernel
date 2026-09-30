"""Independent PyTorch oracles and seeded inputs for agent challenges.

These define *semantics*, not candidate implementations. Keep all candidate
compilation, DSL imports, and optimization outside this module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cases import Case


Inputs = dict[str, torch.Tensor]
Outputs = dict[str, torch.Tensor]


def make_inputs(case: Case, seed: int, device: str = "cpu") -> Inputs:
    p = case.params
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def normal(*shape: int, scale: float = 1.0, dtype=torch.bfloat16) -> torch.Tensor:
        return (torch.randn(shape, generator=gen) * scale).to(dtype)

    if case.family == "stencil":
        inputs = {"x": normal(p["batch"], p["width"], dtype=torch.float32),
                  "alpha": torch.tensor(0.15 + (seed % 7) * 0.01,
                                        dtype=torch.float32)}
    elif case.family == "decoder":
        b, h, s, i = (p[k] for k in ("batch", "hidden", "context", "intermediate"))
        scale = h ** -0.5
        inputs = {"x": normal(b, h), "kcache": normal(b, s, h, scale=0.5),
                  "vcache": normal(b, s, h, scale=0.5),
                  "norm1": normal(h, scale=0.1, dtype=torch.float32) + 1,
                  "norm2": normal(h, scale=0.1, dtype=torch.float32) + 1,
                  "wq": normal(h, h, scale=scale),
                  "wk": normal(h, h, scale=scale),
                  "wv": normal(h, h, scale=scale),
                  "wo": normal(h, h, scale=scale),
                  "wg": normal(i, h, scale=scale),
                  "wu": normal(i, h, scale=scale),
                  "wd": normal(h, i, scale=i ** -0.5)}
    elif case.family == "moe":
        b, h, e, i = (p[k] for k in ("batch", "hidden", "experts",
                                     "intermediate"))
        inputs = {"x": normal(b, h), "router": normal(e, h, scale=h ** -0.5),
                  "w_up": normal(e, i, h, scale=h ** -0.5),
                  "w_down": normal(e, h, i, scale=i ** -0.5)}
    elif case.family == "quant_mlp":
        b, h, i, bits = (p[k] for k in ("batch", "hidden", "intermediate", "bits"))
        qlo, qhi = (-8, 8) if bits == 4 else (-127, 128)

        def weight(rows: int, cols: int) -> torch.Tensor:
            q = torch.randint(qlo, qhi, (rows, cols), generator=gen,
                              dtype=torch.int8)
            if bits == 8:
                return q
            # Two signed 4-bit values per byte, biased by +8.
            lo = (q[:, 0::2].to(torch.int16) + 8).to(torch.uint8)
            hi = (q[:, 1::2].to(torch.int16) + 8).to(torch.uint8)
            return lo | (hi << 4)

        inputs = {"x": normal(b, h), "norm": normal(h, scale=0.1,
                  dtype=torch.float32) + 1,
                  "w_gate": weight(i, h), "w_up": weight(i, h),
                  "w_down": weight(h, i),
                  "s_gate": normal(i, scale=0.001, dtype=torch.float32).abs() + 0.008,
                  "s_up": normal(i, scale=0.001, dtype=torch.float32).abs() + 0.008,
                  "s_down": normal(h, scale=0.001, dtype=torch.float32).abs() + 0.008}
    elif case.family == "spec_verify":
        b, k, v = (p[key] for key in ("batch", "draft_len", "vocab"))
        draft = normal(b, k, v, dtype=torch.float32)
        target = normal(b, k + 1, v, dtype=torch.float32)
        proposed = draft.argmax(-1)
        for row in range(b):
            accept = (seed + 3 * row) % (k + 1)
            for step in range(k):
                token = int(proposed[row, step])
                chosen = token if step < accept else (token + 1) % v
                target[row, step, chosen] = 100.0
        inputs = {"draft_logits": draft, "target_logits": target}
    else:
        raise ValueError(f"no generator for {case.family}")
    return {name: value.to(device) for name, value in inputs.items()}


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight


def _dequant(weight: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        q = weight.float()
    else:
        lo = (weight & 15).to(torch.int16) - 8
        hi = (weight >> 4).to(torch.int16) - 8
        q = torch.stack((lo, hi), dim=-1).flatten(-2).float()
    return q * scale.float().unsqueeze(-1)


def reference(case: Case, t: Inputs) -> Outputs:
    p = case.params
    if case.family == "stencil":
        x, alpha = t["x"], t["alpha"]
        for _ in range(p["steps"]):
            x = (1 - 2 * alpha) * x + alpha * (torch.roll(x, 1, -1) +
                                                torch.roll(x, -1, -1))
        return {"y": x}
    if case.family == "decoder":
        x = t["x"].float()
        n1 = _rmsnorm(x, t["norm1"].float())
        q = n1 @ t["wq"].float().T
        k = n1 @ t["wk"].float().T
        v = n1 @ t["wv"].float().T
        kc = torch.cat((t["kcache"].float(), k[:, None, :]), dim=1)
        vc = torch.cat((t["vcache"].float(), v[:, None, :]), dim=1)
        scores = (kc * q[:, None, :]).sum(-1) / math.sqrt(p["hidden"])
        context = (scores.softmax(-1)[:, :, None] * vc).sum(1)
        z = x + context @ t["wo"].float().T
        n2 = _rmsnorm(z, t["norm2"].float())
        gate = F.silu(n2 @ t["wg"].float().T)
        up = n2 @ t["wu"].float().T
        y = z + (gate * up) @ t["wd"].float().T
        return {"y": y.bfloat16(), "kcache": kc.bfloat16(),
                "vcache": vc.bfloat16()}
    if case.family == "moe":
        x = t["x"].float()
        scores = x @ t["router"].float().T
        values, indices = scores.topk(p["topk"], dim=-1)
        probs = values.softmax(-1)
        result = x.clone()
        for slot in range(p["topk"]):
            expert = indices[:, slot]
            up = torch.bmm(t["w_up"][expert].float(), x[:, :, None]).squeeze(-1)
            act = F.silu(up)
            down = torch.bmm(t["w_down"][expert].float(), act[:, :, None]).squeeze(-1)
            result = result + probs[:, slot, None] * down
        return {"y": result.bfloat16(), "expert_ids": indices.to(torch.int32)}
    if case.family == "quant_mlp":
        x = t["x"].float()
        n = _rmsnorm(x, t["norm"].float())
        bits = p["bits"]
        gate = n @ _dequant(t["w_gate"], t["s_gate"], bits).T
        up = n @ _dequant(t["w_up"], t["s_up"], bits).T
        down = (F.silu(gate) * up) @ _dequant(
            t["w_down"], t["s_down"], bits).T
        return {"y": (x + down).bfloat16()}
    if case.family == "spec_verify":
        draft = t["draft_logits"].argmax(-1)
        target = t["target_logits"].argmax(-1)
        k = p["draft_len"]
        equal_prefix = (draft == target[:, :k]).to(torch.int64).cumprod(-1)
        accepted = equal_prefix.sum(-1)
        committed = torch.full((p["batch"], k + 1), -1, dtype=torch.int64,
                               device=draft.device)
        for step in range(k):
            committed[:, step] = torch.where(
                accepted > step, draft[:, step],
                torch.where(accepted == step, target[:, step], -1))
        committed[:, k] = torch.where(accepted == k, target[:, k], -1)
        return {"accepted_count": accepted,
                "committed_count": accepted + 1,
                "committed_tokens": committed}
    raise ValueError(f"no oracle for {case.family}")
