"""Eager PyTorch TP baseline for the Trainium P0 cases.

Each rank runs the single-device reference's math on its shard
(``tasks/sharding.py``) with ``torch.distributed`` collectives on the
``neuron`` backend: an all-reduce after each row-parallel projection and
after the routed experts, and a global argmax over the vocabulary shards.
It returns outputs in the rank layout a submission returns, so the harness
grades it like a candidate. It is the eager timing baseline, not an
optimized implementation.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .cases import Case
from .tasks.common import attention, rms, rope
from .tasks.gemma import dequant


def _all_reduce(value: torch.Tensor, group) -> torch.Tensor:
    """Sum BF16 partials in FP32 across ranks."""
    total = value.float()
    dist.all_reduce(total, group=group)
    return total.to(value.dtype)


def _greedy(logits: torch.Tensor, offset: int, group) -> torch.Tensor:
    """Global argmax over vocabulary shards; the lowest index wins ties."""
    local = logits.float()
    maximum = local.amax(-1, keepdim=True).clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    index = torch.arange(local.shape[-1], device=local.device, dtype=torch.int32) + offset
    candidate = torch.where(local == maximum, index, torch.iinfo(torch.int32).max).amin(-1)
    dist.all_reduce(candidate, op=dist.ReduceOp.MIN, group=group)
    return candidate.to(torch.int64)


def build_tp_baseline(case: Case, execution: dict) -> Callable[[dict], dict]:
    family, p = case.family, case.params
    world, rank, group = execution["world_size"], execution["tp_rank"], execution["tp_group"]
    qh, kvh, d, h = p["q_heads"] // world, p["kv_heads"] // world, p["head_dim"], p["hidden"]
    context, vocab = p["context"], p["vocab"] // world
    gemma, spec, moe = family == "quant_step", family == "spec_target_step", family == "moe_step"
    qk_norm = family in ("dense_step", "moe_step", "quant_step")
    eps = 1e-5 if spec else 1e-6
    theta = 500_000.0 if spec else 1_000_000.0
    experts = p.get("experts", world) // world

    def norm(x, weight):
        return rms(x, weight, eps, gemma=gemma)

    def run(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        def linear(name: str, layer: int, x: torch.Tensor) -> torch.Tensor:
            matrix = values[name][layer]
            if gemma:
                matrix = dequant(matrix, values[f"{name}_scale"][layer], p["bits"]).to(x.dtype)
            return x @ matrix.T

        if spec:
            ids = torch.cat((values["token"].reshape(1), values["draft_tokens"]))
            steps = p["draft_depth"] + 1
        else:
            ids, steps = values["token"].reshape(1), 1
        x = values["embed"][ids]
        if gemma:
            x = x * torch.tensor(math.sqrt(h), device=x.device, dtype=torch.bfloat16)
        positions = torch.arange(context, context + steps, device=x.device)
        allowed = None
        if spec:
            allowed = ~(torch.arange(context + steps, device=x.device)[None, :] > positions[:, None])
        k_writes, v_writes, hidden = [], [], []
        for layer in range(p["layers"]):
            xn = norm(x, values["ln1"][layer])
            q = linear("wq", layer, xn).view(steps, qh, d)
            k = linear("wk", layer, xn).view(steps, kvh, d)
            v = linear("wv", layer, xn).view(steps, kvh, d)
            if qk_norm:
                q, k = norm(q, values["qn"][layer]), norm(k, values["kn"][layer])
            if gemma:
                local = (layer + 1) % 6 != 0
                q = rope(q, positions, 10_000.0 if local else 1_000_000.0, factor=1.0 if local else 8.0)
                k = rope(k, positions, 10_000.0 if local else 1_000_000.0, factor=1.0 if local else 8.0)
            else:
                q, k = rope(q, positions, theta), rope(k, positions, theta)
            k_writes.append(k.to(torch.bfloat16))
            v_writes.append(v.to(torch.bfloat16))
            start = max(0, context + 1 - p["local_window"]) if gemma and (layer + 1) % 6 else 0
            keys = torch.cat((values["kcache"][layer, start:], k), dim=0)
            vals = torch.cat((values["vcache"][layer, start:], v), dim=0)
            scale = p.get("query_pre_attn_scalar", d) ** -0.5 if gemma else None
            o = attention(q, keys, vals, scale=scale, allowed=allowed).reshape(steps, qh * d)
            out = _all_reduce(linear("wo", layer, o), group)
            x = x + (norm(out, values["ln2"][layer]) if gemma else out)
            xn = norm(x, values["ln3" if gemma else "ln2"][layer])
            if moe:
                router = xn.float() @ values["wrt"][layer].float().T
                top_v, top_i = torch.topk(router[0], p["topk"])
                probs = torch.softmax(top_v, dim=-1)
                acc = torch.zeros_like(x, dtype=torch.float32)
                for slot, expert in enumerate(top_i.tolist()):
                    if rank * experts <= expert < (rank + 1) * experts:
                        e = expert - rank * experts
                        gate = xn @ values["wg"][layer, e].T
                        up = xn @ values["wu"][layer, e].T
                        acc = acc + (F.silu(gate) * up @ values["wd"][layer, e].T).float() * probs[slot]
                dist.all_reduce(acc, group=group)
                x = x + acc.to(x.dtype)
            else:
                gate, up = linear("wg", layer, xn), linear("wu", layer, xn)
                act = F.gelu(gate, approximate="tanh") if gemma else F.silu(gate)
                mlp = _all_reduce(linear("wd", layer, act * up), group)
                x = x + (norm(mlp, values["ln4"][layer]) if gemma else mlp)
            hidden.append(x)
        head = values["lm_head"] if "lm_head" in values else values["embed"][rank * vocab:(rank + 1) * vocab]
        logits = (norm(x, values["fnorm"]) @ head.T).float()
        greedy = _greedy(logits, rank * vocab, group)
        k_write, v_write = torch.stack(k_writes), torch.stack(v_writes)
        if not spec:
            return {"logits": logits[0], "next_token": greedy[0],
                    "k_write": k_write[:, 0].contiguous(), "v_write": v_write[:, 0].contiguous()}
        depth = p["draft_depth"]
        accepted = (greedy[:depth] == values["draft_tokens"]).to(torch.int64).cumprod(0).sum()
        count = int(accepted.item())
        committed = torch.full((steps,), -1, dtype=torch.int64, device=x.device)
        if count:
            committed[:count] = values["draft_tokens"][:count]
        committed[count] = greedy[count]
        keep = (torch.arange(steps, device=x.device) <= accepted)[None, :, None, None]
        feature_layers = (max(0, p["layers"] // 4 - 1), max(0, p["layers"] // 2 - 1), p["layers"] - 1)
        return {"logits": logits, "accepted_count": accepted,
                "committed_count": accepted + 1, "committed_tokens": committed,
                "cache_length": accepted + context + 1,
                "target_features": torch.stack([hidden[i][count] for i in feature_layers]).to(torch.bfloat16),
                "k_write": k_write.masked_fill(~keep, 0), "v_write": v_write.masked_fill(~keep, 0)}

    return run
