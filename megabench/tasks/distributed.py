"""Full Gemma TP and Qwen3 MoE TP/EP decode with rank-local tensors.

Embeddings/logits are vocabulary-sharded; attention uses head TP. FFN
intermediates use TP and experts use contiguous EP ownership. Attention
partial outputs reduce within each TP group; routed expert partials reduce
across TP*EP. Inputs and outputs contain only this rank's KV/head shards.
"""

from __future__ import annotations

import hashlib

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..cases import Case
from .common import rms, rope
from .parallel import initialize


ROW_BLOCK = 128


def seed_for(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest(), "little")


def matrix(seed: int, name: str, rows: int, cols: int, fan_in: int,
           device: str, *, row_range: tuple[int, int] | None = None,
           col_range: tuple[int, int] | None = None) -> torch.Tensor:
    """Sample canonical row blocks, so slices match a separately built oracle.

    No rank materializes a full model or a full embedding matrix. At most
    128 global rows of one matrix exist as an FP32 sampling temporary.
    """
    r0, r1 = row_range or (0, rows)
    c0, c1 = col_range or (0, cols)
    output = torch.empty((r1 - r0, c1 - c0), device=device, dtype=torch.bfloat16)
    for start in range(r0 // ROW_BLOCK * ROW_BLOCK, r1, ROW_BLOCK):
        end = min(start + ROW_BLOCK, rows)
        generator = torch.Generator(device=device).manual_seed(seed_for(seed, f"{name}:{start}"))
        block = (torch.randn((end - start, cols), generator=generator, device=device) *
                 fan_in ** -0.5).to(torch.bfloat16)
        lo, hi = max(start, r0), min(end, r1)
        output[lo - r0:hi - r0].copy_(block[lo - start:hi - start, c0:c1])
    return output


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None) -> dict[str, torch.Tensor]:
    if rank is None:
        rank = dist.get_rank()
    p, tp, ep = case.params, case.tp, case.ep
    tensor_rank, expert_rank = rank % tp, rank // tp
    gemma = "gemma" in case.model.lower()
    h, layers, inter, d = p["hidden"], p["layers"], p["intermediate"], p["head_dim"]
    qdim, kdim, vocab = p["q_heads"] * d, p["kv_heads"] * d, p["vocab"]
    for dimension in (p["q_heads"], p["kv_heads"], inter, vocab):
        if dimension % tp:
            raise ValueError("heads, intermediate, and vocabulary must divide TP")
    if not 0 <= rank < case.gpus or case.gpus != tp * ep:
        raise ValueError("invalid rank or TP/EP topology")
    split = lambda n: (tensor_rank * (n // tp), (tensor_rank + 1) * (n // tp))
    values = {"token": torch.tensor(seed_for(seed, "token") % vocab, device=device)}
    values["embed"] = matrix(seed, "embed", vocab, h, h, device, row_range=split(vocab))
    if not gemma:
        values["lm_head"] = matrix(seed, "lm_head", vocab, h, h, device, row_range=split(vocab))
    for name in (("ln1", "ln2", "ln3", "ln4") if gemma else ("ln1", "ln2")):
        raw = matrix(seed, name, layers, h, 100, device).float()
        values[name] = (raw if gemma else raw + 1).to(torch.bfloat16)
    values["fnorm"] = matrix(seed, "fnorm", 1, h, 100, device)[0]
    if not gemma:
        values["fnorm"] = (values["fnorm"].float() + 1).to(torch.bfloat16)
    for name in ("qn", "kn"):
        values[name] = matrix(seed, name, layers, d, 100, device)
        if not gemma:
            values[name] = (values[name].float() + 1).to(torch.bfloat16)
    shapes = {"wq": (qdim, h, split(qdim), None),
              "wk": (kdim, h, split(kdim), None),
              "wv": (kdim, h, split(kdim), None),
              "wo": (h, qdim, None, split(qdim))}
    if gemma:
        shapes |= {"wg": (inter, h, split(inter), None), "wu": (inter, h, split(inter), None),
                   "wd": (h, inter, None, split(inter))}
    for name, (rows, cols, rr, cr) in shapes.items():
        local_rows, local_cols = (rr[1] - rr[0] if rr else rows), (cr[1] - cr[0] if cr else cols)
        weights = torch.empty((layers, local_rows, local_cols), dtype=torch.bfloat16, device=device)
        for layer in range(layers):
            weights[layer].copy_(matrix(seed, f"{name}:{layer}", rows, cols, cols, device,
                                       row_range=rr, col_range=cr))
        values[name] = weights
    if not gemma:
        experts = p["experts"]
        if experts % ep:
            raise ValueError("expert count must divide EP")
        values["router"] = torch.empty((layers, experts, h), dtype=torch.bfloat16, device=device)
        for layer in range(layers):
            values["router"][layer].copy_(matrix(seed, f"router:{layer}", experts, h, h, device))
        local_experts = experts // ep
        for name in ("wg", "wu", "wd"):
            rows, cols = (h, inter) if name == "wd" else (inter, h)
            rr, cr = (None, split(inter)) if name == "wd" else (split(inter), None)
            local_rows, local_cols = (rows if rr is None else inter // tp), (cols if cr is None else inter // tp)
            weights = torch.empty((layers, local_experts, local_rows, local_cols), dtype=torch.bfloat16, device=device)
            for layer in range(layers):
                for expert in range(local_experts):
                    global_expert = expert_rank * local_experts + expert
                    weights[layer, expert].copy_(matrix(seed, f"{name}:{layer}:{global_expert}",
                                                       rows, cols, cols, device, row_range=rr, col_range=cr))
            values[name] = weights
    for name in ("kcache", "vcache"):
        weights = torch.empty((layers, p["context"], p["kv_heads"] // tp, d), dtype=torch.bfloat16, device=device)
        for layer in range(layers):
            weights[layer].copy_(matrix(seed, f"{name}:{layer}", p["context"], kdim, d, device,
                                       col_range=split(kdim)).view(p["context"], p["kv_heads"] // tp, d))
        values[name] = weights
    return values


def reduce_sum(value: torch.Tensor, group=None) -> torch.Tensor:
    value = value.clone()
    dist.all_reduce(value, group=group)
    return value


def greedy_token(logits: torch.Tensor, offset: int, group) -> torch.Tensor:
    maximum = logits.max().clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    indices = torch.arange(logits.numel(), device=logits.device, dtype=torch.int64) + offset
    token = torch.where(logits == maximum, indices, torch.iinfo(torch.int64).max).min()
    dist.all_reduce(token, op=dist.ReduceOp.MIN, group=group)
    return token


def reference(case: Case, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    ctx = initialize(case)
    p = case.params
    gemma = "gemma" in case.model.lower()
    qh, kvh, d = p["q_heads"] // case.tp, p["kv_heads"] // case.tp, p["head_dim"]
    vocab_size = p["vocab"] // case.tp
    offset, token = ctx.tp_rank * vocab_size, int(values["token"].item())
    x = values["embed"][token - offset].float() if offset <= token < offset + vocab_size else torch.zeros(p["hidden"], device=values["token"].device)
    x = reduce_sum(x, ctx.tp_group)
    if gemma:
        x = x * p["hidden"] ** 0.5
    position = torch.tensor([p["context"]], device=x.device)
    keys_out, content_out, routes = [], [], []
    for layer in range(p["layers"]):
        normalized = rms(x, values["ln1"][layer], gemma=gemma)
        query = rms((values["wq"][layer].float() @ normalized).view(qh, d), values["qn"][layer], gemma=gemma)
        key = rms((values["wk"][layer].float() @ normalized).view(kvh, d), values["kn"][layer], gemma=gemma)
        content = (values["wv"][layer].float() @ normalized).view(kvh, d)
        local = gemma and (layer + 1) % 6 != 0
        theta = (10_000.0 if local else 1_000_000.0) if gemma else 1_000_000.0
        factor = 8.0 if gemma and not local else 1.0
        query, key = rope(query, position, theta, factor=factor), rope(key, position, theta, factor=factor)
        keys_out.append(key.to(torch.bfloat16))
        content_out.append(content.to(torch.bfloat16))
        start = max(0, p["context"] + 1 - p["local_window"]) if local else 0
        keys = torch.cat((values["kcache"][layer, start:].float(), key[None]))
        content_cache = torch.cat((values["vcache"][layer, start:].float(), content[None]))
        scalar = p.get("query_pre_attn_scalar", d)
        probability = (torch.einsum("gqd,tgd->gqt", query.view(kvh, qh // kvh, d), keys) * scalar ** -0.5).softmax(-1)
        attention = torch.einsum("gqt,tgd->gqd", probability, content_cache).flatten()
        output = reduce_sum(values["wo"][layer].float() @ attention, ctx.tp_group)
        x = x + (rms(output, values["ln2"][layer], gemma=True) if gemma else output)
        normalized = rms(x, values["ln3" if gemma else "ln2"][layer], gemma=gemma)
        if gemma:
            hidden = F.gelu(values["wg"][layer].float() @ normalized, approximate="tanh") * (values["wu"][layer].float() @ normalized)
            output = reduce_sum(values["wd"][layer].float() @ hidden, ctx.tp_group)
            x = x + rms(output, values["ln4"][layer], gemma=True)
        else:
            scores = values["router"][layer].float() @ normalized
            selected = torch.argsort(scores, descending=True, stable=True)[:p["topk"]]
            probability = scores[selected].softmax(-1)
            routes.append(selected)
            output = torch.zeros_like(x)
            local_experts = p["experts"] // case.ep
            for slot, expert in enumerate(selected.tolist()):
                if expert // local_experts != ctx.ep_rank:
                    continue
                local_expert = expert % local_experts
                hidden = F.silu(values["wg"][layer, local_expert].float() @ normalized) * (values["wu"][layer, local_expert].float() @ normalized)
                output = output + values["wd"][layer, local_expert].float() @ (hidden * probability[slot])
            x = x + reduce_sum(output)
    head = values["embed" if gemma else "lm_head"]
    logits = head.float() @ rms(x, values["fnorm"], gemma=gemma)
    outputs = {"logits": logits, "next_token": greedy_token(logits, offset, ctx.tp_group),
               "k_write": torch.stack(keys_out), "v_write": torch.stack(content_out)}
    if not gemma:
        outputs["expert_ids"] = torch.stack(routes)
    return outputs
