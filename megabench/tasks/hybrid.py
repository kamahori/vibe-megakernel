"""Full Qwen3.5 text decode with Gated DeltaNet and gated GQA.

    Linear-layer state is FP32 [value_heads, key_dim, value_dim]. The
    depthwise convolution buffer stores the last kernel_width raw projected
    QKV vectors, oldest first. Attention caches cover full-attention layers
    only. Reset discards all prior state, including the attention prefix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..cases import Case
from .common import fill_layer_weights, random_norm, random_weight, rms, rope


def delta_step(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               state: torch.Tensor, log_decay: torch.Tensor,
               beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One FP32 gated delta recurrence; returns output and new immutable state."""
    query = query.float() * torch.rsqrt(query.float().square().sum(-1, keepdim=True) + 1e-6)
    key = key.float() * torch.rsqrt(key.float().square().sum(-1, keepdim=True) + 1e-6)
    query = query * query.shape[-1] ** -0.5
    decayed = state.float() * log_decay.float().exp()[..., None, None]
    prediction = (decayed * key[..., :, None]).sum(-2)
    correction = (value.float() - prediction) * beta.float()[..., None]
    updated = decayed + key[..., :, None] * correction[..., None, :]
    output = (updated * query[..., :, None]).sum(-2)
    return output, updated


def make_inputs(case: Case, seed: int, device: str) -> dict[str, torch.Tensor]:
    p = case.params
    if p["batch"] != 1:
        raise ValueError("Qwen3.5 case is batch one")
    h, layers, inter = p["hidden"], p["layers"], p["intermediate"]
    full = layers // p["full_attention_interval"]
    linear = layers - full
    kh, vh, kd, vd = (p["linear_key_heads"], p["linear_value_heads"],
                       p["linear_key_dim"], p["linear_value_dim"])
    channels = 2 * kh * kd + vh * vd
    qdim, kdim = p["q_heads"] * p["head_dim"], p["kv_heads"] * p["head_dim"]
    g = torch.Generator(device=device).manual_seed(seed)
    values = {
        "token": torch.randint(p["vocab"], (), generator=g, device=device),
        "reset": torch.tensor(seed % 3 == 0, device=device),
        "embed": random_weight(g, device, (p["vocab"], h), h),
        "ln1": random_norm(g, device, (layers, h), gemma=True),
        "ln2": random_norm(g, device, (layers, h), gemma=True),
        "fnorm": random_norm(g, device, (h,), gemma=True),
        "qn": random_norm(g, device, (full, p["head_dim"]), gemma=True),
        "kn": random_norm(g, device, (full, p["head_dim"]), gemma=True),
        "linear_norm": random_norm(g, device, (linear, vd)),
        "A_log": torch.randn((linear, vh), generator=g, device=device),
        "dt_bias": torch.randn((linear, vh), generator=g, device=device),
        "conv_weight": random_weight(g, device, (linear, channels, p["conv_kernel"]), p["conv_kernel"]),
        "conv_state": torch.randn((linear, channels, p["conv_kernel"]), generator=g, device=device) * 0.1,
        "recurrent_state": torch.randn((linear, vh, kd, vd), generator=g, device=device) * 0.1,
    }
    shapes = {
        "wq": (full, 2 * qdim, h), "wk": (full, kdim, h),
        "wv": (full, kdim, h), "wo": (full, h, qdim),
        "in_qkv": (linear, channels, h), "in_z": (linear, vh * vd, h),
        "in_a": (linear, vh, h), "in_b": (linear, vh, h),
        "out_linear": (linear, h, vh * vd),
        "wg": (layers, inter, h), "wu": (layers, inter, h), "wd": (layers, h, inter),
    }
    for name, shape in shapes.items():
        values[name] = fill_layer_weights(g, device, shape, shape[-1])
    shape = (full, p["context"], p["kv_heads"], p["head_dim"])
    values["kcache"] = random_weight(g, device, shape, p["head_dim"])
    values["vcache"] = random_weight(g, device, shape, p["head_dim"])
    return values


def reference(case: Case, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    p = case.params
    reset = bool(values["reset"].item())
    context = 0 if reset else p["context"]
    x = values["embed"][values["token"]].float()
    k_writes, v_writes, states, conv_states = [], [], [], []
    linear_index, full_index = 0, 0
    position = torch.tensor([context], device=x.device)
    qh, kvh, d = p["q_heads"], p["kv_heads"], p["head_dim"]
    kh, vh, kd, vd = (p["linear_key_heads"], p["linear_value_heads"],
                       p["linear_key_dim"], p["linear_value_dim"])
    for layer in range(p["layers"]):
        normalized = rms(x, values["ln1"][layer], gemma=True)
        if (layer + 1) % p["full_attention_interval"]:
            idx = linear_index
            raw = values["in_qkv"][idx].float() @ normalized
            old = torch.zeros_like(values["conv_state"][idx]) if reset else values["conv_state"][idx]
            new_conv = torch.cat((old[:, 1:], raw[:, None]), dim=-1)
            mixed = F.silu((new_conv * values["conv_weight"][idx].float()).sum(-1))
            query, key, content = torch.split(mixed, (kh * kd, kh * kd, vh * vd))
            query = query.view(kh, kd).repeat_interleave(vh // kh, dim=0)
            key = key.view(kh, kd).repeat_interleave(vh // kh, dim=0)
            content = content.view(vh, vd)
            log_decay = -values["A_log"][idx].exp() * F.softplus(
                values["in_a"][idx].float() @ normalized + values["dt_bias"][idx])
            beta = (values["in_b"][idx].float() @ normalized).sigmoid()
            old = torch.zeros_like(values["recurrent_state"][idx]) if reset else values["recurrent_state"][idx]
            output, new_state = delta_step(query, key, content, old, log_decay, beta)
            gate = (values["in_z"][idx].float() @ normalized).view(vh, vd)
            output = rms(output, values["linear_norm"][idx]) * F.silu(gate)
            output = values["out_linear"][idx].float() @ output.flatten()
            states.append(new_state)
            conv_states.append(new_conv)
            linear_index += 1
        else:
            idx = full_index
            query_gate = (values["wq"][idx].float() @ normalized).view(qh, 2 * d)
            query, gate = query_gate.chunk(2, dim=-1)
            key = (values["wk"][idx].float() @ normalized).view(kvh, d)
            content = (values["wv"][idx].float() @ normalized).view(kvh, d)
            query = rms(query, values["qn"][idx], gemma=True)
            key = rms(key, values["kn"][idx], gemma=True)
            # Text positions repeat across all mRoPE axes, reducing to partial RoPE.
            width = p["rotary_dim"]
            query = torch.cat((rope(query[..., :width], position, 10_000_000.0), query[..., width:]), -1)
            key = torch.cat((rope(key[..., :width], position, 10_000_000.0), key[..., width:]), -1)
            k_writes.append(key.to(torch.bfloat16))
            v_writes.append(content.to(torch.bfloat16))
            keys = torch.cat((values["kcache"][idx, :context].float(), key[None]))
            content_cache = torch.cat((values["vcache"][idx, :context].float(), content[None]))
            grouped = query.view(kvh, qh // kvh, d)
            probability = (torch.einsum("gqd,tgd->gqt", grouped, keys) * d ** -0.5).softmax(-1)
            attention = torch.einsum("gqt,tgd->gqd", probability, content_cache).reshape(qh, d)
            output = values["wo"][idx].float() @ (attention * gate.sigmoid()).flatten()
            full_index += 1
        x = x + output
        normalized = rms(x, values["ln2"][layer], gemma=True)
        x = x + values["wd"][layer].float() @ (
            F.silu(values["wg"][layer].float() @ normalized) *
            (values["wu"][layer].float() @ normalized))
    logits = values["embed"].float() @ rms(x, values["fnorm"], gemma=True)
    return {"logits": logits, "next_token": logits.argmax().to(torch.int64),
            "k_write": torch.stack(k_writes), "v_write": torch.stack(v_writes),
            "recurrent_state": torch.stack(states), "conv_state": torch.stack(conv_states),
            "cache_length": torch.tensor(context + 1, device=x.device, dtype=torch.int64)}
