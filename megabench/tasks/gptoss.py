"""Complete GPT-OSS decode with native packed MXFP4 expert inputs.

    Synthetic BF16 attention/router/embedding weights and E2M1/E8M0 experts.
    Expert packing follows the checkpoint: [input, output/32, 16]. Every
    selected expert is unpacked inside the call. Inputs are never mutated.
    Arithmetic accumulates in FP32, matching MegaBench's synthetic tier.
"""

from __future__ import annotations

import math

import torch

from ..cases import Case
from .common import fill_layer_weights, random_norm, random_weight, rms
from .quantization import pack_mxfp4, unpack_mxfp4


def yarn_rotate(x: torch.Tensor, position: int) -> torch.Tensor:
    """GPT-OSS YaRN: factor 32, original context 4096, truncate=False."""
    width = x.shape[-1]
    index = torch.arange(width // 2, device=x.device, dtype=torch.float64)
    inverse = 150_000.0 ** (-2 * index / width)
    correction = lambda rotations: width * math.log(
        4096 / (rotations * 2 * math.pi)) / (2 * math.log(150_000.0))
    low, high = max(correction(32), 0), min(correction(1), width - 1)
    ramp = ((index - low) / max(high - low, 0.001)).clamp(0, 1)
    inverse = inverse * (1 - ramp) + inverse / 32 * ramp
    phase = torch.cat((position * inverse, position * inverse))
    factor = 1 + 0.1 * math.log(32)
    cos, sin = phase.cos().float() * factor, phase.sin().float() * factor
    half = width // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def make_inputs(case: Case, seed: int, device: str) -> dict[str, torch.Tensor]:
    p = case.params
    if p["batch"] != 1:
        raise ValueError("GPT-OSS case is batch one")
    h, layers, inter = p["hidden"], p["layers"], p["intermediate"]
    qdim, kdim = p["q_heads"] * p["head_dim"], p["kv_heads"] * p["head_dim"]
    experts = p["experts"]
    g = torch.Generator(device=device).manual_seed(seed)
    values = {
        "token": torch.randint(p["vocab"], (), generator=g, device=device),
        "embed": random_weight(g, device, (p["vocab"], h), h),
        "lm_head": random_weight(g, device, (p["vocab"], h), h),
        "ln1": random_norm(g, device, (layers, h)),
        "ln2": random_norm(g, device, (layers, h)),
        "fnorm": random_norm(g, device, (h,)),
    }
    for name, rows, cols in (("wq", qdim, h), ("wk", kdim, h),
                             ("wv", kdim, h), ("wo", h, qdim),
                             ("router", experts, h)):
        values[name] = fill_layer_weights(g, device, (layers, rows, cols), cols)
        bias_name = "router_bias" if name == "router" else "b" + name[1:]
        values[bias_name] = (0.02 * torch.randn(
            (layers, rows), generator=g, device=device)).to(torch.bfloat16)
    values["sinks"] = (0.02 * torch.randn(
        (layers, p["q_heads"]), generator=g, device=device)).to(torch.bfloat16)
    for name, inputs, outputs in (("gate_up", h, 2 * inter), ("down", inter, h)):
        blocks = torch.empty((layers, experts, inputs, outputs // 32, 16),
                             dtype=torch.uint8, device=device)
        scales = torch.empty(blocks.shape[:-1], dtype=torch.uint8, device=device)
        for layer in range(layers):
            for expert in range(experts):
                base = random_weight(g, device, (inputs, outputs), inputs)
                blocks[layer, expert], scales[layer, expert] = pack_mxfp4(base)
        values[name + "_blocks"], values[name + "_scales"] = blocks, scales
        values[name + "_bias"] = (0.02 * torch.randn(
            (layers, experts, outputs), generator=g, device=device)).to(torch.bfloat16)
    shape = (layers, p["context"], p["kv_heads"], p["head_dim"])
    values["kcache"] = random_weight(g, device, shape, p["head_dim"])
    values["vcache"] = random_weight(g, device, shape, p["head_dim"])
    return values


def reference(case: Case, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    p = case.params
    qh, kvh, d = p["q_heads"], p["kv_heads"], p["head_dim"]
    context = p["context"]
    x = values["embed"][values["token"]].float()
    k_writes, v_writes, routes = [], [], []
    for layer in range(p["layers"]):
        normalized = rms(x, values["ln1"][layer], eps=1e-5)
        project = lambda name: values["w" + name][layer].float() @ normalized + values["b" + name][layer].float()
        q = yarn_rotate(project("q").view(qh, d), context)
        k = yarn_rotate(project("k").view(kvh, d), context)
        v = project("v").view(kvh, d)
        k_writes.append(k.to(torch.bfloat16))
        v_writes.append(v.to(torch.bfloat16))
        start = max(0, context + 1 - p["sliding_window"]) if layer % 2 == 0 else 0
        keys = torch.cat((values["kcache"][layer, start:].float(), k[None]))
        content = torch.cat((values["vcache"][layer, start:].float(), v[None]))
        grouped = q.view(kvh, qh // kvh, d)
        scores = torch.einsum("gqd,tgd->gqt", grouped, keys) * d ** -0.5
        sinks = values["sinks"][layer].float().view(kvh, qh // kvh, 1)
        probabilities = torch.cat((scores, sinks), dim=-1).softmax(-1)[..., :-1]
        attention = torch.einsum("gqt,tgd->gqd", probabilities, content).flatten()
        x = x + values["wo"][layer].float() @ attention + values["bo"][layer].float()
        normalized = rms(x, values["ln2"][layer], eps=1e-5)
        scores = values["router"][layer].float() @ normalized + values["router_bias"][layer].float()
        # Stable sorting fixes expert-ID tie breaking as part of the contract.
        selected = torch.argsort(scores, descending=True, stable=True)[:p["topk"]]
        probabilities = scores[selected].softmax(-1)
        routes.append(selected)
        output = torch.zeros_like(x)
        for slot, expert in enumerate(selected):
            weights = unpack_mxfp4(values["gate_up_blocks"][layer, expert],
                                   values["gate_up_scales"][layer, expert])
            gate_up = normalized @ weights + values["gate_up_bias"][layer, expert].float()
            gate = gate_up[::2].clamp(max=7.0)
            up = gate_up[1::2].clamp(-7.0, 7.0)
            activated = (up + 1) * gate * (1.702 * gate).sigmoid()
            weights = unpack_mxfp4(values["down_blocks"][layer, expert],
                                   values["down_scales"][layer, expert])
            output = output + probabilities[slot] * (
                activated @ weights + values["down_bias"][layer, expert].float())
        x = x + output
    logits = values["lm_head"].float() @ rms(x, values["fnorm"], eps=1e-5)
    return {"logits": logits, "next_token": logits.argmax().to(torch.int64),
            "k_write": torch.stack(k_writes), "v_write": torch.stack(v_writes),
            "expert_ids": torch.stack(routes)}
