"""Gemma 3 4B text decode with explicit W8A16 or packed W4A16 weights.

This is a synthetic-weight tier. Every projection uses symmetric per-output-row
quantization; embedding and norm parameters stay BF16. The same random base
weight stream is used for the W8 and W4 cells at a given seed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..cases import Case
from .common import random_norm, random_weight, rms, rope


LINEAR_NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")
NORM_NAMES = ("ln1", "ln2", "ln3", "ln4", "qn", "kn", "fnorm")


def _quantized_stack(g: torch.Generator, device: str, shape: tuple[int, int, int],
                     bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    layers, rows, cols = shape
    if bits == 4 and cols % 2:
        raise ValueError("W4 input dimension must be even")
    packed_cols = cols if bits == 8 else cols // 2
    qweight = torch.empty((layers, rows, packed_cols), device=device,
                          dtype=torch.int8 if bits == 8 else torch.uint8)
    scales = torch.empty((layers, rows), device=device, dtype=torch.float32)
    limit = 127 if bits == 8 else 7
    for layer in range(layers):
        base = random_weight(g, device, (rows, cols), cols).float()
        scale = (base.abs().amax(-1) / limit).clamp_min(1e-8)
        quant = torch.round(base / scale[:, None]).clamp(-limit, limit).to(torch.int8)
        if bits == 8:
            qweight[layer].copy_(quant)
        else:
            lo = (quant[:, 0::2].to(torch.int16) + 8).to(torch.uint8)
            hi = (quant[:, 1::2].to(torch.int16) + 8).to(torch.uint8)
            qweight[layer].copy_(lo | (hi << 4))
        scales[layer].copy_(scale)
    return qweight, scales


def dequant(qweight: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        q = qweight.float()
    elif bits == 4:
        lo = (qweight & 15).to(torch.int16) - 8
        hi = (qweight >> 4).to(torch.int16) - 8
        q = torch.stack((lo, hi), dim=-1).flatten(-2).float()
    else:
        raise ValueError(f"unsupported quantization width {bits}")
    return q * scale.float().unsqueeze(-1)


def make_inputs(case: Case, seed: int, device: str) -> dict[str, torch.Tensor]:
    p = case.params
    if p["batch"] != 1:
        raise NotImplementedError("Gemma 3 fixture currently handles batch one")
    h, layers, inter, d = (p["hidden"], p["layers"], p["intermediate"],
                           p["head_dim"])
    qdim, kdim = p["q_heads"] * d, p["kv_heads"] * d
    g = torch.Generator(device=device).manual_seed(seed)
    values = {
        "token": torch.randint(p["vocab"], (), generator=g, device=device,
                               dtype=torch.int64),
        "embed": random_weight(g, device, (p["vocab"], h), h),
        "ln1": random_norm(g, device, (layers, h), gemma=True),
        "ln2": random_norm(g, device, (layers, h), gemma=True),
        "ln3": random_norm(g, device, (layers, h), gemma=True),
        "ln4": random_norm(g, device, (layers, h), gemma=True),
        "qn": random_norm(g, device, (layers, d), gemma=True),
        "kn": random_norm(g, device, (layers, d), gemma=True),
        "fnorm": random_norm(g, device, (h,), gemma=True),
    }
    shapes = {
        "wq": (layers, qdim, h), "wk": (layers, kdim, h),
        "wv": (layers, kdim, h), "wo": (layers, h, qdim),
        "wg": (layers, inter, h), "wu": (layers, inter, h),
        "wd": (layers, h, inter),
    }
    for name, shape in shapes.items():
        values[name], values[f"{name}_scale"] = _quantized_stack(
            g, device, shape, p["bits"])
    cache_shape = (layers, p["context"], p["kv_heads"], d)
    values["kcache"] = random_weight(g, device, cache_shape, d)
    values["vcache"] = random_weight(g, device, cache_shape, d)
    return values


def reference(case: Case, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    p = case.params
    h, d, context, bits = (p["hidden"], p["head_dim"], p["context"], p["bits"])
    qh, kvh = p["q_heads"], p["kv_heads"]
    x = values["embed"][values["token"]].float() * math.sqrt(h)
    positions = torch.tensor([context], device=x.device)
    k_writes, v_writes = [], []

    def linear(name: str, layer: int, vec: torch.Tensor) -> torch.Tensor:
        matrix = dequant(values[name][layer], values[f"{name}_scale"][layer], bits)
        return matrix @ vec

    for layer in range(p["layers"]):
        xn = rms(x, values["ln1"][layer], gemma=True)
        q = linear("wq", layer, xn).view(qh, d)
        k = linear("wk", layer, xn).view(kvh, d)
        v = linear("wv", layer, xn).view(kvh, d)
        q = rms(q, values["qn"][layer], gemma=True)
        k = rms(k, values["kn"][layer], gemma=True)
        local = (layer + 1) % 6 != 0
        theta = 10_000.0 if local else 1_000_000.0
        factor = 1.0 if local else 8.0
        q = rope(q, positions, theta, factor=factor)
        k = rope(k, positions, theta, factor=factor)
        k_writes.append(k.to(torch.bfloat16))
        v_writes.append(v.to(torch.bfloat16))
        start = max(0, context + 1 - p["local_window"]) if local else 0
        ks = torch.cat((values["kcache"][layer, start:].float(), k[None]), dim=0)
        vs = torch.cat((values["vcache"][layer, start:].float(), v[None]), dim=0)
        grouped_q = q.view(kvh, qh // kvh, d)
        scores = torch.einsum("gqd,tgd->gqt", grouped_q, ks) * d ** -0.5
        probs = scores.softmax(dim=-1)
        attn = torch.einsum("gqt,tgd->gqd", probs, vs).reshape(qh * d)
        x = x + rms(linear("wo", layer, attn), values["ln2"][layer], gemma=True)

        xn = rms(x, values["ln3"][layer], gemma=True)
        gate = linear("wg", layer, xn)
        up = linear("wu", layer, xn)
        mlp = linear("wd", layer, F.gelu(gate, approximate="tanh") * up)
        x = x + rms(mlp, values["ln4"][layer], gemma=True)

    logits = values["embed"].float() @ rms(x, values["fnorm"], gemma=True)
    return {
        "logits": logits.float(),
        "next_token": logits.argmax().to(torch.int64),
        "k_write": torch.stack(k_writes),
        "v_write": torch.stack(v_writes),
    }
