"""Synthetic SigLIP image conditioning and full Gemma decoder prefill.

The vision tower and multimodal prefill run only in trusted fixture setup.
The timed task receives their actual BF16 KV state and decodes one text token.
The case's context is the number of text prefix tokens; image tokens are
additional. Image blocks attend bidirectionally within each image, while text
positions use causal attention. Local layers retain the same window rule.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from ..cases import Case
from . import gemma
from .common import fill_layer_weights, random_norm, random_weight, rms, rope


def vision_inputs(case: Case, seed: int, device: str) -> dict[str, torch.Tensor]:
    p = case.params
    g = torch.Generator(device=device).manual_seed(seed ^ 0x51611)
    h, layers, inter = p["vision_hidden"], p["vision_layers"], p["vision_intermediate"]
    patch, size = p["patch_size"], p["image_size"]
    values = {
        "pixels": torch.rand((p["images"], 3, size, size), generator=g, device=device) * 2 - 1,
        "patch_weight": random_weight(g, device, (h, 3, patch, patch), 3 * patch * patch),
        "patch_bias": random_weight(g, device, (h,), h),
        "position": random_weight(g, device, ((size // patch) ** 2, h), h),
        "norm1": random_norm(g, device, (layers, h)),
        "norm2": random_norm(g, device, (layers, h)),
        "norm1_bias": random_weight(g, device, (layers, h), h),
        "norm2_bias": random_weight(g, device, (layers, h), h),
        "final_norm": random_norm(g, device, (h,)),
        "final_bias": random_weight(g, device, (h,), h),
        "projector_norm": random_norm(g, device, (h,), gemma=True),
        "projector": random_weight(g, device, (h, p["hidden"]), h),
    }
    for name, rows, cols in (("q", h, h), ("k", h, h), ("v", h, h),
                             ("o", h, h), ("up", inter, h), ("down", h, inter)):
        values[name] = fill_layer_weights(g, device, (layers, rows, cols), cols)
        values[name + "_bias"] = random_weight(g, device, (layers, rows), rows)
    return values


def image_features(case: Case, values: dict[str, torch.Tensor]) -> torch.Tensor:
    """Full SigLIP encoder, average pooling, Gemma normalization and projector."""
    p = case.params
    heads, h = p["vision_heads"], p["vision_hidden"]
    dim = h // heads
    x = F.conv2d(values["pixels"].float(), values["patch_weight"].float(),
                 values["patch_bias"].float(), stride=p["patch_size"])
    x = x.flatten(2).transpose(1, 2) + values["position"].float()
    for layer in range(p["vision_layers"]):
        normalized = F.layer_norm(x, (h,), values["norm1"][layer].float(),
                                  values["norm1_bias"][layer].float(), eps=1e-6)
        projected = [(normalized @ values[name][layer].float().T +
                      values[name + "_bias"][layer].float()).view(
                          p["images"], -1, heads, dim).transpose(1, 2)
                     for name in ("q", "k", "v")]
        query, key, content = projected
        # SDPA uses the same exact bidirectional mask; its tiled CUDA path
        # avoids materializing 16 full 4096x4096 attention matrices.
        attention = F.scaled_dot_product_attention(query, key, content,
                                                    dropout_p=0.0, is_causal=False)
        attention = attention.transpose(1, 2).reshape_as(x)
        x = x + attention @ values["o"][layer].float().T + values["o_bias"][layer].float()
        normalized = F.layer_norm(x, (h,), values["norm2"][layer].float(),
                                  values["norm2_bias"][layer].float(), eps=1e-6)
        up = normalized @ values["up"][layer].float().T + values["up_bias"][layer].float()
        x = x + F.gelu(up, approximate="tanh") @ values["down"][layer].float().T + values["down_bias"][layer].float()
    x = F.layer_norm(x, (h,), values["final_norm"].float(), values["final_bias"].float(), eps=1e-6)
    side = p["image_size"] // p["patch_size"]
    output_side = math.isqrt(p["image_tokens"])
    if output_side ** 2 != p["image_tokens"] or side % output_side:
        raise ValueError("image tokens must form a square pooling grid dividing patch grid")
    x = x.transpose(1, 2).reshape(p["images"], h, side, side)
    x = F.avg_pool2d(x, side // output_side).flatten(2).transpose(1, 2)
    return rms(x, values["projector_norm"], gemma=True) @ values["projector"].float()


def prefill(case: Case, values: dict[str, torch.Tensor],
            embeddings: torch.Tensor, image_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run every Gemma decoder layer to generate trusted multimodal KV."""
    p = case.params
    steps, d = embeddings.shape[0], p["head_dim"]
    qh, kvh = p["q_heads"], p["kv_heads"]
    x = embeddings.float()
    positions = torch.arange(steps, device=x.device)
    causal = positions[None, :] <= positions[:, None]
    same_image = (image_blocks[:, None] == image_blocks[None, :]) & (image_blocks[:, None] >= 0)
    k_writes, v_writes = [], []
    for layer in range(p["layers"]):
        normalized = rms(x, values["ln1"][layer], gemma=True)
        query = (normalized @ values["wq"][layer].float().T).view(steps, qh, d)
        key = (normalized @ values["wk"][layer].float().T).view(steps, kvh, d)
        content = (normalized @ values["wv"][layer].float().T).view(steps, kvh, d)
        query = rms(query, values["qn"][layer], gemma=True)
        key = rms(key, values["kn"][layer], gemma=True)
        local = (layer + 1) % 6 != 0
        theta, factor = (10_000.0, 1.0) if local else (1_000_000.0, 8.0)
        query, key = rope(query, positions, theta, factor=factor), rope(key, positions, theta, factor=factor)
        k_writes.append(key.to(torch.bfloat16))
        v_writes.append(content.to(torch.bfloat16))
        scores = torch.einsum("tgqd,sgd->tgqs", query.view(steps, kvh, qh // kvh, d), key) * d ** -0.5
        allowed = causal | same_image
        if local:
            allowed = allowed & (positions[None, :] > positions[:, None] - p["local_window"])
        probabilities = scores.masked_fill(~allowed[:, None, None], -torch.inf).softmax(-1)
        attention = torch.einsum("tgqs,sgd->tgqd", probabilities, content).reshape(steps, qh * d)
        x = x + rms(attention @ values["wo"][layer].float().T, values["ln2"][layer], gemma=True)
        normalized = rms(x, values["ln3"][layer], gemma=True)
        gate, up = normalized @ values["wg"][layer].float().T, normalized @ values["wu"][layer].float().T
        x = x + rms((F.gelu(gate, approximate="tanh") * up) @ values["wd"][layer].float().T,
                    values["ln4"][layer], gemma=True)
    return torch.stack(k_writes), torch.stack(v_writes)


def make_inputs(case: Case, seed: int, device: str) -> dict[str, torch.Tensor]:
    p = case.params
    text_case = replace(case, params=p | {"bits": 16})
    values = gemma.make_inputs(text_case, seed, device)
    vision = vision_inputs(case, seed, device)
    features = image_features(case, vision).reshape(-1, p["hidden"])
    generator = torch.Generator(device=device).manual_seed(seed ^ 0xCA5E)
    prompt = torch.randint(p["vocab"], (p["context"],), generator=generator, device=device)
    embeddings = torch.cat((features, values["embed"][prompt].float() * math.sqrt(p["hidden"])))
    blocks = torch.cat((torch.arange(p["images"], device=device).repeat_interleave(p["image_tokens"]),
                        torch.full((p["context"],), -1, device=device, dtype=torch.int64)))
    values["kcache"], values["vcache"] = prefill(text_case, values, embeddings, blocks)
    return values


def reference(case: Case, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    context = case.params["context"] + case.params["images"] * case.params["image_tokens"]
    text_case = replace(case, params=case.params | {"bits": 16, "context": context})
    return gemma.reference(text_case, values)
