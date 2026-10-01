"""Small PyTorch primitives shared by the synthetic whole-model P0 oracles."""

from __future__ import annotations

import torch


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
        *, gemma: bool = False) -> torch.Tensor:
    scale = 1.0 + weight.float() if gemma else weight.float()
    return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * scale


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float,
         *, factor: float = 1.0) -> torch.Tensor:
    """Rotate half the head dimensions, with Gemma's optional position scale."""
    width = x.shape[-1]
    inv = theta ** (-torch.arange(0, width, 2, device=x.device,
                                  dtype=torch.float64) / width)
    phase = torch.outer(positions.to(torch.float64) / factor, inv)
    phase = torch.cat((phase, phase), dim=-1)
    cos, sin = phase.cos().float(), phase.sin().float()
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    half = width // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def random_weight(generator: torch.Generator, device: str, shape: tuple[int, ...],
                  fan_in: int) -> torch.Tensor:
    return (torch.randn(shape, generator=generator, device=device) *
            fan_in ** -0.5).to(torch.bfloat16)


def random_norm(generator: torch.Generator, device: str, shape: tuple[int, ...],
                *, gemma: bool = False) -> torch.Tensor:
    values = 0.1 * torch.randn(shape, generator=generator, device=device)
    return (values if gemma else 1.0 + values).to(torch.bfloat16)


def fill_layer_weights(generator: torch.Generator, device: str,
                       shape: tuple[int, ...], fan_in: int) -> torch.Tensor:
    """Avoid a full-stack fp32 temporary for model-sized BF16 weights."""
    out = torch.empty(shape, device=device, dtype=torch.bfloat16)
    for layer in range(shape[0]):
        out[layer].copy_(random_weight(generator, device, shape[1:], fan_in))
    return out
