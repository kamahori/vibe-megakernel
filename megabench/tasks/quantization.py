"""Explicit native weight formats used by the non-P0 references."""

from __future__ import annotations

import torch


FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MX_BLOCK = 32


def pack_mxfp4(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack contiguous groups of 32 along the last axis into E2M1/E8M0.

    GPT-OSS checkpoints store expert matrices in [input, output] order;
    callers retain that order rather than silently changing the block axis.
    Round ties to an even E2M1 significand. Scales are powers of two with
    exponent bias 127. Finite seeded fixtures never emit reserved scale 255.
    """
    if matrix.shape[-1] % MX_BLOCK:
        raise ValueError("MXFP4 block axis must be divisible by 32")
    groups = matrix.float().unflatten(-1, (-1, MX_BLOCK))
    maximum = groups.abs().amax(-1)
    exponent = torch.ceil(torch.log2((maximum / 6.0).clamp_min(2.0 ** -126)))
    exponent = exponent.clamp(-126, 127).to(torch.int32)
    normalized = torch.ldexp(groups, -exponent[..., None])
    boundaries = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
                              device=matrix.device)
    absolute = normalized.abs().contiguous()
    index = torch.bucketize(absolute, boundaries)
    # The boundaries at indices 1, 3, and 5 have an odd lower code.
    odd_tie = ((absolute == 0.75) | (absolute == 1.75) | (absolute == 3.5))
    index = index + odd_tie.to(index.dtype)
    code = (index | ((normalized < 0).to(index.dtype) << 3)).to(torch.uint8)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed, (exponent + 127).to(torch.uint8)


def unpack_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode to FP32 without changing the checkpoint's matrix orientation."""
    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 blocks and E8M0 scales must be uint8")
    if blocks.shape[:-1] != scales.shape or blocks.shape[-1] != MX_BLOCK // 2:
        raise ValueError("invalid MXFP4 block/scale geometry")
    lut = torch.tensor(FP4_VALUES, device=blocks.device, dtype=torch.float32)
    codes = torch.stack((blocks & 15, blocks >> 4), dim=-1).flatten(-2)
    decoded = torch.ldexp(lut[codes.long()], scales.int()[..., None] - 127)
    # 255 is the native E8M0 NaN encoding, rather than a finite exponent.
    decoded = decoded.masked_fill(scales[..., None] == 255, torch.nan)
    return decoded.flatten(-2)


FP8_BLOCK = 128


def fp8_scale(maximum: torch.Tensor, power_of_two: bool) -> torch.Tensor:
    scale = maximum.float().clamp_min(1e-4) / 448.0
    return torch.exp2(torch.ceil(torch.log2(scale))) if power_of_two else scale


def pack_fp8_activation(value: torch.Tensor, *, power_of_two: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic E4M3 per-token, per-128-channel quantization.

    DeepSeek UE8M0 scales round upward to a power of two; GLM uses FP32
    scales. Padding participates as zero and is discarded from the payload.
    """
    width = value.shape[-1]
    padded = torch.nn.functional.pad(value.float(), (0, (-width) % FP8_BLOCK))
    groups = padded.unflatten(-1, (-1, FP8_BLOCK))
    scales = fp8_scale(groups.abs().amax(-1), power_of_two)
    payload = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.flatten(-2)[..., :width].contiguous(), scales


def unpack_fp8_activation(value: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return value.float() * scales.repeat_interleave(FP8_BLOCK, dim=-1)[..., :value.shape[-1]]


def pack_fp8_weight(value: torch.Tensor, *, power_of_two: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Native E4M3 weight payload plus one scale per 128x128 block."""
    rows, cols = value.shape
    padded = torch.nn.functional.pad(value.float(), (0, (-cols) % FP8_BLOCK, 0, (-rows) % FP8_BLOCK))
    blocks = padded.view(-1, FP8_BLOCK, padded.shape[-1] // FP8_BLOCK, FP8_BLOCK)
    scales = fp8_scale(blocks.abs().amax((1, 3)), power_of_two)
    payload = (blocks / scales[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.reshape(padded.shape)[:rows, :cols].contiguous(), scales


def unpack_fp8_weight(value: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    expanded = scales.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    return value.float() * expanded[:value.shape[0], :value.shape[1]]
