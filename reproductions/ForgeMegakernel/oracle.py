"""Candidate-independent mid-state diagnostics from ForgeMegakernel §4.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch

from .config import ModelConfig, accessed_bytes, config_hash
from .reference import DecodeResult, decode_reference


Candidate = Callable[[ModelConfig, dict[str, torch.Tensor], torch.Tensor,
                      torch.Tensor, tuple[torch.Tensor, torch.Tensor]], DecodeResult]


def relative_row_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Eq. 4, averaged over rows, with the paper's 1e-6 denominator."""
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("matching tensors with a row dimension required")
    x, y = a.double().reshape(-1, a.shape[-1]), b.double().reshape(-1, b.shape[-1])
    return float((2 * (x - y).abs() / (x.abs() + y.abs() + 1e-6)).mean(-1).mean())


def top_logprob_error(result: torch.Tensor, golden: torch.Tensor, topk: int = 64) -> tuple[float, float]:
    """Eq. 5 on the golden's own top-k logits at each lane/tap."""
    if result.shape != golden.shape or result.ndim != 2:
        raise ValueError("matching [tap, vocab] logits required")
    indices = golden.topk(min(topk, golden.shape[-1]), dim=-1).indices
    a = torch.log_softmax(result.double(), dim=-1).gather(-1, indices)
    b = torch.log_softmax(golden.double(), dim=-1).gather(-1, indices)
    error = (a - b).abs()
    return float(error.mean()), float(error.max())


def precision_alpha(kernel: torch.Tensor, exact: torch.Tensor, narrowed: torch.Tensor) -> float | None:
    """Eq. 6 projection. None means the precision probe cannot distinguish widths."""
    difference = (narrowed.double() - exact.double()).reshape(-1)
    denominator = torch.dot(difference, difference)
    if denominator == 0:
        return None
    return float(torch.dot((kernel.double() - exact.double()).reshape(-1), difference) / denominator)


def slope_us(time_16_us: float, time_80_us: float) -> float:
    if not math.isfinite(time_16_us) or not math.isfinite(time_80_us) or time_80_us <= time_16_us:
        raise ValueError("invalid cumulative timings")
    return (time_80_us - time_16_us) / 64


def model_bandwidth_utilization(bytes_per_step: int, step_us: float, peak_gbs: float) -> float:
    if bytes_per_step <= 0 or step_us <= 0 or peak_gbs <= 0:
        raise ValueError("positive bytes, time and bandwidth required")
    # GB/s * microseconds = 10^3 bytes.
    return bytes_per_step / (step_us * peak_gbs * 1000)


@dataclass
class GateResult:
    status: str
    suite: str
    summary: str
    metrics: dict[str, float | int | str | bool | None]
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _written_rows(result: DecodeResult, positions: torch.Tensor) -> torch.Tensor:
    rows = []
    for layer in range(result.keys.shape[0]):
        for lane, position in enumerate(positions.tolist()):
            rows.extend((result.keys[layer, lane, position], result.values[layer, lane, position]))
    return torch.stack(rows)


def production_numerical_evidence(
    actual: DecodeResult,
    pytorch_fp32: DecodeResult,
    golden_fp64: DecodeResult,
    hf_bf16_logits: torch.Tensor,
    sglang_logits: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, float | bool]:
    """Paper Eq. 4/5 bars on checkpoint-matched, teacher-forced outputs.

    The caller must establish checkpoint/tap identity and arithmetic provenance.
    Missing production baselines are an error, never an implicit pass.
    """
    if any(result.logits.shape != golden_fp64.logits.shape
           for result in (actual, pytorch_fp32)):
        raise ValueError("candidate, PyTorch and golden logits must have the same taps")
    if (not isinstance(hf_bf16_logits, torch.Tensor) or
            not isinstance(sglang_logits, torch.Tensor) or
            hf_bf16_logits.shape != golden_fp64.logits.shape or
            sglang_logits.shape != golden_fp64.logits.shape):
        raise ValueError("production baselines must have the same checkpoint taps")
    golden_rows = _written_rows(golden_fp64, positions)
    fp32_rows = _written_rows(pytorch_fp32, positions)
    actual_rows = _written_rows(actual, positions)
    kv_b = relative_row_error(fp32_rows, golden_rows)
    kv_c = relative_row_error(actual_rows, fp32_rows)
    error, error_max = top_logprob_error(actual.logits, golden_fp64.logits)
    hf_error, _ = top_logprob_error(hf_bf16_logits, golden_fp64.logits)
    sglang_error, sglang_error_max = top_logprob_error(sglang_logits, golden_fp64.logits)
    return {
        "kv_b": kv_b, "kv_c": kv_c,
        "logprob_error": error, "logprob_error_max": error_max,
        "hf_bf16_error": hf_error,
        "sglang_error": sglang_error, "sglang_error_max": sglang_error_max,
        "kv_bar_ok": kv_c < 2 * kv_b,
        "production_logprob_bars_ok": (
            error <= 5 * hf_error and error <= sglang_error
            and error_max <= 2 * sglang_error_max
        ),
    }


def evaluate_cpu(
    cfg: ModelConfig, weights: dict[str, torch.Tensor], tokens: torch.Tensor,
    positions: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor],
    candidate: Candidate,
) -> GateResult:
    """Toy CPU functional gate; paper milestones need checkpoint/GPU evidence."""
    golden = decode_reference(cfg, weights, tokens, positions, cache, torch.float64)
    reference = decode_reference(cfg, weights, tokens, positions, cache, torch.float32)
    actual = candidate(cfg, weights, tokens, positions, cache)
    if actual.logits.shape != golden.logits.shape:
        return GateResult("fail", "toy-cpu", "wrong logit shape", {}, {})

    golden_rows = _written_rows(golden, positions)
    reference_rows = _written_rows(reference, positions)
    actual_rows = _written_rows(actual, positions)
    b = relative_row_error(reference_rows, golden_rows)
    c = relative_row_error(actual_rows, reference_rows)
    log_error, log_error_max = top_logprob_error(actual.logits, golden.logits)
    ref_error, _ = top_logprob_error(reference.logits, golden.logits)
    finite = bool(torch.isfinite(actual.logits).all())

    # Writes beyond each lane's position are poisoned; valid logits must be bit-identical.
    poisoned = tuple(tensor.clone() for tensor in cache)
    for tensor in poisoned:
        for lane, position in enumerate(positions.tolist()):
            tensor[:, lane, position + 1:] = 100.0
    poisoned_result = candidate(cfg, weights, tokens, positions, poisoned)
    poison_safe = bool(torch.equal(actual.logits, poisoned_result.logits))

    # A batched call must equal independent calls with the same lane-local position.
    lane_safe = True
    for lane in range(tokens.numel()):
        sliced_cache = tuple(tensor[:, lane:lane + 1].clone() for tensor in cache)
        alone = candidate(cfg, weights, tokens[lane:lane + 1],
                          positions[lane:lane + 1], sliced_cache)
        lane_safe &= bool(torch.allclose(actual.logits[lane], alone.logits[0], rtol=1e-5, atol=1e-5))

    # The paper's full logit bar requires HF bf16 and SGLang on real checkpoints.
    # Here only the independent fp64/float32 and mid-state diagnostics are claimed.
    row_pass = c < 2 * b if b > 0 else c == 0
    passed = finite and poison_safe and lane_safe and row_pass and log_error <= max(5 * ref_error, 1e-6)
    return GateResult(
        "pass" if passed else "fail", "toy-cpu",
        "CPU functional checks passed" if passed else "CPU functional check failed",
        {
            "config_hash": config_hash(cfg, tokens.numel(), int(positions.max()) + 1),
            "eq2_nominal_bytes_bf16": accessed_bytes(cfg, tokens.numel(), int(positions.max()) + 1),
            "kv_C_candidate_vs_fp32": c, "kv_B_fp32_vs_fp64": b,
            "logprob_error_vs_fp64": log_error, "logprob_max_vs_fp64": log_error_max,
            "fp32_logprob_error_vs_fp64": ref_error,
            "finite": finite, "poison_safe": poison_safe, "per_lane_position_safe": lane_safe,
            "gpu_launch_count": None, "mbu": None,
        },
        {"row_bar": row_pass, "paper_milestones": "abstain: toy weights, no checkpoint or GPU megakernel",
         "production_logit_bar": "abstain: no checkpoint-matched HF bf16/SGLang baseline"},
    )
