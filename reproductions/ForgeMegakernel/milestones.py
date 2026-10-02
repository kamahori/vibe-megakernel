"""Paper milestone policy applied to evidence from a trusted GPU gate.

The gate owns measurements and source inspection. This module recomputes the
decision from that evidence; an agent's self-reported status is never enough.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .config import ModelConfig, accessed_bytes


MILESTONES = (
    "M0 toolchain", "M1 operators and precision", "M2 fused layer",
    "M3 layer stack", "M4 real-checkpoint decode", "M5 persistent launch",
    "M6 per-SM instruction streams", "M7 counter dependencies",
    "M8 shared-memory buffer pool", "M9 long-run stability (optional)",
)

MILESTONE_CARDS = (
    "Build and execute the target CUDA toolchain; record the compile/run result.",
    "Implement individual decode operators on oracle taps; check against float64 and paired precision probes.",
    "Fuse one complete transformer layer and compare its outputs with the float64 golden.",
    "Run the full layer stack and compare its outputs with the float64 golden.",
    "Decode a real pinned checkpoint, including embedding and LM head, against the float64 golden.",
    "Use one persistent GPU launch for the entire decode step, including LM head; no operator library calls. "
    "Pass finite-logit, poisoned-KV, ragged-lane, KV-row, log-probability, precision, and traffic gates.",
    "Compile typed per-SM instruction streams: at least five types, six instructions per layer, "
    "and queue imbalance at most 1.35; clear the cell's M6 MBU floor.",
    "Replace grid-wide barriers with at least 4L dependency counters. Measure busy fraction at least 0.85, "
    "tail spread at most 0.05, and timeline/event agreement within 25%; clear the M7 floor.",
    "Stage operands in a real shared-memory buffer pool with safe lifetimes; clear the M8 MBU floor.",
    "Optional: run 1000 greedy tokens with finite outputs and at most five drift positions, "
    "then a free 4000-token rollout.",
)

# Guidance is given to the coding agent. Verdicts still come only from the
# independent gate and the evidence requirements in check_milestone().
MILESTONE_GUIDANCE = (
    "Establish a reproducible CUDA build and an executable entry point for the "
    "target model. Keep the host interface and tensor layouts explicit. Inspect "
    "compiler or import failures before changing the decode algorithm; report "
    "only build and execution results that actually ran.",
    "Implement each decode operator on the oracle's fixed inputs before fusing. "
    "Preserve bf16 operator boundaries and fp32 internal arithmetic, including "
    "reductions, norm gains, softmax, and rotary values. Use the paired exact and "
    "narrowed probes to locate an unintended bf16 rounding point; a matching "
    "final token alone does not establish precision.",
    "Fuse one complete transformer layer while keeping its inputs, output, and "
    "KV writes comparable with the float64 golden. If the gate finds the first "
    "divergent KV row, trace the operators before that row and fix the earliest "
    "mismatch before adding more layers.",
    "Execute the full layer stack with the correct residual path and per-layer "
    "KV writes. Use the first failing layer and lane in the gate report to narrow "
    "the edit. Avoid performance tuning while a functional mismatch remains.",
    "Decode the pinned real checkpoint from embedding through final norm, LM "
    "head, and token selection. Preserve each lane's own position and cache "
    "cursor. Compare teacher-forced taps with the checkpoint-matched float64 "
    "golden and inspect the earliest divergent KV row or logit distribution.",
    "Put the entire decode step, including embedding, every layer, KV update, "
    "LM head, and token selection, inside one persistent CUDA launch. Remove "
    "operator-library launches from the path. Use the gate's launch audit, "
    "finite-logit, poisoned-KV, per-lane-position, KV-row, log-probability, "
    "precision, and measured-traffic results; one launch alone is insufficient.",
    "Compile the model into typed instruction streams with explicit operand "
    "descriptors and assign work per SM. Include at least five instruction types "
    "and six instructions per layer, with queue imbalance at most 1.35. First "
    "preserve the M5 checks, then use the schedule audit and measured MBU to "
    "find underused SMs and improve instruction granularity or assignment.",
    "Replace grid-wide phase barriers with producer-consumer dependency "
    "counters on the actual per-SM instruction streams. Require at least 4L "
    "distinct counters, no grid-wide barriers, busy fraction at least 0.85, "
    "and tail spread at most 0.05. Use the device timeline, CUDA-event "
    "agreement, and MBU result to identify waits and load imbalance; keep "
    "counter initialization and reuse safe across steps.",
    "Stage operands in a real CUDA shared-memory buffer pool. Tie each buffer's "
    "reuse to the completion of every consumer, and overlap the next "
    "instruction's load with the current instruction's store where dependencies "
    "permit it. Check lifetime safety, measured HBM traffic, and the M8 MBU "
    "floor while preserving all earlier gates.",
    "Run the optional stability gate on the same candidate: 1000 greedy "
    "tokens with finite outputs and at most five drift positions from the bf16 "
    "reference, followed by a free 4000-token rollout. Trace the first failure "
    "for position overflow, counter wraparound, or buffer reuse errors.",
)


@dataclass(frozen=True)
class Cell:
    model: ModelConfig
    batch: int
    context: int
    peak_gbs: float
    checkpoint_sha256: str
    mbu_floors: Mapping[str, float]
    timing_scope: str = "end-to-end"

    def __post_init__(self) -> None:
        accessed_bytes(self.model, self.batch, self.context)
        if not math.isfinite(self.peak_gbs) or self.peak_gbs <= 0:
            raise ValueError("peak_gbs must be positive")
        if len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if self.timing_scope not in {"kernel-only", "step-no-sampling", "end-to-end"}:
            raise ValueError("invalid pinned timing scope")
        if set(self.mbu_floors) != {"M6", "M7", "M8"}:
            raise ValueError("cell-specific M6, M7, and M8 MBU floors are required")
        floors = [self.mbu_floors[key] for key in ("M6", "M7", "M8")]
        if any(not 0 < value <= 1 for value in floors) or floors != sorted(floors):
            raise ValueError("MBU floors must be positive, at most one, and nondecreasing")


def check_milestone(milestone: int, evidence: Mapping[str, Any], cell: Cell) -> tuple[list[str], dict[str, float]]:
    """Return failed obligations and host-derived metrics; missing data fails closed."""
    if milestone not in range(len(MILESTONES)):
        raise ValueError("unknown milestone")
    issues: list[str] = []
    derived: dict[str, float] = {}

    def truth(name: str) -> None:
        if evidence.get(name) is not True:
            issues.append(f"{name} must be true")

    def number(name: str) -> float | None:
        value = evidence.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            issues.append(f"{name} must be a finite number")
            return None
        return float(value)

    def bound(name: str, *, low: float | None = None, high: float | None = None) -> None:
        value = number(name)
        if value is not None and ((low is not None and value < low) or (high is not None and value > high)):
            issues.append(f"{name} outside [{low}, {high}]")

    if milestone <= 4:
        truth(("toolchain_ok", "operators_ok", "layer_ok", "stack_ok", "real_checkpoint_decode_ok")[milestone])
        if milestone >= 1:
            truth("float64_golden_ok")
        if milestone == 1:
            truth("operator_precision_probes_ok")
        if milestone == 4 and evidence.get("checkpoint_sha256") != cell.checkpoint_sha256:
            issues.append("checkpoint digest differs from pinned campaign checkpoint")
        return issues, derived

    # M5+ uses the full mid-state oracle on the same real checkpoint.
    if evidence.get("checkpoint_sha256") != cell.checkpoint_sha256:
        issues.append("checkpoint digest differs from pinned campaign checkpoint")
    for key in ("gpu_executed", "float64_golden_ok", "finite_logits",
                "poison_bit_identical", "lane_positions_ok", "precision_contract_ok"):
        truth(key)
    kv_b, kv_c = number("kv_b"), number("kv_c")
    if kv_b is not None and kv_c is not None:
        if kv_b < 0 or kv_c < 0 or not kv_c < 2 * kv_b:
            issues.append("KV error fails C < 2B")
    error = number("logprob_error")
    error_max = number("logprob_error_max")
    hf_error = number("hf_bf16_error")
    sgl_error = number("sglang_error")
    sgl_max = number("sglang_error_max")
    if None not in (error, error_max, hf_error, sgl_error, sgl_max):
        if min(error, error_max, hf_error, sgl_error, sgl_max) < 0:
            issues.append("log-probability errors must be nonnegative")
        if error > 5 * hf_error or error > sgl_error or error_max > 2 * sgl_max:
            issues.append("production log-probability bars failed")

    if evidence.get("timing_scope") != cell.timing_scope:
        issues.append("timing_scope differs from pinned campaign scope")
    timings = evidence.get("timing_replicates_us")
    slopes: list[float] = []
    if not isinstance(timings, list) or not timings:
        issues.append("16/80-step timing replicates are required")
    else:
        for pair in timings:
            if (not isinstance(pair, list) or len(pair) != 2 or
                    any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pair) or
                    pair[1] <= pair[0]):
                issues.append("invalid 16/80-step timing replicate")
                break
            slopes.append((pair[1] - pair[0]) / 64)
    if slopes and len(slopes) == len(timings):
        step_us = min(slopes)
        nominal = accessed_bytes(cell.model, cell.batch, cell.context)
        mbu = nominal / (step_us * cell.peak_gbs * 1000)
        derived.update(step_us=step_us, nominal_bytes=float(nominal), mbu=mbu)
        if mbu > 1:
            issues.append("nominal MBU exceeds 100%")
        traffic = number("ncu_read_bytes")
        if traffic is not None:
            rho = traffic / nominal
            derived["read_amplification"] = rho
            if traffic <= 0 or mbu > 1 / rho:
                issues.append("MBU exceeds measured-traffic roofline")

    bound("launches_per_step", low=1, high=1)
    bound("library_calls", low=0, high=0)
    truth("persistent_launch")
    truth("lm_head_inside_launch")

    if milestone >= 6:
        bound("instruction_types", low=5)
        bound("min_instructions_per_layer", low=6)
        bound("queue_imbalance", low=1, high=1.35)
    if milestone >= 7:
        bound("grid_wide_barriers", low=0, high=0)
        bound("distinct_counters", low=4 * cell.model.layers)
        bound("busy_frac", low=0.85, high=1)
        bound("tail_spread", low=0, high=0.05)
        bound("timeline_event_ratio", low=0.75, high=1.25)
    if milestone >= 8:
        truth("shared_memory_pool")
        truth("pool_lifetimes_safe")
    if milestone == 9:
        truth("rollout_1000_finite")
        bound("greedy_drift_1000", low=0, high=5)
        truth("free_rollout_4000_ok")
    if 6 <= milestone <= 8 and "mbu" in derived:
        if derived["mbu"] < cell.mbu_floors[f"M{milestone}"]:
            issues.append(f"M{milestone} MBU floor not met")
    return issues, derived
