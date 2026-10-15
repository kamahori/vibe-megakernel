"""Validate submission outputs against a task reference."""

from __future__ import annotations

import secrets
import time
from typing import Callable

from ..cases import Case
from ..tasks.workloads import make_inputs, oracle, reference


def _band(want, got, exact, case: Case) -> dict:
    """Bound candidate error by a multiple of the BF16 reference's own error.

    Both errors are measured against the FP32 oracle. The relative L2 bound
    covers distributed drift; the peak bound catches localized corruption.
    """
    import torch

    exact = exact.double()
    error, ref_error = got.double() - exact, want.double() - exact
    scale = exact.norm().clamp_min(torch.finfo(torch.float64).tiny)
    rtol = case.bf16_rtol if want.dtype == torch.bfloat16 else case.rtol
    k = case.noise_factor
    detail = {"relative_error": float(error.norm() / scale),
              "reference_relative_error": float(ref_error.norm() / scale),
              "max_abs_error": float(error.abs().max()),
              "reference_max_abs_error": float(ref_error.abs().max())}
    detail["pass"] = bool(
        detail["relative_error"] <= k * detail["reference_relative_error"] + rtol and
        detail["max_abs_error"] <= (k * detail["reference_max_abs_error"] + case.atol
                                    + rtol * float(exact.abs().max())))
    return detail


def _near_greedy(token, want_logits, exact_logits, case: Case) -> bool:
    """Accept a greedy token whose FP32 logit is within BF16 noise of the top."""
    error = (want_logits.double() - exact_logits.double()).square().mean().sqrt()
    exact_logits = exact_logits.double()
    return bool(exact_logits[token] >= exact_logits.max() - case.noise_factor * error)


# Sequential codebook decoding conditions each code on the earlier ones, so a
# BF16 near-tie early in a frame changes every later code. Grading follows the
# submission's own codes, but only a few near-tie departures are accepted.
MAX_CODEBOOK_DIVERGENCES = 4


def _teacher_forced(case: Case, inputs: dict, expected: dict, exact: dict,
                    actual: dict) -> tuple[dict, dict]:
    """Grade codebook frames against references forced onto the submission's codes.

    The unforced oracle may itself leave the BF16 reference's path at a
    near-tie, so both references always condition on the submitted codes.
    Each code must be the forced BF16 reference's or FP32 oracle's greedy
    choice, or within the BF16 band of the forced oracle's top logit.
    """
    import torch

    from ..tasks.oracle import fp32_reference
    from ..tasks.workloads import _module

    got, want = actual.get("codes"), expected["codes"]
    if (not isinstance(got, torch.Tensor) or got.shape != want.shape or got.dtype != want.dtype
            or got.device != want.device):
        return expected, exact
    module = _module(case)
    forced = got.clone()
    if not torch.equal(got, want):
        expected = module.reference(case, inputs, forced_codes=forced)
    exact = fp32_reference(module, case, inputs, forced_codes=forced)
    departures = []
    for index, code in enumerate(got.tolist()):
        if code in (int(expected["logits"][index].argmax()), int(exact["logits"][index].argmax())):
            continue
        if not _near_greedy(code, expected["logits"][index], exact["logits"][index], case):
            raise AssertionError(f"codes: codebook {index} chose {code}, outside the greedy band")
        departures.append(index)
    if len(departures) > MAX_CODEBOOK_DIVERGENCES:
        raise AssertionError(f"codes: {len(departures)} near-tie departures at {departures}")
    return expected, exact


# TileRT-style per-step gate. Every floating output must point the same way as
# the FP32 oracle and stay within a relative L2 error of it; the greedy token
# must be the reference's or oracle's, or a near-tie of the oracle's top logit.
# Routing and sparse-selection integers are reported, not graded, so any
# implementation of the same math (sparse attention, other summation orders)
# can pass.
TILERT_MIN_COSINE = 0.99
TILERT_MAX_RELATIVE_L2 = 0.15
# Cache writes are graded over the first layers only (DeepSeek-V3.2: three
# dense layers and the first MoE layer); later layers are reported. Near-tie
# sparse-index and expert flips compound with depth: on one 4K prompt the
# BF16 reference's cache writes reach cosine 0.87 to the oracle over all 61
# layers, but stay within 2.3% relative L2 over these four.
TILERT_GRADED_LAYERS = 4
TILERT_CACHE_WRITES = ("kv_write", "pe_write", "index_k_write")
TILERT_DIAGNOSTIC = ("sparse_indices", "expert_ids")


def tilert_metrics(expected: dict, actual: dict, case: Case, device: str, exact: dict) -> dict:
    """Per-output cosine similarity and relative L2, with verdicts.

    Each floating output is measured against both the FP32 oracle and the
    BF16 reference, and passes if it is within the thresholds of either.
    Cache writes are graded over their first ``TILERT_GRADED_LAYERS`` layers
    and reported over all layers.

    In a TP process group the logits are vocabulary shards, so their sums,
    maxima and the greedy token's logit are reduced over the group. Every rank
    reaches every collective, even when its own outputs are malformed.
    """
    import torch
    import torch.distributed as dist

    sharded = dist.is_available() and dist.is_initialized() and case.tp > 1 and dist.get_world_size() == case.tp
    problem = None
    if not isinstance(actual, dict) or set(actual) != set(expected):
        problem = (f"output keys: expected {sorted(expected)}, got "
                   f"{sorted(actual) if isinstance(actual, dict) else type(actual)}")
    else:
        for name, want in expected.items():
            got = actual[name]
            if not isinstance(got, torch.Tensor):
                problem = f"{name}: output must be a Tensor"
            elif got.shape != want.shape or got.dtype != want.dtype:
                problem = (f"{name}: expected {tuple(want.shape)} {want.dtype}, "
                           f"got {tuple(got.shape)} {got.dtype}")
            elif got.device != want.device:
                problem = f"{name}: output is on {got.device}, expected {device}"
            if problem:
                break
    flag = torch.tensor([problem is not None], device=device, dtype=torch.int32)
    if sharded:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if problem or flag.item():
        raise AssertionError(problem or "another rank returned malformed outputs")

    def total(value: torch.Tensor, op=None) -> torch.Tensor:
        if sharded:
            dist.all_reduce(value, op=op or dist.ReduceOp.SUM)
        return value

    def measure(got: torch.Tensor, want: torch.Tensor, oracle: torch.Tensor, reduce: bool) -> dict:
        got, want, oracle = (value.double().flatten() for value in (got, want, oracle))
        sums = torch.stack([got @ oracle, got @ want, got.square().sum(), oracle.square().sum(),
                            want.square().sum(), (got - oracle).square().sum(), (got - want).square().sum(),
                            want @ oracle, (want - oracle).square().sum()])
        if reduce:
            sums = total(sums)
        dot, ref_dot, got_sq, oracle_sq, ref_sq, error_sq, ref_error_sq, agree, drift_sq = sums.tolist()
        tiny = torch.finfo(torch.float64).tiny
        result = {"cosine": dot / max((got_sq * oracle_sq) ** 0.5, tiny),
                  "relative_l2": (error_sq / max(oracle_sq, tiny)) ** 0.5,
                  "cosine_to_reference": ref_dot / max((got_sq * ref_sq) ** 0.5, tiny),
                  "relative_l2_to_reference": (ref_error_sq / max(ref_sq, tiny)) ** 0.5,
                  "reference_cosine": agree / max((ref_sq * oracle_sq) ** 0.5, tiny),
                  "reference_relative_l2": (drift_sq / max(oracle_sq, tiny)) ** 0.5}
        result["pass"] = any(result[cosine] >= TILERT_MIN_COSINE and result[relative] <= TILERT_MAX_RELATIVE_L2
                             for cosine, relative in (("cosine", "relative_l2"),
                                                      ("cosine_to_reference", "relative_l2_to_reference")))
        return result

    details = {}
    for name, want in expected.items():
        if not want.is_floating_point():
            continue
        got, oracle = actual[name], exact[name]
        details[name] = measure(got, want, oracle, name == "logits")
        if name in TILERT_CACHE_WRITES and want.shape[0] > TILERT_GRADED_LAYERS:
            details[name]["graded"] = False
            details[name].pop("pass")
            graded = slice(0, TILERT_GRADED_LAYERS)
            details[f"{name}[:{TILERT_GRADED_LAYERS}]"] = measure(got[graded], want[graded], oracle[graded], False)

    if "next_token" not in expected:
        return details
    # Greedy token: the oracle logit it selects against the oracle maximum.
    token, logits, oracle_logits = int(actual["next_token"]), expected["logits"], exact["logits"].double()
    offset = dist.get_rank() * logits.numel() if sharded else 0
    vocabulary = logits.numel() * (case.tp if sharded else 1)
    picked = torch.zeros((), device=device, dtype=torch.float64)
    if not 0 <= token < vocabulary:
        picked -= torch.inf
    elif offset <= token < offset + logits.numel():
        picked += oracle_logits[token - offset]
    picked = total(picked)
    maximum = total(oracle_logits.max().clone(), dist.ReduceOp.MAX)
    noise = total(torch.stack([(logits.double() - oracle_logits).square().sum(),
                               torch.tensor(float(logits.numel()), device=device, dtype=torch.float64)]))
    margin = case.noise_factor * float((noise[0] / noise[1]).sqrt())
    details["next_token"] = {"token": token, "reference_token": int(expected["next_token"]),
                             "oracle_token": int(exact["next_token"]),
                             "oracle_logit_gap": float(maximum - picked), "near_tie_margin": margin}
    details["next_token"]["pass"] = (token in (details["next_token"]["reference_token"],
                                               details["next_token"]["oracle_token"])
                                     or float(maximum - picked) <= margin)

    for name in (name for name in TILERT_DIAGNOSTIC if name in expected):
        got, want = actual[name].reshape(actual[name].shape[0], -1), expected[name].reshape(expected[name].shape[0], -1)
        recall = [len(set(row.tolist()) & set(ref.tolist()) - {-1}) / max(len(set(ref.tolist()) - {-1}), 1)
                  for row, ref in zip(got.cpu(), want.cpu()) if (ref >= 0).any()]
        details[name] = {"graded": False, "mean_recall": sum(recall) / max(len(recall), 1),
                         "min_recall": min(recall, default=1.0)}
    return details


def _tilert_gate(expected: dict, actual: dict, case: Case, device: str, exact: dict) -> dict:
    """Raise unless every graded output passes ``tilert_metrics``."""
    details = tilert_metrics(expected, actual, case, device, exact)
    failed = [name for name, detail in details.items() if detail.get("pass") is False]
    if failed:
        raise AssertionError("; ".join(f"{name}: {details[name]}" for name in failed))
    return details


def _compare(expected: dict, actual: dict, case: Case, device: str,
             exact: dict | None = None) -> dict:
    """Check outputs against the reference, or within the BF16 band of ``exact``.

    Without ``exact`` the comparison uses fixed tolerances around ``expected``.
    With the FP32 oracle, each integer element must match the BF16 reference
    or the oracle. A greedy token may also be any token whose FP32 logit is
    within the band of the top logit.
    """
    import torch

    if exact is not None and case.gate == "tilert":
        return _tilert_gate(expected, actual, case, device, exact)
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise AssertionError(f"output keys: expected {sorted(expected)}, got "
                             f"{sorted(actual) if isinstance(actual, dict) else type(actual)}")
    details = {}
    for name, want in expected.items():
        got = actual[name]
        if not isinstance(got, torch.Tensor):
            raise AssertionError(f"{name}: output must be a Tensor")
        if got.shape != want.shape or got.dtype != want.dtype:
            raise AssertionError(f"{name}: expected {tuple(want.shape)} {want.dtype}, "
                                 f"got {tuple(got.shape)} {got.dtype}")
        if got.device != want.device:
            raise AssertionError(f"{name}: output is on {got.device}, expected {device}")
        if exact is not None and got.numel() == 0:
            details[name] = {"pass": True}
        elif exact is not None and want.is_floating_point():
            details[name] = _band(want, got, exact[name], case)
            if not details[name]["pass"]:
                raise AssertionError(f"{name}: {details[name]}")
        elif exact is not None:
            oracle_value = exact[name].to(want.dtype)
            neither = (got != want) & (got != oracle_value)
            details[name] = {"mismatched_elements": int((got != want).sum().item()),
                             "oracle_mismatched_elements": int((got != oracle_value).sum().item()),
                             "unmatched_elements": int(neither.sum().item())}
            accepted = details[name]["unmatched_elements"] == 0
            if (not accepted and name == "next_token" and case.tp == 1
                    and expected.get("logits") is not None and expected["logits"].ndim == 1):
                accepted = _near_greedy(int(got), expected["logits"], exact["logits"], case)
            details[name]["pass"] = accepted
            if not accepted:
                raise AssertionError(f"{name}: {details[name]}")
        elif want.is_floating_point():
            delta = (got.float() - want.float()).abs()
            rtol = case.bf16_rtol if want.dtype == torch.bfloat16 else case.rtol
            close = torch.isclose(got.float(), want.float(),
                                  atol=case.atol, rtol=rtol, equal_nan=False)
            details[name] = {"max_abs_error": float(delta.max().item()),
                             "mismatched_elements": int((~close).sum().item())}
            if not bool(close.all()):
                raise AssertionError(f"{name}: {details[name]}")
        else:
            mismatch = int((got != want).sum().item())
            details[name] = {"mismatched_elements": mismatch}
            if mismatch:
                raise AssertionError(f"{name}: {mismatch} integer mismatches")
    return details


def check_trials(case: Case, candidate: Callable, device: str,
                 trials: int) -> tuple[dict, float | None]:
    """Check fresh inputs, input preservation, and special acceptance scenarios."""
    import torch

    result = {"status": "pass", "trials": []}
    first_call_ms: float | None = None
    trial_seeds = [secrets.randbits(32) for _ in range(trials)]
    for trial_index, seed in enumerate(trial_seeds):
        inputs = make_inputs(case, seed, device)
        scenario = "seeded"
        if case.family in ("spec_target_step", "spec_full_iteration"):
            scenario = ("eagle3_raw", "accept_full", "accept_one")[trial_index % 3]
            if scenario != "eagle3_raw":
                if case.family == "spec_target_step":
                    from ..tasks.eagle3 import set_acceptance_scenario
                else:
                    from ..tasks.speculative import set_acceptance_scenario
                accepted_prefix = (case.params["draft_depth"] if
                                   scenario == "accept_full" else 1)
                set_acceptance_scenario(case, inputs, accepted_prefix)
        # Full MoE weights exceed 60 GB. Keep mutation snapshots on host
        # so correctness does not need two full copies on one GPU.
        originals = {name: value.detach().to("cpu", copy=True)
                     for name, value in inputs.items()}
        expected = reference(case, inputs)
        exact = oracle(case, inputs)
        first_start = time.perf_counter() if trial_index == 0 else None
        actual = candidate(inputs)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        if first_start is not None:
            first_call_ms = (time.perf_counter() - first_start) * 1000
        for name in inputs:
            if not torch.equal(inputs[name].cpu(), originals[name]):
                raise AssertionError(f"candidate mutated input {name}")
        if case.family == "tts_frame_step":
            forced = _teacher_forced(case, inputs, expected, exact, actual)
            if forced[0] is not expected:
                scenario = "teacher_forced"
            expected, exact = forced
        details = _compare(expected, actual, case, device, exact)
        trial_record = {"seed": seed, "scenario": scenario, "outputs": details}
        if case.family in ("spec_target_step", "spec_full_iteration"):
            trial_record["spec_accounting"] = {
                "proposed_draft_tokens": case.params["draft_depth"],
                "accepted_draft_tokens": int(expected["accepted_count"].item()),
                "committed_tokens": int(expected["committed_count"].item()),
            }
        result["trials"].append(trial_record)
        del inputs, originals, expected, exact, actual
    return result, first_call_ms
