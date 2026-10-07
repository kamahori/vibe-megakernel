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


def _compare(expected: dict, actual: dict, case: Case, device: str,
             exact: dict | None = None) -> dict:
    """Check outputs against the reference, or within the BF16 band of ``exact``.

    Without ``exact`` the comparison uses fixed tolerances around ``expected``.
    With the FP32 oracle, each integer element must match the BF16 reference
    or the oracle. A greedy token may also be any token whose FP32 logit is
    within the band of the top logit.
    """
    import torch

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
