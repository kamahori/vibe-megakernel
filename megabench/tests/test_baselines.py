"""CPU checks that capture-safe timing baselines match the grading references."""

from __future__ import annotations

import unittest

import torch

from ..baselines import SUPPORTED_FAMILIES, build
from ..harness.correctness import _compare
from ..tasks.eagle3 import set_acceptance_scenario
from ..workloads import make_inputs, oracle, reference
from .test_suite import TINY_P0, TINY_SPEC


def _assert_matches(test: unittest.TestCase, case, inputs) -> None:
    before = {name: value.clone() for name, value in inputs.items()}
    expected = reference(case, inputs)
    exact = oracle(case, inputs)
    actual = build(case, "cpu")(inputs)
    _compare(expected, actual, case, "cpu", exact)
    for name, want in expected.items():
        if not want.is_floating_point():
            test.assertTrue(torch.equal(actual[name], want), name)
    test.assertTrue(all(torch.equal(inputs[k], before[k]) for k in inputs))


class CaptureSafeBaselineTests(unittest.TestCase):
    def test_every_p0_family_has_a_baseline(self) -> None:
        self.assertEqual({case.family for case in TINY_P0}, set(SUPPORTED_FAMILIES))

    def test_baselines_match_reference_within_the_oracle_band(self) -> None:
        for case in TINY_P0:
            for seed in (3, 11):
                with self.subTest(case=case.id, seed=seed), torch.inference_mode():
                    _assert_matches(self, case, make_inputs(case, seed))

    def test_speculative_baseline_follows_every_acceptance_length(self) -> None:
        for accepted in range(TINY_SPEC.params["draft_depth"] + 1):
            with self.subTest(accepted=accepted), torch.inference_mode():
                inputs = make_inputs(TINY_SPEC, 23)
                set_acceptance_scenario(TINY_SPEC, inputs, accepted)
                _assert_matches(self, TINY_SPEC, inputs)


if __name__ == "__main__":
    unittest.main()
