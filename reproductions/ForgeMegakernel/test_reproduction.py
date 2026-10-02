"""Mutation tests for the independent oracle and the schedule model."""

from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from .candidate import decode_candidate
from .config import ModelConfig, QWEN3_06B, accessed_bytes, weight_elements
from .oracle import (evaluate_cpu, model_bandwidth_utilization,
                     precision_alpha, production_numerical_evidence, slope_us)
from .reference import DecodeResult, decode_reference, empty_cache, random_weights
from .schedule import Schedule, audit_schedule, compile_schedule


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = ModelConfig()
        cls.weights = random_weights(cls.cfg, seed=7)
        cls.tokens = torch.tensor([3, 11], dtype=torch.long)
        cls.positions = torch.tensor([0, 5], dtype=torch.long)
        cls.cache = empty_cache(cls.cfg, 2)

    def evaluate(self, candidate=decode_candidate):
        return evaluate_cpu(self.cfg, self.weights, self.tokens,
                            self.positions, self.cache, candidate)

    def test_candidate_passes_toy_gate(self) -> None:
        verdict = self.evaluate()
        self.assertEqual(verdict.status, "pass", verdict.to_dict())
        self.assertTrue(verdict.metrics["poison_safe"])
        self.assertTrue(verdict.metrics["per_lane_position_safe"])
        self.assertIsNone(verdict.metrics["mbu"])

    def test_teacher_forced_cache_persists_across_steps(self) -> None:
        reference_cache = self.cache
        candidate_cache = self.cache
        for step in range(3):
            tokens = (self.tokens + step) % self.cfg.vocab
            positions = self.positions + step
            reference = decode_reference(self.cfg, self.weights, tokens, positions,
                                         reference_cache, torch.float32)
            candidate = decode_candidate(self.cfg, self.weights, tokens, positions,
                                         candidate_cache)
            self.assertTrue(torch.allclose(candidate.logits, reference.logits,
                                           rtol=1e-5, atol=1e-5))
            self.assertTrue(torch.allclose(candidate.keys, reference.keys,
                                           rtol=1e-5, atol=1e-5))
            reference_cache = (reference.keys, reference.values)
            candidate_cache = (candidate.keys, candidate.values)

    def test_poison_catches_future_cache_read(self) -> None:
        def invalid(cfg, weights, tokens, positions, cache):
            result = decode_candidate(cfg, weights, tokens, positions, cache)
            logits = result.logits + cache[0][:, :, -1].sum() * 1e-5
            return DecodeResult(logits, result.keys, result.values)

        verdict = self.evaluate(invalid)
        self.assertEqual(verdict.status, "fail")
        self.assertFalse(verdict.metrics["poison_safe"])

    def test_kv_rows_catch_midstate_corruption(self) -> None:
        def invalid(cfg, weights, tokens, positions, cache):
            result = decode_candidate(cfg, weights, tokens, positions, cache)
            if tokens.numel() > 1:
                result.keys[0, 1, int(positions[1]), 0, 0] += 1
            return result

        verdict = self.evaluate(invalid)
        self.assertEqual(verdict.status, "fail")
        self.assertFalse(verdict.details["row_bar"])

    def test_lane_check_catches_broadcast_position(self) -> None:
        def invalid(cfg, weights, tokens, positions, cache):
            wrong_positions = positions[0].expand_as(positions)
            return decode_candidate(cfg, weights, tokens, wrong_positions, cache)

        verdict = self.evaluate(invalid)
        self.assertEqual(verdict.status, "fail")
        self.assertFalse(verdict.metrics["per_lane_position_safe"])

    def test_eq2_does_not_count_embedding_table_twice(self) -> None:
        cfg = QWEN3_06B
        self.assertEqual(
            accessed_bytes(cfg, 1, 128),
            2 * (weight_elements(cfg) + 2 * cfg.layers * cfg.kv_heads
                 * cfg.head_dim * 128 + cfg.hidden),
        )
        self.assertLess(accessed_bytes(cfg, 1, 128),
                        2 * (weight_elements(cfg) + cfg.vocab * cfg.hidden))

    def test_precision_projection_and_timing_equations(self) -> None:
        exact = torch.tensor([0.1, 0.2, 0.3])
        narrowed = exact.bfloat16().float()
        self.assertAlmostEqual(precision_alpha(exact, exact, narrowed), 0)
        self.assertAlmostEqual(precision_alpha(narrowed, exact, narrowed), 1)
        self.assertIsNone(precision_alpha(exact, exact, exact))
        self.assertEqual(slope_us(100, 740), 10)
        self.assertAlmostEqual(model_bandwidth_utilization(3350000, 1, 3350), 1)

    def test_production_bars_require_matching_baselines(self) -> None:
        golden = decode_reference(self.cfg, self.weights, self.tokens, self.positions,
                                  self.cache, torch.float64)
        fp32 = decode_reference(self.cfg, self.weights, self.tokens, self.positions,
                                self.cache, torch.float32)
        metrics = production_numerical_evidence(
            fp32, fp32, golden, fp32.logits, fp32.logits, self.positions,
        )
        self.assertTrue(metrics["kv_bar_ok"])
        self.assertTrue(metrics["production_logprob_bars_ok"])
        with self.assertRaisesRegex(ValueError, "same checkpoint taps"):
            production_numerical_evidence(
                fp32, fp32, golden, fp32.logits[:, :-1], fp32.logits, self.positions,
            )

    def test_schedule_counters_and_dependency_audit(self) -> None:
        schedule = compile_schedule(self.cfg, sm_count=8)
        audit = audit_schedule(schedule, self.cfg.layers)
        self.assertTrue(audit["all_counters_once"])
        self.assertTrue(audit["paper_M6_static_targets"])
        self.assertTrue(audit["paper_M7_counter_count_target"])
        streams = [list(stream) for stream in schedule.streams]
        first = streams[0][0]
        streams[0][0] = replace(first, wait_for=(len(schedule.instructions),))
        with self.assertRaises(ValueError):
            audit_schedule(Schedule(tuple(tuple(stream) for stream in streams)), self.cfg.layers)


if __name__ == "__main__":
    unittest.main()
