"""The VibeSys progress score is taken against one best baseline pinned per run."""

import tempfile
import unittest
from pathlib import Path

from ..integrations.vibesys_case_evaluator import benchmark_output, pinned_score


def result(cand_ms: float, best_ms: float, status: str = "ok_provisional") -> dict:
    return {"status": status, "candidate_timing": {"cuda_event_p50_ms": cand_ms},
            "speedup_vs_best_baseline_cuda_event": best_ms / cand_ms}


KEY = {"case": {"id": "c"}, "gpu": "B200"}


class PinnedScoreTest(unittest.TestCase):
    def test_first_evaluation_pins_and_later_ones_reuse_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            score, info = pinned_score(result(10.0, 150.0), Path(tmp), KEY)
            self.assertTrue(info["pinned_now"])
            self.assertAlmostEqual(info["pinned_best_baseline_cuda_event_p50_ms"], 150.0)
            self.assertAlmostEqual(score, 1.0 + 15.0 / 16.0)
            _, again = pinned_score(result(10.0, 120.0), Path(tmp), KEY)
            self.assertFalse(again["pinned_now"])
            self.assertAlmostEqual(again["pinned_best_baseline_cuda_event_p50_ms"], 150.0)
            self.assertAlmostEqual(again["measured_best_baseline_cuda_event_p50_ms"], 120.0)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_baseline_drift_does_not_reorder_kernels(self):
        # A slower kernel measured against a high baseline, then a faster one
        # measured after the baseline drifted down: the faster one must win.
        with tempfile.TemporaryDirectory() as tmp:
            slow, _ = pinned_score(result(11.95, 150.98), Path(tmp), KEY)
            fast, _ = pinned_score(result(10.46, 123.68), Path(tmp), KEY)
            self.assertGreater(fast, slow)
        unpinned_slow, _ = pinned_score(result(11.95, 150.98), None, KEY)
        unpinned_fast, _ = pinned_score(result(10.46, 123.68), None, KEY)
        self.assertLess(unpinned_fast, unpinned_slow)  # the bug the pin removes

    def test_a_different_key_pins_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            pinned_score(result(10.0, 150.0), Path(tmp), KEY)
            _, info = pinned_score(result(10.0, 120.0), Path(tmp), {**KEY, "gpu": "other"})
            self.assertTrue(info["pinned_now"])
            self.assertAlmostEqual(info["pinned_best_baseline_cuda_event_p50_ms"], 120.0)

    def test_failed_evaluation_scores_zero_and_pins_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            score, info = pinned_score(result(10.0, 150.0, "non_megakernel"), Path(tmp), KEY)
            self.assertEqual(score, 0.0)
            self.assertIsNone(info)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_without_a_pin_dir_the_measured_baseline_scores(self):
        score, info = pinned_score(result(10.0, 150.0), None, KEY)
        self.assertIsNone(info)
        self.assertAlmostEqual(score, 1.0 + 15.0 / 16.0)

    def test_output_keeps_the_pin_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = result(10.0, 150.0)
            score, info = pinned_score(res, Path(tmp), KEY)
            res["score_baseline"] = info
            out = benchmark_output("c", res)
            self.assertIn("score_baseline", out["result"])


if __name__ == "__main__":
    unittest.main()
