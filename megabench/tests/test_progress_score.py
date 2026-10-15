"""VibeSys progress scores rank candidates by the intended metric."""

from __future__ import annotations

import unittest

from ..cases import select_cases
from ..integrations.vibesys_case_evaluator import progress_score
from ..sol import estimate_case


def _result(ms: float, speedup: float = 20.0, status: str = "ok_provisional") -> dict:
    return {"status": status, "candidate_timing": {"cuda_event_p50_ms": ms},
            "speedup_vs_best_baseline_cuda_event": speedup}


class ProgressScoreTests(unittest.TestCase):
    def test_sol_score_is_one_plus_fraction_of_floor(self) -> None:
        case = select_cases("p0", ["dense-step-qwen3-06b-b1-s128"])[0]
        floor = estimate_case(case)["hbm_floor_ms"]
        self.assertAlmostEqual(progress_score(case, _result(2 * floor), "sol"), 1.5)
        # Independent of the eager baseline, and strictly ordered by candidate time.
        self.assertEqual(progress_score(case, _result(0.3, 10.0), "sol"),
                         progress_score(case, _result(0.3, 40.0), "sol"))
        self.assertGreater(progress_score(case, _result(0.29), "sol"),
                           progress_score(case, _result(0.30), "sol"))

    def test_work_model_cases_use_the_binding_floor(self) -> None:
        case = select_cases("all", ["world-frame-step-waypoint15-1b-f128"])[0]
        floor = estimate_case(case)["floor_ms"]
        self.assertAlmostEqual(progress_score(case, _result(4 * floor), "sol"), 1.25)

    def test_failures_score_zero_and_speedup_mode_is_unchanged(self) -> None:
        case = select_cases("p0")[0]
        self.assertEqual(progress_score(case, _result(0.3, status="incorrect"), "sol"), 0.0)
        self.assertAlmostEqual(progress_score(case, _result(0.3, 3.0)), 1.75)
        with self.assertRaises(ValueError):
            progress_score(case, _result(0.3), "nope")


if __name__ == "__main__":
    unittest.main()
