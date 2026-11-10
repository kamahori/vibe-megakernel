"""Generated VibeSys task files for one MegaBench case."""

from __future__ import annotations

import json
import math
import runpy
import statistics
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from ..integrations.make_vibesys_case_task import GATE_QUEUE_ALLOWANCE_S, make_task
from ..integrations.vibesys_case_evaluator import (
    METRIC, RELATIVE_NOISE, benchmark_output, progress_score)
from ..integrations.vibesys_noise_calibration import summarize

ROOT = Path(__file__).resolve().parents[2]


class VibeSysTaskTests(unittest.TestCase):
    def test_evaluator_only_evaluates_when_run_as_main(self) -> None:
        # Spawned ranks re-run the parent's main script as __mp_main__; the
        # evaluator must not start a second evaluation (and more spawns) there.
        for case_id in ("dense-step-qwen3-06b-b1-s128", "glm53-flash-step"):
            with self.subTest(case=case_id), tempfile.TemporaryDirectory() as directory:
                task = make_task(case_id, Path(directory), ROOT)
                evaluator = task / "evaluator.py"
                argv = sys.argv
                sys.argv = [str(evaluator)]
                try:
                    namespace = runpy.run_path(str(evaluator), run_name="__mp_main__")
                finally:
                    sys.argv = argv
                self.assertIn("main", namespace)
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(evaluator), run_name="__main__")

    def test_gate_timeouts_leave_room_for_the_queue_only_with_a_launcher(self) -> None:
        case_id = "dense-step-qwen3-06b-b1-s128"
        for launcher, extra in (((), 0), (("srun", "--time=00:20:00"), GATE_QUEUE_ALLOWANCE_S)):
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as directory:
                task = make_task(case_id, Path(directory), ROOT, gate_launcher=launcher)
                config = tomllib.loads((task / "vibesys.input.toml").read_text())
                self.assertEqual(config["accuracy"]["timeout_seconds"], 1200 + extra)
                self.assertEqual(config["benchmark"]["timeout_seconds"], 1800 + extra)

    def test_task_minimizes_kernel_latency_within_declared_noise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = make_task("dense-step-qwen3-06b-b1-s128", Path(directory), ROOT)
            manifest = tomllib.loads((task / "vibesys.input.toml").read_text())
            objectives = tomllib.loads((task / "objectives.toml").read_text())
        self.assertEqual(manifest["benchmark"]["result"]["metric"], METRIC)
        self.assertEqual(objectives["objective"], [{"name": METRIC, "direction": "min"}])
        self.assertEqual(objectives["pareto"]["relative_noise"], RELATIVE_NOISE)
        self.assertTrue(0 < RELATIVE_NOISE < 1)


class BenchmarkOutputTests(unittest.TestCase):
    """The metric VibeSys reads is the candidate's own time, and only for a valid megakernel."""

    def _result(self, status: str = "ok_provisional", **extra) -> dict:
        return {"status": status, "speedup_vs_best_baseline_cuda_event": 25.0,
                "candidate_timing": {"cuda_event_p50_ms": 3.8},
                "reference_timing": {"cuda_event_p50_ms": 95.0}, **extra}

    def test_valid_megakernel_reports_its_latency(self) -> None:
        output = benchmark_output("case", self._result())
        self.assertEqual(output[METRIC], 3.8)
        self.assertEqual(output["status"], "ok_provisional")

    def test_latency_ignores_the_eager_baseline(self) -> None:
        slow_eager = self._result(reference_timing={"cuda_event_p50_ms": 110.0},
                                  speedup_vs_best_baseline_cuda_event=28.9)
        self.assertEqual(benchmark_output("case", slow_eager)[METRIC],
                         benchmark_output("case", self._result())[METRIC])

    def test_invalid_candidates_have_no_latency_anywhere(self) -> None:
        # VibeSys also searches nested JSON for the metric, so the name must not
        # appear anywhere in a rejected result, including per-regime rows.
        regimes = [{"id": "r0", "candidate_cuda_event_p50_ms": 1.0,
                    "speedup_vs_best_baseline_cuda_event": 90.0}]
        for status in ("non_megakernel", "authenticity_unverified", "incorrect", "error"):
            with self.subTest(status=status):
                output = benchmark_output("case", self._result(status, regimes=regimes))
                self.assertNotIn(f'"{METRIC}"', json.dumps(output))
                self.assertEqual(output["progress_score"], 0.0)

    def test_error_without_timing_has_no_latency(self) -> None:
        output = benchmark_output("case", {"status": "error", "reason": "boom"})
        self.assertNotIn(METRIC, output)
        self.assertEqual(output["progress_score"], 0.0)

    def test_progress_score_is_unchanged_for_reports(self) -> None:
        self.assertAlmostEqual(progress_score(self._result()), 1 + 25 / 26)


class NoiseCalibrationTests(unittest.TestCase):
    def _row(self, latency: float | None, gpu: str, case: str = "c") -> dict:
        return {"case": case, "kernel_latency_ms": latency, "eager_ms": 90.0, "gpu_uuid": gpu}

    def test_suggestion_is_one_runs_sigma_within_and_across_gpus(self) -> None:
        rows = [self._row(10.00, "a"), self._row(10.01, "a"),
                self._row(10.21, "b"), self._row(10.22, "b")]
        item = summarize(rows)["c"]
        self.assertEqual((item["gpus"], item["runs"]), (2, 4))
        self.assertAlmostEqual(item["across_gpu_spread"], 10.215 / 10.005 - 1)
        between = statistics.stdev([10.005, 10.215]) / 10.11
        within = statistics.stdev([10.00, 10.01]) / 10.005
        self.assertAlmostEqual(item["run_sigma"], math.hypot(between, within))
        # sigma is 1.47%: rounded up to the next half percent, not set by the 2.1% spread.
        self.assertAlmostEqual(item["suggested_relative_noise"], 0.015)

    def test_calibrated_waypoint_sigma_matches_the_declared_noise(self) -> None:
        # Per-GPU Waypoint means measured on 2026-11-07, one row per GPU.
        means = [31.807, 31.219, 31.748, 31.317, 31.276, 31.189, 31.879, 31.854]
        item = summarize([self._row(ms, f"gpu{i}") for i, ms in enumerate(means)])["c"]
        self.assertAlmostEqual(item["suggested_relative_noise"], RELATIVE_NOISE)

    def test_runs_without_a_valid_latency_are_ignored(self) -> None:
        item = summarize([self._row(5.0, "a"), self._row(5.0, "a"), self._row(None, "b")])["c"]
        self.assertEqual((item["gpus"], item["runs"]), (1, 2))
        self.assertEqual(item["across_gpu_spread"], 0)
        self.assertEqual(item["suggested_relative_noise"], 0)


if __name__ == "__main__":
    unittest.main()
