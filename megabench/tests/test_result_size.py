"""Evaluation results stay small enough for agent frameworks to read back."""

from __future__ import annotations

import json
import unittest

from ..harness.benchmark import _kernel_name_summary
from ..integrations.vibesys_case_evaluator import benchmark_output

# VibeSys reads its benchmark result file through a 100,000-character capture.
VIBESYS_CAPTURE_CHARS = 100_000
LONG_NAME = "void at::native::vectorized_elementwise_kernel<4, " + "x" * 400


def _eager_reference_names() -> list[str]:
    return [f"{LONG_NAME}{index % 40}" for index in range(40_000)]


class ResultSizeTests(unittest.TestCase):
    def test_small_launch_sets_list_every_name(self) -> None:
        for names in ([], ["megakernel"], [f"k{index}" for index in range(32)]):
            self.assertEqual(_kernel_name_summary(names), {"gpu_kernel_names": names})

    def test_large_launch_sets_are_summarized_by_frequency(self) -> None:
        names = _eager_reference_names()
        summary = _kernel_name_summary(names)
        self.assertTrue(summary["gpu_kernel_names_truncated"])
        self.assertEqual(summary["gpu_kernel_names"], names[:32])
        self.assertEqual(summary["distinct_gpu_kernel_names"], 40)
        self.assertEqual(len(summary["top_gpu_kernel_names"]), 16)
        self.assertTrue(all(item["count"] == 1_000 for item in summary["top_gpu_kernel_names"]))
        self.assertLess(len(json.dumps(summary)), 30_000)

    def test_multi_gpu_benchmark_output_fits_the_capture(self) -> None:
        reference_audit = {"status": "launch_budget_failed", "gpu_kernel_count": 40_000,
                           **_kernel_name_summary(_eager_reference_names())}
        candidate_audit = {"status": "within_budget", "gpu_kernel_count": 1,
                           "gpu_kernel_names": ["megakernel"]}
        ranks = [{"rank": rank, "status": "pass", "launch_audit": candidate_audit,
                  "reference_launch_audit": reference_audit} for rank in range(4)]
        result = {"status": "ok_provisional", "ranks": ranks,
                  "launch_audit": {"status": "within_budget", "per_rank": [candidate_audit] * 4},
                  "speedup_vs_best_baseline_cuda_event": 28.5}
        output = benchmark_output("glm53-flash-step", result, 1.9658)
        self.assertNotIn("ranks", output["result"])
        self.assertEqual(output["result"]["status"], "ok_provisional")
        self.assertEqual(output["progress_score"], 1.9658)
        self.assertLess(len(json.dumps(output)), VIBESYS_CAPTURE_CHARS)


if __name__ == "__main__":
    unittest.main()
