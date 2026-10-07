"""Validate the public speed-of-light estimate against the P0 task contracts."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ..cases import Case, select_cases
from ..harness.runner import main
from ..sol import estimate_case


class SolTests(unittest.TestCase):
    def test_p0_byte_counts_match_issue_7_floor(self) -> None:
        expected = {
            "dense-step-qwen3-06b-b1-s128": 1207502352,
            "moe-step-qwen3-30b-a3b-b1-s128": 6097028624,
            "quant-step-gemma3-4b-w8-b1-s128": 4574608400,
            "quant-step-gemma3-4b-w4-b1-s128": 2970287120,
            "spec-target-step-llama31-8b-k4": 15029912680,
        }
        for case in select_cases("p0"):
            with self.subTest(case=case.id):
                row = estimate_case(case)
                self.assertEqual(row["minimum_bytes"], expected[case.id])
                self.assertEqual(sum(x["bytes"] for x in row["components"]),
                                 row["minimum_bytes"])
                self.assertAlmostEqual(row["hbm_floor_ms"],
                                       expected[case.id] / 8e9)

    def test_moe_counts_only_selected_experts_and_gemma_window(self) -> None:
        moe = select_cases("p0", ["moe-step-qwen3-30b-a3b-b1-s128"])[0]
        base = estimate_case(moe)
        doubled = Case(**(moe.to_dict() | {
            "params": moe.params | {"topk": 16},
        }))
        change = estimate_case(doubled)["minimum_bytes"] - base["minimum_bytes"]
        self.assertEqual(change, 8 * 3 * moe.params["layers"] *
                         moe.params["intermediate"] * moe.params["hidden"] * 2)

        gemma = select_cases("p0", ["quant-step-gemma3-4b-w4-b1-s128"])[0]
        short_window = Case(**(gemma.to_dict() | {
            "params": gemma.params | {"local_window": 33},
        }))
        parts = {part["name"]: part["bytes"]
                 for part in estimate_case(short_window)["components"]}
        local_layers = gemma.params["layers"] - gemma.params["layers"] // 6
        positions = local_layers * 32 + gemma.params["layers"] // 6 * 128
        self.assertEqual(parts["old_kv_cache_reads"],
                         2 * positions * gemma.params["kv_heads"] *
                         gemma.params["head_dim"] * 2)

    def test_cli_json_result_gap_and_planned_case(self) -> None:
        case = select_cases("p0")[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            path.write_text(json.dumps({
                "case": case.to_dict(),
                "candidate_timing": {"cuda_event_p50_ms": 2.0},
            }) + "\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["sol", "--case", case.id,
                                       "--result", str(path), "--json"]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["method"], "optimistic_one_read_hbm_floor")
            item = report["cases"][0]
            self.assertAlmostEqual(item["gap_to_hbm_floor_x"],
                                   2.0 / item["hbm_floor_ms"])
            self.assertAlmostEqual(item["percent_of_hbm_sol"],
                                   100 / item["gap_to_hbm_floor_x"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["sol", "--suite", "planned", "--json"]), 0)
            self.assertTrue(all(row["status"] == "unavailable"
                                for row in json.loads(output.getvalue())["cases"]))

    def test_bandwidth_scales_floor_and_rejects_invalid_values(self) -> None:
        case = select_cases("p0")[0]
        base = estimate_case(case)
        self.assertAlmostEqual(estimate_case(case, 4)["hbm_floor_ms"],
                               2 * base["hbm_floor_ms"])
        with self.assertRaises(ValueError):
            estimate_case(case, 0)
        with self.assertRaises(ValueError):
            estimate_case(case, measured_ms=float("nan"))


if __name__ == "__main__":
    unittest.main()
