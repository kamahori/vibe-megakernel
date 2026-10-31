from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from megabench import primdb
from megabench.integrations.microbench_gate import check, plan_items
from megabench.primdb import engine_map, phases

# Rejections of earlier VibeSys plans, in their words; none cites a measured row.
PAST_REJECTIONS = [
    {"reasoning": "No candidate exists. Rejected: starting directly with tcgen05/TMA designs, "
                  "because at batch 1 the GEMVs have no tensor-core reuse and that complexity "
                  "adds correctness risk. Rejected: aggressive phase fusion."},
    {"reasoning": "Rejected alternatives: - Tuning CS alone keeps the per-item syncs. - A "
                  "tensor-core QK/PV path is premature while latency and sync dominate. "
                  "Keeping the other phases fixed keeps the attribution clean."},
    {"reasoning": "The rejected alternatives are a tcgen05/tensor-core GEMM and per-phase "
                  "tuning. A tensor-core GEMM does not pay off at b=1, where the step is pure "
                  "GEMV and bandwidth-bound."},
    {"reasoning": "Rejected alternatives: - Tensor-core mma: needs a multi-row tile "
                  "restructure and still pays the same per-element dequant to bf16. - Dropping "
                  "per-element bf16 rounding: risks breaking reference numerics."},
]


class PlanItemTests(unittest.TestCase):
    def test_past_tensor_core_rejections_become_items(self) -> None:
        for plan in PAST_REJECTIONS:
            with self.subTest(plan=plan["reasoning"][:40]):
                items = plan_items(plan)
                self.assertTrue(any("tensor-core" in i or "Tensor-core" in i for i in items), items)
                self.assertFalse(any(i.startswith("Keeping") for i in items), items)

    def test_plan_without_rejections_has_no_items(self) -> None:
        self.assertEqual(plan_items({"reasoning": "Profile first. The parent is HEAD."}), [])


class GateTests(unittest.TestCase):
    def test_gate_needs_files_answers_and_map_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            self.assertEqual(check(ws), [])
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0002.json").write_text(json.dumps(PAST_REJECTIONS[2]))
            problems = check(ws)
            self.assertEqual(len(problems), 3, problems)
            items = (ws / "progress/microbench/items/round-0002.md").read_text()
            self.assertIn("- P1:", items)
            mb = ws / "progress/microbench"
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0002.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("## Status\n### round-0002\n")
            self.assertIn("P1", check(ws)[0])
            n = items.count("- P")
            (mb / "answers/round-0002.md").write_text(" ".join(f"P{i}" for i in range(1, n + 1)))
            self.assertEqual(check(ws), [])


class EngineMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.seed = Path(self.tmp.name) / "seed.jsonl"
        rows = [primdb.normalize({"family": "tc_mma", "primitive": "tcgen05.mma", "p.kind": "f16",
                                  "p.a_src": "smem", "p.cta_group": 1, "p.M": 128, "p.N": n,
                                  "m.fma_per_clk_sm": 32.0 * n, "c.sm_clock_mhz": 1900.0})
                for n in (16, 64, 256)]
        rows.append(primdb.normalize({"family": "mma_sync", "primitive": "mma.sync",
                                      "p.shape": "m16n8k16", "m.fma_per_clk_sm": 1000.0}))
        rows.append(primdb.normalize({"family": "simt", "primitive": "simt.ffma",
                                      "m.flop_per_clk_sm": 256.0}))
        primdb.append(rows, self.seed)
        self.rows = primdb.load(self.seed, Path(self.tmp.name) / "none.jsonl")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mappings_pad_tokens_to_legal_shapes(self) -> None:
        by = {(m["engine"], m["orientation"], m["instr"]): m
              for m in engine_map.mappings(self.rows, 50, 4096, "bfloat16")}
        swap = by[("tcgen05", "cta_group 1, swap-AB (tokens as MMA N)", "M128 N64")]
        self.assertEqual(swap["padded_m"], 64)
        self.assertAlmostEqual(swap["useful"], 50 / 64)
        self.assertEqual(swap["rate"], 32.0 * 64)
        self.assertEqual(by[("tcgen05", "cta_group 1, tokens as MMA M", "M128 N256")]["padded_m"], 128)
        self.assertEqual(by[("SIMT", "FFMA, fp32 accumulate", "-")]["rate"], 128.0)
        # cta_group 2 has no rows here: the map says to measure it.
        self.assertTrue(any(m["rate"] is None for m in by.values()))

    def test_section_reports_bytes_and_rows(self) -> None:
        text = engine_map.section(self.rows, {"M": 50, "N": 4096, "K": 1024, "dtype": "bfloat16"})
        self.assertIn("8.39 MB per call", text)
        self.assertIn("| tcgen05 |", text)
        self.assertIn("no measured GEMM rows", text)

    def test_traced_cases_map_with_the_seed(self) -> None:
        table = phases.load()
        self.assertTrue(table, "case_phases.json is empty")
        rows = primdb.load()
        for case_id, case_phases in table.items():
            with self.subTest(case=case_id):
                self.assertTrue(all({"M", "N", "K", "count"} <= set(p) for p in case_phases))
                out = io.StringIO()
                self.assertEqual(engine_map.main(["--case", case_id], out), 0)
                self.assertIn("E1: ", out.getvalue())


if __name__ == "__main__":
    unittest.main()
