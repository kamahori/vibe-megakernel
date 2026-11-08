"""gpu_noise: stability verdicts, throttle decoding, nvidia-smi parsing and window summaries."""

from __future__ import annotations

import unittest

from ..integrations import gpu_noise


class GpuNoiseTests(unittest.TestCase):
    def test_spread_stable(self) -> None:
        s = gpu_noise.spread([7.58, 7.60, 7.59])
        self.assertEqual(s["verdict"], "stable")
        self.assertEqual(s["slow_processes"], [])
        self.assertAlmostEqual(s["fast_p50_ms"], 7.59)

    def test_spread_slow_mode_majority(self) -> None:
        # Two of three processes slow: the median is slow, the fast cluster is not.
        s = gpu_noise.spread([10.2, 7.59, 10.21])
        self.assertEqual(s["verdict"], "unstable")
        self.assertEqual(s["slow_processes"], [0, 2])
        self.assertAlmostEqual(s["fast_p50_ms"], 7.59)
        self.assertAlmostEqual(s["median_ms"], 10.2)

    def test_canary_note(self) -> None:
        normal = {"gemm_us": 100, "copy_us": 200}
        slow = {"gemm_us": 150, "copy_us": 300}
        self.assertIn("instance", gpu_noise.canary_note([normal, normal], [1]))
        self.assertIn("GPU or node", gpu_noise.canary_note([normal, slow], [1]))
        self.assertEqual(gpu_noise.canary_note([None, None], [1]), "")
        self.assertEqual(gpu_noise.canary_note([normal, normal], []), "")

    def test_decode_reasons_ignores_idle(self) -> None:
        self.assertEqual(gpu_noise.decode_reasons(0x1), [])
        self.assertEqual(gpu_noise.decode_reasons(0x5), ["software power cap"])

    def test_parse_and_summarize(self) -> None:
        lines = []
        for tick in range(3):
            lines.append(f"0, GPU-aaa, 1965, 3996, 700, 50, 0x{0x4 if tick == 1 else 0x0:x}, 95")
            lines.append("1, GPU-bbb, 1900, 3996, 500, 45, 0x0, 80")
            lines.append("2, GPU-ccc, 120, 3996, 140, 25, 0x1, 0")
        rows = [gpu_noise.parse_line(x) for x in lines]
        self.assertIsNone(gpu_noise.parse_line("index, uuid"))
        summary = gpu_noise.summarize_window(rows, 0)
        self.assertEqual(summary["gpu"]["uuid"], "GPU-aaa")
        self.assertEqual(summary["gpu"]["throttle"], ["software power cap"])
        self.assertEqual(summary["node"]["other_gpus"], 2)
        self.assertEqual(summary["node"]["other_gpus_busy"], 1)
        self.assertEqual(summary["node"]["node_power_w_max"], 1340)
        text = gpu_noise.format_window("job-1", summary)
        self.assertIn("1 of 2 busy", text)
        self.assertIn("software power cap", text)

    def test_check_report_uses_evaluate_as_process_zero(self) -> None:
        record = {"candidate_timing": {"cuda_event_p50_ms": 10.2}, "build_ms": 6400,
                  "first_call_ms_including_jit": 12.0}
        probes = [{"p50_ms": 7.6, "canary": {"gemm_us": 100, "copy_us": 200}},
                  {"p50_ms": 7.61, "canary": {"gemm_us": 101, "copy_us": 199}},
                  {"error": "boom"}]
        stab = gpu_noise.check_report(record, probes, 3.0)
        self.assertEqual(stab["slow_processes"], [0])
        self.assertAlmostEqual(stab["fast_p50_ms"], 7.605)
        self.assertEqual(stab["probe_errors"], ["boom"])
        self.assertIn("UNSTABLE", gpu_noise.format_stability("check", stab, stab["processes"]))


class RerunTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "submission.py").write_text("x = 1\n")
        (self.root / "progress").mkdir()
        (self.root / "reports").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def report(self, name, verdict, rerun_of=None):
        import json

        ident = gpu_noise.tree_digest(self.root)
        path = self.root / "reports" / f"{name}.jsonl"
        path.write_text("{}\n")
        path.with_suffix(".stability.json").write_text(json.dumps(
            {"verdict": verdict, "identity": ident, "rerun_of": rerun_of}))
        return path

    def test_unstable_same_code_is_a_rerun(self) -> None:
        prior = self.report("a", "unstable")
        ok, _ = gpu_noise.rerun_status(prior, gpu_noise.tree_digest(self.root))
        self.assertTrue(ok)

    def test_notes_do_not_change_identity(self) -> None:
        ident = gpu_noise.tree_digest(self.root)
        (self.root / "progress" / "round.md").write_text("notes")
        (self.root / "reports" / "x.json").write_text("{}")
        self.assertEqual(ident, gpu_noise.tree_digest(self.root))

    def test_refusals(self) -> None:
        stable = self.report("s", "stable")
        ok, why = gpu_noise.rerun_status(stable, gpu_noise.tree_digest(self.root))
        self.assertFalse(ok)
        self.assertIn("not unstable", why)
        prior = self.report("u", "unstable")
        (self.root / "submission.py").write_text("x = 2\n")
        ok, why = gpu_noise.rerun_status(prior, gpu_noise.tree_digest(self.root))
        self.assertFalse(ok)
        self.assertIn("other code", why)
        ok, why = gpu_noise.rerun_status(self.root / "none.jsonl", "")
        self.assertFalse(ok)

    def test_rerun_limit(self) -> None:
        first = self.report("r0", "unstable")
        second = self.report("r1", "unstable", str(first))
        third = self.report("r2", "unstable", str(second))
        ident = gpu_noise.tree_digest(self.root)
        self.assertTrue(gpu_noise.rerun_status(second, ident)[0])
        ok, why = gpu_noise.rerun_status(third, ident)
        self.assertFalse(ok)
        self.assertIn("at most", why)


if __name__ == "__main__":
    unittest.main()
