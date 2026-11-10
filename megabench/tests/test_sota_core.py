"""CPU-only tests for the SOTA harness core (stats, records, report, accuracy, run)."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from megabench.sota import accuracy, backend, records, report, run, timing
from megabench.sota.shapes import Shape, accessed_bytes, bytes_floor


class FakeStep:
    def __init__(self, arm: str, variant: str):
        self.arm, self.variant, self.precision = arm, variant, "bf16"
        self.config = {"pdl": {"launches_with_pdl": 1, "launches_total": 2}}
        self.x = torch.zeros(4)

    def launch(self) -> None:
        self.x += 1

    def outputs(self) -> dict:
        return {}

    def set_chained(self, on: bool) -> None:
        pass

    def close(self) -> None:
        pass


class FakeArm:
    name = "fake"

    def __init__(self, **options):
        self.options = options

    def variants(self) -> tuple[str, ...]:
        return ("a", "b")

    def supports(self, shape, variant):
        return "no b" if variant == "b" else None

    def prepare(self, shape, inputs, variant, device):
        return FakeStep(self.name, variant)


class StatsTest(unittest.TestCase):
    def test_summarize(self):
        s = timing.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual((s["n"], s["median"], s["min"], s["max"]), (5, 3.0, 1.0, 100.0))
        self.assertAlmostEqual(s["mean"], 22.0)
        self.assertEqual(s["mad"], 1.0)
        self.assertEqual(s, timing.summarize([1.0, 2.0, 3.0, 4.0, 100.0]))
        lo, hi = s["ci95_median"]
        self.assertTrue(1.0 <= lo <= 3.0 <= hi <= 100.0)
        self.assertEqual(timing.summarize([5.0] * 10)["ci95_median"], [5.0, 5.0])

    def test_ratio_ci(self):
        r = timing.ratio_ci([2.0] * 9, [1.0] * 9)
        self.assertEqual(r["ratio"], 2.0)
        self.assertEqual(r["ci95"], [2.0, 2.0])
        self.assertEqual(r, timing.ratio_ci([2.0] * 9, [1.0] * 9))

    def test_gap_stats(self):
        iv = [(0.0, 10.0), (12.0, 20.0), (18.0, 30.0), (40.0, 50.0)]
        g = timing.gap_stats(iv, 2, ["a", "b", "b", "c"])
        self.assertEqual(g["kernels_per_step"], 2)
        self.assertEqual(g["sum_kernel_us_per_step"], (10 + 8 + 12 + 10) / 2)
        self.assertEqual(g["window_us_per_step"], 25.0)
        self.assertEqual(g["busy_us_per_step"], (10 + 18 + 10) / 2)  # union 0-10,12-30,40-50
        self.assertEqual(g["gap_us_per_step"], 25.0 - 19.0)
        self.assertEqual(g["overlap_count"], 1)
        self.assertEqual(g["overlap_mean_us"], 2.0)
        self.assertEqual(g["gap_mean_us"], 6.0)  # gaps 2 and 10
        self.assertEqual(g["top_kernels"][0], {"name": "b", "count": 2, "total_us": 20.0})

    def test_throttle_decode(self):
        self.assertEqual(timing.decode_throttle(0x2 | 0x4), ["applications_clocks_setting",
                                                            "sw_power_cap"])


class RecordsTest(unittest.TestCase):
    def test_exclusive_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.jsonl"
            w = records.RecordWriter(path, "run1")
            rec = w.write("env", foo=1)
            w.close()
            self.assertEqual(rec["schema"], "megabench.sota/v1")
            got = records.read_records(path)
            self.assertEqual(got[0]["run_id"], "run1")
            self.assertEqual((got[0]["record_type"], got[0]["foo"]), ("env", 1))
            self.assertIn("timestamp_utc", got[0])
            self.assertIn("slurm_job_id", got[0])
            with self.assertRaises(FileExistsError):
                records.RecordWriter(path, "run2").write("env")

    def test_env_info(self):
        info = records.env_info()
        self.assertIn("host", info)
        self.assertIn("versions", info)


class AccuracyTest(unittest.TestCase):
    def outputs(self, scale=1.0):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(50, generator=g)
        return {"logits": logits * scale, "next_token": logits.argmax(),
                "k_write": torch.randn(2, 3, 4, generator=g),
                "v_write": torch.randn(2, 3, 4, generator=g)}

    def test_compare_identical(self):
        ref = self.outputs()
        m = accuracy.compare(ref, ref)
        self.assertEqual(m["logits"]["rel_l2"], 0.0)
        self.assertTrue(m["logits"]["argmax_match"])
        self.assertEqual(m["logits"]["top5_overlap"], 1.0)
        self.assertTrue(m["next_token"]["match"])
        self.assertEqual(accuracy.grade(m), "pass")

    def test_compare_noise_and_squeeze(self):
        ref = self.outputs()
        got = {k: v.clone() for k, v in ref.items()}
        got["logits"] = (ref["logits"] * 1.03).unsqueeze(0).bfloat16()
        got["next_token"] = ref["next_token"].reshape(1)
        got["k_write"] = ref["k_write"].unsqueeze(0)
        m = accuracy.compare(got, ref)
        self.assertAlmostEqual(m["logits"]["rel_l2"], 0.03, places=2)
        self.assertAlmostEqual(m["logits"]["cosine"], 1.0, places=3)
        self.assertEqual(accuracy.grade(m), "warn")
        got["logits"] = -ref["logits"]
        m = accuracy.compare(got, ref)
        self.assertFalse(m["logits"]["argmax_match"])
        self.assertEqual(accuracy.grade(m), "fail")

    def test_compare_errors(self):
        ref = self.outputs()
        with self.assertRaises(ValueError):
            accuracy.compare({"logits": ref["logits"]}, ref)
        bad = dict(ref, logits=torch.zeros(7))
        with self.assertRaises(ValueError):
            accuracy.compare(bad, ref)

    def test_oracle_band(self):
        fp32 = self.outputs()
        scaled = lambda s: {k: (v * s if v.is_floating_point() else v)  # noqa: E731
                            for k, v in fp32.items()}
        band = accuracy.oracle_band(scaled(1.05), fp32)
        self.assertAlmostEqual(band["logits"], 0.05, places=4)
        for s, want in ((1.08, "pass"), (1.15, "warn"), (1.30, "fail")):
            m = accuracy.compare(scaled(s), fp32)
            self.assertEqual(accuracy.grade_vs_oracle(m, "bf16", band), want)
        # FP32 arms ignore the band and must match the FP32 reference tightly.
        m = accuracy.compare(scaled(1.08), fp32)
        self.assertEqual(accuracy.grade_vs_oracle(m, "fp32", band), "fail")

    def test_snapshot(self):
        inp = {"a": torch.ones(3), "n": 5}
        snap = accuracy.snapshot_inputs(inp)
        self.assertEqual(accuracy.mutated_inputs(inp, snap), [])
        inp["a"][0] = 2
        self.assertEqual(accuracy.mutated_inputs(inp, snap), ["a"])


class ShapesTest(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(bytes_floor(Shape(1, 128))["bytes"], 1207502352)
        self.assertEqual(accessed_bytes(Shape(1, 128)), 1206650880)
        self.assertGreater(accessed_bytes(Shape(1, 256)), accessed_bytes(Shape(1, 128)))


def synthetic_records() -> list[dict]:
    def timing_rec(arm, variant, ms, mode="steady", l2="warm", ratio=1.0):
        stats = {"median": ms, "ci95_median": [ms * 0.99, ms * 1.01]}
        return {"record_type": "timing", "arm": arm, "variant": variant, "shape": "b1-s128",
                "mode": mode, "l2": l2, "stats": stats,
                "ratio_vs_headline": {"ratio": ratio, "ci95": [ratio * 0.9, ratio * 1.1]},
                "sol": {"gbps": 1.2e3}, "pct_measured_bw": 20.0,
                "clock": {"sm_mhz": {"min": 1800, "max": 1900}, "throttle_reasons": []},
                "throttled_rounds": [],
                "config": {"attn_backend": "fa2", "pdl": {"launches_with_pdl": 3, "launches_total": 9}}}

    def prof(arm, variant, gap):
        return {"record_type": "profile", "arm": arm, "variant": variant, "shape": "b1-s128",
                "stats": {"kernels_per_step": 1.0, "gap_us_per_step": gap, "busy_us_per_step": 900.0,
                          "gap_mean_us": 1.5, "gap_p95_us": 3.0, "overlap_count": 2}}
    corr = {"record_type": "correctness", "arm": "megakernel", "variant": "direct",
            "shape": "b1-s128", "own_ref": "fp32", "grade": "pass",
            "vs_fp32": {"logits": {"rel_l2": 1e-5, "argmax_match": True}},
            "vs_bf16": {"logits": {"rel_l2": 1e-2, "argmax_match": True}}}
    return [
        {"record_type": "env", "env": {"gpu": {"name": "B200", "sm_count": 148, "l2_bytes": 1 << 27},
                                        "versions": {"torch": "2.13"}, "git": {"sha": "abcdef0123456"}}},
        {"record_type": "hbm_bw", "median_gbps": 6000.0, "max_gbps": 6500.0},
        {"record_type": "shape", "shape": {"id": "b1-s128"},
         "sol": {"bytes": 1207502352, "floor_ms": 0.151, "bandwidth_tb_s": 8.0}},
        corr, timing_rec("megakernel", "direct", 0.995),
        timing_rec("megakernel", "direct", 1.0, mode="isolated"),
        prof("megakernel", "direct", 5.0),
        timing_rec("sota-graph", "graph-nopdl", 1.5, ratio=1.5), prof("sota-graph", "graph-nopdl", 100.0),
        timing_rec("sota-graph", "graph-pdl", 1.4, ratio=1.4), prof("sota-graph", "graph-pdl", 60.0),
        {"record_type": "skip", "arm": "torch-compile", "variant": "x", "shape": "b1-s128", "reason": "nope"},
        {"record_type": "error", "arm": "sota-graph", "variant": "eager-pdl", "shape": "b1-s128",
         "stage": "prepare", "error": "Traceback\nValueError: boom"},
        # vllm_worker.py's schema: the shape is a dict, and the ITL record has no l2.
        {"record_type": "timing", "mode": "vllm_slope", "arm": "vllm", "variant": "mp0",
         "shape": {"id": "b1-s128", "batch": 1, "context": 128, "model": "qwen3-0.6b"},
         "slope_ms": 2.0, "slope_ms_ci95": [1.9, 2.1], "r2": 0.99, "itl_ms": 2.1,
         "mean_context": 150.0},
        {"record_type": "timing", "mode": "vllm_engine_itl", "arm": "vllm", "variant": "mp0",
         "shape": {"id": "b1-s128", "batch": 1, "context": 128, "model": "qwen3-0.6b"},
         "samples_ms": [2.1], "mean_context": 156.0},
    ]


class ReportTest(unittest.TestCase):
    def test_render(self):
        text = report.render(synthetic_records())
        self.assertIn("megakernel:direct", text)
        self.assertIn("0.9950 [0.9850, 1.0050]", text)
        self.assertIn("1.50 [1.35, 1.65]", text)
        self.assertIn("pass; fp32 1.0e-05", text)
        self.assertIn("argmax=fp32 1/1", text)
        self.assertIn("3/9", text)
        self.assertIn("-100.0", text)                 # graph-pdl vs graph-nopdl is -100 us
        self.assertIn("SKIP torch-compile:x: nope", text)
        self.assertIn("ERROR sota-graph:eager-pdl [prepare]: ValueError: boom", text)
        self.assertIn("vllm:mp0", text)
        self.assertIn("—", text)

    def test_render_empty(self):
        self.assertIn("SOTA decode benchmark", report.render([]))


class RunTest(unittest.TestCase):
    def test_parse(self):
        a = run.parse_args(["--smoke", "--variants", "sota-graph:graph-pdl,graph-nopdl;megakernel:direct",
                            "--megakernel", "mk2=/x/y.cu:case1"])
        self.assertEqual((a.rounds, a.isolated_samples, a.seed_list), (3, 10, [7301]))
        self.assertEqual(a.variant_filter["sota-graph"], ["graph-pdl", "graph-nopdl"])
        self.assertEqual(a.megakernels, [("mk2", "/x/y.cu", "case1")])
        self.assertEqual(run.parse_args([]).arm_list,
                         ["megakernel", "sota-graph", "torch-compile", "vllm"])

    def test_vllm_commands(self):
        a = run.parse_args(["--vllm-variants", "mp0"])
        (v, cmd, out), = run.vllm_commands(a, Shape(1, 128), Path("/r"), "id")
        self.assertEqual(v, "mp0")
        self.assertIn("megabench.sota.vllm_worker", cmd)
        self.assertEqual(out, Path("/r/vllm-mp0-b1-s128.jsonl"))

    def test_dry_run(self):
        fake = {"fake": f"{__name__}:FakeArm"}
        with mock.patch.dict(backend.ARMS, fake, clear=True):
            a = run.parse_args(["--arms", "fake,vllm", "--dry-run", "--variants", "fake:a,b",
                                "--context", "128,256"])
            rows = run.plan_matrix(a)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(run.run(a), 0)
        self.assertEqual([r["status"] for r in rows if r["arm"] == "fake" and r["shape"] == "b1-s128"],
                         ["run", "skip"])
        self.assertEqual(len([r for r in rows if r["arm"] == "vllm"]), 6)
        self.assertIn("b1-s256", buf.getvalue())
        self.assertIn("no b", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
