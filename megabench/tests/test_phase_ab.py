"""phase_ab with the GPU parts replaced by fakes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..integrations import phase_ab

CASE = "vla-action-step-pi05-b1"


class Fake:
    def __init__(self, case, fail=False) -> None:
        self.log, self.fail = [], fail

    def load(self, directory):
        self.log.append(("load", directory.name))
        return directory.name

    def check(self, candidate):
        self.log.append(("check", candidate))
        if self.fail:
            raise AssertionError("mismatch")

    def make_timer(self, iters, warmup):
        def timer(fn):
            self.log.append(("time", fn))
            return 1.0 if fn == "a" else 0.9
        return timer, lambda fn: self.log.append(("warm", fn))

    def timeline(self, directory, output):
        self.log.append(("timeline", directory.name))
        gain = 0.0 if directory.name == "a" else 5.0
        return {"gemm": {"window_us": 100.0 - gain, "body_us": 90.0, "occurrences": 2},
                "norm": {"window_us": 10.0, "body_us": 9.0, "occurrences": 2}}


def invoke(args, fail=False):
    holder = []

    def factory(case):
        holder.append(Fake(case, fail))
        return holder[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("a", "b"):
            (tmp / name).mkdir()
            (tmp / name / "submission.py").write_text("")
        out = tmp / "out"
        code = phase_ab.main([str(tmp / "a"), str(tmp / "b"), "--case", CASE, "--output", str(out),
                              *args], backend_factory=factory)
        result = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else None
    return code, result, holder[0]


class PhaseAbTests(unittest.TestCase):
    def test_interleave_alternates_a_and_b(self) -> None:
        self.assertEqual("".join(phase_ab.interleave(3)), "ABABAB")

    def test_arguments_and_result_format(self) -> None:
        code, result, fake = invoke(["--reps", "4", "--phase", "gemm"])
        self.assertEqual(code, 0)
        self.assertTrue(result["correct"])
        self.assertEqual(result["order"], list("ABABABAB"))
        self.assertEqual(len(result["e2e"]["a_ms"]), 4)
        self.assertAlmostEqual(result["e2e"]["diff_ms"], -0.1)
        self.assertEqual([r["phase"] for r in result["phases"]], ["gemm"])
        self.assertEqual(result["phases"][0]["diff_window_us"], -5.0)
        times = [fn for kind, fn in fake.log if kind == "time"]
        self.assertEqual(times, ["a", "b"] * 4)
        self.assertEqual([k for k, _ in fake.log if k == "timeline"], ["timeline", "timeline"])
        self.assertIn("gemm", phase_ab.format_result(result))

    def test_all_phases_reported_without_the_filter(self) -> None:
        _, result, _ = invoke([])
        self.assertEqual([r["phase"] for r in result["phases"]], ["gemm", "norm"])
        self.assertEqual(result["reps"], 5)

    def test_wrong_variant_exits_non_zero_and_skips_timing(self) -> None:
        code, result, fake = invoke([], fail=True)
        self.assertEqual(code, 1)
        self.assertFalse(result["correct"])
        self.assertIn("mismatch", result["error"])
        self.assertNotIn("time", [k for k, _ in fake.log])
        self.assertNotIn("e2e", result)

    def test_same_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "submission.py").write_text("")
            with self.assertRaises(SystemExit):
                phase_ab.main([tmp, tmp, "--case", CASE])


if __name__ == "__main__":
    unittest.main()
