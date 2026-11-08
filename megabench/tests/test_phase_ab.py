"""phase_ab with the GPU parts replaced by fakes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..integrations import phase_ab

CASE = "vla-action-step-pi05-b1"


class Fake:
    def __init__(self, case, fail=False, slow=None) -> None:
        self.log, self.fail = [], fail
        self.slow = slow or {}  # process number -> side that runs in the slow mode
        self.process = -1

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
            base = 1.0 if fn == "a" else 0.9
            return base * (1.3 if self.slow.get(self.process) == fn else 1.0)
        return timer, lambda fn: self.log.append(("warm", fn))

    def time_process(self, parent, variant, reps, iters, warmup):
        self.process += 1
        return phase_ab.time_pair(self, parent, variant, reps, iters, warmup)

    def canary(self):
        return {"gemm_us": 100.0, "copy_us": 200.0}

    def timeline(self, directory, output):
        self.log.append(("timeline", directory.name))
        gain = 0.0 if directory.name == "a" else 5.0
        return {"gemm": {"window_us": 100.0 - gain, "body_us": 90.0, "occurrences": 2},
                "norm": {"window_us": 10.0, "body_us": 9.0, "occurrences": 2}}


def invoke(args, fail=False, slow=None):
    holder = []

    def factory(case):
        holder.append(Fake(case, fail, slow))
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
        code, result, fake = invoke(["--reps", "4", "--phase", "gemm", "--processes", "1"])
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

    def test_processes_and_slow_mode(self) -> None:
        code, result, fake = invoke(["--reps", "2"], slow={1: "a"})
        self.assertEqual(code, 0)
        self.assertEqual(result["processes"], 3)
        self.assertEqual(len(result["e2e"]["a_ms"]), 6)
        stab = result["stability"]
        self.assertEqual(stab["verdict"], "unstable")
        self.assertEqual(stab["a"]["slow_processes"], [1])
        self.assertEqual(stab["b"]["verdict"], "stable")
        # The slow process does not move the score: fast-cluster p50s.
        self.assertAlmostEqual(result["e2e"]["a_p50_ms"], 1.0)
        self.assertAlmostEqual(result["e2e"]["diff_ms"], -0.1)
        self.assertIn("instance", stab["a"]["note"])
        text = phase_ab.format_result(result)
        self.assertIn("UNSTABLE", text)
        self.assertIn("slow process(es) [1]", text)

    def test_stable_when_processes_agree(self) -> None:
        _, result, _ = invoke(["--reps", "2"])
        self.assertEqual(result["stability"]["verdict"], "stable")
        self.assertEqual(result["stability"]["diffs_ms"], [-0.1, -0.1, -0.1])

    def test_rerun_of_needs_an_unstable_result(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("a", "b"):
                (tmp / name).mkdir()
                (tmp / name / "submission.py").write_text(name)
            fakes = []

            def factory(case, slow=None):
                fakes.append(Fake(case, False, slow))
                return fakes[-1]

            args = [str(tmp / "a"), str(tmp / "b"), "--case", CASE, "--reps", "2"]
            phase_ab.main(args + ["--output", str(tmp / "o1")],
                          backend_factory=lambda c: factory(c, {1: "a"}))
            first = tmp / "o1/result.json"
            self.assertEqual(json.loads(first.read_text())["stability"]["verdict"], "unstable")
            code = phase_ab.main(args + ["--output", str(tmp / "o2"), "--rerun-of", str(first)],
                                 backend_factory=factory)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((tmp / "o2/result.json").read_text())["rerun_of"],
                             str(first))
            # A stable result cannot be rerun for free.
            with self.assertRaises(SystemExit):
                phase_ab.main(args + ["--output", str(tmp / "o3"), "--rerun-of",
                                      str(tmp / "o2/result.json")], backend_factory=factory)
            # Changed code cannot be rerun for free.
            (tmp / "b" / "submission.py").write_text("changed")
            with self.assertRaises(SystemExit):
                phase_ab.main(args + ["--output", str(tmp / "o4"), "--rerun-of", str(first)],
                              backend_factory=factory)

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
