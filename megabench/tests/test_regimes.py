"""Regime sets: weighted aggregation, CLI plumbing, workspace and evaluator."""

from __future__ import annotations

import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ..cases import select_cases
from ..harness import runner
from ..integrations import vibesys_case_evaluator as evaluator
from ..integrations.make_agent_workspace import make_workspace
from ..integrations.make_vibesys_case_task import make_task

ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "dense-step-qwen3-06b-b1-s128"
REGIMES = {"version": 1, "source": {"kind": "test"}, "regimes": [
    {"id": "short", "weight": 1, "params": {"context": 64}},
    {"id": "long", "weight": 3, "params": {"context": 128}}]}


def timing(p50: float) -> dict:
    return {"host_p50_ms": p50, "host_p95_ms": p50 * 2,
            "cuda_event_p50_ms": p50, "cuda_event_p95_ms": p50 * 2}


def shape_result(cand: float, ref: float, graph: float | None = None,
                 status: str = "ok_provisional", audit: str = "within_budget") -> dict:
    out = {"status": status, "build_ms": 1.0,
           "correctness": {"status": "pass" if status == "ok_provisional" else "fail"},
           "candidate_timing": timing(cand), "reference_timing": timing(ref),
           "launch_audit": {"status": audit, "gpu_kernel_count": 1}}
    if graph is not None:
        out["graph_baseline_timing"] = timing(graph)
    return out


def write_regimes(directory: str) -> Path:
    path = Path(directory) / "in-regimes.json"
    path.write_text(json.dumps(REGIMES))
    return path


class CombineTests(unittest.TestCase):
    def rows(self, *results: dict) -> list:
        return list(zip(select_cases("all", [CASE_ID])[0].regimes or
                        [{"id": "a", "weight": 1}, {"id": "b", "weight": 3}], results))

    def test_weights_normalize_and_speedups_are_ratios_of_sums(self) -> None:
        rows = self.rows(shape_result(1.0, 4.0, graph=3.0), shape_result(2.0, 8.0, graph=2.0))
        out = runner._combine_regimes(rows)
        self.assertEqual(out["status"], "ok_provisional")
        self.assertAlmostEqual(out["candidate_timing"]["cuda_event_p50_ms"], 0.25 + 1.5)
        self.assertAlmostEqual(out["candidate_timing"]["cuda_event_p95_ms"], 2 * 1.75)
        self.assertAlmostEqual(out["reference_timing"]["host_p50_ms"], 1 + 6)
        self.assertAlmostEqual(out["speedup_vs_eager_cuda_event"], 7 / 1.75)
        self.assertAlmostEqual(out["speedup_vs_graph_cuda_event"], (0.75 + 1.5) / 1.75)
        self.assertAlmostEqual(out["speedup_vs_best_baseline_cuda_event"], 2.25 / 1.75)
        self.assertEqual([r["weight"] for r in out["regimes"]], [0.25, 0.75])
        self.assertEqual(out["regimes"][1]["graph_cuda_event_p50_ms"], 2.0)
        self.assertEqual(out["correctness"]["status"], "pass")

    def test_status_and_audit_take_the_worst_regime(self) -> None:
        bad = shape_result(1.0, 2.0, status="incorrect", audit="launch_budget_failed")
        out = runner._combine_regimes(self.rows(shape_result(1.0, 2.0), bad))
        self.assertEqual(out["status"], "incorrect")
        self.assertEqual(out["correctness"]["status"], "fail")
        self.assertEqual(out["launch_audit"]["status"], "launch_budget_failed")
        self.assertEqual(out["correctness"]["regimes"], {"a": "pass", "b": "fail"})

    def test_graph_speedup_needs_every_regime(self) -> None:
        out = runner._combine_regimes(self.rows(shape_result(1.0, 2.0, graph=1.5),
                                                shape_result(1.0, 2.0)))
        self.assertNotIn("graph_baseline_timing", out)
        self.assertNotIn("speedup_vs_graph_cuda_event", out)


class EvaluateCaseTests(unittest.TestCase):
    def test_regimes_dispatch_to_the_single_shape_helper_per_regime(self) -> None:
        case = replace(select_cases("all", [CASE_ID])[0], regimes=tuple(REGIMES["regimes"]))
        seen = []

        def fake(sub, submission, record, **kwargs):
            seen.append((sub.params["context"], sub.regimes, record["case"]["params"]["context"]))
            return record | shape_result(1.0, 2.0)

        with tempfile.TemporaryDirectory() as directory:
            submission = Path(directory) / "submission.py"
            submission.write_text("def build(case): pass\n")
            with patch.object(runner, "_evaluate_shape", fake):
                result = runner.evaluate_case(case, submission, device="cpu", trials=1,
                                              warmup=0, reps=1)
        self.assertEqual(seen, [(64, (), 64), (128, (), 128)])
        self.assertEqual(result["case"], case.to_dict())
        self.assertEqual(result["status"], "ok_provisional")
        self.assertEqual(len(result["regimes"]), 2)
        self.assertAlmostEqual(result["speedup_vs_eager_host"], 2.0)

    def test_failing_regime_is_named(self) -> None:
        case = replace(select_cases("all", [CASE_ID])[0], regimes=tuple(REGIMES["regimes"]))

        def fake(sub, *args, **kwargs):
            raise AssertionError("bad logits")

        with tempfile.TemporaryDirectory() as directory:
            submission = Path(directory) / "s.py"
            submission.write_text("")
            with patch.object(runner, "_evaluate_shape", fake):
                with self.assertRaisesRegex(AssertionError, "regime short: bad logits"):
                    runner.evaluate_case(case, submission, device="cpu", trials=1,
                                         warmup=0, reps=1)

    def test_no_regimes_calls_helper_once_and_adds_no_keys(self) -> None:
        case = select_cases("all", [CASE_ID])[0]
        with tempfile.TemporaryDirectory() as directory:
            submission = Path(directory) / "s.py"
            submission.write_text("")
            with patch.object(runner, "_evaluate_shape",
                              lambda sub, s, record, **k: record | {"status": "x"}) as helper:
                result = runner.evaluate_case(case, submission, device="cpu", trials=1,
                                              warmup=0, reps=1)
        self.assertNotIn("regimes", result)
        self.assertNotIn("regimes", result["case"])


class CliTests(unittest.TestCase):
    def test_evaluate_passes_regimes_to_the_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            regimes = write_regimes(directory)
            submission = Path(directory) / "submission.py"
            submission.write_text("")
            cases = []

            def fake(args, case, tempdir):
                cases.append(case)
                return {"status": "ok_provisional", "speedup_vs_best_baseline_cuda_event": 2.0}

            with patch.object(runner, "_run_one", fake), redirect_stdout(io.StringIO()):
                code = runner.main(["evaluate", "--case", CASE_ID, "--submission", str(submission),
                                    "--method", "m", "--session-id", "s", "--regimes", str(regimes),
                                    "--output", str(Path(directory) / "out.jsonl")])
        self.assertEqual(code, 0)
        self.assertEqual([r["id"] for r in cases[0].regimes], ["short", "long"])

    def test_worker_command_carries_the_regime_file(self) -> None:
        case = replace(select_cases("all", [CASE_ID])[0], regimes=tuple(REGIMES["regimes"]))
        args = type("A", (), dict(submission="s.py", device="cpu", trials=1, warmup=0, reps=1,
                                  timeout=5, docker_image=None, graph_baseline=True,
                                  compiled_baselines=False))()
        commands = []

        def fake_worker(cmd, timeout):
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "_run_local_worker", fake_worker):
            runner._run_one(args, case, Path(directory))
            saved = json.loads((Path(directory) / "regimes.json").read_text())
        self.assertIn("--regimes", commands[0])
        self.assertEqual([r["id"] for r in saved["regimes"]], ["short", "long"])

    def test_sol_reports_each_regime_and_a_weighted_floor(self) -> None:
        case = replace(select_cases("all", [CASE_ID])[0], regimes=tuple(REGIMES["regimes"]))
        item = runner._estimate_regimes(case, 8.0, 0.5)
        floors = [sub["hbm_floor_ms"] for sub in item["regimes"]]
        self.assertLess(floors[0], floors[1])
        self.assertAlmostEqual(item["hbm_floor_ms"], 0.25 * floors[0] + 0.75 * floors[1])
        self.assertAlmostEqual(item["gap_to_hbm_floor_x"], 0.5 / item["hbm_floor_ms"])

    def test_sol_cli_accepts_regimes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            regimes = write_regimes(directory)
            out = io.StringIO()
            with redirect_stdout(out):
                code = runner.main(["sol", "--case", CASE_ID, "--regimes", str(regimes), "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(out.getvalue())["cases"][0]["regimes"]), 2)


class EvaluatorTests(unittest.TestCase):
    def test_accuracy_checks_every_regime(self) -> None:
        checked = []
        with tempfile.TemporaryDirectory() as directory:
            regimes = write_regimes(directory)
            submission = Path(directory) / "submission.py"
            submission.write_text("")
            with patch.object(evaluator, "_load_submission",
                              lambda path, case: case), \
                    patch.object(evaluator, "check_trials",
                                 lambda case, cand, device, trials: checked.append(
                                     case.params["context"])), \
                    redirect_stdout(io.StringIO()):
                code = evaluator.main(CASE_ID, ["accuracy", "--submission", str(submission)],
                                      regimes_path=regimes)
        self.assertEqual((code, checked), (0, [64, 128]))

    def test_benchmark_output_drops_row_contexts_and_stays_small(self) -> None:
        rows = [{"id": str(i), "contexts": list(range(256))} for i in range(4)]
        output = evaluator.benchmark_output("c", {"status": "ok", "regimes": rows}, 1.5)
        self.assertTrue(all("contexts" not in row for row in output["result"]["regimes"]))
        self.assertLess(len(json.dumps(output)), 100_000)


class WorkspaceTests(unittest.TestCase):
    def test_task_pins_the_regime_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            regimes = write_regimes(directory)
            task = make_task(CASE_ID, Path(directory) / "p", ROOT, regimes)
            self.assertEqual((task / "regimes.json").read_bytes(), regimes.read_bytes())
            self.assertIn("regimes", (task / "OBJECTIVE.md").read_text())
            argv = sys.argv
            sys.argv = [str(task / "evaluator.py")]
            try:
                namespace = runpy.run_path(str(task / "evaluator.py"), run_name="__mp_main__")
                self.assertIn("main", namespace)
                (task / "regimes.json").write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "regime set changed"):
                    runpy.run_path(str(task / "evaluator.py"), run_name="__mp_main__")
            finally:
                sys.argv = argv

    def test_task_without_regimes_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = make_task(CASE_ID, Path(directory) / "p", ROOT)
            self.assertFalse((task / "regimes.json").exists())
            self.assertNotIn("regimes", (task / "OBJECTIVE.md").read_text())
            self.assertNotIn("regimes_path", (task / "evaluator.py").read_text())

    def test_workspace_bakes_and_commits_regimes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            regimes = write_regimes(directory)
            project = make_workspace(CASE_ID, Path(directory) / "ws", regimes=regimes)
            self.assertEqual((project / "regimes.json").read_bytes(), regimes.read_bytes())
            tracked = subprocess.run(["git", "ls-files"], cwd=project, text=True,
                                     capture_output=True, check=True).stdout.split()
            self.assertIn("regimes.json", tracked)
            self.assertIn(f".vibesys/tasks/{CASE_ID}/regimes.json", tracked)
            case = json.loads((project / "case.json").read_text())
            self.assertEqual([r["id"] for r in case["regimes"]], ["short", "long"])
            self.assertIn("CASE = Case(", (project / "reference/megabench/cases.py").read_text())
            self.assertIn("regimes", (project / "reference/megabench/cases.py").read_text().split(
                "CASE = Case(")[1])
            self.assertIn("`./trace_regimes`", (project / "TASK.md").read_text())
            self.assertIn("--regimes regimes.json", (project / "check_candidate").read_text())
            wrapper = project / "trace_regimes"
            self.assertTrue(os.access(wrapper, os.X_OK))
            self.assertIn(f"megabench.integrations.trace_regimes --case {CASE_ID}",
                          wrapper.read_text())

    def test_workspace_without_regimes_has_no_regime_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = make_workspace(CASE_ID, Path(directory) / "ws", agent="plain")
            self.assertFalse((project / "regimes.json").exists())
            self.assertNotIn("--regimes", (project / "check_candidate").read_text())
            self.assertTrue((project / "trace_regimes").exists())


if __name__ == "__main__":
    unittest.main()
