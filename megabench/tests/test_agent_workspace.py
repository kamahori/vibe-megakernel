"""Check that one-case agent workspaces contain runnable reference snapshots."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from ..cases import select_cases
from ..integrations.make_agent_workspace import make_workspace


MODULE_BY_FAMILY = {
    "dense_step": "dense",
    "moe_step": "moe",
    "quant_step": "gemma",
    "spec_target_step": "eagle3",
}


class AgentWorkspaceTests(unittest.TestCase):
    def test_each_p0_reference_is_isolated_and_importable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for case in select_cases("p0"):
                with self.subTest(case=case.id):
                    project = make_workspace(case.id, Path(directory) / case.id,
                                             agent="plain")
                    self.assertEqual(json.loads((project / "case.json").read_text())["id"],
                                     case.id)
                    self.assertFalse((project / "megabench").exists())
                    self.assertFalse((project / "reference/megabench/harness").exists())
                    self.assertTrue((project / "check_candidate").is_file())
                    self.assertTrue((project / "profile_candidate").is_file())
                    self.assertTrue((project / "sol_info").is_file())
                    self.assertTrue((project / ".git").is_dir())
                    python_files = sorted((project / "reference/megabench/tasks").rglob("*.py"))
                    self.assertLessEqual(len(python_files), 6)
                    environment = os.environ.copy()
                    environment["PYTHONPATH"] = str(project / "reference")
                    result = subprocess.run(
                        [sys.executable, "-c",
                         "from megabench.cases import CASE; "
                         "from megabench.tasks import "
                         f"{MODULE_BY_FAMILY[case.family]}; "
                         f"assert CASE.id == {case.id!r}"],
                        cwd=project, env=environment, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_vibesys_task_uses_trusted_python_and_single_case(self) -> None:
        case = select_cases("p0")[0]
        with tempfile.TemporaryDirectory() as directory:
            project = make_workspace(case.id, Path(directory) / "vibesys",
                                     agent="vibesys")
            history = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "HEAD"],
                cwd=project, capture_output=True, text=True, check=True)
            self.assertNotIn("agent.toml", history.stdout.splitlines())
            task = project / ".vibesys/tasks" / case.id
            config = tomllib.loads((task / "vibesys.input.toml").read_text())
            self.assertEqual(config["agent"]["domain"], "kernel-writing")
            self.assertEqual(config["accuracy"]["command"][0],
                             f".vibesys/tasks/{case.id}/evaluator.sh")
            self.assertIn("TASK.md", (task / "OBJECTIVE.md").read_text())
            self.assertIn("reference/megabench/tasks/dense.py",
                          (task / "OBJECTIVE.md").read_text())
            result = subprocess.run(
                [*config["accuracy"]["command"], "--help"],
                cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for executable in ("check_candidate", "profile_candidate", "sol_info"):
                with self.subTest(executable=executable):
                    result = subprocess.run(
                        [str(project / executable), "--help"],
                        cwd=project, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [str(project / "sol_info"), "--json"],
                cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["cases"][0]["case_id"],
                             case.id)

    def test_kernelagent_adapter_and_no_overwrite(self) -> None:
        case = select_cases("p0")[0]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "kernelagent"
            project = make_workspace(case.id, destination, agent="kernelagent")
            self.assertIn("from kernel import kernel_function",
                          (project / "submission.py").read_text())
            self.assertIn("Write `kernel.py`", (project / "TASK.md").read_text())
            test_source = (project / "test.py").read_text()
            ast.parse(test_source)
            self.assertIn("to('cpu', copy=True)", test_source)
            with self.assertRaises(FileExistsError):
                make_workspace(case.id, destination, agent="kernelagent")


if __name__ == "__main__":
    unittest.main()
