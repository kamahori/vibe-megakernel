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
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from .. import primdb
from ..cases import select_cases
from ..integrations.make_agent_workspace import make_workspace
from ..tasks.workloads import TASK_MODULES
from .test_non_p0 import tiny_hybrid, tiny_vision, tiny_speculative
from .test_frontier import development as frontier_development
from .test_glm53 import development as glm53_development


MODULE_BY_FAMILY = TASK_MODULES
SKILLS = ("megakernel-phase-profiling", "megakernel-microbench", "megakernel-regimes",
          "blackwell-tensor-core-design")
MEGABENCH = Path(__file__).resolve().parents[1]


class AgentWorkspaceTests(unittest.TestCase):
    def test_new_snapshots_generate_inputs_and_execute_in_isolation(self) -> None:
        # Import-only checks miss fixtures that replace the Case dataclass.
        frontier_cases = [replace(frontier_development(case), gpus=1, tp=1, ready=True)
                          for case in select_cases('p3')
                          if case.id in ('deepseek-v32-step', 'glm52-step')]
        frontier_cases.append(replace(glm53_development(), gpus=1, tp=1, ready=True))
        for case in (tiny_hybrid(), tiny_vision(), tiny_speculative(), *frontier_cases):
            with self.subTest(case=case.id), tempfile.TemporaryDirectory() as directory:
                with patch('megabench.integrations.make_agent_workspace.select_cases',
                           return_value=[case]):
                    project = make_workspace(case.id, Path(directory) / 'workspace',
                                             agent='plain')
                module = MODULE_BY_FAMILY[case.family]
                frontier = case.family in ('deepseek_v32_step', 'glm52_step', 'glm53_flash_step')
                fixture_args = ', rank=0' if frontier else ''
                reference_args = ', serial=True' if frontier else ''
                result = subprocess.run(
                    [sys.executable, '-c',
                     'import torch; torch.set_num_threads(1); '
                     'from megabench.cases import CASE; '
                     f'from megabench.tasks import {module} as task; '
                     f'values = task.make_inputs(CASE, 17, "cpu"{fixture_args}); '
                     f'outputs = task.reference(CASE, values{reference_args}); '
                     'assert torch.isfinite(outputs["logits"]).all()'],
                    cwd=project, env=os.environ | {'PYTHONPATH': str(project / 'reference')},
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_each_ready_reference_is_isolated_and_importable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for case in select_cases("core"):
                with self.subTest(case=case.id):
                    project = make_workspace(case.id, Path(directory) / case.id,
                                             agent="plain")
                    self.assertEqual(json.loads((project / "case.json").read_text())["id"],
                                     case.id)
                    self.assertFalse((project / "megabench").exists())
                    self.assertFalse((project / "reference/megabench/harness").exists())
                    self.assertTrue((project / "check_candidate").is_file())
                    self.assertTrue((project / "profile_candidate").is_file())
                    self.assertTrue((project / "timeline_candidate").is_file())
                    self.assertTrue((project / "timeline_probe.cuh").is_file())
                    for skill in SKILLS:
                        self.assertTrue(
                            (project / "megakernel_skills" / skill / "SKILL.md").is_file())
                    self.assertTrue((project / "megakernel_skills/megakernel-phase-profiling"
                                     / "scripts/codegen_gate.py").is_file())
                    self.assertIn("megakernel_skills/", (project / "TASK.md").read_text())
                    self.assertTrue((project / "sol_info").is_file())
                    self.assertTrue((project / ".git").is_dir())
                    python_files = sorted((project / "reference/megabench/tasks").rglob("*.py"))
                    self.assertLessEqual(len(python_files), 8)
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

    def test_skills_have_frontmatter_and_vibesys_rules(self) -> None:
        root = MEGABENCH / "skills"
        rules = tomllib.loads((root / ".vibesys.toml").read_text())["rule"]
        ruled = {rule["path"] for rule in rules}
        found = sorted(path.parent for path in root.glob("*/SKILL.md"))
        self.assertEqual({path.name for path in found}, set(SKILLS))
        for path in found:
            with self.subTest(skill=path.name):
                text = (path / "SKILL.md").read_text()
                self.assertTrue(text.startswith("---\n"))
                front = dict(line.split(": ", 1) for line in text.split("---\n", 2)[1].splitlines())
                self.assertEqual(front["name"], path.name)
                self.assertTrue(str(front["description"]).strip())
                self.assertIn(path.name, ruled)

    def test_vibesys_example_config_uses_claude_with_implementer_role(self) -> None:
        config = tomllib.loads(
            (MEGABENCH / "integrations/vibesys-agent.example.toml").read_text())
        self.assertEqual(config["agent"]["cli_provider"], "claude")
        role = config["agent"]["roles"]["implementer"]
        self.assertTrue(role["model"])
        self.assertTrue(role["reasoning_effort"])

    def test_phase_agents_add_the_section_the_tool_and_the_subagent(self) -> None:
        case = select_cases("p0")[0]
        with tempfile.TemporaryDirectory() as directory:
            plain = make_workspace(case.id, Path(directory) / "plain", agent="vibesys",
                                   gpu_socket=Path("/s.sock"))
            phased = make_workspace(case.id, Path(directory) / "phased", agent="vibesys",
                                    gpu_socket=Path("/s.sock"), phase_agents=True)
            self.assertNotIn("Phase table and phase rounds", (plain / "TASK.md").read_text())
            self.assertFalse((plain / "phase_ab").exists())
            self.assertFalse((plain / ".claude/agents/phase.md").exists())
            task = (phased / "TASK.md").read_text()
            self.assertIn("## Phase table and phase rounds", task)
            # A background subagent stops when the implementer turn ends: start them in the foreground.
            self.assertIn("`run_in_background: false`). Put all the Agent calls in one", task)
            self.assertNotIn("subagent_type: phase`, `run_in_background: true`", task)
            self.assertIn("subagent_type: phase", task)
            self.assertIn("progress/phases.md", task)
            self.assertIn("/variants/", (phased / ".gitignore").read_text().splitlines())
            self.assertIn("name: phase", (phased / ".claude/agents/phase.md").read_text())
            wrapper = (phased / "phase_ab").read_text()
            self.assertIn("megabench.integrations.phase_ab", wrapper)
            self.assertIn("gpu_run", wrapper)
            self.assertTrue(os.access(phased / "phase_ab", os.X_OK))

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
            for executable in ("check_candidate", "profile_candidate", "timeline_candidate",
                               "sol_info", "primdb", "engine_map"):
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
            result = subprocess.run(
                [str(project / "primdb"), "q", "tcgen05.mma", "--limit", "1", "--json"],
                cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["origin"], "seed")
            # Rows that earlier runs of the case measured are read as origin `cache`.
            cached = primdb.normalize({"family": "cache_probe", "primitive": "cache.probe",
                                       "p.n": 1, "m.us": 2.0})
            primdb.append([cached], project / "progress/microbench/primdb_cache.jsonl")
            result = subprocess.run(
                [str(project / "primdb"), "q", "cache.probe", "--json"],
                cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["origin"], "cache")
            self.assertIn("primdb reuse",
                          (project / ".claude/agents/microbench.md").read_text())
            self.assertTrue((project / ".claude/agents/microbench.md").is_file())
            self.assertTrue((project / "primbench/schema.md").is_file())
            self.assertIn("Microbench phase", (project / "TASK.md").read_text())
            self.assertIn("progress/engine-map.md", (project / "TASK.md").read_text())
            self.assertIn("## Status", (project / "progress/engine-map.md").read_text())
            self.assertTrue((project / "roadmap/engine-selection.md").is_file())
            self.assertIn("# Open questions", (project / "progress/microbench/open.md").read_text())
            self.assertIn("progress/microbench/open.md", (project / "TASK.md").read_text())
            # Every role reads the rule against a fallback to a design measured slower.
            task_text = (project / "TASK.md").read_text()
            self.assertIn("No fallback to a design measured slower", task_text)
            rule = task_text.split("No fallback to a design measured slower")[1]
            self.assertIn("An in-kernel measurement overrules the proxy rows", rule)
            self.assertIn("built correctly in the same kernel and measured slower", rule)
            self.assertIn("Per-phase timing and measured rounds", task_text)
            for role in ("Candidate contract", "Implementer", "Orchestrator",
                         "Forecast miss", "Judge"):
                self.assertIn(f"**{role}.**", task_text.split("Per-phase timing")[1])
            self.assertIn("in the foreground", task_text)
            for role in ("**Orchestrator.**", "**Implementer.**", "**Judge.**"):
                self.assertIn(role, task_text)
            # mb_build runs in the caller's directory and refuses a build without -o.
            result = subprocess.run([str(project / "mb_build"), "--", "x.cu"],
                                    cwd=project / "progress", capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("-o", result.stderr)
            # The microbench gate: a round with a plan needs its requests and answers, an
            # answer per plan claim, and an engine-map status entry for the round.
            (project / "progress/plans").mkdir(parents=True, exist_ok=True)
            (project / "progress/plans/round-0001.json").write_text(json.dumps(
                {"reasoning": "Rejected alternatives: - Tensor-core designs: premature here."}))
            result = subprocess.run([str(project / "check_candidate"), "--help"],
                                    cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3)
            self.assertIn("progress/microbench/requests/round-0001.md is missing", result.stderr)
            # The microbench agent runs the same gate before it returns its answers.
            result = subprocess.run([str(project / "microbench_check")],
                                    cwd=project, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("progress/microbench/requests/round-0001.md is missing", result.stderr)
            result = subprocess.run([str(project / "microbench_items")],
                                    cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("P1", result.stdout)
            for kind in ("requests", "answers"):
                (project / "progress/microbench" / kind).mkdir(parents=True)
                (project / "progress/microbench" / kind / "round-0001.md").write_text("- item\n")
            result = subprocess.run([str(project / "check_candidate"), "--help"],
                                    cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3)
            self.assertIn("P1", result.stderr)
            # The seeded engine question is due in round 1, with a matched measurement.
            self.assertIn("opened round-0000", (project / "progress/microbench/open.md").read_text())
            (project / "progress/microbench/answers/round-0001.md").write_text("P1: rows ...\n")
            result = subprocess.run([str(project / "check_candidate"), "--help"],
                                    cwd=project, capture_output=True, text=True)
            self.assertIn("Q1 has gone longest", result.stderr)
            (project / "progress/microbench/primdb_local.jsonl").write_text("".join(json.dumps(
                {"id": i, "params": {"engine": e, "phase": "E1"}, "metrics": {"us": us}, "source": {"round": "round-0001", "question": "Q1",
                                                              "conditions": "matched"}}) + "\n"
                for i, e, us in (("eeeeeeeeee", "mma_sync_a", 6.0), ("eeeeeeeeef", "tcgen05_b", 4.0),
                                 ("eeeeeeeeeg", "simt_c", 20.0))))
            (project / "progress/microbench/answers/round-0001.md").write_text(
                "P1 P2: rows ...\n## Q1\neeeeeeeeee\n")
            with (project / "progress/engine-map.md").open("a") as f:
                f.write("### round-0001\n")
            self.assertIn("# Design space", (project / "progress/design-space.md").read_text())
            result = subprocess.run([str(project / "check_candidate"), "--help"],
                                    cwd=project, capture_output=True, text=True)
            self.assertIn("design-space.md has no reconcile entry", result.stderr)
            with (project / "progress/design-space.md").open("a") as f:
                f.write("### round-0001\n")
            # The answers pass, so ./check_candidate runs.
            result = subprocess.run([str(project / "microbench_check")],
                                    cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(project / "check_candidate"), "--help"],
                                    cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

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
