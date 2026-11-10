"""Generated VibeSys task files for one MegaBench case."""

from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path

from ..integrations.make_vibesys_case_task import GATE_QUEUE_ALLOWANCE_S, make_task

ROOT = Path(__file__).resolve().parents[2]


class VibeSysTaskTests(unittest.TestCase):
    def test_evaluator_only_evaluates_when_run_as_main(self) -> None:
        # Spawned ranks re-run the parent's main script as __mp_main__; the
        # evaluator must not start a second evaluation (and more spawns) there.
        for case_id in ("dense-step-qwen3-06b-b1-s128", "glm53-flash-step"):
            with self.subTest(case=case_id), tempfile.TemporaryDirectory() as directory:
                task = make_task(case_id, Path(directory), ROOT)
                evaluator = task / "evaluator.py"
                argv = sys.argv
                sys.argv = [str(evaluator)]
                try:
                    namespace = runpy.run_path(str(evaluator), run_name="__mp_main__")
                finally:
                    sys.argv = argv
                self.assertIn("main", namespace)
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(evaluator), run_name="__main__")

    def test_gate_timeouts_leave_room_for_the_queue_only_with_a_launcher(self) -> None:
        import tomllib
        case_id = "dense-step-qwen3-06b-b1-s128"
        for launcher, extra in (((), 0), (("srun", "--time=00:20:00"), GATE_QUEUE_ALLOWANCE_S)):
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as directory:
                task = make_task(case_id, Path(directory), ROOT, gate_launcher=launcher)
                config = tomllib.loads((task / "vibesys.input.toml").read_text())
                self.assertEqual(config["accuracy"]["timeout_seconds"], 1200 + extra)
                self.assertEqual(config["benchmark"]["timeout_seconds"], 1800 + extra)


if __name__ == "__main__":
    unittest.main()
