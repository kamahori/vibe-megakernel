"""Isolated VibeSys run preparation: the trusted snapshot and the agent environment."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from ..integrations import clean_vibesys_run as clean


class TrustedSnapshotTests(unittest.TestCase):
    def test_snapshot_drops_past_run_material(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trusted = Path(directory) / "trusted"
            clean.build_trusted("HEAD", trusted, Path(directory) / "venv")
            root = trusted / "megabench"
            self.assertTrue((root / "cases.py").is_file())
            self.assertTrue((root / "skills").is_dir())
            for name in clean.DROP_DIRS:
                self.assertFalse((root / name).exists(), name)
            self.assertLessEqual({p.name for p in (root / "docs").iterdir()}, set(clean.KEEP_DOCS))
            self.assertFalse((trusted / "reproductions").exists())
            self.assertTrue((trusted / ".venv").is_symlink())


class AgentEnvironmentTests(unittest.TestCase):
    def test_environment_is_closed_and_home_is_the_run(self) -> None:
        args = Namespace(case="c", venv=Path("/v"), cuda=Path("/cuda"))
        env = clean.agent_env(args, Path("/run"), Path("/bin/dir/claude"), key="k")
        self.assertEqual(env["HOME"], "/run/home")
        self.assertFalse([name for name in env if name.startswith("CLAUDE")])
        self.assertNotIn("SLURM_SUBMIT_DIR", env)
        self.assertEqual(env["VIBESYS_AGENT_SANDBOX_ALLOW"].split(":")[0], "/run/trusted")


if __name__ == "__main__":
    unittest.main()
