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
        args = Namespace(case="c", venv=Path("/v"), cuda=Path("/cuda"),
                         api_key_file=Path("/keys/k"), oauth_token_file=None)
        env = clean.agent_env(args, Path("/run"), Path("/bin/dir/claude"), key="k")
        self.assertEqual(env["HOME"], "/run/home")
        self.assertFalse([name for name in env if name.startswith("CLAUDE")])
        self.assertNotIn("SLURM_SUBMIT_DIR", env)
        self.assertEqual(env["VIBESYS_AGENT_SANDBOX_ALLOW"].split(":")[0], "/run/trusted")
        self.assertIn("/run/gpu", env["VIBESYS_AGENT_SANDBOX_ALLOW"].split(":"))

    def test_launch_runs_vibesys_without_gpus_and_the_broker_without_the_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            args = Namespace(case="c", venv=Path("/v"), cuda=Path("/cuda"), rounds=10,
                             skills=["s"], vibesys=Path("/vs/bin/vibesys"),
                             vibesys_skills=Path("/skills"), api_key_file=Path("/keys/k"),
                             oauth_token_file=None, timeout_seconds=None, max_gpu_jobs=1, gpu_partition="main",
                             gpu_cpus=8, gpu_mem="64G", gpu_time="00:30:00")
            env = clean.agent_env(args, run, Path("/bin/dir/claude"), key="")
            srun = clean.srun_command(args, 1)
            self.assertIn("--gres=gpu:1", srun)
            clean.write_launch(args, run, env, srun)
            lines = (run / "launch.sh").read_text().splitlines()
            broker = next(line for line in lines if "gpu_broker serve" in line)
            self.assertNotIn("ANTHROPIC_API_KEY", broker)
            self.assertIn("-- srun --partition=main --gres=gpu:1", broker)
            self.assertIn(" ".join(clean.NO_GPU) + " timeout", "\n".join(lines))
            self.assertIn("36000s", "\n".join(lines))
            self.assertFalse((run / "job.sbatch").exists())

    def test_subscription_token_replaces_the_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            args = Namespace(case="c", venv=Path("/v"), cuda=Path("/cuda"), rounds=1,
                             skills=[], vibesys=Path("/vs/bin/vibesys"),
                             vibesys_skills=Path("/skills"), api_key_file=None,
                             oauth_token_file=Path("/keys/t"), timeout_seconds=None,
                             max_gpu_jobs=1, gpu_partition="main", gpu_cpus=8, gpu_mem="64G",
                             gpu_time="00:30:00")
            env = clean.agent_env(args, run, Path("/bin/dir/claude"), key="")
            self.assertIn("CLAUDE_CODE_OAUTH_TOKEN", env)
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            clean.write_launch(args, run, env, clean.srun_command(args, 1))
            text = (run / "launch.sh").read_text()
            self.assertIn("key=$(</keys/t)", text)
            self.assertIn('CLAUDE_CODE_OAUTH_TOKEN="$key"', text)
            broker = next(line for line in text.splitlines() if "gpu_broker serve" in line)
            self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", broker)


if __name__ == "__main__":
    unittest.main()
