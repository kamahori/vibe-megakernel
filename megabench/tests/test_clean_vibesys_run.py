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
            self.assertNotIn("VIBESYS_MICROBENCH_PHASE", "\n".join(lines))
            args.microbench_phase = True
            clean.write_launch(args, run, env, srun)
            vibesys = [line for line in (run / "launch.sh").read_text().splitlines()
                       if "VIBESYS_MICROBENCH_PHASE=1" in line]
            self.assertEqual(len(vibesys), 1)
            self.assertNotIn("gpu_broker", vibesys[0])
            self.assertNotIn("VIBESYS_MICROBENCH_MINUTES", vibesys[0])
            args.microbench_minutes = 40.0
            clean.write_launch(args, run, env, srun)
            self.assertIn("VIBESYS_MICROBENCH_MINUTES=40 ", (run / "launch.sh").read_text())
            self.assertIn("VIBESYS_MEASUREMENT_DIR=reports ", (run / "launch.sh").read_text())
            self.assertNotIn("VIBESYS_MICROBENCH_FULL_ROUNDS", (run / "launch.sh").read_text())
            args.microbench_full_rounds = 2
            clean.write_launch(args, run, env, srun)
            self.assertIn("VIBESYS_MICROBENCH_FULL_ROUNDS=2 ", (run / "launch.sh").read_text())
            args.microbench_full_rounds = -1
            clean.write_launch(args, run, env, srun)
            self.assertNotIn("VIBESYS_MICROBENCH_FULL_ROUNDS", (run / "launch.sh").read_text())
            self.assertNotIn("VIBESYS_PHASE_AGENTS", (run / "launch.sh").read_text())
            # Phase agents: the flag and the round count sit on the VibeSys line, even without
            # the microbench phase, and never on the broker line.
            args.microbench_phase, args.microbench_full_rounds, args.phase_agents = False, 2, True
            clean.write_launch(args, run, env, srun)
            text = (run / "launch.sh").read_text()
            line = next(l for l in text.splitlines() if "VIBESYS_PHASE_AGENTS=1" in l)
            self.assertIn("VIBESYS_MICROBENCH_FULL_ROUNDS=2 ", line)
            self.assertNotIn("VIBESYS_MICROBENCH_PHASE=1", line)
            self.assertNotIn("gpu_broker", line)
            self.assertEqual(text.count("VIBESYS_PHASE_AGENTS"), 1)

    def test_help_shows_the_phase_agents_flag_and_the_gpu_time_default(self) -> None:
        import contextlib
        import io
        import sys
        from unittest import mock

        text = io.StringIO()
        with mock.patch.object(sys, "argv", ["x", "--help"]), contextlib.redirect_stdout(text):
            with self.assertRaises(SystemExit):
                clean.main()
        self.assertIn("--phase-agents", text.getvalue())
        self.assertIn("default: 00:20:00", " ".join(text.getvalue().split()))

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


class SeedTests(unittest.TestCase):
    def test_seed_copies_the_candidate_and_progress_and_moves_round_files(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            src, ws = Path(directory) / "src", Path(directory) / "ws"
            files = {"submission.py": "seed sub", "k.cu": "kernel", "TASK.md": "old task",
                     "progress/microbench/open.md": "old ledger",
                     "progress/plans/round-0002.json": "{}",
                     "progress/microbench/answers/round-0002.md": "a",
                     "progress/round-0002.md": "notes", ".build/x.so": "bin",
                     "progress/microbench/.build-cache/k/o": "obj",
                     ".agents/skills/s/SKILL.md": "skill", "some-skill/SKILL.md": "skill",
                     "some-skill/notes.md": "n"}
            for name, text in files.items():
                (src / name).parent.mkdir(parents=True, exist_ok=True)
                (src / name).write_text(text)
            git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run([*git, "init", "-q"], check=True)
            subprocess.run([*git, "add", "-A", "-f"], check=True)
            subprocess.run([*git, "commit", "-qm", "s"], check=True)
            (src / "k.cu").write_text("later edit")
            for name, text in {"TASK.md": "new task", "submission.py": "template",
                               "progress/microbench/open.md": "header"}.items():
                (ws / name).parent.mkdir(parents=True, exist_ok=True)
                (ws / name).write_text(text)
            subprocess.run(["git", "-C", str(ws), "init", "-q"], check=True)
            clean.seed_workspace(ws, src, "HEAD")
            self.assertEqual((ws / "submission.py").read_text(), "seed sub")
            self.assertEqual((ws / "k.cu").read_text(), "kernel")
            self.assertEqual((ws / "TASK.md").read_text(), "new task")
            self.assertEqual((ws / "progress/microbench/open.md").read_text(), "old ledger")
            self.assertFalse((ws / "progress/plans/round-0002.json").exists())
            self.assertTrue((ws / "progress/prior/plans/round-0002.json").is_file())
            self.assertTrue((ws / "progress/prior/microbench/answers/round-0002.md").is_file())
            self.assertTrue((ws / "progress/prior/round-0002.md").is_file())
            self.assertFalse((ws / ".build").exists())
            self.assertFalse((ws / "progress/microbench/.build-cache").exists())
            self.assertFalse((ws / ".agents").exists())
            self.assertFalse((ws / "some-skill").exists())
            status = subprocess.run(["git", "-C", str(ws), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout
            self.assertEqual(status, "", "the seeded workspace is committed")
