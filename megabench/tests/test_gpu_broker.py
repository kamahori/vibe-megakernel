"""GPU broker: requests, output, exit status, secrets and cancellation, with a fake srun."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Drops the srun options and runs the job command.
FAKE_SRUN = """#!/bin/sh
while [ "${1#--}" != "$1" ]; do shift; done
exec "$@"
"""
# Stands in for gpu_job.py: runs the request without a sandbox.
FAKE_JOB = """
import json, os, sys
request = json.load(open(sys.argv[3]))
os.chdir(request["cwd"])
os.execvpe(request["argv"][0], request["argv"], request["env"])
"""


class BrokerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.socket = root / "gpu/broker.sock"
        self.socket.parent.mkdir()
        srun = root / "srun"
        srun.write_text(FAKE_SRUN)
        srun.chmod(0o755)
        job = root / "job.py"
        job.write_text(FAKE_JOB)
        self.work = root / "work"
        self.work.mkdir()
        env = os.environ | {"PYTHONPATH": str(REPO)}
        self.server = subprocess.Popen(
            [sys.executable, "-m", "megabench.integrations.gpu_broker", "serve",
             "--socket", str(self.socket), "--requests", str(root / "requests"),
             "--spec", str(root / "spec.json"), "--job-python", sys.executable,
             "--job-script", str(job), "--", str(srun), "--gres=gpu:1"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if self.socket.exists():
                break
            time.sleep(0.05)

    def tearDown(self) -> None:
        self.server.kill()
        self.server.wait()
        self.tmp.cleanup()

    def client(self, *command: str, **kwargs) -> subprocess.CompletedProcess:
        env = os.environ | {"PYTHONPATH": str(REPO), "ANTHROPIC_API_KEY": "secret",
                            "MARK": "kept"}
        return subprocess.run(
            [sys.executable, "-m", "megabench.integrations.gpu_broker", "run",
             "--socket", str(self.socket), "--", *command],
            cwd=self.work, env=env, capture_output=True, text=True, **kwargs)

    def test_output_status_cwd_and_environment(self) -> None:
        script = ("import json, os, sys; print(json.dumps([os.getcwd(), "
                  "os.environ.get('MARK'), os.environ.get('ANTHROPIC_API_KEY')])); "
                  "print('err', file=sys.stderr); sys.exit(3)")
        out = self.client(sys.executable, "-c", script, timeout=60)
        self.assertEqual(out.returncode, 3)
        self.assertEqual(json.loads(out.stdout), [str(self.work), "kept", None])
        self.assertIn("err", out.stderr)

    def test_client_exit_cancels_the_job(self) -> None:
        marker = self.work / "finished"
        command = [sys.executable, "-c", f"import time, pathlib; time.sleep(5); "
                                          f"pathlib.Path({str(marker)!r}).touch()"]
        with self.assertRaises(subprocess.TimeoutExpired):
            self.client(*command, timeout=1)
        time.sleep(7)
        self.assertFalse(marker.exists())
        self.assertEqual(self.client("true", timeout=60).returncode, 0)


class WorkspaceGpuTests(unittest.TestCase):
    def test_gpu_commands_use_gpu_run_and_gates_use_the_launcher(self) -> None:
        from ..integrations.make_agent_workspace import make_workspace
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "gpu/broker.sock"
            project = make_workspace("dense-step-qwen3-06b-b1-s128", Path(directory) / "w",
                                     gpu_socket=socket_path, gate_launcher=("srun", "--gres=gpu:1"))
            self.assertIn(str(socket_path), (project / "gpu_run").read_text())
            for tool in ("check_candidate", "profile_candidate", "timeline_candidate"):
                self.assertIn(str(project / "gpu_run"), (project / tool).read_text(), tool)
            self.assertNotIn("gpu_run", (project / "sol_info").read_text())
            self.assertIn("./gpu_run", (project / "TASK.md").read_text())
            gate = project / ".vibesys/tasks/dense-step-qwen3-06b-b1-s128/evaluator.sh"
            self.assertIn("exec srun --gres=gpu:1 ", gate.read_text())

    def test_without_a_broker_the_workspace_does_not_change(self) -> None:
        from ..integrations.make_agent_workspace import make_workspace
        with tempfile.TemporaryDirectory() as directory:
            project = make_workspace("dense-step-qwen3-06b-b1-s128", Path(directory) / "w")
            self.assertFalse((project / "gpu_run").exists())
            self.assertNotIn("gpu_run", (project / "check_candidate").read_text())
            self.assertNotIn("gpu_run", (project / "TASK.md").read_text())


if __name__ == "__main__":
    unittest.main()
