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
if request["env"].get("STARTED"):  # gpu_job.py writes this file when the job starts
    open(sys.argv[3].rsplit(".", 1)[0] + ".gpu", "w").write("0")
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

    def client(self, *command: str, extra_env: dict | None = None,
               **kwargs) -> subprocess.CompletedProcess:
        env = os.environ | {"PYTHONPATH": str(REPO), "ANTHROPIC_API_KEY": "secret",
                            "MARK": "kept"} | (extra_env or {})
        return subprocess.run(
            [sys.executable, "-m", "megabench.integrations.gpu_broker", "run",
             "--socket", str(self.socket), "--", *command],
            cwd=self.work, env=env, capture_output=True, text=True, **kwargs)

    def test_missing_command_fails_before_a_job(self) -> None:
        out = self.client("./no_such_tool", "--x")
        self.assertEqual(out.returncode, 125)
        self.assertIn("no such file", out.stderr)
        self.assertIn("no GPU job", out.stderr)
        (self.work / "plain.txt").write_text("x")
        self.assertIn("not executable", self.client("./plain.txt").stderr)
        self.assertIn("not found", self.client("no-such-command-xyz").stderr)

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

    def wait_s(self) -> float:
        path = self.socket.parent / "queue_wait_s"
        return float(path.read_text()) if path.exists() else 0.0

    def test_queue_wait_counts_only_jobs_that_did_not_start(self) -> None:
        sleep = [sys.executable, "-c", "import time; time.sleep(7)"]
        out = self.client(*sleep, extra_env={"STARTED": "1"}, timeout=60)
        self.assertEqual(out.returncode, 0)
        time.sleep(6)
        self.assertLess(self.wait_s(), 3)
        out = self.client(*sleep, timeout=60)  # never writes .gpu: it waits in the "queue"
        self.assertEqual(out.returncode, 0)
        time.sleep(6)
        self.assertGreaterEqual(self.wait_s(), 5)


FAKE_SALLOC = """#!/bin/sh
echo "salloc $*" >> "$(dirname "$0")/slurm.log"
sleep "${FAKE_SALLOC_SLEEP:-0}"
echo "salloc: Pending job allocation 4242" >&2
echo "salloc: Granted job allocation 4242" >&2
"""
FAKE_SCANCEL = """#!/bin/sh
echo "scancel $*" >> "$(dirname "$0")/slurm.log"
"""
FAKE_SQUEUE = """#!/bin/sh
echo RUNNING
"""


class HeldAllocationTests(unittest.TestCase):
    """--hold-s: one allocation serves consecutive commands and is released when idle."""

    def start(self, hold: str = "2", salloc_sleep: str = "0") -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        self.socket = root / "gpu/broker.sock"
        self.socket.parent.mkdir()
        for name, text in (("srun", FAKE_SRUN), ("salloc", FAKE_SALLOC),
                           ("scancel", FAKE_SCANCEL), ("squeue", FAKE_SQUEUE)):
            (root / name).write_text(text)
            (root / name).chmod(0o755)
        (root / "job.py").write_text(FAKE_JOB)
        self.work = root / "work"
        self.work.mkdir()
        env = os.environ | {"PYTHONPATH": str(REPO), "FAKE_SALLOC_SLEEP": salloc_sleep}
        self.server = subprocess.Popen(
            [sys.executable, "-m", "megabench.integrations.gpu_broker", "serve",
             "--socket", str(self.socket), "--requests", str(root / "requests"),
             "--spec", str(root / "spec.json"), "--job-python", sys.executable,
             "--job-script", str(root / "job.py"), "--partition-policy", "fixed",
             "--hold-s", hold, "--min-left-s", "0",
             "--", str(root / "srun"), "--partition=main", "--gres=gpu:1",
             "--time=00:20:00", "--kill-on-bad-exit=1"],
            env=env, stdout=open(root / "broker.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(100):
            if self.socket.exists():
                break
            time.sleep(0.05)

    def tearDown(self) -> None:
        self.server.kill()
        self.server.wait()
        self.tmp.cleanup()

    def client(self, *command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "megabench.integrations.gpu_broker", "run",
             "--socket", str(self.socket), "--", *command],
            cwd=self.work, env=os.environ | {"PYTHONPATH": str(REPO)},
            capture_output=True, text=True, timeout=60)

    def slurm(self) -> list[str]:
        path = self.root / "slurm.log"
        return path.read_text().splitlines() if path.exists() else []

    def test_consecutive_commands_share_one_allocation(self) -> None:
        self.start(hold="3")
        self.assertEqual(self.client("echo", "a").stdout.strip(), "a")
        self.assertEqual(self.client("echo", "b").stdout.strip(), "b")
        sallocs = [l for l in self.slurm() if l.startswith("salloc")]
        self.assertEqual(len(sallocs), 1)
        self.assertIn("--no-shell", sallocs[0])
        self.assertNotIn("--kill-on-bad-exit=1", sallocs[0])
        self.assertNotIn("scancel 4242", self.slurm())
        time.sleep(6)  # idle longer than --hold-s
        self.assertIn("scancel 4242", self.slurm())
        self.assertEqual(self.client("true").returncode, 0)  # a new allocation after release
        self.assertEqual(len([l for l in self.slurm() if l.startswith("salloc")]), 2)

    def test_status_passes_through(self) -> None:
        self.start()
        self.assertEqual(self.client(sys.executable, "-c", "raise SystemExit(4)").returncode, 4)

    def test_waiting_for_the_allocation_is_queue_wait(self) -> None:
        self.start(salloc_sleep="7")
        self.assertEqual(self.client("true").returncode, 0)
        time.sleep(6)
        self.assertGreaterEqual(float((self.socket.parent / "queue_wait_s").read_text()), 5)

    def test_stopping_the_broker_releases_held_allocations(self) -> None:
        self.start(hold="600")
        self.assertEqual(self.client("true").returncode, 0)
        self.server.terminate()
        self.server.wait(timeout=30)
        self.assertIn("scancel 4242", self.slurm())


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


class PartitionTests(unittest.TestCase):
    WINDOWS = {"now": "2026-11-07T17:00:00", "partitions": [
        {"partition": "main", "groups": [{"gpus_min": 1, "gpus_max": 4, "options": [
            {"max_time": "00:15:00", "start": "2026-11-07T17:00"},
            {"max_time": "02:00:00", "start": "2026-11-07T18:00"}]}]},
        {"partition": "priority", "groups": [{"gpus_min": 1, "gpus_max": 8, "options": [
            {"max_time": "00:30:00", "start": "2026-11-07T17:00"}]}]}]}

    def test_main_when_it_starts_now(self) -> None:
        from megabench.integrations.gpu_broker import choose_partition
        self.assertEqual(choose_partition(self.WINDOWS, 1, 10), "main")

    def test_priority_when_main_waits(self) -> None:
        from megabench.integrations.gpu_broker import choose_partition
        self.assertEqual(choose_partition(self.WINDOWS, 1, 20), "priority")

    def test_none_when_nothing_starts_now(self) -> None:
        from megabench.integrations.gpu_broker import choose_partition
        self.assertIsNone(choose_partition(self.WINDOWS, 1, 45))

    def test_minutes(self) -> None:
        from megabench.integrations.gpu_broker import _minutes
        self.assertEqual(_minutes("00:20:00"), 20)
        self.assertEqual(_minutes("1-00:00:00"), 1440)
        self.assertEqual(_minutes("30"), 30)
