import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from megabench.integrations import scorecard


def commit(ws: Path, msg: str, when: str) -> None:
    env = {"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when, "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, env=env)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", msg], cwd=ws, check=True, env=env)


def check(ws: Path, name: str, sub: Path, ms: float, when: str, ok: bool = True) -> None:
    (ws / "reports").mkdir(exist_ok=True)
    sha = hashlib.sha256((ws / "submission.py").read_bytes()).hexdigest()
    row = {"submission": str(sub), "submission_sha256": sha, "timestamp_utc": when,
           "correctness": {"status": "pass" if ok else "fail"},
           "launch_audit": {"status": "within_budget"},
           "candidate_timing": {"cuda_event_p50_ms": ms}}
    (ws / "reports" / name).write_text(json.dumps(row) + "\n")


class ScorecardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        run = self.run_dir = Path(self.tmp.name) / "run-a"
        ws = run / "case-x"
        ws.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        (ws / "TASK.md").write_text("task")
        (ws / "submission.py").write_text("v0")
        (ws / "mk").mkdir()
        (ws / "mk" / "k.cu").write_text("k0")
        commit(ws, "Prepare workspace", "2026-01-01T00:00:00+00:00")
        # Round 1: the kernel source changes (submission.py does not); 10 ms kept, a variant at 8 ms.
        (ws / "mk" / "k.cu").write_text("k1")
        check(ws, "check-1.jsonl", ws / "submission.py", 10.0, "2026-01-01T00:10:00+00:00")
        check(ws, "v-check-1.jsonl", ws / "variants/r1/a/submission.py", 8.0,
              "2026-01-01T00:20:00+00:00")
        commit(ws, "multi: close round 1", "2026-01-01T00:30:00+00:00")
        # Round 2: no change to the candidate, progress notes only.
        (ws / "progress").mkdir()
        (ws / "progress" / "n.md").write_text("n")
        commit(ws, "multi: close round 2", "2026-01-01T01:00:00+00:00")
        # Round 3: attempt 1 (9 ms) is not kept; attempt 2, slower, is kept. A failed check
        # does not count.
        (ws / "mk" / "k.cu").write_text("k3a")
        check(ws, "check-3a.jsonl", ws / "submission.py", 9.0, "2026-01-01T01:05:00+00:00")
        commit(ws, "implementer-session-turn-1", "2026-01-01T01:07:00+00:00")
        (ws / "mk" / "k.cu").write_text("k3")
        check(ws, "check-3.jsonl", ws / "submission.py", 11.0, "2026-01-01T01:10:00+00:00")
        check(ws, "check-3b.jsonl", ws / "submission.py", 1.0, "2026-01-01T01:15:00+00:00", ok=False)
        commit(ws, "multi: close round 3", "2026-01-01T01:30:00+00:00")
        logs = run / "state/projects/p/runs/r/logs"
        logs.mkdir(parents=True)
        ev = [("2026-01-01T00:00:00Z", "run_started", {}, None),
              ("2026-01-01T00:01:00Z", "phase_started", {"phase": "orchestrator"}, "a"),
              ("2026-01-01T00:03:00Z", "phase_finished", {"phase": "orchestrator"}, "a"),
              ("2026-01-01T00:03:00Z", "phase_started", {"phase": "implementer"}, "b"),
              ("2026-01-01T00:13:00Z", "phase_finished", {"phase": "implementer"}, "b"),
              ("2026-01-01T00:13:00Z", "phase_started", {"phase": "orchestrator"}, "c"),
              ("2026-01-01T00:14:00Z", "phase_finished", {"phase": "orchestrator"}, "c"),
              ("2026-01-01T00:14:00Z", "phase_started", {"phase": "implementer"}, "d"),
              ("2026-01-01T00:29:00Z", "phase_finished", {"phase": "implementer"}, "d"),
              ("2026-01-01T00:30:00Z", "round_finished", {"judge_verdict": "pass", "attempts": 2}, None),
              ("2026-01-01T01:00:00Z", "round_finished", {"judge_verdict": "skipped", "attempts": 1}, None),
              ("2026-01-01T01:30:00Z", "round_finished", {"judge_verdict": "pass", "attempts": 1}, None)]
        (logs / "core-events.jsonl").write_text("".join(
            json.dumps({"timestamp": t, "type": k, "data": d, "execution_id": e}) + "\n"
            for t, k, d, e in ev))
        (logs / "usage.jsonl").write_text(json.dumps({"total_cost_usd": 1.5, "output_tokens": 10}) + "\n")
        (run / "gpu-broker.log").write_text(
            "2026-01-01T00:05:00 vs-gpu-x-1 start [\"check\"]\n"
            "2026-01-01T00:06:00 vs-gpu-x-1 exit 0 after 60s\n"
            "2026-01-01T00:07:00 vs-gpu-x-2 start [\"check\", \"--rerun-of\", \"r.json\"]\n"
            "2026-01-01T00:07:30 vs-gpu-x-2 exit 1 after 30s\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_row(self):
        row = scorecard.score_run(self.run_dir, [10.5], 0.01)
        self.assertEqual(row["case"], "case-x")
        self.assertEqual(row["verdicts"], "PSP")
        self.assertEqual(row["retries"], 1)
        self.assertEqual(row["wall_min"], 90.0)
        self.assertEqual(row["kept_trail_ms"], "10.000 10.000 11.000")
        self.assertEqual(row["best_kept_ms"], 10.0)
        self.assertEqual(row["best_check_ms"], 9.0)
        self.assertEqual(row["best_variant_ms"], 8.0)
        self.assertEqual(row["kept_slower"], ["r3 10.000->11.000"])
        self.assertEqual(row["rejected_faster"], ["r1 variants/r1/a 8.000<10.000", "r3 . 9.000<11.000"])
        self.assertEqual(row["kept_unmeasured"], 0)
        self.assertEqual(row["checks_failed"], 1)
        self.assertEqual(row["min_microbench"], 10.0)
        self.assertEqual(row["min_implementer"], 15.0)
        self.assertEqual(row["min_orchestrator"], 3.0)
        self.assertEqual(row["min_to_10.5"], 10.0)
        self.assertEqual(row["gpu_jobs"], 2)
        self.assertEqual(row["gpu_failed_min"], 0.5)
        self.assertEqual(row["gpu_rerun_jobs"], 1)
        self.assertEqual(row["cost_usd"], 1.5)

    def test_missing_inputs(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        row = scorecard.score_run(empty, [], 0.01)
        self.assertIsNone(row["case"])
        self.assertIsNone(row["best_kept_ms"])


if __name__ == "__main__":
    unittest.main()
