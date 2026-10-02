"""Check P0 campaign transitions without paid agents or GPU execution."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from megabench.cases import select_cases
from megabench.harness.runner import _contract_digest

from .p0_campaign import METHOD, run_p0_case


CASE = "dense-step-qwen3-06b-b1-s128"


class P0CampaignTests(unittest.TestCase):
    def test_no_cuda_preflight_keeps_agent_unstarted_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "campaign"
            with patch("reproductions.ForgeMegakernel.p0_campaign.torch.cuda.is_available",
                       return_value=False):
                first = run_p0_case(CASE, destination)
                second = run_p0_case(CASE, destination)
            self.assertEqual(first[0]["status"], "abstain")
            self.assertFalse(first[0]["agent_started"])
            self.assertEqual(second, first)
            self.assertFalse((destination / "ledger.jsonl").exists())
            (destination / "workspace/submission.py").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "accepted workspace differs"):
                run_p0_case(CASE, destination)

    def test_gate_and_review_pass_commits_only_passing_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "campaign"

            def fake_run(argv: list[str], cwd: Path, stdin: str, env: dict[str, str],
                         timeout: int) -> subprocess.CompletedProcess[str]:
                if any("codex_adapter" in arg for arg in argv) and "edit" in argv:
                    path = Path(env["FORGE_CANDIDATE_DIR"]) / "submission.py"
                    path.write_text("def build(case):\n    return lambda inputs: inputs\n")
                    Path(env["FORGE_AGENT_REPORT"]).write_text(
                        json.dumps({"hypothesis": "fuse the decode step"}))
                    return subprocess.CompletedProcess(argv, 0, "", "")
                if any("codex_adapter" in arg for arg in argv) and "review" in argv:
                    request = json.loads(stdin)
                    return subprocess.CompletedProcess(argv, 0, json.dumps(
                        {"run_id": request["run_id"], "status": "pass",
                         "summary": "source reviewed"}), "")
                if "megabench" in argv:
                    submission = Path(argv[argv.index("--submission") + 1])
                    session_id = argv[argv.index("--session-id") + 1]
                    import hashlib
                    Path(argv[argv.index("--output") + 1]).write_text(json.dumps({
                        "case": select_cases("p0", [CASE])[0].to_dict(),
                        "contract_sha256": _contract_digest(),
                        "session": {"id": session_id, "method": METHOD, "case_id": CASE},
                        "status": "ok_provisional",
                        "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
                        "candidate_timing": {"cuda_event_p50_ms": 1.0},
                        "speedup_vs_best_baseline_cuda_event": 1.1,
                    }) + "\n")
                    return subprocess.CompletedProcess(argv, 0, "", "")
                raise AssertionError(argv)

            with patch("reproductions.ForgeMegakernel.p0_campaign.torch.cuda.is_available",
                       return_value=True), patch(
                           "reproductions.ForgeMegakernel.p0_campaign._run",
                           side_effect=fake_run):
                record = run_p0_case(CASE, destination)[0]
            self.assertEqual(record["decision"], "keep")
            self.assertEqual(record["review_status"], "pass")
            self.assertEqual(record["hypothesis"], "fuse the decode step")
            state = json.loads((destination / "state.json").read_text())
            self.assertEqual(state["next_round"], 2)
            self.assertTrue((destination / "rounds/0001/diff.patch").is_file())
            prompt = json.loads((destination / "rounds/0001/prompt.json").read_text())
            self.assertIn("one persistent CUDA launch", prompt["current_guidance"])
            self.assertIn("A correct one-launch kernel can still be much too slow",
                          prompt["adaptation_guidance"])
            self.assertEqual(len((destination / "ledger.jsonl").read_text().splitlines()), 1)

    def test_timed_out_gate_is_recorded_and_reverted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "campaign"

            def fake_run(argv: list[str], cwd: Path, stdin: str, env: dict[str, str],
                         timeout: int) -> subprocess.CompletedProcess[str]:
                if any("codex_adapter" in arg for arg in argv):
                    self.assertIn("edit", argv)
                    (Path(env["FORGE_CANDIDATE_DIR"]) / "submission.py").write_text(
                        "def build(case):\n    pass\n")
                    Path(env["FORGE_AGENT_REPORT"]).write_text(
                        json.dumps({"hypothesis": "try full decode"}))
                    return subprocess.CompletedProcess(argv, 0, "", "")
                self.assertEqual(argv[argv.index("--timeout") + 1], "600")
                Path(argv[argv.index("--output") + 1]).write_text(json.dumps({
                    "case": select_cases("p0", [CASE])[0].to_dict(),
                    "contract_sha256": _contract_digest(),
                    "session": {"id": argv[argv.index("--session-id") + 1],
                                "method": METHOD, "case_id": CASE},
                    "status": "timeout", "reason": "worker exceeded 600s",
                }) + "\n")
                return subprocess.CompletedProcess(argv, 1, "", "")

            with patch("reproductions.ForgeMegakernel.p0_campaign.torch.cuda.is_available",
                       return_value=True), patch(
                           "reproductions.ForgeMegakernel.p0_campaign._run",
                           side_effect=fake_run):
                record = run_p0_case(CASE, destination, timeout=600)[0]
            self.assertEqual(record["gate_status"], "timeout")
            self.assertEqual(record["review_status"], "not_run")
            self.assertEqual(record["decision"], "revert")
            self.assertNotIn("error", record)
            self.assertIn("NotImplementedError",
                          (destination / "workspace/submission.py").read_text())


if __name__ == "__main__":
    unittest.main()
