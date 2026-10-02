"""Fail-closed behavior of the independent Qwen3 development gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from .qwen3_gate import evaluate


class Qwen3GateTests(unittest.TestCase):
    def test_missing_gpu_abstains_and_cannot_pass_a_paper_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, round_dir = root / "candidate", root / "round"
            candidate.mkdir()
            round_dir.mkdir()
            for name in ("submission.py", "dense_binding.cpp", "dense_launch.cu", "dense_kernel.cu"):
                (candidate / name).write_text("// source\n")
            checkpoint = root / "checkpoint"
            checkpoint.write_bytes(b"pinned")
            request = {
                "milestone": "M0", "run_id": "0001", "candidate_sha256": "b" * 64,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(b"pinned").hexdigest(),
            }
            with patch("torch.cuda.is_available", return_value=False):
                result = evaluate(request, candidate, round_dir)
            self.assertEqual(result["status"], "abstain")
            self.assertEqual(result["metrics"], {})
            self.assertFalse((round_dir / "megabench_dense.jsonl").exists())
            checkpoint.write_bytes(b"changed")
            with patch("torch.cuda.is_available", return_value=False):
                result = evaluate(request, candidate, round_dir)
            self.assertEqual(result["status"], "fail")

    def test_synthetic_megabench_pass_still_abstains_on_m5(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, round_dir = root / "candidate", root / "round"
            candidate.mkdir()
            round_dir.mkdir()
            for name in ("submission.py", "dense_binding.cpp", "dense_launch.cu", "dense_kernel.cu"):
                (candidate / name).write_text("// source\n")
            checkpoint = root / "checkpoint"
            checkpoint.write_bytes(b"pinned")
            request = {
                "milestone": "M5", "run_id": "0005", "candidate_sha256": "b" * 64,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(b"pinned").hexdigest(),
            }

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({
                    "case": {"id": "dense-step-qwen3-06b-b1-s128"},
                    "correctness": {"status": "pass"},
                    "launch_audit": {"status": "within_budget", "gpu_kernel_count": 1},
                    "candidate_timing": {"cuda_event_p50_ms": 1.5},
                    "status": "ok_provisional",
                }) + "\n")
                return subprocess.CompletedProcess(command, 0, "", "")

            with (patch("torch.cuda.is_available", return_value=True),
                  patch("reproductions.ForgeMegakernel.qwen3_gate.subprocess.run",
                        side_effect=fake_run)):
                result = evaluate(request, candidate, round_dir)
            self.assertEqual(result["status"], "abstain")
            self.assertTrue(result["metrics"]["megabench_correct"])
            self.assertEqual(result["metrics"]["launches_per_step"], 1)
            self.assertTrue(result["details"]["synthetic_weights"])


if __name__ == "__main__":
    unittest.main()
