"""Campaign transition and fail-closed gate tests using isolated fake processes."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from .campaign import run_campaign
from .config import ModelConfig, accessed_bytes
from .milestones import Cell, check_milestone


FAKE_PROCESS = '''\
import json
import os
from pathlib import Path
import sys

role, verdict = sys.argv[1:3]
if role == "agent":
    prompt = json.load(sys.stdin)
    path = Path(os.environ["FORGE_CANDIDATE_DIR"]) / "kernel.cu"
    path.write_text(path.read_text() + "// " + os.environ["FORGE_RUN_ID"] + "\\n")
    Path(os.environ["FORGE_AGENT_REPORT"]).write_text(json.dumps({"hypothesis": "add a stage"}))
elif role == "gate":
    request = json.load(sys.stdin)
    milestone = request["milestone"]
    metrics = {"toolchain_ok": True, "operators_ok": True,
               "float64_golden_ok": True, "operator_precision_probes_ok": True}
    print(json.dumps({"status": "pass" if verdict == "missing" else verdict,
                      "suite": milestone, "summary": "measured fake gate",
                      "metrics": {} if verdict == "missing" else metrics,
                      "details": {}, "run_id": request["run_id"],
                      "candidate_sha256": request["candidate_sha256"]}))
else:
    request = json.load(sys.stdin)
    assert request["run_id"] == os.environ["FORGE_RUN_ID"]
    print(json.dumps({"status": verdict, "summary": "diff reviewed",
                      "run_id": request["run_id"]}))
'''


class CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seed = self.root / "seed"
        self.seed.mkdir()
        (self.seed / "kernel.cu").write_text("// initial\n")
        self.process = self.root / "fake.py"
        self.process.write_text(FAKE_PROCESS)
        self.checkpoint = self.root / "model.safetensors"
        self.checkpoint.write_bytes(b"canonical checkpoint fixture")

    def spec(self, reviewer: str = "pass", gate: str = "pass") -> Path:
        path = self.root / f"spec_{reviewer}_{gate}.json"
        spec = {
            "seed": str(self.seed), "model": asdict(ModelConfig()),
            "batch": 1, "context": 4, "peak_gbs": 3350,
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "mbu_floors": {"M6": 0.28, "M7": 0.35, "M8": 0.38},
            "agent_command": [sys.executable, str(self.process), "agent", "pass"],
            "gate_command": [sys.executable, str(self.process), "gate", gate],
            "review_command": [sys.executable, str(self.process), "review", reviewer],
            "protected_paths": [str(self.process)],
            "rules": "Edit only the candidate source and use the gate measurements.",
        }
        path.write_text(json.dumps(spec))
        return path

    def test_fresh_rounds_advance_only_after_gate_and_review(self) -> None:
        spec = self.spec()
        directory = self.root / "campaign"
        first = run_campaign(spec, directory)
        self.assertEqual(first[0]["decision"], "keep")
        self.assertEqual(json.loads((directory / "state.json").read_text())["milestone"], 1)
        second = run_campaign(spec, directory)
        self.assertEqual(second[0]["decision"], "keep")
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["milestone"], 2)
        self.assertEqual((directory / state["accepted"] / "kernel.cu").read_text(),
                         "// initial\n// 0001\n// 0002\n")
        prompt = json.loads((directory / "rounds/0002/prompt.json").read_text())
        self.assertEqual(prompt["last_gate_summary"]["summary"], "measured fake gate")
        self.assertIn("paired exact and narrowed probes", prompt["current_guidance"])
        self.assertNotIn("hypothesis", json.dumps(prompt))
        self.assertEqual(len((directory / "ledger.jsonl").read_text().splitlines()), 2)
        self.assertEqual((self.seed / "kernel.cu").read_text(), "// initial\n")

    def test_review_failure_reverts(self) -> None:
        directory = self.root / "campaign"
        record = run_campaign(self.spec(reviewer="fail"), directory)[0]
        self.assertEqual(record["gate_status"], "pass")
        self.assertEqual(record["review_status"], "fail")
        self.assertEqual(record["decision"], "revert")
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual((state["milestone"], state["accepted"]), (0, "seed"))

    def test_gate_abstention_reverts_without_review(self) -> None:
        record = run_campaign(self.spec(gate="abstain"), self.root / "campaign")[0]
        self.assertEqual(record["decision"], "revert")
        self.assertEqual(record["review_status"], "not_run")

    def test_self_reported_gate_pass_cannot_override_missing_evidence(self) -> None:
        record = run_campaign(self.spec(gate="missing"), self.root / "campaign")[0]
        self.assertEqual(record["gate_status"], "pass")
        self.assertEqual(record["decision"], "revert")
        self.assertIn("toolchain_ok must be true", record["gate_issues"])

    def test_mutated_accepted_candidate_blocks_resume(self) -> None:
        spec = self.spec()
        directory = self.root / "campaign"
        run_campaign(spec, directory)
        state = json.loads((directory / "state.json").read_text())
        (directory / state["accepted"] / "kernel.cu").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "accepted candidate changed"):
            run_campaign(spec, directory)

    def test_mutated_checkpoint_blocks_campaign(self) -> None:
        spec = self.spec()
        self.checkpoint.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint file is missing or differs"):
            run_campaign(spec, self.root / "campaign")

    def test_full_gate_requires_real_evidence_and_host_derived_mbu(self) -> None:
        cfg = ModelConfig()
        cell = Cell(cfg, 1, 4, 3350, "a" * 64,
                    {"M6": 0.28, "M7": 0.35, "M8": 0.38})
        self.assertTrue(check_milestone(5, {"gpu_executed": True}, cell)[0])
        nominal = accessed_bytes(cfg, 1, 4)
        step_us = nominal / (0.4 * cell.peak_gbs * 1000)
        evidence = {
            "checkpoint_sha256": cell.checkpoint_sha256,
            "gpu_executed": True, "float64_golden_ok": True,
            "finite_logits": True, "poison_bit_identical": True,
            "lane_positions_ok": True, "precision_contract_ok": True,
            "kv_b": 0.01, "kv_c": 0.01, "logprob_error": 0.01,
            "logprob_error_max": 0.02, "hf_bf16_error": 0.01,
            "sglang_error": 0.02, "sglang_error_max": 0.02,
            "timing_scope": "end-to-end", "timing_replicates_us": [[10, 10 + 64 * step_us]],
            "ncu_read_bytes": nominal, "launches_per_step": 1,
            "library_calls": 0, "persistent_launch": True,
            "lm_head_inside_launch": True,
        }
        issues, derived = check_milestone(5, evidence, cell)
        self.assertEqual(issues, [])
        self.assertAlmostEqual(derived["mbu"], 0.4)
        self.assertTrue(check_milestone(6, evidence, cell)[0])
        evidence.update(instruction_types=6, min_instructions_per_layer=8, queue_imbalance=1.1)
        self.assertEqual(check_milestone(6, evidence, cell)[0], [])
        evidence["timing_scope"] = "kernel-only"
        self.assertIn("timing_scope differs from pinned campaign scope",
                      check_milestone(6, evidence, cell)[0])
        evidence["timing_scope"] = "end-to-end"
        evidence["ncu_read_bytes"] = nominal * 4
        self.assertIn("MBU exceeds measured-traffic roofline", check_milestone(6, evidence, cell)[0])


if __name__ == "__main__":
    unittest.main()
