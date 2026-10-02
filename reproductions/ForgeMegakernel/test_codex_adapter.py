"""Check the Codex subprocess contract without making an API request."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from .codex_adapter import _invoke


FAKE_CODEX = f'''#!{sys.executable}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
prompt = sys.stdin.read()
Path(os.environ["FORGE_ROUND_DIR"], "codex_args.json").write_text(json.dumps(args))
Path(os.environ["FORGE_ROUND_DIR"], "codex_prompt.txt").write_text(prompt)
value = ({{"status": "pass", "summary": "reviewed"}}
         if args[args.index("--sandbox") + 1] == "read-only"
         else {{"hypothesis": "try a different instruction schedule"}})
Path(args[args.index("--output-last-message") + 1]).write_text(json.dumps(value))
'''


class CodexAdapterTests(unittest.TestCase):
    def test_editor_and_reviewer_are_fresh_pinned_model_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            round_dir = root / "round"
            candidate.mkdir()
            round_dir.mkdir()
            fake = root / "codex"
            fake.write_text(FAKE_CODEX)
            fake.chmod(0o755)
            env = {
                "FORGE_CANDIDATE_DIR": str(candidate),
                "FORGE_ROUND_DIR": str(round_dir),
                "FORGE_AGENT_REPORT": str(round_dir / "agent_report.json"),
            }
            with patch.dict(os.environ, env):
                _invoke("edit", {"current_milestone": 6,
                                 "current_guidance": "assign typed instructions per SM",
                                 "last_gate_summary": {"metrics": {"queue_imbalance": 1.5}}},
                        str(fake))
                args = json.loads((round_dir / "codex_args.json").read_text())
                prompt = (round_dir / "codex_prompt.txt").read_text()
                self.assertEqual(args[args.index("--model") + 1], "gpt-6-sol")
                self.assertEqual(args[args.index("--sandbox") + 1], "workspace-write")
                self.assertIn("--ephemeral", args)
                self.assertEqual(args[-1], "-")
                self.assertIn("assign typed instructions per SM", prompt)
                self.assertIn('"queue_imbalance": 1.5', prompt)
                self.assertIn("one falsifiable hypothesis", prompt)
                self.assertTrue(json.loads((round_dir / "agent_report.json").read_text())["hypothesis"])
                review = _invoke("review", {"run_id": "0001", "diff": "change",
                                            "gate": {"status": "pass"}}, str(fake))
                args = json.loads((round_dir / "codex_args.json").read_text())
                self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
                self.assertIn("active preprocessor branches",
                              (round_dir / "codex_prompt.txt").read_text())
                self.assertEqual(review, {"status": "pass", "summary": "reviewed", "run_id": "0001"})


if __name__ == "__main__":
    unittest.main()
