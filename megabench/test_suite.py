"""CPU protocol tests for the whole-model challenge catalog."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from .cases import CASES, Case, select_cases
from .runner import (_compare, _docker_worker_command, _run_one,
                     _stop_owned_container, evaluate_case, main)
from .workloads import make_inputs, reference


REFERENCE_SUBMISSION = Path(__file__).parent / "examples" / "reference_submission.py"
TINY_DENSE = Case(
    "dense-step-dev", "dense_step", "Qwen/Qwen3-0.6B", "decode",
    {"batch": 1, "context": 4, "layers": 2, "hidden": 64,
     "q_heads": 2, "kv_heads": 1, "head_dim": 32,
     "intermediate": 128, "vocab": 256},
    "dense_llm", "p0", ready=True, atol=0.002, rtol=0.002,
)


class CatalogTests(unittest.TestCase):
    def test_whole_model_catalog_and_priorities(self) -> None:
        self.assertEqual(len(CASES), len({case.id for case in CASES}))
        self.assertTrue(all("step" in case.id or "iteration" in case.id
                            for case in CASES))
        self.assertEqual([case.family for case in select_cases("core")],
                         ["dense_step"])
        self.assertEqual(len(select_cases("planned")), len(CASES) - 1)
        self.assertEqual({c.params["bits"] for c in CASES
                          if c.family == "quant_step"}, {4, 8})
        self.assertTrue(any("EAGLE3" in c.model for c in CASES))
        self.assertTrue(any(c.tp > 1 or c.ep > 1 for c in CASES))
        p3_models = " ".join(case.model for case in select_cases("p3"))
        self.assertTrue(all(name in p3_models for name in
                            ("DeepSeek", "GLM", "Kimi")))
        self.assertFalse(any(c.id in {"vl-ingest", "serving-replay"} for c in CASES))

    def test_dense_oracle_is_deterministic_and_input_dependent(self) -> None:
        inputs = make_inputs(TINY_DENSE, 123)
        before = {name: value.clone() for name, value in inputs.items()}
        actual = reference(TINY_DENSE, inputs)
        again = reference(TINY_DENSE, make_inputs(TINY_DENSE, 123))
        other = reference(TINY_DENSE, make_inputs(TINY_DENSE, 987))
        self.assertEqual(set(actual), {"logits", "next_token", "k_write", "v_write"})
        self.assertEqual(actual["logits"].shape, (256,))
        self.assertEqual(actual["k_write"].shape, (2, 1, 32))
        self.assertEqual(actual["v_write"].shape, (2, 1, 32))
        self.assertEqual(actual["next_token"].dtype, torch.int64)
        self.assertEqual(int(actual["next_token"]), int(actual["logits"].argmax()))
        self.assertTrue(all(torch.equal(actual[k], again[k]) for k in actual))
        self.assertTrue(any(not torch.equal(actual[k], other[k]) for k in actual))
        self.assertTrue(all(torch.equal(inputs[k], before[k]) for k in inputs))


class HarnessTests(unittest.TestCase):
    def test_docker_command_uses_only_scoped_mounts_and_gpu(self) -> None:
        case = select_cases("core")[0]
        args = Namespace(submission=str(REFERENCE_SUBMISSION), device="cuda:0",
                         trials=2, warmup=1, reps=3, graph_baseline=False,
                         docker_image="existing-image:tag", docker_gpus="device=6")
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            cmd, name, run_id = _docker_worker_command(
                args, case, Path(directory), result)
        self.assertEqual(cmd[:3], ["docker", "run", "--rm"])
        self.assertIn("--read-only", cmd)
        self.assertEqual(cmd[cmd.index("--tmpfs") + 1],
                         "/tmp:rw,exec,nosuid,size=8g")
        self.assertIn("--network", cmd)
        self.assertIn("none", cmd)
        self.assertIn("--gpus", cmd)
        self.assertEqual(cmd[cmd.index("--gpus") + 1], "device=6")
        self.assertEqual(cmd[cmd.index("--pull") + 1], "never")
        self.assertIn("--no-graph-baseline", cmd)
        self.assertIn("/workspace/megabench/examples/reference_submission.py", cmd)
        self.assertIn(f"org.megabench.run={run_id}", cmd)
        self.assertTrue(name.startswith("megabench-"))
        self.assertFalse(any("docker.sock" in part for part in cmd))

    def test_timeout_cleanup_never_stops_unverified_container(self) -> None:
        with patch("megabench.runner.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "someone-else\n"
            warning = _stop_owned_container("megabench-test", "expected")
            self.assertIn("could not be verified", warning)
            run.assert_called_once()
            run.return_value.stdout = "expected\n"
            self.assertIsNone(_stop_owned_container("megabench-test", "expected"))
            self.assertEqual(run.call_args_list[-1].args[0][:2], ["docker", "stop"])

    def test_reference_submission_passes_tiny_cpu_case(self) -> None:
        result = evaluate_case(TINY_DENSE, REFERENCE_SUBMISSION,
                               device="cpu", trials=2, warmup=0, reps=1)
        self.assertEqual(result["status"], "correctness_only")
        self.assertEqual(len(result["correctness"]["trials"]), 2)

    def test_wrong_output_is_rejected(self) -> None:
        want = reference(TINY_DENSE, make_inputs(TINY_DENSE, 1))
        bad = want | {"logits": torch.zeros_like(want["logits"])}
        with self.assertRaises(AssertionError):
            _compare(want, bad, TINY_DENSE, "cpu")

    def test_planned_case_is_unscoreable(self) -> None:
        case = select_cases("planned")[0]
        with tempfile.TemporaryDirectory() as directory:
            result = _run_one(Namespace(), case, Path(directory))
        self.assertEqual(result["status"], "not_implemented")

    def test_cli_lists_new_cases(self) -> None:
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(["list", "--suite", "core", "--json"]), 0)
        cases = json.loads(output.getvalue())
        self.assertEqual([case["id"] for case in cases],
                         ["dense-step-qwen3-06b-b1-s128"])


if __name__ == "__main__":
    unittest.main()
