"""CPU protocol tests for the whole-model challenge catalog."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from ..cases import CASES, Case, select_cases
from ..harness.correctness import _compare
from ..harness.runner import (_contract_digest, _docker_worker_command,
                              _output_path, _run_one, _stop_owned_container,
                              aggregate_sessions, evaluate_case, main)
from ..workloads import make_inputs, reference


REFERENCE_SUBMISSION = Path(__file__).resolve().parents[1] / "examples" / "reference_submission.py"
TINY_DENSE = Case(
    "dense-step-dev", "dense_step", "Qwen/Qwen3-0.6B", "decode",
    {"batch": 1, "context": 4, "layers": 2, "hidden": 64,
     "q_heads": 2, "kv_heads": 1, "head_dim": 32,
     "intermediate": 128, "vocab": 256},
    "dense_llm", "p0", ready=True, atol=0.002, rtol=0.002,
)
TINY_MOE = Case(
    "moe-step-dev", "moe_step", "Qwen/Qwen3-30B-A3B", "decode",
    {"batch": 1, "context": 4, "layers": 2, "hidden": 64,
     "q_heads": 2, "kv_heads": 1, "head_dim": 32, "experts": 4,
     "topk": 2, "intermediate": 32, "vocab": 256},
    "routed_moe", "p0", ready=True, atol=0.003, rtol=0.003,
)
GEMMA_PARAMS = {"batch": 1, "context": 4, "layers": 6, "hidden": 64,
                "q_heads": 2, "kv_heads": 1, "head_dim": 32,
                "intermediate": 128, "vocab": 256, "local_window": 3}
TINY_W8 = Case("quant-step-w8-dev", "quant_step", "google/gemma-3-4b-it",
               "decode", GEMMA_PARAMS | {"bits": 8}, "quantized_llm", "p0",
               ready=True, atol=0.003, rtol=0.003)
TINY_W4 = Case("quant-step-w4-dev", "quant_step", "google/gemma-3-4b-it",
               "decode", GEMMA_PARAMS | {"bits": 4}, "quantized_llm", "p0",
               ready=True, atol=0.003, rtol=0.003)
TINY_SPEC = Case(
    "spec-target-step-dev", "spec_target_step",
    "meta-llama/Llama-3.1-8B-Instruct + EAGLE3", "verify",
    {"batch": 1, "context": 4, "layers": 2, "hidden": 64,
     "q_heads": 2, "kv_heads": 1, "head_dim": 32,
     "intermediate": 128, "vocab": 256, "draft_vocab": 64,
     "draft_depth": 3},
    "speculative_decoding", "p0", ready=True, atol=0.003, rtol=0.003,
)
TINY_P0 = (TINY_DENSE, TINY_MOE, TINY_W8, TINY_W4, TINY_SPEC)


class CatalogTests(unittest.TestCase):
    def test_whole_model_catalog_and_priorities(self) -> None:
        self.assertEqual(len(CASES), len({case.id for case in CASES}))
        self.assertTrue(all("step" in case.id or "iteration" in case.id
                            for case in CASES))
        self.assertEqual({case.id for case in select_cases("core")},
                         {case.id for case in select_cases("p0")})
        self.assertEqual(len(select_cases("core")), 5)
        self.assertEqual(len(select_cases("planned")), len(CASES) - 5)
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

    def test_each_p0_oracle_is_deterministic_and_input_dependent(self) -> None:
        for case in TINY_P0:
            with self.subTest(case=case.id), torch.inference_mode():
                inputs = make_inputs(case, 17)
                before = {name: value.clone() for name, value in inputs.items()}
                actual = reference(case, inputs)
                again = reference(case, make_inputs(case, 17))
                other = reference(case, make_inputs(case, 19))
                self.assertEqual(set(actual), set(again))
                self.assertTrue(all(torch.equal(actual[k], again[k]) for k in actual))
                self.assertTrue(any(not torch.equal(actual[k], other[k]) for k in actual))
                self.assertTrue(all(torch.equal(inputs[k], before[k]) for k in inputs))

    def test_quantized_weight_layout_and_shared_base_stream(self) -> None:
        from ..tasks.gemma import dequant

        w8 = make_inputs(TINY_W8, 4)
        w4 = make_inputs(TINY_W4, 4)
        self.assertEqual(w8["wg"].dtype, torch.int8)
        self.assertEqual(w4["wg"].dtype, torch.uint8)
        self.assertEqual(w4["wg"].shape[-1] * 2, w8["wg"].shape[-1])
        a = dequant(w8["wg"][0], w8["wg_scale"][0], 8)
        b = dequant(w4["wg"][0], w4["wg_scale"][0], 4)
        self.assertLess(float((a - b).abs().mean()), 0.03)

    def test_moe_uses_independent_lm_head(self) -> None:
        values = make_inputs(TINY_MOE, 7)
        self.assertFalse(torch.equal(values["embed"], values["lm_head"]))
        with torch.inference_mode():
            normal = reference(TINY_MOE, values)["logits"]
            values["lm_head"].zero_()
            zero_head = reference(TINY_MOE, values)["logits"]
        self.assertTrue(bool((zero_head == 0).all()))
        self.assertFalse(torch.equal(normal, zero_head))

    def test_speculative_acceptance_and_rollback(self) -> None:
        from ..tasks.eagle3 import set_acceptance_scenario

        values = make_inputs(TINY_SPEC, 17)
        with torch.inference_mode():
            set_acceptance_scenario(TINY_SPEC, values, 3)
            accepted = reference(TINY_SPEC, values)
            self.assertEqual(int(accepted["accepted_count"]), 3)
            self.assertEqual(int(accepted["committed_count"]), 4)
            self.assertEqual(int(accepted["cache_length"]), 8)
            values["draft_tokens"][0] = (values["draft_tokens"][0] + 1) % 256
            rejected = reference(TINY_SPEC, values)
            self.assertEqual(int(rejected["accepted_count"]), 0)
            self.assertEqual(int(rejected["committed_count"]), 1)
            self.assertEqual(int(rejected["cache_length"]), 5)
            self.assertTrue(bool((rejected["committed_tokens"][1:] == -1).all()))
            self.assertTrue(bool((rejected["k_write"][:, 1:] == 0).all()))
            set_acceptance_scenario(TINY_SPEC, values, 1)
            middle = reference(TINY_SPEC, values)
            self.assertEqual(int(middle["accepted_count"]), 1)
            self.assertEqual(int(middle["committed_count"]), 2)


class HarnessTests(unittest.TestCase):
    def test_default_results_stay_under_megabench(self) -> None:
        self.assertEqual(_output_path(None).parent,
                         Path(__file__).resolve().parents[1] / "runs")

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
        with patch("megabench.harness.runner.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "someone-else\n"
            warning = _stop_owned_container("megabench-test", "expected")
            self.assertIn("could not be verified", warning)
            run.assert_called_once()
            run.return_value.stdout = "expected\n"
            self.assertIsNone(_stop_owned_container("megabench-test", "expected"))
            self.assertEqual(run.call_args_list[-1].args[0][:2], ["docker", "stop"])

    def test_reference_submission_passes_tiny_cpu_case(self) -> None:
        for case in TINY_P0:
            with self.subTest(case=case.id):
                result = evaluate_case(case, REFERENCE_SUBMISSION,
                                       device="cpu", trials=2, warmup=0, reps=1)
                self.assertEqual(result["status"], "correctness_only")
                self.assertEqual(len(result["correctness"]["trials"]), 2)
                if case.family == "spec_target_step":
                    self.assertEqual(result["spec_accounting"]["proposed_draft_tokens"], 3)

    def test_wrong_output_is_rejected(self) -> None:
        want = reference(TINY_DENSE, make_inputs(TINY_DENSE, 1))
        bad = want | {"logits": torch.zeros_like(want["logits"])}
        with self.assertRaises(AssertionError):
            _compare(want, bad, TINY_DENSE, "cpu")

    def test_bf16_kv_accepts_one_rounding_step_only(self) -> None:
        want = {"k_write": torch.tensor([1.0], dtype=torch.bfloat16)}
        one_step = {"k_write": torch.tensor([1.0078125], dtype=torch.bfloat16)}
        _compare(want, one_step, TINY_DENSE, "cpu")
        too_far = {"k_write": torch.tensor([1.03125], dtype=torch.bfloat16)}
        with self.assertRaises(AssertionError):
            _compare(want, too_far, TINY_DENSE, "cpu")

    def test_planned_case_is_unscoreable(self) -> None:
        case = select_cases("planned")[0]
        with tempfile.TemporaryDirectory() as directory:
            result = _run_one(Namespace(), case, Path(directory))
        self.assertEqual(result["status"], "not_implemented")

    def test_timeout_records_submission_for_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "candidate" / "submission.py"
            args = Namespace(submission=str(submission), docker_image=None,
                             device="cpu", trials=1, warmup=0, reps=1,
                             graph_baseline=False, timeout=1)
            with patch("megabench.harness.runner.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("worker", 1)):
                result = _run_one(args, TINY_DENSE, root)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["submission"], str(submission))

    def test_cli_lists_new_cases(self) -> None:
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(["list", "--suite", "core", "--json"]), 0)
        cases = json.loads(output.getvalue())
        self.assertEqual(len(cases), 5)
        self.assertEqual({case["id"] for case in cases},
                         {case.id for case in select_cases("p0")})

    def test_aggregate_requires_independent_case_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, case in enumerate(select_cases("p0")):
                path = root / f"{index}.jsonl"
                row = {"case": case.to_dict(), "status": "ok_provisional",
                       "contract_sha256": _contract_digest(),
                       "speedup_vs_best_baseline_cuda_event": 2.0,
                       "submission": str(root / str(index) / "submission.py"),
                       "session": {"id": f"trial-{index}", "method": "codex",
                                   "case_id": case.id}}
                path.write_text(json.dumps(row) + "\n")
                paths.append(path)
            summary = aggregate_sessions(paths, "p0", "codex")
            self.assertEqual(summary["passed"], 5)
            self.assertAlmostEqual(summary["provisional_geomean_speedup"], 2.0)
            self.assertIsNone(aggregate_sessions(paths[:4], "p0", "codex")
                              ["provisional_geomean_speedup"])
            timed_out = json.loads(paths[-1].read_text())
            timed_out["status"] = "timeout"
            timed_out.pop("speedup_vs_best_baseline_cuda_event")
            paths[-1].write_text(json.dumps(timed_out) + "\n")
            failed_summary = aggregate_sessions(paths, "p0", "codex")
            self.assertEqual(failed_summary["passed"], 4)
            self.assertIsNone(failed_summary["provisional_geomean_speedup"])
            duplicate = json.loads(paths[-1].read_text())
            duplicate["session"]["id"] = "trial-0"
            paths[-1].write_text(json.dumps(duplicate) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate session ID"):
                aggregate_sessions(paths, "p0", "codex")
            duplicate["session"]["id"] = "trial-4"
            duplicate["submission"] = str(root / "0" / "submission.py")
            paths[-1].write_text(json.dumps(duplicate) + "\n")
            with self.assertRaisesRegex(ValueError, "shared candidate directory"):
                aggregate_sessions(paths, "p0", "codex")


if __name__ == "__main__":
    unittest.main()
