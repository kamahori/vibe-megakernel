"""CPU protocol tests; GPU examples are exercised separately."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch

from .cases import CASES, select_cases
from .runner import _compare, evaluate_case, main
from .workloads import make_inputs, reference


REFERENCE_SUBMISSION = Path(__file__).parent / "examples" / "reference_submission.py"


class CatalogTests(unittest.TestCase):
    def test_unique_cases_and_requested_families(self) -> None:
        self.assertEqual(len(CASES), len({case.id for case in CASES}))
        families = {case.family for case in select_cases("core")}
        self.assertEqual(families, {"stencil", "decoder", "moe",
                                    "quant_mlp", "spec_verify"})
        self.assertTrue(any(c.params.get("bits") == 4 for c in CASES))
        self.assertTrue(any(c.params.get("bits") == 8 for c in CASES))
        self.assertTrue(any(c.tp > 1 or c.ep > 1 for c in select_cases("planned")))

    def test_all_oracles_are_deterministic_and_input_dependent(self) -> None:
        for case in select_cases("core"):
            with self.subTest(case=case.id):
                a = reference(case, make_inputs(case, 123))
                again = reference(case, make_inputs(case, 123))
                other = reference(case, make_inputs(case, 987))
                self.assertEqual(set(a), set(again))
                self.assertTrue(all(torch.equal(a[k], again[k]) for k in a))
                self.assertTrue(any(not torch.equal(a[k], other[k]) for k in a))

    def test_speculative_token_accounting(self) -> None:
        case = select_cases("all", ["spec-b8-k4-v256"])[0]
        result = reference(case, make_inputs(case, 123))
        accepted = result["accepted_count"]
        committed = result["committed_count"]
        self.assertTrue(bool(((accepted >= 0) & (accepted <= 4)).all()))
        self.assertTrue(torch.equal(committed, accepted + 1))
        for row in range(8):
            count = int(committed[row])
            self.assertTrue(bool((result["committed_tokens"][row, :count] >= 0).all()))
            self.assertTrue(bool((result["committed_tokens"][row, count:] == -1).all()))

    def test_quantized_weight_formats(self) -> None:
        q4 = select_cases("all", ["quant-w4-b1-h64-i128"])[0]
        q8 = select_cases("all", ["quant-w8-b1-h64-i128"])[0]
        self.assertEqual(make_inputs(q4, 1)["w_gate"].dtype, torch.uint8)
        self.assertEqual(make_inputs(q4, 1)["w_gate"].shape[-1], 32)
        self.assertEqual(make_inputs(q8, 1)["w_gate"].dtype, torch.int8)
        self.assertEqual(make_inputs(q8, 1)["w_gate"].shape[-1], 64)


class HarnessTests(unittest.TestCase):
    def test_reference_submission_passes_cpu_correctness(self) -> None:
        for case_id in ("stencil-b4-n128-t4", "quant-w4-b1-h64-i128",
                        "spec-b1-k2-v128"):
            with self.subTest(case=case_id):
                case = select_cases("all", [case_id])[0]
                result = evaluate_case(case, REFERENCE_SUBMISSION,
                                       device="cpu", trials=2, warmup=0, reps=1)
                self.assertEqual(result["status"], "correctness_only")
                self.assertEqual(len(result["correctness"]["trials"]), 2)

    def test_wrong_output_is_rejected(self) -> None:
        case = select_cases("all", ["stencil-b4-n128-t4"])[0]
        want = reference(case, make_inputs(case, 1))
        bad = {"y": torch.zeros_like(want["y"])}
        with self.assertRaises(AssertionError):
            _compare(want, bad, case, "cpu")

    def test_parent_writes_exclusive_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            args = ["evaluate", "--submission", str(REFERENCE_SUBMISSION),
                    "--case", "stencil-b4-n128-t4", "--device", "cpu",
                    "--trials", "1", "--warmup", "0", "--reps", "1",
                    "--output", str(path)]
            with redirect_stdout(StringIO()):
                self.assertEqual(main(args), 1)  # CPU has no GPU launch audit.
            record = json.loads(path.read_text().strip())
            self.assertEqual(record["status"], "correctness_only")
            with redirect_stdout(StringIO()), self.assertRaises(FileExistsError):
                main(args)


if __name__ == "__main__":
    unittest.main()
