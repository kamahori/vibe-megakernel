"""CPU tests for the Sesame CSM-1B audio-frame task."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ..cases import select_cases
from ..harness.correctness import _teacher_forced
from ..harness.runner import evaluate_case
from ..tasks import csm
from ..tasks.oracle import fp32_reference
from ..verify_csm_primary import check_sources, compare, development, hf_frame, hf_model

REFERENCE_SUBMISSION = Path(__file__).resolve().parents[1] / "examples" / "reference_submission.py"


class CsmTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = development()

    def test_catalog_geometry_matches_pinned_config(self) -> None:
        from ..verify_csm_primary import pinned_spec

        case = select_cases("all", ["tts-frame-step-csm-1b-b1-s128"])[0]
        config = pinned_spec()["config"]
        depth = config["depth_decoder_config"]
        p = case.params
        self.assertEqual(
            (p["layers"], p["hidden"], p["q_heads"], p["kv_heads"], p["head_dim"],
             p["intermediate"], p["text_vocab"], p["codebooks"], p["audio_vocab"]),
            (config["num_hidden_layers"], config["hidden_size"], config["num_attention_heads"],
             config["num_key_value_heads"], config["head_dim"], config["intermediate_size"],
             config["text_vocab_size"], config["num_codebooks"], config["vocab_size"]))
        self.assertEqual(
            (p["depth_layers"], p["depth_hidden"], p["depth_q_heads"], p["depth_kv_heads"],
             p["depth_head_dim"], p["depth_intermediate"]),
            (depth["num_hidden_layers"], depth["hidden_size"], depth["num_attention_heads"],
             depth["num_key_value_heads"], depth["head_dim"], depth["intermediate_size"]))
        for scaling, pinned in ((csm.BACKBONE_ROPE, config), (csm.DEPTH_ROPE, depth)):
            self.assertEqual(pinned["rope_theta"], csm.ROPE_THETA)
            self.assertEqual(pinned["rms_norm_eps"], csm.RMS_EPS)
            self.assertEqual(pinned["rope_scaling"]["rope_type"], "llama3")
            self.assertEqual({key: pinned["rope_scaling"][key] for key in scaling}, scaling)

    def test_output_contract(self) -> None:
        p = self.case.params
        actual = csm.reference(self.case, csm.make_inputs(self.case, 17, "cpu"))
        shapes = {"codes": ((p["codebooks"],), torch.int64),
                  "logits": ((p["codebooks"], p["audio_vocab"]), torch.float32),
                  "k_write": ((p["layers"], p["kv_heads"], p["head_dim"]), torch.bfloat16),
                  "v_write": ((p["layers"], p["kv_heads"], p["head_dim"]), torch.bfloat16)}
        self.assertEqual(set(actual), set(shapes))
        for name, (shape, dtype) in shapes.items():
            self.assertEqual((tuple(actual[name].shape), actual[name].dtype), (shape, dtype))
        self.assertTrue(torch.equal(actual["codes"], actual["logits"].argmax(-1)))

    def test_determinism_input_dependence_and_immutability(self) -> None:
        with torch.inference_mode():
            values = csm.make_inputs(self.case, 17, "cpu")
            before = {name: value.clone() for name, value in values.items()}
            actual = csm.reference(self.case, values)
            repeat = csm.reference(self.case, csm.make_inputs(self.case, 17, "cpu"))
            other = csm.reference(self.case, csm.make_inputs(self.case, 19, "cpu"))
            self.assertTrue(all(torch.equal(actual[name], repeat[name]) for name in actual))
            self.assertFalse(torch.equal(actual["logits"], other["logits"]))
            self.assertTrue(all(torch.equal(values[name], before[name]) for name in values))
            self.assertTrue(torch.isfinite(actual["logits"]).all())
            # The previous frame and the cache both reach the outputs.
            for name in ("prev_codes", "kcache"):
                changed = dict(values)
                changed[name] = (values[name] + 1) % self.case.params["audio_vocab"] \
                    if name == "prev_codes" else values[name].flip(1)
                result = csm.reference(self.case, changed)
                self.assertFalse(torch.equal(result["logits"][0], actual["logits"][0]), name)

    def test_forced_codes(self) -> None:
        values = csm.make_inputs(self.case, 17, "cpu")
        actual = csm.reference(self.case, values)
        forced = csm.reference(self.case, values, forced_codes=actual["codes"].clone())
        self.assertTrue(all(torch.equal(actual[name], forced[name]) for name in actual))
        # A different history changes later codebooks but not codebook 0 or the K/V writes.
        other = actual["codes"].clone()
        other[3] = (other[3] + 1) % self.case.params["audio_vocab"]
        moved = csm.reference(self.case, values, forced_codes=other)
        self.assertTrue(torch.equal(moved["codes"], other))
        self.assertTrue(torch.equal(moved["logits"][:4], actual["logits"][:4]))
        self.assertFalse(torch.equal(moved["logits"][4:], actual["logits"][4:]))
        self.assertTrue(torch.equal(moved["k_write"], actual["k_write"]))
        exact = fp32_reference(csm, self.case, values, forced_codes=other)
        self.assertTrue(torch.equal(exact["codes"], other))
        self.assertEqual(exact["logits"].dtype, torch.float32)
        with self.assertRaises(ValueError):
            csm.reference(self.case, values,
                          forced_codes=torch.full_like(other, self.case.params["audio_vocab"]))

    def test_grader_regrades_on_submission_codes(self) -> None:
        values = csm.make_inputs(self.case, 17, "cpu")
        expected = csm.reference(self.case, values)
        exact = fp32_reference(csm, self.case, values)
        same = {name: value.clone() for name, value in expected.items()}
        self.assertIs(_teacher_forced(self.case, values, expected, exact, same)[0], expected)
        # A self-consistent frame that takes the least likely code at codebook 5.
        codes = expected["codes"].clone()
        codes[5] = expected["logits"][5].argmin()
        far = csm.reference(self.case, values, forced_codes=codes)
        with self.assertRaisesRegex(AssertionError, "codebook 5"):
            _teacher_forced(self.case, values, expected, exact, far)
        # Seed 2 has an exact BF16 tie at codebook 22. Taking the runner-up and
        # following the forced reference afterwards is accepted, and the logits
        # are regraded under the submission's own history.
        values = csm.make_inputs(self.case, 2, "cpu")
        expected = csm.reference(self.case, values)
        exact = fp32_reference(csm, self.case, values)
        codes = expected["codes"].clone()
        codes[22] = expected["logits"][22].topk(2).indices[1]
        for index in range(23, self.case.params["codebooks"]):
            codes[index] = csm.reference(self.case, values, forced_codes=codes)["logits"][index].argmax()
        tied = csm.reference(self.case, values, forced_codes=codes)
        self.assertFalse(torch.equal(tied["codes"], expected["codes"]))
        regraded, oracle = _teacher_forced(self.case, values, expected, exact, tied)
        self.assertTrue(torch.equal(regraded["logits"], tied["logits"]))
        self.assertTrue(torch.equal(oracle["codes"], tied["codes"]))

    def test_matches_transformers_csm(self) -> None:
        check_sources()
        for seed in (17, 18, 23):
            with self.subTest(seed=seed):
                values = csm.make_inputs(self.case, seed, "cpu")
                model = hf_model(self.case, values)
                report = compare(self.case, values, model)
                self.assertTrue(report["codes_equal"], report)
                upstream = hf_frame(model, self.case, values)
                ours = csm.reference(self.case, values)
                self.assertTrue(torch.equal(ours["codes"], upstream["codes"]))
                torch.testing.assert_close(ours["logits"], upstream["logits"],
                                           rtol=self.case.bf16_rtol, atol=self.case.atol)
                forced = csm.reference(self.case, values, forced_codes=upstream["codes"])
                torch.testing.assert_close(forced["logits"], upstream["forced_logits"],
                                           rtol=self.case.bf16_rtol, atol=self.case.atol)

    def test_reference_submission_passes_harness(self) -> None:
        case = replace(self.case, id="tts-frame-step-dev")
        result = evaluate_case(case, REFERENCE_SUBMISSION, device="cpu", trials=2,
                               warmup=0, reps=1)
        self.assertEqual(result["status"], "correctness_only", result)
        self.assertEqual(len(result["correctness"]["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
