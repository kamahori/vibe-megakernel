"""pi0.5 action-chunk task at a tiny CPU geometry."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ..cases import CASES
from ..harness.runner import evaluate_case
from ..tasks import pi05
from ..tasks.oracle import fp32_reference
from .test_non_p0 import serving_model

REFERENCE_SUBMISSION = Path(__file__).resolve().parents[1] / "examples" / "reference_submission.py"


def tiny_pi05():
    case = next(case for case in CASES if case.id == "vla-action-step-pi05-b1")
    return replace(case, ready=True, params=case.params | {
        "images": 2, "image_tokens": 4, "image_size": 8, "patch_size": 4, "text_tokens": 64,
        "vision_hidden": 32, "vision_layers": 2, "vision_intermediate": 48, "vision_heads": 4,
        "layers": 3, "hidden": 64, "intermediate": 128, "q_heads": 4, "kv_heads": 1,
        "head_dim": 16, "expert_hidden": 32, "expert_intermediate": 64, "horizon": 6,
        "action_dim": 8, "denoise_steps": 4})


class Pi05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = tiny_pi05()
        cls.values = pi05.make_inputs(cls.case, 7, "cpu")

    def run_reference(self, values):
        return pi05.reference(self.case, values)["actions"]

    def test_layout_and_dtypes(self):
        p, v = self.case.params, self.values
        length = pi05.prefix_length(self.case)
        self.assertEqual(tuple(v["prefix_k"].shape), (p["layers"], length, p["kv_heads"], p["head_dim"]))
        self.assertEqual(v["prefix_k"].dtype, torch.bfloat16)
        self.assertEqual(v["prefix_mask"].dtype, torch.bool)
        self.assertEqual(v["noise"].dtype, torch.float32)
        self.assertEqual(v["attn_norm_weight"].dtype, torch.float32)
        self.assertEqual(v["wq"].dtype, torch.bfloat16)
        self.assertFalse({"embed", "pixels", "lq", "q"} & set(v), "VLM weights must be dropped")
        images = p["images"] * p["image_tokens"]
        mask = v["prefix_mask"]
        # Images are valid; the prompt is right padded.
        self.assertTrue(bool(mask[:images].all()))
        self.assertFalse(bool(mask[-1]))
        self.assertTrue(bool((mask[images:].long().diff() <= 0).all()))
        out = self.run_reference(v)
        self.assertEqual(tuple(out.shape), (p["horizon"], p["action_dim"]))
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_deterministic_and_seed_dependent(self):
        again = pi05.make_inputs(self.case, 7, "cpu")
        self.assertEqual(set(again), set(self.values))
        for name, value in self.values.items():
            self.assertTrue(torch.equal(value, again[name]), name)
        self.assertTrue(torch.equal(self.run_reference(self.values), self.run_reference(again)))
        other = pi05.make_inputs(self.case, 8, "cpu")
        self.assertFalse(torch.equal(other["prefix_k"], self.values["prefix_k"]))
        self.assertFalse(torch.equal(self.run_reference(other), self.run_reference(self.values)))

    def test_no_input_mutation(self):
        before = {name: value.clone() for name, value in self.values.items()}
        self.run_reference(self.values)
        fp32_reference(pi05, self.case, self.values)
        for name, value in self.values.items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_noise_and_prefix_dependence(self):
        base = self.run_reference(self.values)
        mask = self.values["prefix_mask"]
        for name in ("noise", "prefix_k", "prefix_v"):
            changed = dict(self.values)
            changed[name] = self.values[name].clone()
            if name == "noise":
                changed[name][0, 0] += 1.0
            else:
                changed[name][:, mask] = changed[name][:, mask].flip(1)
            self.assertFalse(torch.allclose(self.run_reference(changed), base), name)
        # Padded prefix entries are masked out.
        padded = dict(self.values)
        for name in ("prefix_k", "prefix_v"):
            padded[name] = self.values[name].clone()
            padded[name][:, ~mask] = 100.0
        self.assertTrue(torch.equal(self.run_reference(padded), base))
        # Unmasking padding changes the positions and the visible keys.
        unmasked = self.values | {"prefix_mask": torch.ones_like(mask)}
        self.assertFalse(torch.allclose(self.run_reference(unmasked), base))

    def test_bf16_reference_is_near_fp32_oracle(self):
        want = self.run_reference(self.values)
        exact = fp32_reference(pi05, self.case, self.values)["actions"]
        error = float((want - exact).norm() / exact.norm())
        self.assertLess(error, 0.02)
        self.assertGreater(float((want - self.values["noise"]).norm()), 0.1)

    def test_time_embedding_matches_transformers_pi0(self):
        from transformers.models.pi0.modeling_pi0 import PI0TimestepEmbeddings

        p = self.case.params
        config = type("Config", (), {"min_period": pi05.MIN_PERIOD, "max_period": pi05.MAX_PERIOD,
                                     "dit_config": type("Dit", (), {"hidden_size": p["expert_hidden"]})})
        times = torch.tensor([1.0, 0.75, 0.25, 0.001])
        # Transformers forms FP32 phases; openpi/LeRobot form FP64 phases
        # (up to 2*pi/0.004 rad per unit time), so they agree to ~1e-4.
        torch.testing.assert_close(pi05.time_embedding(times, p["expert_hidden"]),
                                   PI0TimestepEmbeddings(config)(times), rtol=0, atol=2e-4)

    def test_neutral_adarms_expert_matches_transformers_gemma_over_prefix_cache(self):
        """With scale=w, shift=0, gate=1 adaRMS is Gemma RMSNorm; pi0's expert is GemmaModel."""
        from transformers import DynamicCache, GemmaConfig, GemmaModel

        p = self.case.params
        w = p["expert_hidden"]
        values = dict(self.values)
        g = torch.Generator().manual_seed(3)
        scales = {}

        def neutral(shape_prefix):
            scale = (0.1 * torch.randn((*shape_prefix, w), generator=g)).bfloat16().float()
            bias = torch.cat((scale, torch.zeros_like(scale), torch.ones_like(scale)), -1)
            return scale, torch.zeros((*shape_prefix, 3 * w, w)), bias

        for name in ("attn_norm", "ffn_norm"):
            scales[name], values[name + "_weight"], values[name + "_bias"] = neutral((p["layers"],))
        scales["final"], values["final_norm_weight"], values["final_norm_bias"] = neutral(())
        config = GemmaConfig(hidden_size=w, intermediate_size=p["expert_intermediate"],
                             num_hidden_layers=p["layers"], num_attention_heads=p["q_heads"],
                             num_key_value_heads=p["kv_heads"], head_dim=p["head_dim"],
                             vocab_size=16, hidden_act="gelu_pytorch_tanh", rms_norm_eps=1e-6)
        config._attn_implementation = "eager"
        model = serving_model(GemmaModel(config))
        with torch.no_grad():
            for index, layer in enumerate(model.layers):
                layer.input_layernorm.weight.copy_(scales["attn_norm"][index])
                layer.post_attention_layernorm.weight.copy_(scales["ffn_norm"][index])
                for module, key in ((layer.self_attn.q_proj, "wq"), (layer.self_attn.k_proj, "wk"),
                                    (layer.self_attn.v_proj, "wv"), (layer.self_attn.o_proj, "wo"),
                                    (layer.mlp.gate_proj, "wg"), (layer.mlp.up_proj, "wu"),
                                    (layer.mlp.down_proj, "wd")):
                    module.weight.copy_(values[key][index])
            model.norm.weight.copy_(scales["final"])
            positions, allowed = pi05.suffix_layout(self.case, values["prefix_mask"])
            h = torch.randn((p["horizon"], w), generator=g).bfloat16()
            cond = pi05.time_condition(self.case, values, torch.tensor([0.4]))
            ours = pi05.expert(self.case, values, h, cond, positions, allowed)
            cache = DynamicCache()
            for index in range(p["layers"]):
                cache.update(values["prefix_k"][index].transpose(0, 1)[None].clone(),
                             values["prefix_v"][index].transpose(0, 1)[None].clone(), index)
            additive = torch.where(allowed, 0.0, pi05.MASK_VALUE)[None, None]
            upstream = model(inputs_embeds=h[None], attention_mask=additive,
                             position_ids=positions[None], past_key_values=cache,
                             use_cache=True).last_hidden_state[0]
        self.assertTrue(torch.equal(ours, upstream))

    def test_reference_submission_passes_cpu_harness(self):
        result = evaluate_case(self.case, REFERENCE_SUBMISSION, device="cpu",
                               trials=2, warmup=0, reps=1)
        self.assertEqual(result["status"], "correctness_only", result)
        self.assertEqual(len(result["correctness"]["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
