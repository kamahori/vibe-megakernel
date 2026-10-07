"""Non-P0 format, model, and state checks at small development geometries."""

from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from ..cases import select_cases
from ..tasks import gemma, gptoss, hybrid, speculative, vision
from ..tasks.quantization import FP4_VALUES, pack_mxfp4, unpack_mxfp4


def tiny_gptoss():
    case = select_cases("p1", ["gptoss-step-20b-b1-s128"])[0]
    return replace(case, params=case.params | {
        "context": 4, "layers": 2, "hidden": 64, "q_heads": 4,
        "kv_heads": 2, "head_dim": 16, "intermediate": 64,
        "vocab": 128, "experts": 4, "topk": 2, "sliding_window": 3,
    }, atol=0.002, rtol=0.002)


def tiny_hybrid():
    case = select_cases("p1", ["hybrid-step-qwen35-08b-b1-s128"])[0]
    return replace(case, params=case.params | {
        "context": 4, "layers": 4, "hidden": 32, "q_heads": 4,
        "kv_heads": 2, "head_dim": 16, "rotary_dim": 4,
        "intermediate": 64, "vocab": 128, "linear_layers": 3,
        "full_attention_layers": 1, "linear_key_heads": 2,
        "linear_value_heads": 2, "linear_key_dim": 8,
        "linear_value_dim": 8,
    }, atol=0.002, rtol=0.002)


def tiny_vision():
    case = select_cases("p1", ["vl-decode-step-gemma3-4b-b1-s128"])[0]
    return replace(case, params=case.params | {
        "context": 4, "layers": 6, "hidden": 64, "q_heads": 2,
        "kv_heads": 1, "head_dim": 32, "intermediate": 128, "vocab": 256,
        "local_window": 3, "image_tokens": 4, "image_size": 8,
        "patch_size": 2, "vision_hidden": 32, "vision_layers": 2,
        "vision_intermediate": 48, "vision_heads": 4,
    }, atol=0.003, rtol=0.003)


def tiny_speculative():
    case = select_cases("p2", ["spec-full-iteration-llama31-8b-k4"])[0]
    return replace(case, params=case.params | {
        "context": 4, "layers": 4, "hidden": 64, "q_heads": 4,
        "kv_heads": 2, "head_dim": 16, "intermediate": 128,
        "vocab": 256, "draft_vocab": 64,
    }, atol=0.003, rtol=0.003)


class NativeQuantizationTests(unittest.TestCase):
    def test_all_native_codes_and_scales(self):
        blocks = torch.arange(16, dtype=torch.uint8).reshape(1, 1, 16)
        blocks = blocks | (blocks << 4)
        scales = torch.tensor([[127]], dtype=torch.uint8)
        expected = torch.tensor(FP4_VALUES).repeat_interleave(2).reshape(1, 32)
        self.assertTrue(torch.equal(unpack_mxfp4(blocks, scales), expected))
        scales.fill_(125)
        self.assertTrue(torch.equal(unpack_mxfp4(blocks, scales), expected / 4))
        scales.fill_(255)
        self.assertTrue(torch.isnan(unpack_mxfp4(blocks, scales)).all())

    def test_representable_weights_and_even_rounding(self):
        weights = torch.tensor(FP4_VALUES).repeat(2).reshape(1, 32)
        blocks, scales = pack_mxfp4(weights)
        self.assertTrue(torch.equal(unpack_mxfp4(blocks, scales), weights))
        weights = torch.zeros((1, 32))
        weights[0, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
        blocks, scales = pack_mxfp4(weights)
        expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0])
        self.assertTrue(torch.equal(unpack_mxfp4(blocks, scales)[0, :8], expected))

    def test_native_checkpoint_orientation_matches_transformers(self):
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors
        matrix = torch.randn((2, 64, 128)) * 0.1
        blocks, scales = pack_mxfp4(matrix)
        expected = _convert_moe_packed_tensors(blocks, scales, dtype=torch.float32)
        self.assertTrue(torch.equal(unpack_mxfp4(blocks, scales).transpose(1, 2), expected))


class NonP0ReferenceTests(unittest.TestCase):
    def test_speculative_tree_target_matches_upstream(self):
        from transformers import DynamicCache
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        case = tiny_speculative()
        p = case.params
        values = speculative.make_inputs(case, 17, "cpu")
        self.assertNotIn("draft_tokens", values)
        config = LlamaConfig(hidden_size=p["hidden"], intermediate_size=p["intermediate"],
                              num_hidden_layers=p["layers"], num_attention_heads=p["q_heads"],
                              num_key_value_heads=p["kv_heads"], head_dim=p["head_dim"],
                              vocab_size=p["vocab"], rms_norm_eps=1e-5,
                              rope_parameters={"rope_type": "llama3", "rope_theta": 500_000.0,
                                               "factor": 8.0, "low_freq_factor": 1.0,
                                               "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
                              max_position_embeddings=131072)
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).float().eval()
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(values["embed"])
            model.lm_head.weight.copy_(values["lm_head"])
            model.model.norm.weight.copy_(values["fnorm"])
            for index, layer in enumerate(model.model.layers):
                layer.input_layernorm.weight.copy_(values["ln1"][index])
                layer.post_attention_layernorm.weight.copy_(values["ln2"][index])
                for suffix in ("q", "k", "v", "o"):
                    getattr(layer.self_attn, suffix + "_proj").weight.copy_(values["w" + suffix][index])
                for suffix, key in (("gate", "wg"), ("up", "wu"), ("down", "wd")):
                    getattr(layer.mlp, suffix + "_proj").weight.copy_(values[key][index])
            for width in (1, 2):
                tree_case = replace(case, params=p | {"tree_width": width})
                tree = speculative.draft_tree(tree_case, values)
                expected = speculative.target_tree(tree_case, values, tree)
                for node in range(tree["tokens"].numel()):
                    path, parent = [], node
                    while parent >= 0:
                        path.append(parent)
                        parent = int(tree["parents"][parent])
                    path.reverse()
                    cache = DynamicCache()
                    for index in range(p["layers"]):
                        cache.update(values["kcache"][index].float().permute(1, 0, 2)[None],
                                     values["vcache"][index].float().permute(1, 0, 2)[None], index)
                    actual = model(input_ids=tree["tokens"][path][None], past_key_values=cache, use_cache=True, output_hidden_states=True)
                    torch.testing.assert_close(expected["logits"][node], actual.logits[0, -1], rtol=2e-5, atol=2e-5)
                    feature_layers = (min(2, p['layers'] - 1), p['layers'] // 2, max(0, p['layers'] - 3))
                    for slot, index in enumerate(feature_layers):
                        torch.testing.assert_close(expected['features'][node, slot],
                                                   actual.hidden_states[index][0, -1], rtol=2e-5, atol=2e-5)
                    for index in range(p["layers"]):
                        torch.testing.assert_close(expected["k_write"][index, node].float(),
                                                   cache.layers[index].keys[0, :, -1], rtol=case.bf16_rtol, atol=case.atol)

    def test_speculative_acceptance_and_both_cache_rollbacks(self):
        case = tiny_speculative()
        for depth in (2, 4, 8):
            current = replace(case, params=case.params | {"draft_depth": depth})
            for prefix in (0, 1, depth):
                with self.subTest(depth=depth, prefix=prefix):
                    values = speculative.make_inputs(current, 17, "cpu")
                    speculative.set_acceptance_scenario(current, values, prefix)
                    actual = speculative.reference(current, values)
                    self.assertEqual(actual["accepted_count"].item(), prefix)
                    self.assertEqual(actual["committed_count"].item(), prefix + 1)
                    self.assertEqual(actual["cache_length"].item(), current.params["context"] + prefix + 1)
                    self.assertEqual(actual["draft_cache_length"].item(), actual["cache_length"].item())
                    self.assertTrue((actual["committed_tokens"][:prefix + 1] == 0).all())
                    self.assertTrue((actual["committed_tokens"][prefix + 1:] == -1).all())
                    self.assertTrue((actual["k_write"][:, prefix + 1:] == 0).all())
                    self.assertTrue((actual["v_write"][:, prefix + 1:] == 0).all())
                    self.assertTrue((actual["draft_k_write"][prefix + 1:] == 0).all())
                    self.assertTrue((actual["draft_v_write"][prefix + 1:] == 0).all())

    def test_vision_encoder_and_projector_match_upstream(self):
        from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel
        from ..tasks.common import rms

        case = tiny_vision()
        p = case.params
        values = vision.vision_inputs(case, 17, "cpu")
        config = SiglipVisionConfig(hidden_size=p["vision_hidden"], intermediate_size=p["vision_intermediate"],
                                    num_hidden_layers=p["vision_layers"], num_attention_heads=p["vision_heads"],
                                    image_size=p["image_size"], patch_size=p["patch_size"], layer_norm_eps=1e-6,
                                    vision_use_head=False, hidden_act="gelu_pytorch_tanh")
        config._attn_implementation = "eager"
        model = SiglipVisionModel(config).float().eval()
        with torch.no_grad():
            model.embeddings.patch_embedding.weight.copy_(values["patch_weight"])
            model.embeddings.patch_embedding.bias.copy_(values["patch_bias"])
            model.embeddings.position_embedding.weight.copy_(values["position"])
            model.post_layernorm.weight.copy_(values["final_norm"])
            model.post_layernorm.bias.copy_(values["final_bias"])
            for index, layer in enumerate(model.encoder.layers):
                for num in (1, 2):
                    getattr(layer, f"layer_norm{num}").weight.copy_(values[f"norm{num}"][index])
                    getattr(layer, f"layer_norm{num}").bias.copy_(values[f"norm{num}_bias"][index])
                for suffix in ("q", "k", "v", "o"):
                    projection = getattr(layer.self_attn, "out_proj" if suffix == "o" else suffix + "_proj")
                    projection.weight.copy_(values[suffix][index])
                    projection.bias.copy_(values[suffix + "_bias"][index])
                for name, suffix in (("fc1", "up"), ("fc2", "down")):
                    getattr(layer.mlp, name).weight.copy_(values[suffix][index])
                    getattr(layer.mlp, name).bias.copy_(values[suffix + "_bias"][index])
            encoded = model(pixel_values=values["pixels"]).last_hidden_state
            pooled = torch.nn.functional.avg_pool2d(encoded.transpose(1, 2).reshape(1, 32, 4, 4), 2)
            pooled = pooled.flatten(2).transpose(1, 2)
            expected = rms(pooled, values["projector_norm"], gemma=True) @ values["projector"].float()
            actual = vision.image_features(case, values)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_multimodal_prefill_matches_upstream_and_depends_on_image(self):
        from transformers import DynamicCache
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
        from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

        case = tiny_vision()
        p = case.params
        values = gemma.make_inputs(replace(case, params=p | {"bits": 16}), 17, "cpu")
        images = vision.vision_inputs(case, 17, "cpu")
        features = vision.image_features(case, images).reshape(-1, p["hidden"])
        prompt = torch.arange(p["context"])
        embeddings = torch.cat((features, values["embed"][prompt].float() * p["hidden"] ** 0.5))
        blocks = torch.tensor([0] * p["image_tokens"] + [-1] * p["context"])
        expected_k, expected_v = vision.prefill(case, values, embeddings, blocks)
        config = Gemma3TextConfig(hidden_size=p["hidden"], intermediate_size=p["intermediate"],
                                  num_hidden_layers=p["layers"], num_attention_heads=p["q_heads"],
                                  num_key_value_heads=p["kv_heads"], head_dim=p["head_dim"],
                                  vocab_size=p["vocab"], sliding_window=p["local_window"],
                                  query_pre_attn_scalar=p["head_dim"],
                                  rope_parameters={"full_attention": {"rope_type": "linear", "factor": 8.0,
                                                                      "rope_theta": 1_000_000.0},
                                                   "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0}})
        config._attn_implementation = "eager"
        model = Gemma3TextModel(config).float().eval()
        with torch.no_grad():
            model.embed_tokens.weight.copy_(values["embed"])
            model.norm.weight.copy_(values["fnorm"])
            for index, layer in enumerate(model.layers):
                for key, name in (("ln1", "input_layernorm"), ("ln2", "post_attention_layernorm"),
                                  ("ln3", "pre_feedforward_layernorm"), ("ln4", "post_feedforward_layernorm")):
                    getattr(layer, name).weight.copy_(values[key][index])
                for suffix in ("q", "k", "v", "o"):
                    getattr(layer.self_attn, suffix + "_proj").weight.copy_(values["w" + suffix][index])
                layer.self_attn.q_norm.weight.copy_(values["qn"][index])
                layer.self_attn.k_norm.weight.copy_(values["kn"][index])
                for suffix, key in (("gate", "wg"), ("up", "wu"), ("down", "wd")):
                    getattr(layer.mlp, suffix + "_proj").weight.copy_(values[key][index])
            positions = torch.arange(embeddings.shape[0])
            allowed = positions[None] <= positions[:, None]
            allowed |= (blocks[:, None] == blocks[None]) & (blocks[:, None] >= 0)
            local = allowed & (positions[None] > positions[:, None] - p["local_window"])
            masks = {"full_attention": torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, -torch.inf)[None, None],
                     "sliding_attention": torch.zeros_like(local, dtype=torch.float32).masked_fill(~local, -torch.inf)[None, None]}
            cache = DynamicCache()
            model(inputs_embeds=embeddings[None], attention_mask=masks,
                  past_key_values=cache, use_cache=True)
            for index in range(p["layers"]):
                torch.testing.assert_close(expected_k[index].float(), cache.layers[index].keys[0].transpose(0, 1),
                                           rtol=case.bf16_rtol, atol=case.atol)
                torch.testing.assert_close(expected_v[index].float(), cache.layers[index].values[0].transpose(0, 1),
                                           rtol=case.bf16_rtol, atol=case.atol)
        values["kcache"], values["vcache"] = expected_k, expected_v
        first = vision.reference(case, values)
        images["pixels"].neg_()
        changed_features = vision.image_features(case, images).reshape(-1, p["hidden"])
        changed = torch.cat((changed_features, embeddings[p["image_tokens"]:]))
        values["kcache"], values["vcache"] = vision.prefill(case, values, changed, blocks)
        second = vision.reference(case, values)
        self.assertFalse(torch.equal(first["logits"], second["logits"]))

    def test_gptoss_whole_step_matches_upstream(self):
        from transformers import DynamicCache
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

        case = tiny_gptoss()
        p = case.params
        values = gptoss.make_inputs(case, 17, "cpu")
        config = GptOssConfig(hidden_size=p["hidden"], intermediate_size=p["intermediate"],
                              num_hidden_layers=p["layers"], num_attention_heads=p["q_heads"],
                              num_key_value_heads=p["kv_heads"], head_dim=p["head_dim"],
                              num_local_experts=p["experts"], num_experts_per_tok=p["topk"],
                              vocab_size=p["vocab"], sliding_window=p["sliding_window"],
                              tie_word_embeddings=False)
        config._attn_implementation = "eager"
        model = GptOssForCausalLM(config).float().eval()
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(values["embed"])
            model.lm_head.weight.copy_(values["lm_head"])
            model.model.norm.weight.copy_(values["fnorm"])
            for index, layer in enumerate(model.model.layers):
                layer.input_layernorm.weight.copy_(values["ln1"][index])
                layer.post_attention_layernorm.weight.copy_(values["ln2"][index])
                for suffix in ("q", "k", "v", "o"):
                    projection = getattr(layer.self_attn, suffix + "_proj")
                    projection.weight.copy_(values["w" + suffix][index])
                    projection.bias.copy_(values["b" + suffix][index])
                layer.self_attn.sinks.copy_(values["sinks"][index])
                layer.mlp.router.weight.copy_(values["router"][index])
                layer.mlp.router.bias.copy_(values["router_bias"][index])
                layer.mlp.experts.gate_up_proj.copy_(unpack_mxfp4(
                    values["gate_up_blocks"][index], values["gate_up_scales"][index]))
                layer.mlp.experts.down_proj.copy_(unpack_mxfp4(
                    values["down_blocks"][index], values["down_scales"][index]))
                layer.mlp.experts.gate_up_proj_bias.copy_(values["gate_up_bias"][index])
                layer.mlp.experts.down_proj_bias.copy_(values["down_bias"][index])
            cache = DynamicCache()
            for index in range(p["layers"]):
                cache.update(values["kcache"][index].float().permute(1, 0, 2)[None],
                             values["vcache"][index].float().permute(1, 0, 2)[None], index)
            actual = model(input_ids=values["token"].reshape(1, 1), past_key_values=cache,
                           use_cache=True)
            expected = gptoss.reference(case, values)
        torch.testing.assert_close(expected["logits"], actual.logits[0, 0], rtol=1e-5, atol=1e-5)
        for index in range(p["layers"]):
            key = cache.layers[index].keys[0, :, -1].to(torch.bfloat16)
            value = cache.layers[index].values[0, :, -1].to(torch.bfloat16)
            torch.testing.assert_close(expected["k_write"][index], key, rtol=case.bf16_rtol, atol=case.atol)
            torch.testing.assert_close(expected["v_write"][index], value, rtol=case.bf16_rtol, atol=case.atol)

    def test_hybrid_whole_step_and_rollout_match_upstream(self):
        from transformers import DynamicCache
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        case = tiny_hybrid()
        p = case.params
        config = Qwen3_5TextConfig(
            hidden_size=p["hidden"], intermediate_size=p["intermediate"],
            num_hidden_layers=p["layers"], num_attention_heads=p["q_heads"],
            num_key_value_heads=p["kv_heads"], head_dim=p["head_dim"],
            vocab_size=p["vocab"], tie_word_embeddings=True,
            linear_num_key_heads=p["linear_key_heads"], linear_num_value_heads=p["linear_value_heads"],
            linear_key_head_dim=p["linear_key_dim"], linear_value_head_dim=p["linear_value_dim"],
            linear_conv_kernel_dim=p["conv_kernel"],
            rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.0,
                             "partial_rotary_factor": p["rotary_dim"] / p["head_dim"],
                             "mrope_section": [1, 1, 0], "mrope_interleaved": True})
        config._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(config).float().eval()
        values = hybrid.make_inputs(case, 17, "cpu")
        values["reset"].fill_(False)
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(values["embed"])
            model.model.norm.weight.copy_(values["fnorm"])
            linear, full = 0, 0
            for index, layer in enumerate(model.model.layers):
                layer.input_layernorm.weight.copy_(values["ln1"][index])
                layer.post_attention_layernorm.weight.copy_(values["ln2"][index])
                for suffix in ("gate", "up", "down"):
                    name = {"gate": "wg", "up": "wu", "down": "wd"}[suffix]
                    getattr(layer.mlp, suffix + "_proj").weight.copy_(values[name][index])
                if (index + 1) % p["full_attention_interval"]:
                    attention = layer.linear_attn
                    for suffix in ("qkv", "z", "a", "b"):
                        getattr(attention, "in_proj_" + suffix).weight.copy_(values["in_" + suffix][linear])
                    attention.out_proj.weight.copy_(values["out_linear"][linear])
                    attention.conv1d.weight.copy_(values["conv_weight"][linear][:, None])
                    attention.A_log.copy_(values["A_log"][linear])
                    attention.dt_bias.copy_(values["dt_bias"][linear])
                    attention.norm.weight.copy_(values["linear_norm"][linear])
                    linear += 1
                else:
                    for suffix in ("q", "k", "v", "o"):
                        getattr(layer.self_attn, suffix + "_proj").weight.copy_(values["w" + suffix][full])
                    layer.self_attn.q_norm.weight.copy_(values["qn"][full])
                    layer.self_attn.k_norm.weight.copy_(values["kn"][full])
                    full += 1
            cache = DynamicCache(config=config)
            linear, full = 0, 0
            for index in range(p["layers"]):
                if (index + 1) % p["full_attention_interval"]:
                    cache.update_conv_state(values["conv_state"][linear][None].clone(), index)
                    cache.update_recurrent_state(values["recurrent_state"][linear][None].clone(), index)
                    linear += 1
                else:
                    cache.update(values["kcache"][full].float().permute(1, 0, 2)[None],
                                 values["vcache"][full].float().permute(1, 0, 2)[None], index)
                    full += 1
            for step in range(3):
                current = replace(case, params=p | {"context": p["context"] + step})
                expected = hybrid.reference(current, values)
                actual = model(input_ids=values["token"].reshape(1, 1), past_key_values=cache,
                               position_ids=torch.tensor([[p["context"] + step]]),
                               attention_mask={"full_attention": None, "linear_attention": None},
                               use_cache=True)
                torch.testing.assert_close(expected["logits"], actual.logits[0, 0], rtol=2e-5, atol=2e-5)
                linear, full = 0, 0
                for index in range(p["layers"]):
                    layer_cache = cache.layers[index]
                    if (index + 1) % p["full_attention_interval"]:
                        torch.testing.assert_close(expected["recurrent_state"][linear],
                                                   layer_cache.recurrent_states[0][0], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(expected["conv_state"][linear],
                                                   layer_cache.conv_states[0][0], rtol=1e-5, atol=1e-5)
                        linear += 1
                    else:
                        torch.testing.assert_close(expected["k_write"][full].float(),
                                                   layer_cache.keys[0, :, -1], rtol=case.bf16_rtol, atol=case.atol)
                        torch.testing.assert_close(expected["v_write"][full].float(),
                                                   layer_cache.values[0, :, -1], rtol=case.bf16_rtol, atol=case.atol)
                        # MegaBench retains BF16 KV between calls.
                        layer_cache.keys = layer_cache.keys.to(torch.bfloat16).float()
                        layer_cache.values = layer_cache.values.to(torch.bfloat16).float()
                        full += 1
                values["kcache"] = torch.cat((values["kcache"], expected["k_write"][:, None]), 1)
                values["vcache"] = torch.cat((values["vcache"], expected["v_write"][:, None]), 1)
                values["recurrent_state"] = expected["recurrent_state"]
                values["conv_state"] = expected["conv_state"]
                values["token"] = expected["next_token"]

    def test_delta_recurrence_matches_upstream(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
        g = torch.Generator().manual_seed(93)
        query, key, value = [torch.randn((2, 8), generator=g) for _ in range(3)]
        state = torch.randn((2, 8, 8), generator=g)
        decay = -torch.rand(2, generator=g)
        beta = torch.rand(2, generator=g)
        output, updated = hybrid.delta_step(query, key, value, state, decay, beta)
        want_output, want_state = torch_recurrent_gated_delta_rule(
            query[None, None], key[None, None], value[None, None],
            decay[None, None], beta[None, None], initial_state=state[None],
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(output, want_output[0, 0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(updated, want_state[0], rtol=1e-6, atol=1e-6)

    def test_fresh_inputs_determinism_and_immutability(self):
        for module, case in ((gptoss, tiny_gptoss()), (hybrid, tiny_hybrid()),
                             (vision, tiny_vision()), (speculative, tiny_speculative())):
            with self.subTest(family=case.family), torch.inference_mode():
                values = module.make_inputs(case, 17, "cpu")
                before = {name: value.clone() for name, value in values.items()}
                actual = module.reference(case, values)
                repeat = module.reference(case, module.make_inputs(case, 17, "cpu"))
                other = module.reference(case, module.make_inputs(case, 19, "cpu"))
                self.assertTrue(all(torch.equal(actual[name], repeat[name]) for name in actual))
                self.assertFalse(torch.equal(actual["logits"], other["logits"]))
                self.assertTrue(all(torch.equal(values[name], before[name]) for name in values))
                self.assertTrue(torch.isfinite(actual["logits"]).all())

    def test_hybrid_reset_discards_prior_state(self):
        case = tiny_hybrid()
        values = hybrid.make_inputs(case, 17, "cpu")
        values["reset"].fill_(True)
        first = hybrid.reference(case, values)
        for name in ("conv_state", "recurrent_state", "kcache", "vcache"):
            values[name].fill_(100)
        second = hybrid.reference(case, values)
        self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))
        self.assertEqual(first["cache_length"].item(), 1)


if __name__ == "__main__":
    unittest.main()
