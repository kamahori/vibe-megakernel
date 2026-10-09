"""GLM-5.3-Flash decode against the upstream transformers glm5_next text model."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from ..cases import select_cases
from ..tasks import glm53
from ..tasks.oracle import WidenBf16
from ..tasks.quantization import pack_fp8_activation, unpack_fp8_activation, unpack_fp8_weight


def development(case=None):
    case = case or select_cases('all', ['glm53-flash-step'])[0]
    # Eight layers place MLA/DSA at 3 and 7. Ten cached-plus-current tokens
    # form two complete k-pools and a two-token tail; index_topk keeps one
    # pool, so the selection is genuinely sparse.
    return replace(case, params=case.params | {
        'layers': 8, 'hidden': 128, 'linear_heads': 4, 'linear_head_dim': 32,
        'q_heads': 4, 'q_lora_rank': 128, 'kv_lora_rank': 64, 'qk_nope_dim': 32,
        'v_head_dim': 32, 'index_heads': 4, 'index_dim': 32, 'index_topk': 4,
        'experts': 8, 'topk': 2, 'intermediate': 128, 'dense_intermediate': 256,
        'first_dense': 1, 'vocab': 512, 'context': 9,
    }, atol=0.003, rtol=0.003)


def quantized(value):
    payload, scales = pack_fp8_activation(value)
    return unpack_fp8_activation(payload, scales).to(value.dtype)


def upstream_model(case, values):
    """Load reference inputs into transformers' Glm5NextTextModel (FP32)."""
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    p = case.params
    config = Glm5NextTextConfig(
        vocab_size=p['vocab'], hidden_size=p['hidden'], intermediate_size=p['dense_intermediate'],
        moe_intermediate_size=p['intermediate'], num_hidden_layers=p['layers'],
        num_attention_heads=p['q_heads'], num_key_value_heads=p['q_heads'],
        n_shared_experts=p['shared_experts'], n_routed_experts=p['experts'], num_experts_per_tok=p['topk'],
        kv_lora_rank=p['kv_lora_rank'], q_lora_rank=p['q_lora_rank'], qk_rope_head_dim=0,
        qk_nope_head_dim=p['qk_nope_dim'], v_head_dim=p['v_head_dim'], index_topk=p['index_topk'],
        index_head_dim=p['index_dim'], index_n_heads=p['index_heads'], index_kpool=p['index_kpool'],
        linear_head_dim=p['linear_head_dim'], linear_num_heads=p['linear_heads'],
        linear_conv_kernel_dim=p['conv_kernel'], hc_mult=p['hc_mult'],
        hc_sinkhorn_iters=p['hc_sinkhorn_iters'], pad_token_id=None,
        mlp_layer_types=['dense' if layer < p['first_dense'] else 'sparse' for layer in range(p['layers'])])
    config._attn_implementation = 'eager'
    config._experts_implementation = 'eager'
    model = Glm5NextTextModel(config).float().eval()
    dense = lambda name: unpack_fp8_weight(values[name], values[name+'_scale'])

    def fp8_input(module):
        module.register_forward_pre_hook(lambda module, args: (quantized(args[0]), *args[1:]))

    with torch.no_grad():
        model.embed_tokens.weight.copy_(values['embed'])
        model.norm.weight.copy_(values['fnorm'])
        for layer, block in enumerate(model.layers):
            prefix = f'l{layer}_'
            block.input_layernorm.weight.copy_(values['ln1'][layer])
            block.post_attention_layernorm.weight.copy_(values['ln2'][layer])
            for site in ('attn_hc', 'ffn_hc'):
                module = getattr(block, site)
                for name in ('fn', 'base', 'scale'):
                    getattr(module, name).copy_(values[f'{site}_{name}'][layer])
            attention = block.self_attn
            if glm53.mla_layer(layer):
                for short, long in (('qa', 'q_a_proj'), ('qb', 'q_b_proj'), ('ka', 'kv_a_proj_with_mqa'), ('o', 'o_proj')):
                    getattr(attention, long).weight.copy_(dense(prefix+short))
                    fp8_input(getattr(attention, long))
                attention.kv_b_proj.weight.copy_(values[prefix+'kb'])
                attention.q_a_layernorm.weight.copy_(values[prefix+'qn'])
                attention.kv_a_layernorm.weight.copy_(values[prefix+'kn'])
                indexer = attention.indexer
                for short, long in (('iq', 'wq_b'), ('ik', 'wk'), ('iw', 'weights_proj')):
                    getattr(indexer, long).weight.copy_(values[prefix+short])
                indexer.k_norm.weight.copy_(values[prefix+'inorm'])
                indexer.k_norm.bias.copy_(values[prefix+'ibias'])
                indexer.index_kpool_compress_gate.copy_(values[prefix+'igate'])
                indexer.index_kpool_compress_ape.copy_(values[prefix+'iape'])
            else:
                for short, long in (('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'o_proj'),
                                    ('beta', 'b_proj'), ('ga', 'g_a_proj'), ('gb', 'g_b_proj')):
                    getattr(attention, long).weight.copy_(values[prefix+short])
                attention.forget_gate.f_a_proj.weight.copy_(values[prefix+'fa'])
                attention.forget_gate.f_b_proj.weight.copy_(values[prefix+'fb'])
                attention.forget_gate.A_log.copy_(values[prefix+'A_log'])
                attention.forget_gate.dt_bias.copy_(values[prefix+'dt_bias'])
                attention.conv1d.weight.copy_(values[prefix+'conv_weight'].flatten(0, 1)[:, None])
                attention.o_norm.weight.copy_(values[prefix+'onorm'])
            mlp = block.mlp
            if layer < p['first_dense']:
                for name in ('gate', 'up', 'down'):
                    getattr(mlp, name+'_proj').weight.copy_(dense(prefix+name))
                    fp8_input(getattr(mlp, name+'_proj'))
            else:
                mlp.gate.weight.copy_(values[prefix+'router'])
                mlp.gate.e_score_correction_bias.copy_(values[prefix+'router_bias'])
                for name in ('gate', 'up', 'down'):
                    getattr(mlp.shared_experts, name+'_proj').weight.copy_(dense(prefix+'shared_'+name))
                    fp8_input(getattr(mlp.shared_experts, name+'_proj'))
                expert = lambda name, e: unpack_fp8_weight(values[prefix+name][e], values[prefix+name+'_scale'][e])
                mlp.experts.gate_up_proj.copy_(torch.stack([torch.cat((expert('gate', e), expert('up', e)))
                                                            for e in range(p['experts'])]))
                mlp.experts.down_proj.copy_(torch.stack([expert('down', e) for e in range(p['experts'])]))
                fp8_input(mlp.experts)
                gate = mlp.experts._apply_gate
                mlp.experts._apply_gate = lambda gate_up, gate=gate: quantized(gate(gate_up))
    return config, model


def seeded_cache(config, model, case, values):
    from transformers import DynamicCache
    p = case.params
    cache = DynamicCache(config=config)
    with torch.no_grad():
        for layer in range(p['layers']):
            prefix = f'l{layer}_'
            if glm53.mla_layer(layer):
                nope, vd = p['qk_nope_dim'], p['v_head_dim']
                expanded = model.layers[layer].self_attn.kv_b_proj(values[prefix+'kv_cache'].float())
                expanded = expanded.view(p['context'], p['q_heads'], nope+vd).transpose(0, 1)[None]
                cache.update(expanded[..., :nope].contiguous(), expanded[..., nope:].contiguous(), layer)
                valid = torch.ones(p['context'], 1)
                cache.update_indexer(torch.cat((values[prefix+'index_k_cache'].float(),
                                                values[prefix+'index_gate_cache'].float(), valid), -1)[None], layer)
            else:
                cache.layers[layer].update_conv_state(values[prefix+'conv_state'].flatten(0, 1).float()[None])
                cache.layers[layer].update_recurrent_state(values[prefix+'recurrent_state'][None].clone())
    return cache


class Glm53ReferenceTests(unittest.TestCase):
    def test_decode_matches_upstream_glm5_next_text_model(self):
        case = replace(development(), gpus=1, tp=1, ready=True)
        p = case.params
        for seed in (17, 19, 23):
            with self.subTest(seed=seed):
                values = glm53.make_inputs(case, seed, 'cpu', rank=0)
                with WidenBf16():
                    expected = glm53.reference(case, values, serial=True)
                config, model = upstream_model(case, values)
                cache = seeded_cache(config, model, case, values)
                captured = {'latent': [], 'masks': [], 'pools': [], 'scores': [], 'experts': []}
                for layer, block in enumerate(model.layers):
                    attention = block.self_attn
                    if glm53.mla_layer(layer):
                        attention.kv_a_layernorm.register_forward_hook(
                            lambda module, args, out: captured['latent'].append(out[0, 0]))
                        mask_builder = attention.build_attention_mask_from_topk
                        def build(topk_indices, query_states, kv_length, mask_builder=mask_builder):
                            captured['pools'].append(topk_indices[0, 0])
                            mask = mask_builder(topk_indices, query_states, kv_length)
                            captured['masks'].append(mask[0, 0, 0] == 0)
                            return mask
                        attention.build_attention_mask_from_topk = build
                    if layer >= p['first_dense']:
                        block.mlp.gate.register_forward_hook(
                            lambda module, args, out: captured['experts'].append(out[2][0].sort().values))
                matmul = torch.matmul
                def observe(left, right, *args, **kwargs):
                    result = matmul(left, right, *args, **kwargs)
                    if left.shape[-2:] == (1, p['index_heads']) and right.shape[-2] == p['index_heads']:
                        captured['scores'].append(result[0, 0, 0])
                    return result
                with torch.no_grad(), patch('torch.matmul', observe):
                    hidden = model(input_ids=values['token'].view(1, 1), past_key_values=cache, use_cache=True).last_hidden_state
                logits = values['lm_head'].float() @ hidden[0, 0]
                torch.testing.assert_close(expected['logits'], logits, rtol=2e-3, atol=2e-3)
                self.assertEqual(int(expected['next_token']), int(logits.argmax()))
                torch.testing.assert_close(expected['kv_write'], torch.stack(captured['latent']), rtol=1e-4, atol=1e-4)
                self.assertTrue(torch.equal(expected['sparse_mask'], torch.stack(captured['masks'])))
                self.assertFalse(bool(expected['sparse_mask'].all()))
                pools = p['context']+1 >> 2
                torch.testing.assert_close(expected['index_scores'], torch.stack(captured['scores'])[:, :pools],
                                           rtol=1e-4, atol=1e-5)
                expert_ids = expected['expert_ids'][p['first_dense']:].sort(-1).values
                self.assertTrue(torch.equal(expert_ids, torch.stack(captured['experts'])))
                mla = [layer for layer in range(p['layers']) if glm53.mla_layer(layer)]
                for slot, layer in enumerate(mla):
                    packed = cache.layers[layer].indexer_keys[0, -1]
                    torch.testing.assert_close(expected['index_k_write'][slot], packed[:p['index_dim']], rtol=1e-4, atol=1e-4)
                    torch.testing.assert_close(expected['index_gate_write'][slot], packed[p['index_dim']:-1], rtol=1e-4, atol=1e-4)
                kda = [layer for layer in range(p['layers']) if not glm53.mla_layer(layer)]
                for slot, layer in enumerate(kda):
                    torch.testing.assert_close(expected['recurrent_state'][slot], cache.layers[layer].recurrent_states[0][0],
                                               rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(expected['conv_state'][slot].flatten(0, 1).float(),
                                               cache.layers[layer].conv_states[0][0], rtol=1e-4, atol=1e-5)

    def test_bf16_reference_tracks_fp32_oracle_and_preserves_inputs(self):
        case = replace(development(), gpus=1, tp=1, ready=True)
        values = glm53.make_inputs(case, 29, 'cpu', rank=0)
        before = {name: value.clone() for name, value in values.items()}
        actual = glm53.reference(case, values, serial=True)
        with WidenBf16():
            exact = glm53.reference(case, values, serial=True)
        for name, value in values.items():
            self.assertTrue(torch.equal(value.view(torch.uint8) if value.dtype == torch.float8_e4m3fn else value,
                                        before[name].view(torch.uint8) if value.dtype == torch.float8_e4m3fn else before[name]), name)
        for name, value in actual.items():
            if value.is_floating_point():
                self.assertTrue(torch.isfinite(value.float()).all(), name)
        # Dynamic FP8 activations amplify BF16-vs-FP32 differences that cross
        # E4M3 rounding thresholds; GLM-5.2's development geometry shows the
        # same 20-60% logit drift. Require the schedules to stay correlated.
        cosine = torch.nn.functional.cosine_similarity(actual['logits'], exact['logits'], dim=0)
        self.assertGreater(float(cosine), 0.8)
        self.assertEqual(actual['recurrent_state'].dtype, torch.float32)
        self.assertEqual(actual['conv_state'].dtype, torch.bfloat16)

    def test_sinkhorn_comb_is_doubly_stochastic(self):
        torch.manual_seed(3)
        streams = torch.randn(4, 64).to(torch.bfloat16)
        post, comb, collapsed = glm53.hyper_connection(streams, torch.randn(24, 256)*0.1, torch.randn(24)*0.1,
                                                       torch.ones(3), 20)
        torch.testing.assert_close(comb.sum(0), torch.ones(4), rtol=0, atol=1e-5)
        torch.testing.assert_close(comb.sum(1), torch.ones(4), rtol=0, atol=1e-3)
        self.assertTrue(bool(((post >= 0) & (post <= 2)).all()))
        self.assertEqual(collapsed.dtype, torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
