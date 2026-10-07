"""Serving precision and independent BF16 decoder regression checks."""

import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ..tasks.common import attention, rms
from ..workloads import make_inputs, reference
from .test_non_p0 import serving_model
from .test_suite import TINY_DENSE, TINY_P0


class MatrixPrecision(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.dtypes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.mm.default, torch.ops.aten.mv.default,
                    torch.ops.aten.bmm.default, torch.ops.aten.addmm.default):
            self.dtypes.append(result.dtype)
        return result


class ReferencePrecisionTests(unittest.TestCase):
    def test_p0_uses_bf16_matrix_outputs_and_preserves_packed_inputs(self):
        for case in TINY_P0:
            with self.subTest(case=case.id):
                values = make_inputs(case, 17)
                before = {key: value.clone() for key, value in values.items()}
                trace = MatrixPrecision()
                with trace, torch.no_grad():
                    actual = reference(case, values)
                self.assertTrue(trace.dtypes)
                # Qwen3 MoE explicitly retains one FP32 router per layer.
                self.assertEqual(trace.dtypes.count(torch.float32),
                                 case.params['layers'] if case.family == 'moe_step' else 0)
                self.assertTrue(all(dtype in (torch.bfloat16, torch.float32)
                                    for dtype in trace.dtypes))
                self.assertEqual(actual['logits'].dtype, torch.float32)
                self.assertTrue(torch.equal(actual['logits'], actual['logits'].bfloat16().float()))
                for key in ('k_write', 'v_write'):
                    self.assertEqual(actual[key].dtype, torch.bfloat16)
                self.assertTrue(all(torch.equal(value, before[key])
                                    for key, value in values.items()))

    def test_sensitive_reductions_return_to_activation_dtype(self):
        x = torch.tensor([1., 2., 16., 256.], dtype=torch.bfloat16)
        weight = torch.ones_like(x)
        expected = (x.float() * (x.float().square().mean() + 1e-6).rsqrt()).bfloat16()
        self.assertTrue(torch.equal(rms(x, weight), expected))
        q = torch.randn(2, 4).bfloat16()
        k, v = torch.randn(3, 1, 4).bfloat16(), torch.randn(3, 1, 4).bfloat16()
        self.assertEqual(attention(q, k, v).dtype, torch.bfloat16)

    def test_dense_matches_upstream_bf16_qwen3(self):
        from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

        case, p = TINY_DENSE, TINY_DENSE.params
        config = Qwen3Config(hidden_size=p['hidden'], num_hidden_layers=p['layers'],
                            num_attention_heads=p['q_heads'], num_key_value_heads=p['kv_heads'],
                            head_dim=p['head_dim'], intermediate_size=p['intermediate'],
                            vocab_size=p['vocab'], rope_theta=1e6, tie_word_embeddings=True)
        config._attn_implementation = 'eager'
        model = serving_model(Qwen3ForCausalLM(config))
        for seed in (17, 19, 104729):
            with self.subTest(seed=seed), torch.no_grad():
                values = make_inputs(case, seed)
                model.model.embed_tokens.weight.copy_(values['embed'])
                model.lm_head.weight.copy_(values['embed'])
                model.model.norm.weight.copy_(values['fnorm'])
                cache = DynamicCache()
                for index, layer in enumerate(model.model.layers):
                    layer.input_layernorm.weight.copy_(values['ln1'][index])
                    layer.post_attention_layernorm.weight.copy_(values['ln2'][index])
                    for name in ('q', 'k', 'v', 'o'):
                        getattr(layer.self_attn, name + '_proj').weight.copy_(values['w' + name][index])
                    layer.self_attn.q_norm.weight.copy_(values['qn'][index])
                    layer.self_attn.k_norm.weight.copy_(values['kn'][index])
                    for name, key in (('gate', 'wg'), ('up', 'wu'), ('down', 'wd')):
                        getattr(layer.mlp, name + '_proj').weight.copy_(values[key][index])
                    cache.update(values['kcache'][index].permute(1, 0, 2)[None],
                                 values['vcache'][index].permute(1, 0, 2)[None], index)
                actual = model(input_ids=values['token'].reshape(1, 1),
                               past_key_values=cache, use_cache=True)
                expected = reference(case, values)
                torch.testing.assert_close(expected['logits'], actual.logits[0, 0].float(), rtol=0, atol=0)
                for index in range(p['layers']):
                    torch.testing.assert_close(expected['k_write'][index], cache.layers[index].keys[0, :, -1], rtol=0, atol=0)
                    torch.testing.assert_close(expected['v_write'][index], cache.layers[index].values[0, :, -1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
