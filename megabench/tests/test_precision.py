"""Serving precision and independent BF16 decoder regression checks."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from ..harness.correctness import _compare
from ..tasks.common import attention, rms
from ..workloads import make_inputs, oracle, reference
from .test_non_p0 import serving_model, tiny_gptoss, tiny_hybrid, tiny_speculative, tiny_vision
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


def sdpa_attention(query, key, value, *, scale=None, allowed=None):
    """An alternative BF16 attention schedule, as a fused kernel might use."""
    single = query.ndim == 2
    query = query.unsqueeze(0) if single else query
    groups = query.shape[-2] // key.shape[-2]
    key, value = (tensor.repeat_interleave(groups, dim=-2) for tensor in (key, value))
    result = F.scaled_dot_product_attention(
        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
        attn_mask=allowed, scale=scale).transpose(0, 1)
    return result.squeeze(0) if single else result


class OracleBandTests(unittest.TestCase):
    def test_oracle_widens_bf16_compute_and_preserves_inputs(self):
        for case in (*TINY_P0, tiny_gptoss(), tiny_hybrid(), tiny_vision(), tiny_speculative()):
            with self.subTest(case=case.id):
                values = make_inputs(case, 23)
                before = {key: value.clone() for key, value in values.items()}
                trace = MatrixPrecision()
                with trace:
                    exact = oracle(case, values)
                with torch.inference_mode():
                    again = oracle(case, values)
                self.assertTrue(trace.dtypes)
                self.assertNotIn(torch.bfloat16, trace.dtypes)
                expected = reference(case, values)
                for key, value in exact.items():
                    self.assertEqual(value.shape, expected[key].shape)
                    self.assertNotEqual(value.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(value, again[key]))
                self.assertTrue(all(torch.equal(value, before[key])
                                    for key, value in values.items()))

    def test_band_accepts_alternative_bf16_schedule(self):
        for seed in (3, 5, 7):
            with self.subTest(seed=seed), torch.no_grad():
                values = make_inputs(TINY_DENSE, seed)
                expected, exact = reference(TINY_DENSE, values), oracle(TINY_DENSE, values)
                with patch('megabench.tasks.references.qwen3.attention', sdpa_attention):
                    actual = reference(TINY_DENSE, values)
                self.assertFalse(torch.equal(actual['logits'], expected['logits']))
                with self.assertRaises(AssertionError):
                    _compare(expected, actual, TINY_DENSE, 'cpu')
                details = _compare(expected, actual, TINY_DENSE, 'cpu', exact)
                self.assertTrue(all(detail['pass'] for detail in details.values()))
                _compare(expected, exact | {'next_token': expected['next_token'],
                                            'k_write': exact['k_write'].bfloat16(),
                                            'v_write': exact['v_write'].bfloat16()},
                         TINY_DENSE, 'cpu', exact)

    def test_band_rejects_errors_beyond_bf16_noise(self):
        values = make_inputs(TINY_DENSE, 3)
        expected, exact = reference(TINY_DENSE, values), oracle(TINY_DENSE, values)
        logits = expected['logits']
        spike = logits.clone()
        spike[0] += 1.0
        writes = expected['k_write'].clone()
        writes[-1] = 0
        order = exact['logits'].argsort()
        corruptions = {
            'scaled logits': {'logits': logits * 1.05},
            'one corrupted logit': {'logits': spike},
            'one missing layer write': {'k_write': writes},
            'far from greedy': {'next_token': order[0]},
            'non-finite': {'logits': logits * torch.nan},
        }
        for label, change in corruptions.items():
            with self.subTest(label), self.assertRaises(AssertionError):
                _compare(expected, expected | change, TINY_DENSE, 'cpu', exact)

    def test_integer_outputs_follow_reference_or_oracle(self):
        case = TINY_DENSE
        logits = torch.tensor([0.0, 1.0, 0.98, -3.0])
        exact = {'logits': logits, 'next_token': torch.tensor(1)}
        expected = {'logits': logits + torch.tensor([0.0, 0.0, 0.05, 0.0]),
                    'next_token': torch.tensor(2)}
        for token in (1, 2):
            _compare(expected, expected | {'next_token': torch.tensor(token)}, case, 'cpu', exact)
        with self.assertRaises(AssertionError):
            _compare(expected, expected | {'next_token': torch.tensor(0)}, case, 'cpu', exact)
        routes = torch.tensor([[0, 1], [2, 3], [1, 0]])
        exact, expected = {'expert_ids': routes}, {'expert_ids': routes.clone()}
        expected['expert_ids'][0, 1] = 2
        changed = routes.clone()
        changed[0, 1] = 2
        changed[2, 1] = 0
        _compare(expected, {'expert_ids': changed}, case, 'cpu', exact)
        changed[1, 1] = 0
        with self.assertRaises(AssertionError):
            _compare(expected, {'expert_ids': changed}, case, 'cpu', exact)


if __name__ == '__main__':
    unittest.main()
