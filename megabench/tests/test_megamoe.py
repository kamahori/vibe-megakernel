"""MegaMoE routed-expert EP layer: inputs, expert math and the EP all-to-all."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from ..cases import select_cases
from ..harness.distributed import evaluate_distributed
from ..tasks import megamoe
from ..tasks.oracle import fp32_reference
from ..tasks.quantization import (e8m0_exponent, pack_fp8_activation, unpack_fp8_activation,
                                  unpack_mxfp4)
from ..verify_megamoe import development, serial_case


def megamoe_cases():
    return [case for case in select_cases('all') if case.family == 'megamoe_layer']


def tiny(ranks=1):
    return replace(development(megamoe_cases()[0], ranks), ready=True)


def dequantized_x(values):
    return unpack_fp8_activation(values['x'], values['x_scales'], block=megamoe.ACT_BLOCK)


def ceil_to_ue8m0(maximum):
    """DeepGEMM ``per_token_cast_to_fp8`` scale: ceil of clamped amax/448."""
    bits = (maximum.clamp_min(1e-4)/448.0).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF)+(bits & 0x7FFFFF).bool().int()
    return exponent.clamp(1, 254).to(torch.uint8)


class MegaMoETests(unittest.TestCase):
    def test_catalog_geometry(self):
        cases = megamoe_cases()
        self.assertEqual(len(cases), 2)
        for case in cases:
            self.assertEqual((case.gpus, case.tp, case.ep), (8, 1, 8))
            self.assertEqual(case.params['experts'] % case.ep, 0)
            self.assertEqual(case.params['hidden'] % megamoe.ACT_BLOCK, 0)

    def test_e8m0_activation_scales_match_deepgemm_rounding(self):
        generator = torch.Generator().manual_seed(3)
        value = torch.randn((64, 256), generator=generator)*torch.logspace(-6, 3, 64)[:, None]
        # Mantissas at and around 1.75 (448's) and exact powers of two.
        edges = torch.tensor([448., 448.00003, 447.99997, 224., 1e-4, 1e-6, 0., 2.**-22*448, 896.])
        value[:len(edges), 0] = edges
        payload, scales = pack_fp8_activation(value, block=32, scale_format='e8m0')
        maximum = value.unflatten(-1, (-1, 32)).abs().amax(-1)
        self.assertTrue(torch.equal(scales, ceil_to_ue8m0(maximum)))
        self.assertTrue(torch.equal(scales, e8m0_exponent(maximum)))
        restored = unpack_fp8_activation(payload, scales, block=32)
        self.assertTrue(torch.isfinite(restored).all())
        # E4M3 keeps 3 mantissa bits; subnormals step by 2**-9 of the scale.
        scale = torch.exp2(scales.float()-127).repeat_interleave(32, -1)
        self.assertTrue(bool(((restored-value).abs() <= value.abs()*2**-4+scale*2**-10).all()))

    def test_default_fp8_activation_packing_is_unchanged(self):
        value = torch.randn((3, 300), generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
        for power in (False, True):
            payload, scales = pack_fp8_activation(value, power_of_two=power)
            padded = torch.nn.functional.pad(value.float(), (0, 84)).unflatten(-1, (-1, 128))
            want = padded.abs().amax(-1).clamp_min(1e-4)/448.0
            want = torch.exp2(torch.ceil(torch.log2(want))) if power else want
            self.assertTrue(torch.equal(scales, want))
            expected = (padded/want[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2)[..., :300]
            self.assertTrue(torch.equal(payload.view(torch.uint8), expected.view(torch.uint8)))

    def test_inputs_layout_and_routing(self):
        case = tiny()
        values = megamoe.make_inputs(case, 7, 'cpu', rank=0)
        p = case.params
        self.assertEqual(values['x'].dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(values['x_scales'].shape), (p['tokens'], p['hidden']//32))
        self.assertEqual(values['x_scales'].dtype, torch.uint8)
        self.assertEqual(values['topk_idx'].dtype, torch.int64)
        self.assertEqual(values['topk_weights'].dtype, torch.float32)
        self.assertEqual(tuple(values['l1_blocks'].shape), (p['experts'], 2*p['intermediate'], p['hidden']//32, 16))
        self.assertEqual(tuple(values['l2_scales'].shape), (p['experts'], p['hidden'], p['intermediate']//32))
        index = values['topk_idx']
        self.assertTrue(all(len(set(row)) == p['topk'] for row in index.tolist()))
        self.assertTrue(bool(((index >= 0) & (index < p['experts'])).all()))
        torch.testing.assert_close(values['topk_weights'].sum(-1), torch.full((p['tokens'],), megamoe.ROUTED_SCALING))
        # Popularity is mildly imbalanced at catalog scale.
        many = megamoe.route(7, 4096, 384, 6, 'cpu', (0, 4096))[0]
        load = torch.bincount(many.flatten(), minlength=384).float()
        self.assertTrue(1.5 < float(load.max()/load.mean()) < 5)

    def test_reference_is_deterministic_input_dependent_and_does_not_mutate(self):
        case = tiny()
        values = megamoe.make_inputs(case, 11, 'cpu', rank=0)
        self.assertTrue(all(torch.equal(values[k].view(torch.uint8) if values[k].dtype == torch.float8_e4m3fn else values[k],
                                        other.view(torch.uint8) if other.dtype == torch.float8_e4m3fn else other)
                            for k, other in megamoe.make_inputs(case, 11, 'cpu', rank=0).items()))
        originals = {key: value.clone() for key, value in values.items()}
        output = megamoe.reference(case, values, serial=True)
        self.assertEqual(set(output), {'y'})
        self.assertEqual(output['y'].dtype, torch.bfloat16)
        self.assertEqual(tuple(output['y'].shape), (case.params['tokens'], case.params['hidden']))
        self.assertTrue(torch.equal(output['y'], megamoe.reference(case, values, serial=True)['y']))
        for key, value in values.items():
            self.assertTrue(torch.equal(value.view(torch.uint8), originals[key].view(torch.uint8)), key)
        changes = {'x_scales':values['x_scales']+1, 'l1_blocks':values['l1_blocks'] ^ 0x11,
                   'l2_scales':values['l2_scales']-1, 'topk_weights':values['topk_weights']*0.5,
                   'topk_idx':(values['topk_idx']+1) % case.params['experts']}
        for key, changed in changes.items():
            moved = megamoe.reference(case, values | {key:changed}, serial=True)['y']
            self.assertFalse(torch.equal(moved, output['y']), key)
        self.assertFalse(torch.equal(output['y'], megamoe.reference(case, megamoe.make_inputs(case, 12, 'cpu', rank=0),
                                                                    serial=True)['y']))

    def test_expert_math_matches_transformers_deepseek_v4_experts(self):
        case = tiny()
        p = case.params
        config = DeepseekV4Config(num_local_experts=p['experts'], n_routed_experts=p['experts'],
                                  hidden_size=p['hidden'], intermediate_size=p['intermediate'],
                                  moe_intermediate_size=p['intermediate'], swiglu_limit=megamoe.SWIGLU_LIMIT,
                                  hidden_act='silu')
        module = DeepseekV4Experts(config)
        for seed in (1, 2):
            values = megamoe.make_inputs(case, seed, 'cpu', rank=0)
            with torch.no_grad():
                module.gate_up_proj.copy_(unpack_mxfp4(values['l1_blocks'], values['l1_scales']))
                module.down_proj.copy_(unpack_mxfp4(values['l2_blocks'], values['l2_scales']))
                expected = module(dequantized_x(values), values['topk_idx'], values['topk_weights'])
            # The clamp must be exercised by the synthetic gain.
            l1 = dequantized_x(values)@module.gate_up_proj[0].T
            self.assertGreater(int((l1.abs() > megamoe.SWIGLU_LIMIT).sum()), 0)
            with patch.object(megamoe, 'requantize', lambda value: value.to(torch.bfloat16)):
                exact = fp32_reference(megamoe, case, values, serial=True)['y']
            torch.testing.assert_close(exact, expected, rtol=1e-5, atol=1e-5*float(expected.abs().max()))
            # The served reference adds BF16 and FP8 intermediate rounding only.
            served = megamoe.reference(case, values, serial=True)['y'].double()
            self.assertLess(float((served-expected).norm()/expected.norm()), 0.05)

    def test_union_of_rank_inputs_is_the_serial_problem(self):
        case = tiny(4)
        single = serial_case(case)
        whole = megamoe.make_inputs(single, 5, 'cpu', rank=0)
        parts = [megamoe.make_inputs(case, 5, 'cpu', rank=rank) for rank in range(4)]
        for name, value in whole.items():
            joined = torch.cat([part[name] for part in parts])
            self.assertTrue(torch.equal(value.view(torch.uint8) if value.dtype == torch.float8_e4m3fn else value,
                                        joined.view(torch.uint8) if joined.dtype == torch.float8_e4m3fn else joined), name)

    def test_expert_parallel_all_to_all_matches_serial(self):
        case = megamoe_cases()[0]
        for ranks in (2, 8):
            with self.subTest(ranks=ranks), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                     f'--nproc-per-node={ranks}', '-m', 'megabench.verify_megamoe',
                     '--case', case.id, '--device', 'cpu', '--trials', '2', '--output', directory],
                    env=os.environ | {'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1'},
                    capture_output=True, text=True, timeout=300)
                self.assertEqual(result.returncode, 0, (result.stdout+result.stderr)[-12000:])
                for rank in range(ranks):
                    report = json.loads((Path(directory)/f'rank-{rank}.json').read_text())
                    self.assertEqual(report['status'], 'pass')
                    self.assertEqual(len(report['trials']), 2)
                    self.assertTrue(all(trial['serial_comparison']['y']['bitwise_equal'] for trial in report['trials']))

    def test_reference_submission_passes_the_distributed_harness(self):
        submission = Path(__file__).parents[1]/'examples/reference_submission.py'
        report = evaluate_distributed(tiny(2), submission, device='cpu', trials=2, warmup=0, reps=1)
        self.assertEqual(report['status'], 'correctness_only', report)


if __name__ == '__main__':
    unittest.main()
