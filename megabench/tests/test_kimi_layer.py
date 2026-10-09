"""Kimi K3 single-layer DP/EP cases against the whole-model Kimi reference."""

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

from ..cases import select_cases
from ..harness.distributed import evaluate_distributed
from ..tasks import kimi, kimi_layer
from ..verify_kimi_layer import development, serial_case
from .test_kimi import development as kimi_development


def layer_cases():
    return [case for case in select_cases('p3') if case.family == 'kimi_k3_layer']


def model_slice(case, seed):
    """Run the whole dev model; capture one layer's inputs and outputs."""
    model = replace(kimi_development(), gpus=1, tp=1)
    model = replace(model, params=model.params | {key:case.params[key] for key in case.params
                                                  if key in model.params and key != 'batch'})
    values = kimi.make_inputs(model, seed, 'cpu', rank=0)
    calls = []
    original = kimi.attention_residual

    def capture(prefix, blocks, norm, projection):
        calls.append((prefix, list(blocks)))
        return original(prefix, blocks, norm, projection)
    with patch.object(kimi, 'attention_residual', capture):
        result = kimi.reference(model, values, serial=True)
    layer, name = case.params['layer'], f'l{case.params["layer"]}_'
    prefix, blocks = calls[2*layer-1]
    row = {key[len(name):]:value for key, value in values.items() if key.startswith(name)}
    row |= {key:values[key][layer] for key in ('ln1', 'ln2', 'attention_res_norm', 'mlp_res_norm',
                                                'attention_res_proj', 'mlp_res_proj')}
    row |= {'blocks':torch.stack(blocks)[:, None], 'prefix':prefix[None]}
    for key in ('kv_cache', 'pe_cache', 'conv_state', 'recurrent_state'):
        if key in row:
            row[key] = row[key][None]
    kinds = [index for index in range(model.params['layers'])
             if kimi.full_layer(model, index) == kimi.full_layer(model, layer)]
    writes = ('kv_write', 'pe_write') if kimi.full_layer(model, layer) else ('recurrent_state', 'conv_state')
    expected = {'prefix':calls[2*layer+1][0][None],
                'expert_ids':result['expert_ids'][layer].sort().values[None]}
    expected |= {key:result[key][kinds.index(layer)][None] for key in writes}
    return row, expected


class KimiLayerTests(unittest.TestCase):
    def test_catalog_layers_cover_both_attention_variants_mid_block(self):
        cases = layer_cases()
        self.assertEqual(sorted(kimi_layer.attention_kind(case) for case in cases), ['kda', 'mla'])
        for case in cases:
            self.assertEqual((case.gpus, case.tp, case.ep), (8, 1, 8))
            self.assertEqual(kimi_layer.residual_blocks(case), 6)
            self.assertTrue(case.ready)

    def test_layer_matches_whole_model_reference_slice(self):
        for original in layer_cases():
            case = replace(development(original), gpus=1, ep=1, params=development(original).params | {'batch':1})
            for seed in (11, 13, 17):  # Kimi resets its caches when seed%3 == 0.
                with self.subTest(case=case.id, seed=seed):
                    values, expected = model_slice(case, seed)
                    actual = kimi_layer.reference(case, values, serial=True)
                    self.assertEqual(set(actual), set(expected))
                    # Both compute each token with the same per-token operations.
                for name in actual:
                    self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_union_of_rank_inputs_is_the_serial_batch(self):
        case = replace(development(layer_cases()[0]), gpus=4, ep=4)
        single = serial_case(case)
        whole = kimi_layer.make_inputs(single, 5, 'cpu', rank=0)
        parts = [kimi_layer.make_inputs(case, 5, 'cpu', rank=rank) for rank in range(4)]
        for name, value in whole.items():
            if name.endswith(('_blocks', '_scales')):
                self.assertTrue(torch.equal(value, torch.cat([part[name] for part in parts])), name)
            elif name in ('blocks',):
                self.assertTrue(torch.equal(value, torch.cat([part[name] for part in parts], 1)), name)
            elif value.shape[0] == single.params['batch'] and name in parts[0] and parts[0][name].shape[0] == 2:
                self.assertTrue(torch.equal(value, torch.cat([part[name] for part in parts])), name)
            else:
                self.assertTrue(all(torch.equal(value, part[name]) for part in parts), name)

    def test_expert_parallel_all_to_all_matches_serial(self):
        for case in layer_cases():
            for ranks in (2, 8):
                with self.subTest(case=case.id, ranks=ranks), tempfile.TemporaryDirectory() as directory:
                    result = subprocess.run(
                        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                         f'--nproc-per-node={ranks}', '-m', 'megabench.verify_kimi_layer',
                         '--case', case.id, '--device', 'cpu', '--trials', '2', '--output', directory],
                        env=os.environ | {'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1'},
                        capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, (result.stdout+result.stderr)[-12000:])
                    for rank in range(ranks):
                        report = json.loads((Path(directory)/f'rank-{rank}.json').read_text())
                        self.assertEqual(report['status'], 'pass')
                        self.assertEqual(len(report['trials']), 2)

    def test_reference_submission_passes_the_distributed_harness(self):
        submission = Path(__file__).parents[1]/'examples/reference_submission.py'
        for original in layer_cases():
            with self.subTest(case=original.id):
                case = replace(development(original), gpus=2, ep=2)
                report = evaluate_distributed(case, submission, device='cpu', trials=2, warmup=0, reps=1)
                self.assertEqual(report['status'], 'correctness_only', report)


if __name__ == '__main__':
    unittest.main()
