"""Compare the MegaMoE reference with pinned DeepGEMM and Transformers code on CPU.

``--src`` is a local DeepGEMM checkout at the pinned commit. Only the pure
torch casts in ``deep_gemm/utils/math.py`` are loaded (``load_nodes``, after
a sha256 check); the fused kernel and the test's DeepEP/TileLang baseline
need SM100 GPUs and are not run. The other pinned DeepGEMM files are
hash-checked as documentation of the transcribed epilogue/combine rules.

Checks on a small geometry:

1. Quantization: ``pack_fp8_activation(block=32, scale_format="e8m0")``
   equals DeepGEMM ``per_token_cast_to_fp8(use_ue8m0=True, gran_k=32)``
   bit for bit on the input rows and on the reference's own SwiGLU outputs;
   ``pack_mxfp4`` decodes to the same values as ``per_token_cast_to_fp4``.
2. Inputs quantized by DeepGEMM's casts run through ``megamoe.reference``.
3. Expert math: with the intermediate FP8 requantization removed, the FP32
   oracle equals the pinned ``DeepseekV4Experts`` on dequantized weights
   (it applies routing weights after linear2; equal in exact arithmetic).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from .cases import select_cases
from .tasks import megamoe
from .tasks.oracle import fp32_reference
from .tasks.quantization import FP4_VALUES, pack_fp8_activation, pack_mxfp4, unpack_mxfp4
from .verify_kimi_primary import load_nodes
from .verify_megamoe import DEVELOPMENT

MODEL = 'deepseek-ai/DeepSeek-V4-Pro'
DEEPGEMM_FILES = {
    'deepgemm/math.py':'deep_gemm/utils/math.py',
    'deepgemm/test_mega_moe.py':'tests/test_mega_moe.py',
    'deepgemm/swiglu_apply_weight_to_fp8.py':'third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py',
    'deepgemm/tilelang_utils.py':'third-party/tilelang_ops/utils.py',
    'deepgemm/mega.py':'deep_gemm/mega/__init__.py',
    'deepgemm/sm100_fp8_fp4_mega_moe.cuh':'deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh',
    'deepgemm/math.cuh':'deep_gemm/include/deep_gemm/common/math.cuh',
}


def pins():
    root = Path(__file__).parent/'tasks'
    return json.loads((root/'model_specs.json').read_text())[MODEL]


def check_hash(path, pin):
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin['sha256']:
        raise ValueError(f'primary source hash differs: {path}')


def e8m0(sf):
    """DeepGEMM FP32 power-of-two scales as biased uint8 exponents."""
    bits = sf.float().contiguous().view(torch.int32)
    if (bits & 0x807FFFFF).any():
        raise AssertionError('scale is not a positive power of two')
    return (bits >> 23).to(torch.uint8)


def fp4_native(namespace, weight):
    """DeepGEMM FP4 cast in the task layout: uint8 [n, k/32, 16] plus E8M0."""
    packed, sf = namespace['per_token_cast_to_fp4'](weight, use_ue8m0=True, gran_k=32)
    return packed.view(torch.uint8).unflatten(-1, (-1, 16)), e8m0(sf)


def decode_fp4(blocks, scales):
    lut = torch.tensor(FP4_VALUES)
    codes = torch.stack((blocks & 15, blocks >> 4), -1).flatten(-2).long()
    return (lut[codes]*torch.exp2(scales.float()-127)[..., None]).flatten(-2)


def same_fp8(namespace, value, label):
    payload, sf = namespace['per_token_cast_to_fp8'](value, use_ue8m0=True, gran_k=32)
    ours, scales = pack_fp8_activation(value, block=32, scale_format='e8m0')
    if not torch.equal(payload.view(torch.uint8), ours.view(torch.uint8)) or not torch.equal(e8m0(sf), scales):
        raise AssertionError(f'{label}: E4M3/E8M0 cast differs from DeepGEMM')
    return payload, sf


def primary_inputs(namespace, case, seed):
    """Synthetic BF16 tensors quantized by DeepGEMM's own casts."""
    p = case.params
    values = megamoe.make_inputs(case, seed, 'cpu', rank=0)
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn((p['tokens'], p['hidden']), generator=generator)*2).to(torch.bfloat16)
    payload, sf = same_fp8(namespace, x, 'x')
    values['x'], values['x_scales'] = payload, e8m0(sf)
    for name, rows, cols, gain in (('l1', 2*p['intermediate'], p['hidden'], megamoe.L1_GAIN),
                                   ('l2', p['hidden'], p['intermediate'], 1)):
        blocks, scales = [], []
        for _ in range(p['experts']):
            weight = (torch.randn((rows, cols), generator=generator)*gain/math.sqrt(cols)).to(torch.bfloat16)
            native = fp4_native(namespace, weight)
            ours = pack_mxfp4(weight)
            if not torch.equal(decode_fp4(*native), unpack_mxfp4(*ours)):
                raise AssertionError(f'{name}: MXFP4 values differ from DeepGEMM per_token_cast_to_fp4')
            blocks.append(native[0])
            scales.append(native[1])
        values[name+'_blocks'], values[name+'_scales'] = torch.stack(blocks), torch.stack(scales)
    return values


def transformers_experts(namespace, case, values):
    p = case.params
    config = type('Config', (), {'num_local_experts':p['experts'], 'hidden_size':p['hidden'],
                                 'intermediate_size':p['intermediate'], 'hidden_act':'silu',
                                 'swiglu_limit':megamoe.limit(case)})()
    module = namespace['DeepseekV4Experts'](config)
    with torch.no_grad():
        module.gate_up_proj.copy_(torch.stack([decode_fp4(values['l1_blocks'][e], values['l1_scales'][e])
                                               for e in range(p['experts'])]))
        module.down_proj.copy_(torch.stack([decode_fp4(values['l2_blocks'][e], values['l2_scales'][e])
                                            for e in range(p['experts'])]))
        x = values['x'].float()*torch.exp2(values['x_scales'].float()-127).repeat_interleave(32, -1)
        return module(x, values['topk_idx'], values['topk_weights'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, type=Path, help='DeepGEMM checkout at the pinned commit')
    parser.add_argument('--transformers-source', type=Path,
                        help='modeling_deepseek_v4.py (default: the installed Transformers copy)')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    spec = pins()['source_files']
    for key, relative in DEEPGEMM_FILES.items():
        check_hash(args.src/relative, spec[key])
    namespace = {'torch':torch, 'Tuple':Tuple}
    load_nodes(args.src/DEEPGEMM_FILES['deepgemm/math.py'],
               ['ceil_div', 'align', 'ceil_to_ue8m0', 'pack_ue8m0_to_int', 'per_token_cast_to_fp8',
                '_quantize_to_fp4_e2m1', 'per_token_cast_to_fp4'], namespace, spec['deepgemm/math.py'])
    source = args.transformers_source
    if source is None:
        import transformers.models.deepseek_v4.modeling_deepseek_v4 as installed
        source = Path(installed.__file__)
    namespace.update({'nn':nn, 'F':F, 'ACT2FN':ACT2FN, 'DeepseekV4Config':object,
                      'use_experts_implementation':lambda cls: cls})
    load_nodes(source, ['DeepseekV4Experts'], namespace, spec['transformers/modeling_deepseek_v4.py'])
    base = select_cases('all', ['megamoe-layer-deepseek-v4-pro-ep8-t512'])[0]
    case = replace(base, ready=True, gpus=1, ep=1, params=base.params | DEVELOPMENT | {'tokens':32})
    report = {'case':case.to_dict(), 'tier':'development_geometry', 'primary_sources':spec, 'trials':[]}
    with torch.inference_mode():
        for seed in (17, 18):
            values = primary_inputs(namespace, case, seed)
            captured = []
            original = megamoe.requantize

            def capture(value):
                captured.append(value.clone())
                return original(value)
            with patch.object(megamoe, 'requantize', capture):
                actual = megamoe.reference(case, values, serial=True)['y']
            for index, value in enumerate(captured):
                same_fp8(namespace, value, f'intermediate {index}')
            expected = transformers_experts(namespace, case, values)
            with patch.object(megamoe, 'requantize', lambda value: value.to(torch.bfloat16)):
                exact = fp32_reference(megamoe, case, values, serial=True)['y']
            torch.testing.assert_close(exact, expected, rtol=1e-5, atol=1e-5*float(expected.abs().max()))
            quantized = fp32_reference(megamoe, case, values, serial=True)['y']
            report['trials'].append({
                'seed':seed, 'requantized_expert_groups':len(captured),
                'unquantized_oracle_max_error':float((exact-expected).abs().max()),
                'bf16_reference_relative_error':float((actual.double()-quantized.double()).norm()/quantized.double().norm()),
                'fp8_intermediate_relative_effect':float((quantized-expected).norm()/expected.norm())})
    report['status'] = 'pass'
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report['trials']))


if __name__ == '__main__':
    main()
