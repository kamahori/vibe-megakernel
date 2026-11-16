"""DeepSeek-V4.1-Flash DSpark iteration against DeepSeek's reference inference code.

The upstream test loads the pinned ``inference/model.py`` and ``engram.py``
(downloaded once into ``MEGABENCH_DSV41_SOURCE`` or ``~/.cache/megabench``)
with small PyTorch stand-ins for its TileLang kernels, prefills a short
prompt, decodes the verified tokens one at a time and drafts with
``forward_spec`` after every committed token, as a serving loop would. The
task reference must reproduce that from the prefilled caches in one 6-row
step.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import types
import unittest
import urllib.request
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..cases import select_cases
from ..harness.correctness import dspark_metrics
from ..tasks import dsv41
from ..tasks.oracle import fp32_reference

CASE_ID = 'dsv41-flash-dspark-step-k5-ctx2k'
SPECS = Path(__file__).resolve().parents[1]/'tasks'/'model_specs.json'
DEV = {'context': 20, 'layers': 8, 'hidden': 128, 'q_heads': 4, 'head_dim': 64, 'rope_dim': 16,
       'q_lora_rank': 64, 'o_lora_rank': 32, 'o_groups': 2, 'window': 8, 'window_layers': 1,
       'c2_layers': 4, 'c2_group': 2, 'c1_index_group': 2, 'index_heads': 4, 'index_dim': 32,
       'index_topk': 4, 'candidate_blocks': 2, 'candidate_block_size': 4, 'experts': 8, 'topk': 2,
       'intermediate': 64, 'vocab': 512, 'engram_first': 1, 'engram_second': 4, 'engram_heads': 2,
       'engram_dim': 32, 'engram_buckets': 100, 'engram_compressed_vocab': 300, 'draft_experts': 4,
       'draft_topk': 2, 'markov_rank': 32, 'noise_token': 500}


def development(**changes):
    """Eight layers: sliding 0, 2:1 at 1-4 (sources 1, 3), 1:1 at 5-7 (source 5), indexers 1/3/5/7,
    Engram before 1 and 4, DSpark reading 5-7. Twenty cached tokens make every top-k sparse and
    leave fewer candidate blocks than blocks."""
    case = select_cases('all', [CASE_ID])[0]
    return replace(case, params=case.params | DEV | changes, gpus=1, tp=1)


def spec():
    return json.loads(SPECS.read_text())['deepseek-ai/DeepSeek-V4.1-Flash']


# --------------------------------------------------------------------------- upstream


def upstream_sources() -> Path:
    files = spec()['source_files']
    root = Path(os.environ.get('MEGABENCH_DSV41_SOURCE',
                               Path.home()/'.cache'/'megabench'/'dsv41-inference'/spec()['revision']))
    root.mkdir(parents=True, exist_ok=True)
    for name in ('inference/model.py', 'inference/engram.py'):
        path = root/Path(name).name
        if not path.exists():
            try:
                with urllib.request.urlopen(files[name]['url'], timeout=60) as response:
                    path.write_bytes(response.read())
            except OSError as error:
                raise unittest.SkipTest(f'upstream {name} unavailable: {error}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != files[name]['sha256']:
            raise AssertionError(f'{path} does not match the pinned sha256')
    return root


FP4 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0))


def _fp4_nearest(value):
    """Nearest E2M1 value, ties to the even code, by exhaustive comparison."""
    grid = FP4[:8]
    distance = (value.abs()[..., None]-grid).abs()
    distance = distance+(torch.arange(8) % 2)*1e-6  # an exact tie prefers the even code
    return torch.copysign(grid[distance.argmin(-1)], value)


def _kernel_module():
    """Plain PyTorch versions of the TileLang kernels ``model.py`` imports."""
    kernel = types.ModuleType('kernel')

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        groups = x.float().unflatten(-1, (-1, block_size))
        scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)*(1/448.0)
        if scale_fmt is not None:
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        quant = (groups/scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        if inplace:
            x.copy_((quant.float()*scale).flatten(-2))
            return x
        return quant.flatten(-2), scale.squeeze(-1).to(scale_dtype)

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        assert inplace
        groups = x.float().unflatten(-1, (-1, block_size))
        amax = groups.abs().amax(-1, keepdim=True)
        if scale_dtype == torch.float8_e4m3fn:
            scale = (amax.clamp_min(6*2.0**-9)/6.0).to(torch.float8_e4m3fn).float()
        else:
            scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6*2.0**-126)/6.0)))
        x.copy_((_fp4_nearest((groups/scale).clamp(-6, 6))*scale).flatten(-2))
        return x

    def blocks(scale, rows, cols):
        return scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)[:rows, :cols]

    def fp8_gemm(a, a_s, b, b_s, scale_dtype, block_size=32):
        a = a.float()*a_s.float().repeat_interleave(block_size, -1)
        return (a @ (b.float()*blocks(b_s, *b.shape)).T).to(torch.bfloat16)

    def fp4_gemm(a, a_s, b, b_s, scale_dtype, act_block_size=32):
        a = a.float()*a_s.float().repeat_interleave(act_block_size, -1)
        codes = b.view(torch.uint8)
        values = torch.stack((FP4[(codes & 15).long()], FP4[(codes >> 4).long()]), -1).flatten(-2)
        return (a @ (values*b_s.float().repeat_interleave(32, -1)).T).to(torch.bfloat16)

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        out = torch.empty_like(q)
        for b in range(q.shape[0]):
            for m in range(q.shape[1]):
                idx = topk_idxs[b, m]
                keys = kv[b, idx[idx >= 0].long()].float()
                scores = q[b, m].float() @ keys.T*softmax_scale
                peak = scores.amax(-1, keepdim=True).clamp_min(-1e30)
                weights = torch.exp(scores-peak)
                total = weights.sum(-1, keepdim=True)+torch.exp(attn_sink[:, None]-peak)
                out[b, m] = ((weights.to(torch.bfloat16).float() @ keys)/total).to(q.dtype)
        return out

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc]*hc_scale[0]+hc_base[:hc])+eps
        post = 2*torch.sigmoid(mixes[..., hc:2*hc]*hc_scale[1]+hc_base[hc:2*hc])
        comb = mixes[..., 2*hc:].unflatten(-1, (hc, hc))*hc_scale[2]+hc_base[2*hc:].view(hc, hc)
        comb = comb.softmax(-1)+eps
        comb = comb/(comb.sum(-2, keepdim=True)+eps)
        for _ in range(sinkhorn_iters-1):
            comb = comb/(comb.sum(-1, keepdim=True)+eps)
            comb = comb/(comb.sum(-2, keepdim=True)+eps)
        return pre, post, comb

    for function in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
        setattr(kernel, function.__name__, function)
    return kernel


def load_upstream(root: Path):
    """Import the pinned model.py with kernel and vision stand-ins, in an isolated namespace."""
    saved = {name: sys.modules.get(name) for name in ('kernel', 'vision', 'image_processor', 'engram')}
    vision = types.ModuleType('vision')
    vision.Aligner = vision.ViT = object
    processor = types.ModuleType('image_processor')
    processor.IMAGE, processor.IMAGE_END, processor.IMAGE_NEW_LINE, processor.IMAGE_START = 0, 1, 2, 3
    sys.modules.update(kernel=_kernel_module(), vision=vision, image_processor=processor)
    try:
        engram_spec = importlib.util.spec_from_file_location('engram', root/'engram.py')
        engram = importlib.util.module_from_spec(engram_spec)
        sys.modules['engram'] = engram
        engram_spec.loader.exec_module(engram)
        model_spec = importlib.util.spec_from_file_location('dsv41_upstream_model', root/'model.py')
        model = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return model, engram


def upstream_args(model, case):
    p, lay = case.params, dsv41.layout(case.params)
    return model.ModelArgs(
        max_batch_size=1, max_seq_len=64, temperature=0, vocab_size=p['vocab'], dim=p['hidden'],
        moe_inter_dim=p['intermediate'], n_layers=p['layers'], n_mtp_layers=p['draft_layers'],
        n_heads=p['q_heads'], n_routed_experts=p['experts'], n_shared_experts=1,
        n_activated_experts=p['topk'], route_scale=dsv41.ROUTE_SCALE, swiglu_limit=dsv41.SWIGLU_LIMIT,
        q_lora_rank=p['q_lora_rank'], head_dim=p['head_dim'], rope_head_dim=p['rope_dim'],
        o_groups=p['o_groups'], o_lora_rank=p['o_lora_rank'], window_size=p['window'],
        compress_ratios=tuple(lay.ratios)+(0,)*p['draft_layers'], kv_source_layers=tuple(lay.kv_sources),
        index_source_layers=tuple(lay.index_sources), compress_rope_theta=dsv41.COMPRESS_ROPE_THETA,
        original_seq_len=dsv41.YARN_ORIGINAL, rope_factor=dsv41.YARN_FACTOR, index_n_heads=p['index_heads'],
        index_head_dim=p['index_dim'], index_topk=p['index_topk'], candidate_source_layer=lay.candidate_source,
        candidate_topk_blocks=p['candidate_blocks'], candidate_block_size=p['candidate_block_size'],
        hc_mult=p['hc_mult'], hc_sinkhorn_iters=p['hc_sinkhorn_iters'], engram_layer_ids=lay.engram,
        engram_num_embeddings=dsv41.engram_rows(p), engram_max_ngram_size=p['engram_ngram'],
        engram_vocab_size=p['engram_buckets'], engram_n_heads=p['engram_heads'],
        engram_head_dim=p['engram_dim'], engram_compressed_vocab_size=p['engram_compressed_vocab'],
        dspark_block_size=p['draft_depth'], dspark_noise_token_id=p['noise_token'],
        dspark_target_layer_ids=lay.targets, dspark_markov_rank=p['markov_rank'],
        dspark_n_routed_experts=p['draft_experts'], dspark_n_activated_experts=p['draft_topk'])


def upstream_state(case, values) -> dict[str, torch.Tensor]:
    """Upstream parameter names for every task weight (TP1, converted-checkpoint dtypes)."""
    p, lay = case.params, dsv41.layout(case.params)
    e8m0 = lambda tensor: tensor.view(torch.float8_e8m0fnu)
    fp4 = lambda tensor: tensor.view(torch.float4_e2m1fn_x2)
    state = {'embed.weight': values['embed'], 'head.weight': values['head'].float(), 'norm.weight': values['fnorm']}

    def block(prefix, stack, index, experts):
        get = lambda name: values[stack+name][index]
        for name in ('attn_norm', 'ffn_norm'):
            state[f'{prefix}{name}.weight'] = get(name)
        for site in ('hc_attn', 'hc_ffn'):
            for part in ('fn', 'base', 'scale'):
                state[f'{prefix}{site}_{part}'] = get(f'{site}_{part}')
        state[prefix+'attn.attn_sink'] = get('attn_sink')
        for name in ('wq_a', 'wq_b', 'wkv', 'wo_b'):
            state[f'{prefix}attn.{name}.weight'] = get(name)
            state[f'{prefix}attn.{name}.scale'] = e8m0(get(name+'_scale'))
        state[prefix+'attn.wo_a.weight'] = dsv41.fp8_weight(get('wo_a'), get('wo_a_scale')).to(torch.bfloat16)
        state[prefix+'attn.q_norm.weight'] = get('q_norm')
        state[prefix+'attn.kv_norm.weight'] = get('kv_norm')
        state[prefix+'ffn.gate.weight'] = get('router')
        state[prefix+'ffn.gate.bias'] = get('router_bias')
        for name in ('w1', 'w2', 'w3'):
            for expert in range(experts):
                state[f'{prefix}ffn.experts.{expert}.{name}.weight'] = fp4(get(name)[expert])
                state[f'{prefix}ffn.experts.{expert}.{name}.scale'] = e8m0(get(name+'_scale')[expert])
            state[f'{prefix}ffn.shared_experts.{name}.weight'] = get('shared_'+name)
            state[f'{prefix}ffn.shared_experts.{name}.scale'] = e8m0(get('shared_'+name+'_scale'))

    for layer in range(p['layers']):
        prefix = f'layers.{layer}.'
        block(prefix, '', layer, p['experts'])
        source = f'l{layer}_'
        if layer in lay.kv_sources:
            ratio = lay.ratios[layer]
            state[prefix+'attn.compressor.wkv.weight'] = values[source+'comp_wkv'].float() if ratio > 1 else values[source+'comp_wkv']
            if ratio > 1:
                state[prefix+'attn.compressor.wgate.weight'] = values[source+'comp_wgate'].float()
            state[prefix+'attn.compressor.norm.weight'] = values[source+'comp_norm']
            state[prefix+'attn.indexer.wk.weight'] = values[source+'idx_wk']
            state[prefix+'attn.indexer.k_norm.weight'] = values[source+'idx_knorm']
        if layer in lay.index_sources:
            state[prefix+'attn.indexer.wq_b.weight'] = values[source+'idx_wq_b']
            state[prefix+'attn.indexer.wq_b.scale'] = e8m0(values[source+'idx_wq_b_scale'])
            state[prefix+'attn.indexer.weights_proj.weight'] = values[source+'idx_weights']
        if layer in lay.engram:
            engram = f'l{layer}_engram_'
            state[prefix+'engram.embed.weight'] = values[engram+'table']
            state[prefix+'engram.embed.scale'] = e8m0(values[engram+'table_scale'])
            state[prefix+'engram.wkv.weight'] = values[engram+'wkv']
            state[prefix+'engram.wkv.scale'] = e8m0(values[engram+'wkv_scale'])
            state[prefix+'engram.q_weight'] = values[engram+'q']
            state[prefix+'engram.k_weight'] = values[engram+'k']
    for stage in range(p['draft_layers']):
        block(f'mtp.{stage}.', 'draft_', stage, p['draft_experts'])
    state['mtp.0.main_proj.weight'] = values['draft_main_proj']
    state['mtp.0.main_proj.scale'] = e8m0(values['draft_main_proj_scale'])
    state['mtp.0.main_norm.weight'] = values['draft_main_norm']
    last = f"mtp.{p['draft_layers']-1}."
    state[last+'norm.weight'] = values['draft_norm']
    state[last+'markov_head.embed.weight'] = values['markov_embed']
    state[last+'markov_head.head.weight'] = values['markov_head'].float()
    state[last+'confidence_head.proj.weight'] = values['confidence'][None].float()
    return state


class UpstreamDecode:
    """The pinned model with task weights, prefilled with ``context`` prompt tokens."""

    def __init__(self, case, values, prompt):
        root = upstream_sources()
        self.model_module, engram = load_upstream(root)
        p = case.params
        token_map = values['engram_token_map'].tolist()
        original = engram.build_compressed_token_map
        engram.build_compressed_token_map = lambda tokenizer: (token_map, p['engram_compressed_vocab'])
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            self.model = self.model_module.Transformer(upstream_args(self.model_module, case), tokenizer=object())
        finally:
            torch.set_default_dtype(previous)
            engram.build_compressed_token_map = original
        parameters = dict(self.model.named_parameters())
        state = upstream_state(case, values)
        missing = sorted(set(parameters)-set(state))
        if missing:
            raise AssertionError(f'task inputs lack upstream weights: {missing[:8]}')
        with torch.no_grad():
            for name, tensor in state.items():
                parameter = parameters[name]
                if parameter.shape != tensor.shape or parameter.dtype != tensor.dtype:
                    raise AssertionError(f'{name}: upstream {tuple(parameter.shape)} {parameter.dtype}, '
                                         f'task {tuple(tensor.shape)} {tensor.dtype}')
                parameter.data = tensor.clone()
        self.case = case
        with torch.inference_mode():
            ids, _, hidden = self.model(prompt[None], 0)
            self.model.forward_spec(ids, hidden, 0)
        self.snapshot = {name: buffer.clone() for name, buffer in self.model.named_buffers()}

    def restore(self):
        buffers = dict(self.model.named_buffers())
        for name, value in self.snapshot.items():
            buffers[name].copy_(value)

    def task_state(self) -> dict[str, torch.Tensor]:
        """The prefilled caches in task input layout."""
        p, lay, model = self.case.params, dsv41.layout(self.case.params), self.model
        state = {'window_kv': torch.stack([layer.attn.window_kv_cache[0].clone() for layer in model.layers]),
                 'draft_window_kv': torch.stack([stage.attn.window_kv_cache[0].clone() for stage in model.mtp])}
        for source in lay.kv_sources:
            rows = p['context']//lay.ratios[source]
            attention = model.layers[source].attn
            state[f'l{source}_compress_kv'] = attention.compress_kv_cache[0, :rows].clone()
            state[f'l{source}_index_k'] = attention.indexer.k_cache[0, :rows].clone()
        return state

    def iterate(self, tokens):
        """Sequential decode of the verified tokens, then forward_spec per committed token."""
        self.restore()
        context = self.case.params['context']
        logits, greedy, hiddens = [], [], []
        with torch.inference_mode():
            for row, token in enumerate(tokens.tolist()):
                ids, row_logits, hidden = self.model(torch.tensor([[token]]), context+row)
                logits.append(row_logits[0])
                greedy.append(int(ids[0]))
                hiddens.append(hidden)
            accepted = dsv41.accept(tokens, torch.tensor(greedy))
            for row in range(accepted+1):
                drafted = self.model.forward_spec(torch.tensor([greedy[row]]), hiddens[row], context+row)
        windows = torch.stack([layer.attn.window_kv_cache[0].clone() for layer in self.model.layers])
        output_ids, draft_logits, confidence = drafted
        return {'logits': torch.stack(logits), 'verify_tokens': torch.tensor(greedy), 'accepted': accepted,
                'draft_tokens': output_ids[0, 1:], 'draft_logits': draft_logits[0], 'confidence': confidence[0],
                'window_ring': windows}


def relative(got, want):
    return float((got.double()-want.double()).norm()/want.double().norm().clamp_min(1e-30))


class Dsv41ReferenceTests(unittest.TestCase):
    def test_full_layout_matches_pinned_config(self):
        case = select_cases('all', [CASE_ID])[0]
        config = spec()['config']['text_config']
        p, lay = case.params, dsv41.layout(case.params)
        self.assertEqual(lay.ratios+[0]*p['draft_layers'], config['compress_ratios'])
        self.assertEqual(lay.kv_sources, config['kv_source_layer_ids'])
        self.assertEqual(lay.index_sources, config['index_source_layer_ids'])
        self.assertEqual(lay.candidate_source, config['candidate_source_layer_id'])
        self.assertEqual(list(lay.engram), config['engram_layer_ids'])
        self.assertEqual(list(lay.targets), config['dspark_target_layer_ids'])
        self.assertEqual(list(dsv41.engram_rows(p)), config['engram_num_embeddings'])
        for key, name in (('vocab_size', 'vocab'), ('hidden_size', 'hidden'), ('num_hidden_layers', 'layers'),
                          ('num_attention_heads', 'q_heads'), ('head_dim', 'head_dim'),
                          ('qk_rope_head_dim', 'rope_dim'), ('q_lora_rank', 'q_lora_rank'),
                          ('o_lora_rank', 'o_lora_rank'), ('o_groups', 'o_groups'), ('sliding_window', 'window'),
                          ('index_n_heads', 'index_heads'), ('index_head_dim', 'index_dim'),
                          ('index_topk', 'index_topk'), ('candidate_topk_blocks', 'candidate_blocks'),
                          ('candidate_block_size', 'candidate_block_size'), ('hc_mult', 'hc_mult'),
                          ('hc_sinkhorn_iters', 'hc_sinkhorn_iters'), ('n_routed_experts', 'experts'),
                          ('num_experts_per_tok', 'topk'), ('moe_intermediate_size', 'intermediate'),
                          ('engram_max_ngram_size', 'engram_ngram'), ('engram_n_heads', 'engram_heads'),
                          ('engram_head_dim', 'engram_dim'), ('engram_vocab_size', 'engram_buckets'),
                          ('engram_compressed_vocab_size', 'engram_compressed_vocab'),
                          ('num_nextn_predict_layers', 'draft_layers'), ('dspark_block_size', 'draft_depth'),
                          ('dspark_noise_token_id', 'noise_token'), ('dspark_markov_rank', 'markov_rank'),
                          ('dspark_n_routed_experts', 'draft_experts'),
                          ('dspark_num_experts_per_tok', 'draft_topk')):
            self.assertEqual(config[key], p[name], key)
        self.assertEqual(config['routed_scaling_factor'], dsv41.ROUTE_SCALE)
        self.assertEqual(config['swiglu_limit'], dsv41.SWIGLU_LIMIT)
        self.assertEqual(config['rms_norm_eps'], dsv41.NORM_EPS)
        self.assertEqual(config['compress_rope_theta'], dsv41.COMPRESS_ROPE_THETA)
        self.assertEqual(config['rope_scaling']['factor'], dsv41.YARN_FACTOR)
        self.assertEqual(config['rope_scaling']['original_max_position_embeddings'], dsv41.YARN_ORIGINAL)
        self.assertEqual(spec()['sample_native_tensors']['layers.2.ffn.experts.0.w1.weight']['shape'],
                         [p['intermediate'], p['hidden']//2])

    def test_quantizers_follow_kernel_rules(self):
        values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 7.0, 0.1])
        self.assertEqual(dsv41.fp4_round(values.clamp(-6, 6)).tolist(), [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 6.0, 0.0])
        self.assertTrue(torch.equal(_fp4_nearest(values.clamp(-6, 6)), dsv41.fp4_round(values.clamp(-6, 6))))
        x = torch.randn(4, 64).to(torch.bfloat16)
        kernel = _kernel_module()
        for ours, theirs in ((dsv41.fp8_qdq, lambda v: kernel.act_quant(v.clone(), 32, 'ue8m0', None, True)),
                             (dsv41.fp4_qdq, lambda v: kernel.fp4_act_quant(v.clone(), 32, True)),
                             (dsv41.kv_fp4_qdq, lambda v: kernel.fp4_act_quant(v.clone(), 16, True, torch.float8_e4m3fn))):
            self.assertTrue(torch.equal(ours(x), theirs(x)))

    def test_iteration_matches_upstream_sequential_decode(self):
        case = development()
        p = case.params
        for seed, prefix in ((3, 0), (4, 2), (5, 5)):
            with self.subTest(seed=seed, accepted=prefix):
                values = dsv41.make_inputs(case, seed, 'cpu', rank=0)
                generator = torch.Generator().manual_seed(seed)
                prompt = torch.randint(0, p['vocab'], (p['context'],), generator=generator)
                upstream = UpstreamDecode(case, values, prompt)
                values.update(upstream.task_state())
                values['history_tokens'] = prompt[-3:].clone()
                # Proposals: the upstream greedy continuation for ``prefix`` tokens, then a miss.
                tokens = values['tokens'].clone()
                tokens[0] = int(torch.randint(0, p['vocab'], (1,), generator=generator))
                for row in range(prefix):
                    tokens[row+1] = int(upstream.iterate(tokens)['verify_tokens'][row])
                if prefix < p['draft_depth']:
                    expected = int(upstream.iterate(tokens)['verify_tokens'][prefix])
                    tokens[prefix+1] = (expected+1) % p['vocab']
                values['tokens'] = tokens
                want = upstream.iterate(tokens)
                got = dsv41.reference(case, values, serial=True)
                exact = fp32_reference(dsv41, case, values, serial=True)
                self.assertEqual(want['accepted'], prefix)
                self.assertEqual(int(got['accepted_count']), prefix)
                self.assertTrue(torch.equal(got['verify_tokens'], want['verify_tokens']))
                self.assertTrue(torch.equal(got['draft_tokens'], want['draft_tokens']))
                slots = (p['context']+torch.arange(p['draft_depth']+1)) % p['window']
                # Upstream decodes one row per GEMM; the reference batches six rows, so FP32
                # accumulation order can flip a BF16 ulp that FP8 re-quantization amplifies.
                # Each output must stay within the gate's band: twice the reference's own error
                # against the FP32 oracle.
                for name, theirs in (('logits', want['logits']), ('draft_logits', want['draft_logits']),
                                     ('draft_confidence', want['confidence']),
                                     ('window_kv_write', want['window_ring'][:, slots])):
                    self.assertLessEqual(relative(got[name], theirs), 2*relative(got[name], exact[name])+0.01, name)
                committed = torch.full((p['draft_depth']+1,), -1)
                committed[:prefix] = tokens[1:prefix+1]
                committed[prefix] = want['verify_tokens'][prefix]
                self.assertTrue(torch.equal(got['committed_tokens'], committed))

    def test_outputs_scenarios_and_oracle(self):
        case = development()
        p = case.params
        verify, depth = p['draft_depth']+1, p['draft_depth']
        lay = dsv41.layout(p)
        for trial, accepted in enumerate((0, depth, 2)):
            values = dsv41.make_inputs(case, 11, 'cpu', rank=0)
            scenario = dsv41.set_trial_scenario(case, values, trial)
            self.assertEqual(scenario, dsv41.SCENARIOS[trial])
            out = dsv41.reference(case, values, serial=True)
            self.assertEqual(int(out['accepted_count']), accepted)
            self.assertEqual(out['committed_tokens'][:accepted+1].ge(0).all().item(), True)
            self.assertTrue(out['committed_tokens'][accepted+1:].eq(-1).all())
        shapes = {'logits': (verify, p['vocab']), 'draft_logits': (depth, p['vocab']),
                  'window_kv_write': (p['layers'], verify, p['head_dim']),
                  'compress_kv_write_r2': (len(lay.c2_sources), verify//2, p['head_dim']),
                  'compress_kv_write_r1': (1, verify, p['head_dim']),
                  'index_k_write_r1': (1, verify, p['index_dim']),
                  'draft_window_kv_write': (p['draft_layers'], verify, p['head_dim']),
                  'sparse_indices': (len(lay.index_sources), verify, p['index_topk'])}
        for name, shape in shapes.items():
            self.assertEqual(tuple(out[name].shape), shape, name)
        self.assertEqual(out['window_kv_write'].dtype, torch.bfloat16)
        exact = fp32_reference(dsv41, case, values, serial=True)
        self.assertEqual(set(exact), set(out))
        self.assertLess(relative(exact['window_kv_write'][0], out['window_kv_write'][0]), 0.1)

    def test_forcing_and_gate(self):
        case = development()
        values = dsv41.make_inputs(case, 21, 'cpu', rank=0)
        out = dsv41.reference(case, values, serial=True)
        forced = dsv41.reference(case, values, serial=True,
                                 forced={'verify_tokens': out['verify_tokens'], 'draft_tokens': out['draft_tokens']})
        for name in out:
            self.assertTrue(torch.equal(out[name], forced[name]), name)
        exact = fp32_reference(dsv41, case, values, serial=True)
        details = dspark_metrics(out, {k: v.clone() for k, v in out.items()}, case, 'cpu', exact)
        self.assertFalse([k for k, v in details.items() if v.get('pass') is False])
        broken = {k: v.clone() for k, v in out.items()}
        broken['window_kv_write'][3] = 0
        broken['accepted_count'] += 1
        broken['draft_tokens'] = (broken['draft_tokens']+7) % case.params['vocab']
        oracle = lambda c, v, **options: fp32_reference(dsv41, c, v, **options)
        want, oracle_out = dsv41.regrade(case, values, out, exact, broken, oracle, serial=True)
        self.assertIsNot(want, out)
        details = dspark_metrics(want, broken, case, 'cpu', oracle_out)
        failed = {k for k, v in details.items() if v.get('pass') is False}
        self.assertTrue({'window_kv_write[3]', 'accepted_count', 'draft_logits'} <= failed, failed)


def _tp_worker(rank: int, root: str, results: dict) -> None:
    dist.init_process_group('gloo', init_method=f'file://{root}/store', rank=rank, world_size=2)
    try:
        case = replace(development(), gpus=2, tp=2)
        values = dsv41.make_inputs(case, 31, 'cpu', rank=rank)
        out = dsv41.reference(case, values)
        shards = [torch.empty_like(out['logits']) for _ in range(2)]
        dist.all_gather(shards, out['logits'])
        draft = [torch.empty_like(out['draft_logits']) for _ in range(2)]
        dist.all_gather(draft, out['draft_logits'])
        if rank == 0:
            torch.save({**out, 'logits': torch.cat(shards, -1), 'draft_logits': torch.cat(draft, -1)},
                       f'{root}/tp.pt')
    finally:
        dist.destroy_process_group()


class Dsv41ParallelTests(unittest.TestCase):
    def test_tp2_matches_serial_run(self):
        """Rank-local shards (heads, groups, vocabulary, experts, Engram rows) reproduce the TP1 run."""
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(_tp_worker, args=(root, {}), nprocs=2)
            tp = torch.load(f'{root}/tp.pt')
        case = development()
        values = dsv41.make_inputs(case, 31, 'cpu', rank=0)
        serial = dsv41.reference(case, values, serial=True)
        exact = fp32_reference(dsv41, case, values, serial=True)
        # Partial projections round to BF16 before the cross-rank sum, so TP is not bitwise
        # serial; the TP run must pass the case's gate against the serial reference instead.
        self.assertTrue(torch.equal(tp['window_kv_write'][0], serial['window_kv_write'][0]))
        oracle = lambda c, v, **options: fp32_reference(dsv41, c, v, **options)
        want, oracle_out = dsv41.regrade(case, values, serial, exact, tp, oracle, serial=True)
        details = dspark_metrics(want, tp, case, 'cpu', oracle_out)
        self.assertFalse({k: v for k, v in details.items() if v.get('pass') is False})


if __name__ == '__main__':
    unittest.main()
