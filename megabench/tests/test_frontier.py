"""Native FP8, YaRN/indexer transforms and frontier routing checks."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from ..cases import select_cases
from ..tasks import frontier
from ..tasks.common import rms
from ..tasks.quantization import (pack_fp8_activation, pack_fp8_weight,
                                  unpack_fp8_activation, unpack_fp8_weight)


def development(case):
    return replace(case, params=case.params | {
        'hidden': 128, 'q_heads': 8, 'q_lora_rank': 128, 'kv_lora_rank': 128,
        'qk_nope_dim': 32, 'qk_rope_dim': 32, 'v_head_dim': 32,
        'intermediate': 256, 'dense_intermediate': 256, 'experts': 4, 'topk': 2,
        'index_heads': 4, 'index_dim': 128, 'index_topk': 3, 'vocab': 512,
        'layers': 8, 'context': 6, 'router_groups': 2 if case.family == 'deepseek_v32_step' else 1,
        'router_top_groups': 1,
    }, atol=0.003, rtol=0.003)


class FrontierMathTests(unittest.TestCase):
    def test_deepseek_sparse_selection_matches_upstream_indexer(self):
        from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer, DeepseekV32RotaryEmbedding
        original = next(case for case in select_cases('p3') if case.family == 'deepseek_v32_step')
        case = development(original)
        case = replace(case,gpus=1,tp=1,params=case.params | {'layers':1,'first_dense':1})
        p = case.params
        rope = {'rope_type':'yarn','rope_theta':10000.,'factor':40.,'original_max_position_embeddings':4096,
                'beta_fast':32,'beta_slow':1,'mscale':1.,'mscale_all_dim':1.}
        config = DeepseekV32Config(hidden_size=p['hidden'],q_lora_rank=p['q_lora_rank'],qk_rope_head_dim=p['qk_rope_dim'],
                                   index_n_heads=p['index_heads'],index_head_dim=p['index_dim'],index_topk=p['index_topk'],rope_parameters=rope)
        # This independently constructs the normalized Hadamard matrix.
        hadamard = torch.ones(1,1)
        while hadamard.shape[0] < p['index_dim']:
            hadamard = torch.cat((torch.cat((hadamard,hadamard),1),torch.cat((hadamard,-hadamard),1)),0)
        for seed in (17,19):
            with self.subTest(case=case.id,seed=seed), torch.no_grad():
                values = frontier.make_inputs(case,seed,'cpu',rank=0)
                values['l0_iw'].abs_()  # Positive scores avoid source-specific zero-score topk ties.
                indexer = DeepseekV32Indexer(config,0).float().eval()
                def quantize(value):
                    payload,scale = pack_fp8_activation(value.to(torch.bfloat16),power_of_two=True)
                    return unpack_fp8_activation(payload,scale)
                for short,long in (('iq','wq_b'),('ik','wk')):
                    module = getattr(indexer,long)
                    module.weight.copy_(unpack_fp8_weight(values['l0_'+short],values['l0_'+short+'_scale']))
                    # TileRT's DeepSeek wq_b consumes the BF16 q residual (W8A16).
                    activation = (lambda value: value.float()) if short == 'iq' else quantize
                    module.register_forward_pre_hook(lambda module,args,activation=activation:(activation(args[0]),))
                    module.register_forward_hook(lambda module,args,out:out.to(torch.bfloat16))
                indexer.k_norm.to(torch.bfloat16)
                indexer.k_norm.weight.copy_(values['l0_inorm'])
                indexer.k_norm.bias.copy_(values['l0_ibias'])
                indexer.weights_proj.weight.copy_(values['l0_iw'])
                # TileRT publishes the head weights as BF16.
                indexer.weights_proj.register_forward_hook(lambda module,args,out:out.to(torch.bfloat16))
                qa = torch.nn.Linear(p['hidden'],p['q_lora_rank'],bias=False)
                qa.weight.copy_(unpack_fp8_weight(values['l0_qa'],values['l0_qa_scale']))
                # TileRT multiplies the FP32 gamma before rounding once.
                hidden = rms(values['embed'][values['token']],values['ln1'][0],eps=1e-6,weight_in_fp32=True)[None,None]
                q_resid = rms(qa(quantize(hidden)).to(torch.bfloat16),values['l0_qn'],eps=1e-6,weight_in_fp32=True)
                def native_coordinates(value):
                    value = value.to(torch.bfloat16)
                    return (value.float()@hadamard/p['index_dim']**0.5).to(torch.bfloat16)
                written = {}
                class NativeCache:
                    def update_indexer(self,key,layer):
                        # TileRT keeps a BF16 Hadamard-domain key cache.
                        written.update(key=native_coordinates(key)[0,0])
                        return torch.cat((values['l0_index_cache'][None],native_coordinates(key)),1).float()
                matmul = torch.matmul
                def native_matmul(left,right,*args,**kwargs):
                    if left.shape[-1] == p['index_dim'] and right.shape[-2] == p['index_dim']:
                        left = native_coordinates(left).float()
                    return matmul(left,right,*args,**kwargs)
                position = torch.tensor([[p['context']]])
                # DeepSeek rotates with FP32 tables (TileRT).
                rotary = DeepseekV32RotaryEmbedding(config)(hidden.float(),position)
                with patch('torch.matmul',native_matmul):
                    selected = indexer(hidden,q_resid,rotary,
                                       torch.zeros(1,1,p['context']+1),position,NativeCache())[0,0]
                actual = frontier.reference(case,values,serial=True)
                torch.testing.assert_close(actual['sparse_indices'][0],selected,rtol=0,atol=0)
                torch.testing.assert_close(actual['index_k_write'][0],written['key'],rtol=0,atol=0)

    def test_glm_indexer_matches_tilert_rocm_goldens(self):
        """GLM index keys and selection against TileRT's ROCm goldens.

        Re-implements tilert/models/glm_5_2_rocm/ops/rotate.py (lane-wise
        hadamard128, FP32 LayerNorm and interleaved RoPE, one rounding),
        llm_preprocess.make_freqs_cis and sparse_index.select_golden, whose
        top-k keeps every strictly larger logit and leaves threshold ties open.
        """
        case = replace(development(next(case for case in select_cases('p3') if case.family == 'glm52_step')),
                       gpus=1, tp=1)
        case = replace(case, params=case.params | {'layers': 1, 'first_dense': 1})
        p = case.params
        rd, ih, idim, position = p['qk_rope_dim'], p['index_heads'], p['index_dim'], p['context']

        def hadamard128(x):
            v = x.reshape(-1, 16, 8).clone()
            for stage in range(3):
                stride = 1 << stage
                for j in range(4):
                    low = j & stride - 1
                    index = (j - low) * 2 + low
                    a, b = v[:, :, index].clone(), v[:, :, index + stride].clone()
                    v[:, :, index], v[:, :, index + stride] = a + b, a - b
            lane = torch.arange(16)
            for stage in range(4):
                mask = 1 << stage
                partner = v.index_select(1, lane ^ mask)
                sign = torch.where((lane & mask).bool(), torch.tensor(-1.0), torch.tensor(1.0))
                v = sign[None, :, None] * v + partner
            return v.reshape(x.shape)

        inv = 1.0 / 8_000_000.0 ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd)
        angle = torch.outer(torch.arange(position + 1), inv)
        freqs = torch.view_as_real(torch.polar(torch.ones_like(angle), angle)).reshape(position + 1, rd)[position]

        def rope_rotate(x):
            out = x.clone()
            pairs = x[..., :rd].reshape(*x.shape[:-1], rd // 2, 2)
            c, d = freqs.reshape(rd // 2, 2)[:, 0], freqs.reshape(rd // 2, 2)[:, 1]
            a, b = pairs[..., 0], pairs[..., 1]
            out[..., :rd] = torch.stack([a * c - b * d, a * d + b * c], -1).reshape(*x.shape[:-1], rd)
            return (hadamard128(out) * 0.08838834764831843).to(torch.bfloat16)

        for seed in (17, 19):
            with self.subTest(seed=seed), torch.no_grad():
                values = frontier.make_inputs(case, seed, 'cpu', rank=0)
                actual = frontier.reference(case, values, serial=True)
                gemv = lambda name, value: (unpack_fp8_weight(values['l0_'+name], values['l0_'+name+'_scale'])
                                            @ value.float()).to(torch.bfloat16)
                normalized = rms(values['embed'][values['token']], values['ln1'][0], eps=1e-5, weight_in_fp32=True)
                q_norm = rms(gemv('qa', normalized), values['l0_qn'], eps=1e-5, weight_in_fp32=True)
                key = gemv('ik', normalized).float()
                mean = key.mean(-1, keepdim=True)
                key = (key - mean) * torch.rsqrt(((key - mean) ** 2).mean(-1, keepdim=True) + 1e-6)
                key = rope_rotate(key * values['l0_inorm'].float() + values['l0_ibias'].float())
                torch.testing.assert_close(actual['index_k_write'][0], key, rtol=0, atol=0)
                query = rope_rotate(gemv('iq', q_norm).float().view(ih, idim))
                weights = (values['l0_iw'].float() @ normalized.float()).to(torch.bfloat16).float()
                cache = torch.cat((values['l0_index_cache'], key[None])).float()
                logits = (torch.relu(query.float() @ cache.T * (ih**-0.5 * idim**-0.5)) * weights[:, None]).sum(0)
                threshold = logits.sort(descending=True).values[p['index_topk'] - 1]
                chosen = set(actual['sparse_indices'][0].tolist())
                self.assertEqual(len(chosen), p['index_topk'])
                self.assertTrue(set(torch.nonzero(logits > threshold).flatten().tolist()) <= chosen)
                self.assertTrue(chosen <= set(torch.nonzero(logits >= threshold).flatten().tolist()))

    def test_absorbed_mla_matches_independent_expanded_attention(self):
        from transformers import DynamicCache
        from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention, DeepseekV32RotaryEmbedding
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding
        for original in select_cases('p3')[:2]:
            deepseek = original.family == 'deepseek_v32_step'
            case = development(original)
            case = replace(case,gpus=1,tp=1,params=case.params | {'layers':1,'index_topk':case.params['context']+1,'first_dense':1})
            p = case.params
            config_class, attention_class, rotary_class = ((DeepseekV32Config,DeepseekV32Attention,DeepseekV32RotaryEmbedding)
                if deepseek else (GlmMoeDsaConfig,GlmMoeDsaAttention,GlmMoeDsaRotaryEmbedding))
            rope = {'rope_type':'yarn','rope_theta':10000.,'factor':40.,'original_max_position_embeddings':4096,
                    'beta_fast':32,'beta_slow':1,'mscale':1.,'mscale_all_dim':1.} if deepseek else {'rope_type':'default','rope_theta':8_000_000.}
            config = config_class(hidden_size=p['hidden'],num_hidden_layers=1,num_attention_heads=p['q_heads'],
                num_key_value_heads=p['q_heads'],q_lora_rank=p['q_lora_rank'],kv_lora_rank=p['kv_lora_rank'],
                qk_nope_head_dim=p['qk_nope_dim'],qk_rope_head_dim=p['qk_rope_dim'],v_head_dim=p['v_head_dim'],
                index_n_heads=p['index_heads'],index_head_dim=p['index_dim'],index_topk=p['index_topk'],rope_parameters=rope)
            config._attn_implementation = 'eager'
            attention = attention_class(config,0).to(torch.bfloat16).eval()
            attention.indexer.weights_proj.float()
            values = frontier.make_inputs(case,17,'cpu',rank=0)
            expanded_output = {}
            def quantized(value):
                payload,scales = pack_fp8_activation(value.to(torch.bfloat16),power_of_two=deepseek)
                return unpack_fp8_activation(payload,scales)
            with torch.no_grad():
                for short,long in (('qa','q_a_proj'),('qb','q_b_proj'),('ka','kv_a_proj_with_mqa'),('kb','kv_b_proj'),('o','o_proj')):
                    module = getattr(attention,long)
                    if short != 'kb':
                        module.float()
                    module.weight.copy_(unpack_fp8_weight(values['l0_'+short],values['l0_'+short+'_scale']))
                    if short != 'kb':
                        # DeepSeek (TileRT) quantizes only the input-norm projections;
                        # GLM (TileRT ROCm) keeps BF16 activations everywhere (W8A16).
                        w8a16 = short not in frontier.DEEPSEEK_FP8_ACTIVATIONS if deepseek else short not in frontier.GLM_FP8_ACTIVATIONS
                        activation = (lambda value: value.float()) if w8a16 else quantized
                        module.register_forward_pre_hook(lambda module,args,activation=activation: (activation(args[0]),))
                        module.register_forward_hook(lambda module,args,out:out.to(torch.bfloat16))
                    else:
                        module.to(torch.bfloat16)
                attention.o_proj.register_forward_pre_hook(
                    lambda module,args: expanded_output.update(attention=args[0].detach().clone()))
                attention.q_a_layernorm.to(torch.bfloat16)
                attention.kv_a_layernorm.to(torch.bfloat16)
                attention.q_a_layernorm.weight.copy_(values['l0_qn'])
                attention.kv_a_layernorm.weight.copy_(values['l0_kn'])
                # TileRT caches the BF16 latent without FP8 scales.
                latent = values['l0_kv_cache']
                cache = DynamicCache(config=config)
                nr,rd,vd = p['qk_nope_dim'],p['qk_rope_dim'],p['v_head_dim']
                expanded = (latent@attention.kv_b_proj.weight.T).view(p['context'],p['q_heads'],nr+vd)
                # HF represents the result of interleaved rotation as half-split pairs.
                pe = values['l0_pe_cache']
                pe = torch.cat((pe[:,0::2],pe[:,1::2]),-1)
                keys = torch.cat((expanded[:,:,:nr],pe[:,None].expand(-1,p['q_heads'],-1)),-1)
                cache.update(keys.permute(1,0,2)[None],expanded[:,:,nr:].permute(1,0,2)[None],0)
                cache.update_indexer(torch.zeros(1,p['context'],p['index_dim']),0)
                token_embedding = values['embed'][values['token']]
                normalized = rms(token_embedding,values['ln1'][0],eps=1e-6 if deepseek else 1e-5)
                position = torch.tensor([[p['context']]])
                output = attention(normalized[None,None],position_embeddings=rotary_class(config)(normalized,position),
                    attention_mask=torch.zeros(1,1,1,p['context']+1),position_ids=position,past_key_values=cache)[0][0,0]
                # Disable only the FFN so full reference logits expose the attention residual.
                values['l0_down'].zero_()
                absorbed_output = {}
                native_linear, native_partial = frontier.linear, frontier.rank_partial
                def observe_linear(value, weight, scale, power, **options):
                    if weight is values['l0_o']:
                        absorbed_output['attention'] = value.detach().clone()
                    return native_linear(value, weight, scale, power, **options)
                def observe_partial(terms):
                    # GLM fuses o_proj into TileRT's residual reduce.
                    for value, weight, _, _ in terms:
                        if weight is values['l0_o']:
                            absorbed_output['attention'] = value.detach().clone()
                    return native_partial(terms)
                with patch.object(frontier, 'linear', observe_linear), patch.object(frontier, 'rank_partial', observe_partial):
                    frontier.reference(case,values,serial=True)
                # Compare before the next FP8 quantization: different BF16
                # schedules can cross its discrete thresholds. This check
                # isolates the absorption identity from those thresholds.
                expected = expanded_output['attention'].flatten().float()
                actual = absorbed_output['attention'].float()
                relative_l2 = (actual-expected).norm()/expected.norm()
                # FP8 input quantization (DeepSeek q_a/kv_a) and BF16
                # norm/expansion order differ between the schedules. Bound RMS
                # error rather than requiring bit-identical activations.
                self.assertLess(float(relative_l2), 0.05)

    def test_agent_workspace_output_contract_matches_reference(self):
        from ..harness.correctness import UNGRADED_OUTPUTS
        from ..integrations.make_agent_workspace import OUTPUT_KEYS
        for original in select_cases('p3')[:2]:
            with self.subTest(case=original.id):
                case = replace(development(original), gpus=1, tp=1)
                outputs = frontier.reference(case, frontier.make_inputs(case, 5, 'cpu', rank=0), serial=True)
                graded = set(outputs) - UNGRADED_OUTPUTS.get(case.family, frozenset())
                self.assertEqual(graded, set(OUTPUT_KEYS[case.family]))
                # Both follow TileRT's BF16 caches, without FP8 scales.
                for name in ('kv_write', 'pe_write', 'index_k_write'):
                    self.assertEqual(outputs[name].dtype, torch.bfloat16)
                self.assertEqual(outputs['logits'].dtype, torch.float32)

    def test_activation_quantization_sites(self):
        """GLM quantizes activations only for experts; DeepSeek only for q_a/kv_a/indexer wk."""
        for family in ('deepseek_v32_step', 'glm52_step'):
            with self.subTest(family=family):
                case = replace(development(next(case for case in select_cases('p3') if case.family == family)),
                               gpus=1, tp=1)
                p = case.params
                values = frontier.make_inputs(case, 7, 'cpu', rank=0)
                native_pack, native_quantize = frontier.pack_fp8_activation, frontier.tilert_quantize
                packed, quantized = [], []
                def pack(value, **options):
                    packed.append(value.shape[-1])
                    return native_pack(value, **options)
                def quantize(value):
                    quantized.append(value.shape[-1])
                    return native_quantize(value)
                with patch.object(frontier, 'pack_fp8_activation', pack), patch.object(frontier, 'tilert_quantize', quantize):
                    frontier.reference(case, values, serial=True)
                moe_layers = p['layers'] - p['first_dense']
                if family == 'deepseek_v32_step':
                    # q_a and kv_a read the same normalized input; indexer wk too.
                    self.assertEqual(packed, [p['hidden']]*3*p['layers'])
                    self.assertEqual(quantized, [])
                else:
                    self.assertEqual(packed, [])
                    inter = p['intermediate']
                    # Normalized input, then the shared and routed mids.
                    self.assertEqual(quantized, ([p['hidden']] + [inter]*(1+p['topk']))*moe_layers)

    def test_glm_residual_reduce_rounds_once(self):
        """o_proj and dense down partials join the residual in FP32 before one rounding."""
        case = replace(development(next(case for case in select_cases('p3') if case.family == 'glm52_step')),
                       gpus=1, tp=1)
        case = replace(case, params=case.params | {'layers': 1, 'first_dense': 1})
        values = frontier.make_inputs(case, 11, 'cpu', rank=0)
        native, partials = frontier.rank_partial, []
        def record(terms):
            partials.append(native(terms))
            return partials[-1]
        with torch.no_grad(), patch.object(frontier, 'rank_partial', record):
            actual = frontier.reference(case, values, serial=True)
        self.assertEqual(len(partials), 2)
        self.assertTrue(all(partial.dtype == torch.bfloat16 for partial in partials))
        x = values['embed'][values['token']]
        for partial in partials:
            x = (x.float() + partial.float()).to(torch.bfloat16)
        final = rms(x, values['fnorm'], eps=1e-5, weight_in_fp32=True)
        self.assertTrue(torch.equal(actual['logits'], values['lm_head'].float() @ final.float()))
        # The partial itself is one FP32 accumulation rounded once.
        weight, scale = values['l0_down'], values['l0_down_scale']
        mid = torch.randn(weight.shape[1]).to(torch.bfloat16)
        expected = (unpack_fp8_weight(weight, scale) @ mid.float() * 0.5 + unpack_fp8_weight(weight, scale) @ mid.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(native([(mid, weight, scale, 0.5), (mid, weight, scale, 1.0)]), expected))

    def test_glm_moe_matches_tilert_w8a8_goldens(self):
        """Routing, W8A8 experts and the down partial against TileRT ROCm goldens.

        Re-implements moe_router.select_topk_golden, upgate_silu_w8a8.quant_std_blocks
        and fused_moe_w8a8.golden_forward_w8a8/golden_down_w8a8 (per-K-block
        scale folding); only FP32 summation order differs from the reference.
        """
        case = replace(development(next(case for case in select_cases('p3') if case.family == 'glm52_step')),
                       gpus=1, tp=1)
        # Layer 0 is dense (first_dense >= 1); layer 1 is the MoE layer under test.
        case = replace(case, params=case.params | {'layers': 2, 'first_dense': 1})
        p = case.params

        def quant(x):
            blocks = x.float().reshape(-1, 128)
            scale = blocks.abs().amax(-1).clamp_min(1e-4) * torch.tensor(1.0 / 448, dtype=torch.float32)
            return (blocks * (torch.ones_like(scale) / scale)[:, None]).to(torch.float8_e4m3fn).float(), scale

        def blocked(weight, scale, payload, act_scale, factor=1.0):
            out = torch.zeros(weight.shape[0])
            rows = torch.arange(weight.shape[0]) // 128
            for kb in range(weight.shape[1] // 128):
                part = weight.float()[:, kb*128:(kb+1)*128] @ payload[kb]
                out += part * (scale[rows, kb] * factor * float(act_scale[kb]))
            return out

        for seed in (5, 23):
            with self.subTest(seed=seed), torch.no_grad():
                values = frontier.make_inputs(case, seed, 'cpu', rank=0)
                native, partials = frontier.rank_partial, []
                def record(terms):
                    partials.append(native(terms))
                    return partials[-1]
                with patch.object(frontier, 'rank_partial', record):
                    actual = frontier.reference(case, values, serial=True)
                x = values['embed'][values['token']]
                for partial in partials[:3]:  # o_proj, dense down, o_proj
                    x = (x.float() + partial.float()).to(torch.bfloat16)
                normalized = rms(x, values['ln2'][1], eps=1e-5, weight_in_fp32=True)
                logits = values['l1_router'].float() @ normalized.float()
                scores = torch.sigmoid(logits)
                ranked, ids, picked = scores + values['l1_router_bias'].float(), [], []
                for _ in range(p['topk']):
                    expert = int(torch.argmax(ranked))
                    ids.append(expert)
                    picked.append(float(scores[expert]))
                    ranked[expert] = -float('inf')
                total = sum(picked)
                probabilities = [v * (2.5 / total) for v in picked]
                self.assertEqual(actual['expert_ids'][1].tolist(), ids)
                activation, activation_scale = quant(normalized)
                out = torch.zeros(p['hidden'])
                slots = [('shared_', None, 1.0)] + [('', e, w) for e, w in zip(ids, probabilities)]
                for prefix, expert, factor in slots:
                    def weight(name):
                        w, s = values[f'l1_{prefix}{name}'], values[f'l1_{prefix}{name}_scale']
                        return (w, s) if expert is None else (w[expert], s[expert])
                    gate = blocked(*weight('gate'), activation, activation_scale)
                    up = blocked(*weight('up'), activation, activation_scale)
                    mid = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
                    out += blocked(*weight('down'), *quant(mid), factor)
                torch.testing.assert_close(partials[3].float(), out.to(torch.bfloat16).float(), rtol=0.008, atol=1e-6)

    def test_glm_route_and_quantizer_formulas(self):
        ids, weights = frontier.route_tilert(torch.zeros(8), torch.zeros(8), 2)
        self.assertTrue(torch.equal(ids, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(weights, torch.full((2,), 1.25)))
        torch.manual_seed(3)
        value = (torch.randn(2, 256)*3).to(torch.bfloat16)
        value[1, 128:] = 0
        payload, scale = frontier.tilert_quantize(value)
        blocks = value.float().view(2, 2, 128)
        expected_scale = blocks.abs().amax(-1).clamp_min(1e-4) * torch.tensor(1/448, dtype=torch.float32)
        self.assertTrue(torch.equal(scale, expected_scale))
        self.assertEqual(float(scale[1, 1]), float(torch.tensor(1e-4) * torch.tensor(1/448)))
        expected = (blocks * (torch.ones_like(expected_scale)/expected_scale)[..., None]).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(payload.view(torch.uint8), expected.view(2, 256).view(torch.uint8)))

    def test_fp8_native_block_and_dynamic_scales(self):
        torch.manual_seed(13)
        value = torch.randn(137, 259).to(torch.bfloat16)
        for power in (False, True):
            payload, scales = pack_fp8_weight(value, power_of_two=power)
            expected = torch.empty_like(value, dtype=torch.float32)
            for row in range(0, 137, 128):
                for col in range(0, 259, 128):
                    block = value[row:row+128, col:col+128].float()
                    scale = block.abs().max().clamp_min(1e-4)/448
                    if power:
                        scale = 2**torch.ceil(torch.log2(scale))
                    self.assertEqual(scales[row//128, col//128], scale)
                    expected[row:row+128, col:col+128] = (block/scale).to(torch.float8_e4m3fn).float()*scale
            self.assertTrue(torch.equal(unpack_fp8_weight(payload, scales), expected))
            payload, scales = pack_fp8_activation(value, power_of_two=power)
            for row in (0, 136):
                for col in range(0, 259, 128):
                    block = value[row, col:col+128].float()
                    scale = block.abs().max().clamp_min(1e-4)/448
                    if power:
                        scale = 2**torch.ceil(torch.log2(scale))
                    self.assertEqual(scales[row, col//128], scale)
                    torch.testing.assert_close(unpack_fp8_activation(payload, scales)[row, col:col+128],
                                               (block/scale).to(torch.float8_e4m3fn).float()*scale, rtol=0, atol=0)

    def test_global_quantized_blocks_are_invariant_to_tp(self):
        for power in (False, True):
            full, scales = frontier.fp8_matrix(7, 'matrix', 512, 256, 'cpu', power)
            for row in (0, 256):
                shard, shard_scales = frontier.fp8_matrix(7, 'matrix', 512, 256, 'cpu', power, row_range=(row,row+256))
                self.assertTrue(torch.equal(shard, full[row:row+256]))
                self.assertTrue(torch.equal(shard_scales, scales[row//128:(row+256)//128]))
            for col in (0, 128):
                shard, shard_scales = frontier.fp8_matrix(7, 'matrix', 512, 256, 'cpu', power, col_range=(col,col+128))
                self.assertTrue(torch.equal(shard, full[:,col:col+128]))
                self.assertTrue(torch.equal(shard_scales, scales[:,col//128:(col+128)//128]))

    def test_hadamard_and_yarn_match_primary_oracles(self):
        from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RotaryEmbedding
        value = torch.randn(4,128).to(torch.bfloat16)
        h = torch.ones(1,1)
        for _ in range(7):
            h = torch.cat((torch.cat((h,h),1), torch.cat((h,-h),1)),0)
        self.assertTrue(torch.equal(frontier.hadamard(value), (value.float()@h/128**0.5).to(torch.bfloat16)))
        config = DeepseekV32Config(qk_rope_head_dim=64, rope_parameters={
            'rope_type':'yarn', 'rope_theta':10000., 'factor':40.,
            'original_max_position_embeddings':4096, 'beta_fast':32, 'beta_slow':1,
            'mscale':1., 'mscale_all_dim':1., 'truncate':True})
        model = DeepseekV32RotaryEmbedding(config)
        value = torch.randn(4,64)
        for position in (128,4096,163000):
            cos,sin = model(value, torch.tensor([[position]]))
            a,b = value[:,0::2],value[:,1::2]
            c,s = cos[0,0,:32],sin[0,0,:32]
            expected = torch.stack((a*c-b*s,b*c+a*s),-1).flatten(-2)
            torch.testing.assert_close(frontier.rotate(value, position, deepseek=True, interleaved=True), expected, rtol=1e-5, atol=1e-5)

    def test_router_correction_group_selection_and_probabilities(self):
        from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32TopkRouter
        config = DeepseekV32Config(hidden_size=128,n_routed_experts=8,num_experts_per_tok=2,n_group=2,topk_group=1)
        router = DeepseekV32TopkRouter(config).float()
        with torch.no_grad():
            router.weight.normal_()
            router.e_score_correction_bias.copy_(torch.linspace(-0.2,0.2,8))
            value = torch.randn(1,128)
            logits, weights, ids = router(value)
            actual_ids,actual_weights = frontier.route(logits[0],router.e_score_correction_bias,2,2,1)
            order, actual_order = ids[0].argsort(), actual_ids.argsort()
            self.assertTrue(torch.equal(ids[0][order],actual_ids[actual_order]))
            torch.testing.assert_close(weights[0][order],actual_weights[actual_order])
        ids, weights = frontier.route(torch.zeros(8), torch.zeros(8),2,2,1)
        self.assertTrue(torch.equal(ids,torch.tensor([0,1])))
        self.assertTrue(torch.equal(weights,torch.full((2,),1.25)))


if __name__ == '__main__':
    unittest.main()
