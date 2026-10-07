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
    def test_native_sparse_selection_matches_upstream_indexers(self):
        from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer, DeepseekV32RotaryEmbedding, DeepseekV32RMSNorm
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer, GlmMoeDsaRotaryEmbedding, GlmMoeDsaRMSNorm
        for original in select_cases('p3')[:2]:
            case = development(original)
            case = replace(case,gpus=1,tp=1,params=case.params | {'layers':1,'first_dense':1})
            p = case.params
            deepseek = case.family == 'deepseek_v32_step'
            config_class,indexer_class,rotary_class,norm_class = (
                (DeepseekV32Config,DeepseekV32Indexer,DeepseekV32RotaryEmbedding,DeepseekV32RMSNorm)
                if deepseek else (GlmMoeDsaConfig,GlmMoeDsaIndexer,GlmMoeDsaRotaryEmbedding,GlmMoeDsaRMSNorm))
            rope = {'rope_type':'yarn','rope_theta':10000.,'factor':40.,'original_max_position_embeddings':4096,
                    'beta_fast':32,'beta_slow':1,'mscale':1.,'mscale_all_dim':1.} if deepseek else {'rope_type':'default','rope_theta':8_000_000.}
            config = config_class(hidden_size=p['hidden'],q_lora_rank=p['q_lora_rank'],qk_rope_head_dim=p['qk_rope_dim'],
                                  index_n_heads=p['index_heads'],index_head_dim=p['index_dim'],index_topk=p['index_topk'],rope_parameters=rope)
            # This independently constructs the normalized Hadamard matrix.
            hadamard = torch.ones(1,1)
            while hadamard.shape[0] < p['index_dim']:
                hadamard = torch.cat((torch.cat((hadamard,hadamard),1),torch.cat((hadamard,-hadamard),1)),0)
            for seed in (17,19):
                with self.subTest(case=case.id,seed=seed), torch.no_grad():
                    values = frontier.make_inputs(case,seed,'cpu',rank=0)
                    values['l0_iw'].abs_()  # Positive scores avoid source-specific zero-score topk ties.
                    indexer = indexer_class(config,0).float().eval()
                    def quantize(value):
                        payload,scale = pack_fp8_activation(value.to(torch.bfloat16),power_of_two=deepseek)
                        return unpack_fp8_activation(payload,scale)
                    for short,long in (('iq','wq_b'),('ik','wk')):
                        module = getattr(indexer,long)
                        module.weight.copy_(unpack_fp8_weight(values['l0_'+short],values['l0_'+short+'_scale']))
                        module.register_forward_pre_hook(lambda module,args:(quantize(args[0]),))
                        module.register_forward_hook(lambda module,args,out:out.to(torch.bfloat16))
                    indexer.k_norm.to(torch.bfloat16)
                    indexer.k_norm.weight.copy_(values['l0_inorm'])
                    indexer.k_norm.bias.copy_(values['l0_ibias'])
                    indexer.weights_proj.weight.copy_(values['l0_iw'])
                    norm = norm_class(p['hidden'],eps=1e-6 if deepseek else 1e-5).to(torch.bfloat16)
                    norm.weight.copy_(values['ln1'][0])
                    hidden = norm(values['embed'][values['token']])[None,None]
                    qnorm = norm_class(p['q_lora_rank'],eps=1e-6).to(torch.bfloat16)
                    qnorm.weight.copy_(values['l0_qn'])
                    qa = torch.nn.Linear(p['hidden'],p['q_lora_rank'],bias=False)
                    qa.weight.copy_(unpack_fp8_weight(values['l0_qa'],values['l0_qa_scale']))
                    q_resid = qnorm(qa(quantize(hidden)).to(torch.bfloat16))
                    def native_coordinates(value):
                        if not deepseek:
                            # HF emits half-split rotated pairs; native MLA/index cache uses interleaved pairs.
                            rotated,passed = value.split((p['qk_rope_dim'],p['index_dim']-p['qk_rope_dim']),-1)
                            left,right = rotated.chunk(2,-1)
                            value = torch.cat((torch.stack((left,right),-1).flatten(-2),passed),-1)
                        value = value.to(torch.bfloat16)
                        return (value.float()@hadamard/p['index_dim']**0.5).to(torch.bfloat16) if deepseek else value
                    written = {}
                    class NativeCache:
                        def update_indexer(self,key,layer):
                            payload,scale = pack_fp8_activation(native_coordinates(key),power_of_two=deepseek)
                            written.update(payload=payload[0,0],scale=scale[0,0])
                            old = unpack_fp8_activation(values['l0_index_cache'],values['l0_index_cache_scale'])[None]
                            return torch.cat((old,unpack_fp8_activation(payload,scale)),1)
                    matmul = torch.matmul
                    def native_matmul(left,right,*args,**kwargs):
                        if left.shape[-1] == p['index_dim'] and right.shape[-2] == p['index_dim']:
                            left = quantize(native_coordinates(left))
                        return matmul(left,right,*args,**kwargs)
                    position = torch.tensor([[p['context']]])
                    with patch('torch.matmul',native_matmul):
                        selected = indexer(hidden,q_resid,rotary_class(config)(hidden,position),
                                           torch.zeros(1,1,p['context']+1),position,NativeCache())[0,0]
                    actual = frontier.reference(case,values,serial=True)
                    torch.testing.assert_close(actual['sparse_indices'][0],selected,rtol=0,atol=0)
                    torch.testing.assert_close(actual['index_k_write'][0].float(),written['payload'].float(),rtol=0,atol=0)
                    torch.testing.assert_close(actual['index_scale_write'][0],written['scale'],rtol=0,atol=0)

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
                        module.register_forward_pre_hook(lambda module,args: (quantized(args[0]),))
                        module.register_forward_hook(lambda module,args,out:out.to(torch.bfloat16))
                    else:
                        module.to(torch.bfloat16)
                attention.o_proj.register_forward_pre_hook(
                    lambda module,args: expanded_output.update(attention=args[0].detach().clone()))
                attention.q_a_layernorm.to(torch.bfloat16)
                attention.kv_a_layernorm.to(torch.bfloat16)
                attention.q_a_layernorm.weight.copy_(values['l0_qn'])
                attention.kv_a_layernorm.weight.copy_(values['l0_kn'])
                attention.kv_a_layernorm.register_forward_hook(lambda module,args,out: quantized(out).to(torch.bfloat16))
                cache = DynamicCache(config=config)
                latent = unpack_fp8_activation(values['l0_kv_cache'],values['l0_kv_cache_scale']).to(torch.bfloat16)
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
                native_linear = frontier.linear
                def observe_linear(value, weight, scale, power):
                    if weight is values['l0_o']:
                        absorbed_output['attention'] = value.detach().clone()
                    return native_linear(value, weight, scale, power)
                with patch.object(frontier, 'linear', observe_linear):
                    frontier.reference(case,values,serial=True)
                # Compare before the next FP8 quantization: different BF16
                # schedules can cross its discrete thresholds. This check
                # isolates the absorption identity from those thresholds.
                expected = expanded_output['attention'].flatten().float()
                actual = absorbed_output['attention'].float()
                relative_l2 = (actual-expected).norm()/expected.norm()
                # The native latent cache also passes through FP8 before
                # either schedule; BF16 norm/expansion order can change that
                # quantization. Bound RMS error rather than requiring the
                # distinct schedules to produce bit-identical activations.
                self.assertLess(float(relative_l2), 0.05)

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
