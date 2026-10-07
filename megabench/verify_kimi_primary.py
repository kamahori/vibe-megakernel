"""Compare the K3 reference with pinned Moonshot decoder classes on CPU.

Provide local official source files whose URLs/hashes are in model_specs.
Only the named math classes/functions are loaded. CPU adapters replace FLA
GPU dispatch with torch convolution and FLA's own naive KDA recurrence; the
original decoder, MLA, SiTU, latent MoE, router and attention residuals run
unchanged. This verifies development geometry, not 16-GPU full execution.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN

from .tasks import kimi
from .tasks.quantization import unpack_mxfp4
from .tests.test_kimi import development
from .verify_frontier import advance_state, input_digest


def load_nodes(path, names, namespace, pin):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != pin['sha256']:
        raise ValueError(f'primary source hash differs: {path}')
    selected = [node for node in ast.parse(source).body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f'missing primary definitions in {path}')
    future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future,*selected],type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)


class CpuConvolution(nn.Conv1d):
    def __init__(self,hidden_size,kernel_size,activation='silu'):
        super().__init__(hidden_size,hidden_size,kernel_size,groups=hidden_size,bias=False)

    def forward(self,x,cache=None,**kwargs):
        old = torch.zeros(x.shape[0],x.shape[-1],self.kernel_size[0],dtype=torch.bfloat16) if cache is None else cache
        # Native state is BF16 even while the oracle accumulates in FP32.
        window = torch.cat((old[...,1:],x.transpose(1,2).to(torch.bfloat16)),-1)
        result = F.silu(F.conv1d(window.float(),self.weight.float(),groups=self.groups)).transpose(1,2)
        return result,window


class CpuGatedNorm(nn.Module):
    def __init__(self,width,eps,activation):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self,x,gate):
        return x*(x.square().mean(-1,keepdim=True)+self.eps).rsqrt()*self.weight*gate.sigmoid()


def primary_oracle(case,values,namespace):
    p = case.params
    full = [layer+1 for layer in range(p['layers']) if kimi.full_layer(case,layer)]
    config = namespace['KimiLinearConfig'](hidden_size=p['hidden'],intermediate_size=p['dense_intermediate'],
        num_hidden_layers=p['layers'],num_attention_heads=p['q_heads'],num_key_value_heads=p['q_heads'],
        vocab_size=p['vocab'],hidden_act='situ',rms_norm_eps=1e-5,
        q_lora_rank=p['q_lora_rank'],kv_lora_rank=p['kv_lora_rank'],qk_nope_head_dim=p['qk_nope_dim'],
        qk_rope_head_dim=p['qk_rope_dim'],v_head_dim=p['v_head_dim'],mla_use_nope=True,mla_use_output_gate=True,
        moe_intermediate_size=p['intermediate'],num_experts=p['experts'],num_experts_per_token=p['topk'],
        num_shared_experts=p['shared_experts'],first_k_dense_replace=p['first_dense'],
        routed_expert_hidden_size=p['latent_hidden'],latent_moe_use_norm=True,attn_res_block_size=p['attn_res_block_size'],
        activation_situ_beta=4.,activation_situ_linear_beta=25.,linear_attn_config={
            'full_attn_layers':full,'kda_layers':[i+1 for i in range(p['layers']) if i+1 not in full],
            'head_dim':p['head_dim'],'num_heads':p['q_heads'],'short_conv_kernel_size':p['conv_kernel'],
            'gate_lower_bound':-5.,'use_full_rank_gate':True})
    config._attn_implementation = 'eager'
    cache = namespace['KimiDynamicCache'](config)
    layers = [namespace['KimiDecoderLayer'](config,i).float().eval() for i in range(p['layers'])]
    captured = {}
    reset = bool(values['reset'])
    def copy(parameter,value):
        parameter.copy_(value)
    def record(key, transform):
        def hook(module, inputs, output):
            captured[key] = transform(output).detach().clone()
        return hook
    with torch.no_grad():
        for index,layer in enumerate(layers):
            name = f'l{index}_'
            for attribute,key in (('input_layernorm','ln1'),('post_attention_layernorm','ln2'),
                                  ('self_attention_res_norm','attention_res_norm'),('mlp_res_norm','mlp_res_norm')):
                copy(getattr(layer,attribute).weight,values[key][index])
            copy(layer.self_attention_res_proj.weight,values['attention_res_proj'][index][None])
            copy(layer.mlp_res_proj.weight,values['mlp_res_proj'][index][None])
            attention = layer.self_attn
            if kimi.full_layer(case,index):
                for short,long in (('qa','q_a_proj'),('qb','q_b_proj'),('ka','kv_a_proj_with_mqa'),('kb','kv_b_proj'),
                                   ('o','o_proj'),('output_gate','g_proj')):
                    copy(getattr(attention,long).weight,values[name+short])
                copy(attention.q_a_layernorm.weight,values[name+'qn'])
                copy(attention.kv_a_layernorm.weight,values[name+'kn'])
                attention.kv_a_layernorm.register_forward_hook(record(name+'latent',lambda out:out[0,0].to(torch.bfloat16)))
                attention.kv_a_proj_with_mqa.register_forward_hook(record(name+'pe',lambda out:out[0,0,p['kv_lora_rank']:].to(torch.bfloat16)))
                if not reset:
                    nr,rd,vd = p['qk_nope_dim'],p['qk_rope_dim'],p['v_head_dim']
                    expanded = (values[name+'kv_cache'].float()@attention.kv_b_proj.weight.T).view(p['context'],p['q_heads'],nr+vd)
                    key = torch.cat((expanded[:,:,:nr],values[name+'pe_cache'].float()[:,None].expand(-1,p['q_heads'],-1)),-1)
                    cache.update(key.permute(1,0,2)[None],expanded[:,:,nr:].permute(1,0,2)[None],index)
            else:
                for short,long in (('q','q_proj'),('k','k_proj'),('v','v_proj'),('output_gate','g_proj'),('o','o_proj'),
                                   ('fa','f_a_proj'),('fb','f_b_proj'),('beta','b_proj')):
                    copy(getattr(attention,long).weight,values[name+short])
                copy(attention.A_log,values[name+'A_log'])
                copy(attention.dt_bias,values[name+'dt_bias'])
                copy(attention.o_norm.weight,values[name+'onorm'])
                for slot,key in enumerate(('q','k','v')):
                    copy(getattr(attention,key+'_conv1d').weight,values[name+'conv_weight'][slot][:,None])
                if not reset:
                    cache.conv_states[index] = tuple(state[None].clone() for state in values[name+'conv_state'])
                    cache.recurrent_states[index] = values[name+'recurrent_state'][None].clone()
            if index < p['first_dense']:
                for short,long in (('gate','gate_proj'),('up','up_proj'),('down','down_proj')):
                    copy(getattr(layer.mlp,long).weight,values[name+short])
            else:
                moe = layer.block_sparse_moe
                copy(moe.gate.weight,values[name+'router'])
                copy(moe.gate.e_score_correction_bias,values[name+'router_bias'])
                moe.gate.register_forward_hook(record(name+'experts',lambda out:out[0][0]))
                for short,long in (('latent_down','routed_expert_down_proj'),('latent_up','routed_expert_up_proj')):
                    copy(getattr(moe,long).weight,values[name+short])
                copy(moe.routed_expert_norm.weight,values[name+'latent_norm'])
                for short,long in (('shared_gate','gate_proj'),('shared_up','up_proj'),('shared_down','down_proj')):
                    copy(getattr(moe.shared_experts,long).weight,values[name+short])
                for expert in range(p['experts']):
                    for short,long in (('gate','w1'),('up','w3'),('down','w2')):
                        copy(getattr(moe.experts[expert],long).weight,unpack_mxfp4(values[name+short+'_blocks'][expert],values[name+short+'_scales'][expert]))
        prefix = values['embed'][values['token']].float()[None,None]
        residual = torch.empty(1,0,p['hidden'])
        for layer in layers:
            prefix,residual = layer(prefix,past_key_values=cache,use_cache=True,block_residual=residual)
        norm = namespace['KimiRMSNorm'](p['hidden'],eps=1e-5)
        copy(norm.weight,values['output_res_norm'])
        projection = nn.Linear(p['hidden'],1,bias=False)
        copy(projection.weight,values['output_res_proj'][None])
        hidden = namespace['_apply_attn_res'](prefix[0],residual,projection,norm)
        copy(norm.weight,values['fnorm'])
        logits = (norm(hidden)@values['lm_head'].float().T)[0]
    return logits,cache,captured


def compare_step(case, actual, expected, cache, captured):
    torch.testing.assert_close(actual['logits'],expected,rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(actual['next_token'],expected.argmax(),rtol=0,atol=0)
    full_slot, recurrent_slot = 0, 0
    first_full = None
    for layer in range(case.params['layers']):
        if kimi.full_layer(case,layer):
            first_full = layer if first_full is None else first_full
            for short,key in (('kv_write','latent'),('pe_write','pe')):
                torch.testing.assert_close(actual[short][full_slot],captured[f'l{layer}_{key}'],
                                           rtol=case.bf16_rtol,atol=case.atol)
            full_slot += 1
        else:
            torch.testing.assert_close(actual['recurrent_state'][recurrent_slot],cache.recurrent_states[layer][0],rtol=2e-5,atol=2e-5)
            for component in range(3):
                torch.testing.assert_close(actual['conv_state'][recurrent_slot,component],cache.conv_states[layer][component][0],
                                           rtol=case.bf16_rtol,atol=case.atol)
            recurrent_slot += 1
        if layer >= case.params['first_dense']:
            # The official router's topk is unsorted; the public contract
            # orders the same selected IDs by corrected score with stable ties.
            torch.testing.assert_close(actual['expert_ids'][layer].sort().values,captured[f'l{layer}_experts'].sort().values,rtol=0,atol=0)
    if int(actual['cache_length']) != cache.get_seq_length(first_full):
        raise AssertionError('compressed/expanded MLA cache lengths disagree')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-source',required=True,type=Path)
    parser.add_argument('--config-source',required=True,type=Path)
    parser.add_argument('--fla-source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--steps',type=int,default=3)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    pin = json.loads((Path(__file__).parent/'tasks/model_specs.json').read_text())['moonshotai/Kimi-K3']['source_files']
    namespace = {'torch':torch,'nn':nn,'F':F,'math':math,'rearrange':rearrange,'ACT2FN':ACT2FN,'PretrainedConfig':PretrainedConfig,
                 'ShortConvolution':CpuConvolution,'FusedRMSNormGated':CpuGatedNorm}
    load_nodes(args.config_source,['KimiLinearConfig'],namespace,pin['configuration_kimi_k3.py'])
    load_nodes(args.fla_source,['naive_recurrent_kda'],namespace,pin['fla/naive.py'])
    def cpu_kda(q,k,v,g,beta,A_log,dt_bias,initial_state,**kwargs):
        q = q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt()
        k = k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()
        log_decay = -5*torch.sigmoid(A_log.float().exp()[None,None,:,None]*(g+dt_bias.view(g.shape[-2:])))
        output,state = namespace['naive_recurrent_kda'](q,k,v,log_decay,beta.sigmoid(),
            initial_state=initial_state.transpose(-1,-2) if initial_state is not None else None,output_final_state=True)
        return output,state.transpose(-1,-2)
    namespace.update(chunk_kda=cpu_kda,fused_recurrent_kda=cpu_kda)
    names = ['SituAndMul','_get_situ_activation_params','KimiDynamicCache','KimiRMSNorm','KimiBlockSparseMLP',
             'KimiMLP','repeat_kv','eager_attention_forward','KimiMLAAttention','KimiDeltaAttention',
             'KimiMoEGate','KimiSparseMoeBlock','KimiDecoderLayer','_apply_attn_res']
    load_nodes(args.model_source,names,namespace,pin['modeling_kimi_linear.py'])
    case = replace(development(),gpus=1,tp=1)
    report = {'case':case.to_dict(),'tier':'development_geometry','steps':args.steps,'primary_sources':pin,'trials':[]}
    with torch.inference_mode():
        for seed in (17,18):
            current = case
            values = kimi.make_inputs(current,seed,'cpu',rank=0)
            trajectory = []
            for step in range(args.steps):
                before = input_digest(values)
                expected,cache,captured = primary_oracle(current,values,namespace)
                actual = kimi.reference(current,values,serial=True)
                compare_step(current,actual,expected,cache,captured)
                if before != input_digest(values):
                    raise AssertionError('reference or primary oracle mutated runtime inputs')
                trajectory.append({'step':step,'input_context':current.params['context'],'reset':bool(values['reset']),
                                   'next_token':int(actual['next_token']),'cache_length':int(actual['cache_length']),
                                   'max_logit_error':float((actual['logits']-expected).abs().max())})
                if step+1 < args.steps:
                    current,values = advance_state(current,values,actual)
            report['trials'].append({'seed':seed,'trajectory':trajectory})
    report['status'] = 'pass'
    with args.output.open('x') as file:
        json.dump(report,file,indent=2)
    print(json.dumps(report['trials']))


if __name__ == '__main__':
    main()
