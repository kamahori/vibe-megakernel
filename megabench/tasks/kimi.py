"""Kimi K3 text decode: KDA, gated MLA, attention residuals and latent MoE.

Expert matrices retain native row-major MXFP4/E8M0 groups of 32. Attention,
shared experts, dense FFNs and latent projections remain BF16, matching the
checkpoint's quantization exclusions. TP shards attention heads and FFN
intermediates; latent MoE reduction precedes its normalization/up projection.
K3's published MLA uses no rotary transform. KDA stores V-first FP32 state.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..cases import Case
from .common import rms
from .distributed import matrix, seed_for, reduce_sum, greedy_token
from .frontier import route
from .parallel import initialize
from .quantization import pack_mxfp4, unpack_mxfp4


def full_layer(case: Case, layer: int) -> bool:
    return (layer+1)%4 == 0 or layer == case.params['layers']-1


def mxfp4_matrix(seed, name, rows, cols, device, *, row_range=None, col_range=None):
    r0,r1 = row_range or (0,rows)
    c0,c1 = col_range or (0,cols)
    if c0%32 or c1%32:
        raise ValueError('MXFP4 input-channel shards must align to groups of 32')
    blocks = torch.empty((r1-r0,(c1-c0)//32,16),device=device,dtype=torch.uint8)
    scales = torch.empty((r1-r0,(c1-c0)//32),device=device,dtype=torch.uint8)
    for start in range(r0,r1,128):
        end = min(start+128,r1)
        raw = matrix(seed,name,rows,cols,cols,device,row_range=(start,end),col_range=(c0,c1))
        packed,scale = pack_mxfp4(raw)
        blocks[start-r0:end-r0].copy_(packed)
        scales[start-r0:end-r0].copy_(scale)
    return blocks,scales


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None):
    rank = dist.get_rank() if rank is None else rank
    p,tp = case.params,case.tp
    if case.ep != 1 or case.gpus != tp or not 0 <= rank < case.gpus:
        raise ValueError('Kimi requires one TP group spanning all ranks')
    h,lh,d = p['hidden'],p['latent_hidden'],p['head_dim']
    heads,qdim = p['q_heads'],p['q_heads']*d
    for n in (heads,p['vocab'],p['intermediate'],p['dense_intermediate']):
        if n%tp:
            raise ValueError('Kimi attention heads, vocabulary and FFN intermediate must divide TP')
    split = lambda n:(rank*(n//tp),(rank+1)*(n//tp))
    values = {'token':torch.tensor(seed_for(seed,'token')%p['vocab'],device=device),
              'reset':torch.tensor(seed%3 == 0,device=device)}
    for name in ('embed','lm_head'):
        values[name] = matrix(seed,name,p['vocab'],h,h,device,row_range=split(p['vocab']))
    for name in ('fnorm','output_res_norm'):
        values[name] = (matrix(seed,name,1,h,100,device)[0].float()+1).to(torch.bfloat16)
    values['output_res_proj'] = matrix(seed,'output_res_proj',1,h,h,device)[0]
    for name in ('ln1','ln2','attention_res_norm','mlp_res_norm'):
        values[name] = (matrix(seed,name,p['layers'],h,100,device).float()+1).to(torch.bfloat16)
    for name in ('attention_res_proj','mlp_res_proj'):
        values[name] = matrix(seed,name,p['layers'],h,h,device)
    for layer in range(p['layers']):
        prefix = f'l{layer}_'
        def weight(name,rows,cols,rr=None,cr=None):
            values[prefix+name] = matrix(seed,prefix+name,rows,cols,cols,device,row_range=rr,col_range=cr)
        def norm(name,width):
            values[prefix+name] = (matrix(seed,prefix+name,1,width,100,device)[0].float()+1).to(torch.bfloat16)
        if full_layer(case,layer):
            qr,kr,nr,rd,vd = p['q_lora_rank'],p['kv_lora_rank'],p['qk_nope_dim'],p['qk_rope_dim'],p['v_head_dim']
            weight('qa',qr,h)
            weight('qb',heads*(nr+rd),qr,rr=split(heads*(nr+rd)))
            weight('ka',kr+rd,h)
            weight('kb',heads*(nr+vd),kr,rr=split(heads*(nr+vd)))
            weight('o',h,heads*vd,cr=split(heads*vd))
            weight('output_gate',heads*vd,h,rr=split(heads*vd))
            norm('qn',qr)
            norm('kn',kr)
            values[prefix+'kv_cache'] = matrix(seed,prefix+'kv_cache',p['context'],kr,kr,device)
            values[prefix+'pe_cache'] = matrix(seed,prefix+'pe_cache',p['context'],rd,rd,device)
        else:
            for name in ('q','k','v','output_gate'):
                weight(name,qdim,h,rr=split(qdim))
            weight('o',h,qdim,cr=split(qdim))
            weight('fa',d,h)
            weight('fb',qdim,d,rr=split(qdim))
            weight('beta',heads,h,rr=split(heads))
            norm('onorm',d)
            values[prefix+'A_log'] = matrix(seed,prefix+'A_log',1,heads,100,device,col_range=split(heads))[0].float()+1
            values[prefix+'dt_bias'] = matrix(seed,prefix+'dt_bias',1,qdim,100,device,col_range=split(qdim))[0].float()-2
            conv,states = [],[]
            for name in ('q','k','v'):
                conv.append(matrix(seed,prefix+name+'_conv',qdim,p['conv_kernel'],p['conv_kernel'],device,row_range=split(qdim)))
                states.append(matrix(seed,prefix+name+'_conv_cache',qdim,p['conv_kernel'],p['conv_kernel'],device,row_range=split(qdim)))
            values[prefix+'conv_weight'] = torch.stack(conv)
            values[prefix+'conv_state'] = torch.stack(states)
            state = matrix(seed,prefix+'recurrent',heads*d,d,d,device,row_range=split(heads*d))
            values[prefix+'recurrent_state'] = state.view(heads//tp,d,d).float()
        if layer < p['first_dense']:
            inter = p['dense_intermediate']
            weight('gate',inter,h,rr=split(inter))
            weight('up',inter,h,rr=split(inter))
            weight('down',h,inter,cr=split(inter))
        else:
            inter,experts = p['intermediate'],p['experts']
            weight('latent_down',lh,h)
            weight('latent_up',h,lh)
            norm('latent_norm',lh)
            values[prefix+'router'] = matrix(seed,prefix+'router',experts,h,h,device)
            values[prefix+'router_bias'] = matrix(seed,prefix+'router_bias',1,experts,100,device)[0].float()
            for name,rows,cols,rr,cr in (('gate',inter,lh,split(inter),None),
                                       ('up',inter,lh,split(inter),None),
                                       ('down',lh,inter,None,split(inter))):
                r,c = rr[1]-rr[0] if rr else rows,cr[1]-cr[0] if cr else cols
                blocks = torch.empty((experts,r,c//32,16),dtype=torch.uint8,device=device)
                scales = torch.empty((experts,r,c//32),dtype=torch.uint8,device=device)
                for expert in range(experts):
                    packed,scale = mxfp4_matrix(seed,prefix+name+f':{expert}',rows,cols,device,row_range=rr,col_range=cr)
                    blocks[expert].copy_(packed)
                    scales[expert].copy_(scale)
                values[prefix+name+'_blocks'],values[prefix+name+'_scales'] = blocks,scales
            shared = inter*p['shared_experts']
            weight('shared_gate',shared,h,rr=split(shared))
            weight('shared_up',shared,h,rr=split(shared))
            weight('shared_down',h,shared,cr=split(shared))
    return values


def situ(gate,up):
    return 4*torch.tanh(gate.float()/4)*gate.float().sigmoid()*(25*torch.tanh(up.float()/25))


def attention_residual(prefix,blocks,norm,projection):
    values = torch.stack([*blocks,prefix])
    normalized = rms(values,norm,eps=1e-5)
    probability = (normalized@projection.float()).softmax(0)
    return (probability[:,None]*values).sum(0)


def kda_step(query,key,value,gate,beta,A_log,dt_bias,state):
    """Per-key decay and delta update in native V-first FP32 state layout."""
    query = query.float()*torch.rsqrt(query.float().square().sum(-1,keepdim=True)+1e-6)/query.shape[-1]**0.5
    key = key.float()*torch.rsqrt(key.float().square().sum(-1,keepdim=True)+1e-6)
    decay = (-5*torch.sigmoid(A_log.float().exp()[:,None]*(gate.float()+dt_bias.float()))).exp()
    updated = state.float()*decay[:,None,:]
    error = value.float()-torch.einsum('hvk,hk->hv',updated,key)
    updated = updated+torch.einsum('hv,hk->hvk',error*beta.float().sigmoid()[:,None],key)
    return torch.einsum('hvk,hk->hv',updated,query),updated


def reference(case: Case,values,*,serial=False):
    if serial and (case.gpus,case.tp,case.ep) != (1,1,1):
        raise ValueError('serial validation requires unsharded geometry')
    ctx = SimpleNamespace(tp_rank=0,tp_group=None) if serial else initialize(case)
    sum_value = (lambda value:value) if serial else (lambda value:reduce_sum(value,ctx.tp_group))
    p,tp = case.params,case.tp
    h,d,heads = p['hidden'],p['head_dim'],p['q_heads']//tp
    chunk,offset = p['vocab']//tp,ctx.tp_rank*(p['vocab']//tp)
    token = int(values['token'])
    prefix = values['embed'][token-offset].float() if offset <= token < offset+chunk else torch.zeros(h,device=values['token'].device)
    prefix = sum_value(prefix)
    blocks,latent_writes,pe_writes,recurrent,conv_writes,routes = [],[],[],[],[],[]
    reset = bool(values['reset'])
    for layer in range(p['layers']):
        name = f'l{layer}_'
        def project(key,value):
            return values[name+key].float()@value
        hidden = attention_residual(prefix,blocks,values['attention_res_norm'][layer],values['attention_res_proj'][layer]) if blocks else prefix
        if layer%p['attn_res_block_size'] == 0:
            blocks.append(prefix)
            prefix = None
        normalized = rms(hidden,values['ln1'][layer],eps=1e-5)
        if full_layer(case,layer):
            kr,nr,rd,vd = p['kv_lora_rank'],p['qk_nope_dim'],p['qk_rope_dim'],p['v_head_dim']
            query = project('qb',rms(project('qa',normalized),values[name+'qn'],eps=1e-6)).view(heads,nr+rd)
            raw = project('ka',normalized)
            latent = rms(raw[:kr],values[name+'kn'],eps=1e-6)
            pe = raw[kr:]
            latent_writes.append(latent.to(torch.bfloat16))
            pe_writes.append(pe.to(torch.bfloat16))
            old_latent = values[name+'kv_cache'][:0] if reset else values[name+'kv_cache']
            old_pe = values[name+'pe_cache'][:0] if reset else values[name+'pe_cache']
            all_latent = torch.cat((old_latent.float(),latent[None]))
            all_pe = torch.cat((old_pe.float(),pe[None]))
            weight = values[name+'kb'].float().view(heads,nr+vd,kr)
            absorbed = torch.einsum('hd,hdc->hc',query[:,:nr],weight[:,:nr])
            probability = ((absorbed@all_latent.T+query[:,nr:]@all_pe.T)/(nr+rd)**0.5).softmax(-1)
            output = torch.einsum('hc,hdc->hd',probability@all_latent,weight[:,nr:]).flatten()
            output = output*project('output_gate',normalized).sigmoid()
            attention = sum_value(project('o',output))
        else:
            raw = torch.stack([project(key,normalized) for key in ('q','k','v')])
            old = torch.zeros_like(values[name+'conv_state']) if reset else values[name+'conv_state']
            conv = torch.cat((old[...,1:],raw.to(torch.bfloat16)[...,None]),-1)
            q,k,v = F.silu((conv.float()*values[name+'conv_weight'].float()).sum(-1)).view(3,heads,d)
            gate = project('fb',project('fa',normalized)).view(heads,d)
            beta = project('beta',normalized)
            state = torch.zeros_like(values[name+'recurrent_state']) if reset else values[name+'recurrent_state']
            output,new_state = kda_step(q,k,v,gate,beta,values[name+'A_log'],values[name+'dt_bias'].view(heads,d),state)
            recurrent.append(new_state)
            conv_writes.append(conv)
            output = rms(output,values[name+'onorm'],eps=1e-5)*project('output_gate',normalized).view(heads,d).sigmoid()
            attention = sum_value(project('o',output.flatten()))
        prefix = attention if prefix is None else prefix+attention
        hidden = attention_residual(prefix,blocks,values['mlp_res_norm'][layer],values['mlp_res_proj'][layer])
        normalized = rms(hidden,values['ln2'][layer],eps=1e-5)
        if layer < p['first_dense']:
            output = sum_value(project('down',situ(project('gate',normalized),project('up',normalized))))
            routes.append(torch.full((p['topk'],),-1,device=normalized.device,dtype=torch.int64))
        else:
            indices,weights = route(values[name+'router'].float()@normalized,values[name+'router_bias'],p['topk'],1,1,scaling=1.)
            routes.append(indices)
            latent = project('latent_down',normalized)
            expert_sum = torch.zeros(p['latent_hidden'],device=latent.device)
            for slot,expert in enumerate(indices.tolist()):
                def expert_weight(key):
                    return unpack_mxfp4(values[name+key+'_blocks'][expert],values[name+key+'_scales'][expert])
                expert_sum = expert_sum+(expert_weight('down')@situ(expert_weight('gate')@latent,expert_weight('up')@latent))*weights[slot]
            expert_sum = sum_value(expert_sum)
            routed = project('latent_up',rms(expert_sum,values[name+'latent_norm'],eps=1e-5))
            shared = sum_value(project('shared_down',situ(project('shared_gate',normalized),project('shared_up',normalized))))
            output = routed+shared
        prefix = prefix+output
    hidden = attention_residual(prefix,blocks,values['output_res_norm'],values['output_res_proj'])
    logits = values['lm_head'].float()@rms(hidden,values['fnorm'],eps=1e-5)
    return {'logits':logits,'next_token':logits.argmax() if serial else greedy_token(logits,offset,ctx.tp_group),
            'kv_write':torch.stack(latent_writes),'pe_write':torch.stack(pe_writes),
            'recurrent_state':torch.stack(recurrent),'conv_state':torch.stack(conv_writes),
            'expert_ids':torch.stack(routes),'cache_length':torch.tensor(1 if reset else p['context']+1,device=logits.device)}
