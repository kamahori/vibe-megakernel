"""Allocate and exercise one full K3 TP16 shard on a scheduled GPU.

This is a storage/local-component check, NOT a full distributed decode.
It preserves the catalog's full layers, experts and native tensor geometry.
Use verify_frontier --full with 16 ranks for full distributed validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .cases import select_cases
from .tasks import kimi
from .tasks.common import rms
from .tasks.quantization import unpack_mxfp4
from .verify_frontier import input_digest


def verify(device,rank):
    case = select_cases('all',['kimi-k3-step'])[0]
    p = case.params
    with torch.inference_mode():
        values = kimi.make_inputs(case,104729,device,rank=rank)
        before = input_digest(values)
        generator = torch.Generator(device=device).manual_seed(104729)
        hidden = torch.randn(p['hidden'],generator=generator,device=device)
        heads,d = p['q_heads']//case.tp,p['head_dim']
        counts = {'mla':0,'kda':0,'dense_ffn':0,'routed_moe':0}
        for layer in range(p['layers']):
            prefix = f'l{layer}_'
            project = lambda key,x:values[prefix+key].float()@x
            x = rms(hidden,values['ln1'][layer],eps=1e-5)
            if kimi.full_layer(case,layer):
                kr,nr,rd,vd = p['kv_lora_rank'],p['qk_nope_dim'],p['qk_rope_dim'],p['v_head_dim']
                q = project('qb',rms(project('qa',x),values[prefix+'qn'],eps=1e-6)).view(heads,nr+rd)
                raw = project('ka',x)
                latent = rms(raw[:kr],values[prefix+'kn'],eps=1e-6)
                cache = torch.cat((values[prefix+'kv_cache'].float(),latent[None]))
                pe = torch.cat((values[prefix+'pe_cache'].float(),raw[kr:][None]))
                weight = values[prefix+'kb'].float().view(heads,nr+vd,kr)
                scores = (torch.einsum('hd,hdc->hc',q[:,:nr],weight[:,:nr])@cache.T+q[:,nr:]@pe.T)/(nr+rd)**0.5
                output = torch.einsum('hc,hdc->hd',scores.softmax(-1)@cache,weight[:,nr:]).flatten()
                output = project('o',output*project('output_gate',x).sigmoid())
                counts['mla'] += 1
            else:
                raw = torch.stack([project(key,x) for key in ('q','k','v')])
                conv = torch.cat((values[prefix+'conv_state'][...,1:],raw.to(torch.bfloat16)[...,None]),-1)
                q,k,v = F.silu((conv.float()*values[prefix+'conv_weight'].float()).sum(-1)).view(3,heads,d)
                gate = project('fb',project('fa',x)).view(heads,d)
                output,state = kimi.kda_step(q,k,v,gate,project('beta',x),values[prefix+'A_log'],
                                             values[prefix+'dt_bias'].view(heads,d),values[prefix+'recurrent_state'])
                if not torch.isfinite(state).all():
                    raise AssertionError(f'non-finite recurrent state at layer {layer}')
                output = rms(output,values[prefix+'onorm'],eps=1e-5)*project('output_gate',x).view(heads,d).sigmoid()
                output = project('o',output.flatten())
                counts['kda'] += 1
            if not torch.isfinite(output).all():
                raise AssertionError(f'non-finite attention at layer {layer}')
            if layer < p['first_dense']:
                output = project('down',kimi.situ(project('gate',x),project('up',x)))
                counts['dense_ffn'] += 1
            else:
                # Exercise runtime-routed experts with full native quantization.
                ids,weights = kimi.route(values[prefix+'router'].float()@x,values[prefix+'router_bias'],p['topk'],1,1,scaling=1.)
                latent = project('latent_down',x)
                output = torch.zeros(p['latent_hidden'],device=device)
                for slot,expert in enumerate(ids.tolist()):
                    weight = lambda key:unpack_mxfp4(values[prefix+key+'_blocks'][expert],values[prefix+key+'_scales'][expert])
                    output += (weight('down')@kimi.situ(weight('gate')@latent,weight('up')@latent))*weights[slot]
                # This is a local partial; cross-rank reduction is not simulated.
                output = project('latent_up',rms(output,values[prefix+'latent_norm'],eps=1e-5))
                output += project('shared_down',kimi.situ(project('shared_gate',x),project('shared_up',x)))
                counts['routed_moe'] += 1
            if not torch.isfinite(output).all():
                raise AssertionError(f'non-finite FFN at layer {layer}')
        if before != input_digest(values):
            raise AssertionError('local component checks mutated the full native fixture')
        torch.cuda.synchronize(device)
        return {'case':case.to_dict(),'rank':rank,'gpu':torch.cuda.get_device_name(device),
                'verification_scope':'one_rank_input_storage_and_local_components',
                'full_distributed_decode_verified':False,'status':'pass','component_counts':counts,
                'input_bytes':sum(v.numel()*v.element_size() for v in values.values()),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = verify(args.device,args.rank)
    with args.output.open('x') as file:
        json.dump(result,file,indent=2)
    print(json.dumps({key:result[key] for key in ('status','input_bytes','peak_allocated_bytes','component_counts','verification_scope')}))


if __name__ == '__main__':
    main()
