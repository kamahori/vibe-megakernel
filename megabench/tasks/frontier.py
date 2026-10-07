"""Synthetic native-FP8 DeepSeek V3.2 and GLM 5.2 distributed decode.

Attention heads and FFN intermediates use TP. Compressed MLA KV and DSA
indexer state are replicated; logits are vocabulary shards. FP8 payloads,
block scales, dynamic activation quantization, sparse selection, and all
collectives are consumed inside reference(), with no cached dequantization.
The pinned model geometries and primary-source revisions are in model_specs.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..cases import Case
from .common import rms
from .distributed import matrix, seed_for, reduce_sum, greedy_token
from .parallel import initialize
from .quantization import (FP8_BLOCK, pack_fp8_activation, pack_fp8_weight,
                           unpack_fp8_activation, unpack_fp8_weight)


def indexer_layer(case: Case, layer: int) -> bool:
    return case.family == 'deepseek_v32_step' or layer < 3 or (layer >= 6 and (layer - 6) % 4 == 0)


def fp8_matrix(seed: int, name: str, rows: int, cols: int, device: str,
               power_of_two: bool, *, row_range=None, col_range=None):
    """Quantize canonical global blocks before selecting rank-local shards."""
    r0, r1 = row_range or (0, rows)
    c0, c1 = col_range or (0, cols)
    if r0 % FP8_BLOCK or c0 % FP8_BLOCK or (r1 != rows and r1 % FP8_BLOCK) or (c1 != cols and c1 % FP8_BLOCK):
        raise ValueError('FP8 TP shards must align to native 128-channel blocks')
    payload = torch.empty((r1 - r0, c1 - c0), device=device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((math.ceil((r1-r0)/FP8_BLOCK), math.ceil((c1-c0)/FP8_BLOCK)),
                         device=device, dtype=torch.float32)
    for start in range(r0, r1, FP8_BLOCK):
        end = min(start + FP8_BLOCK, rows)
        # Sampling is independent of TP, including each global block's scale.
        source = matrix(seed, name, rows, cols, cols, device, row_range=(start, end))
        block, scale = pack_fp8_weight(source, power_of_two=power_of_two)
        payload[start-r0:end-r0].copy_(block[:, c0:c1])
        scales[(start-r0)//FP8_BLOCK].copy_(scale[0, c0//FP8_BLOCK:math.ceil(c1/FP8_BLOCK)])
    return payload, scales


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None) -> dict[str, torch.Tensor]:
    rank = dist.get_rank() if rank is None else rank
    if case.ep != 1 or case.gpus != case.tp or not 0 <= rank < case.gpus:
        raise ValueError('frontier reference requires one TP group covering all ranks')
    p, tp = case.params, case.tp
    h, heads, qrank, krank = p['hidden'], p['q_heads'], p['q_lora_rank'], p['kv_lora_rank']
    nope, rotary, vd = p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim']
    power = case.family == 'deepseek_v32_step'
    for dimension in (heads, p['vocab'], p['intermediate'], p['dense_intermediate']):
        if dimension % tp:
            raise ValueError('heads, FFN intermediates and vocabulary must divide TP')
    split = lambda n: (rank * (n // tp), (rank + 1) * (n // tp))
    values = {'token': torch.tensor(seed_for(seed, 'token') % p['vocab'], device=device)}
    for name in ('embed', 'lm_head'):
        values[name] = matrix(seed, name, p['vocab'], h, h, device, row_range=split(p['vocab']))
    values['fnorm'] = (matrix(seed, 'fnorm', 1, h, 100, device)[0].float() + 1).to(torch.bfloat16)
    values['ln1'] = (matrix(seed, 'ln1', p['layers'], h, 100, device).float() + 1).to(torch.bfloat16)
    values['ln2'] = (matrix(seed, 'ln2', p['layers'], h, 100, device).float() + 1).to(torch.bfloat16)
    for layer in range(p['layers']):
        prefix = f'l{layer}_'
        def weight(name, rows, cols, rr=None, cr=None):
            quant, scale = fp8_matrix(seed, prefix+name, rows, cols, device, power, row_range=rr, col_range=cr)
            values[prefix+name], values[prefix+name+'_scale'] = quant, scale
        for name, width in (('qn', qrank), ('kn', krank)):
            values[prefix+name] = (matrix(seed, prefix+name, 1, width, 100, device)[0].float()+1).to(torch.bfloat16)
        weight('qa', qrank, h)
        weight('qb', heads*(nope+rotary), qrank, rr=split(heads*(nope+rotary)))
        weight('ka', krank+rotary, h)
        weight('kb', heads*(nope+vd), krank, rr=split(heads*(nope+vd)))
        weight('o', h, heads*vd, cr=split(heads*vd))
        latent = matrix(seed, prefix+'latent_cache', p['context'], krank, krank, device)
        values[prefix+'kv_cache'], values[prefix+'kv_cache_scale'] = pack_fp8_activation(latent, power_of_two=power)
        values[prefix+'pe_cache'] = matrix(seed, prefix+'pe_cache', p['context'], rotary, rotary, device)
        if indexer_layer(case, layer):
            ih, idim = p['index_heads'], p['index_dim']
            weight('iq', ih*idim, qrank)
            weight('ik', idim, h)
            values[prefix+'inorm'] = (matrix(seed, prefix+'inorm', 1, idim, 100, device)[0].float()+1).to(torch.bfloat16)
            values[prefix+'ibias'] = matrix(seed, prefix+'ibias', 1, idim, 100, device)[0]
            values[prefix+'iw'] = matrix(seed, prefix+'iw', ih, h, h, device)
            old = matrix(seed, prefix+'index_cache', p['context'], idim, idim, device)
            values[prefix+'index_cache'], values[prefix+'index_cache_scale'] = pack_fp8_activation(old, power_of_two=power)
        if layer < p['first_dense']:
            inter = p['dense_intermediate']
            weight('gate', inter, h, rr=split(inter))
            weight('up', inter, h, rr=split(inter))
            weight('down', h, inter, cr=split(inter))
        else:
            inter, experts = p['intermediate'], p['experts']
            router = matrix(seed, prefix+'router', experts, h, h, device)
            values[prefix+'router'] = router if power else router.float()
            values[prefix+'router_bias'] = matrix(seed, prefix+'router_bias', 1, experts, 100, device)[0].float()
            for name, rows, cols, rr, cr in (('gate', inter, h, split(inter), None),
                                          ('up', inter, h, split(inter), None),
                                          ('down', h, inter, None, split(inter))):
                r, c = (rr[1]-rr[0] if rr else rows), (cr[1]-cr[0] if cr else cols)
                weights = torch.empty((experts, r, c), device=device, dtype=torch.float8_e4m3fn)
                scales = torch.empty((experts, math.ceil(r/128), math.ceil(c/128)), device=device)
                for expert in range(experts):
                    quant, scale = fp8_matrix(seed, prefix+name+f':{expert}', rows, cols, device, power, row_range=rr, col_range=cr)
                    weights[expert].copy_(quant)
                    scales[expert].copy_(scale)
                values[prefix+name], values[prefix+name+'_scale'] = weights, scales
            shared = inter*p['shared_experts']
            weight('shared_gate', shared, h, rr=split(shared))
            weight('shared_up', shared, h, rr=split(shared))
            weight('shared_down', h, shared, cr=split(shared))
    return values


def linear(value: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, power: bool) -> torch.Tensor:
    quant, act_scale = pack_fp8_activation(value.to(torch.bfloat16), power_of_two=power)
    return unpack_fp8_weight(weight, scale) @ unpack_fp8_activation(quant, act_scale)


def rotate(value: torch.Tensor, position: int, *, deepseek: bool, interleaved: bool) -> torch.Tensor:
    dim = value.shape[-1]
    theta = 10000.0 if deepseek else 8_000_000.0
    frequencies = theta ** (torch.arange(0, dim, 2, device=value.device, dtype=torch.float32)/dim)
    inv = 1.0/frequencies
    if deepseek:
        lo = max(0, math.floor(dim*math.log(4096/(32*2*math.pi))/(2*math.log(theta))))
        hi = min(dim-1, math.ceil(dim*math.log(4096/(2*math.pi))/(2*math.log(theta))))
        ramp = ((torch.arange(dim//2, device=value.device)-lo)/max(hi-lo, 0.001)).clamp(0,1)
        extrapolation = 1-ramp
        inv = (1.0/(40*frequencies))*(1-extrapolation) + inv*extrapolation
    angle = position*inv
    a, b = (value[..., 0::2], value[..., 1::2]) if interleaved else value.chunk(2, -1)
    a, b = a*angle.cos()-b*angle.sin(), b*angle.cos()+a*angle.sin()
    return torch.stack((a,b), -1).flatten(-2) if interleaved else torch.cat((a,b), -1)


def hadamard(value: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform used before DeepSeek FP8 indexing."""
    width = value.shape[-1]
    if width & (width-1):
        raise ValueError('Hadamard width must be a power of two')
    output, stride = value.float(), 1
    while stride < width:
        blocks = output.unflatten(-1, (-1, 2, stride))
        left, right = blocks[..., 0, :], blocks[..., 1, :]
        output = torch.stack((left+right, left-right), -2).flatten(-3)
        stride *= 2
    return (output/width**0.5).to(torch.bfloat16)


def route(scores: torch.Tensor, bias: torch.Tensor, topk: int, groups: int, selected_groups: int,
          scaling: float = 2.5) -> tuple[torch.Tensor, torch.Tensor]:
    probability = scores.sigmoid()
    choices = probability+bias
    if groups > 1:
        grouped = choices.view(groups, -1)
        group_scores = grouped.sort(descending=True, stable=True).values[:, :2].sum(-1)
        selected = group_scores.argsort(descending=True, stable=True)[:selected_groups]
        mask = torch.zeros(groups, device=scores.device, dtype=torch.bool)
        mask[selected] = True
        choices = grouped.masked_fill(~mask[:, None], -torch.inf).flatten()
    indices = choices.argsort(descending=True, stable=True)[:topk]
    weights = probability[indices]
    return indices, weights/(weights.sum()+1e-20)*scaling


def reference(case: Case, values: dict[str, torch.Tensor], *, serial: bool = False) -> dict[str, torch.Tensor]:
    if serial and (case.gpus, case.tp, case.ep) != (1, 1, 1):
        raise ValueError('serial validation requires unsharded geometry')
    ctx = SimpleNamespace(tp_rank=0, tp_group=None) if serial else initialize(case)
    sum_value = (lambda value: value) if serial else (lambda value: reduce_sum(value, ctx.tp_group))
    p, tp = case.params, case.tp
    deepseek = case.family == 'deepseek_v32_step'
    power, h, heads = deepseek, p['hidden'], p['q_heads']//tp
    nr, rd, vd, kr = p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim'], p['kv_lora_rank']
    chunk, offset = p['vocab']//tp, ctx.tp_rank*(p['vocab']//tp)
    token = int(values['token'])
    x = values['embed'][token-offset].float() if offset <= token < offset+chunk else torch.zeros(h, device=values['token'].device)
    x = sum_value(x)
    latent_out, latent_scales, pe_out, index_keys, index_scales, indices_out, expert_ids = [], [], [], [], [], [], []
    selected = None
    for layer in range(p['layers']):
        prefix = f'l{layer}_'
        def project(name, value, expert=None):
            weight, scale = values[prefix+name], values[prefix+name+'_scale']
            if expert is not None:
                weight, scale = weight[expert], scale[expert]
            return linear(value, weight, scale, power)
        normalized = rms(x, values['ln1'][layer], eps=1e-6 if deepseek else 1e-5)
        qr = rms(project('qa', normalized), values[prefix+'qn'], eps=1e-6)
        query = project('qb', qr).view(heads, nr+rd)
        raw = project('ka', normalized)
        latent = rms(raw[:kr], values[prefix+'kn'], eps=1e-6)
        latent_payload, latent_scale = pack_fp8_activation(latent.to(torch.bfloat16), power_of_two=power)
        latent = unpack_fp8_activation(latent_payload, latent_scale).to(torch.bfloat16).float()
        pe = rotate(raw[kr:], p['context'], deepseek=deepseek, interleaved=True)
        query_pe = rotate(query[:, nr:], p['context'], deepseek=deepseek, interleaved=True)
        latent_out.append(latent_payload)
        latent_scales.append(latent_scale)
        pe_out.append(pe.to(torch.bfloat16))
        all_latent = torch.cat((unpack_fp8_activation(values[prefix+'kv_cache'], values[prefix+'kv_cache_scale']).to(torch.bfloat16).float(), latent[None]))
        all_pe = torch.cat((values[prefix+'pe_cache'].float(), pe[None]))
        if indexer_layer(case, layer):
            ih, idim = p['index_heads'], p['index_dim']
            iq = project('iq', qr).view(ih, idim)
            ik = F.layer_norm(project('ik', normalized), (idim,), values[prefix+'inorm'].float(), values[prefix+'ibias'].float(), eps=1e-6)
            iq = torch.cat((rotate(iq[:, :rd], p['context'], deepseek=deepseek, interleaved=not deepseek), iq[:, rd:]), -1)
            ik = torch.cat((rotate(ik[:rd], p['context'], deepseek=deepseek, interleaved=not deepseek), ik[rd:]), -1)
            # GLM's published eager indexer works directly in BF16/FP32;
            # DeepSeek's native indexer rotates and quantizes both Q and K.
            if deepseek:
                iq = hadamard(iq.to(torch.bfloat16))
                ik = hadamard(ik.to(torch.bfloat16))
            iq_payload, iq_scale = pack_fp8_activation(iq.to(torch.bfloat16), power_of_two=power)
            ik_payload, ik_scale = pack_fp8_activation(ik.to(torch.bfloat16), power_of_two=power)
            iq = unpack_fp8_activation(iq_payload, iq_scale)
            index_cache = torch.cat((unpack_fp8_activation(values[prefix+'index_cache'], values[prefix+'index_cache_scale']),
                                     unpack_fp8_activation(ik_payload, ik_scale)[None]))
            iw = values[prefix+'iw'].float() @ normalized / ih**0.5
            scores = (F.relu(iq @ index_cache.T / idim**0.5)*iw[:, None]).sum(0)
            selected = scores.argsort(descending=True, stable=True)[:min(p['index_topk'], p['context']+1)].to(torch.int32)
            index_keys.append(ik_payload)
            index_scales.append(ik_scale)
        indices_out.append(selected)
        weight = unpack_fp8_weight(values[prefix+'kb'], values[prefix+'kb_scale']).view(heads, nr+vd, kr)
        absorbed = torch.einsum('hd,hdc->hc', query[:, :nr], weight[:, :nr])
        scores = (absorbed @ all_latent.T + query_pe @ all_pe.T)/(nr+rd)**0.5
        if deepseek:
            scores = scores*(1+0.1*math.log(40))**2
        mask = torch.ones(p['context']+1, device=x.device, dtype=torch.bool)
        mask[selected.long()] = False
        probability = scores.masked_fill(mask[None], -torch.inf).softmax(-1)
        attention_latent = probability @ all_latent
        attention = torch.einsum('hc,hdc->hd', attention_latent, weight[:, nr:]).flatten()
        x = x+sum_value(project('o', attention))
        normalized = rms(x, values['ln2'][layer], eps=1e-6 if deepseek else 1e-5)
        if layer < p['first_dense']:
            output = project('down', F.silu(project('gate', normalized))*project('up', normalized))
            expert_ids.append(torch.full((p['topk'],), -1, device=x.device, dtype=torch.int64))
        else:
            ids, weights = route(values[prefix+'router'].float() @ normalized, values[prefix+'router_bias'],
                                 p['topk'], p['router_groups'], p['router_top_groups'])
            expert_ids.append(ids)
            output = torch.zeros_like(x)
            for slot, expert in enumerate(ids.tolist()):
                hidden = F.silu(project('gate', normalized, expert))*project('up', normalized, expert)
                output = output+project('down', hidden, expert)*weights[slot]
            output = output+project('shared_down', F.silu(project('shared_gate', normalized))*project('shared_up', normalized))
        x = x+sum_value(output)
    logits = values['lm_head'].float() @ rms(x, values['fnorm'], eps=1e-6 if deepseek else 1e-5)
    # Native FP8 cache payloads and their scales must travel together.
    return {'logits': logits, 'next_token': logits.argmax() if serial else greedy_token(logits, offset, ctx.tp_group),
            'kv_write': torch.stack(latent_out), 'kv_scale_write': torch.stack(latent_scales), 'pe_write': torch.stack(pe_out),
            'index_k_write': torch.stack(index_keys), 'index_scale_write': torch.stack(index_scales),
            'sparse_indices': torch.stack(indices_out), 'expert_ids': torch.stack(expert_ids)}
