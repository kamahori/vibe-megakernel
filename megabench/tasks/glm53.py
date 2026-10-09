"""Synthetic GLM-5.3-Flash text decode: mHC streams, KDA, NoPE MLA/DSA and FP8 MoE.

Every fourth layer is MLA with a DSA k-pool indexer; the rest are KDA linear
attention. Four hyper-connection residual streams are mixed by Sinkhorn-
projected matrices around each sublayer. MLA projections, dense and expert
FFNs carry native FP8 E4M3 blocks with dynamic activation quantization; KDA,
indexer, kv_b, routers and mHC parameters stay unquantized, as in the
checkpoint's exclusions; storage dtypes follow its safetensors headers. TP shards attention/KDA heads, FFN intermediates and
the vocabulary; the indexer and mHC are replicated. The latent MLA cache and
indexer cache are BF16. KDA state is FP32 in the upstream [head, key, value]
layout. The pinned geometry and primary-source revisions are in model_specs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..cases import Case
from .common import rms
from .distributed import matrix, seed_for, reduce_sum, greedy_token
from .frontier import fp8_matrix, linear, route
from .parallel import initialize

EPS = 1e-5
HC_EPS = 1e-6
SWIGLU_LIMIT = 10.0
GATE_LOWER_BOUND = -5.0


def mla_layer(layer: int) -> bool:
    return (layer+1) % 4 == 0


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None) -> dict[str, torch.Tensor]:
    rank = dist.get_rank() if rank is None else rank
    if case.ep != 1 or case.gpus != case.tp or not 0 <= rank < case.gpus:
        raise ValueError('GLM-5.3 reference requires one TP group covering all ranks')
    p, tp = case.params, case.tp
    h, streams, d = p['hidden'], p['hc_mult'], p['linear_head_dim']
    heads, qdim = p['q_heads'], p['linear_heads']*d
    for dimension in (heads, p['linear_heads'], p['vocab'], p['intermediate'], p['dense_intermediate']):
        if dimension % tp:
            raise ValueError('heads, FFN intermediates and vocabulary must divide TP')
    split = lambda n: (rank*(n//tp), (rank+1)*(n//tp))
    values = {'token': torch.tensor(seed_for(seed, 'token') % p['vocab'], device=device)}
    for name in ('embed', 'lm_head'):
        values[name] = matrix(seed, name, p['vocab'], h, h, device, row_range=split(p['vocab']))
    values['fnorm'] = (matrix(seed, 'fnorm', 1, h, 100, device)[0].float()+1).to(torch.bfloat16)
    for name in ('ln1', 'ln2'):
        values[name] = (matrix(seed, name, p['layers'], h, 100, device).float()+1).to(torch.bfloat16)
    mix = (2+streams)*streams
    for site in ('attn_hc', 'ffn_hc'):
        values[site+'_fn'] = torch.stack([matrix(seed, f'l{layer}_{site}_fn', mix, streams*h, streams*h, device)
                                          for layer in range(p['layers'])])
        values[site+'_base'] = matrix(seed, site+'_base', p['layers'], mix, 100, device).float()
        values[site+'_scale'] = matrix(seed, site+'_scale', p['layers'], 3, 100, device).float()+1
    for layer in range(p['layers']):
        prefix = f'l{layer}_'
        def bf16(name, rows, cols, rr=None, cr=None, fan_in=None):
            values[prefix+name] = matrix(seed, prefix+name, rows, cols, fan_in or cols, device, row_range=rr, col_range=cr)
        def fp8(name, rows, cols, rr=None, cr=None):
            values[prefix+name], values[prefix+name+'_scale'] = fp8_matrix(
                seed, prefix+name, rows, cols, device, False, row_range=rr, col_range=cr)
        def norm(name, width):
            values[prefix+name] = (matrix(seed, prefix+name, 1, width, 100, device)[0].float()+1).to(torch.bfloat16)
        if mla_layer(layer):
            qrank, krank, nope, vd = p['q_lora_rank'], p['kv_lora_rank'], p['qk_nope_dim'], p['v_head_dim']
            ih, idim, pool = p['index_heads'], p['index_dim'], p['index_kpool']
            fp8('qa', qrank, h)
            fp8('qb', heads*nope, qrank, rr=split(heads*nope))
            fp8('ka', krank, h)
            bf16('kb', heads*(nope+vd), krank, rr=split(heads*(nope+vd)))
            fp8('o', h, heads*vd, cr=split(heads*vd))
            norm('qn', qrank)
            norm('kn', krank)
            # Cached rows are RMS-normalized latents and LayerNorm keys: unit scale.
            bf16('kv_cache', p['context'], krank, fan_in=1)
            bf16('iq', ih*idim, qrank)
            bf16('ik', idim, h)
            bf16('iw', ih, h)
            bf16('igate', idim, h)
            bf16('iape', pool, idim, fan_in=100)
            norm('inorm', idim)
            bf16('ibias', 1, idim, fan_in=100)
            values[prefix+'ibias'] = values[prefix+'ibias'][0]
            bf16('index_k_cache', p['context'], idim, fan_in=1)
            bf16('index_gate_cache', p['context'], idim, fan_in=1)
        else:
            for name in ('q', 'k', 'v'):
                bf16(name, qdim, h, rr=split(qdim))
            bf16('o', h, qdim, cr=split(qdim))
            bf16('fa', d, h)
            bf16('fb', qdim, d, rr=split(qdim))
            bf16('beta', p['linear_heads'], h, rr=split(p['linear_heads']))
            bf16('ga', d, h)
            bf16('gb', qdim, d, rr=split(qdim))
            norm('onorm', d)
            values[prefix+'A_log'] = matrix(seed, prefix+'A_log', 1, p['linear_heads'], 100, device,
                                            col_range=split(p['linear_heads']))[0].float()+1
            values[prefix+'dt_bias'] = matrix(seed, prefix+'dt_bias', 1, qdim, 100, device, col_range=split(qdim))[0].float()-2
            conv, states = [], []
            for name in ('q', 'k', 'v'):
                conv.append(matrix(seed, prefix+name+'_conv', qdim, p['conv_kernel'], p['conv_kernel'], device, row_range=split(qdim)))
                states.append(matrix(seed, prefix+name+'_conv_cache', qdim, p['conv_kernel'], 1, device, row_range=split(qdim)))
            values[prefix+'conv_weight'] = torch.stack(conv)
            values[prefix+'conv_state'] = torch.stack(states)
            state = matrix(seed, prefix+'recurrent', p['linear_heads']*d, d, d, device, row_range=split(p['linear_heads']*d))
            values[prefix+'recurrent_state'] = state.view(p['linear_heads']//tp, d, d).float()
        if layer < p['first_dense']:
            inter = p['dense_intermediate']
            fp8('gate', inter, h, rr=split(inter))
            fp8('up', inter, h, rr=split(inter))
            fp8('down', h, inter, cr=split(inter))
        else:
            inter, experts = p['intermediate'], p['experts']
            values[prefix+'router'] = matrix(seed, prefix+'router', experts, h, h, device)
            values[prefix+'router_bias'] = matrix(seed, prefix+'router_bias', 1, experts, 100, device)[0].float()
            for name, rows, cols, rr, cr in (('gate', inter, h, split(inter), None),
                                             ('up', inter, h, split(inter), None),
                                             ('down', h, inter, None, split(inter))):
                r, c = (rr[1]-rr[0] if rr else rows), (cr[1]-cr[0] if cr else cols)
                weights = torch.empty((experts, r, c), device=device, dtype=torch.float8_e4m3fn)
                scales = torch.empty((experts, -(-r//128), -(-c//128)), device=device)
                for expert in range(experts):
                    quant, scale = fp8_matrix(seed, prefix+name+f':{expert}', rows, cols, device, False, row_range=rr, col_range=cr)
                    weights[expert].copy_(quant)
                    scales[expert].copy_(scale)
                values[prefix+name], values[prefix+name+'_scale'] = weights, scales
            shared = inter*p['shared_experts']
            fp8('shared_gate', shared, h, rr=split(shared))
            fp8('shared_up', shared, h, rr=split(shared))
            fp8('shared_down', h, shared, cr=split(shared))
    return values


def hyper_connection(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, iterations: int):
    """mHC pre/post/comb weights and the collapsed sublayer input."""
    count = streams.shape[0]
    flat = streams.flatten().float()
    flat = flat*torch.rsqrt(flat.square().mean()+EPS)
    pre, post, comb = (fn.float() @ flat).split([count, count, count*count])
    pre_bias, post_bias, comb_bias = base.split([count, count, count*count])
    pre = torch.sigmoid(pre*scale[0]+pre_bias)+HC_EPS
    post = 2*torch.sigmoid(post*scale[1]+post_bias)
    comb = torch.softmax(comb.view(count, count)*scale[2]+comb_bias.view(count, count), -1)+HC_EPS
    comb = comb/(comb.sum(-2, keepdim=True)+HC_EPS)
    for _ in range(iterations-1):
        comb = comb/(comb.sum(-1, keepdim=True)+HC_EPS)
        comb = comb/(comb.sum(-2, keepdim=True)+HC_EPS)
    collapsed = (pre[:, None]*streams).sum(0).to(streams.dtype)
    return post, comb, collapsed


def hyper_combine(output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    dtype = residual.dtype
    return post.to(dtype)[:, None]*output[None]+comb.to(dtype).T @ residual


def kda_step(query, key, value, gate, beta, state):
    """Upstream recurrent KDA: L2-normalized q/k, per-key decay, [head, key, value] state."""
    query, key, value = query.float(), key.float(), value.float()
    query = query/torch.sqrt(query.square().sum(-1, keepdim=True)+1e-6)/query.shape[-1]**0.5
    key = key/torch.sqrt(key.square().sum(-1, keepdim=True)+1e-6)
    updated = state.float()*gate.exp()[..., None]
    delta = (value-(updated*key[..., None]).sum(-2))*beta.float()[:, None]
    updated = updated+key[..., None]*delta[:, None, :]
    return (updated*query[..., None]).sum(-2), updated


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate.clamp(max=SWIGLU_LIMIT))*up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)


def pool_scores(p, values, prefix, normalized, query_residual, index_key, index_gate):
    """DSA k-pool indexer scores over complete pools, plus the selected-token mask."""
    ih, idim, pool = p['index_heads'], p['index_dim'], p['index_kpool']
    length = p['context']+1
    keys = torch.cat((values[prefix+'index_k_cache'], index_key[None]))
    gates = torch.cat((values[prefix+'index_gate_cache'], index_gate[None]))
    pools = length//pool
    logits = gates[:pools*pool].view(pools, pool, idim).float()+values[prefix+'iape'].float()
    weights = logits.softmax(1).to(keys.dtype)
    pooled = (weights*keys[:pools*pool].view(pools, pool, idim)).sum(1)
    query = (values[prefix+'iq'] @ query_residual).view(ih, idim)
    scores = F.relu(query.float() @ pooled.float().T * idim**-0.5)
    head_weights = (values[prefix+'iw'] @ normalized).float()*ih**-0.5
    scores = head_weights @ scores
    chosen = scores.argsort(descending=True, stable=True)[:min(p['index_topk']//pool, pools)]
    selected = torch.zeros(length, device=scores.device, dtype=torch.bool)
    selected[:pools*pool].view(pools, pool)[chosen] = True
    selected[pools*pool:] = True  # the incomplete tail pool is always attended
    return scores, selected


def reference(case: Case, values: dict[str, torch.Tensor], *, serial: bool = False) -> dict[str, torch.Tensor]:
    if serial and (case.gpus, case.tp, case.ep) != (1, 1, 1):
        raise ValueError('serial validation requires unsharded geometry')
    ctx = SimpleNamespace(tp_rank=0, tp_group=None) if serial else initialize(case)
    sum_value = (lambda value: value) if serial else (lambda value: reduce_sum(value, ctx.tp_group))
    p, tp = case.params, case.tp
    h, d, heads, kheads = p['hidden'], p['linear_head_dim'], p['q_heads']//tp, p['linear_heads']//tp
    nope, vd, kr = p['qk_nope_dim'], p['v_head_dim'], p['kv_lora_rank']
    chunk, offset = p['vocab']//tp, ctx.tp_rank*(p['vocab']//tp)
    token = int(values['token'])
    x = values['embed'][token-offset] if offset <= token < offset+chunk else torch.zeros(h, device=values['token'].device, dtype=torch.bfloat16)
    streams = sum_value(x)[None].expand(p['hc_mult'], h).contiguous()
    latent_out, index_keys, index_gates, index_scores, masks = [], [], [], [], []
    recurrent, conv_out, expert_ids = [], [], []
    for layer in range(p['layers']):
        prefix = f'l{layer}_'
        def fp8(name, value, expert=None):
            weight, scale = values[prefix+name], values[prefix+name+'_scale']
            if expert is not None:
                weight, scale = weight[expert], scale[expert]
            return linear(value, weight, scale, False)
        def hc(site):
            return hyper_connection(streams, values[site+'_fn'][layer], values[site+'_base'][layer],
                                    values[site+'_scale'][layer], p['hc_sinkhorn_iters'])
        post, comb, collapsed = hc('attn_hc')
        normalized = rms(collapsed, values['ln1'][layer], eps=EPS)
        if mla_layer(layer):
            query_residual = rms(fp8('qa', normalized), values[prefix+'qn'], eps=EPS)
            query = fp8('qb', query_residual).view(heads, nope)
            latent = rms(fp8('ka', normalized), values[prefix+'kn'], eps=EPS)
            index_key = F.layer_norm(values[prefix+'ik'] @ normalized, (p['index_dim'],),
                                     values[prefix+'inorm'], values[prefix+'ibias'], eps=1e-6)
            index_gate = values[prefix+'igate'] @ normalized
            scores, selected = pool_scores(p, values, prefix, normalized, query_residual, index_key, index_gate)
            latent_out.append(latent)
            index_keys.append(index_key)
            index_gates.append(index_gate)
            index_scores.append(scores)
            masks.append(selected)
            all_latent = torch.cat((values[prefix+'kv_cache'], latent[None]))
            weight = values[prefix+'kb'].view(heads, nope+vd, kr)
            absorbed = torch.einsum('hd,hdc->hc', query, weight[:, :nope])
            logits = (absorbed.float() @ all_latent.float().T)*nope**-0.5
            probability = logits.masked_fill(~selected[None], -torch.inf).softmax(-1)
            attention = torch.einsum('hc,hdc->hd', probability.to(x.dtype) @ all_latent, weight[:, nope:]).flatten()
            output = sum_value(fp8('o', attention))
        else:
            raw = torch.stack([values[prefix+key] @ normalized for key in ('q', 'k', 'v')])
            conv = torch.cat((values[prefix+'conv_state'][..., 1:], raw[..., None]), -1)
            q, k, v = F.silu((conv.float()*values[prefix+'conv_weight'].float()).sum(-1)).to(x.dtype).view(3, kheads, d)
            forget = (values[prefix+'fb'] @ (values[prefix+'fa'] @ normalized)).float()+values[prefix+'dt_bias']
            gate = GATE_LOWER_BOUND*torch.sigmoid(values[prefix+'A_log'].exp()[:, None]*forget.view(kheads, d))
            beta = torch.sigmoid(values[prefix+'beta'] @ normalized)
            core, state = kda_step(q, k, v, gate, beta, values[prefix+'recurrent_state'])
            recurrent.append(state)
            conv_out.append(conv)
            output_gate = (values[prefix+'gb'] @ (values[prefix+'ga'] @ normalized)).view(kheads, d)
            core = core.to(x.dtype).float()
            core = core*torch.rsqrt(core.square().mean(-1, keepdim=True)+EPS)*values[prefix+'onorm'].float()
            core = (core*torch.sigmoid(output_gate.float())).to(x.dtype)
            output = sum_value(values[prefix+'o'] @ core.flatten())
        streams = hyper_combine(output, streams, post, comb)
        post, comb, collapsed = hc('ffn_hc')
        normalized = rms(collapsed, values['ln2'][layer], eps=EPS)
        if layer < p['first_dense']:
            output = sum_value(fp8('down', swiglu(fp8('gate', normalized), fp8('up', normalized))))
            expert_ids.append(torch.full((p['topk'],), -1, device=x.device, dtype=torch.int64))
        else:
            ids, weights = route(values[prefix+'router'].float() @ normalized.float(), values[prefix+'router_bias'], p['topk'], 1, 1)
            expert_ids.append(ids)
            routed = torch.zeros(h, device=x.device, dtype=torch.float32)
            for slot, expert in enumerate(ids.tolist()):
                hidden = swiglu(fp8('gate', normalized, expert), fp8('up', normalized, expert))
                routed = routed+fp8('down', hidden, expert).float()*weights[slot]
            shared = fp8('shared_down', swiglu(fp8('shared_gate', normalized), fp8('shared_up', normalized)))
            output = sum_value(routed+shared.float()).to(x.dtype)
        streams = hyper_combine(output, streams, post, comb)
    logits = values['lm_head'] @ rms(streams.mean(0), values['fnorm'], eps=EPS)
    return {'logits': logits.float(), 'next_token': logits.argmax() if serial else greedy_token(logits, offset, ctx.tp_group),
            'kv_write': torch.stack(latent_out), 'index_k_write': torch.stack(index_keys),
            'index_gate_write': torch.stack(index_gates), 'index_scores': torch.stack(index_scores),
            'sparse_mask': torch.stack(masks), 'recurrent_state': torch.stack(recurrent),
            'conv_state': torch.stack(conv_out), 'expert_ids': torch.stack(expert_ids)}
