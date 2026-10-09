"""One Kimi K3 decoder layer with data-parallel attention and expert parallelism.

The full model does not fit eight GPUs, so each case is one mid-block layer
of one attention variant (KDA or gated MLA). Every rank holds replicated
attention, residual, router, latent and shared-expert weights and decodes its
own ``batch`` sequences. Routed experts are owned contiguously by rank
(``experts // ep`` each). Each rank dispatches its BF16 latent rows to expert
owners with an all-to-all, owners apply native MXFP4 experts, and a second
all-to-all returns expert outputs, which the source rank combines in FP32
in routing-slot order. The math per token follows ``kimi.reference``.

Per-sequence tensors sample global rows ``rank*batch ...``, so the union of
all ranks equals one serial rank with ``batch*gpus`` sequences.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..cases import Case
from .common import rms
from .distributed import matrix
from .frontier import route
from .kimi import attention_residual, full_layer, kda_step, mxfp4_matrix, situ
from .parallel import exchange, exchange_counts, initialize
from .quantization import unpack_mxfp4


PER_SEQUENCE = ('prefix', 'kv_cache', 'pe_cache', 'conv_state', 'recurrent_state')


def attention_kind(case: Case) -> str:
    return 'mla' if full_layer(case, case.params['layer']) else 'kda'


def residual_blocks(case: Case) -> int:
    """Completed attention-residual blocks visible to a mid-block layer."""
    p = case.params
    if p['layer'] % p['attn_res_block_size'] == 0 or p['layer'] < p['first_dense']:
        raise ValueError('Kimi layer cases must be mid-block MoE layers')
    return p['layer']//p['attn_res_block_size'] + 1


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None):
    rank = dist.get_rank() if rank is None else rank
    p = case.params
    if case.tp != 1 or case.gpus != case.ep or not 0 <= rank < case.gpus:
        raise ValueError('Kimi layer cases use data-parallel attention and EP over all ranks')
    if p['experts'] % case.ep:
        raise ValueError('expert count must divide EP')
    h, lh, d, ctx = p['hidden'], p['latent_hidden'], p['head_dim'], p['context']
    heads, qdim, batch = p['q_heads'], p['q_heads']*d, p['batch']
    total = batch*case.gpus
    rows = lambda n: (rank*batch*n, (rank+1)*batch*n)
    values = {}

    def weight(name, rows, cols):
        values[name] = matrix(seed, name, rows, cols, cols, device)

    def norm(name, width):
        values[name] = (matrix(seed, name, 1, width, 100, device)[0].float()+1).to(torch.bfloat16)

    def sequence(name, per_sequence, cols, fan_in, shape):
        values[name] = matrix(seed, name, total*per_sequence, cols, fan_in, device,
                              row_range=rows(per_sequence)).view(batch, *shape)

    sequence('blocks', residual_blocks(case), h, 1, (residual_blocks(case), h))
    values['blocks'] = values['blocks'].transpose(0, 1).contiguous()
    sequence('prefix', 1, h, 1, (h,))
    for name in ('ln1', 'ln2', 'attention_res_norm', 'mlp_res_norm'):
        norm(name, h)
    for name in ('attention_res_proj', 'mlp_res_proj'):
        values[name] = matrix(seed, name, 1, h, h, device)[0]
    if attention_kind(case) == 'mla':
        qr, kr, nr, rd, vd = p['q_lora_rank'], p['kv_lora_rank'], p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim']
        weight('qa', qr, h)
        weight('qb', heads*(nr+rd), qr)
        weight('ka', kr+rd, h)
        weight('kb', heads*(nr+vd), kr)
        weight('o', h, heads*vd)
        weight('output_gate', heads*vd, h)
        norm('qn', qr)
        norm('kn', kr)
        sequence('kv_cache', ctx, kr, 1, (ctx, kr))
        sequence('pe_cache', ctx, rd, 1, (ctx, rd))
    else:
        for name in ('q', 'k', 'v', 'output_gate'):
            weight(name, qdim, h)
        weight('o', h, qdim)
        weight('fa', d, h)
        weight('fb', qdim, d)
        weight('beta', heads, h)
        norm('onorm', d)
        values['A_log'] = matrix(seed, 'A_log', 1, heads, 100, device)[0].float()+1
        values['dt_bias'] = matrix(seed, 'dt_bias', 1, qdim, 100, device)[0].float()-2
        k = p['conv_kernel']
        values['conv_weight'] = torch.stack([matrix(seed, name+'_conv', qdim, k, k, device) for name in ('q', 'k', 'v')])
        sequence('conv_state', 3*qdim, k, 1, (3, qdim, k))
        sequence('recurrent_state', heads*d, d, d, (heads, d, d))
        values['recurrent_state'] = values['recurrent_state'].float()
    inter, experts = p['intermediate'], p['experts']
    local = experts//case.ep
    weight('latent_down', lh, h)
    weight('latent_up', h, lh)
    norm('latent_norm', lh)
    weight('router', experts, h)
    values['router_bias'] = matrix(seed, 'router_bias', 1, experts, 100, device)[0].float()
    for name, rows_, cols in (('gate', inter, lh), ('up', inter, lh), ('down', lh, inter)):
        blocks = torch.empty((local, rows_, cols//32, 16), dtype=torch.uint8, device=device)
        scales = torch.empty((local, rows_, cols//32), dtype=torch.uint8, device=device)
        for expert in range(local):
            packed, scale = mxfp4_matrix(seed, f'{name}:{rank*local+expert}', rows_, cols, device)
            blocks[expert].copy_(packed)
            scales[expert].copy_(scale)
        values[name+'_blocks'], values[name+'_scales'] = blocks, scales
    shared = inter*p['shared_experts']
    weight('shared_gate', shared, h)
    weight('shared_up', shared, h)
    weight('shared_down', h, shared)
    return values


def sequence_values(values, row):
    """One sequence's view: per-sequence tensors indexed, weights shared."""
    return values | {key:values[key][row] for key in PER_SEQUENCE if key in values} | {
        'blocks':values['blocks'][:, row]}


def mla(case: Case, values, normalized):
    p = case.params
    heads, kr, nr, rd, vd = p['q_heads'], p['kv_lora_rank'], p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim']
    project = lambda key, value: values[key]@value.to(torch.bfloat16)
    query = project('qb', rms(project('qa', normalized), values['qn'], eps=1e-6)).view(heads, nr+rd)
    raw = project('ka', normalized)
    latent = rms(raw[:kr], values['kn'], eps=1e-6)
    pe = raw[kr:]
    all_latent = torch.cat((values['kv_cache'], latent[None]))
    all_pe = torch.cat((values['pe_cache'], pe[None]))
    weight = values['kb'].view(heads, nr+vd, kr)
    absorbed = torch.einsum('hd,hdc->hc', query[:, :nr], weight[:, :nr])
    probability = ((absorbed.float()@all_latent.float().T+query[:, nr:].float()@all_pe.float().T)/(nr+rd)**0.5).softmax(-1)
    output = torch.einsum('hc,hdc->hd', probability.to(torch.bfloat16)@all_latent, weight[:, nr:]).flatten()
    output = output*project('output_gate', normalized).sigmoid()
    return project('o', output), {'kv_write':latent.to(torch.bfloat16), 'pe_write':pe.to(torch.bfloat16)}


def kda(case: Case, values, normalized):
    p = case.params
    heads, d = p['q_heads'], p['head_dim']
    project = lambda key, value: values[key]@value.to(torch.bfloat16)
    raw = torch.stack([project(key, normalized) for key in ('q', 'k', 'v')])
    conv = torch.cat((values['conv_state'][..., 1:], raw.to(torch.bfloat16)[..., None]), -1)
    q, k, v = F.silu((conv.float()*values['conv_weight'].float()).sum(-1).to(torch.bfloat16)).view(3, heads, d)
    gate = project('fb', project('fa', normalized)).view(heads, d)
    beta = project('beta', normalized)
    output, state = kda_step(q, k, v, gate, beta, values['A_log'], values['dt_bias'].view(heads, d),
                             values['recurrent_state'])
    output = rms(output, values['onorm'], eps=1e-5)*project('output_gate', normalized).view(heads, d).sigmoid()
    return project('o', output.flatten()), {'recurrent_state':state, 'conv_state':conv}


def attention_block(case: Case, values):
    """Attention residual, attention and MLP residual for one sequence."""
    blocks, prefix = list(values['blocks']), values['prefix']
    hidden = attention_residual(prefix, blocks, values['attention_res_norm'], values['attention_res_proj'])
    normalized = rms(hidden, values['ln1'], eps=1e-5)
    attention, writes = (mla if attention_kind(case) == 'mla' else kda)(case, values, normalized)
    prefix = prefix+attention
    hidden = attention_residual(prefix, blocks, values['mlp_res_norm'], values['mlp_res_proj'])
    return prefix, rms(hidden, values['ln2'], eps=1e-5), writes


def experts(values, rows, ids, first):
    """Apply local MXFP4 experts to dispatched BF16 latent rows."""
    output = torch.empty_like(rows)
    for expert in ids.unique().tolist():
        selected = (ids == expert).nonzero().flatten()
        local = expert-first

        def weight(key):
            return unpack_mxfp4(values[key+'_blocks'][local], values[key+'_scales'][local]).to(torch.bfloat16)
        x = rows[selected]
        output[selected] = situ(x@weight('gate').T, x@weight('up').T)@weight('down').T
    return output


def moe(case: Case, values, normalized, *, serial, group):
    """Routed latent experts over EP plus shared experts; rows are sequences.

    Dense projections run one token at a time so that results do not depend
    on how sequences are batched across ranks. Each expert sees its rows in
    global (rank, sequence, slot) order in both the serial and EP layouts.
    """
    p = case.params
    batch, topk = normalized.shape[0], p['topk']
    local = p['experts']//(1 if serial else case.ep)
    rank = 0 if serial else dist.get_rank(group)
    routes = [route(values['router'].float()@token.float(), values['router_bias'], topk, 1, 1, scaling=1.)
              for token in normalized]
    indices = torch.stack([index for index, _ in routes])
    weights = torch.stack([weight for _, weight in routes])
    latent = torch.stack([values['latent_down']@token for token in normalized])
    flat = indices.flatten()
    owner = torch.div(flat, local, rounding_mode='floor')
    # Stable order by owner keeps (sequence, slot) order inside each destination.
    order = torch.sort(owner, stable=True).indices
    rows = latent.repeat_interleave(topk, 0)[order]
    ids = flat[order]
    if serial:
        output = experts(values, rows, ids, 0)
    else:
        send, receive = exchange_counts(owner, group)
        arrived = exchange(rows, send, receive, group)
        arrived_ids = exchange(ids, send, receive, group)
        computed = experts(values, arrived, arrived_ids, rank*local)
        output = exchange(computed, receive, send, group)
    combined = torch.empty_like(output)
    combined[order] = output
    combined = combined.view(batch, topk, -1)
    results = []
    for row, token in enumerate(normalized):
        expert_sum = torch.zeros(p['latent_hidden'], device=token.device)
        for slot in range(topk):
            expert_sum = expert_sum+combined[row, slot].float()*weights[row, slot]
        routed = values['latent_up']@rms(expert_sum.to(torch.bfloat16), values['latent_norm'], eps=1e-5)
        shared = values['shared_down']@situ(values['shared_gate']@token, values['shared_up']@token)
        results.append(routed+shared)
    return torch.stack(results), indices.sort(-1).values


def reference(case: Case, values, *, serial=False):
    if serial and case.gpus != 1:
        raise ValueError('serial validation requires a single-rank geometry')
    group = None if serial else initialize(case).ep_group
    rows = [attention_block(case, sequence_values(values, row)) for row in range(case.params['batch'])]
    prefix = torch.stack([row[0] for row in rows])
    output, expert_ids = moe(case, values, torch.stack([row[1] for row in rows]), serial=serial, group=group)
    writes = {key:torch.stack([row[2][key] for row in rows]) for key in rows[0][2]}
    return {'prefix':prefix+output, **writes, 'expert_ids':expert_ids}
