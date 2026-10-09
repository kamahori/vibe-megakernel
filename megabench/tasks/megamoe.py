"""Routed experts of one DeepSeek-V4-Pro MoE layer under expert parallelism.

The task mirrors DeepGEMM's ``fp8_fp4_mega_moe`` (dispatch, linear1, SwiGLU,
FP8 requantization, linear2, combine) without the shared expert. Routing is
an input. Each rank holds ``tokens`` FP8 rows with per-32-channel E8M0 scales
and owns experts ``[rank*E/ep, (rank+1)*E/ep)`` as MXFP4 weights (E2M1 codes
with one E8M0 scale per 32 input channels).

Per routed (token, slot) row, following DeepGEMM's fused epilogue:

1. ``l1 = x @ [gate | up].T`` with exact FP8/FP4 operands, FP32
   accumulation and a BF16 result.
2. ``gate = min(gate, limit)``, ``up = clamp(up, -limit, limit)`` in BF16;
   ``a = (gate / (1 + exp(-gate)) * up) * topk_weight`` in FP32.
3. ``a`` is requantized to E4M3 with one E8M0 scale per 32 channels.
4. ``l2 = a @ down.T`` with FP32 accumulation and a BF16 result.
5. The source rank sums its token's ``topk`` BF16 rows in FP32, in slot
   order, and rounds once to BF16.

Every expert processes its rows in global (rank, token, slot) order in both
the serial and EP layouts, so per-expert GEMMs see identical operands and
row counts, and the EP result equals the serial result bit for bit.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..cases import Case
from .distributed import ROW_BLOCK, matrix, seed_for
from .parallel import exchange, exchange_counts, initialize
from .quantization import MX_BLOCK, pack_fp8_activation, pack_mxfp4, unpack_fp8_activation, unpack_mxfp4


ACT_BLOCK = 32             # DeepGEMM mega MoE recipe (1, 1, 32): per-token, per-32-channel UE8M0
SWIGLU_LIMIT = 10.0        # DeepSeek-V4-Pro config.json swiglu_limit
ROUTED_SCALING = 2.5       # DeepSeek-V4-Pro config.json routed_scaling_factor
POPULARITY_SIGMA = 0.4     # log-normal expert popularity: hottest of 384 is about 3x the mean
L1_GAIN = 4                # linear1 output std; about 1% of gate/up values reach the clamp


def limit(case: Case) -> float:
    return float(case.params.get('swiglu_limit', SWIGLU_LIMIT))


def random_rows(seed: int, name: str, rows: int, cols: int, device: str, *,
                row_range: tuple[int, int], normal: bool) -> torch.Tensor:
    """FP32 uniform or normal rows sampled in canonical 128-row blocks."""
    r0, r1 = row_range
    output = torch.empty((r1-r0, cols), device=device)
    for start in range(r0//ROW_BLOCK*ROW_BLOCK, r1, ROW_BLOCK):
        end = min(start+ROW_BLOCK, rows)
        generator = torch.Generator(device=device).manual_seed(seed_for(seed, f'{name}:{start}'))
        sample = torch.randn if normal else torch.rand
        block = sample((end-start, cols), generator=generator, device=device)
        lo, hi = max(start, r0), min(end, r1)
        output[lo-r0:hi-r0].copy_(block[lo-start:hi-start])
    return output


def route(seed: int, total: int, experts: int, topk: int, device: str, row_range: tuple[int, int]):
    """Seeded, mildly imbalanced routing for global tokens ``row_range``.

    Distinct experts per token are drawn without replacement from a
    log-normal popularity (Gumbel top-k). Weights follow DeepSeek-V4's
    normalized sqrt-softplus gate scaled by ``routed_scaling_factor``.
    """
    popularity = POPULARITY_SIGMA*random_rows(seed, 'expert_popularity', 1, experts, device,
                                              row_range=(0, 1), normal=True)[0]
    uniform = random_rows(seed, 'route_noise', total, experts, device, row_range=row_range, normal=False)
    gumbel = -torch.log(-torch.log(uniform.clamp(1e-12, 1-1e-7)))
    indices = (popularity+gumbel).topk(topk, -1).indices
    logits = random_rows(seed, 'route_logits', total, topk, device, row_range=row_range, normal=True)
    score = torch.nn.functional.softplus(logits).sqrt()
    return indices, score/score.sum(-1, keepdim=True)*ROUTED_SCALING


def mxfp4_expert(seed: int, name: str, rows: int, cols: int, fan_in: float, device: str):
    """Packed E2M1 codes [rows, cols/32, 16] and E8M0 scales [rows, cols/32]."""
    blocks = torch.empty((rows, cols//MX_BLOCK, MX_BLOCK//2), dtype=torch.uint8, device=device)
    scales = torch.empty((rows, cols//MX_BLOCK), dtype=torch.uint8, device=device)
    for start in range(0, rows, ROW_BLOCK):
        end = min(start+ROW_BLOCK, rows)
        blocks[start:end], scales[start:end] = pack_mxfp4(
            matrix(seed, name, rows, cols, fan_in, device, row_range=(start, end)))
    return blocks, scales


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None):
    rank = dist.get_rank() if rank is None else rank
    p = case.params
    tokens, experts, topk, hidden, inter = p['tokens'], p['experts'], p['topk'], p['hidden'], p['intermediate']
    if case.tp != 1 or case.gpus != case.ep or not 0 <= rank < case.gpus:
        raise ValueError('MegaMoE cases use expert parallelism over all ranks')
    if experts % case.ep or hidden % ACT_BLOCK or inter % ACT_BLOCK or not 0 < topk <= experts:
        raise ValueError('invalid MegaMoE geometry')
    if p.get('act_block', ACT_BLOCK) != ACT_BLOCK:
        raise ValueError(f'MegaMoE activations use {ACT_BLOCK}-channel scale blocks')
    total, rows = tokens*case.gpus, (rank*tokens, (rank+1)*tokens)
    values = {}
    activation = matrix(seed, 'x', total, hidden, 1, device, row_range=rows)
    values['x'], values['x_scales'] = pack_fp8_activation(activation, block=ACT_BLOCK, scale_format='e8m0')
    values['topk_idx'], values['topk_weights'] = route(seed, total, experts, topk, device, rows)
    local = experts//case.ep
    for name, out, cols, fan_in in (('l1', 2*inter, hidden, hidden/L1_GAIN**2), ('l2', hidden, inter, inter)):
        blocks = torch.empty((local, out, cols//MX_BLOCK, MX_BLOCK//2), dtype=torch.uint8, device=device)
        scales = torch.empty((local, out, cols//MX_BLOCK), dtype=torch.uint8, device=device)
        for expert in range(local):
            blocks[expert], scales[expert] = mxfp4_expert(seed, f'{name}:{rank*local+expert}', out, cols,
                                                          fan_in, device)
        values[name+'_blocks'], values[name+'_scales'] = blocks, scales
    return values


def swiglu(case: Case, l1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Clamped SwiGLU on BF16 linear1 outputs, weighted in FP32."""
    gate, up = l1.chunk(2, dim=-1)
    bound = limit(case)
    gate = gate.clamp(max=bound).float()
    up = up.clamp(-bound, bound).float()
    return gate/(1+torch.exp(-gate))*up*weights[:, None]


def requantize(value: torch.Tensor) -> torch.Tensor:
    """E4M3 with per-32-channel E8M0 scales, dequantized for linear2."""
    payload, scales = pack_fp8_activation(value, block=ACT_BLOCK, scale_format='e8m0')
    return unpack_fp8_activation(payload, scales, block=ACT_BLOCK).to(torch.bfloat16)


def experts(case: Case, values, rows, scales, local_ids, weights):
    """Apply local experts to dispatched FP8 rows; returns weighted BF16 rows.

    Rows are grouped per expert in arrival order, then one GEMM per expert
    covers all its rows.
    """
    output = torch.empty((rows.shape[0], case.params['hidden']), dtype=torch.bfloat16, device=rows.device)
    order = torch.sort(local_ids, stable=True).indices
    counts = torch.bincount(local_ids, minlength=values['l1_blocks'].shape[0]).tolist()
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        selected = order[start:start+count]
        start += count
        x = unpack_fp8_activation(rows[selected], scales[selected], block=ACT_BLOCK).to(torch.bfloat16)
        w1 = unpack_mxfp4(values['l1_blocks'][expert], values['l1_scales'][expert]).to(torch.bfloat16)
        w2 = unpack_mxfp4(values['l2_blocks'][expert], values['l2_scales'][expert]).to(torch.bfloat16)
        output[selected] = requantize(swiglu(case, x@w1.T, weights[selected]))@w2.T
    return output


def reference(case: Case, values, *, serial=False):
    if serial and case.gpus != 1:
        raise ValueError('serial validation requires a single-rank geometry')
    group = None if serial else initialize(case).ep_group
    p = case.params
    tokens, topk = values['x'].shape[0], p['topk']
    local = p['experts']//(1 if serial else case.ep)
    rank = 0 if serial else dist.get_rank(group)
    flat = values['topk_idx'].flatten()
    owner = torch.div(flat, local, rounding_mode='floor')
    # Stable order by owner keeps (token, slot) order inside each destination.
    order = torch.sort(owner, stable=True).indices
    source = torch.div(order, topk, rounding_mode='floor')
    # FP8 travels as bytes: collectives need not support float8 dtypes.
    payload = (values['x'].view(torch.uint8)[source], values['x_scales'][source],
               flat[order], values['topk_weights'].flatten()[order])
    if serial:
        output = experts(case, values, payload[0].view(torch.float8_e4m3fn), payload[1], payload[2], payload[3])
    else:
        send, receive = exchange_counts(owner, group)
        rows, scales, ids, weights = (exchange(value, send, receive, group) for value in payload)
        computed = experts(case, values, rows.view(torch.float8_e4m3fn), scales, ids-rank*local, weights)
        output = exchange(computed, receive, send, group)
    slots = torch.empty_like(output)
    slots[order] = output
    slots = slots.view(tokens, topk, -1)
    combined = torch.zeros((tokens, slots.shape[-1]), dtype=torch.float32, device=slots.device)
    for slot in range(topk):
        combined = combined+slots[:, slot].float()
    return {'y':combined.to(torch.bfloat16)}
