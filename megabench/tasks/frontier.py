"""Synthetic native-FP8 DeepSeek V3.2 and GLM 5.2 distributed decode.

Attention heads and FFN intermediates use TP. Compressed MLA KV and DSA
indexer state are replicated; logits are vocabulary shards. FP8 payloads,
block scales, dynamic activation quantization, sparse selection, and all
collectives are consumed inside reference(), with no cached dequantization.
The pinned model geometries and primary-source revisions are in model_specs.

Both families follow TileRT 0.1.6's non-MTP B=1 decode numerics
(tile-ai/TileRT), so the TileRT engine can be graded and timed on the same
task. Shared: BF16 latent, rope and index-key caches without scales; FP32
gamma applied before one BF16 rounding; FP32 rotary tables and products; a
BF16 Hadamard-domain indexer; BF16 head weights with FP32 logits.

DeepSeek (CUDA backend): FP8 activations only for q_a/kv_a/indexer wk, YaRN,
grouped routing, power-of-two scales.

GLM (ROCm backend, tilert/models/glm_5_2_rocm): W8A16 attention and dense
FFN GEMVs with FP32 accumulation; FP32 LayerNorm/RoPE/Hadamard index keys and
queries rounded once; MLA probabilities rounded to BF16 before the latent
product; W8A8 routed and shared experts (TileRT's per-128 activation
quantizer); each output projection's rank-local BF16 partial is summed in
FP32 with the residual and rounded once. Choices the public sources cannot
settle are the GLM_* constants below.
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


# Families whose decode numerics follow TileRT 0.1.6 (every frontier family).
TILERT_FAMILIES = frozenset({'deepseek_v32_step', 'glm52_step'})

# GLM-5.2 choices that TileRT's binary-only ROCm kernels leave open. Each is
# the most likely reading of the public goldens; flip after MI350X checks
# (probes/tilert_glm52_probe.py --check-ops).
# Names of attention/dense-FFN projections that quantize their BF16 input to
# FP8 (W8A8). The goldens (unprojo_allreduce.partial_golden) use BF16 inputs.
GLM_FP8_ACTIVATIONS: frozenset[str] = frozenset()
# Dense SwiGLU: FP32 gate/up accumulators, silu(gate)*up in FP32, one BF16
# rounding (as the W8A8 MoE golden). False rounds gate/up to BF16 first.
GLM_DENSE_SWIGLU_FP32 = True
# Sparse MLA rounds the FP32 softmax probabilities to BF16 before the latent
# product (flash_sparse_mla.golden_attention); split-K kernels may not.
GLM_PROBABILITIES_BF16 = True
# q_a and kv_a RMSNorm epsilon (rmsnorm_projq_wqb.EPS, rmsnorm_kv.EPS).
GLM_LATENT_NORM_EPS = 1e-5


# Synthetic GLM weights are conditioned so the 78-layer step is not chaotic.
# With fan-in-scaled embeddings, the embedding is about sqrt(hidden) times
# smaller than one residual branch, so the pre-norm stack is invariant to any
# branch gain and rounding differences (amplified by MoE routing flips) grow
# until the BF16 reference is uncorrelated with its FP32 oracle. GLM
# embeddings therefore have unit element RMS, and the residual-branch output
# projections (attention o, dense and expert down, shared-expert down) are
# sampled with gain 1/(4 * sqrt(2 * layers)), about 0.02 at 78 layers. In a
# scaled-down serial simulation this keeps the reference about 2% (relative
# L2) from the oracle on logits. Trained checkpoints are not chaotic.
GLM_BRANCH_OUTPUTS = frozenset({'o', 'down', 'shared_down'})


def branch_gain(case: Case, name: str) -> float:
    if case.family == 'glm52_step' and name in GLM_BRANCH_OUTPUTS:
        return (2 * case.params['layers']) ** -0.5 / 4
    return 1.0


def indexer_layer(case: Case, layer: int) -> bool:
    return case.family == 'deepseek_v32_step' or layer < 3 or (layer >= 6 and (layer - 6) % 4 == 0)


def fp8_matrix(seed: int, name: str, rows: int, cols: int, device: str,
               power_of_two: bool, *, row_range=None, col_range=None, gain: float = 1.0):
    """Quantize canonical global blocks before selecting rank-local shards.

    ``gain`` scales the sampled matrix before quantization; FP8 block scales
    absorb it, so relative FP8 precision is unchanged.
    """
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
        if gain != 1.0:
            source = (source.float() * gain).to(source.dtype)
        block, scale = pack_fp8_weight(source, power_of_two=power_of_two)
        payload[start-r0:end-r0].copy_(block[:, c0:c1])
        scales[(start-r0)//FP8_BLOCK].copy_(scale[0, c0//FP8_BLOCK:math.ceil(c1/FP8_BLOCK)])
    return payload, scales


def _check_topology(case: Case, rank: int) -> None:
    if case.family not in TILERT_FAMILIES:
        raise ValueError(f'{case.family} is not a frontier family')
    if case.ep != 1 or case.gpus != case.tp or not 0 <= rank < case.gpus:
        raise ValueError('frontier reference requires one TP group covering all ranks')
    p = case.params
    for dimension in (p['q_heads'], p['vocab'], p['intermediate'], p['dense_intermediate']):
        if dimension % case.tp:
            raise ValueError('heads, FFN intermediates and vocabulary must divide TP')


def global_inputs(case: Case, seed: int, device: str, *, rank: int) -> dict[str, torch.Tensor]:
    """Token, embedding/head shards and the per-layer norm stacks."""
    _check_topology(case, rank)
    p, tp, h = case.params, case.tp, case.params['hidden']
    rows = (rank * (p['vocab'] // tp), (rank + 1) * (p['vocab'] // tp))
    values = {'token': torch.tensor(seed_for(seed, 'token') % p['vocab'], device=device)}
    for name in ('embed', 'lm_head'):
        fan_in = 1 if name == 'embed' and case.family == 'glm52_step' else h
        values[name] = matrix(seed, name, p['vocab'], h, fan_in, device, row_range=rows)
    values['fnorm'] = (matrix(seed, 'fnorm', 1, h, 100, device)[0].float() + 1).to(torch.bfloat16)
    values['ln1'] = (matrix(seed, 'ln1', p['layers'], h, 100, device).float() + 1).to(torch.bfloat16)
    values['ln2'] = (matrix(seed, 'ln2', p['layers'], h, 100, device).float() + 1).to(torch.bfloat16)
    return values


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None) -> dict[str, torch.Tensor]:
    rank = dist.get_rank() if rank is None else rank
    values = global_inputs(case, seed, device, rank=rank)
    for layer in range(case.params['layers']):
        values |= layer_inputs(case, seed, layer, device, rank=rank)
    return values


def layer_inputs(case: Case, seed: int, layer: int, device: str, *, rank: int) -> dict[str, torch.Tensor]:
    """One layer's rank-local weights and cache prefix, keyed ``l{layer}_*``.

    With ``gpus=tp=1`` this is the unsharded layer: shards are slices of the
    same canonical blocks, so external engines can consume global tensors.
    """
    _check_topology(case, rank)
    p, tp = case.params, case.tp
    h, heads, qrank, krank = p['hidden'], p['q_heads'], p['q_lora_rank'], p['kv_lora_rank']
    nope, rotary, vd = p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim']
    power = case.family == 'deepseek_v32_step'
    split = lambda n: (rank * (n // tp), (rank + 1) * (n // tp))
    values = {}
    prefix = f'l{layer}_'
    def weight(name, rows, cols, rr=None, cr=None):
        quant, scale = fp8_matrix(seed, prefix+name, rows, cols, device, power, row_range=rr, col_range=cr,
                                  gain=branch_gain(case, name))
        values[prefix+name], values[prefix+name+'_scale'] = quant, scale
    for name, width in (('qn', qrank), ('kn', krank)):
        values[prefix+name] = (matrix(seed, prefix+name, 1, width, 100, device)[0].float()+1).to(torch.bfloat16)
    weight('qa', qrank, h)
    weight('qb', heads*(nope+rotary), qrank, rr=split(heads*(nope+rotary)))
    weight('ka', krank+rotary, h)
    weight('kb', heads*(nope+vd), krank, rr=split(heads*(nope+vd)))
    weight('o', h, heads*vd, cr=split(heads*vd))
    # TileRT keeps BF16 latent, rope and index-key caches without scales.
    values[prefix+'kv_cache'] = matrix(seed, prefix+'latent_cache', p['context'], krank, krank, device)
    values[prefix+'pe_cache'] = matrix(seed, prefix+'pe_cache', p['context'], rotary, rotary, device)
    if indexer_layer(case, layer):
        ih, idim = p['index_heads'], p['index_dim']
        weight('iq', ih*idim, qrank)
        weight('ik', idim, h)
        values[prefix+'inorm'] = (matrix(seed, prefix+'inorm', 1, idim, 100, device)[0].float()+1).to(torch.bfloat16)
        values[prefix+'ibias'] = matrix(seed, prefix+'ibias', 1, idim, 100, device)[0]
        values[prefix+'iw'] = matrix(seed, prefix+'iw', ih, h, h, device)
        values[prefix+'index_cache'] = matrix(seed, prefix+'index_cache', p['context'], idim, idim, device)
    if layer < p['first_dense']:
        inter = p['dense_intermediate']
        weight('gate', inter, h, rr=split(inter))
        weight('up', inter, h, rr=split(inter))
        weight('down', h, inter, cr=split(inter))
    else:
        inter, experts = p['intermediate'], p['experts']
        # TileRT stores BF16 router weights and forms FP32 logits.
        values[prefix+'router'] = matrix(seed, prefix+'router', experts, h, h, device)
        values[prefix+'router_bias'] = matrix(seed, prefix+'router_bias', 1, experts, 100, device)[0].float()
        for name, rows, cols, rr, cr in (('gate', inter, h, split(inter), None),
                                      ('up', inter, h, split(inter), None),
                                      ('down', h, inter, None, split(inter))):
            r, c = (rr[1]-rr[0] if rr else rows), (cr[1]-cr[0] if cr else cols)
            weights = torch.empty((experts, r, c), device=device, dtype=torch.float8_e4m3fn)
            scales = torch.empty((experts, math.ceil(r/128), math.ceil(c/128)), device=device)
            for expert in range(experts):
                quant, scale = fp8_matrix(seed, prefix+name+f':{expert}', rows, cols, device, power,
                                          row_range=rr, col_range=cr, gain=branch_gain(case, name))
                weights[expert].copy_(quant)
                scales[expert].copy_(scale)
            values[prefix+name], values[prefix+name+'_scale'] = weights, scales
        shared = inter*p['shared_experts']
        weight('shared_gate', shared, h, rr=split(shared))
        weight('shared_up', shared, h, rr=split(shared))
        weight('shared_down', h, shared, cr=split(shared))
    return values


def linear(value: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, power: bool,
           *, quantize: bool = True) -> torch.Tensor:
    """FP8 weight GEMV; ``quantize=False`` keeps the BF16 activation (W8A16)."""
    if not quantize:
        return (unpack_fp8_weight(weight, scale) @ value.float()).to(value.dtype)
    quant, act_scale = pack_fp8_activation(value.to(torch.bfloat16), power_of_two=power)
    return (unpack_fp8_weight(weight, scale) @ unpack_fp8_activation(quant, act_scale)).to(value.dtype)


# TileRT quantizes activations only where the input RMSNorm feeds FP8 GEMVs.
DEEPSEEK_FP8_ACTIVATIONS = frozenset({'qa', 'ka', 'ik'})


def tilert_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TileRT ROCm W8A8 activation quantizer (upgate_silu_w8a8.quant_std_blocks).

    Per 128 channels: ``scale = max(amax, 1e-4) * fp32(1/448)`` and
    ``q = x * (1/scale)`` without clamping; returns FP8 payload and FP32 scales.
    """
    groups = value.float().unflatten(-1, (-1, FP8_BLOCK))
    scale = groups.abs().amax(-1).clamp_min(1e-4) * torch.tensor(1/448, dtype=torch.float32, device=value.device)
    payload = (groups * (1/scale)[..., None]).to(torch.float8_e4m3fn)
    return payload.flatten(-2), scale


def rank_partial(terms) -> torch.Tensor:
    """One rank's residual-reduce operand, rounded once to BF16.

    ``terms`` are ``(activation, fp8_weight, scale, factor)``: FP32 GEMVs over
    this rank's K shard, scaled by ``factor`` and summed in FP32. TileRT then
    sums the ranks' BF16 partials and the residual in FP32 and rounds once
    (unprojo_allreduce/moe_down_allreduce ``partial_golden``/``reduce_golden``).
    """
    total = None
    for activation, weight, scale, factor in terms:
        part = (unpack_fp8_weight(weight, scale) @ activation.float()) * factor
        total = part if total is None else total + part
    return total.to(torch.bfloat16)


def rotate(value: torch.Tensor, position: int, *, deepseek: bool, interleaved: bool) -> torch.Tensor:
    """Rotary embedding in FP32 with FP32 tables, rounded once to ``value``'s dtype."""
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
    # TileRT rotates in FP32 with FP32 tables and rounds once.
    a, b = a.float(), b.float()
    cos, sin = angle.cos().float(), angle.sin().float()
    a, b = a*cos-b*sin, b*cos+a*sin
    result = torch.stack((a,b), -1).flatten(-2) if interleaved else torch.cat((a,b), -1)
    return result.to(value.dtype)


def walsh_hadamard(value: torch.Tensor) -> torch.Tensor:
    """Unnormalized natural-order (Sylvester) Walsh-Hadamard transform in FP32."""
    width = value.shape[-1]
    if width & (width-1):
        raise ValueError('Hadamard width must be a power of two')
    output, stride = value.float(), 1
    while stride < width:
        blocks = output.unflatten(-1, (-1, 2, stride))
        left, right = blocks[..., 0, :], blocks[..., 1, :]
        output = torch.stack((left+right, left-right), -2).flatten(-3)
        stride *= 2
    return output


def hadamard(value: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform used before DeepSeek indexing."""
    return (walsh_hadamard(value)/value.shape[-1]**0.5).to(torch.bfloat16)


def glm_index_transform(value: torch.Tensor, position: int, rotary: int) -> torch.Tensor:
    """TileRT GLM indexer rows (ops/rotate.py): FP32 interleaved RoPE on the
    first ``rotary`` dims, Hadamard times width**-0.5, one BF16 rounding."""
    value = value.float()
    value = torch.cat((rotate(value[..., :rotary], position, deepseek=False, interleaved=True),
                       value[..., rotary:]), -1)
    return (walsh_hadamard(value)*value.shape[-1]**-0.5).to(torch.bfloat16)


def layer_norm_fp32(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 LayerNorm with FP32 affine terms (ops/rotate.py rows_golden_f32)."""
    value = value.float()
    mean = value.mean(-1, keepdim=True)
    variance = ((value-mean)**2).mean(-1, keepdim=True)
    return (value-mean)*torch.rsqrt(variance+eps)*weight.float() + bias.float()


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


def route_tilert(scores: torch.Tensor, bias: torch.Tensor, topk: int,
                 scaling: float = 2.5) -> tuple[torch.Tensor, torch.Tensor]:
    """GLM routing as moe_router.select_topk_golden: sigmoid plus FP32 bias,
    top-k with the lowest index winning ties, ``v * (scale / sum(v))``."""
    probability = scores.float().sigmoid()
    indices = (probability+bias.float()).argsort(descending=True, stable=True)[:topk]
    weights = probability[indices]
    return indices, weights*(scaling/weights.sum())


def reference(case: Case, values: dict[str, torch.Tensor], *, serial: bool = False) -> dict[str, torch.Tensor]:
    if serial and (case.gpus, case.tp, case.ep) != (1, 1, 1):
        raise ValueError('serial validation requires unsharded geometry')
    if case.family not in TILERT_FAMILIES:
        raise ValueError(f'{case.family} is not a frontier family')
    ctx = SimpleNamespace(tp_rank=0, tp_group=None) if serial else initialize(case)
    sum_value = (lambda value: value) if serial else (lambda value: reduce_sum(value, ctx.tp_group))
    p, tp = case.params, case.tp
    deepseek = case.family == 'deepseek_v32_step'
    power, h, heads = deepseek, p['hidden'], p['q_heads']//tp
    nr, rd, vd, kr = p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim'], p['kv_lora_rank']
    chunk, offset = p['vocab']//tp, ctx.tp_rank*(p['vocab']//tp)
    token = int(values['token'])
    x = values['embed'][token-offset] if offset <= token < offset+chunk else torch.zeros(h, device=values['token'].device, dtype=torch.bfloat16)
    x = sum_value(x)
    eps = 1e-6 if deepseek else 1e-5
    latent_eps = 1e-6 if deepseek else GLM_LATENT_NORM_EPS
    # TileRT applies the FP32 gamma before its single BF16 rounding.
    norm = lambda value, gamma, eps=eps: rms(value, gamma, eps=eps, weight_in_fp32=True)
    activations = DEEPSEEK_FP8_ACTIVATIONS if deepseek else GLM_FP8_ACTIVATIONS

    def glm_input(name, value):
        """A GLM GEMV input: BF16 (W8A16) unless listed in GLM_FP8_ACTIVATIONS."""
        if name not in GLM_FP8_ACTIVATIONS:
            return value
        return unpack_fp8_activation(*pack_fp8_activation(value.to(torch.bfloat16)))

    def residual_add(residual, partial):
        """TileRT's fused reduce: FP32 sum of BF16 rank partials and the residual, one rounding."""
        return (residual.float()+sum_value(partial.float())).to(residual.dtype)

    latent_out, pe_out, index_keys, indices_out, expert_ids = [], [], [], [], []
    selected = None
    for layer in range(p['layers']):
        prefix = f'l{layer}_'
        def project(name, value, expert=None):
            weight, scale = values[prefix+name], values[prefix+name+'_scale']
            if expert is not None:
                weight, scale = weight[expert], scale[expert]
            return linear(value, weight, scale, power, quantize=name in activations)
        def gemv(name, value, expert=None):
            """FP32 GLM GEMV output, before any BF16 rounding."""
            weight, scale = values[prefix+name], values[prefix+name+'_scale']
            if expert is not None:
                weight, scale = weight[expert], scale[expert]
            return unpack_fp8_weight(weight, scale) @ value.float()
        normalized = norm(x, values['ln1'][layer])
        qr = norm(project('qa', normalized), values[prefix+'qn'], eps=latent_eps)
        query = project('qb', qr).view(heads, nr+rd)
        raw = project('ka', normalized)
        latent = norm(raw[:kr], values[prefix+'kn'], eps=latent_eps)
        latent_out.append(latent)
        pe = rotate(raw[kr:], p['context'], deepseek=deepseek, interleaved=True)
        query_pe = rotate(query[:, nr:], p['context'], deepseek=deepseek, interleaved=True)
        pe_out.append(pe.to(torch.bfloat16))
        all_latent = torch.cat((values[prefix+'kv_cache'], latent[None]))
        all_pe = torch.cat((values[prefix+'pe_cache'], pe[None]))
        if indexer_layer(case, layer):
            ih, idim = p['index_heads'], p['index_dim']
            iq = project('iq', qr).view(ih, idim)
            if deepseek:
                ik = F.layer_norm(project('ik', normalized), (idim,), values[prefix+'inorm'], values[prefix+'ibias'], eps=1e-6)
                iq = torch.cat((rotate(iq[:, :rd], p['context'], deepseek=deepseek, interleaved=False), iq[:, rd:]), -1)
                ik = torch.cat((rotate(ik[:rd], p['context'], deepseek=deepseek, interleaved=False), ik[rd:]), -1)
                # TileRT rotates both sides by a Hadamard transform and keeps
                # BF16 query, key cache and head weights (no FP8 indexer).
                iq = hadamard(iq.to(torch.bfloat16))
                ik = hadamard(ik.to(torch.bfloat16))
                index_cache = torch.cat((values[prefix+'index_cache'], ik[None])).float()
                iw = (values[prefix+'iw'].float() @ normalized.float()).to(torch.bfloat16).float() / ih**0.5
                scores = (F.relu(iq.float() @ index_cache.T / idim**0.5)*iw[:, None]).sum(0)
            else:
                # TileRT ROCm (ops/rotate.py, ops/sparse_index.py): the BF16
                # key is LayerNormed, rotated and Hadamard-transformed in FP32
                # and rounded once, as is each query head; BF16 head weights;
                # FP32 logits with both scales inside the ReLU.
                ik = layer_norm_fp32(project('ik', normalized), values[prefix+'inorm'], values[prefix+'ibias'], 1e-6)
                ik = glm_index_transform(ik, p['context'], rd)
                iq = glm_index_transform(iq, p['context'], rd)
                index_cache = torch.cat((values[prefix+'index_cache'], ik[None])).float()
                iw = (values[prefix+'iw'].float() @ normalized.float()).to(torch.bfloat16)
                scores = (F.relu(iq.float() @ index_cache.T * (ih**-0.5*idim**-0.5))*iw.float()[:, None]).sum(0)
            index_keys.append(ik)
            # Lowest index wins ties (TileRT's radix select leaves them open).
            selected = scores.argsort(descending=True, stable=True)[:min(p['index_topk'], p['context']+1)].to(torch.int32)
        indices_out.append(selected)
        if deepseek:
            weight = unpack_fp8_weight(values[prefix+'kb'], values[prefix+'kb_scale']).to(x.dtype).view(heads, nr+vd, kr)
            absorbed = torch.einsum('hd,hdc->hc', query[:, :nr], weight[:, :nr])
        else:
            # Absorptions keep the FP32-dequantized kv_b block (W8A16, BF16 out).
            weight = unpack_fp8_weight(values[prefix+'kb'], values[prefix+'kb_scale']).view(heads, nr+vd, kr)
            absorbed = torch.einsum('hd,hdc->hc', query[:, :nr].float(), weight[:, :nr]).to(x.dtype)
        scores = (absorbed.float() @ all_latent.float().T + query_pe.float() @ all_pe.float().T)/(nr+rd)**0.5
        if deepseek:
            scores = scores*(1+0.1*math.log(40))**2
        mask = torch.ones(p['context']+1, device=x.device, dtype=torch.bool)
        mask[selected.long()] = False
        probability = scores.masked_fill(mask[None], -torch.inf).softmax(-1)
        if deepseek:
            attention_latent = probability.to(x.dtype) @ all_latent
            attention = torch.einsum('hc,hdc->hd', attention_latent, weight[:, nr:]).flatten()
            x = x+sum_value(project('o', attention))
        else:
            if GLM_PROBABILITIES_BF16:
                probability = probability.to(x.dtype).float()
            attention_latent = (probability @ all_latent.float()).to(x.dtype)
            attention = torch.einsum('hc,hdc->hd', attention_latent.float(), weight[:, nr:]).to(x.dtype).flatten()
            x = residual_add(x, rank_partial([(glm_input('o', attention), values[prefix+'o'], values[prefix+'o_scale'], 1.0)]))
        normalized = norm(x, values['ln2'][layer])
        if deepseek:
            # TileRT forms silu(gate)*up in FP32 and rounds once.
            swiglu = lambda gate, up: (F.silu(gate.float())*up.float()).to(gate.dtype)
            if layer < p['first_dense']:
                output = project('down', swiglu(project('gate', normalized), project('up', normalized)))
                expert_ids.append(torch.full((p['topk'],), -1, device=x.device, dtype=torch.int64))
            else:
                ids, weights = route(values[prefix+'router'].float() @ normalized.float(), values[prefix+'router_bias'],
                                     p['topk'], p['router_groups'], p['router_top_groups'])
                expert_ids.append(ids)
                # Experts accumulate in FP32. TileRT rounds each weighted routed
                # expert's TP partial to BF16; that rank-local rounding is not
                # TP-invariant, so it is left to the tolerance band.
                output = torch.zeros_like(x, dtype=torch.float32)
                for slot, expert in enumerate(ids.tolist()):
                    hidden = swiglu(project('gate', normalized, expert), project('up', normalized, expert))
                    output = output+project('down', hidden, expert).float()*weights[slot]
                output = output+project('shared_down', swiglu(project('shared_gate', normalized), project('shared_up', normalized))).float()
            x = x+sum_value(output).to(x.dtype)
        elif layer < p['first_dense']:
            gate, up = gemv('gate', glm_input('gate', normalized)), gemv('up', glm_input('up', normalized))
            if not GLM_DENSE_SWIGLU_FP32:
                gate, up = gate.to(x.dtype).float(), up.to(x.dtype).float()
            mid = (gate*torch.sigmoid(gate)*up).to(x.dtype)
            x = residual_add(x, rank_partial([(glm_input('down', mid), values[prefix+'down'], values[prefix+'down_scale'], 1.0)]))
            expert_ids.append(torch.full((p['topk'],), -1, device=x.device, dtype=torch.int64))
        else:
            ids, weights = route_tilert(values[prefix+'router'].float() @ normalized.float(), values[prefix+'router_bias'], p['topk'])
            expert_ids.append(ids)
            # W8A8 experts (fused_moe_w8a8 goldens): FP32 gate/up, mid rounded
            # to BF16 and re-quantized; slot 0 is the shared expert (weight 1).
            activation = unpack_fp8_activation(*tilert_quantize(normalized))
            def expert_term(gate, up, down, expert, factor):
                gate, up = gemv(gate, activation, expert), gemv(up, activation, expert)
                mid = (gate*torch.sigmoid(gate)*up).to(x.dtype)
                weight, scale = values[prefix+down], values[prefix+down+'_scale']
                if expert is not None:
                    weight, scale = weight[expert], scale[expert]
                return unpack_fp8_activation(*tilert_quantize(mid)), weight, scale, factor
            terms = [expert_term('shared_gate', 'shared_up', 'shared_down', None, 1.0)]
            terms += [expert_term('gate', 'up', 'down', expert, weights[slot]) for slot, expert in enumerate(ids.tolist())]
            x = residual_add(x, rank_partial(terms))
    final = norm(x, values['fnorm'])
    # TileRT keeps FP32 logits from the BF16 head GEMV.
    logits = values['lm_head'].float() @ final.float()
    return {'logits': logits, 'next_token': logits.argmax() if serial else greedy_token(logits, offset, ctx.tp_group),
            'kv_write': torch.stack(latent_out), 'pe_write': torch.stack(pe_out),
            'index_k_write': torch.stack(index_keys),
            'sparse_indices': torch.stack(indices_out), 'expert_ids': torch.stack(expert_ids)}
