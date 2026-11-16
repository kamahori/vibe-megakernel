"""DeepSeek-V4.1-Flash DSpark speculative decode iteration at B=1.

One iteration of DSpark speculative decoding (DeepSeek's confidence-scheduled,
semi-autoregressive drafter) after ``context`` cached tokens:

1. **Verify.** The 40-layer backbone runs ``draft_depth + 1`` tokens: the token
   committed by the previous iteration and the ``draft_depth`` DSpark
   proposals, at positions ``context ..``. Each row sees the same state that
   sequential decode would: a 128-token sliding window, plus, on compressed
   layers, the indexer's top-k compressed positions, including compression
   groups completed earlier in the same chunk.
2. **Accept.** The longest proposal prefix that matches the backbone's greedy
   tokens is accepted, and the greedy token after it is the bonus token.
3. **Draft.** The three DSpark stages read the mean hyper-connection stream
   entering backbone layers 37-39 at the last accepted row, attend to a
   128-token window of those projected hiddens plus a block of
   ``[bonus, noise x4]`` embeddings, and propose the next ``draft_depth``
   tokens through a Markov head chained greedily, plus a per-row confidence.

Architecture and numerics follow DeepSeek's reference inference code
(``inference/model.py`` of deepseek-ai/DeepSeek-V4.1-Flash, pinned in
model_specs): 4-stream mHC with Sinkhorn mixing; latent MQA (one 512-wide KV
head shared by 64 query heads, RoPE on the last 64 dims, attention sinks,
grouped low-rank output); layers 0-1 sliding-window only; layers 2-19 add 2:1
compressed KV (sources 2/8/14) and layers 20-39 1:1 compressed KV (source 20),
selected by FP4 indexers (sources 2/8/14/20/24/28/32/36) with candidate-block
prefiltering from layer 20; n-gram Engram lookups before layers 1 and 14; MoE
with 384 FP4 experts/top-6 (sqrt-softplus gate, bias-steered selection) plus an
FP8 shared expert. Dense weights are FP8 E4M3 with 32x32 UE8M0 blocks and
activations are quantized per 32 channels with power-of-two scales. Caches hold
the dequantized values the upstream kernels write: FP8 window KV, FP4 (E4M3
scale per 16) compressed KV, FP4 (UE8M0 per 32) index keys.

Weights and caches are seeded synthetic tensors in the checkpoint's native
formats. Placement follows the upstream TP8 runtime: attention, indexer heads,
vocabulary, and the grouped output projection are split by rank; routed experts
are owned whole by rank ``e // (experts / tp)``; the shared expert, router,
compressors, Engram projection and mHC are replicated; Engram tables are split
by rows. Integer E8M0 scales are stored as uint8 biased exponents.

Rounding departs from upstream in two documented places, both so that the
result does not depend on accumulation or collective order: cross-rank sums
accumulate in FP32 (upstream sums BF16 index scores and embeddings with NCCL),
and top-k ties are broken toward the lower index (upstream's order is
unspecified).
"""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..cases import Case
from .common import rms
from .distributed import matrix, reduce_sum, seed_for
from .parallel import initialize
from .quantization import e8m0_exponent, e8m0_power, pack_mxfp4, unpack_mxfp4

BLOCK = 32                  # FP8 weight/activation and FP4 expert block
KV_FP4_BLOCK = 16           # compressed KV: FP4 with an E4M3 scale per 16
NORM_EPS = 1e-20            # rms_norm_eps, also the mHC and Engram epsilon
HC_EPS = 1e-6
SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5
ROPE_THETA = 10000.0
COMPRESS_ROPE_THETA = 160000.0
YARN_ORIGINAL, YARN_FACTOR, YARN_BETA_FAST, YARN_BETA_SLOW = 65536, 16, 32, 1
ENGRAM_PAD_ID = 2
ENGRAM_CLAMP = 1e-6
DRAFT_TARGETS = 3           # dspark_target_layer_ids are the last three layers
ENGRAM_ROW_CHUNK = 1 << 16  # canonical sampling chunk of the Engram tables
RESIDUAL_DEPTH = 2          # residual-update fan-in multiplier per block (see block_weights)

SCENARIOS = ('raw', 'accept_full', 'accept_two')


# --------------------------------------------------------------------------- layout

def layout(p: dict) -> SimpleNamespace:
    """Per-layer roles derived from the case parameters.

    With the full geometry this reproduces ``compress_ratios``,
    ``kv_source_layer_ids``, ``index_source_layer_ids``,
    ``candidate_source_layer_id``, ``engram_layer_ids`` and
    ``dspark_target_layer_ids`` of the pinned config.
    """
    layers, window, c2 = p['layers'], p['window_layers'], p['c2_layers']
    if window + c2 >= layers or c2 % p['c2_group']:
        raise ValueError('invalid DeepSeek-V4.1 layer layout')
    c1 = window + c2
    ratios = [0]*window + [2]*c2 + [1]*(layers-c1)
    c2_sources = list(range(window, c1, p['c2_group']))
    kv_sources = c2_sources + [c1]
    index_sources = c2_sources + list(range(c1, layers, p['c1_index_group']))
    kv_source = [max((s for s in kv_sources if s <= layer), default=-1) if ratios[layer] else -1
                 for layer in range(layers)]
    index_source = [max((s for s in index_sources if s <= layer), default=-1) if ratios[layer] else -1
                    for layer in range(layers)]
    engram = (p['engram_first'], p['engram_second'])
    if not engram[0] < engram[1] < layers or layers < DRAFT_TARGETS:
        raise ValueError('invalid Engram or DSpark target layers')
    return SimpleNamespace(ratios=ratios, kv_sources=kv_sources, c2_sources=c2_sources, c1_source=c1,
                           index_sources=index_sources, kv_source=kv_source, index_source=index_source,
                           candidate_source=c1, engram=engram,
                           targets=tuple(range(layers-DRAFT_TARGETS, layers)))


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small:
        if n % prime == 0:
            return n == prime
    d, r = n-1, 0
    while d % 2 == 0:
        d, r = d//2, r+1
    for base in small:  # deterministic for n < 3.3e24
        x = pow(base, d, n)
        if x in (1, n-1):
            continue
        for _ in range(r-1):
            x = x*x % n
            if x == n-1:
                break
        else:
            return False
    return True


@lru_cache(None)
def engram_primes(layers: int, ngram: int, heads: int, buckets: int) -> tuple:
    """Bucket moduli [engram layer][n-gram size][head], as ``EngramLayout.from_args``."""
    seen, primes = set(), []
    for _ in range(layers):
        per_ngram = []
        for _ in range(ngram-1):
            sizes, current = [], buckets-1
            for _ in range(heads):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return tuple(primes)


def engram_rows(p: dict) -> tuple[int, int]:
    """Rows of each Engram table: every (n-gram size, head) bucket range end to end."""
    primes = engram_primes(2, p['engram_ngram'], p['engram_heads'], p['engram_buckets'])
    return tuple(sum(sum(sizes) for sizes in layer) for layer in primes)


@lru_cache(None)
def engram_multipliers(layer_ids: tuple[int, ...], ngram: int, compressed_vocab: int) -> tuple:
    """Odd per-(layer, look-back) hash multipliers from ``compute_hash_multipliers``."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rows = []
    for layer in layer_ids:
        values = np.random.default_rng(10007*layer).integers(low=0, high=bound, size=(ngram,), dtype=np.int64)
        rows.append(tuple(int(value)*2+1 for value in values))
    return tuple(rows)


# --------------------------------------------------------------------------- numerics

def _pow2_ceil(value: torch.Tensor) -> torch.Tensor:
    """Exact ``2**ceil(log2(value))`` from the FP32 bits, as the kernels' fast_round_scale."""
    bits = value.float().contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    return (exponent << 23).view(torch.float32)


_FP4_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fp4_round(value: torch.Tensor) -> torch.Tensor:
    """Round FP32 values in [-6, 6] to the nearest E2M1 value, ties to an even code."""
    magnitude = value.abs().contiguous()
    index = torch.bucketize(magnitude, torch.tensor(_FP4_BOUNDS, device=value.device))
    index = index + ((magnitude == 0.75) | (magnitude == 1.75) | (magnitude == 3.5)).to(index.dtype)
    grid = torch.tensor(_FP4_GRID, device=value.device)
    return torch.copysign(grid[index], value)


def fp8_qdq(value: torch.Tensor) -> torch.Tensor:
    """``act_quant(..., inplace=True)``: E4M3 per 32 channels, power-of-two scale."""
    groups = value.float().unflatten(-1, (-1, BLOCK))
    scale = _pow2_ceil(groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)*(1/448.0))
    quant = (groups/scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()*scale
    return quant.flatten(-2).to(value.dtype)


def fp4_qdq(value: torch.Tensor) -> torch.Tensor:
    """``fp4_act_quant(..., 32, inplace=True)``: E2M1 with a UE8M0 scale per 32 (indexer)."""
    groups = value.float().unflatten(-1, (-1, BLOCK))
    scale = _pow2_ceil(groups.abs().amax(-1, keepdim=True).clamp_min(6*2.0**-126)*(1/6.0))
    return (fp4_round((groups/scale).clamp(-6, 6))*scale).flatten(-2).to(value.dtype)


def kv_fp4_qdq(value: torch.Tensor) -> torch.Tensor:
    """``fp4_act_quant(..., 16, inplace=True, E4M3 scale)`` used for compressed KV."""
    groups = value.float().unflatten(-1, (-1, KV_FP4_BLOCK))
    scale = (groups.abs().amax(-1, keepdim=True).clamp_min(6*2.0**-9)/6.0).to(torch.float8_e4m3fn).float()
    return (fp4_round((groups/scale).clamp(-6, 6))*scale).flatten(-2).to(value.dtype)


def fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP32 values of an E4M3 weight with one UE8M0 exponent per 32x32 block."""
    rows, cols = weight.shape
    expanded = e8m0_power(scale).repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return weight.float()*expanded[:rows, :cols]


def fp4_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP32 values of packed E2M1 [rows, cols/2] with one UE8M0 exponent per 32 inputs."""
    rows = weight.shape[0]
    return unpack_mxfp4(weight.view(rows, -1, BLOCK//2), scale)


def linear(value: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Upstream ``linear``: quantized weights take an FP8 activation; others run in their dtype."""
    if scale is None:
        return F.linear(value, weight)
    dense = fp4_weight(weight, scale) if weight.dtype == torch.uint8 else fp8_weight(weight, scale)
    return (fp8_qdq(value).float() @ dense.T).to(torch.bfloat16)


def norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms(value, weight, eps=NORM_EPS, weight_in_fp32=True)


def rope_angles(positions: torch.Tensor, dim: int, compressed: bool) -> torch.Tensor:
    """``precompute_freqs_cis`` rows: YaRN at the compressed theta, plain RoPE otherwise."""
    base = COMPRESS_ROPE_THETA if compressed else ROPE_THETA
    freqs = 1.0/(base**(torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device)/dim))
    if compressed:
        def corrected(rotations):
            return dim*np.log(YARN_ORIGINAL/(rotations*2*np.pi))/(2*np.log(base))
        low = max(int(np.floor(corrected(YARN_BETA_FAST))), 0)
        high = min(int(np.ceil(corrected(YARN_BETA_SLOW))), dim-1)
        ramp = ((torch.arange(dim//2, dtype=torch.float32, device=positions.device)-low)/max(high-low, 1e-3)).clamp(0, 1)
        smooth = 1-ramp
        freqs = freqs/YARN_FACTOR*(1-smooth)+freqs*smooth
    return torch.outer(positions.to(torch.float32), freqs)


def rotate(value: torch.Tensor, angles: torch.Tensor, rope_dim: int, *, inverse: bool = False) -> torch.Tensor:
    """Interleaved-pair rotation of the last ``rope_dim`` channels; angles [rows, rope_dim/2]."""
    head, tail = value[..., :-rope_dim], value[..., -rope_dim:].float().unflatten(-1, (-1, 2))
    while angles.ndim < tail.ndim-1:
        angles = angles.unsqueeze(-2)
    cos, sin = angles.cos(), angles.sin()
    if inverse:
        sin = -sin
    a, b = tail[..., 0], tail[..., 1]
    rotated = torch.stack((a*cos-b*sin, a*sin+b*cos), -1).flatten(-2)
    return torch.cat((head, rotated.to(value.dtype)), -1)


def sparse_attention(query: torch.Tensor, keys: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """``sparse_attn`` for one query row: query [heads, d], keys [n, d] (K = V), FP32 sink."""
    scores = (query.float() @ keys.float().T)*query.shape[-1]**-0.5
    maximum = scores.amax(-1, keepdim=True).clamp_min(-1e30)
    weights = torch.exp(scores-maximum)
    total = weights.sum(-1, keepdim=True)+torch.exp(sink.float()[:, None]-maximum)
    return ((weights.to(torch.bfloat16).float() @ keys.float())/total).to(torch.bfloat16)


def descending(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Top ``count`` indices, ties toward the lower index."""
    return scores.argsort(dim=-1, descending=True, stable=True)[..., :count]


def hc_mixes(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, iterations: int):
    """mHC pre/post/comb coefficients of each row (``hc_split_sinkhorn``)."""
    rows, count = stream.shape[0], stream.shape[1]
    flat = stream.flatten(1).float()
    mixes = (flat @ fn.float().T)*torch.rsqrt(flat.square().mean(-1, keepdim=True)+NORM_EPS)
    pre = torch.sigmoid(mixes[:, :count]*scale[0]+base[:count])+HC_EPS
    post = 2*torch.sigmoid(mixes[:, count:2*count]*scale[1]+base[count:2*count])
    comb = mixes[:, 2*count:].view(rows, count, count)*scale[2]+base[2*count:].view(count, count)
    comb = torch.softmax(comb, -1)+HC_EPS
    comb = comb/(comb.sum(-2, keepdim=True)+HC_EPS)
    for _ in range(iterations-1):
        comb = comb/(comb.sum(-1, keepdim=True)+HC_EPS)
        comb = comb/(comb.sum(-2, keepdim=True)+HC_EPS)
    return pre, post, comb


def hc_pre(stream: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return (pre[..., None]*stream.float()).sum(1).to(stream.dtype)


def hc_post(output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    # Upstream's expression and reduction order: comb[j, k] weighs residual stream j into k.
    mixed = torch.sum(comb[..., None]*residual.float()[:, :, None], dim=1)
    return (post[..., None]*output.float()[:, None]+mixed).to(residual.dtype)


# --------------------------------------------------------------------------- inputs

def fp8_matrix(seed: int, name: str, rows: int, cols: int, device: str, *, row_range=None, col_range=None,
               fan_in: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded E4M3 weight with UE8M0 32x32 block scales, sharded after quantization."""
    r0, r1 = row_range or (0, rows)
    c0, c1 = col_range or (0, cols)
    if any(edge % BLOCK for edge in (r0, r1, c0, c1)):
        raise ValueError('FP8 shards must align to 32-channel blocks')
    source = matrix(seed, name, rows, cols, fan_in or cols, device, row_range=(r0, r1), col_range=(c0, c1))
    blocks = source.float().view((r1-r0)//BLOCK, BLOCK, (c1-c0)//BLOCK, BLOCK)
    exponent = e8m0_exponent(blocks.abs().amax((1, 3)))
    payload = (blocks*e8m0_power(exponent, inverse=True)[:, None, :, None]).to(torch.float8_e4m3fn)
    return payload.view(r1-r0, c1-c0), exponent


def fp4_matrix(seed: int, name: str, rows: int, cols: int, device: str,
               fan_in: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One whole expert matrix: packed E2M1 [rows, cols/2] and UE8M0 scales [rows, cols/32]."""
    generator = torch.Generator(device=device).manual_seed(seed_for(seed, name))
    source = (torch.randn((rows, cols), generator=generator, device=device)*(fan_in or cols)**-0.5).to(torch.bfloat16)
    packed, scales = pack_mxfp4(source)
    return packed.flatten(-2), scales


def kv_rows(seed: int, name: str, rows: int, cols: int, device: str, quantize) -> torch.Tensor:
    """Unit-scale cache rows already on the quantization grid their kernel writes."""
    return quantize(matrix(seed, name, rows, cols, 1, device))


def engram_table(seed: int, name: str, rows: int, dim: int, device: str, row_range: tuple[int, int]):
    """FP8 Engram rows ``row_range`` (zero padded past ``rows``) and UE8M0 scales per 32.

    Bytes are drawn per canonical chunk of rows, so a shard does not depend on
    the TP degree. Codes keep magnitudes in [1/8, 2) before scaling.
    """
    r0, r1 = row_range
    payload = torch.zeros((r1-r0, dim), dtype=torch.uint8, device=device)
    scales = torch.full((r1-r0, dim//BLOCK), 127, dtype=torch.uint8, device=device)
    for start in range(r0//ENGRAM_ROW_CHUNK*ENGRAM_ROW_CHUNK, min(r1, rows), ENGRAM_ROW_CHUNK):
        end = min(start+ENGRAM_ROW_CHUNK, rows)
        generator = torch.Generator(device=device).manual_seed(seed_for(seed, f'{name}:{start}'))
        raw = torch.randint(0, 256, (end-start, dim), generator=generator, device=device, dtype=torch.uint8)
        exponents = torch.randint(122, 125, (end-start, dim//BLOCK), generator=generator, device=device,
                                  dtype=torch.uint8)
        code = (raw & 0x87) | ((((raw >> 3) & 3)+4) << 3)
        lo, hi = max(start, r0), min(end, r1)
        payload[lo-r0:hi-r0] = code[lo-start:hi-start]
        scales[lo-r0:hi-r0] = exponents[lo-start:hi-start]
    return payload.view(torch.float8_e4m3fn), scales


def _check(case: Case, rank: int) -> None:
    p, tp = case.params, case.tp
    if case.ep != 1 or case.gpus != tp or not 0 <= rank < case.gpus:
        raise ValueError('DeepSeek-V4.1 reference requires one TP group covering all ranks')
    for name in ('q_heads', 'o_groups', 'index_heads', 'vocab', 'experts', 'draft_experts'):
        if p[name] % tp:
            raise ValueError(f'{name} must divide TP')
    if p['q_heads'] % p['o_groups'] or p['context'] < max(p['window'], 3):
        raise ValueError('invalid DeepSeek-V4.1 geometry')
    verify = p['draft_depth']+1
    if p['context'] % 2 or verify % 2:
        raise ValueError('2:1 compression groups must start at the cached and verified boundaries')
    if p['candidate_blocks']*p['candidate_block_size'] < p['index_topk']:
        raise ValueError('candidate blocks must cover index_topk positions')
    if not 0 <= p['noise_token'] < p['vocab']:
        raise ValueError('noise token outside the vocabulary')


def make_inputs(case: Case, seed: int, device: str, *, rank: int | None = None) -> dict[str, torch.Tensor]:
    rank = dist.get_rank() if rank is None else rank
    _check(case, rank)
    p, tp, lay = case.params, case.tp, layout(case.params)
    h, d, rd, layers = p['hidden'], p['head_dim'], p['rope_dim'], p['layers']
    heads, groups, qrank, orank = p['q_heads'], p['o_groups'], p['q_lora_rank'], p['o_lora_rank']
    ih, idim, inter = p['index_heads'], p['index_dim'], p['intermediate']
    window, context, streams = p['window'], p['context'], p['hc_mult']
    split = lambda n: (rank*(n//tp), (rank+1)*(n//tp))
    group_width = heads//groups*d
    values: dict[str, torch.Tensor] = {}

    def tokens(name, count):
        generator = torch.Generator(device=device).manual_seed(seed_for(seed, name))
        return torch.randint(0, p['vocab'], (count,), generator=generator, device=device)

    values['tokens'] = tokens('tokens', p['draft_depth']+1)
    values['history_tokens'] = tokens('history_tokens', 3)
    generator = torch.Generator(device=device).manual_seed(seed_for(seed, 'engram_token_map'))
    values['engram_token_map'] = torch.randint(0, p['engram_compressed_vocab'], (p['vocab'],),
                                               generator=generator, device=device)
    values['embed'] = matrix(seed, 'embed', p['vocab'], h, 1, device, row_range=split(p['vocab']))
    values['head'] = matrix(seed, 'head', p['vocab'], h, h, device, row_range=split(p['vocab']))
    values['fnorm'] = (matrix(seed, 'fnorm', 1, h, 100, device)[0].float()+1).to(torch.bfloat16)

    def stack(names: list[str], build):
        built = [build(name) for name in names]
        if isinstance(built[0], tuple):
            return tuple(torch.stack(part) for part in zip(*built))
        return torch.stack(built)

    def block_weights(prefix: str, ids: list[str], experts: int):
        """One mHC/attention/MoE block per id, stacked under ``prefix``.

        Projections back into the residual stream (wo_b, w2) are scaled down
        by the stack depth, as in trained models whose per-layer updates are
        small relative to the stream. Unit-scale random updates make the
        synthetic network chaotic: FP8 activation scales, expert routes and
        sparse selections then flip on BF16-level noise and every layer
        amplifies the difference.
        """
        local = experts//tp
        depth = RESIDUAL_DEPTH*len(ids)
        values[prefix+'attn_norm'] = stack(ids, lambda i: (matrix(seed, i+'attn_norm', 1, h, 100, device)[0].float()+1).to(torch.bfloat16))
        values[prefix+'ffn_norm'] = stack(ids, lambda i: (matrix(seed, i+'ffn_norm', 1, h, 100, device)[0].float()+1).to(torch.bfloat16))
        mix = (2+streams)*streams
        for site in ('hc_attn', 'hc_ffn'):
            values[prefix+site+'_fn'] = stack(ids, lambda i: matrix(seed, i+site+'_fn', mix, streams*h, streams*h, device).float())
            values[prefix+site+'_base'] = stack(ids, lambda i: matrix(seed, i+site+'_base', 1, mix, 100, device)[0].float())
            values[prefix+site+'_scale'] = stack(ids, lambda i: matrix(seed, i+site+'_scale', 1, 3, 100, device)[0].float()+1)
        values[prefix+'attn_sink'] = stack(ids, lambda i: matrix(seed, i+'attn_sink', 1, heads, 1, device,
                                                                 col_range=split(heads))[0].float())
        for name, rows, cols, rr, cr, gain in (('wq_a', qrank, h, None, None, 1),
                                               ('wq_b', heads*d, qrank, split(heads*d), None, 1),
                                               ('wkv', d, h, None, None, 1),
                                               ('wo_a', groups*orank, group_width, split(groups*orank), None, 1),
                                               ('wo_b', h, groups*orank, None, split(groups*orank), depth)):
            values[prefix+name], values[prefix+name+'_scale'] = stack(
                ids, lambda i: fp8_matrix(seed, i+name, rows, cols, device, row_range=rr, col_range=cr,
                                          fan_in=cols*gain))
        values[prefix+'q_norm'] = stack(ids, lambda i: (matrix(seed, i+'q_norm', 1, qrank, 100, device)[0].float()+1).to(torch.bfloat16))
        values[prefix+'kv_norm'] = stack(ids, lambda i: (matrix(seed, i+'kv_norm', 1, d, 100, device)[0].float()+1).to(torch.bfloat16))
        values[prefix+'router'] = stack(ids, lambda i: matrix(seed, i+'router', experts, h, h, device))
        values[prefix+'router_bias'] = stack(ids, lambda i: matrix(seed, i+'router_bias', 1, experts, 100, device)[0].float())
        owned = range(rank*local, (rank+1)*local)
        for name, rows, cols, gain in (('w1', inter, h, 1), ('w3', inter, h, 1), ('w2', h, inter, depth)):
            values[prefix+name], values[prefix+name+'_scale'] = stack(
                ids, lambda i: stack([f'{i}{name}:{e}' for e in owned],
                                     lambda key: fp4_matrix(seed, key, rows, cols, device, cols*gain)))
            values[prefix+'shared_'+name], values[prefix+'shared_'+name+'_scale'] = stack(
                ids, lambda i: fp8_matrix(seed, i+'shared_'+name, rows, cols, device, fan_in=cols*gain))

    block_weights('', [f'l{layer}_' for layer in range(layers)], p['experts'])
    values['window_kv'] = stack([f'l{layer}_' for layer in range(layers)],
                                lambda i: kv_rows(seed, i+'window_kv', window, d, device, fp8_qdq))
    for source in lay.kv_sources:
        prefix, ratio = f'l{source}_', lay.ratios[source]
        values[prefix+'comp_wkv'] = matrix(seed, prefix+'comp_wkv', d, h, h, device)
        if ratio > 1:
            values[prefix+'comp_wgate'] = matrix(seed, prefix+'comp_wgate', d, h, h, device)
        values[prefix+'comp_norm'] = (matrix(seed, prefix+'comp_norm', 1, d, 100, device)[0].float()+1).to(torch.bfloat16)
        values[prefix+'idx_wk'] = matrix(seed, prefix+'idx_wk', idim, d, d, device)
        values[prefix+'idx_knorm'] = (matrix(seed, prefix+'idx_knorm', 1, idim, 100, device)[0].float()+1).to(torch.bfloat16)
        values[prefix+'compress_kv'] = kv_rows(seed, prefix+'compress_kv', context//ratio, d, device, kv_fp4_qdq)
        values[prefix+'index_k'] = kv_rows(seed, prefix+'index_k', context//ratio, idim, device, fp4_qdq)
    for source in lay.index_sources:
        prefix = f'l{source}_'
        values[prefix+'idx_wq_b'], values[prefix+'idx_wq_b_scale'] = fp8_matrix(
            seed, prefix+'idx_wq_b', ih*idim, qrank, device, row_range=split(ih*idim))
        values[prefix+'idx_weights'] = matrix(seed, prefix+'idx_weights', ih, h, h, device, row_range=split(ih))

    rows, edim = engram_rows(p), p['engram_dim']
    cols = (p['engram_ngram']-1)*p['engram_heads']
    for slot, layer in enumerate(lay.engram):
        prefix = f'l{layer}_engram_'
        part = -(-rows[slot]//tp)
        values[prefix+'table'], values[prefix+'table_scale'] = engram_table(
            seed, prefix+'table', rows[slot], edim, device, (rank*part, (rank+1)*part))
        values[prefix+'wkv'], values[prefix+'wkv_scale'] = fp8_matrix(seed, prefix+'wkv', h*(streams+1), cols*edim, device)
        for name in ('q', 'k'):
            values[prefix+name] = (matrix(seed, prefix+name, streams, h, 100, device).float()+1).to(torch.bfloat16)

    ids = [f'draft{stage}_' for stage in range(p['draft_layers'])]
    block_weights('draft_', ids, p['draft_experts'])
    values['draft_window_kv'] = stack(ids, lambda i: kv_rows(seed, i+'window_kv', window, d, device, fp8_qdq))
    values['draft_main_proj'], values['draft_main_proj_scale'] = fp8_matrix(seed, 'draft_main_proj', h, DRAFT_TARGETS*h, device)
    for name in ('draft_main_norm', 'draft_norm'):
        values[name] = (matrix(seed, name, 1, h, 100, device)[0].float()+1).to(torch.bfloat16)
    values['markov_embed'] = matrix(seed, 'markov_embed', p['vocab'], p['markov_rank'], 1, device, row_range=split(p['vocab']))
    values['markov_head'] = matrix(seed, 'markov_head', p['vocab'], p['markov_rank'], p['markov_rank'], device,
                                   row_range=split(p['vocab']))
    values['confidence'] = matrix(seed, 'confidence', 1, h+p['markov_rank'], h+p['markov_rank'], device)[0]
    return values


# --------------------------------------------------------------------------- reference

def _runtime(case: Case, serial: bool) -> SimpleNamespace:
    if serial:
        if (case.gpus, case.tp, case.ep) != (1, 1, 1):
            raise ValueError('serial validation requires unsharded geometry')
        return SimpleNamespace(rank=0, group=None, sum=lambda value: value, maximum=lambda value: value,
                               minimum=lambda value: value)
    ctx = initialize(case)

    def collective(op):
        def apply(value):
            value = value.clone()
            dist.all_reduce(value, op=op, group=ctx.tp_group)
            return value
        return apply
    return SimpleNamespace(rank=ctx.tp_rank, group=ctx.tp_group, sum=lambda value: reduce_sum(value, ctx.tp_group),
                           maximum=collective(dist.ReduceOp.MAX), minimum=collective(dist.ReduceOp.MIN))


def greedy(rt, logits: torch.Tensor, offset: int) -> torch.Tensor:
    """Global argmax of each row of vocabulary-sharded logits, ties to the lower id."""
    maximum = rt.maximum(logits.amax(-1))
    ids = torch.arange(logits.shape[-1], device=logits.device)+offset
    candidates = torch.where(logits == maximum[..., None], ids, torch.iinfo(torch.int64).max)
    return rt.minimum(candidates.amin(-1))


def embed_rows(rt, table: torch.Tensor, ids: torch.Tensor, offset: int) -> torch.Tensor:
    local = ids-offset
    inside = (local >= 0) & (local < table.shape[0])
    rows = table[local.clamp(0, table.shape[0]-1)]*inside[..., None].to(table.dtype)
    return rt.sum(rows)


def engram_ids(case: Case, values, slot: int) -> torch.Tensor:
    """Hashed table rows [verify, (ngram-1)*heads] for every verified position."""
    p, lay = case.params, layout(case.params)
    ngram = p['engram_ngram']
    token_map = values['engram_token_map']
    sequence = token_map[torch.cat((values['history_tokens'], values['tokens']))]
    first = values['history_tokens'].numel()
    pad = token_map[ENGRAM_PAD_ID]
    multipliers = engram_multipliers(lay.engram, ngram, p['engram_compressed_vocab'])[slot]
    primes = engram_primes(2, ngram, p['engram_heads'], p['engram_buckets'])[slot]
    flat = [prime for sizes in primes for prime in sizes]
    offsets = torch.tensor(np.cumsum([0, *flat[:-1]]), device=sequence.device)
    rows = []
    for index in range(values['tokens'].numel()):
        position = p['context']+index
        tokens = [int(sequence[first+index-shift]) if position >= shift else int(pad) for shift in range(ngram)]
        rolling, hashes = tokens[0]*multipliers[0], []
        for size in range(1, ngram):
            rolling ^= tokens[size]*multipliers[size]
            hashes.extend(rolling % prime for prime in primes[size-1])
        rows.append(hashes)
    return torch.tensor(rows, device=sequence.device)+offsets


def engram(case: Case, values, rt, slot: int, layer: int, stream: torch.Tensor) -> torch.Tensor:
    p = case.params
    prefix = f'l{layer}_engram_'
    table, scales = values[prefix+'table'], values[prefix+'table_scale']
    ids = engram_ids(case, values, slot)
    local = ids-rt.rank*table.shape[0]
    inside = (local >= 0) & (local < table.shape[0])
    local = local.clamp(0, table.shape[0]-1)
    rows = table[local].float().unflatten(-1, (-1, BLOCK))*e8m0_power(scales[local])[..., None]
    rows = (rows.flatten(-2).to(torch.bfloat16)*inside[..., None].to(torch.bfloat16))
    rows = rt.sum(rows)
    kv = linear(rows.flatten(-2), values[prefix+'wkv'], values[prefix+'wkv_scale'])
    h, streams = p['hidden'], p['hc_mult']
    key = kv[:, :streams*h].float().unflatten(-1, (streams, h))
    value = kv[:, streams*h:]
    weight = values[prefix+'q'].float()*values[prefix+'k'].float()
    x = stream.float()
    scale = torch.rsqrt(x.square().mean(-1)+NORM_EPS)*torch.rsqrt(key.square().mean(-1)+NORM_EPS)
    dot = (x*weight*key).sum(-1)*scale*h**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(ENGRAM_CLAMP).sqrt(), dot))
    return (x+gate[..., None]*value.float()[:, None]).to(stream.dtype)


class Block:
    """One stacked block (backbone layer or DSpark stage) of ``values``."""

    def __init__(self, case: Case, values, rt, prefix: str, index: int):
        self.p, self.values, self.rt, self.prefix, self.index = case.params, values, rt, prefix, index

    def __call__(self, name: str) -> torch.Tensor:
        return self.values[self.prefix+name][self.index]

    def fp8(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return linear(value, self(name), self(name+'_scale'))

    def query(self, x: torch.Tensor, angles: torch.Tensor):
        p = self.p
        qr = norm(self.fp8(x, 'wq_a'), self('q_norm'))
        q = self.fp8(qr, 'wq_b').view(x.shape[0], -1, p['head_dim'])
        return qr, rotate(q, angles, p['rope_dim'])

    def window_kv(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        kv = norm(self.fp8(x, 'wkv'), self('kv_norm'))
        return fp8_qdq(rotate(kv, angles, self.p['rope_dim']))

    def output(self, attended: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Inverse RoPE, grouped low-rank wo_a (dequantized to BF16 as convert.py) and wo_b."""
        p = self.p
        rows = attended.shape[0]
        attended = rotate(attended, angles, p['rope_dim'], inverse=True)
        local_groups = self('wo_a').shape[0]//p['o_lora_rank']
        grouped = fp8_qdq(attended.reshape(rows, local_groups, -1))
        wo_a = fp8_weight(self('wo_a'), self('wo_a_scale')).to(torch.bfloat16).view(local_groups, p['o_lora_rank'], -1)
        low = torch.einsum('rgd,god->rgo', grouped.float(), wo_a.float()).to(torch.bfloat16)
        return self.rt.sum(self.fp8(low.flatten(1), 'wo_b'))

    def expert(self, x: torch.Tensor, prefix: str, weight=None, expert=None) -> torch.Tensor:
        def project(value, name):
            tensor, scale = self(prefix+name), self(prefix+name+'_scale')
            if expert is not None:
                tensor, scale = tensor[expert], scale[expert]
            return linear(value, tensor, scale)
        gate = project(x, 'w1').float().clamp(max=SWIGLU_LIMIT)
        up = project(x, 'w3').float().clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        hidden = F.silu(gate)*up
        if weight is not None:
            hidden = weight*hidden
        return project(hidden.to(torch.bfloat16), 'w2')

    def moe(self, x: torch.Tensor, topk: int):
        scores = F.softplus(x.float() @ self('router').float().T).sqrt()
        ids = descending(scores+self('router_bias'), topk)
        weights = scores.gather(-1, ids)
        weights = weights/(weights.sum(-1, keepdim=True)+1e-20)*ROUTE_SCALE
        local = self('w1').shape[0]
        first = self.rt.rank*local
        routed = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        for expert in range(first, first+local):
            rows, slots = torch.nonzero(ids == expert, as_tuple=True)
            if rows.numel():
                out = self.expert(x[rows], '', weights[rows, slots, None], expert-first)
                routed[rows] += out.float()
        routed = self.rt.sum(routed)+self.expert(x, 'shared_').float()
        return routed.to(x.dtype), ids

    def forward(self, stream: torch.Tensor, pre_mix: torch.Tensor, attention, topk: int):
        iterations = self.p['hc_sinkhorn_iters']
        attn_pre, attn_post, attn_comb = hc_mixes(stream, self('hc_attn_fn'), self('hc_attn_scale'),
                                                  self('hc_attn_base'), iterations)
        x = norm(hc_pre(stream, pre_mix), self('attn_norm'))
        stream = hc_post(attention(self, x), stream, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = hc_mixes(stream, self('hc_ffn_fn'), self('hc_ffn_scale'),
                                               self('hc_ffn_base'), iterations)
        x = norm(hc_pre(stream, attn_pre), self('ffn_norm'))
        output, ids = self.moe(x, topk)
        return hc_post(output, stream, ffn_post, ffn_comb), ffn_pre, ids


def select_candidates(scores: torch.Tensor, visible: int, blocks: int, size: int) -> torch.Tensor:
    """``select_candidate_blocks`` for one query: keep the best ``blocks`` blocks, pin the newest."""
    width = scores.shape[-1]
    block_scores = F.pad(scores, (0, -width % size), value=-torch.inf).unflatten(-1, (-1, size)).amax(-1)
    count = block_scores.shape[-1]
    block_scores = block_scores.masked_fill(torch.arange(count, device=scores.device) == (visible-1)//size, torch.inf)
    top = descending(block_scores, min(blocks, count))
    keep = torch.zeros(count, device=scores.device, dtype=torch.bool)
    keep[top] = block_scores[top] > -torch.inf
    return keep.repeat_interleave(size)[:width]


def verify(case: Case, values, rt):
    """Backbone over the verified tokens: logits, captured target hiddens and state writes."""
    p, lay = case.params, layout(case.params)
    rows, context, window = p['draft_depth']+1, p['context'], p['window']
    d, rd, ih = p['head_dim'], p['rope_dim'], p['index_heads']
    device = values['tokens'].device
    positions = torch.arange(context, context+rows, device=device)
    offset = rt.rank*values['embed'].shape[0]
    stream = embed_rows(rt, values['embed'], values['tokens'], offset)[:, None].expand(rows, p['hc_mult'], -1).contiguous()
    pre_mix = torch.zeros(rows, p['hc_mult'], device=device)
    pre_mix[:, 0] = 1
    shared = SimpleNamespace(compress_kv=None, index_k=None, topk=None, candidates=None)
    writes = {name: [] for name in ('window', 'compress', 'index_k', 'kv_state', 'score_state', 'experts', 'sparse')}
    captured = []

    def attention(block: Block, x: torch.Tensor) -> torch.Tensor:
        layer, ratio = block.index, lay.ratios[block.index]
        angles = rope_angles(positions, rd, ratio > 0)
        qr, q = block.query(x, angles)
        kv = block.window_kv(x, angles)
        writes['window'].append(kv)
        if layer == lay.kv_source[layer]:
            prefix = f'l{layer}_'
            starts = positions[::ratio]
            group_angles = rope_angles(starts, rd, True)
            if ratio == 1:
                latent = norm(F.linear(x, values[prefix+'comp_wkv']), values[prefix+'comp_norm'])
            else:
                raw = x.float() @ values[prefix+'comp_wkv'].float().T
                score = x.float() @ values[prefix+'comp_wgate'].float().T
                writes['kv_state'].append(raw)
                writes['score_state'].append(score)
                pooled = (raw.view(-1, ratio, d)*score.view(-1, ratio, d).softmax(1)).sum(1)
                latent = norm(pooled.to(torch.bfloat16), values[prefix+'comp_norm'])
            index_key = norm(F.linear(latent, values[prefix+'idx_wk']), values[prefix+'idx_knorm'])
            index_key = fp4_qdq(rotate(index_key, group_angles, rd))
            latent = kv_fp4_qdq(rotate(latent, group_angles, rd))
            writes['compress'].append(latent)
            writes['index_k'].append(index_key)
            shared.compress_kv = torch.cat((values[prefix+'compress_kv'], latent))
            shared.index_k = torch.cat((values[prefix+'index_k'], index_key))
        if ratio and layer == lay.index_source[layer]:
            prefix = f'l{layer}_'
            iq = linear(qr, values[prefix+'idx_wq_b'], values[prefix+'idx_wq_b_scale'])
            iq = fp4_qdq(rotate(iq.view(rows, -1, p['index_dim']), angles, rd))
            head_weights = F.linear(x, values[prefix+'idx_weights'])*(p['index_dim']**-0.5*ih**-0.5)
            # Upstream scores in BF16; heads are summed per rank, ranks in FP32.
            scores = torch.einsum('rhd,td->rht', iq.float(), shared.index_k.float()).to(torch.bfloat16)
            scores = (scores.relu()*head_weights[..., None]).float().sum(1).to(torch.bfloat16)
            scores = rt.sum(scores).float()
            selected, candidates = [], []
            for row in range(rows):
                visible = (context+row+1)//ratio
                row_scores = scores[row, :visible]
                if layer == lay.candidate_source:
                    candidates.append(select_candidates(row_scores, visible, p['candidate_blocks'],
                                                        p['candidate_block_size']))
                elif layer > lay.candidate_source:
                    row_scores = row_scores.masked_fill(~shared.candidates[row][:visible], -torch.inf)
                selected.append(descending(row_scores, min(p['index_topk'], visible)).sort().values)
            if candidates:
                shared.candidates = candidates
            shared.topk = selected
            padded = torch.full((rows, p['index_topk']), -1, device=device, dtype=torch.int64)
            for row, chosen in enumerate(selected):
                padded[row, :chosen.numel()] = chosen
            writes['sparse'].append(padded)
        ring = values['window_kv'][layer]
        attended = []
        for row in range(rows):
            position = context+row
            old = torch.arange(max(0, position-window+1), context, device=device)
            keys = torch.cat((ring[old % window], kv[:row+1]))
            if ratio:
                keys = torch.cat((keys, shared.compress_kv[shared.topk[row]]))
            attended.append(sparse_attention(q[row], keys, block('attn_sink')))
        return block.output(torch.stack(attended), angles)

    engram_slot = {layer: slot for slot, layer in enumerate(lay.engram)}
    for layer in range(p['layers']):
        if layer in engram_slot:
            stream = engram(case, values, rt, engram_slot[layer], layer, stream)
        if layer in lay.targets:
            captured.append(stream.mean(1))
        stream, pre_mix, ids = Block(case, values, rt, '', layer).forward(stream, pre_mix, attention, p['topk'])
        writes['experts'].append(ids)
    final = norm(hc_pre(stream, pre_mix), values['fnorm'])
    logits = final.float() @ values['head'].float().T
    return logits, torch.cat(captured, -1), writes


def draft(case: Case, values, rt, main_hidden: torch.Tensor, accepted: int, bonus: torch.Tensor,
          forced_tokens: torch.Tensor | None = None):
    """Three DSpark stages over ``[bonus, noise...]`` and the greedy Markov chain."""
    p = case.params
    depth, context, window = p['draft_depth'], p['context'], p['window']
    rd, device = p['rope_dim'], bonus.device
    main_rows = main_hidden.shape[0]
    main_positions = torch.arange(context, context+main_rows, device=device)
    start = context+accepted
    positions = torch.arange(start+1, start+1+depth, device=device)
    main_x = norm(linear(main_hidden, values['draft_main_proj'], values['draft_main_proj_scale']),
                  values['draft_main_norm'])
    offset = rt.rank*values['embed'].shape[0]
    ids = torch.full((depth,), p['noise_token'], device=device, dtype=torch.int64)
    ids[0] = bonus
    stream = embed_rows(rt, values['embed'], ids, offset)[:, None].expand(depth, p['hc_mult'], -1).contiguous()
    pre_mix = torch.zeros(depth, p['hc_mult'], device=device)
    pre_mix[:, 0] = 1
    window_writes, expert_ids = [], []

    def attention(block: Block, x: torch.Tensor) -> torch.Tensor:
        main_kv = block.window_kv(main_x, rope_angles(main_positions, rd, False))
        window_writes.append(main_kv)
        ring = values['draft_window_kv'][block.index].index_copy(
            0, main_positions[:accepted+1] % window, main_kv[:accepted+1])
        angles = rope_angles(positions, rd, False)
        _, q = block.query(x, angles)
        keys = torch.cat((ring, block.window_kv(x, angles)))
        attended = torch.stack([sparse_attention(q[row], keys, block('attn_sink')) for row in range(depth)])
        return block.output(attended, angles)

    for stage in range(p['draft_layers']):
        stream, pre_mix, routed = Block(case, values, rt, 'draft_', stage).forward(
            stream, pre_mix, attention, p['draft_topk'])
        expert_ids.append(routed)
    hidden = hc_pre(stream, pre_mix)
    logits = norm(hidden, values['draft_norm']).float() @ values['head'].float().T
    tokens = torch.empty(depth+1, device=device, dtype=torch.int64)
    tokens[0] = bonus
    markov = []
    for row in range(depth):
        embedded = embed_rows(rt, values['markov_embed'], tokens[row], offset)
        markov.append(embedded)
        logits[row] = logits[row]+embedded.float() @ values['markov_head'].float().T
        tokens[row+1] = forced_tokens[row] if forced_tokens is not None else greedy(rt, logits[row], offset)
    confidence = torch.cat((hidden, torch.stack(markov)), -1).float() @ values['confidence'].float()
    return {'draft_tokens': tokens[1:], 'draft_logits': logits, 'draft_confidence': confidence,
            'draft_window_kv_write': torch.stack(window_writes), 'draft_expert_ids': torch.stack(expert_ids)}


def accept(tokens: torch.Tensor, verified: torch.Tensor) -> int:
    """Length of the proposal prefix equal to the backbone's greedy tokens."""
    matches = (tokens[1:] == verified[:-1]).to(torch.int64)
    return int(matches.cumprod(0).sum())


def reference(case: Case, values: dict[str, torch.Tensor], *, serial: bool = False,
              forced: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """One DSpark iteration. ``forced`` replaces greedy picks with given tokens (regrading)."""
    rt = _runtime(case, serial)
    p, lay = case.params, layout(case.params)
    logits, main_hidden, writes = verify(case, values, rt)
    offset = rt.rank*values['head'].shape[0]
    verified = forced['verify_tokens'] if forced else greedy(rt, logits, offset)
    accepted = accept(values['tokens'], verified)
    rows = p['draft_depth']+1
    committed = torch.full((rows,), -1, device=logits.device, dtype=torch.int64)
    committed[:accepted] = values['tokens'][1:accepted+1]
    committed[accepted] = verified[accepted]
    drafted = draft(case, values, rt, main_hidden, accepted, verified[accepted],
                    forced['draft_tokens'] if forced else None)
    c2 = len(lay.c2_sources)
    return {'logits': logits, 'verify_tokens': verified,
            'accepted_count': torch.tensor(accepted, device=logits.device),
            'committed_tokens': committed, **drafted,
            'window_kv_write': torch.stack(writes['window']),
            'compress_kv_write_r2': torch.stack(writes['compress'][:c2]),
            'compress_kv_write_r1': torch.stack(writes['compress'][c2:]),
            'index_k_write_r2': torch.stack(writes['index_k'][:c2]),
            'index_k_write_r1': torch.stack(writes['index_k'][c2:]),
            'compressor_kv_state': torch.stack(writes['kv_state']),
            'compressor_score_state': torch.stack(writes['score_state']),
            'expert_ids': torch.stack(writes['experts']),
            'sparse_indices': torch.stack(writes['sparse'])}


# --------------------------------------------------------------------------- grading hooks

def set_trial_scenario(case: Case, values: dict[str, torch.Tensor], trial: int) -> str:
    """Cycle natural rejection, full acceptance and a two-token prefix.

    Random proposals are rejected at the first row. The controlled scenarios
    zero the shared LM head, so every greedy token is 0 (ties go to the lowest
    id), and choose proposals ``0...0[1]`` for the wanted prefix. The draft
    then follows the Markov head alone, but the backbone, DSpark stages and
    every state write are still graded.
    """
    scenario = SCENARIOS[trial % len(SCENARIOS)]
    if scenario != 'raw':
        accepted = case.params['draft_depth'] if scenario == 'accept_full' else 2
        values['head'].zero_()
        values['tokens'][1:] = 0
        if accepted < case.params['draft_depth']:
            values['tokens'][accepted+1] = 1
    return scenario


TOKEN_OUTPUTS = ('verify_tokens', 'draft_tokens')


def regrade(case: Case, values: dict[str, torch.Tensor], expected: dict, exact: dict, actual: dict,
            oracle, *, serial: bool = False) -> tuple[dict, dict]:
    """Teacher-force the reference and oracle onto the candidate's greedy picks.

    Verify rows and Markov steps can tie within BF16 noise; a different pick
    changes the accepted prefix, bonus token and every later draft. The gate
    then checks each submitted pick against its own row (near-tie rule) and the
    remaining outputs against reference and oracle runs forced onto those
    picks. Every rank takes the same branch: picks are compared to rank 0's.
    """
    distributed = not serial and dist.is_available() and dist.is_initialized() and case.gpus > 1
    device = expected['logits'].device
    usable = isinstance(actual, dict) and all(
        isinstance(actual.get(name), torch.Tensor) and actual[name].shape == expected[name].shape
        and actual[name].dtype == expected[name].dtype and actual[name].device == device for name in TOKEN_OUTPUTS)
    reference_picks = torch.cat([expected[name] for name in TOKEN_OUTPUTS])
    oracle_picks = torch.cat([exact[name].to(reference_picks.dtype) for name in TOKEN_OUTPUTS])
    picks = torch.cat([actual[name] for name in TOKEN_OUTPUTS]).clone() if usable else reference_picks.clone()
    if usable:
        usable = bool(((picks >= 0) & (picks < case.params['vocab'])).all())
    flags = torch.tensor([0 if usable else 1, int(not torch.equal(picks, reference_picks)),
                          int(not torch.equal(picks, oracle_picks))], device=device)
    if distributed:
        first = picks.clone()
        dist.broadcast(first, 0)
        flags[0] += int(not torch.equal(first, picks))
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    if flags[0]:
        return expected, exact
    rows = case.params['draft_depth']+1
    forced = {'verify_tokens': picks[:rows], 'draft_tokens': picks[rows:]}
    options = {'forced': forced, **({'serial': True} if serial else {})}
    if flags[1]:
        expected = reference(case, values, **options)
    if flags[2]:
        exact = oracle(case, values, **options)
    return expected, exact
