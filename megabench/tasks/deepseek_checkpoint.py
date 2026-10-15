"""Real DeepSeek-V3.2 weights, prompts and prefilled caches for the decode cases.

Cases with ``Case.checkpoint`` set read the pinned FP8 checkpoint instead of
seeded synthetic weights. Synthetic weights at this depth make the step
numerically chaotic (expert routing and sparse selection flip on any rounding
change), so only a bit-exact copy of the reference could pass; real weights
keep the step well conditioned.

The cache prefix is real too. Prompts are fixed windows of the checkpoint's
own chat-encoded transcripts (``encoding/test_output*.txt``), and their
latent, rope and index-key caches come from a one-time distributed prefill
(``python -m megabench.prepare_deepseek_v32``) stored under
``$MEGABENCH_CACHE_DIR``. A trial seed selects the prompt; the decode token
is the prompt's next token at position ``context``.

The checkpoint resolves from ``$MEGABENCH_DEEPSEEK_V32_CHECKPOINT`` (a
snapshot directory) or the Hugging Face cache at the pinned revision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..cases import Case
from .common import rms
from .quantization import FP8_BLOCK, unpack_fp8_weight

CHECKPOINT_ENV = 'MEGABENCH_DEEPSEEK_V32_CHECKPOINT'
CACHE_ENV = 'MEGABENCH_CACHE_DIR'
# Chat transcripts rendered by DeepSeek's own encoder, in corpus order.
CORPUS_FILES = ('encoding/test_output_search_w_date.txt', 'encoding/test_output_search_wo_date.txt',
                'encoding/test_output.txt')
BOS = 0
# Bump when prefill numerics or the cache layout change; old caches are rejected.
PREFILL_VERSION = 1


def repository(case: Case) -> tuple[str, str]:
    model, _, revision = case.checkpoint.partition('@')
    if not model or not revision:
        raise ValueError(f'{case.id}: checkpoint must be "repo@revision"')
    return model, revision


def checkpoint_dir(case: Case) -> Path:
    override = os.environ.get(CHECKPOINT_ENV)
    if override:
        path = Path(override)
    else:
        from huggingface_hub import snapshot_download

        model, revision = repository(case)
        try:
            path = Path(snapshot_download(model, revision=revision, local_files_only=True))
        except Exception as error:
            raise FileNotFoundError(
                f'{case.checkpoint} is not in the Hugging Face cache; download it '
                f'(hf download {model} --revision {revision}) or set {CHECKPOINT_ENV}') from error
    if not (path / 'model.safetensors.index.json').is_file():
        raise FileNotFoundError(f'{path} has no model.safetensors.index.json')
    return path


class Checkpoint:
    """Lazy safetensors reader that loads only the requested rank slice."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.weight_map = json.loads((self.path / 'model.safetensors.index.json').read_text())['weight_map']
        self._files = {}

    def get(self, name: str, device: str, rows=None, cols=None) -> torch.Tensor:
        from safetensors import safe_open

        file = self.weight_map[name]
        if file not in self._files:
            self._files[file] = safe_open(self.path / file, framework='pt', device='cpu')
        tensor = self._files[file].get_slice(name)
        shape = tensor.get_shape()
        if len(shape) == 1:
            return tensor[:].to(device)
        r0, r1 = rows or (0, shape[0])
        c0, c1 = cols or (0, shape[1])
        return tensor[r0:r1, c0:c1].to(device).contiguous()

    def fp8(self, name: str, device: str, rows=None, cols=None) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8 payload and its 128x128 block scales, sliced on block boundaries."""
        block = lambda span: None if span is None else (span[0] // FP8_BLOCK, math.ceil(span[1] / FP8_BLOCK))
        if any(span and (span[0] % FP8_BLOCK) for span in (rows, cols)):
            raise ValueError('FP8 TP shards must align to native 128-channel blocks')
        return (self.get(name + '.weight', device, rows, cols),
                self.get(name + '.weight_scale_inv', device, block(rows), block(cols)))


_OPEN: dict[Path, Checkpoint] = {}


def open_checkpoint(case: Case) -> Checkpoint:
    path = checkpoint_dir(case)
    if path not in _OPEN:
        _OPEN[path] = Checkpoint(path)
    return _OPEN[path]


def _split(case: Case, rank: int, n: int) -> tuple[int, int]:
    return rank * (n // case.tp), (rank + 1) * (n // case.tp)


# Task names of one layer's tensors and their checkpoint names under
# ``model.layers.{layer}.``. FP8 entries also carry ``.weight_scale_inv``.
ATTENTION_FP8 = (('qa', 'self_attn.q_a_proj'), ('qb', 'self_attn.q_b_proj'),
                 ('ka', 'self_attn.kv_a_proj_with_mqa'), ('kb', 'self_attn.kv_b_proj'),
                 ('o', 'self_attn.o_proj'), ('iq', 'self_attn.indexer.wq_b'), ('ik', 'self_attn.indexer.wk'))
ATTENTION_PLAIN = (('qn', 'self_attn.q_a_layernorm.weight'), ('kn', 'self_attn.kv_a_layernorm.weight'),
                   ('inorm', 'self_attn.indexer.k_norm.weight'), ('ibias', 'self_attn.indexer.k_norm.bias'),
                   ('iw', 'self_attn.indexer.weights_proj.weight'))
DENSE_FP8 = (('gate', 'mlp.gate_proj'), ('up', 'mlp.up_proj'), ('down', 'mlp.down_proj'))
SHARED_FP8 = (('shared_gate', 'mlp.shared_experts.gate_proj'), ('shared_up', 'mlp.shared_experts.up_proj'),
              ('shared_down', 'mlp.shared_experts.down_proj'))
ROUTER_PLAIN = (('router', 'mlp.gate.weight'), ('router_bias', 'mlp.gate.e_score_correction_bias'))


def global_weights(case: Case, device: str, *, rank: int) -> dict[str, torch.Tensor]:
    """Vocabulary shards of the embedding and head, and the FP32 norm stacks."""
    ckpt, p = open_checkpoint(case), case.params
    rows = _split(case, rank, p['vocab'])
    stack = lambda name: torch.stack([ckpt.get(f'model.layers.{layer}.{name}.weight', device)
                                      for layer in range(p['layers'])])
    return {'embed': ckpt.get('model.embed_tokens.weight', device, rows),
            'lm_head': ckpt.get('lm_head.weight', device, rows),
            'fnorm': ckpt.get('model.norm.weight', device),
            'ln1': stack('input_layernorm'), 'ln2': stack('post_attention_layernorm')}


def _shards(case: Case, rank: int) -> dict[str, tuple]:
    """TP slice (rows, cols) of every sharded matrix; the rest are replicated."""
    p = case.params
    split = lambda n: _split(case, rank, n)
    dense, expert = split(p['dense_intermediate']), split(p['intermediate'])
    shared = split(p['intermediate'] * p['shared_experts'])
    return {'qb': (split(p['q_heads'] * (p['qk_nope_dim'] + p['qk_rope_dim'])), None),
            'kb': (split(p['q_heads'] * (p['qk_nope_dim'] + p['v_head_dim'])), None),
            'o': (None, split(p['q_heads'] * p['v_head_dim'])),
            'dense_gate': (dense, None), 'dense_up': (dense, None), 'dense_down': (None, dense),
            'gate': (expert, None), 'up': (expert, None), 'down': (None, expert),
            'shared_gate': (shared, None), 'shared_up': (shared, None), 'shared_down': (None, shared)}


def layer_weights(case: Case, layer: int, device: str, *, rank: int) -> dict[str, torch.Tensor]:
    """One layer's rank-local weights under the task's ``l{layer}_*`` names."""
    ckpt, p, prefix = open_checkpoint(case), case.params, f'l{layer}_'
    scope, shards, values = f'model.layers.{layer}.', _shards(case, rank), {}

    def fp8(key: str, name: str, shard: str) -> None:
        rows, cols = shards.get(shard, (None, None))
        values[prefix + key], values[prefix + key + '_scale'] = ckpt.fp8(scope + name, device, rows, cols)

    for key, name in ATTENTION_FP8:
        fp8(key, name, key)
    for key, name in ATTENTION_PLAIN:
        values[prefix + key] = ckpt.get(scope + name, device)
    if layer < p['first_dense']:
        for key, name in DENSE_FP8:
            fp8(key, name, 'dense_' + key)
        return values
    for key, name in ROUTER_PLAIN:
        values[prefix + key] = ckpt.get(scope + name, device)
    for key, name in DENSE_FP8:
        rows, cols = shards[key]
        pairs = [ckpt.fp8(f'{scope}mlp.experts.{expert}.{name.split(".")[-1]}', device, rows, cols)
                 for expert in range(p['experts'])]
        values[prefix + key] = torch.stack([weight for weight, _ in pairs])
        values[prefix + key + '_scale'] = torch.stack([scale for _, scale in pairs])
        del pairs
    for key, name in SHARED_FP8:
        fp8(key, name, key)
    return values


def hf_layer(values: dict[str, torch.Tensor], layer: int, case: Case) -> dict[str, torch.Tensor]:
    """Name one unsharded layer of task inputs as checkpoint tensors."""
    prefix, scope, out = f'l{layer}_', f'model.layers.{layer}.', {}

    def fp8(name: str, key: str, value=None, scale=None) -> None:
        out[scope + name + '.weight'] = values[prefix + key] if value is None else value
        out[scope + name + '.weight_scale_inv'] = values[prefix + key + '_scale'] if scale is None else scale

    out[scope + 'input_layernorm.weight'] = values['ln1'][layer]
    out[scope + 'post_attention_layernorm.weight'] = values['ln2'][layer]
    for key, name in ATTENTION_FP8:
        fp8(name, key)
    for key, name in ATTENTION_PLAIN:
        out[scope + name] = values[prefix + key]
    if layer < case.params['first_dense']:
        for key, name in DENSE_FP8:
            fp8(name, key)
        return out
    for key, name in ROUTER_PLAIN:
        out[scope + name] = values[prefix + key]
    for key, name in SHARED_FP8:
        fp8(name, key)
    for key, name in DENSE_FP8:
        for expert in range(case.params['experts']):
            fp8(f'mlp.experts.{expert}.{name.split(".")[-1]}', key,
                values[prefix + key][expert], values[prefix + key + '_scale'][expert])
    return out


# ---------------------------------------------------------------- prompts


def corpus_tokens(case: Case) -> tuple[list[int], str]:
    """Token ids of the pinned transcripts and a digest of the source text."""
    from tokenizers import Tokenizer

    path = checkpoint_dir(case)
    tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
    digest, tokens = hashlib.sha256(), []
    for name in CORPUS_FILES:
        text = (path / name).read_text(encoding='utf-8')
        digest.update(name.encode() + b'\0' + text.encode())
        tokens += tokenizer.encode(text, add_special_tokens=False).ids
    return tokens, digest.hexdigest()


def prompt_window(case: Case, tokens: list[int], prompt: int) -> list[int]:
    """``context + 1`` tokens: the cached prefix and the decode token.

    Windows start at evenly spaced corpus offsets; each begins with BOS like
    a fresh request.
    """
    count, length = case.params['prompts'], case.params['context'] + 1
    if len(tokens) < length + 1:
        raise ValueError(f'{case.id}: corpus has {len(tokens)} tokens, needs {length + 1}')
    start = prompt * (len(tokens) - length - 1) // max(count - 1, 1)
    return [BOS] + tokens[start + 1:start + length]


def cache_path(case: Case, prompt: int) -> Path:
    _, revision = repository(case)
    root = Path(os.environ.get(CACHE_ENV) or Path.home() / '.cache' / 'megabench')
    return root / 'deepseek-v32' / revision[:12] / f'ctx{case.params["context"]}-prompt{prompt}.pt'


def _prefill_params(case: Case) -> dict[str, int]:
    # Every geometry parameter shapes the caches except the prompt count.
    return {key: value for key, value in sorted(case.params.items()) if key != 'prompts'}


def load_prefix(case: Case, prompt: int) -> dict:
    path = cache_path(case, prompt)
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} is missing; build the prefill caches with torch.distributed.run --nproc-per-node={case.gpus} '
            f'-m megabench.prepare_deepseek_v32 --case {case.id}')
    record = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'checkpoint': case.checkpoint, 'prompt': prompt, 'params': _prefill_params(case),
                'prefill_version': PREFILL_VERSION}
    if {key: record.get(key) for key in expected} != expected:
        stale = sorted(key for key in expected if record.get(key) != expected[key])
        raise ValueError(f'{path} was built with a different {", ".join(stale)}; move it aside and rebuild')
    return record


def build_prompt_caches(jobs: list[tuple[Case, int]], device: str, *, rank: int, group=None,
                        keep_caches: bool = True, log=None) -> list[dict]:
    """Prefill (case, prompt) windows in one pass; return the records ``load_prefix`` reads.

    The cases may differ only in context length and prompt count.
    """
    import time

    case = jobs[0][0]
    shared = lambda other: (other.checkpoint, other.tp, {key: value for key, value in other.params.items()
                                                         if key not in ('context', 'prompts')})
    if any(shared(other) != shared(case) for other, _ in jobs):
        raise ValueError('prefill jobs must share the checkpoint, TP degree and geometry')
    tokens, digest = corpus_tokens(case)
    windows = [prompt_window(other, tokens, prompt) for other, prompt in jobs]
    start = time.perf_counter()
    records = prefill(case, [window[:-1] for window in windows], device,
                      rank=rank, group=group, keep_caches=keep_caches, log=log)
    elapsed = time.perf_counter() - start
    return [record | {'tokens': window, 'checkpoint': other.checkpoint, 'prompt': prompt,
                      'params': _prefill_params(other), 'prefill_version': PREFILL_VERSION,
                      'corpus_sha256': digest, 'prefill_s': elapsed, 'torch': str(torch.__version__)}
            for (other, prompt), window, record in zip(jobs, windows, records)]


def save_prompt_cache(case: Case, prompt: int, record: dict) -> Path:
    """Write a record atomically to the prompt's cache path."""
    path = cache_path(case, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f'.partial-{os.getpid()}')
    torch.save(record, partial)
    os.replace(partial, path)
    return path


# ---------------------------------------------------------------- task inputs


_WEIGHTS: dict[tuple, dict[str, torch.Tensor]] = {}


def make_inputs(case: Case, seed: int, device: str, *, rank: int) -> dict[str, torch.Tensor]:
    """Rank-local real weights (loaded once per process) plus a prompt's caches.

    Weights do not depend on the seed, so trials share one copy; the harness
    still checks every input tensor for mutation after each call.
    """
    key = (case.checkpoint, case.tp, rank, str(device), tuple(sorted(case.params.items())))
    if key not in _WEIGHTS:
        _WEIGHTS.clear()
        weights = global_weights(case, device, rank=rank)
        for layer in range(case.params['layers']):
            weights |= layer_weights(case, layer, device, rank=rank)
        _WEIGHTS[key] = weights
    return dict(_WEIGHTS[key]) | prefix_inputs(case, prompt_record(case, seed), device)


_PREFIX: dict[tuple, dict] = {}


def prompt_record(case: Case, seed: int) -> dict:
    """The seed's prompt and prefilled caches (the last one stays in memory)."""
    prompt = seed % case.params['prompts']
    key = (str(cache_path(case, prompt)), case.checkpoint, case.params['layers'])
    if key not in _PREFIX:
        _PREFIX.clear()
        _PREFIX[key] = load_prefix(case, prompt)
    return _PREFIX[key]


def prefix_inputs(case: Case, record: dict, device: str) -> dict[str, torch.Tensor]:
    values = {'token': torch.tensor(record['tokens'][case.params['context']], device=device)}
    for layer in range(case.params['layers']):
        for name in ('kv_cache', 'pe_cache', 'index_cache'):
            values[f'l{layer}_{name}'] = record[name][layer].to(device)
    return values


# ---------------------------------------------------------------- prefill


def _rope(value: torch.Tensor, positions: torch.Tensor, *, interleaved: bool) -> torch.Tensor:
    from .frontier import rotate

    shape = (-1,) + (1,) * (value.dim() - 1)
    return rotate(value, positions.view(shape).float(), deepseek=True, interleaved=interleaved)


def _route(scores: torch.Tensor, bias: torch.Tensor, p: dict, scaling: float = 2.5):
    """Batched noaux_tc routing: sigmoid, grouped top-k, normalized weights."""
    probability = scores.sigmoid()
    grouped = (probability + bias).view(scores.shape[0], p['router_groups'], -1)
    groups = grouped.topk(2, -1).values.sum(-1).topk(p['router_top_groups'], -1).indices
    keep = torch.zeros(grouped.shape[:2], device=scores.device, dtype=torch.bool).scatter_(1, groups, True)
    indices = grouped.masked_fill(~keep[..., None], -torch.inf).flatten(1).topk(p['topk'], -1).indices
    weights = probability.gather(1, indices)
    return indices, weights / (weights.sum(-1, keepdim=True) + 1e-20) * scaling


@torch.inference_mode()
def prefill(case: Case, sequences: list[list[int]], device: str, *, rank: int, group=None,
            chunk: int = 512, keep_caches: bool = True, log=None) -> list[dict]:
    """Run token sequences through the model; return their caches and next-token stats.

    Follows the official DeepSeek-V3.2 prefill (causal MLA restricted to each
    query's indexer top-k) with the task's TileRT cache formats: BF16 latent,
    rope and Hadamard-rotated index keys. GEMMs use BF16-dequantized FP8
    weights. The caches are model state for the graded decode step, not a
    graded output, so their numerics need not match the decode reference.
    All sequences advance layer by layer, so each layer is read once. The
    caches are replicated across ranks; ``keep_caches=False`` drops them.
    """
    from .frontier import hadamard

    p, tp = case.params, case.tp

    def reduce(value: torch.Tensor) -> torch.Tensor:
        if tp > 1:
            dist.all_reduce(value, group=group)
        return value

    h, heads = p['hidden'], p['q_heads'] // tp
    nr, rd, vd, kr = p['qk_nope_dim'], p['qk_rope_dim'], p['v_head_dim'], p['kv_lora_rank']
    ih, idim = p['index_heads'], p['index_dim']
    scale = (nr + rd) ** -0.5 * (1 + 0.1 * math.log(40)) ** 2
    offset, chunk_vocab = rank * (p['vocab'] // tp), p['vocab'] // tp
    weights = global_weights(case, device, rank=rank)
    swiglu = lambda gate, up: (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    states = []
    for tokens in sequences:
        ids = torch.tensor(tokens, device=device)
        local = (ids >= offset) & (ids < offset + chunk_vocab)
        x = torch.zeros(len(tokens), h, device=device, dtype=torch.float32)
        x[local] = weights['embed'][ids[local] - offset].float()
        states.append({'ids': ids, 'x': reduce(x).to(torch.bfloat16),
                       'caches': {name: [] for name in ('kv_cache', 'pe_cache', 'index_cache')}})

    def attention_block(state: dict, layer: int, values: dict, dense) -> None:
        x, count = state['x'], len(state['ids'])
        positions = torch.arange(count, device=device)
        # Bound the [chunk, index heads, tokens] FP32 score temporary near 1 GiB.
        rows = max(1, min(chunk, (1 << 28) // (max(ih, heads) * count)))
        normalized = rms(x, weights['ln1'][layer], eps=1e-6, weight_in_fp32=True)
        qr = rms(normalized @ dense('qa').T, values['qn'], eps=1e-6, weight_in_fp32=True)
        query = (qr @ dense('qb').T).view(count, heads, nr + rd)
        raw = normalized @ dense('ka').T
        latent = rms(raw[:, :kr], values['kn'], eps=1e-6, weight_in_fp32=True)
        pe = _rope(raw[:, kr:], positions, interleaved=True)
        query_pe = _rope(query[..., nr:], positions, interleaved=True)
        iq = (qr @ dense('iq').T).view(count, ih, idim)
        ik = F.layer_norm((normalized @ dense('ik').T).float(), (idim,),
                          values['inorm'].float(), values['ibias'].float(), eps=1e-6).to(torch.bfloat16)
        iq = hadamard(torch.cat((_rope(iq[..., :rd], positions, interleaved=False), iq[..., rd:]), -1)).float()
        ik = hadamard(torch.cat((_rope(ik[:, :rd], positions, interleaved=False), ik[:, rd:]), -1))
        iw = (normalized.float() @ values['iw'].float().T).to(torch.bfloat16).float() / ih ** 0.5
        for name, value in (('kv_cache', latent), ('pe_cache', pe), ('index_cache', ik)) if keep_caches else ():
            state['caches'][name].append(value.cpu())
        kb = dense('kb').view(heads, nr + vd, kr)
        absorbed = torch.einsum('thd,hdc->thc', query[..., :nr], kb[:, :nr])
        attention = torch.empty(count, heads, vd, device=device, dtype=torch.bfloat16)
        for start in range(0, count, rows):
            end = min(start + rows, count)
            causal = positions[None, :] <= positions[start:end, None]
            index = (F.relu(torch.einsum('qhd,kd->qhk', iq[start:end], ik.float()) / idim ** 0.5)
                     * iw[start:end, :, None]).sum(1).masked_fill(~causal, -torch.inf)
            selected = index.topk(min(p['index_topk'], count), -1).indices
            allowed = torch.zeros_like(causal).scatter_(1, selected, True) & causal
            scores = (torch.einsum('qhc,kc->hqk', absorbed[start:end].float(), latent.float())
                      + torch.einsum('qhr,kr->hqk', query_pe[start:end].float(), pe.float())) * scale
            probability = scores.masked_fill(~allowed[None], -torch.inf).softmax(-1)
            latent_out = torch.einsum('hqk,kc->qhc', probability.to(torch.bfloat16), latent)
            attention[start:end] = torch.einsum('qhc,hdc->qhd', latent_out, kb[:, nr:])
        state['x'] = x + reduce((attention.flatten(1) @ dense('o').T).float()).to(torch.bfloat16)

    def ffn_block(state: dict, layer: int, values: dict, dense) -> None:
        x = state['x']
        normalized = rms(x, weights['ln2'][layer], eps=1e-6, weight_in_fp32=True)
        if layer < p['first_dense']:
            output = (swiglu(normalized @ dense('gate').T, normalized @ dense('up').T) @ dense('down').T).float()
        else:
            indices, routing = _route(normalized.float() @ values['router'].float().T, values['router_bias'], p)
            output = torch.zeros(len(x), h, device=device, dtype=torch.float32)
            for expert in indices.unique().tolist():
                rows, slots = (indices == expert).nonzero(as_tuple=True)
                part = lambda name: unpack_fp8_weight(values[name][expert], values[name + '_scale'][expert]).to(torch.bfloat16)
                hidden = swiglu(normalized[rows] @ part('gate').T, normalized[rows] @ part('up').T)
                output.index_add_(0, rows, (hidden @ part('down').T).float() * routing[rows, slots, None])
            output += (swiglu(normalized @ dense('shared_gate').T, normalized @ dense('shared_up').T)
                       @ dense('shared_down').T).float()
        state['x'] = x + reduce(output).to(torch.bfloat16)

    for layer in range(p['layers']):
        values = {name[len(f'l{layer}_'):]: tensor
                  for name, tensor in layer_weights(case, layer, device, rank=rank).items()}
        dequantized = {}

        def dense(name: str) -> torch.Tensor:
            if name not in dequantized:
                dequantized[name] = unpack_fp8_weight(values[name], values[name + '_scale']).to(torch.bfloat16)
            return dequantized[name]

        for state in states:
            attention_block(state, layer, values, dense)
            ffn_block(state, layer, values, dense)
        del values, dequantized
        if log:
            log(f'layer {layer} done')

    records = []
    for state in states:
        ids, count = state['ids'], len(state['ids'])
        final = rms(state['x'], weights['fnorm'], eps=1e-6, weight_in_fp32=True)
        # Teacher-forced next-token statistics over vocabulary shards (sanity only).
        targets, nll, hits = ids[1:], torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device)
        for start in range(0, count - 1, chunk):
            end = min(start + chunk, count - 1)
            logits = final[start:end].float() @ weights['lm_head'].float().T
            maximum = logits.max(-1).values
            if tp > 1:
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
            total = reduce((logits - maximum[:, None]).exp().sum(-1))
            target = targets[start:end] - offset
            mine = (target >= 0) & (target < chunk_vocab)
            picked = torch.zeros(end - start, device=device)
            picked[mine] = logits[mine, target[mine]]
            picked = reduce(picked)
            nll += (maximum + total.log() - picked).double().sum()
            hits += (picked >= maximum).sum()
        records.append({name: torch.stack(stack) for name, stack in state['caches'].items() if stack} | {
            'mean_nll': float(nll) / (count - 1), 'top1_accuracy': float(hits) / (count - 1)})
    return records
