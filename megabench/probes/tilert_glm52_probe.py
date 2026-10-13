"""Grade and time TileRT's GLM-5.2 ROCm decode engine on a MegaBench case.

TileRT (tile-ai/TileRT 0.1.6, ``tilert/models/glm_5_2_rocm``) is one host
process driving eight MI350X/MI355X GPUs, so it cannot run as a per-rank
MegaBench submission. This probe runs it on the case's seeded synthetic
weights, cache prefix and token instead:

``reference`` (MegaBench venv, ``torch.distributed.run`` on eight ranks)
    Builds the case inputs and stores the BF16 reference outputs, the FP32
    oracle outputs, and a digest of replicated inputs, one file per rank.

``tilert`` (TileRT venv, one process, eight GPUs)
    Regenerates the case weights layer by layer without sharding, converts
    them in memory with TileRT's own ``shard_layer``/``shard_tail``/
    ``pack_rank_params`` (no checkpoint on disk; non-MTP), injects the BF16
    cache prefix, and runs one greedy ``glm52_show_hands`` step at position
    ``context``. Outputs are graded with MegaBench's ``_compare`` bands; host
    latency is timed per step with all eight devices synchronized.

``check-ops`` (TileRT venv, one GPU)
    Calls TileRT's exported single ops on random inputs and compares each to
    the candidate formulas in ``tasks/frontier.py`` (and to the alternative
    reading where the public sources leave a choice open), so the GLM_*
    constants there can be confirmed or flipped. Single-PE calls of the
    all-reduce ops (``npes=1``, no symmetric buffer) follow the wrappers'
    defaults but are unverified until run on MI350X.

``--layers N`` truncates the model in both ``reference`` and ``tilert`` to
bisect numerics. TileRT exposes the KV/rope/index-key cache rows it wrote,
FP32 vocabulary-shard logits and the greedy token; it keeps sparse indices
and expert IDs only for the last full and last MoE layer, so those are
reported, not graded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from megabench.cases import select_cases

GRADED = ('logits', 'next_token', 'kv_write', 'pe_write', 'index_k_write')
# TileRT ROCm levers (weight_converter.py): W8A8 MoE, BF16 latent and index-key caches.
TILERT_ENV = {'TILERT_GLM5_MOE_W8A8': '1', 'TILERT_GLM5_FP8_KV': '0', 'TILERT_GLM5_FP8_KI': '0'}


def _case(case_id: str, layers: int | None):
    (case,) = select_cases('all', [case_id])
    if case.family != 'glm52_step':
        raise SystemExit(f'{case_id}: this probe supports GLM-5.2 cases only')
    if layers is not None:
        if not 1 <= layers <= case.params['layers']:
            raise SystemExit(f'--layers must be in [1, {case.params["layers"]}]')
        case = replace(case, params=case.params | {'layers': layers})
    return case


def _digest(values: dict[str, torch.Tensor], layers: int) -> str:
    """Hash replicated inputs that are identical on every rank and unsharded."""
    digest = hashlib.sha256()
    names = ['token'] + [f'l{layer}_{name}' for layer in sorted({0, min(3, layers - 1), layers - 1})
                         for name in ('qa', 'qa_scale', 'ka', 'kv_cache', 'pe_cache',
                                      'index_cache', 'iw', 'router', 'router_bias')]
    for name in names:
        if name in values:
            digest.update(name.encode())
            digest.update(values[name].detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def run_reference(args) -> None:
    import torch.distributed as dist

    from megabench.tasks import frontier
    from megabench.tasks.workloads import oracle, reference

    case = _case(args.case, args.layers)
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    with torch.inference_mode():
        values = frontier.make_inputs(case, args.seed, device)
        expected = reference(case, values)
        exact = oracle(case, values)
    record = {'case': case.id, 'seed': args.seed, 'rank': rank, 'layers': case.params['layers'],
              'torch': torch.__version__, 'input_digest': _digest(values, case.params['layers']),
              'expected': {name: value.cpu() for name, value in expected.items()},
              'exact': {name: value.cpu() for name, value in exact.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f'reference-rank-{rank}.pt').open('xb') as file:
        torch.save(record, file)
    dist.barrier()
    dist.destroy_process_group()


def _hf_layer(values: dict[str, torch.Tensor], layer: int, case) -> dict[str, torch.Tensor]:
    """Name one unsharded MegaBench layer as a GLM-5.2-FP8 HF checkpoint."""
    from megabench.tasks import frontier

    prefix, scope, out = f'l{layer}_', f'model.layers.{layer}.', {}

    def fp8(hf_name: str, name: str) -> None:
        out[scope + hf_name + '.weight'] = values[prefix + name]
        out[scope + hf_name + '.weight_scale_inv'] = values[prefix + name + '_scale']

    out[scope + 'input_layernorm.weight'] = values['ln1'][layer]
    out[scope + 'post_attention_layernorm.weight'] = values['ln2'][layer]
    out[scope + 'self_attn.q_a_layernorm.weight'] = values[prefix + 'qn']
    out[scope + 'self_attn.kv_a_layernorm.weight'] = values[prefix + 'kn']
    for hf_name, name in (('q_a_proj', 'qa'), ('q_b_proj', 'qb'), ('kv_a_proj_with_mqa', 'ka'),
                          ('kv_b_proj', 'kb'), ('o_proj', 'o')):
        fp8('self_attn.' + hf_name, name)
    if frontier.indexer_layer(case, layer):
        fp8('self_attn.indexer.wq_b', 'iq')
        fp8('self_attn.indexer.wk', 'ik')
        out[scope + 'self_attn.indexer.k_norm.weight'] = values[prefix + 'inorm']
        out[scope + 'self_attn.indexer.k_norm.bias'] = values[prefix + 'ibias']
        out[scope + 'self_attn.indexer.weights_proj.weight'] = values[prefix + 'iw']
    if layer < case.params['first_dense']:
        for name in ('gate', 'up', 'down'):
            fp8(f'mlp.{name}_proj', name)
        return out
    out[scope + 'mlp.gate.weight'] = values[prefix + 'router']
    out[scope + 'mlp.gate.e_score_correction_bias'] = values[prefix + 'router_bias']
    for name in ('gate', 'up', 'down'):
        fp8(f'mlp.shared_experts.{name}_proj', 'shared_' + name)
        for expert in range(case.params['experts']):
            out[f'{scope}mlp.experts.{expert}.{name}_proj.weight'] = values[prefix + name][expert]
            out[f'{scope}mlp.experts.{expert}.{name}_proj.weight_scale_inv'] = values[prefix + name + '_scale'][expert]
    return out


class _Reader:
    """The ``CheckpointReader.get`` interface over in-memory HF tensors."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def get(self, key: str) -> torch.Tensor:
        return self.tensors[key]


def _check_geometry(case, args) -> None:
    p, a = case.params, args
    want = {'hidden': a.dim, 'dense_intermediate': a.inter_dim, 'intermediate': a.moe_inter_dim,
            'q_heads': a.n_heads, 'q_lora_rank': a.q_lora_rank, 'kv_lora_rank': a.kv_lora_rank,
            'qk_nope_dim': a.qk_nope_head_dim, 'qk_rope_dim': a.qk_rope_head_dim, 'v_head_dim': a.v_head_dim,
            'index_heads': a.index_n_heads, 'index_dim': a.index_head_dim, 'index_topk': a.index_topk,
            'experts': a.n_routed_experts, 'shared_experts': a.n_shared_experts, 'topk': a.n_activated_experts,
            'vocab': a.vocab_size, 'first_dense': a.n_dense_layers}
    wrong = {name: (p[name], value) for name, value in want.items() if p[name] != value}
    if wrong:
        raise SystemExit(f'case geometry differs from TileRT ModelArgsGlm52: {wrong}')


def build_tilert_params(case, seed: int, model_args, generate_on: str, log):
    """TileRT per-rank parameter lists, per-layer BF16 cache prefixes, token, digest."""
    from megabench.tasks import frontier
    from tilert.models.glm_5_2_rocm.model_args import KIND_MOE_SHARED, layer_kind
    from tilert.models.glm_5_2_rocm.ops.llm_preprocess import make_freqs_cis
    from tilert.models.glm_5_2_rocm.weight_converter import pack_rank_params, shard_layer, shard_tail

    devices, layers = model_args.num_devices, case.params['layers']
    unsharded = replace(case, gpus=1, tp=1)
    stores = {rank: {} for rank in range(devices)}
    caches, digest_values = [], {}
    top = frontier.global_inputs(unsharded, seed, generate_on, rank=0)
    digest_values['token'] = top['token']
    norms = {'ln1': top['ln1'].cpu(), 'ln2': top['ln2'].cpu()}
    for layer in range(layers):
        start = time.perf_counter()
        values = frontier.layer_inputs(unsharded, seed, layer, generate_on, rank=0)
        if layer in {0, min(3, layers - 1), layers - 1}:
            digest_values |= values
        cpu = norms | {name: value.cpu() for name, value in values.items()}
        del values
        prefix = f'l{layer}_'
        caches.append((cpu[prefix + 'kv_cache'], cpu[prefix + 'pe_cache'], cpu.get(prefix + 'index_cache')))
        reader = _Reader(_hf_layer(cpu, layer, case))
        for rank in range(devices):
            # attn_tp8=True: shared layers run TP8 attention (the converter's default).
            shard = shard_layer(reader, layer, rank, model_args, 'cpu', attn_tp8=True)
            if layer_kind(layer) == KIND_MOE_SHARED:
                # Shared layers read only the attn_tp8 attention set.
                shard = {key: value for key, value in shard.items() if f'attn_tp8.{key}' not in shard}
            stores[rank] |= {f'layer_{layer}.{key}': value.to(f'cuda:{rank}') for key, value in shard.items()}
        del cpu, reader
        log(f'layer {layer}: generated and sharded in {time.perf_counter() - start:.1f}s')
    tail = _Reader({'model.norm.weight': top['fnorm'].cpu(), 'lm_head.weight': top['lm_head'].cpu()})
    embed = top['embed'].cpu()
    freqs = make_freqs_cis(model_args.max_seq_len, theta=model_args.rope_theta, device='cpu')
    params = []
    for rank in range(devices):
        stores[rank] |= {key: value.to(f'cuda:{rank}') for key, value in shard_tail(tail, rank, model_args).items()}
        store = stores.pop(rank)
        with torch.cuda.device(rank):
            # Each tensor is read once; popping frees the unpacked copy.
            params.append(pack_rank_params(store.pop, embed, model_args, rank, f'cuda:{rank}', layers,
                                           num_mtp=0, freqs_cis=freqs))
        del store
        log(f'rank {rank}: packed')
    return params, caches, int(top['token']), _digest(digest_values, layers)


def _load_reference(directory: Path, case, seed: int) -> dict:
    # The reference command writes these files (tensors plus metadata), so full unpickling is trusted.
    ranks = [torch.load(directory / f'reference-rank-{rank}.pt', weights_only=False) for rank in range(case.gpus)]
    if any(record['case'] != case.id or record['seed'] != seed or record['layers'] != case.params['layers']
           for record in ranks):
        raise SystemExit('reference files were produced for a different case, seed or --layers')
    merged = {'input_digest': ranks[0]['input_digest'], 'torch': ranks[0]['torch']}
    for kind in ('expected', 'exact'):
        # Logits are contiguous vocabulary shards; every other output is replicated.
        merged[kind] = ranks[0][kind] | {'logits': torch.cat([record[kind]['logits'] for record in ranks])}
    return merged


def _load_tilert() -> None:
    os.environ.update(TILERT_ENV)  # Read by the converter and the engine.
    import tilert

    if hasattr(tilert, 'load_backend'):
        tilert.load_backend('glm5_2_rocm')
    from tilert.tilert_init import tilert_init

    tilert_init()


def run_tilert(args) -> None:
    from megabench.harness.correctness import _compare

    case = _case(args.case, args.layers)
    _load_tilert()
    from tilert.models.glm_5_2_rocm.end2end import Glm52ShowHands, Idx
    from tilert.models.glm_5_2_rocm.model_args import ModelArgsGlm52, full_layer_ordinals

    devices, position, layers = 8, case.params['context'], case.params['layers']
    if torch.cuda.device_count() < devices:
        raise SystemExit(f'TileRT GLM-5.2 needs {devices} visible GPUs')
    log = lambda message: print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open('x')  # Claim the result path before the long build.
    reference = _load_reference(args.reference, case, args.seed)
    # The engine needs max_seq_len >= the 2048-token selection budget and > position.
    model_args = ModelArgsGlm52(max_seq_len=max(4096, -(-(position + 1) // 256) * 256), num_mtp=0)
    _check_geometry(case, model_args)
    phases = {}
    start = time.perf_counter()
    with torch.inference_mode():
        params, caches, token, digest = build_tilert_params(case, args.seed, model_args, 'cuda:0', log)
    phases['generate_and_convert_s'] = time.perf_counter() - start
    if digest != reference['input_digest']:
        raise SystemExit('regenerated inputs differ from the reference run (torch RNG drift?)')

    start = time.perf_counter()
    engine = Glm52ShowHands(model_args=model_args, use_topp=False, n_layers=layers, num_mtp=0)
    # from_pretrained without a converted directory: lease, state, weights.
    engine._acquire_lease()
    engine._alloc_state()
    engine._prepare(params)
    del params
    engine.reset_sequence()
    full = full_layer_ordinals(layers)
    for rank in range(devices):
        device_caches = engine._caches[rank]
        for layer, (kv, pe, _) in enumerate(caches):
            device_caches[2*layer][0, :position].copy_(kv.to(f'cuda:{rank}'))
            device_caches[2*layer + 1][0, :position].copy_(pe.to(f'cuda:{rank}'))
    # Rank 0 holds the index-key caches after the kv/pe pairs, in full-layer order.
    for slot, layer in enumerate(full):
        engine._caches[0][2*layers + slot][0, :position].copy_(caches[layer][2].to('cuda:0'))
    phases['engine_build_s'] = time.perf_counter() - start

    token_input = torch.tensor([token], dtype=torch.int32)

    def synchronize() -> None:
        for device in range(devices):
            torch.cuda.synchronize(device)

    def step() -> float:
        # The engine advances its position after each step; pin it to the case.
        engine.set_cur_pos(position)
        synchronize()
        begin = time.perf_counter()
        engine.forward(token_input)
        synchronize()
        return (time.perf_counter() - begin) * 1000

    step()
    temps, cache = engine._temps, engine._caches
    actual = {
        'logits': torch.cat([temps[rank][Idx.LOGITS][0].cpu() for rank in range(devices)]),
        'next_token': temps[0][Idx.TOKEN_OUT][0].to(torch.int64).cpu(),
        # Rank 1 runs attention on every layer (full layers: ranks 1-7; shared: TP8).
        'kv_write': torch.stack([cache[1][2*layer][0, position].cpu() for layer in range(layers)]),
        'pe_write': torch.stack([cache[1][2*layer + 1][0, position].cpu() for layer in range(layers)]),
        'index_k_write': torch.stack([cache[0][2*layers + slot][0, position].cpu() for slot in range(len(full))]),
    }
    expected = {name: reference['expected'][name] for name in GRADED}
    exact = {name: reference['exact'][name] for name in GRADED}
    correctness = {}
    # Logits are gathered here, so the greedy token may use the harness's
    # single-rank near-tie rule against full-vocabulary logits.
    gathered = replace(case, gpus=1, tp=1)
    for name in GRADED:
        names = (name, 'logits') if name == 'next_token' else (name,)
        try:
            correctness[name] = _compare({key: expected[key] for key in names}, {key: actual[key] for key in names},
                                         gathered, 'cpu', {key: exact[key] for key in names})[name]
        except AssertionError as error:
            correctness[name] = {'pass': False, 'reason': str(error)}
    logits, want = actual['logits'].double(), reference['exact']['logits'].double()
    correctness['logits_cosine_vs_oracle'] = float(torch.nn.functional.cosine_similarity(logits, want, dim=0))
    # Per-layer write errors help bisect (first layer whose bands fail).
    correctness['first_failing_layer'] = {
        name: next((layer for layer in range(len(actual[name]))
                    if not torch.allclose(actual[name][layer].float(), expected[name][layer].float(), rtol=0.016, atol=0.016)), None)
        for name in ('kv_write', 'pe_write', 'index_k_write')}
    topk = min(case.params['index_topk'], position + 1)
    last_indices = temps[0][Idx.IDX_SELECTS][0, :topk].long().cpu()
    want_indices = reference['expected']['sparse_indices'][-1].long()
    correctness['last_full_layer_sparse_index_overlap'] = (
        len(set(last_indices.tolist()) & set(want_indices.tolist())) / topk)
    if layers > case.params['first_dense']:
        last_experts = temps[0][Idx.MOE_IDX][0, :case.params['topk']].long().cpu()
        correctness['last_layer_expert_ids'] = {'tilert': sorted(last_experts.tolist()),
                                                'reference': sorted(reference['expected']['expert_ids'][-1].tolist())}
    graded_pass = all(correctness[name]['pass'] for name in GRADED)

    for _ in range(args.warmup):
        step()
    host_ms = [step() for _ in range(args.reps)]
    ordered = sorted(host_ms)
    at = (len(ordered) - 1) * 0.95
    p95 = ordered[int(at)] + (ordered[min(int(at) + 1, len(ordered) - 1)] - ordered[int(at)]) * (at - int(at))
    report = {
        'probe': 'tilert_glm52', 'case': case.id, 'layers': layers, 'seed': args.seed, 'position': position,
        'tilert_version': __import__('tilert').__version__, 'torch': torch.__version__,
        'reference_torch': reference['torch'], 'env': TILERT_ENV, 'max_seq_len': model_args.max_seq_len,
        'gpus': [torch.cuda.get_device_name(device) for device in range(devices)],
        'mode': 'non-MTP greedy decode, one step per call; sampling, MTP and prefill excluded',
        'status': 'pass' if graded_pass else 'incorrect', 'correctness': correctness, 'phases': phases,
        'timing': {'host_ms': host_ms, 'host_p50_ms': statistics.median(host_ms), 'host_p95_ms': p95,
                   'tokens_per_s_at_p50': 1000 / statistics.median(host_ms),
                   'warmup': args.warmup, 'reps': args.reps,
                   'method': 'perf_counter around glm52_show_hands with all eight devices synchronized; position re-set each step'},
    }
    with output:
        json.dump(report, output, indent=2)
    print(json.dumps({key: report[key] for key in ('case', 'layers', 'status', 'phases')} |
                     {'host_p50_ms': report['timing']['host_p50_ms'],
                      'logits': correctness['logits'], 'next_token': correctness['next_token']}, indent=2))
    engine.cleanup()


def _error(got: torch.Tensor, want: torch.Tensor) -> dict:
    got, want = got.float().cpu().flatten(), want.float().cpu().flatten()
    delta = (got - want).abs()
    return {'max_abs': float(delta.max()), 'rel_l2': float(delta.norm() / want.norm().clamp_min(1e-30)),
            'exact_fraction': float((got == want).float().mean())}


def run_check_ops(args) -> None:
    """Single-op TileRT calls versus candidate formulas. Needs on-hardware confirmation."""
    _load_tilert()
    from megabench.tasks import frontier
    from megabench.tasks.common import rms
    from megabench.tasks.quantization import pack_fp8_activation, pack_fp8_weight, unpack_fp8_activation, unpack_fp8_weight

    device = 'cuda:0'
    torch.manual_seed(args.seed)
    hidden = 6144
    report = {'probe': 'tilert_glm52_check_ops', 'seed': args.seed, 'env': TILERT_ENV,
              'tilert_version': __import__('tilert').__version__, 'gpu': torch.cuda.get_device_name(device),
              'note': 'Single-PE (npes=1, sym=None) calls follow the op wrappers defaults; unverified until run.'}

    def fp8_weight(rows, cols):
        payload, scale = pack_fp8_weight((torch.randn(rows, cols) * cols**-0.5).to(torch.bfloat16))
        return payload, scale

    def run(name, check):
        try:
            report[name] = check()
        except Exception as error:  # Report and continue: each op is an independent probe.
            import traceback
            report[name] = {'error': f'{type(error).__name__}: {error}',
                            'traceback': traceback.format_exc(limit=8)}
        print(json.dumps({name: report[name]}), flush=True)

    x = (torch.randn(1, hidden) * 2).to(torch.bfloat16)
    gamma = 1 + 0.1 * torch.randn(hidden)

    def head_proj():
        from tilert.models.glm_5_2_rocm.ops.rmsnorm_head_proj import VOCAB_SHARD
        from tilert.models.glm_5_2_rocm.weight_converter import swizzle_bf16_16x32

        head = (torch.randn(VOCAB_SHARD, hidden) * hidden**-0.5).to(torch.bfloat16)
        logits = torch.empty(1, VOCAB_SHARD, dtype=torch.float32, device=device)
        norm_out = torch.empty(1, hidden, dtype=torch.bfloat16, device=device)
        torch.ops.tilert.glm5_rmsnorm_head_proj_op(x.to(device), gamma.to(device),
                                                   swizzle_bf16_16x32(head).view(torch.uint8).to(device), logits, norm_out)
        fp32_gamma = rms(x[0], gamma, eps=1e-5, weight_in_fp32=True)
        cast_first = rms(x[0], gamma.to(torch.bfloat16), eps=1e-5)
        return {'norm_vs_fp32_gamma_one_rounding': _error(norm_out[0], fp32_gamma),
                'norm_vs_round_then_gamma': _error(norm_out[0], cast_first),
                'logits_vs_fp32_head_gemv': _error(logits[0], head.float() @ norm_out[0].float().cpu())}

    def unproj_o():
        from tilert.models.glm_5_2_rocm.ops.unprojo_allreduce import UnprojOAllReduceGlm5
        from tilert.models.glm_5_2_rocm.weight_converter import swizzle_fp8_v2

        heads = 10
        op = UnprojOAllReduceGlm5(num_heads=heads, device=device)
        weight, scale = fp8_weight(hidden, heads * 256)
        op.w_fp8, op.scales = weight, scale.to(device)
        op.packed = swizzle_fp8_v2(weight).to(device)
        proj_o = (torch.randn(1, heads * 256)).to(torch.bfloat16)
        residual = (torch.randn(1, hidden)).to(torch.bfloat16)
        out = op.tilert_forward(proj_o.to(device), residual.to(device))
        w8a16 = frontier.rank_partial([(proj_o[0], weight, scale, 1.0)])
        quantized = unpack_fp8_activation(*pack_fp8_activation(proj_o[0]))
        w8a8 = frontier.rank_partial([(quantized, weight, scale, 1.0)])
        fused = lambda partial: (residual[0].float() + partial.float()).to(torch.bfloat16)
        return {'vs_w8a16_fused_residual': _error(out[0], fused(w8a16)),
                'vs_w8a8_fused_residual': _error(out[0], fused(w8a8)),
                'vs_w8a16_two_roundings': _error(out[0], residual[0] + w8a16)}

    def dense_upgate():
        from tilert.models.glm_5_2_rocm.ops.upgate_silu import DENSE_INTER
        from tilert.models.glm_5_2_rocm.weight_converter import swizzle_pair_interleaved

        weight, scale = fp8_weight(2 * DENSE_INTER, hidden)
        mid = torch.empty(1, DENSE_INTER, dtype=torch.bfloat16, device=device)
        torch.ops.tilert.glm5_rmsnorm_upgate_silu_op(x.to(device), gamma.to(device),
                                                    swizzle_pair_interleaved(weight, DENSE_INTER).to(device),
                                                    scale.to(device), mid)
        normalized = rms(x[0], gamma, eps=1e-5, weight_in_fp32=True)
        both = unpack_fp8_weight(weight, scale) @ normalized.float()
        gate, up = both[:DENSE_INTER], both[DENSE_INTER:]
        fp32 = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        g16, u16 = gate.to(torch.bfloat16).float(), up.to(torch.bfloat16).float()
        rounded = (g16 * torch.sigmoid(g16) * u16).to(torch.bfloat16)
        q = unpack_fp8_activation(*pack_fp8_activation(normalized))
        both8 = unpack_fp8_weight(weight, scale) @ q
        w8a8 = (both8[:DENSE_INTER] * torch.sigmoid(both8[:DENSE_INTER]) * both8[DENSE_INTER:]).to(torch.bfloat16)
        return {'vs_fp32_swiglu (GLM_DENSE_SWIGLU_FP32=True)': _error(mid[0], fp32),
                'vs_bf16_gate_up (False)': _error(mid[0], rounded),
                'vs_w8a8_input': _error(mid[0], w8a8)}

    def sparse_index():
        from tilert.models.glm_5_2_rocm.ops.sparse_index import SparseIndexGlm5

        length, position = 6144, 6000
        iq = torch.randn(1, 32 * 128).to(torch.bfloat16)
        keys = torch.zeros(1, length, 128, dtype=torch.bfloat16)
        keys[0, :position + 1] = torch.randn(position + 1, 128).to(torch.bfloat16)
        weights = torch.randn(1, 32).to(torch.bfloat16)
        op = SparseIndexGlm5(device=device)
        logits, selected, _ = op.tilert_forward(iq.to(device), keys.to(device), weights.to(device),
                                                torch.tensor([position], dtype=torch.int32, device=device), 1)
        want = (torch.relu(iq.float().view(32, 128) @ keys[0, :position + 1].float().T * (32**-0.5 * 128**-0.5))
                * weights.float().view(32, 1)).sum(0)
        chosen = set(selected[0].cpu().tolist())
        ours = set(want.argsort(descending=True, stable=True)[:2048].tolist())
        return {'logits_vs_candidate': _error(logits[0, :position + 1], want),
                'selection_overlap_with_candidate': len(chosen & ours) / 2048}

    def fused_moe():
        from tilert.models.glm_5_2_rocm.ops.fused_moe_w8a8 import FusedMoeAllreduceW8A8Glm5

        inter, banks = 256, 257
        op = FusedMoeAllreduceW8A8Glm5(device=device)
        router = (torch.randn(256, hidden) * hidden**-0.5).to(torch.bfloat16)
        bias = 0.01 * torch.randn(256)
        up_gate = [fp8_weight(2 * inter, hidden) for _ in range(banks)]
        down = [fp8_weight(hidden, inter) for _ in range(banks)]
        # TileRT packs the router weight on the CPU before moving it to the device.
        op.front.router.w, op.front.router.gamma = router, gamma.to(device)
        op.front.moe.w_fp8 = torch.stack([w for w, _ in up_gate])
        op.front.moe.scales = torch.stack([s for _, s in up_gate]).to(device)
        op.down.w_fp8 = torch.stack([w for w, _ in down])
        op.down.scales = torch.stack([s for _, s in down]).to(device)
        residual = torch.randn(1, hidden).to(torch.bfloat16)
        norm, scores, mid, probs, indices, out = op.tilert_forward_w8a8_v4(x.to(device), bias.to(device), residual.to(device))
        normalized = rms(x[0], gamma, eps=1e-5, weight_in_fp32=True)
        ids, weights = frontier.route_tilert(router.float() @ normalized.float(), bias, 8)
        activation = unpack_fp8_activation(*frontier.tilert_quantize(normalized))
        terms, mids = [], []
        for expert, factor in [(0, 1.0)] + [(1 + int(e), w) for e, w in zip(ids, weights)]:
            both = unpack_fp8_weight(*up_gate[expert]) @ activation
            mids.append((both[:inter] * torch.sigmoid(both[:inter]) * both[inter:]).to(torch.bfloat16))
            terms.append((unpack_fp8_activation(*frontier.tilert_quantize(mids[-1])), *down[expert], factor))
        partial = frontier.rank_partial(terms)
        return {'norm': _error(norm[0], normalized),
                'expert_ids_match': indices[0].cpu().long().tolist() == ids.tolist(),
                'probs': _error(probs[0], weights), 'mid': _error(mid[0], torch.stack(mids)),
                'out_vs_fused_residual': _error(out[0], (residual[0].float() + partial.float()).to(torch.bfloat16))}

    run('glm5_rmsnorm_head_proj_op', head_proj)
    run('glm5_unprojo_allreduce_op', unproj_o)
    run('glm5_rmsnorm_upgate_silu_op', dense_upgate)
    run('glm5_sparse_index_op+glm5_topk_select_op', sparse_index)
    run('glm5_fused_moe_allreduce_w8a8_v4_op', fused_moe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    reference = commands.add_parser('reference', help='store reference outputs (torchrun, eight ranks)')
    tilert = commands.add_parser('tilert', help='grade and time TileRT (TileRT venv, one process)')
    check = commands.add_parser('check-ops', help='compare TileRT single ops with candidate formulas (one GPU)')
    for command in (reference, tilert):
        command.add_argument('--case', default='glm52-step')
        command.add_argument('--layers', type=int, help='truncate the model to its first N layers')
    for command in (reference, tilert, check):
        command.add_argument('--seed', type=int, default=20261008)
    reference.add_argument('--output', type=Path, required=True, help='directory for reference-rank-*.pt')
    tilert.add_argument('--reference', type=Path, required=True, help='directory written by the reference command')
    for command in (tilert, check):
        command.add_argument('--output', type=Path, required=True, help='new JSON report path (exclusive create)')
    tilert.add_argument('--warmup', type=int, default=10)
    tilert.add_argument('--reps', type=int, default=50)
    args = parser.parse_args()
    {'reference': run_reference, 'tilert': run_tilert, 'check-ops': run_check_ops}[args.command](args)


if __name__ == '__main__':
    main()
