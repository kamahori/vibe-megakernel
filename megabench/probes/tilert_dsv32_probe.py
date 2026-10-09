"""Grade and time TileRT's DeepSeek-V3.2 decode engine on a MegaBench case.

TileRT (tile-ai/TileRT 0.1.6) is one host process driving eight GPUs, so it
cannot run as a per-rank MegaBench submission. This probe runs it on the
same seeded synthetic weights, cache prefix and token as the case instead:

``reference`` (MegaBench venv, ``torch.distributed.run`` on eight ranks)
    Builds the case inputs and stores the BF16 reference outputs, the FP32
    oracle outputs, and a digest of replicated inputs, one file per rank.

``tilert`` (TileRT venv, one process, eight GPUs)
    Regenerates the case weights layer by layer without sharding, converts
    them in memory with TileRT's own checkpoint transforms (no checkpoint on
    disk), injects the BF16 cache prefix, and runs one non-MTP greedy step at
    position ``context``. Outputs are graded with MegaBench's ``_compare``
    bands; host latency is timed per step with all eight devices
    synchronized, matching ``harness.benchmark._measure``'s host timer.

TileRT exposes the KV/rope/index-key cache rows it wrote, full FP32 logits
(vocabulary shards) and the greedy token. Its sparse indices and expert IDs
are only kept for the last layer, so those are reported, not graded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from megabench.cases import select_cases

GRADED = ('logits', 'next_token', 'kv_write', 'pe_write', 'index_k_write')


def _case(case_id: str):
    (case,) = select_cases('all', [case_id])
    if case.family != 'deepseek_v32_step':
        raise SystemExit(f'{case_id}: TileRT probe supports DeepSeek-V3.2 cases only')
    return case


def _digest(values: dict[str, torch.Tensor], layers: int) -> str:
    """Hash replicated inputs that are identical on every rank and unsharded."""
    digest = hashlib.sha256()
    names = ['token'] + [f'l{layer}_{name}' for layer in (0, 3, layers - 1)
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

    case = _case(args.case)
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    with torch.inference_mode():
        values = frontier.make_inputs(case, args.seed, device)
        expected = reference(case, values)
        exact = oracle(case, values)
    record = {'case': case.id, 'seed': args.seed, 'rank': rank, 'torch': torch.__version__,
              'input_digest': _digest(values, case.params['layers']),
              'expected': {name: value.cpu() for name, value in expected.items()},
              'exact': {name: value.cpu() for name, value in exact.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f'reference-rank-{rank}.pt').open('xb') as file:
        torch.save(record, file)
    dist.barrier()
    dist.destroy_process_group()


def _hf_layer(values: dict[str, torch.Tensor], layer: int, case) -> dict[str, torch.Tensor]:
    """Name one unsharded MegaBench layer as a DeepSeek-V3.2 HF checkpoint."""
    prefix, out = f'l{layer}_', {}
    scope = f'model.layers.{layer}.'

    def fp8(hf_name: str, name: str) -> None:
        out[scope + hf_name + '.weight'] = values[prefix + name]
        out[scope + hf_name + '.weight_scale_inv'] = values[prefix + name + '_scale']

    out[scope + 'input_layernorm.weight'] = values['ln1'][layer]
    out[scope + 'post_attention_layernorm.weight'] = values['ln2'][layer]
    out[scope + 'self_attn.q_a_layernorm.weight'] = values[prefix + 'qn']
    out[scope + 'self_attn.kv_a_layernorm.weight'] = values[prefix + 'kn']
    for hf_name, name in (('q_a_proj', 'qa'), ('q_b_proj', 'qb'), ('kv_a_proj_with_mqa', 'ka'),
                          ('kv_b_proj', 'kb'), ('o_proj', 'o'), ('indexer.wq_b', 'iq'),
                          ('indexer.wk', 'ik')):
        fp8('self_attn.' + hf_name, name)
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


def convert_layer(converter, values: dict[str, torch.Tensor], layer: int, case) -> dict[int, dict[str, torch.Tensor]]:
    """Apply TileRT's checkpoint transforms; return per-device TileRT keys."""
    hf = _hf_layer(values, layer, case)
    groups = [converter.transform_mla(hf, layer),
              (converter.transform_mlp if layer < case.params['first_dense'] else converter.transform_moe)(hf, layer)]
    devices = {}
    for device in range(converter.num_devices):
        devices[device] = {f'layer_{layer}_{name}_dev_{device}': tensor
                           for group in groups for name, tensor in group[f'dev_{device}'].items()}
    return devices


def make_converter(devices: int = 8):
    from tilert.models.deepseek_v3_2.model_args import ModelArgs
    from tilert.models.preprocess.weight_converter import WeightConverter

    # The constructor indexes an on-disk checkpoint; the transforms need only these.
    converter = object.__new__(WeightConverter)
    converter.model_args, converter.num_devices = ModelArgs(), devices
    return converter


def build_tilert_inputs(case, seed: int, devices: int, generate_on: str, log):
    """TileRT per-device state dicts and per-layer BF16 cache prefixes."""
    from megabench.tasks import frontier
    from tilert.models.deepseek_v3_2.model_args import ModelArgs
    from tilert.models.deepseek_v3_2.ops.rmsnorm_head_proj import RMSNormHeadProj

    unsharded = replace(case, gpus=1, tp=1)
    converter = make_converter(devices)
    states = {device: {} for device in range(devices)}
    caches, digest_values = [], {}
    top = frontier.global_inputs(unsharded, seed, generate_on, rank=0)
    digest_values['token'] = top['token']
    for layer in range(case.params['layers']):
        start = time.perf_counter()
        values = top | frontier.layer_inputs(unsharded, seed, layer, generate_on, rank=0)
        if layer in (0, 3, case.params['layers'] - 1):
            digest_values |= {name: value for name, value in values.items() if name.startswith(f'l{layer}_')}
        cpu = {name: value.cpu() for name, value in values.items()}
        prefix = f'l{layer}_'
        caches.append((cpu[prefix + 'index_cache'], cpu[prefix + 'kv_cache'], cpu[prefix + 'pe_cache']))
        for device, tensors in convert_layer(converter, cpu, layer, case).items():
            states[device] |= {name: tensor.to(f'cuda:{device}') for name, tensor in tensors.items()}
        del values, cpu
        log(f'layer {layer}: generated and converted in {time.perf_counter() - start:.1f}s')
    gamma, head = RMSNormHeadProj(ModelArgs(), device_id=0, num_devices=devices).device_sharding(
        {'model.norm.weight': top['fnorm'].cpu(), 'lm_head.weight': top['lm_head'].cpu()})
    last = case.params['layers']
    for device in range(devices):
        states[device] |= {f'layer_{last}_lm_head.weight_dev_{device}': head[device].to(f'cuda:{device}'),
                           f'layer_{last}_model.norm.weight_dev_{device}': gamma[device].to(f'cuda:{device}'),
                           'model.embed_tokens.weight': top['embed'].to(f'cuda:{device}')}
    return states, caches, int(top['token']), _digest(digest_values, case.params['layers'])


def _load_reference(directory: Path, case, seed: int) -> dict:
    ranks = [torch.load(directory / f'reference-rank-{rank}.pt', weights_only=True) for rank in range(case.gpus)]
    if any(record['case'] != case.id or record['seed'] != seed for record in ranks):
        raise SystemExit('reference files were produced for a different case or seed')
    merged = {'input_digest': ranks[0]['input_digest'], 'torch': ranks[0]['torch']}
    for kind in ('expected', 'exact'):
        # Logits are contiguous vocabulary shards; every other output is replicated.
        merged[kind] = ranks[0][kind] | {'logits': torch.cat([record[kind]['logits'] for record in ranks])}
    return merged


def run_tilert(args) -> None:
    from megabench.harness.correctness import _compare
    from tilert.models.deepseek_v3_2.model_args import ModelArgs
    from tilert.models.deepseek_v3_2.modules.end2end import ShowHandsDSALayer
    from tilert.models.deepseek_v3_2.temp_var_indices import Idx
    from tilert.tilert_init import tilert_init

    case = _case(args.case)
    devices, position = 8, case.params['context']
    if torch.cuda.device_count() < devices:
        raise SystemExit(f'TileRT DeepSeek-V3.2 needs {devices} visible GPUs')
    log = lambda message: print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open('x')  # Claim the result path before the long build.
    reference = _load_reference(args.reference, case, args.seed)
    phases = {}
    start = time.perf_counter()
    with torch.inference_mode():
        states, caches, token, digest = build_tilert_inputs(case, args.seed, devices, 'cuda:0', log)
    phases['generate_and_convert_s'] = time.perf_counter() - start
    if digest != reference['input_digest']:
        raise SystemExit('regenerated inputs differ from the reference run (torch RNG drift?)')

    start = time.perf_counter()
    tilert_init()
    engine = ShowHandsDSALayer(ModelArgs(), model_path='megabench-synthetic', with_mtp=False, use_topp=False)
    # Hand each device its dict once, so the engine holds the only reference
    # (as when TileRT loads converted safetensors from disk).
    engine.load_device_weights = lambda path, device, extra, skip_keys=None: (
        states.pop(device) | {'freqs_cis': engine._gen_freqs_cis().to(device)})
    engine._init_weights('megabench-synthetic')
    for device in range(devices):
        device_caches = engine._get_device_result(device)[1]
        for layer, prefix_caches in enumerate(caches):
            for slot, cache in enumerate(prefix_caches):
                device_caches[3*layer + slot][0, :position].copy_(cache.to(f'cuda:{device}'))
    phases['engine_build_s'] = time.perf_counter() - start

    token_input = torch.tensor(token)

    def synchronize() -> None:
        for device in range(devices):
            torch.cuda.synchronize(device)

    def step() -> float:
        # The engine advances its position after each step; pin it to the case.
        torch.ops.tilert.dsa_show_hands_set_cur_pos(position)
        synchronize()
        begin = time.perf_counter()
        engine.forward(token_input)
        synchronize()
        return (time.perf_counter() - begin) * 1000

    step()
    results = [engine._get_device_result(device) for device in range(devices)]
    layers = case.params['layers']
    actual = {
        'logits': torch.cat([results[device][0][Idx.LOGITS_OUT].reshape(-1, results[device][0][Idx.LOGITS_OUT].shape[-1])[0].cpu()
                             for device in range(devices)]),
        'next_token': results[0][0][Idx.TOKEN_OUT].reshape(-1)[0].to(torch.int64).cpu(),
        # MLA devices 1-7 write the latent and rope rows; device 0 the index key.
        'kv_write': torch.stack([results[1][1][3*layer + 1][0, position].cpu() for layer in range(layers)]),
        'pe_write': torch.stack([results[1][1][3*layer + 2][0, position].cpu() for layer in range(layers)]),
        'index_k_write': torch.stack([results[0][1][3*layer][0, position].cpu() for layer in range(layers)]),
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
    topk = min(case.params['index_topk'], position + 1)
    last_indices = results[0][0][Idx.IDX_SELECTS].reshape(-1)[:topk].long().cpu()
    want_indices = reference['expected']['sparse_indices'][-1].long()
    last_experts = results[0][0][Idx.SEL_INDICES].reshape(-1)[:case.params['topk']].long().cpu()
    correctness['last_layer_sparse_index_overlap'] = len(set(last_indices.tolist()) & set(want_indices.tolist())) / topk
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
        'probe': 'tilert_dsv32', 'case': case.id, 'seed': args.seed, 'position': position,
        'tilert_version': __import__('tilert').__version__, 'torch': torch.__version__,
        'reference_torch': reference['torch'], 'gpus': [torch.cuda.get_device_name(device) for device in range(devices)],
        'mode': 'non-MTP greedy decode, one step per call; sampling, MTP and prefill excluded',
        'status': 'pass' if graded_pass else 'incorrect', 'correctness': correctness, 'phases': phases,
        'timing': {'host_ms': host_ms, 'host_p50_ms': statistics.median(host_ms), 'host_p95_ms': p95,
                   'tokens_per_s_at_p50': 1000 / statistics.median(host_ms),
                   'warmup': args.warmup, 'reps': args.reps,
                   'method': 'perf_counter around engine.forward with all eight devices synchronized before and after'},
    }
    with output:
        json.dump(report, output, indent=2)
    print(json.dumps({key: report[key] for key in ('case', 'status', 'phases')} |
                     {'host_p50_ms': report['timing']['host_p50_ms'],
                      'logits': correctness['logits'], 'next_token': correctness['next_token']}, indent=2))
    engine.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    reference = commands.add_parser('reference', help='store reference outputs (torchrun, eight ranks)')
    tilert = commands.add_parser('tilert', help='grade and time TileRT (TileRT venv, one process)')
    for command in (reference, tilert):
        command.add_argument('--case', default='deepseek-v32-step')
        command.add_argument('--seed', type=int, default=20261008)
    reference.add_argument('--output', type=Path, required=True, help='directory for reference-rank-*.pt')
    tilert.add_argument('--reference', type=Path, required=True, help='directory written by the reference command')
    tilert.add_argument('--output', type=Path, required=True, help='new JSON report path (exclusive create)')
    tilert.add_argument('--warmup', type=int, default=10)
    tilert.add_argument('--reps', type=int, default=50)
    args = parser.parse_args()
    (run_reference if args.command == 'reference' else run_tilert)(args)


if __name__ == '__main__':
    main()
