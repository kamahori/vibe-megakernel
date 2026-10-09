"""Verify the Kimi K3 DP-attention/EP layer reference via torchrun.

Every rank builds only its own sequences and experts. Rank 0 additionally
builds the serial geometry (one rank holding all ``batch*gpus`` sequences and
every expert) and runs the reference without communication; each rank's
all-to-all output must match its slice. Full mode does this at catalog
geometry, so it checks the expert dispatch and combine of the real case.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .cases import select_cases
from .harness.benchmark import _audit_launches, _measure
from .harness.correctness import _compare
from .tasks import kimi_layer
from .tasks.parallel import initialize
from .verify_frontier import input_digest


def development(case):
    """Reduced geometry with the catalog's attention variant and block position."""
    layer = 3 if kimi_layer.attention_kind(case) == 'mla' else 1
    return replace(case, params=case.params | {
        'layers':5, 'layer':layer, 'batch':2, 'context':4, 'hidden':64, 'latent_hidden':32,
        'intermediate':64, 'q_heads':4, 'head_dim':16, 'q_lora_rank':32, 'kv_lora_rank':32,
        'qk_nope_dim':8, 'qk_rope_dim':8, 'v_head_dim':16, 'experts':8, 'topk':2,
        'attn_res_block_size':2})


def serial_case(case):
    return replace(case, gpus=1, ep=1, params=case.params | {'batch':case.params['batch']*case.gpus})


def compare_serial(case, actual, expected, device):
    """Broadcast rank 0's serial outputs, then compare this rank's batch rows."""
    rank, batch = dist.get_rank(), case.params['batch']
    details = {}
    for name, value in actual.items():
        wanted = torch.empty((batch*case.gpus, *value.shape[1:]), device=device, dtype=value.dtype)
        if rank == 0:
            wanted.copy_(expected[name])
        dist.broadcast(wanted, src=0)
        details.update(_compare({name:wanted[rank*batch:(rank+1)*batch]}, {name:value}, case, device))
    return details


def verify(case, device, trials, full):
    rank = dist.get_rank()
    if not full:
        case = replace(development(case), gpus=dist.get_world_size(), ep=dist.get_world_size())
    report = {'case':case.to_dict(), 'rank':rank, 'full_geometry':full, 'trials':[],
              'torch':torch.__version__, 'cuda':torch.version.cuda}
    if device.startswith('cuda'):
        report['gpu'] = torch.cuda.get_device_name(device)
    initialize(case)
    with torch.inference_mode():
        for index in range(trials):
            seed = 104729+index
            start = time.perf_counter()
            print(json.dumps({'case':case.id, 'rank':rank, 'seed':seed, 'phase':'building_inputs',
                              'full_geometry':full}), flush=True)
            values = kimi_layer.make_inputs(case, seed, device)
            before = input_digest(values)
            actual = kimi_layer.reference(case, values)
            repeated = kimi_layer.reference(case, values)
            _compare(actual, repeated, case, device)
            for name, value in actual.items():
                if value.is_floating_point() and not torch.isfinite(value.float()).all():
                    raise AssertionError(f'non-finite {name}')
            if before != input_digest(values):
                raise AssertionError('reference mutated its runtime inputs')
            print(json.dumps({'case':case.id, 'rank':rank, 'seed':seed, 'phase':'ep_reference_done',
                              'elapsed_seconds':time.perf_counter()-start}), flush=True)
            expected = None
            if rank == 0:
                single = serial_case(case)
                expected = kimi_layer.reference(single, kimi_layer.make_inputs(single, seed, device, rank=0),
                                                serial=True)
            details = compare_serial(case, actual, expected, device)
            del expected
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            print(json.dumps({'case':case.id, 'rank':rank, 'seed':seed, 'phase':'serial_verified',
                              'elapsed_seconds':time.perf_counter()-start}), flush=True)
            # Every output must depend on the experts. Rescaling would cancel
            # in latent_norm, so change the FP4 codes instead.
            changed = values | {'down_blocks':values['down_blocks']^0x11}
            moved = kimi_layer.reference(case, changed)['prefix']
            dependent = torch.tensor([float(not torch.equal(moved, actual['prefix']))], device=device)
            dist.all_reduce(dependent)
            if dependent.item() < case.gpus:
                raise AssertionError('some rank output is independent of the expert weights')
            timing = _measure(lambda t:kimi_layer.reference(case, t), values, device, 1, 3)
            audit = _audit_launches(lambda t:kimi_layer.reference(case, t), values, device, case.max_gpu_launches)
            report['trials'].append({'seed':seed, 'input_bytes':sum(v.numel()*v.element_size() for v in values.values()),
                                     'elapsed_seconds':time.perf_counter()-start, 'serial_comparison':details,
                                     'outputs':{k:{'shape':list(v.shape), 'dtype':str(v.dtype)} for k, v in actual.items()},
                                     'reference_timing':timing, 'reference_launch_audit':audit})
            print(json.dumps({'case':case.id, 'rank':rank, 'seed':seed, 'status':'pass',
                              'full_geometry':full}), flush=True)
            del values, actual, repeated, changed, moved
    report['status'] = 'pass'
    if device.startswith('cuda'):
        report['gpu_memory'] = {'total_bytes':torch.cuda.get_device_properties(device).total_memory,
                                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{rank}' if args.device == 'cuda' else 'cpu'
    if args.full and args.device != 'cuda':
        parser.error('full geometry verification requires scheduled GPUs')
    if args.device == 'cuda':
        torch.cuda.set_device(rank)
    dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo', timeout=timedelta(minutes=20),
                            device_id=torch.device(device) if args.device == 'cuda' else None)
    try:
        result = verify(select_cases('all', [args.case])[0], device, args.trials, args.full)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output/f'rank-{rank}.json').open('x') as file:
            json.dump(result, file, indent=2)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
