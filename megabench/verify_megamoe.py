"""Verify the MegaMoE routed-expert EP reference via torchrun.

Every rank builds only its own tokens, routing and experts. Rank 0
additionally builds the serial geometry (one rank holding all
``tokens*gpus`` tokens and every expert) and runs the reference without
communication; each rank's all-to-all output must equal its slice bit for
bit. Full mode does this at catalog geometry on scheduled GPUs.
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
from .tasks import megamoe
from .tasks.oracle import fp32_reference
from .tasks.parallel import initialize
from .verify_frontier import input_digest


DEVELOPMENT = {'tokens':8, 'experts':16, 'topk':4, 'hidden':256, 'intermediate':128}


def development(case, ranks=None):
    ranks = case.gpus if ranks is None else ranks
    return replace(case, gpus=ranks, ep=ranks, params=case.params | DEVELOPMENT)


def serial_case(case):
    return replace(case, gpus=1, ep=1, params=case.params | {'tokens':case.params['tokens']*case.gpus})


def compare_serial(case, actual, expected, device):
    """Broadcast rank 0's serial outputs, then compare this rank's token rows."""
    rank, tokens = dist.get_rank(), case.params['tokens']
    details = {}
    for name, value in actual.items():
        wanted = torch.empty((tokens*case.gpus, *value.shape[1:]), device=device, dtype=value.dtype)
        if rank == 0:
            wanted.copy_(expected[name])
        dist.broadcast(wanted, src=0)
        wanted = wanted[rank*tokens:(rank+1)*tokens]
        details.update(_compare({name:wanted}, {name:value}, case, device))
        details[name]['bitwise_equal'] = bool(torch.equal(wanted, value))
    return details


def all_ranks(flag, device):
    value = torch.tensor([float(flag)], device=device)
    dist.all_reduce(value)
    return int(value.item())


def verify(case, device, trials, full):
    rank = dist.get_rank()
    if not full:
        case = development(case, dist.get_world_size())
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
            values = megamoe.make_inputs(case, seed, device)
            built = time.perf_counter()-start
            before = input_digest(values)
            actual = megamoe.reference(case, values)
            repeated = megamoe.reference(case, values)
            if not torch.equal(actual['y'], repeated['y']):
                raise AssertionError('reference is not deterministic')
            if not torch.isfinite(actual['y'].float()).all():
                raise AssertionError('non-finite y')
            if before != input_digest(values):
                raise AssertionError('reference mutated its runtime inputs')
            print(json.dumps({'case':case.id, 'rank':rank, 'seed':seed, 'phase':'ep_reference_done',
                              'elapsed_seconds':time.perf_counter()-start}), flush=True)
            expected = None
            if rank == 0:
                single = serial_case(case)
                expected = megamoe.reference(single, megamoe.make_inputs(single, seed, device, rank=0), serial=True)
            details = compare_serial(case, actual, expected, device)
            del expected
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            bitwise = all_ranks(details['y']['bitwise_equal'], device)
            print(json.dumps({'case':case.id, 'rank':rank, 'seed':seed, 'phase':'serial_verified',
                              'bitwise_ranks':bitwise, 'elapsed_seconds':time.perf_counter()-start}), flush=True)
            if not full and bitwise < case.gpus:
                raise AssertionError('EP output differs bitwise from the serial union')
            exact = fp32_reference(megamoe, case, values)['y'].double()
            error = actual['y'].double()-exact
            oracle_error = {'relative_error':float(error.norm()/exact.norm()),
                            'max_abs_error':float(error.abs().max()),
                            'max_abs_exact':float(exact.abs().max())}
            # Every rank's output must depend on remote experts' FP4 codes.
            changed = values | {'l2_blocks':values['l2_blocks']^0x11}
            moved = megamoe.reference(case, changed)['y']
            if all_ranks(not torch.equal(moved, actual['y']), device) < case.gpus:
                raise AssertionError('some rank output is independent of the expert weights')
            timing = _measure(lambda t:megamoe.reference(case, t), values, device, 1, 3)
            audit = _audit_launches(lambda t:megamoe.reference(case, t), values, device, case.max_gpu_launches)
            report['trials'].append({'seed':seed, 'input_bytes':sum(v.numel()*v.element_size() for v in values.values()),
                                     'build_seconds':built, 'elapsed_seconds':time.perf_counter()-start,
                                     'serial_comparison':details, 'oracle_error':oracle_error,
                                     'routed_rows':int(values['topk_idx'].numel()),
                                     'outputs':{k:{'shape':list(v.shape), 'dtype':str(v.dtype)} for k, v in actual.items()},
                                     'reference_timing':timing, 'reference_launch_audit':audit})
            print(json.dumps({'case':case.id, 'rank':rank, 'seed':seed, 'status':'pass',
                              'full_geometry':full}), flush=True)
            del values, actual, repeated, changed, moved, exact, error
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
