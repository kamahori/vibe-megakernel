"""Verify native FP8 frontier reference state and TP protocol via torchrun.

Development mode compares TP against an unsharded native-format execution. Independent
MLA, routing, rotary, and format checks live in tests/test_frontier.py. Full
mode checks catalog geometry, rank agreement, determinism and input hashes
over three steps that commit native KV, index, convolution and recurrent state;
it never calls a reduced geometry a full-model verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .cases import select_cases
from .harness.benchmark import _audit_launches, _measure
from .harness.correctness import _compare
from .tasks import frontier, kimi
from .tests.test_frontier import development
from .tests.test_kimi import development as kimi_development


def input_digest(values):
    """Hash raw bytes in bounded host copies, including native FP8 payloads."""
    digest = hashlib.sha256()
    for name,value in sorted(values.items()):
        digest.update(name.encode())
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
        for start in range(0,raw.numel(),16*1024*1024):
            digest.update(raw[start:start+16*1024*1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def advance_state(case, values, outputs):
    """Commit native writes into the next step without changing prior inputs."""
    is_kimi = case.family == 'kimi_k3_step'
    reset = is_kimi and bool(values['reset'])
    result = values | {'token':outputs['next_token'].clone()}
    full_slot, recurrent_slot, index_slot = 0, 0, 0
    def append(key, write):
        old = values[key][:0] if reset else values[key]
        # CPU cat/collectives do not uniformly support FP8 arithmetic types.
        if old.dtype == torch.float8_e4m3fn:
            return torch.cat((old.view(torch.uint8), write[None].view(torch.uint8))).view(old.dtype)
        return torch.cat((old, write[None]))
    for layer in range(case.params['layers']):
        prefix = f'l{layer}_'
        if not is_kimi or kimi.full_layer(case, layer):
            result[prefix+'kv_cache'] = append(prefix+'kv_cache', outputs['kv_write'][full_slot])
            result[prefix+'pe_cache'] = append(prefix+'pe_cache', outputs['pe_write'][full_slot])
            if not is_kimi:
                result[prefix+'kv_cache_scale'] = append(prefix+'kv_cache_scale', outputs['kv_scale_write'][full_slot])
                if frontier.indexer_layer(case, layer):
                    result[prefix+'index_cache'] = append(prefix+'index_cache', outputs['index_k_write'][index_slot])
                    result[prefix+'index_cache_scale'] = append(prefix+'index_cache_scale', outputs['index_scale_write'][index_slot])
                    index_slot += 1
            full_slot += 1
        else:
            result[prefix+'recurrent_state'] = outputs['recurrent_state'][recurrent_slot].clone()
            result[prefix+'conv_state'] = outputs['conv_state'][recurrent_slot].clone()
            recurrent_slot += 1
    if is_kimi:
        result['reset'] = torch.zeros_like(values['reset'])
    context = int(outputs['cache_length']) if is_kimi else case.params['context']+1
    return replace(case, params=case.params | {'context':context}), result


def compare_serial(case, actual, expected, device, is_kimi):
    """Broadcast the complete oracle output, then compare this rank's slice."""
    rank = dist.get_rank()
    details = {}
    for name, value in actual.items():
        shape = list(value.shape)
        if name == 'logits':
            shape[0] *= case.tp
        elif is_kimi and name in ('recurrent_state', 'conv_state'):
            shape[1 if name == 'recurrent_state' else 2] *= case.tp
        wanted = torch.empty(shape, device=device, dtype=value.dtype)
        if rank == 0:
            wanted.copy_(expected[name])
        wire = wanted.view(torch.uint8) if wanted.dtype == torch.float8_e4m3fn else wanted
        dist.broadcast(wire, src=0)
        if name == 'logits':
            wanted = wanted.chunk(case.tp)[rank]
        elif is_kimi and name in ('recurrent_state', 'conv_state'):
            wanted = wanted.chunk(case.tp, dim=1 if name == 'recurrent_state' else 2)[rank]
        details.update(_compare({name:wanted}, {name:value}, case, device))
    return details


def verify(case, device, trials, full, steps=3):
    rank = dist.get_rank()
    module = kimi if case.family == 'kimi_k3_step' else frontier
    if not full:
        if module is kimi:
            case = kimi_development(case)
            if dist.get_world_size() > 2:
                case = replace(case,params=case.params | {'q_heads':32,'head_dim':8,'v_head_dim':8,
                                                         'intermediate':512,'dense_intermediate':512})
        else:
            case = development(case)
        case = replace(case,gpus=dist.get_world_size(),tp=dist.get_world_size())
    report = {'case':case.to_dict(),'rank':rank,'full_geometry':full,'steps':steps,'trials':[],
              'torch':torch.__version__,'cuda':torch.version.cuda}
    if device.startswith('cuda'):
        report['gpu'] = torch.cuda.get_device_name(device)
    module.initialize(case)
    with torch.inference_mode():
        for index in range(trials):
            seed = 104729+index
            current = case
            values = module.make_inputs(current,seed,device)
            perf_inputs = values
            serial_case, serial_values = None, None
            if not full and rank == 0:
                serial_case = replace(case,gpus=1,tp=1)
                serial_values = module.make_inputs(serial_case,seed,device,rank=0)
            trajectory = []
            for step in range(steps):
                before = input_digest(values)
                actual = module.reference(current,values)
                repeated = module.reference(current,values)
                _compare(actual,repeated,current,device)
                for name,value in actual.items():
                    if value.is_floating_point() and not torch.isfinite(value.float()).all():
                        raise AssertionError(f'non-finite {name}')
                if before != input_digest(values):
                    raise AssertionError('reference mutated its runtime inputs')
                keys = ('next_token','expert_ids','cache_length') if module is kimi else ('next_token','sparse_indices','expert_ids')
                replicated = {key:actual[key].cpu().tolist() for key in keys}
                all_ranks = [None]*current.gpus
                dist.all_gather_object(all_ranks,replicated)
                if any(value != replicated for value in all_ranks):
                    raise AssertionError('ranks disagree on global token, index selection or expert routing')
                details = None
                if not full:
                    expected = module.reference(serial_case,serial_values,serial=True) if rank == 0 else None
                    details = compare_serial(current,actual,expected,device,module is kimi)
                    if rank == 0 and step+1 < steps:
                        serial_case,serial_values = advance_state(serial_case,serial_values,expected)
                trajectory.append({'step':step,'input_context':current.params['context'],
                                   'reset':bool(values['reset']) if module is kimi else False,
                                   'next_token':int(actual['next_token']),'serial_comparison':details})
                if step+1 < steps:
                    current,values = advance_state(current,values,actual)
            if not full:
                del expected,serial_values
            values = values | {'token':(values['token']+1)%current.params['vocab']}
            changed = module.reference(current,values)
            if torch.equal(actual['logits'],changed['logits']):
                raise AssertionError('logits are independent of the runtime token')
            timing = _measure(lambda t:module.reference(case,t),perf_inputs,device,1,3)
            audit = _audit_launches(lambda t:module.reference(case,t),perf_inputs,device,case.max_gpu_launches)
            report['trials'].append({'seed':seed,'input_bytes':sum(v.numel()*v.element_size() for v in perf_inputs.values()),
                                     'trajectory':trajectory,'timing_context':case.params['context'],
                                     'verification_final_context':current.params['context'],
                                     'outputs':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in actual.items()},
                                     'reference_timing':timing,'reference_launch_audit':audit})
            print(json.dumps({'case':case.id,'rank':rank,'seed':seed,'status':'pass','full_geometry':full}),flush=True)
            del values,perf_inputs,actual,repeated,changed
    report['status'] = 'pass'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--trials',type=int,default=2)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    if min(args.trials,args.steps) < 1:
        parser.error('trials and steps must be positive')
    rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{rank}' if args.device == 'cuda' else 'cpu'
    if args.full and args.device != 'cuda':
        parser.error('full geometry verification requires scheduled GPUs')
    if args.device == 'cuda':
        torch.cuda.set_device(rank)
    dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo',timeout=timedelta(minutes=20),
                            device_id=torch.device(device) if args.device == 'cuda' else None)
    try:
        result = verify(select_cases('all',[args.case])[0],device,args.trials,args.full,args.steps)
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/f'rank-{rank}.json').open('x') as file:
            json.dump(result,file,indent=2)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
