"""Independent serial-oracle check for rank-local TP/EP references.

Launch with torchrun. Full geometries require the corresponding Slurm GPU
allocation; CPU mode is for small protocol checks only.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from ..cases import Case, select_cases
from ..harness.benchmark import _audit_launches, _measure
from ..harness.correctness import _compare
from ..tasks import distributed, gemma, moe
from ..verify_frontier import input_digest


def development(case: Case) -> Case:
    params = case.params | {"context": 4, "layers": 6, "hidden": 64,
                           "intermediate": 128, "q_heads": 4, "kv_heads": 2,
                           "head_dim": 16, "vocab": 256}
    if "gemma" in case.model.lower():
        params |= {"local_window": 3, "query_pre_attn_scalar": 16}
    else:
        params |= {"experts": 4, "topk": 2}
    return replace(case, params=params, atol=0.003, rtol=0.003)


def serial_oracle(case: Case, values: dict) -> dict:
    if "gemma" in case.model.lower():
        return gemma.reference(replace(case, params=case.params | {"bits": 16}), values)
    renamed = values | {"wrt": values["router"]}
    selected = []
    topk = torch.topk
    def capture_routing(logits, *args, **kwargs):
        result = topk(logits, *args, **kwargs)
        if logits.shape == (case.params['experts'],):
            selected.append(result.indices.clone())
        return result
    # Observe the independent decoder's router without changing its math.
    with patch('torch.topk', capture_routing):
        result = moe.reference(case, renamed)
    if len(selected) != case.params['layers']:
        raise AssertionError('serial oracle did not expose every layer\'s expert routing')
    return result | {'expert_ids':torch.stack(selected)}


def verify(case: Case, device: str, trials: int, full: bool) -> dict:
    rank = dist.get_rank()
    if not full:
        case = development(case)
    # The serial oracle owns complete weights only on rank zero, and never
    # imports the distributed forward. Canonical sampling makes shards exact.
    serial = replace(case, gpus=1, tp=1, ep=1)
    report = {"case": case.to_dict(), "rank": rank, "trials": []}
    distributed.initialize(case)
    with torch.inference_mode():
        for index in range(trials):
            seed = 104729 + index
            values = distributed.make_inputs(case, seed, device)
            before = input_digest(values)
            actual = distributed.reference(case, values)
            expected = None
            if rank == 0:
                all_values = distributed.make_inputs(serial, seed, device, rank=0)
                expected = serial_oracle(serial, all_values)
                del all_values
            dist.barrier()
            expected_shapes = {"logits": (case.params["vocab"],), "next_token": (),
                               "k_write": (case.params["layers"], case.params["kv_heads"], case.params["head_dim"]),
                               "v_write": (case.params["layers"], case.params["kv_heads"], case.params["head_dim"])}
            if 'experts' in case.params:
                expected_shapes['expert_ids'] = (case.params['layers'],case.params['topk'])
            if rank != 0:
                expected = {name: torch.empty(shape, device=device,
                            dtype=torch.int64 if name in ("next_token", "expert_ids") else
                            torch.bfloat16 if name.endswith("write") else torch.float32)
                            for name, shape in expected_shapes.items()}
            for name in expected_shapes:
                dist.broadcast(expected[name], src=0)
            tensor_rank = rank % case.tp
            chunk = case.params["vocab"] // case.tp
            wanted = {"logits": expected["logits"][tensor_rank * chunk:(tensor_rank + 1) * chunk],
                      "next_token": expected["next_token"],
                      "k_write": expected["k_write"].chunk(case.tp, dim=1)[tensor_rank],
                      "v_write": expected["v_write"].chunk(case.tp, dim=1)[tensor_rank]}
            if 'experts' in case.params:
                wanted['expert_ids'] = expected['expert_ids']
            detail = _compare(wanted, {name: actual[name] for name in wanted}, case, device)
            if before != input_digest(values):
                raise AssertionError('reference mutated its runtime inputs')
            timing = _measure(lambda inputs: distributed.reference(case, inputs), values, device, 1, 3)
            audit = _audit_launches(lambda inputs: distributed.reference(case, inputs), values, device, 1)
            # The eager multi-rank reference is a baseline, not a one-launch candidate.
            report["trials"].append({"seed": seed, "outputs": detail,
                                     "reference_timing": timing, "reference_launch_audit": audit,
                                     "input_bytes": sum(value.numel() * value.element_size() for value in values.values())})
            print(json.dumps({"case": case.id, "rank": rank, "seed": seed, "status": "pass"}), flush=True)
            del values, before, actual, expected, wanted
    report["status"] = "pass"
    if device.startswith('cuda'):
        report['gpu_memory'] = {'total_bytes':torch.cuda.get_device_properties(device).total_memory,
                                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                                'includes_serial_oracle_on_rank_zero':rank == 0}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{rank}" if args.device == "cuda" else "cpu"
    if args.device == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo", timeout=timedelta(minutes=10),
                            device_id=torch.device(device) if args.device == 'cuda' else None)
    try:
        case = select_cases("all", [args.case])[0]
        result = verify(case, device, args.trials, args.full)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"rank-{rank}.json").open("x") as file:
            json.dump(result, file, indent=2)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
