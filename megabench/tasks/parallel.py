"""Rank mapping and process groups shared by distributed model references."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist

from ..cases import Case


@dataclass(frozen=True)
class ParallelContext:
    rank: int
    tp_rank: int
    ep_rank: int
    tp_group: object
    ep_group: object


_CONTEXT: tuple[object, int, int, int, ParallelContext] | None = None


def initialize(case: Case) -> ParallelContext:
    """Every rank creates groups in the same order: rank = ep_rank*tp+tp_rank."""
    global _CONTEXT
    if not dist.is_initialized() or dist.get_world_size() != case.gpus:
        raise ValueError(f"{case.id} requires an initialized {case.gpus}-rank process group")
    if case.gpus != case.tp * case.ep:
        raise ValueError("GPU count must equal TP times EP")
    rank = dist.get_rank()
    world_group = dist.group.WORLD
    if _CONTEXT and _CONTEXT[:4] == (world_group, case.tp, case.ep, rank):
        return _CONTEXT[4]
    tp_group, ep_group = None, None
    for expert_rank in range(case.ep):
        ranks = [expert_rank * case.tp + tensor_rank for tensor_rank in range(case.tp)]
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
    for tensor_rank in range(case.tp):
        ranks = [expert_rank * case.tp + tensor_rank for expert_rank in range(case.ep)]
        group = dist.new_group(ranks)
        if rank in ranks:
            ep_group = group
    context = ParallelContext(rank, rank % case.tp, rank // case.tp, tp_group, ep_group)
    _CONTEXT = (world_group, case.tp, case.ep, rank, context)
    return context
