"""Offline model of M6/M7 instruction streams and dependency counters.

This is a schedule validator, not a CUDA interpreter or an M6/M7 gate pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig


@dataclass(frozen=True)
class Instruction:
    number: int
    kind: str
    layer: int
    tile: int
    sm: int
    wait_for: tuple[int, ...]
    shared_slot: int


@dataclass(frozen=True)
class Schedule:
    streams: tuple[tuple[Instruction, ...], ...]

    @property
    def instructions(self) -> tuple[Instruction, ...]:
        return tuple(sorted((task for stream in self.streams for task in stream),
                            key=lambda task: task.number))


def compile_schedule(cfg: ModelConfig, sm_count: int = 8, pool_slots: int = 2) -> Schedule:
    """Produce per-SM queues with explicit producer IDs and recycled slot IDs."""
    if sm_count <= 0 or pool_slots <= 0:
        raise ValueError("SM and slot counts must be positive")
    streams: list[list[Instruction]] = [[] for _ in range(sm_count)]
    next_id = 0

    def emit(kind: str, layer: int, tile: int, deps: tuple[int, ...]) -> int:
        nonlocal next_id
        sm = min(range(sm_count), key=lambda index: len(streams[index]))
        task = Instruction(next_id, kind, layer, tile, sm, tuple(sorted(set(deps))),
                           len(streams[sm]) % pool_slots)
        streams[sm].append(task)
        next_id += 1
        return task.number

    prior = (emit("embed", -1, 0, ()),)
    group_size = cfg.q_heads // cfg.kv_heads
    for layer in range(cfg.layers):
        norm = emit("rms_qkv", layer, 0, prior)
        q_proj = [emit("q_projection", layer, head, (norm,)) for head in range(cfg.q_heads)]
        kv_proj = [emit("kv_projection", layer, head, (norm,)) for head in range(cfg.kv_heads)]
        q_rope = [emit("q_rope", layer, head, (q_proj[head],)) for head in range(cfg.q_heads)]
        kv_write = [emit("kv_rope_store", layer, head, (kv_proj[head],)) for head in range(cfg.kv_heads)]
        attention = [
            emit("attention", layer, head, (q_rope[head], kv_write[head // group_size]))
            for head in range(cfg.q_heads)
        ]
        o_tiles = [emit("o_projection", layer, tile, tuple(attention)) for tile in range(4)]
        mlp_norm = emit("rms_mlp", layer, 0, tuple(o_tiles))
        gate_up = [emit("gate_up", layer, tile, (mlp_norm,)) for tile in range(4)]
        prior = tuple(emit("down_projection", layer, tile, tuple(gate_up)) for tile in range(4))
    emit("lm_head", cfg.layers, 0, prior)
    return Schedule(tuple(tuple(stream) for stream in streams))


def audit_schedule(schedule: Schedule, layers: int) -> dict[str, int | float | bool]:
    """Static checks and a monotonic-counter interpreter for deadlock testing."""
    tasks = schedule.instructions
    ids = {task.number for task in tasks}
    if ids != set(range(len(tasks))):
        raise ValueError("duplicate or missing instruction ID")
    for task in tasks:
        if any(dep >= task.number or dep not in ids for dep in task.wait_for):
            raise ValueError("dependency must name an earlier producer")
    lengths = [len(stream) for stream in schedule.streams]
    mean = sum(lengths) / len(lengths)
    imbalance = max(lengths) / mean if mean else 0.0
    types = len({task.kind for task in tasks})
    min_layer_instructions = min(
        (sum(task.layer == layer for task in tasks) for layer in range(layers)), default=0,
    )

    # Each instruction publishes one completion counter. An SM may advance only
    # its own queue, and waits only on the producers of its next instruction.
    cursor = [0] * len(schedule.streams)
    counters = [0] * len(tasks)
    steps = 0
    while sum(cursor) < len(tasks):
        progressed = False
        for sm, stream in enumerate(schedule.streams):
            if cursor[sm] == len(stream):
                continue
            task = stream[cursor[sm]]
            if all(counters[dep] == 1 for dep in task.wait_for):
                counters[task.number] += 1
                cursor[sm] += 1
                progressed = True
        if not progressed:
            raise ValueError("schedule deadlocked")
        steps += 1
    return {
        "instruction_types": types,
        "min_instructions_per_layer": min_layer_instructions,
        "queue_imbalance": imbalance,
        "distinct_completion_counters": len(counters),
        "all_counters_once": all(count == 1 for count in counters),
        "interpreter_rounds": steps,
        "paper_M6_static_targets": types >= 5 and min_layer_instructions >= 6 and imbalance <= 1.35,
        "paper_M7_counter_count_target": len(counters) >= 4 * layers,
    }
