"""Read ``ls::stamp`` records and report where a megakernel's time goes.

Device side: build with ``-DLS_PROFILE`` and pass ``StampBuffer.ptr`` /
``.capacity`` as the kernel's ``ls::Stamps``. Host side::

    stamps = StampBuffer(grid, capacity=256, device="cuda")
    stamps.clear(); launch(...); torch.cuda.synchronize()
    print(report(stamps.read(), {1: "start", 2: "wait_x", ...}, bytes_total=...))

Name tags "role:name" (e.g. "prod:layer", "cons:wait_x") so producer and
consumer stamps form separate sequences. The report gives, for every
consecutive pair of stamps of one role in a CTA (``prev_tag -> tag``), the total, mean and max time across CTAs: the gaps
named ``wait_* -> got_*`` are bubbles, the rest is streaming or compute. It
also gives per-CTA end-time skew (imbalance) and the achieved bandwidth.
``chrome_trace`` writes a Perfetto/chrome://tracing JSON timeline.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import torch


class StampBuffer:
    def __init__(self, grid: int, capacity: int, device: str = "cuda"):
        self.grid, self.capacity = grid, capacity
        self.tensor = torch.zeros(grid, capacity, 2, dtype=torch.int64, device=device)

    @property
    def ptr(self) -> int:
        return self.tensor.data_ptr()

    def clear(self) -> None:
        self.tensor.zero_()

    def read(self) -> list[list[tuple[int, int, int]]]:
        """Per CTA: [(time_ns, tag, value), ...] sorted by time."""
        data = self.tensor.cpu()
        rows = []
        for cta in range(self.grid):
            count = min(int(data[cta, 0, 0]), self.capacity - 1)
            if int(data[cta, 0, 0]) > self.capacity - 1:
                print(f"warning: CTA {cta} dropped {int(data[cta, 0, 0]) - count} stamps; raise capacity")
            recs = [(int(data[cta, i, 0]), int(data[cta, i, 1]) >> 32, int(data[cta, i, 1]) & 0xFFFFFFFF)
                    for i in range(1, count + 1)]
            rows.append(sorted(recs))
        return rows


@dataclass
class Gap:
    total_us: float = 0.0
    count: int = 0
    max_us: float = 0.0

    def add(self, us: float) -> None:
        self.total_us += us
        self.count += 1
        self.max_us = max(self.max_us, us)


def _name(tags: dict[int, str], tag: int) -> str:
    return tags.get(tag, f"tag{tag}")


def _by_role(row: list[tuple[int, int, int]], tags: dict[int, str]) -> dict[str, list[tuple[int, int, int]]]:
    roles: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for rec in row:
        name = _name(tags, rec[1])
        roles[name.split(":", 1)[0] if ":" in name else "main"].append(rec)
    return roles


def report(rows: list[list[tuple[int, int, int]]], tags: dict[int, str], *, bytes_total: int | None = None,
           bandwidth_tb_s: float = 8.0, top: int = 25) -> str:
    stamps = [r for row in rows for r in row]
    if not stamps:
        return "no stamps recorded (built without -DLS_PROFILE, or buffer not passed?)"
    t0 = min(t for t, _, _ in stamps)
    t1 = max(t for t, _, _ in stamps)
    span_us = (t1 - t0) / 1e3
    gaps: dict[str, Gap] = defaultdict(Gap)
    ends = []
    for row in rows:
        if not row:
            continue
        ends.append((row[-1][0] - t0) / 1e3)
        # Intervals are between consecutive stamps of the same role; a tag
        # named "role:name" belongs to that role ("main" otherwise).
        for role_row in _by_role(row, tags).values():
            for (ta, a, _), (tb, b, _) in zip(role_row, role_row[1:]):
                gaps[f"{_name(tags, a)} -> {_name(tags, b)}"].add((tb - ta) / 1e3)
    n = len([r for r in rows if r])
    lines = [f"span {span_us:.1f} us over {n} CTAs; CTA end skew: min {min(ends):.1f} / "
             f"median {sorted(ends)[len(ends) // 2]:.1f} / max {max(ends):.1f} us"]
    if bytes_total:
        floor = bytes_total / (bandwidth_tb_s * 1e12) * 1e6
        lines.append(f"{bytes_total / 1e6:.1f} MB in {span_us:.1f} us = {bytes_total / span_us / 1e6:.2f} TB/s; "
                     f"floor {floor:.1f} us at {bandwidth_tb_s} TB/s = {100 * floor / span_us:.1f}% of SOL")
    lines.append(f"{'interval (per CTA, summed over occurrences)':58s} {'mean/CTA us':>11s} {'max us':>9s} {'count':>7s}")
    for name, g in sorted(gaps.items(), key=lambda kv: -kv[1].total_us)[:top]:
        lines.append(f"{name:58s} {g.total_us / n:11.2f} {g.max_us:9.2f} {g.count:7d}")
    return "\n".join(lines)


def per_value(rows: list[list[tuple[int, int, int]]], tag: int) -> dict[int, list[float]]:
    """Times (us from kernel start) at which each CTA recorded ``tag``, keyed by the stamp value
    (e.g. the layer index) — use to see when each layer's phase starts across CTAs."""
    t0 = min(t for row in rows for t, _, _ in row)
    out: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        for t, g, v in row:
            if g == tag:
                out[v].append((t - t0) / 1e3)
    return dict(out)


def chrome_trace(rows: list[list[tuple[int, int, int]]], tags: dict[int, str], path: str) -> None:
    """Write intervals between consecutive stamps as complete events, one track per CTA."""
    t0 = min(t for row in rows for t, _, _ in row)
    events = []
    for cta, row in enumerate(rows):
        for role, role_row in _by_role(row, tags).items():
            for (ta, a, va), (tb, b, _) in zip(role_row, role_row[1:]):
                events.append({"name": f"{_name(tags, a)}->{_name(tags, b)}", "ph": "X", "pid": role,
                               "tid": cta, "ts": (ta - t0) / 1e3, "dur": max((tb - ta) / 1e3, 0.001),
                               "args": {"value": va}})
    with open(path, "w") as f:
        json.dump({"traceEvents": events}, f)
