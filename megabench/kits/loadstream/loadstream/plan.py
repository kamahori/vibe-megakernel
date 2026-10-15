"""Plan a load-centric megakernel before writing it: a byte ledger per phase,
per-SM balance of row bands, ring sizing, and the time each phase should take.

    from loadstream.plan import Ledger
    led = Ledger(sms=148, tb_s=7.0)                   # use the measured streaming ceiling
    for l in range(28):
        led.matrix(f"L{l}.qkv", rows=4096, row_bytes=1024 * 2)
        led.matrix(f"L{l}.o", rows=1024, row_bytes=2048 * 2)
        led.matrix(f"L{l}.gate_up", rows=2 * 3072, row_bytes=1024 * 2, granule=2)
        led.matrix(f"L{l}.down", rows=1024, row_bytes=3072 * 2)
        led.sync(f"L{l}", events=4)                    # dependency points on the critical path
    led.matrix("lm_head", rows=151936, row_bytes=1024 * 2)
    print(led.report())

``report`` prints the bytes, the ideal streaming time, the worst-SM
imbalance of each matrix's row bands, and the overhead that the listed
dependency points add if each costs ``sync_us`` with nothing streaming
behind it (the reason the producer must run ahead).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def band(units: int, parts: int, p: int) -> tuple[int, int]:
    """Same split as ``ls::band`` in plan.cuh."""
    base, extra = divmod(units, parts)
    begin = p * base + min(p, extra)
    return begin, begin + base + (1 if p < extra else 0)


def littles_law_bytes(tb_s: float = 8.0, latency_us: float = 0.8, sms: int = 148) -> int:
    """Bytes each SM must keep in flight to sustain ``tb_s`` at ``latency_us`` memory latency."""
    return int(tb_s * 1e12 * latency_us * 1e-6 / sms)


@dataclass
class Entry:
    name: str
    bytes: int
    worst_sm_bytes: int
    sms_used: int


@dataclass
class Ledger:
    sms: int = 148
    tb_s: float = 8.0
    sync_us: float = 1.3
    entries: list[Entry] = field(default_factory=list)
    syncs: int = 0

    def matrix(self, name: str, rows: int, row_bytes: int, granule: int = 1, sms: int | None = None) -> Entry:
        """A weight matrix streamed once, row-banded over ``sms`` SMs (all by default)."""
        parts = sms or self.sms
        units = -(-rows // granule)
        worst = max(band(units, parts, p)[1] - band(units, parts, p)[0] for p in range(parts))
        entry = Entry(name, rows * row_bytes, min(worst * granule, rows) * row_bytes, min(parts, units))
        self.entries.append(entry)
        return entry

    def bytes(self, name: str, nbytes: int) -> Entry:
        """Other traffic (KV cache, embeddings, outputs), assumed perfectly balanced."""
        entry = Entry(name, nbytes, -(-nbytes // self.sms), self.sms)
        self.entries.append(entry)
        return entry

    def sync(self, name: str, events: int = 1) -> None:
        self.syncs += events

    def report(self, top: int = 12) -> str:
        per_sm_bw = self.tb_s * 1e12 / self.sms  # bytes/s
        total = sum(e.bytes for e in self.entries)
        ideal_us = total / (self.tb_s * 1e12) * 1e6
        banded_us = sum(e.worst_sm_bytes for e in self.entries) / per_sm_bw * 1e6
        sync_us = self.syncs * self.sync_us
        lines = [
            f"{total / 1e6:.1f} MB -> {ideal_us:.1f} us at {self.tb_s} TB/s "
            f"(8 TB/s SOL {total / 8e6:.1f} us)",
            f"worst-SM banding: {banded_us:.1f} us (+{100 * (banded_us / ideal_us - 1):.1f}% imbalance)",
            f"{self.syncs} dependency points x {self.sync_us} us = {sync_us:.1f} us if nothing streams behind them "
            f"(serial worst case {banded_us + sync_us:.1f} us = {100 * total / 8e6 / (banded_us + sync_us):.0f}% of SOL)",
            f"Little's law: keep >= {littles_law_bytes(self.tb_s, 0.8, self.sms) / 1024:.0f} KB in flight per SM; "
            f"a {self.sync_us} us stall needs {per_sm_bw * self.sync_us * 1e-6 / 1024:.0f} KB of ring to hide",
            f"{'entry':32s} {'MB':>9s} {'SMs':>5s} {'worst-SM KB':>12s} {'imbalance':>9s}",
        ]
        worst = sorted(self.entries, key=lambda e: -(e.worst_sm_bytes * self.sms - e.bytes))
        for e in worst[:top]:
            lines.append(f"{e.name:32s} {e.bytes / 1e6:9.2f} {e.sms_used:5d} {e.worst_sm_bytes / 1024:12.1f} "
                         f"{100 * (e.worst_sm_bytes * self.sms / max(e.bytes, 1) - 1):8.1f}%")
        return "\n".join(lines)
