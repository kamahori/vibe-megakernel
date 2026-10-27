"""Check the timeline decoder and, on a GPU, the probe header and capture tool end to end."""

from __future__ import annotations

import json
import random
import struct
import tempfile
import textwrap
import unittest
from collections import Counter
from pathlib import Path

from ..integrations import timeline
from ..integrations.timeline import TraceError

WRAP = 1 << 32
HEADER = Path(__file__).resolve().parents[1] / "integrations/timeline_probe.cuh"


def tag(sm: int, block: int, event: int, kind: int) -> int:
    return (sm << 24) | (block << 12) | (event << 2) | kind


def encode(writers: list[list[tuple[int, int, int]]], blocks: int, groups: int,
           base: int) -> bytes:
    """Lay out (event, gap, duration) lists the way the probes write them."""
    stride = blocks * groups
    words = [0] * (stride * 2 * max(1, max(len(events) for events in writers)))
    for slot, events in enumerate(writers):
        clock = base + slot
        for k, (event, gap, duration) in enumerate(events):
            begin = clock + gap
            clock = begin + duration
            for offset, (kind, stamp) in enumerate(((0, begin), (1, clock))):
                words[slot + (2 * k + offset) * stride] = (
                    # SM ids from 1 keep every tag nonzero: a zero word is an unused slot.
                    tag(slot % 255 + 1, slot // groups, event, kind) | ((stamp % WRAP) << 32))
    return struct.pack("<ii", blocks, groups) + struct.pack(f"<{len(words)}Q", *words)


def shape(decoded: timeline.Timeline) -> Counter:
    """Intervals by track, event and duration, to 1 ns (the Chrome export is in float µs)."""
    return Counter((item.track, item.name, round(item.duration_ns)) for item in decoded.intervals)


class DecoderTests(unittest.TestCase):
    def test_random_buffers_decode_to_exactly_the_recorded_intervals(self) -> None:
        rng = random.Random(7301)
        names = ("load", "compute", "reduce")
        for _ in range(300):
            blocks, groups = rng.randint(1, 5), rng.randint(1, 3)
            writers = [[(rng.randrange(3), rng.randint(0, 5000), rng.randint(1, 50000))
                        for _ in range(rng.randint(0, 4))] for _ in range(blocks * groups)]
            writers[0] = writers[0] or [(0, 0, 1)]
            data = encode(writers, blocks, groups, base=rng.randrange(WRAP))  # may wrap
            decoded = timeline.decode_buffer(data, names)
            expected = Counter((names[e], slot // groups, slot % groups, d)
                               for slot, events in enumerate(writers) for e, _, d in events)
            actual = Counter((item.name, *map(int, item.track.split()[0][3:].split("_g")),
                              round(item.duration_ns)) for item in decoded.intervals)
            self.assertEqual(actual, expected)
            self.assertEqual(decoded.unmatched, 0)
            round_trip = timeline.decode_chrome_trace(json.dumps(timeline.to_chrome_trace(decoded)))
            self.assertEqual(shape(round_trip), shape(decoded))

    def test_event_skew_names_the_block_that_finished_last(self) -> None:
        # Block 1 starts 100 ns later and runs 300 ns longer.
        data = encode([[(0, 0, 500)], [(0, 99, 800)]], blocks=2, groups=1, base=1000)
        skew = timeline.event_skew(timeline.decode_buffer(data, ["load"]), "load", top=5)
        self.assertEqual(skew["totals"], {"wall_us": 0.9, "begin_skew_us": 0.1,
                                          "slowest_us": 0.8, "mean_us": 0.65,
                                          "imbalance_us": 0.15})
        self.assertEqual(skew["last_finishers"], [{"track": "blk1_g0 sm2", "times_last": 1}])
        self.assertIn("blk1_g0 sm2", timeline.format_table(
            timeline.phase_table(timeline.decode_buffer(data, ["load"]))))

    def test_invalid_input_raises_only_trace_error(self) -> None:
        rng = random.Random(17)
        samples = [rng.randbytes(rng.randint(0, 96)) for _ in range(2000)]
        samples += [b"", b"{", b"[]", b'{"traceEvents": 3}', b'[{"ph": "X", "ts": 1e400}]',
                    b"[" * 100000, struct.pack("<ii", 0, 1) + bytes(8),
                    struct.pack("<ii", 1, 1) + struct.pack("<Q", 3)]
        for data in samples:
            for decode in (timeline.decode_buffer, timeline.decode_chrome_trace):
                try:
                    decoded = decode(data)
                except TraceError:
                    continue
                self.assertTrue(decoded.intervals)

    def test_unnamed_event_index_is_rejected(self) -> None:
        data = encode([[(2, 0, 10)]], blocks=1, groups=1, base=0)
        with self.assertRaisesRegex(TraceError, "event index 2 has no name"):
            timeline.decode_buffer(data, ["a", "b"])


SUBMISSION = textwrap.dedent('''
    import os
    from pathlib import Path

    import torch
    from torch.utils.cpp_extension import load_inline

    SOURCE = r"""
    #include <torch/extension.h>
    #include "timeline_probe.cuh"
    __global__ void phases(unsigned long long* buffer, unsigned long long words) {
      MB_TIMELINE_INIT(reinterpret_cast<uint64_t*>(buffer), words, threadIdx.x / 32, 2,
                       threadIdx.x % 32 == 0);
      for (int layer = 0; layer < 3; ++layer) {
        MB_TIMELINE_BEGIN(0);
        for (volatile int i = 0; i < 1000 * (1 + blockIdx.x % 4); ++i) {}
        MB_TIMELINE_END(0);
        MB_TIMELINE_BEGIN(1);
        __syncthreads();
        MB_TIMELINE_END(1);
      }
    }
    void launch(torch::Tensor buffer, int64_t blocks) {
      phases<<<blocks, 64>>>(reinterpret_cast<unsigned long long*>(buffer.data_ptr()),
                             buffer.numel());
    }
    """

    def build(case):
        raise NotImplementedError

    def build_timeline(case):
        ext = load_inline("megabench_timeline_test", cpp_sources="void launch(torch::Tensor, int64_t);",
                          cuda_sources=SOURCE, functions=["launch"],
                          extra_include_paths=[os.environ["MEGABENCH_TIMELINE_HEADER_DIR"]],
                          extra_cuda_cflags=["-DMEGABENCH_TIMELINE"],
                          build_directory=os.environ["MEGABENCH_TIMELINE_BUILD_DIR"])
        blocks = 8

        def run(inputs):
            buffer = torch.zeros(1 + blocks * 2 * 12, dtype=torch.int64, device=inputs["token"].device)
            ext.launch(buffer, blocks)
            return buffer

        return run, ["work", "sync"]
''')


class CaptureTests(unittest.TestCase):
    def test_probe_header_and_capture_tool_record_every_block_and_phase(self) -> None:
        import os

        import torch

        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA GPU")
        from ..cases import select_cases
        from ..integrations import timeline_profile_case

        case = next(case for case in select_cases("p0") if case.gpus == 1 and case.ready)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "submission.py").write_text(SUBMISSION)
            (root / "build").mkdir()
            os.environ["MEGABENCH_TIMELINE_HEADER_DIR"] = str(HEADER.parent)
            os.environ["MEGABENCH_TIMELINE_BUILD_DIR"] = str(root / "build")
            timeline_profile_case.main(["--case", case.id, "--submission",
                                        str(root / "submission.py"), "--output",
                                        str(root / "out")])
            decoded = timeline.load(root / "out/timeline.u64", ["work", "sync"])
            self.assertEqual(decoded.unmatched, 0)
            self.assertEqual(len({item.track for item in decoded.intervals}), 16)
            self.assertEqual(Counter(item.name for item in decoded.intervals),
                             {"work": 48, "sync": 48})
            last = timeline.event_skew(decoded, "work", top=1)["last_finishers"][0]["track"]
            self.assertIn(last.split("_")[0], {"blk3", "blk7"})  # the longest-looping blocks
            self.assertEqual(shape(timeline.load(root / "out/trace.json")), shape(decoded))


if __name__ == "__main__":
    unittest.main()
