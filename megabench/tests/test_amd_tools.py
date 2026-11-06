"""CPU tests for the AMD (ROCm) profiling tools: timeline tick scaling, the AMD codegen gate,
the stage-linking contract checker and the trace report's analysis."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from ..integrations import amd_trace_report, timeline

SKILLS = Path(__file__).resolve().parents[1] / "skills"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("codegen_gate_amd", SKILLS / "megakernel-phase-profiling" / "scripts" / "codegen_gate_amd.py")
plk = _load("plk_link", SKILLS.parent / "skills-optional" / "megakernel-stage-linking" / "scripts" / "plk_link.py")


def _buffer(blocks: int, stamps: dict[int, list[tuple[int, int, int]]]) -> bytes:
    """One group per block; stamps[block] = [(event, type, stamp)] in record order."""
    depth = max(len(v) for v in stamps.values())
    words = [0] * (1 + blocks * depth)
    words[0] = blocks | (1 << 32)
    for b, recs in stamps.items():
        for k, (event, typ, stamp) in enumerate(recs):
            words[1 + b + k * blocks] = (stamp << 32) | (b << 12) | (event << 2) | typ
    return struct.pack(f"<{len(words)}Q", *words)


class TimelineTickTests(unittest.TestCase):
    def test_tick_ns_scales_raw_wall_clock_ticks(self) -> None:
        data = _buffer(2, {0: [(0, 0, 1000), (0, 1, 1100)], 1: [(0, 0, 1000), (0, 1, 1150)]})
        ns = timeline.decode_buffer(data, ["phase"])
        scaled = timeline.decode_buffer(data, ["phase"], tick_ns=40.0)
        self.assertEqual(max(i.end_ns for i in ns.intervals), 150.0)
        self.assertEqual(max(i.end_ns for i in scaled.intervals), 6000.0)

    def test_load_reads_clock_json_beside_the_buffer(self) -> None:
        data = _buffer(1, {0: [(0, 0, 10), (0, 1, 35)]})
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "timeline.u64"
            raw.write_bytes(data)
            self.assertEqual(timeline.load(raw, ["p"]).intervals[0].end_ns, 25.0)
            (Path(d) / "clock.json").write_text(json.dumps({"tick_ns": 10.0}))
            self.assertEqual(timeline.load(raw, ["p"]).intervals[0].end_ns, 250.0)
            self.assertEqual(timeline.load(raw, ["p"], tick_ns=1.0).intervals[0].end_ns, 25.0)


NOTES = """
AMDGPU Metadata
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           32
        .value_kind:     by_value
    .group_segment_fixed_size: 2048
    .max_flat_workgroup_size: 512
    .name:           k1
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         k1.kd
    .uses_dynamic_stack: false
    .vgpr_count:     64
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     4
    .args:
      - .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .name:           k2
    .private_segment_fixed_size: 96
    .sgpr_count:     40
    .sgpr_spill_count: 3
    .symbol:         k2.kd
    .uses_dynamic_stack: true
    .vgpr_count:     132
    .vgpr_spill_count: 2
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
"""

DISASM = """
Disassembly of section .text:

<k1>:
\ts_load_dword s4, s[0:1], 0x2c                              // 000000001800: C0020100 0000002C
\ts_cbranch_scc1 <L0>
<L0>:
\tglobal_load_dword v1, v[2:3], off                          // 000000001808: DC508000 017F0002
\ts_endpgm
\ts_code_end
"""


class CodegenGateAmdTests(unittest.TestCase):
    def test_kernel_metadata_reads_per_kernel_resources(self) -> None:
        with unittest.mock.patch.object(gate, "run_tool", return_value=NOTES):
            meta = gate.kernel_metadata("x.co")
        self.assertEqual(set(meta), {"k1", "k2"})
        self.assertEqual(meta["k1"]["vgpr"], 64)
        self.assertEqual(meta["k1"]["lds"], 2048)
        self.assertEqual((meta["k2"]["scratch"], meta["k2"]["vgpr_spill"], meta["k2"]["sgpr_spill"]), (96, 2, 3))
        self.assertEqual(meta["k2"]["dyn_stack"], 1)

    def test_waves_per_simd_follow_the_512_vgpr_budget(self) -> None:
        r = {"vgpr": 64, "agpr": 0}
        self.assertEqual(gate.waves_per_simd(r), 8)
        self.assertEqual(gate.waves_per_simd({"vgpr": 129, "agpr": 0}), 3)
        self.assertEqual(gate.waves_per_simd({"vgpr": 256, "agpr": 0}), 2)

    def test_disassembly_keeps_labels_inside_the_kernel(self) -> None:
        with unittest.mock.patch.object(gate, "run_tool", return_value=DISASM):
            ks = gate.disassemble("x.co")
        self.assertEqual(list(ks), ["k1"])
        self.assertIn("<L0>:", ks["k1"])
        self.assertTrue(all("0000" not in line for line in ks["k1"]))
        self.assertEqual(ks["k1"][-1], "s_endpgm")

    def _resources(self, prod: dict, cand: dict, **kw) -> int:
        args = dict(production="p", candidate="c", kernel=None, arch=None, allow=None, threads_per_cta=0,
                    smem_per_cta=0, lds_per_cu=65536, max_vgprs=0, strict_regs=False, json=False)
        args.update(kw)
        base = {"vgpr": 64, "agpr": 0, "sgpr": 30, "lds": 0, "scratch": 0, "vgpr_spill": 0,
                "sgpr_spill": 0, "dyn_stack": 0, "max_wg": 512, "wavefront": 64}
        metas = iter([{"k": base | prod}, {"k": base | cand}])
        with unittest.mock.patch.object(gate, "code_objects", side_effect=lambda p, w, a: [p]), \
             unittest.mock.patch.object(gate, "kernel_metadata", side_effect=lambda co: next(metas)), \
             contextlib.redirect_stdout(io.StringIO()):
            return gate.cmd_resources(argparse.Namespace(**args))

    def test_scratch_and_spill_growth_fail_register_growth_warns(self) -> None:
        self.assertEqual(self._resources({}, {"scratch": 704}), 1)
        self.assertEqual(self._resources({}, {"vgpr_spill": 1}), 1)
        self.assertEqual(self._resources({}, {"scratch": 704}, allow="scratch"), 0)
        self.assertEqual(self._resources({"vgpr": 66}, {"vgpr": 72}), 0)  # both 7 waves/SIMD: warn only
        self.assertEqual(self._resources({"vgpr": 66}, {"vgpr": 72}, strict_regs=True), 1)
        self.assertEqual(self._resources({"vgpr": 120}, {"vgpr": 136}), 1)  # 4 -> 3 waves per SIMD


PKGS = {
    "a": {"src": "a.cu", "exports": ["stage_a"], "needs": ["x@l-1"], "provides": ["y@l", "reg"]},
    "b": {"src": "b.cu", "exports": ["stage_b"], "needs": ["y@l", "reg"], "consumes": ["reg"], "provides": ["x@l"]},
    "b_noreg": {"src": "b.cu", "exports": ["stage_b"], "needs": ["y@l"], "provides": ["x@l"]},
    "a_noreg": {"src": "a.cu", "exports": ["stage_a"], "needs": ["x@l-1"], "provides": ["y@l"]},
    "init": {"src": "i.cu", "exports": ["stage_init"], "provides": ["x@l"]},
    "drv": {"src": "d.cu", "exports": ["k"], "calls": ["stage_init", "stage_a", "stage_b"]},
}


class StageLinkingTests(unittest.TestCase):
    def pipe(self, a="a", b="b"):
        return {"slots": {"init": "init", "a": a, "b": b, "drv": "drv"},
                "sequence": ["init", {"repeat": ["a", "b"]}]}

    def test_valid_pipeline_and_loop_carried_handoff_pass(self) -> None:
        self.assertEqual(plk.check(PKGS, self.pipe()), [])
        self.assertEqual(plk.check(PKGS, self.pipe(b="b_noreg")), [])

    def test_missing_provider_is_rejected_by_name(self) -> None:
        errs = plk.check(PKGS, self.pipe(a="a_noreg"))
        self.assertTrue(errs and all(e.startswith("L-NEED slot b") for e in errs))

    def test_call_to_unlinked_export_is_rejected(self) -> None:
        pipe = self.pipe()
        pipe["slots"]["a"] = "b"
        errs = plk.check(PKGS, pipe)
        self.assertTrue(any(e.startswith("L-CALL package drv calls 'stage_a'") for e in errs))

    def test_unknown_package_is_rejected(self) -> None:
        pipe = self.pipe(a="missing")
        self.assertEqual(plk.check(PKGS, pipe), ["L-PKG slot a: unknown package missing"])


class TraceReportTests(unittest.TestCase):
    def test_candidate_is_the_capture_s_last_dispatch(self) -> None:
        disp = [{"name": "void at::native::fill<float>(int)", "dur_us": 900.0},
                {"name": "qwen3_step", "dur_us": 1400.0}, {"name": "qwen3_step", "dur_us": 1430.0}]
        self.assertEqual(amd_trace_report.pick_kernel(disp), "^qwen3_step")
        self.assertIsNone(amd_trace_report.pick_kernel([]))

    def test_findings_rank_wait_classes_and_bandwidth(self) -> None:
        rep = {"kernel": {"dur_us": 1000.0},
               "counters": {"values": {"FETCH_SIZE": 2.65e6, "TCC_HIT_sum": 1.0, "TCC_MISS_sum": 3.0}},
               "pc_sampling": {"samples": 100, "classes": {"barrier": 0.44, "mem-wait": 0.33, "valu": 0.23}},
               "timeline": {"phases": [{"event": "attn", "wall_us": 180.0, "slowest_us": 183.0, "mean_us": 66.0,
                                        "imbalance_us": 117.0, "begin_skew_us": 5.0}]}}
        text = "\n".join(amd_trace_report.findings(rep, 5.3, 0.5))
        self.assertIn("50.0% of SOL", text)
        self.assertIn("L2 hit rate 25.0%", text)
        self.assertIn("44% of samples are in sleep/barrier", text)
        self.assertIn("33% of samples are in s_waitcnt", text)
        self.assertIn("Largest imbalance: attn", text)

    def test_fatal_rocprof_log_kills_the_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "run.log"
            cmd = ["sh", "-c", "echo 'Could not construct profile cfg failed with error code 38'; sleep 60"]
            self.assertEqual(amd_trace_report._run(cmd, log, timeout=30), 125)
            self.assertIn("FATAL: killed process group", log.read_text())


if __name__ == "__main__":
    unittest.main()
