"""Tests for the megakernel-phase-profiling codegen gate script."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] / "skills" / "megakernel-phase-profiling"
          / "scripts" / "codegen_gate.py")
spec = importlib.util.spec_from_file_location("codegen_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

SASS_A = """
\tcode for sm_100a
\t\tFunction : k1
\t.headerflags\t@"EF_CUDA_SM100"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;               /* 0x0000df00ff017b82 */
                                                                         /* 0x000fe20000000800 */
        /*0010*/                   BRA 0x1a0 ;                           /* 0x0000000000fc7947 */
                                                                         /* 0x000fea0003800000 */
"""
SASS_B = SASS_A.replace("/*0000*/", "/*0abc*/").replace("0x1a0", "0x2b0").replace("0x0000df00ff017b82", "0x1")

RES = """Resource usage:
 Common:
  GLOBAL:0
 Function k1:
  REG:12 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:912 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function k2:
  REG:40 STACK:16 SHARED:1024 LOCAL:16 CONSTANT[0]:100
"""


class PureTests(unittest.TestCase):
    def test_normalize_strips_offsets_encodings_and_addresses(self) -> None:
        a, b = gate.normalize_sass(SASS_A), gate.normalize_sass(SASS_B)
        self.assertEqual(a, b)
        self.assertEqual(a[1], "LDC R1, c[0x0][0x37c] ;")
        self.assertTrue(all("code for" not in x and "headerflags" not in x for x in a))

    def test_normalize_keeps_real_differences(self) -> None:
        self.assertNotEqual(gate.normalize_sass(SASS_A), gate.normalize_sass(SASS_A.replace("R1", "R2")))

    def test_split_by_function(self) -> None:
        funcs = gate.split_by_function(SASS_A + "\t\tFunction : k2\n        /*0000*/ EXIT ; /* 0x0 */\n")
        self.assertEqual(set(funcs), {"k1", "k2"})

    def test_parse_res_usage(self) -> None:
        res = gate.parse_res_usage(RES)
        self.assertEqual(res["k1"]["REG"], 12)
        self.assertEqual(res["k2"]["SHARED"], 1024)
        self.assertEqual(res["k2"]["LOCAL"], 16)

    def test_parse_ptxas_log(self) -> None:
        log = ("ptxas info    : Compiling entry function 'k1' for 'sm_100a'\n"
               "ptxas info    : Used 12 registers, 8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads\n")
        # The line above is split by ptxas over two lines in practice; both forms must parse.
        log = ("ptxas info    : Compiling entry function 'k1' for 'sm_100a'\n"
               "ptxas info    : 8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads\n"
               "ptxas info    : Used 12 registers\n")
        self.assertEqual(gate.parse_ptxas_log(log)["k1"], {"SPILL": 8, "STACK": 8, "REG": 12})

    def test_compare_resources(self) -> None:
        prod = {"k": {"REG": 10, "STACK": 0, "SHARED": 0, "LOCAL": 0}}
        more = {"k": {"REG": 12, "STACK": 0, "SHARED": 0, "LOCAL": 0}}
        warn = gate.compare_resources(prod, more, set())
        self.assertTrue(warn["ok"])
        self.assertTrue(warn["kernels"][0]["status"].startswith("WARN"))
        self.assertFalse(gate.compare_resources(prod, more, set(), strict_regs=True)["ok"])
        self.assertTrue(gate.compare_resources(prod, more, {"reg"}, strict_regs=True)["ok"])
        self.assertFalse(gate.compare_resources(prod, more, set(), max_regs=11)["ok"])
        self.assertTrue(gate.compare_resources(more, prod, set())["ok"])
        self.assertFalse(gate.compare_resources(prod, {}, set())["ok"])

    def test_spill_stack_shared_growth_fails(self) -> None:
        base = {"REG": 10, "STACK": 0, "SHARED": 0, "LOCAL": 0}
        for field in ("LOCAL", "STACK", "SHARED"):
            grown = dict(base, **{field: 16})
            self.assertFalse(gate.compare_resources({"k": base}, {"k": grown}, set())["ok"], field)
            self.assertTrue(gate.compare_resources({"k": base}, {"k": grown}, {"smem", "stack", "spill"})["ok"], field)
        self.assertFalse(gate.compare_resources({"k": {"REG": 1, "SPILL": 0}}, {"k": {"REG": 1, "SPILL": 8}}, set())["ok"])

    def test_occupancy_drop_fails_and_no_drop_passes(self) -> None:
        # 128 threads: 64 regs -> 8 CTAs/SM by registers; 72 regs -> 7.
        self.assertEqual(gate.ctas_per_sm(64, 128, 0), 8)
        self.assertEqual(gate.ctas_per_sm(72, 128, 0), 7)
        mk = lambda r: {"k": {"REG": r, "STACK": 0, "SHARED": 0, "LOCAL": 0}}
        drop = gate.compare_resources(mk(64), mk(72), set(), threads_per_cta=128)
        self.assertFalse(drop["ok"])
        self.assertEqual(drop["kernels"][0]["ctas_per_sm"], {"production": 8, "candidate": 7})
        # 24 -> 32 regs: still 16 CTAs/SM, so no occupancy change.
        self.assertTrue(gate.compare_resources(mk(24), mk(32), set(), threads_per_cta=128)["ok"])
        # shared memory limit: dynamic smem decides occupancy, so register growth does not lower it.
        self.assertTrue(gate.compare_resources(mk(24), mk(32), set(), threads_per_cta=128,
                                               smem_per_cta=100 * 1024)["ok"])


@unittest.skipUnless(shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists(), "nvcc not found")
class IntegrationTests(unittest.TestCase):
    KERNEL = r"""
extern "C" __global__ void scale(float* a, long long* t) {
#ifdef PROF
  if (threadIdx.x == 0) t[blockIdx.x] = clock64();
#endif
  a[blockIdx.x * blockDim.x + threadIdx.x] *= 2.f;
}
"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cuda_bin = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin")
        self.env = dict(os.environ, PATH=os.environ["PATH"] + os.pathsep + cuda_bin + os.pathsep + "/usr/local/cuda/bin")
        patcher = unittest.mock.patch.dict(os.environ, {"PATH": self.env["PATH"]})
        patcher.start()
        self.addCleanup(patcher.stop)
        src = Path(self.tmp.name) / "k.cu"
        src.write_text(self.KERNEL)
        for name, flags in (("prod", []), ("off", []), ("on", ["-DPROF"])):
            out = Path(self.tmp.name) / f"{name}.cubin"
            subprocess.run(["nvcc", "-cubin", "-arch=sm_100a", *flags, str(src), "-o", str(out)],
                           check=True, env=self.env, capture_output=True)
        self.p = lambda n: str(Path(self.tmp.name) / f"{n}.cubin")

    def run_gate(self, *args: str) -> int:
        return gate.main(list(args))

    def test_identical_passes_for_off_and_fails_for_on(self) -> None:
        self.assertEqual(self.run_gate("identical", "--production", self.p("prod"), "--candidate", self.p("off")), 0)
        self.assertEqual(self.run_gate("identical", "--production", self.p("prod"), "--candidate", self.p("on"),
                                       "--kernel", "scale"), 1)

    def test_resources_reports_both(self) -> None:
        res = gate.resources(self.p("on"), None, None)
        self.assertIn("scale", res)
        self.assertGreater(res["scale"]["REG"], 0)
        rc = self.run_gate("resources", "--production", self.p("prod"), "--candidate", self.p("on"), "--threads-per-cta", "128")
        self.assertEqual(rc, 0)
        self.assertEqual(self.run_gate("resources", "--production", self.p("prod"), "--candidate", self.p("on"),
                                       "--strict-regs"), 1)
        self.assertEqual(self.run_gate("resources", "--production", self.p("prod"), "--candidate", self.p("off")), 0)

    def test_usage_error_exit_code(self) -> None:
        self.assertEqual(self.run_gate("identical", "--production", "/nonexistent.cubin",
                                       "--candidate", "/nonexistent.cubin"), 2)


if __name__ == "__main__":
    unittest.main()
