"""Megakernel arm: the Astra cooperative Qwen3-0.6B decode kernel, several launch mechanisms.

The candidate (``submission.py`` + ``candidate_helpers/decode.cu``) is read-only.
It is copied into ``$MEGABENCH_SOTA_CACHE/candidates/<sha16>/`` and the copy is
imported and built (original nvcc flags, original ``_library``/``_host_library``),
so nothing is ever built inside the source tree.

Variants (``direct`` is the headline):

* ``direct``: one ctypes call of ``launch`` per step, all buffers preallocated.
* ``graph``: the same call captured in a CUDA graph (cooperative graph node).
* ``direct-pdl`` / ``direct-pdl-early``: step-to-step Programmatic Dependent
  Launch on a harness-owned patched copy of decode.cu (see ``patch_pdl``).
* ``submission``: the candidate's own runner (torch op, allocates per call).

PDL overlap: the next step's launch latency and RoPE prologue (register-only
sin/cos) run while the previous step's LM-head/argmax tail is still executing;
``griddepcontrol.wait`` then blocks until the previous grid completed and its
writes are visible, so token/next aliasing in chained mode stays correct. In
the early variant the first statement is ``griddepcontrol.launch_dependents``,
so the next grid may start filling the free third CTA slot per SM (three CTAs
fit, the kernel uses two) while the previous grid is still running.
"""

from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from ..shapes import BASE_CASE_ID, Shape, megabench_case

DEFAULT_SUBMISSION = Path(
    "/raid/keisuke/vibe-megakernel/megabench/experiments/2026-10-06/"
    "20-40-09-astra-dense-1ms/bulk_cpp_launch/submission.py")
DEFAULT_CACHE = "/raid/garv901/.cache/megabench_sota"
VARIANTS = ("direct", "graph", "direct-pdl", "direct-pdl-early", "submission")
RESOURCE_NAMES = ("cooperative", "sms", "max_blocks_per_sm", "regs", "smem",
                  "local", "scratch_bytes")
OUTPUT_KEYS = ("logits", "next_token", "k_write", "v_write")
INPUT_ORDER = ("token", "ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2", "wg",
               "wu", "wd", "fnorm", "embed", "kcache", "vcache")

# ---------------------------------------------------------------------------
# PDL source patch (exact-string insertions, fail closed)

_KERNEL_HEAD = "__global__ void decode(Args a){\n"
_ROPE_LINE = ("float rope_cos=(float)cos(rope_angle),"
              "rope_sin=(float)sin(rope_angle);\n")
_FLAG_ZERO = "if(tid<FLAG_COUNT)ready[tid]=0;"
_LAUNCH_DEPS = ' asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
_WAIT = ' asm volatile("griddepcontrol.wait;" ::: "memory");\n'
_LAUNCH_PDL = r'''
// MegaBench harness addition: PDL launch of the same cooperative kernel.
extern "C" int launch_pdl(void** p,void* stream){
 Args a;static_assert(sizeof(a)==22*sizeof(void*),"pointer layout");memcpy(&a,p,sizeof(a));
 cudaLaunchConfig_t cfg={};
 cfg.gridDim=dim3(296);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=65536;cfg.stream=(cudaStream_t)stream;
 cudaLaunchAttribute at[2];
 at[0].id=cudaLaunchAttributeCooperative;at[0].val.cooperative=1;
 at[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;at[1].val.programmaticStreamSerializationAllowed=1;
 cfg.attrs=at;cfg.numAttrs=2;
 return (int)cudaLaunchKernelEx(&cfg,decode,a);
}
'''


def _once(src: str, anchor: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise RuntimeError(f"PDL patch anchor must occur exactly once, found {n}: {anchor!r}")


def patch_pdl(src: str, early: bool) -> str:
    """Return decode.cu with griddepcontrol.wait after the register-only RoPE prologue.

    Everything between the kernel head and the wait is register/kernel-param
    only (pow/cos/sin); the shared declarations and pointer arithmetic that
    follow touch no global memory, and the flag zeroing ``ready[tid]=0`` (a
    global write) comes after the wait.
    """
    for anchor in (_KERNEL_HEAD, _ROPE_LINE, _FLAG_ZERO,
                   'extern "C" int launch(void** p,void* stream){'):
        _once(src, anchor)
    if not src.index(_KERNEL_HEAD) < src.index(_ROPE_LINE) < src.index(_FLAG_ZERO):
        raise RuntimeError("unexpected decode.cu layout")
    out = src.replace(_ROPE_LINE, _ROPE_LINE + _WAIT)
    if early:
        out = out.replace(_KERNEL_HEAD, _KERNEL_HEAD + _LAUNCH_DEPS)
        out = "#define MEGABENCH_PDL_EARLY 1\n" + out
    return out + _LAUNCH_PDL


# ---------------------------------------------------------------------------
# Candidate copy / import


def cache_root() -> Path:
    return Path(os.environ.get("MEGABENCH_SOTA_CACHE", DEFAULT_CACHE))


def _sha(*blobs: bytes) -> str:
    h = hashlib.sha256()
    for b in blobs:
        h.update(b)
    return h.hexdigest()


def _ensure_nvcc_on_path() -> None:
    if shutil.which("nvcc") is None and Path("/usr/local/cuda/bin/nvcc").exists():
        os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


def copy_candidate(submission: Path, decode_src: str | None = None) -> tuple[Path, str]:
    """Copy submission.py and decode.cu (optionally replaced by ``decode_src``)."""
    submission = Path(submission)
    cu_bytes = (decode_src.encode() if decode_src is not None
                else (submission.parent / "candidate_helpers" / "decode.cu").read_bytes())
    sub_bytes = submission.read_bytes()
    dest = cache_root() / "candidates" / _sha(sub_bytes, cu_bytes)[:16]
    (dest / "candidate_helpers").mkdir(parents=True, exist_ok=True)
    for path, data in ((dest / "submission.py", sub_bytes),
                       (dest / "candidate_helpers" / "decode.cu", cu_bytes)):
        if not path.exists() or path.read_bytes() != data:
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, path)
    return dest, _sha(cu_bytes)


def _import_copy(dest: Path):
    _ensure_nvcc_on_path()
    name = f"megabench_candidate_{dest.name}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, dest / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _nvcc_flags(dest: Path) -> list[str]:
    m = re.search(r"flags\s*=\s*(\[[^\]]*\])", (dest / "submission.py").read_text())
    return ast.literal_eval(m.group(1)) if m else []


def _rc_text(rc: int) -> str:
    try:
        libc = ctypes.CDLL("libcudart.so")
        libc.cudaGetErrorName.restype = ctypes.c_char_p
        return f"{rc} ({libc.cudaGetErrorName(rc).decode()})"
    except Exception:
        pass
    try:
        import torch
        return f"{rc} ({torch._C._cuda_getErrorName(rc)})"  # type: ignore[attr-defined]
    except Exception:
        return str(rc)


def _decode_resources(res: dict[str, int], rc: int) -> str:
    why = []
    if not res["cooperative"]:
        why.append("cooperative launch unsupported")
    if res["max_blocks_per_sm"] < 2:
        why.append(f"fits only {res['max_blocks_per_sm']} CTA/SM (needs >=2)")
    if res["sms"] != 148:
        why.append(f"needs exactly 148 SMs (this GPU has {res['sms']})")
    return f"megakernel resources() rc={_rc_text(rc)}: " + ("; ".join(why) or str(res))


class _Library:
    """A built decode.so (via the copied module) with resources checked."""

    def __init__(self, dest: Path):
        self.dest = dest
        self.cu_sha = ""
        self.module = _import_copy(dest)
        self.cdll = self.module._library()
        arr = (ctypes.c_int * 7)()
        self.cdll.resources.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.cdll.resources.restype = ctypes.c_int
        rc = self.cdll.resources(arr)
        self.resources = dict(zip(RESOURCE_NAMES, (int(v) for v in arr)))
        if rc != 0:
            raise RuntimeError(_decode_resources(self.resources, rc))
        if hasattr(self.cdll, "launch_pdl"):
            self.cdll.launch_pdl.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
            self.cdll.launch_pdl.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Step


class MegakernelStep:
    precision = "fp32"

    def __init__(self, name: str, variant: str, lib: _Library, inputs: dict,
                 config: dict[str, Any], runner=None, scratch=None):
        import torch
        self.torch = torch
        self.arm = name
        self.variant = variant
        self.config = config
        self._lib = lib  # keeps the CDLL alive for the Step's lifetime
        self._inputs = inputs  # their device pointers are baked into the arrays
        self._chained = False
        self._graphs: dict[bool, Any] = {}
        self._runner = runner
        self._outs: dict | None = None
        dev = inputs["token"].device
        self._token_fixture = inputs["token"].detach().clone()
        if variant == "submission":
            self._outs = self._runner(inputs)
            return
        self._pdl = variant.startswith("direct-pdl")
        n_scratch = (lib.resources["scratch_bytes"] + 3) // 4
        self.token_buf = self._token_fixture.clone()
        self.next_buf = torch.zeros((), device=dev, dtype=torch.int64)
        self.chain_buf = self._token_fixture.clone()
        self.logits = torch.zeros(151936, device=dev, dtype=torch.float32)
        self.kw = torch.zeros(28, 8, 128, device=dev, dtype=torch.bfloat16)
        self.vw = torch.zeros(28, 8, 128, device=dev, dtype=torch.bfloat16)
        self.scratch = (scratch if scratch is not None
                        else torch.zeros(n_scratch, device=dev, dtype=torch.float32))
        config["scratch_ptr"] = hex(self.scratch.data_ptr())
        config["scratch_shared"] = scratch is not None
        self._arrays = {False: self._pointers(self.token_buf, self.next_buf),
                        True: self._pointers(self.chain_buf, self.chain_buf)}
        self._ptrs = self._arrays[False]
        self._fn = lib.cdll.launch_pdl if self._pdl else lib.cdll.launch
        if variant == "graph":
            self._capture()

    def _pointers(self, token, nxt):
        ptrs = (ctypes.c_void_p * 22)()
        for i, n in enumerate(INPUT_ORDER):
            ptrs[i] = (token if n == "token" else self._inputs[n]).data_ptr()
        ptrs[16] = self.logits.data_ptr()
        ptrs[17] = nxt.data_ptr()
        ptrs[18] = self.kw.data_ptr()
        ptrs[19] = self.vw.data_ptr()
        ptrs[20] = None
        ptrs[21] = self.scratch.data_ptr()
        return ptrs

    def _error(self, rc: int) -> RuntimeError:
        what = "launch_pdl" if self._pdl else "launch"
        msg = f"megakernel {what} failed: CUDA {_rc_text(rc)}"
        if self._pdl:
            msg = "cooperative+PDL launch rejected: " + msg
        return RuntimeError(msg)

    def _capture(self) -> None:
        torch = self.torch
        try:
            rc = self._fn(self._arrays[False], torch.cuda.current_stream().cuda_stream)
            if rc:
                raise self._error(rc)
            torch.cuda.synchronize()
            for chained in (False, True):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    rc = self._fn(self._arrays[chained],
                                  torch.cuda.current_stream().cuda_stream)
                    if rc:
                        raise self._error(rc)
                self._graphs[chained] = g
        except Exception as e:  # capture failures surface as several exception types
            self._graphs.clear()
            raise RuntimeError(f"cooperative graph capture unavailable: {e}") from e

    # -- Step protocol
    def launch(self) -> None:
        if self.variant == "graph":
            self._graphs[self._chained].replay()
        elif self.variant == "submission":
            # Documented exception to the no-allocation rule: the candidate's own
            # runner allocates its outputs per call (original 0.995 ms method).
            self._outs = self._runner(self._inputs)
        else:
            rc = self._fn(self._ptrs, self.torch.cuda.current_stream().cuda_stream)
            if rc:
                raise self._error(rc)

    def outputs(self) -> dict[str, Any]:
        if self.variant == "submission":
            return self._outs
        return {"logits": self.logits,
                "next_token": self.chain_buf if self._chained else self.next_buf,
                "k_write": self.kw, "v_write": self.vw}

    def set_chained(self, on: bool) -> None:
        if self.variant == "submission":
            return  # unsupported: config["chained_supported"] is False
        on = bool(on)
        if on:
            self.chain_buf.copy_(self._token_fixture)
        self._chained = on
        self._ptrs = self._arrays[on]

    def rebind_scratch(self, scratch) -> None:
        """Point the kernel at ``scratch`` (fp32, >= resources scratch_bytes); direct variants only."""
        if scratch.numel() * 4 < self.lib_scratch_bytes or scratch.dtype != self.torch.float32:
            raise ValueError("scratch too small or not float32")
        self.scratch = scratch
        self.config["scratch_ptr"] = hex(scratch.data_ptr())
        self._arrays = {False: self._pointers(self.token_buf, self.next_buf),
                        True: self._pointers(self.chain_buf, self.chain_buf)}
        self._ptrs = self._arrays[self._chained]

    @property
    def lib_scratch_bytes(self) -> int:
        return self._lib.resources["scratch_bytes"]

    def close(self) -> None:
        self._graphs.clear()
        for n in ("token_buf", "next_buf", "chain_buf", "logits", "kw", "vw", "scratch"):
            if hasattr(self, n):
                delattr(self, n)
        self._arrays = {}
        self._outs = None
        self._runner = None
        self._inputs = {}


# ---------------------------------------------------------------------------
# Arm


class MegakernelArm:
    precision = "fp32"

    def __init__(self, name: str = "megakernel", submission: Path | str = DEFAULT_SUBMISSION,
                 case_id: str = BASE_CASE_ID, **_: Any):
        self.name = name
        self.submission = Path(submission)
        self.case_id = case_id
        self._libs: dict[str, _Library] = {}
        self._submission_runner = None
        self._scratch: dict[str, Any] = {}
        self._case_params: dict | None = None

    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def _params(self) -> dict:
        if self._case_params is None:
            from ...cases import select_cases
            self._case_params = dict(select_cases("all", [self.case_id])[0].params)
        return self._case_params

    def supports(self, shape: Shape, variant: str) -> str | None:
        if variant not in VARIANTS:
            return f"unknown variant {variant!r}"
        want, have = self._params(), megabench_case(shape).params
        keys = ("batch", "context", "layers", "hidden", "q_heads", "kv_heads",
                "head_dim", "intermediate", "vocab")
        if any(want.get(k) != have.get(k) for k in keys):
            return "compiled for b1-s128 only"
        return None

    def _library(self, variant: str) -> tuple[_Library, dict]:
        kind = {"direct-pdl": "pdl", "direct-pdl-early": "pdl-early"}.get(variant, "base")
        orig = (self.submission.parent / "candidate_helpers" / "decode.cu").read_bytes()
        if kind not in self._libs:
            if kind == "base":
                dest, cu_sha = copy_candidate(self.submission)
            else:
                dest, cu_sha = copy_candidate(
                    self.submission, patch_pdl(orig.decode(), early=(kind == "pdl-early")))
            lib = _Library(dest)
            lib.cu_sha = cu_sha
            self._libs[kind] = lib
        lib = self._libs[kind]
        cfg = {"source_sha256": {"submission.py": _sha(self.submission.read_bytes()),
                                 "decode.cu": lib.cu_sha,
                                 "decode.cu_original": _sha(orig)},
               "candidate_copy": str(lib.dest),
               "nvcc_flags": _nvcc_flags(lib.dest),
               "resources": dict(lib.resources),
               "source": str(self.submission)}
        return lib, cfg

    def _shared_scratch(self, lib: _Library, device: str):
        """One scratch per arm and device, so every variant has the same placement."""
        import torch
        n = (lib.resources["scratch_bytes"] + 3) // 4
        cur = self._scratch.get(device)
        if cur is None or cur.numel() < n:
            self._scratch[device] = cur = torch.zeros(n, device=device, dtype=torch.float32)
        return cur

    def _submission(self):
        if self._submission_runner is None:
            lib, _ = self._library("submission")
            self._submission_runner = lib.module.build({})
        return self._submission_runner

    def prepare(self, shape: Shape, inputs: dict, variant: str, device: str):
        import torch
        why = self.supports(shape, variant)
        if why:
            raise ValueError(why)
        lib, cfg = self._library(variant)
        cfg.update(variant=variant, launch_mechanism={
            "direct": "cudaLaunchCooperativeKernel via ctypes",
            "graph": "cudaLaunchCooperativeKernel captured in a CUDA graph",
            "direct-pdl": "cudaLaunchKernelEx cooperative + programmatic stream serialization",
            "direct-pdl-early": "same, plus griddepcontrol.launch_dependents at kernel entry",
            "submission": "candidate torch op (allocates outputs per call)"}[variant],
            pdl=variant.startswith("direct-pdl"), graph=variant == "graph",
            chained_supported=variant != "submission")
        runner = self._submission() if variant == "submission" else None
        scratch = None if variant == "submission" else self._shared_scratch(lib, device)
        step = MegakernelStep(self.name, variant, lib, inputs, cfg, runner, scratch)
        if variant != "submission":
            ref = self._submission()(inputs)
            torch.cuda.synchronize()
            step.set_chained(False)
            step.launch()
            torch.cuda.synchronize()
            got = step.outputs()
            same = all(torch.equal(got[k], ref[k]) for k in OUTPUT_KEYS)
            cfg["bitexact_vs_submission"] = bool(same)
            if not same:
                cfg["bitexact_max_abs_diff"] = {
                    k: float((got[k].double() - ref[k].double()).abs().max())
                    for k in OUTPUT_KEYS}
        return step


# ---------------------------------------------------------------------------
# GPU self-test


def _selftest() -> int:
    import torch

    from ..shapes import make_shape_inputs, reference_outputs
    shape = Shape(1, 128)
    inputs = make_shape_inputs(shape, 7301, "cuda")
    ref = reference_outputs(shape, inputs)
    arm = MegakernelArm()
    status = 0
    for variant in arm.variants():
        print(f"=== {variant}")
        try:
            step = arm.prepare(shape, inputs, variant, "cuda")
        except Exception as e:
            print(f"  PREPARE FAILED: {type(e).__name__}: {e}")
            status = 1
            continue
        c = step.config
        print(f"  resources={c['resources']}")
        print(f"  scratch_ptr={c.get('scratch_ptr')} shared={c.get('scratch_shared')}")
        print(f"  bitexact_vs_submission={c.get('bitexact_vs_submission')} "
              f"{c.get('bitexact_max_abs_diff', '')}")
        step.set_chained(False)
        step.launch()
        torch.cuda.synchronize()
        out = step.outputs()
        lg, rl = out["logits"].float(), ref["logits"].float()
        rel = float((lg - rl).norm() / rl.norm())
        print(f"  logits rel-L2 vs reference={rel:.3e}; next_token={int(out['next_token'])} "
              f"ref={int(ref['next_token'])}")
        for chained in (True, False):
            if chained and variant == "submission":
                continue
            step.set_chained(chained)
            for _ in range(10):
                step.launch()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(200):
                step.launch()
            e.record()
            torch.cuda.synchronize()
            steady = s.elapsed_time(e) / 200
            iso = []
            for _ in range(50):
                torch.cuda.synchronize()
                s.record()
                step.launch()
                e.record()
                torch.cuda.synchronize()
                iso.append(s.elapsed_time(e))
            iso.sort()
            print(f"  {'chained' if chained else 'unchained'}: steady={steady * 1e3:.1f} us "
                  f"isolated median={iso[25] * 1e3:.1f} us")
        step.set_chained(False)
        step.close()
    return status


def _corr(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def _placement_probe(count: int, json_out: str | None) -> int:
    """Chained steady time of one ``direct`` step as its scratch moves through a big pool."""
    import json
    import statistics

    import torch

    from ..shapes import make_shape_inputs
    MiB = 1 << 20
    shape = Shape(1, 128)
    inputs = make_shape_inputs(shape, 7301, "cuda")
    step = MegakernelArm().prepare(shape, inputs, "direct", "cuda")
    n_f32 = (step.lib_scratch_bytes + 3) // 4
    pool = torch.zeros(count * 4 * MiB + 2 * MiB, device="cuda", dtype=torch.uint8)
    base = (-pool.data_ptr()) % (2 * MiB)  # first 2 MiB-aligned byte

    def bind(i: int, off: int = 0) -> int:
        start = base + i * 4 * MiB + off
        step.rebind_scratch(pool[start:start + n_f32 * 4].view(torch.float32))
        return start

    def measure() -> float:
        step.set_chained(True)
        for _ in range(20):
            step.launch()
        torch.cuda.synchronize()
        reps = []
        for _ in range(5):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(50):
                step.launch()
            e.record()
            torch.cuda.synchronize()
            reps.append(s.elapsed_time(e) / 50 * 1e3)
        return statistics.median(reps)

    def outs() -> dict:
        step.set_chained(False)
        step.launch()
        torch.cuda.synchronize()
        return {k: v.clone() for k, v in step.outputs().items()}

    ref = outs()
    bind(0)
    pass1, pass2, control = {}, {}, []
    every = max(1, count // 10)
    for i in range(count):
        bind(i)
        pass1[i] = measure()
        if i % every == 0 and len(control) < 10:
            bind(0)
            control.append(measure())
    for i in list(range(1, count, 2)) + list(range(0, count, 2)):
        bind(i)
        pass2[i] = measure()
    offsets = {}
    for off in (0, 4 << 10, 64 << 10, 512 << 10, 1 * MiB):
        bind(1 % count, off)
        offsets[off] = measure()
    exact = {}
    for i in sorted({0, count // 2, count - 1}):
        bind(i)
        o = outs()
        exact[i] = all(torch.equal(o[k], ref[k]) for k in ref)
    print(f"placement probe: {count} x 4 MiB regions, us per chained launch (median of 5x50)")
    print(f"{'i':>3} {'addr_mod_2GiB':>14} {'pass1':>9} {'pass2':>9}")
    for i in range(count):
        print(f"{i:>3} {(base + i * 4 * MiB + pool.data_ptr()) % (2 << 30):>#14x} "
              f"{pass1[i]:>9.1f} {pass2[i]:>9.1f}")
    a, b = [pass1[i] for i in range(count)], [pass2[i] for i in range(count)]
    mean = [(x + y) / 2 for x, y in zip(a, b)]
    print(f"placement mean-of-passes: min {min(mean):.1f} median {statistics.median(mean):.1f} "
          f"max {max(mean):.1f} spread {max(mean) - min(mean):.1f} us")
    print(f"pass1 vs pass2 correlation: {_corr(a, b):.3f}; mean |pass1-pass2| "
          f"{sum(abs(x - y) for x, y in zip(a, b)) / count:.1f} us")
    print(f"control (placement 0 x{len(control)}): min {min(control):.1f} median "
          f"{statistics.median(control):.1f} max {max(control):.1f} spread "
          f"{max(control) - min(control):.1f} us")
    print("sub-page offsets (placement 1): " +
          ", ".join(f"+{o >> 10}KiB={t:.1f}" for o, t in offsets.items()))
    print(f"bit-exact at placements {list(exact)}: {all(exact.values())}")
    if json_out:
        Path(json_out).write_text(json.dumps(
            dict(count=count, pass1=a, pass2=b, control=control, offsets_us=offsets,
                 bitexact=exact), indent=1))
    step.close()
    return 0 if all(exact.values()) else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="megakernel arm GPU self-test and probes")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--placement-probe", action="store_true")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--json-out")
    ns = ap.parse_args()
    if ns.placement_probe:
        raise SystemExit(_placement_probe(ns.count, ns.json_out))
    if ns.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
    raise SystemExit(0)
