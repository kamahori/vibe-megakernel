#!/usr/bin/env python3
"""AMD (ROCm/HIP) counterpart of codegen_gate.py: keep profiling builds faithful.

Two checks, built on the ROCm LLVM tools (llvm-objdump, llvm-readelf, llvm-objcopy, clang-offload-bundler), found on
PATH, in $ROCM_PATH/llvm/bin, or in /opt/rocm/llvm/bin:

  identical   A build with the profiling switch OFF must equal the production build.
              Disassembles each kernel (GCN/CDNA ISA), strips addresses and encodings,
              and requires equal instruction streams.
  resources   A build with the profiling switch ON must not change resource use.
              Read from the code object's AMDGPU metadata (llvm-readelf --notes):
              VGPR, AGPR, SGPR, LDS (group segment), scratch (private segment),
              VGPR/SGPR spills and dynamic stack. Scratch, spill, stack and LDS growth
              fail. VGPR/AGPR/SGPR growth only warns, and fails if it lowers waves per
              SIMD or workgroups per CU (--threads-per-cta), exceeds --max-vgprs, or
              --strict-regs is given. --allow overrides.

Inputs may be a code object (.hsaco/.co, ELF machine AMDGPU), or any host binary that
embeds one (.so from hipcc or torch.utils.cpp_extension, .o, an executable, a fatbin);
embedded code objects are extracted with clang-offload-bundler. Use --arch to pick one target
(default: the first, e.g. gfx942 on MI300X).

Exit codes: 0 = pass, 1 = mismatch, 2 = usage or tool error.

  codegen_gate_amd.py identical --production prod.so --candidate off.so
  codegen_gate_amd.py resources --production prod.hsaco --candidate on.hsaco \
      --threads-per-cta 256 --allow lds
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


class ToolError(RuntimeError):
    pass


def find_tool(name: str) -> str:
    dirs = [os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), d) for d in ("llvm/bin", "bin")]
    dirs += ["/opt/rocm/llvm/bin", "/opt/rocm/bin"]
    path = shutil.which(name) or next(
        (p for d in dirs if (p := shutil.which(name, path=d))), None)
    if not path:
        raise ToolError(f"{name} not found on PATH or under $ROCM_PATH")
    return path


def run_tool(name: str, *args: str) -> str:
    proc = subprocess.run([find_tool(name), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise ToolError(f"{name} {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


# ------------------------------------------------------------ code objects

def _is_amdgpu_elf(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(20)
    return head[:4] == b"\x7fELF" and int.from_bytes(head[18:20], "little") == 224


def _unbundle(bundle: str, out: str, arch: str | None) -> list[str]:
    """Split a clang offload bundle (plain or CCOB-compressed) into AMDGPU code objects."""
    listing = run_tool("clang-offload-bundler", "--list", "--type=o", f"--input={bundle}")
    targets = [t.strip() for t in listing.splitlines() if "amdgcn" in t]
    if arch:
        targets = [t for t in targets if re.search(arch, t)]
    found = []
    for i, t in enumerate(targets):
        dst = os.path.join(out, f"co{i}-{re.sub(r'[^A-Za-z0-9_.-]', '_', t)}.co")
        run_tool("clang-offload-bundler", "--unbundle", "--type=o", f"--input={bundle}",
                 f"--targets={t}", f"--output={dst}")
        if os.path.getsize(dst) and _is_amdgpu_elf(dst):
            found.append(dst)
    return found


def code_objects(path: str, workdir: str, arch: str | None) -> list[str]:
    """Return AMDGPU code objects for `path`, extracting embedded ones if needed."""
    if not os.path.isfile(path):
        raise ToolError(f"no such file: {path}")
    if _is_amdgpu_elf(path):
        return [path]
    out = tempfile.mkdtemp(dir=workdir)
    with open(path, "rb") as f:
        magic = f.read(24)
    if magic.startswith(b"__CLANG_OFFLOAD_BUNDLE__") or magic.startswith(b"CCOB"):
        found = _unbundle(path, out, arch)
    else:
        # Host ELF (.so/.o/executable): the device bundle lives in .hip_fatbin.
        fatbin = os.path.join(out, "hip_fatbin.bin")
        try:
            run_tool("llvm-objcopy", "--dump-section", f".hip_fatbin={fatbin}", path,
                     os.path.join(out, "host.discard"))
        except ToolError as e:
            raise ToolError(f"{path}: no .hip_fatbin section ({e})")
        found = _unbundle(fatbin, out, arch)
    if not found:
        raise ToolError(f"no AMDGPU code object found in {path}"
                        + (f" for target {arch}" if arch else ""))
    if arch is None:
        names = sorted({re.sub(r"^co\d+-", "", os.path.basename(p)) for p in found})
        if len(names) > 1:
            print(f"note: {path} carries {names}; using the first. Pass --arch to choose.",
                  file=sys.stderr)
            found = [p for p in found if p.endswith(names[0])]
    return found


# ------------------------------------------------------------------- ISA

_KERNEL_HDR = re.compile(r"^[0-9a-fA-F]*\s*<([^>]+)>:\s*$")
_COMMENT = re.compile(r"\s*//.*$")
_ADDR = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")


def disassemble(co: str) -> dict[str, list[str]]:
    text = run_tool("llvm-objdump", "-d", "--no-show-raw-insn", "--no-leading-addr",
                    "--symbolize-operands", co)
    kernels: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = _KERNEL_HDR.match(line.strip())
        if m and re.fullmatch(r"L\d+", m.group(1)) and current is not None:
            kernels[current].append(f"<{m.group(1)}>:")  # local branch label
            continue
        if m:
            current = m.group(1)
            kernels.setdefault(current, [])
            continue
        if current is None:
            continue
        ins = _ADDR.sub("ADDR", _COMMENT.sub("", line)).strip()
        if ins and not ins.startswith("Disassembly of section"):
            kernels[current].append(re.sub(r"\s+", " ", ins))
    # Local labels (<L0>:) are symbolized; drop trailing s_code_end padding.
    for name, body in kernels.items():
        while body and body[-1] in ("s_code_end", "s_nop 0"):
            body.pop()
    return {k: v for k, v in kernels.items() if v}


# -------------------------------------------------------------- metadata

_KV = re.compile(r"^(\s*)(?:-\s+)?\.([A-Za-z0-9_]+):\s*(.*)$")


def kernel_metadata(co: str) -> dict[str, dict]:
    text = run_tool("llvm-readelf", "--notes", co)
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if "amdhsa.kernels:" in l)
    except StopIteration:
        raise ToolError(f"no amdhsa.kernels metadata in {co}")
    kernels, cur, indent = [], None, None
    for line in lines[start + 1:]:
        if not line.strip():
            continue
        m_item = re.match(r"^(\s*)-\s+\.", line)
        lead = len(line) - len(line.lstrip())
        if indent is None and m_item:
            indent = len(m_item.group(1))
        if indent is None:
            continue
        if lead < indent and not line.lstrip().startswith("-"):
            break  # left the kernels list (e.g. amdhsa.printf / amdhsa.target)
        if m_item and len(m_item.group(1)) == indent:
            cur = {}
            kernels.append(cur)
        m = _KV.match(line)
        if cur is None or not m:
            continue
        depth = len(m.group(1)) + (2 if line.lstrip().startswith("-") else 0)
        if depth == indent + 2:
            key, val = m.group(2), m.group(3).strip().strip("'\"")
            cur[key] = int(val) if re.fullmatch(r"-?\d+", val) else val
    out = {}
    for k in kernels:
        name = k.get("name") or k.get("symbol", "?")
        out[name] = {
            "vgpr": int(k.get("vgpr_count", 0)),
            "agpr": int(k.get("agpr_count", 0)),
            "sgpr": int(k.get("sgpr_count", 0)),
            "lds": int(k.get("group_segment_fixed_size", 0)),
            "scratch": int(k.get("private_segment_fixed_size", 0)),
            "vgpr_spill": int(k.get("vgpr_spill_count", 0)),
            "sgpr_spill": int(k.get("sgpr_spill_count", 0)),
            "dyn_stack": 1 if str(k.get("uses_dynamic_stack", "false")).lower() == "true" else 0,
            "max_wg": int(k.get("max_flat_workgroup_size", 0)),
            "wavefront": int(k.get("wavefront_size", 64)),
        }
    return out


# CDNA2/CDNA3 (gfx90a, gfx942, gfx950): 512 unified VGPRs per lane per SIMD
# (arch VGPRs then AGPRs, arch part aligned to 4), granule 8, max 8 waves/SIMD,
# 4 SIMDs and 64 KiB LDS per CU (160 KiB on gfx950).
def unified_vgprs(r: dict) -> int:
    # On gfx90a+ the metadata .vgpr_count already totals ArchVGPRs (aligned to 4) and AGPRs.
    return max(r["vgpr"], r["agpr"])


def waves_per_simd(r: dict) -> int:
    v = max(1, unified_vgprs(r))
    v = ((v + 7) // 8) * 8
    return max(0, min(8, 512 // v))


def wgs_per_cu(r: dict, threads: int, dyn_lds: int, lds_per_cu: int) -> int:
    waves = (threads + r["wavefront"] - 1) // r["wavefront"]
    by_waves = (4 * waves_per_simd(r)) // max(1, waves)
    lds = r["lds"] + dyn_lds
    by_lds = lds_per_cu // lds if lds else 1 << 30
    return min(by_waves, by_lds)


def _select(d: dict, rx: str | None) -> dict:
    if not rx:
        return d
    pat = re.compile(rx)
    return {k: v for k, v in d.items() if pat.search(k)}


def _match(prod: dict, cand: dict) -> tuple[list[str], list[str], list[str]]:
    both = sorted(set(prod) & set(cand))
    return both, sorted(set(prod) - set(cand)), sorted(set(cand) - set(prod))


def cmd_identical(a) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        prod, cand = {}, {}
        for co in code_objects(a.production, tmp, a.arch):
            prod.update(disassemble(co))
        for co in code_objects(a.candidate, tmp, a.arch):
            cand.update(disassemble(co))
    prod, cand = _select(prod, a.kernel), _select(cand, a.kernel)
    if not prod:
        raise ToolError("no kernels selected in the production build")
    both, only_p, only_c = _match(prod, cand)
    bad = 0
    for k in only_p + only_c:
        print(f"FAIL kernel only in {'production' if k in only_p else 'candidate'}: {k}")
        bad += 1
    for k in both:
        if prod[k] == cand[k]:
            print(f"PASS {k}: {len(prod[k])} instructions identical")
            continue
        bad += 1
        print(f"FAIL {k}: ISA differs ({len(prod[k])} vs {len(cand[k])} instructions)")
        diff = list(difflib.unified_diff(prod[k], cand[k], "production", "candidate", lineterm="", n=2))
        print("\n".join(diff[:a.diff_lines]))
        if len(diff) > a.diff_lines:
            print(f"... {len(diff) - a.diff_lines} more diff lines")
    return 1 if bad else 0


def cmd_resources(a) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        prod, cand = {}, {}
        for co in code_objects(a.production, tmp, a.arch):
            prod.update(kernel_metadata(co))
        for co in code_objects(a.candidate, tmp, a.arch):
            cand.update(kernel_metadata(co))
    prod, cand = _select(prod, a.kernel), _select(cand, a.kernel)
    if not prod:
        raise ToolError("no kernels selected in the production build")
    allow = {x.strip() for x in (a.allow or "").split(",") if x.strip()}
    both, only_p, only_c = _match(prod, cand)
    report, bad = [], 0
    for k in only_p + only_c:
        print(f"FAIL kernel only in {'production' if k in only_p else 'candidate'}: {k}")
        bad += 1
    hard = {"scratch": "scratch", "vgpr_spill": "spill", "sgpr_spill": "spill",
            "dyn_stack": "stack", "lds": "lds"}
    for k in both:
        p, c = prod[k], cand[k]
        row = {"kernel": k, "production": p, "candidate": c, "status": "PASS", "notes": []}
        print(f"{k}\n  {'':10}{'VGPR':>6}{'AGPR':>6}{'SGPR':>6}{'LDS':>8}{'SCRATCH':>9}"
              f"{'VSPILL':>8}{'SSPILL':>8}{'WAVES/SIMD':>12}")
        for tag, r in (("production", p), ("candidate", c)):
            print(f"  {tag:10}{r['vgpr']:>6}{r['agpr']:>6}{r['sgpr']:>6}{r['lds']:>8}"
                  f"{r['scratch']:>9}{r['vgpr_spill']:>8}{r['sgpr_spill']:>8}{waves_per_simd(r):>12}")
        fail = []
        for field, group in hard.items():
            if c[field] > p[field] and group not in allow:
                fail.append(f"{field} grew {p[field]} -> {c[field]}")
        reg_grew = [f for f in ("vgpr", "agpr", "sgpr") if c[f] > p[f]]
        if reg_grew and "reg" not in allow:
            msg = ", ".join(f"{f} {p[f]} -> {c[f]}" for f in reg_grew)
            if a.strict_regs:
                fail.append(f"register growth ({msg}) with --strict-regs")
            elif a.max_vgprs and unified_vgprs(c) > a.max_vgprs:
                fail.append(f"unified VGPRs {unified_vgprs(c)} exceed --max-vgprs {a.max_vgprs}")
            elif waves_per_simd(c) < waves_per_simd(p):
                fail.append(f"waves/SIMD dropped {waves_per_simd(p)} -> {waves_per_simd(c)} ({msg})")
            elif a.threads_per_cta and (wgs_per_cu(c, a.threads_per_cta, a.smem_per_cta, a.lds_per_cu)
                                        < wgs_per_cu(p, a.threads_per_cta, a.smem_per_cta, a.lds_per_cu)):
                fail.append(f"workgroups/CU dropped ({msg})")
            else:
                row["notes"].append(f"WARN register growth without occupancy loss: {msg}")
                print(f"  WARN register growth without occupancy loss: {msg}")
        if fail:
            bad += 1
            row["status"] = "FAIL"
            row["notes"] += fail
            for f in fail:
                print(f"  FAIL {f}")
        else:
            print("  PASS")
        report.append(row)
    if a.json:
        print(json.dumps(report, indent=2))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("identical", "resources"):
        s = sub.add_parser(name)
        s.add_argument("--production", required=True)
        s.add_argument("--candidate", required=True)
        s.add_argument("--kernel", help="regex on kernel (mangled) names")
        s.add_argument("--arch", help="target id regex, e.g. gfx942")
        if name == "identical":
            s.add_argument("--diff-lines", type=int, default=60)
        else:
            s.add_argument("--allow", help="comma list of reg,stack,lds,spill,scratch")
            s.add_argument("--threads-per-cta", type=int, default=0)
            s.add_argument("--smem-per-cta", type=int, default=0, help="dynamic LDS bytes")
            s.add_argument("--lds-per-cu", type=int, default=65536)
            s.add_argument("--max-vgprs", type=int, default=0)
            s.add_argument("--strict-regs", action="store_true")
            s.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        return cmd_identical(a) if a.cmd == "identical" else cmd_resources(a)
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
