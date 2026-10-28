#!/usr/bin/env python3
"""Compare two compiled CUDA artifacts so profiling stays faithful to production.

Two checks, both built on `cuobjdump` (CUDA toolkit, found on PATH or $CUDA_HOME/bin):

  identical   A build with the profiling switch OFF must equal the production build.
              Dumps SASS per kernel, strips addresses and encoding comments, and
              requires equal instruction streams.
  resources   A build with the profiling switch ON must not change resource use.
              Spill (local), stack and shared growth fail. Register growth only warns,
              and fails if it lowers CTAs per SM (--threads-per-cta), exceeds
              --max-regs, or --strict-regs is given. --allow overrides.

Inputs may be .cubin, .o, .so, or an executable that carries a fatbin.
Exit codes: 0 = pass, 1 = mismatch, 2 = usage or tool error.

  codegen_gate.py identical --production prod.cubin --candidate off.cubin
  codegen_gate.py resources --production prod.cubin --candidate on.cubin --allow smem
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


class ToolError(RuntimeError):
    pass


def find_tool(name: str) -> str:
    cuda_bin = os.path.join(os.environ.get("CUDA_HOME", ""), "bin")
    path = shutil.which(name) or shutil.which(name, path=cuda_bin)
    if not path:
        raise ToolError(f"{name} not found on PATH or in $CUDA_HOME/bin")
    return path


def run_tool(name: str, *args: str) -> str:
    proc = subprocess.run([find_tool(name), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise ToolError(f"{name} {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


# ---------------------------------------------------------------- SASS

_ENC_COMMENT = re.compile(r"/\*.*?\*/")
_FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")
_HEX_TARGET = re.compile(r"\b0x[0-9a-fA-F]+\b(?=\s*[;,)]|$)")
_BRANCH_OP = re.compile(r"\b(BRA|BRX|CALL|JMP|JMX|BSSY|BSYNC|SSY|RET|EXIT)\b")


def normalize_sass(text: str) -> list[str]:
    """Return instruction text without offsets, encodings, blanks, or file/time headers."""
    out = []
    for line in text.splitlines():
        line = _ENC_COMMENT.sub("", line).strip()
        if not line or line.startswith(("code for", ".target", "Fatbin", "arch =", "code version")):
            continue
        if line.startswith((".headerflags", "producer", "host =", "compile_size", "identifier")):
            continue
        if _BRANCH_OP.search(line):  # literal branch targets are code addresses
            line = _HEX_TARGET.sub("<addr>", line)
        out.append(re.sub(r"\s+", " ", line))
    return out


def split_by_function(sass: str) -> dict[str, str]:
    """Map kernel name -> its SASS text, from `cuobjdump -sass` output."""
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in sass.splitlines():
        m = _FUNC.match(line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, [])
        elif cur is not None:
            funcs[cur].append(line)
    return {k: "\n".join(v) for k, v in funcs.items()}


def sass_by_kernel(path: str, kernel: str | None) -> dict[str, list[str]]:
    funcs = split_by_function(run_tool("cuobjdump", "-sass", path))
    rx = re.compile(kernel) if kernel else None
    return {k: normalize_sass(v) for k, v in funcs.items() if rx is None or rx.search(k)}


# ------------------------------------------------------- resource usage

_RES_FUNC = re.compile(r"^\s*Function\s+(\S+?):\s*$")
_RES_KV = re.compile(r"([A-Z]+)(?:\[\d+\])?:(\d+)")
FIELDS = ("REG", "STACK", "SHARED", "LOCAL")
ALLOW_NAMES = {"reg": ("REG",), "stack": ("STACK",), "smem": ("SHARED",), "shared": ("SHARED",),
               "spill": ("LOCAL", "SPILL"), "local": ("LOCAL",)}


def parse_res_usage(text: str) -> dict[str, dict[str, int]]:
    """Parse `cuobjdump -res-usage` into {kernel: {REG, STACK, SHARED, LOCAL, ...}}."""
    out: dict[str, dict[str, int]] = {}
    cur = None
    for line in text.splitlines():
        m = _RES_FUNC.match(line)
        if m:
            cur = m.group(1)
            out[cur] = {}
        elif cur is not None and "REG:" in line:
            out[cur].update({k: int(v) for k, v in _RES_KV.findall(line)})
    return out


_PTXAS_FUNC = re.compile(r"Compiling entry function '([^']+)'")
_PTXAS_SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
_PTXAS_USED = re.compile(r"Used (\d+) registers")
_PTXAS_STACK = re.compile(r"(\d+) bytes stack frame")


def parse_ptxas_log(text: str) -> dict[str, dict[str, int]]:
    """Parse `ptxas -v` lines. Adds SPILL (stores + loads) to what cuobjdump reports."""
    out: dict[str, dict[str, int]] = {}
    cur = None
    for line in text.splitlines():
        m = _PTXAS_FUNC.search(line)
        if m:
            cur = m.group(1)
            out[cur] = {}
        elif cur is not None:
            if m := _PTXAS_SPILL.search(line):
                out[cur]["SPILL"] = int(m.group(1)) + int(m.group(2))
            if m := _PTXAS_USED.search(line):
                out[cur]["REG"] = int(m.group(1))
            if m := _PTXAS_STACK.search(line):
                out[cur]["STACK"] = int(m.group(1))
    return out


def resources(path: str, kernel: str | None, log: str | None) -> dict[str, dict[str, int]]:
    res = parse_res_usage(run_tool("cuobjdump", "-res-usage", path))
    if log:
        with open(log) as f:
            for name, vals in parse_ptxas_log(f.read()).items():
                for k, v in vals.items():
                    if k == "SPILL" or name in res:
                        res.setdefault(name, {}).setdefault(k, v)
    rx = re.compile(kernel) if kernel else None
    return {k: v for k, v in res.items() if rx is None or rx.search(k)}


# Hardware constants for the occupancy estimate. Defaults suit sm_100; each can be
# overridden by a flag or by passing a different value to ctas_per_sm().
REGS_PER_SM = 64 * 1024          # 32-bit registers per SM
REG_ALLOC_UNIT = 256             # registers are allocated per warp in units of this many
REG_ROUND = 8                    # per-thread register count is rounded up to a multiple of this
MAX_REGS_PER_THREAD = 255
WARP_SIZE = 32
MAX_THREADS_PER_SM = 2048
MAX_CTAS_PER_SM = 32
SMEM_PER_SM = 228 * 1024         # shared memory per SM (--smem-per-sm)
SMEM_RESERVED_PER_CTA = 1024     # driver-reserved shared memory per resident CTA


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def ctas_per_sm(regs: int, threads: int, smem: int, smem_per_sm: int = SMEM_PER_SM) -> int:
    """Resident CTAs per SM limited by registers, shared memory, threads and CTA count."""
    regs = min(_ceil(max(regs, 1), REG_ROUND) * REG_ROUND, MAX_REGS_PER_THREAD)
    warps = _ceil(threads, WARP_SIZE)
    per_warp = _ceil(regs * WARP_SIZE, REG_ALLOC_UNIT) * REG_ALLOC_UNIT
    limits = [REGS_PER_SM // (per_warp * warps), MAX_THREADS_PER_SM // (warps * WARP_SIZE), MAX_CTAS_PER_SM]
    if smem + SMEM_RESERVED_PER_CTA > 0:
        limits.append(smem_per_sm // (smem + SMEM_RESERVED_PER_CTA))
    return min(limits)


def compare_resources(prod, cand, allow: set[str], threads_per_cta: int | None = None,
                      smem_per_cta: int = 0, smem_per_sm: int = SMEM_PER_SM,
                      max_regs: int | None = None, strict_regs: bool = False) -> dict:
    """Fail on spill/stack/local/shared growth; treat register growth as a warning
    unless it lowers occupancy, passes max_regs, or strict_regs is set."""
    allowed = {f for a in allow for f in ALLOW_NAMES.get(a, ())}
    rows, ok = [], True
    for name in sorted(set(prod) | set(cand)):
        if name not in prod or name not in cand:
            rows.append({"kernel": name, "status": "missing in " + ("candidate" if name in prod else "production")})
            ok = False
            continue
        p, c = prod[name], cand[name]
        fails, warns = [], []
        for f in ("STACK", "SHARED", "LOCAL", "SPILL"):
            if p.get(f) is not None and c.get(f) is not None and c[f] > p[f] and f not in allowed:
                fails.append(f"{f} {p[f]}->{c[f]}")
        if c.get("REG", 0) > p.get("REG", 0):
            msg = f"REG {p.get('REG')}->{c.get('REG')}"
            (fails if strict_regs and "REG" not in allowed else warns).append(msg)
        if max_regs is not None and c.get("REG", 0) > max_regs and "REG" not in allowed:
            fails.append(f"REG {c['REG']} exceeds --max-regs {max_regs}")
        row = {"kernel": name, "production": p, "candidate": c}
        if threads_per_cta:
            occ = [ctas_per_sm(r.get("REG", 0), threads_per_cta, r.get("SHARED", 0) + smem_per_cta, smem_per_sm)
                   for r in (p, c)]
            row["ctas_per_sm"] = {"production": occ[0], "candidate": occ[1]}
            if occ[1] < occ[0] and not ({"REG", "SHARED"} & allowed):
                fails.append(f"CTAs/SM {occ[0]}->{occ[1]}")
        row["status"] = ("FAIL " + "; ".join(fails)) if fails else ("WARN " + "; ".join(warns)) if warns else "ok"
        rows.append(row)
        ok = ok and not fails
    return {"ok": ok, "kernels": rows}


# ------------------------------------------------------------- commands

def cmd_identical(args) -> int:
    prod, cand = sass_by_kernel(args.production, args.kernel), sass_by_kernel(args.candidate, args.kernel)
    report, ok = [], True
    for name in sorted(set(prod) | set(cand)):
        if name not in prod or name not in cand:
            report.append({"kernel": name, "status": "missing in " + ("candidate" if name in prod else "production")})
            ok = False
            continue
        same = prod[name] == cand[name]
        diff = [] if same else list(difflib.unified_diff(prod[name], cand[name], "production", "candidate", lineterm="", n=2))
        report.append({"kernel": name, "status": "identical" if same else "DIFFERENT",
                       "instructions": [len(prod[name]), len(cand[name])], "diff": diff[:40]})
        ok = ok and same
    if not report:
        raise ToolError("no kernels matched")
    if args.json:
        print(json.dumps({"ok": ok, "kernels": report}, indent=2))
    else:
        for r in report:
            print(f"{r['kernel']}: {r['status']}" + (f" (instructions {r['instructions']})" if "instructions" in r else ""))
            for d in r.get("diff", []):
                print("  " + d)
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_resources(args) -> int:
    prod = resources(args.production, args.kernel, args.production_log)
    cand = resources(args.candidate, args.kernel, args.candidate_log)
    if not prod and not cand:
        raise ToolError("no kernels matched")
    allow = {a.strip() for a in args.allow.split(",") if a.strip()}
    unknown = allow - set(ALLOW_NAMES)
    if unknown:
        raise ToolError(f"unknown --allow entries: {sorted(unknown)}; use reg,stack,smem,spill")
    result = compare_resources(prod, cand, allow, args.threads_per_cta, args.smem_per_cta,
                               args.smem_per_sm, args.max_regs, args.strict_regs)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{'kernel':40s} " + " ".join(f"{f:>13s}" for f in (*FIELDS, "status")))
        for r in result["kernels"]:
            if "production" not in r:
                print(f"{r['kernel'][:40]:40s} {r['status']}")
                continue
            cells = [f"{r['production'].get(f, '-')}->{r['candidate'].get(f, '-')}" for f in FIELDS]
            print(f"{r['kernel'][:40]:40s} " + " ".join(f"{c:>13s}" for c in cells) + f" {r['status']}")
            if "ctas_per_sm" in r:
                o = r["ctas_per_sm"]
                print(f"{'':40s} CTAs/SM production {o['production']}, candidate {o['candidate']}")
        print("PASS" if result["ok"] else "FAIL")
    return 0 if result["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("identical", "resources"):
        p = sub.add_parser(name)
        p.add_argument("--production", required=True)
        p.add_argument("--candidate", required=True)
        p.add_argument("--kernel", help="regex selecting kernel names")
        p.add_argument("--json", action="store_true")
    r = sub.choices["resources"]
    r.add_argument("--allow", default="", help="comma list of resources allowed to grow: reg,stack,smem,spill")
    r.add_argument("--threads-per-cta", type=int, help="enable the occupancy check for this launch")
    r.add_argument("--smem-per-cta", type=int, default=0, help="dynamic shared bytes per CTA (default 0)")
    r.add_argument("--smem-per-sm", type=int, default=SMEM_PER_SM, help="shared bytes per SM (default sm_100)")
    r.add_argument("--max-regs", type=int, help="register cap of the launch (launch bounds / maxrregcount)")
    r.add_argument("--strict-regs", action="store_true", help="fail on any register growth")
    r.add_argument("--production-log", help="ptxas -v text log of the production build")
    r.add_argument("--candidate-log", help="ptxas -v text log of the candidate build")
    args = ap.parse_args(argv)
    try:
        return cmd_identical(args) if args.mode == "identical" else cmd_resources(args)
    except (ToolError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
