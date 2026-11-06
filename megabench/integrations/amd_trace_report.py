"""Export and analyze an AMD trace of one warmed MegaBench candidate call (ROCm, rocprofv3).

Three sources, each optional and independent, merged into one report:

1. Per-block phase timeline (timeline_profile_case; needs build_timeline in the submission):
   per-phase wall / slowest / mean / begin_skew / imbalance and a Perfetto trace.json.
2. rocprofv3 dispatch trace + hardware counters (several passes; counters are discovered with
   ``rocprofv3 -L`` and only available ones are requested): kernel time, launch geometry,
   VGPR/SGPR/LDS/scratch, HBM bytes and achieved bandwidth, L2 hit rate, instruction mix.
3. rocprofv3 PC sampling (host_trap): where waves are when sampled, grouped by instruction class
   (spin/sleep, memory wait, VMEM, LDS, VALU, SALU, branch, barrier) plus the top instructions.

Writes reports/trace-<ts>/{report.md, report.json, timeline/, rocprof-*/} and prints report.md.
Findings are signals for hypotheses, not proof. Correctness and latency come from check_candidate.
  python -m megabench.integrations.amd_trace_report --case CASE --submission submission.py
         [--bandwidth-tb-s 5.3] [--skip timeline,counters,pc] [--kernel REGEX]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CAPTURE = ["-m", "megabench.integrations.ncu_profile_case"]
# Wanted counters by purpose; only those reported by `rocprofv3 -L` are used, one pass per group.
COUNTER_GROUPS = [
    ("memory", ["FETCH_SIZE", "WRITE_SIZE", "GRBM_GUI_ACTIVE"]),
    ("l2", ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum"]),
    ("waves", ["SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES", "SQ_WAIT_INST_ANY"]),
    ("insts", ["SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_LDS",
               "SQ_INSTS_SMEM"]),
    ("lds", ["SQ_LDS_BANK_CONFLICT", "SQ_LDS_IDX_ACTIVE", "SQ_INSTS_LDS"]),
]
CLASSES = [  # (class, regex on the mnemonic), first match wins
    ("spin/sleep", r"^s_sleep|^s_nop"),
    ("mem-wait", r"^s_waitcnt"),
    ("barrier", r"^s_barrier"),
    ("branch", r"^s_cbranch|^s_branch|^s_setpc|^s_swappc"),
    ("vmem", r"^(global|buffer|flat|scratch)_"),
    ("lds", r"^ds_"),
    ("smem", r"^s_(load|buffer_load|store|dcache)"),
    ("valu-mfma", r"^v_mfma|^v_smfmac"),
    ("valu", r"^v_"),
    ("salu", r"^s_"),
]


def _rocprof() -> str:
    tool = shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3"
    if not os.path.exists(tool):
        raise SystemExit("rocprofv3 not found; load the rocm module")
    return tool


def _capture_cmd(case: str, submission: Path, seed: int) -> list[str]:
    return [sys.executable, *CAPTURE, "--case", case, "--submission", str(submission), "--seed", str(seed)]


# rocprofv3 can hang in its own error handler after a fatal error (seen on gfx942 when a counter set
# does not fit in one pass), holding GPU memory. Watch the log and kill the whole process group.
FATAL = re.compile(r"Could not construct profile cfg|rocprofv3 caught signal|exceeds the capabilities")


def _run(cmd: list[str], log: Path, timeout: int = 600) -> int:
    import signal
    with open(log, "w") as f:
        f.write("+ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        rc = None
        while rc is None:
            time.sleep(2)
            rc = proc.poll()
            if rc is not None:
                break
            text = log.read_text(errors="replace")[-20000:]
            reason = "FATAL" if FATAL.search(text) else ("TIMEOUT" if time.time() - t0 > timeout else None)
            if reason:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                f.write(f"\n{reason}: killed process group\n")
                return 124 if reason == "TIMEOUT" else 125
        return rc


def _csvs(d: Path, name: str) -> list[dict]:
    rows = []
    for p in glob.glob(str(d / "**" / f"*{name}*.csv"), recursive=True):
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def _col(row: dict, *names: str):
    low = {k.lower(): v for k, v in row.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def available_counters(out: Path) -> set[str]:
    log = out / "rocprof-avail.txt"
    _run([_rocprof(), "-L"], log, timeout=300)
    text = log.read_text(errors="replace")
    return set(re.findall(r"\b([A-Z][A-Z0-9_]+(?:_sum)?)\b", text))


def kernel_dispatches(out: Path, case: str, sub: Path, seed: int, kernel: str | None) -> list[dict]:
    d = out / "rocprof-trace"
    cmd = [_rocprof(), "--kernel-trace", "-f", "csv", "-d", str(d), "-o", "run"]
    _run(cmd + ["--"] + _capture_cmd(case, sub, seed), out / "rocprof-trace.log")
    rows = []
    for r in _csvs(d, "kernel_trace"):
        name = _col(r, "Kernel_Name") or ""
        if kernel and not re.search(kernel, name):
            continue
        start, end = _col(r, "Start_Timestamp"), _col(r, "End_Timestamp")
        rows.append({"name": name, "dur_us": (int(end) - int(start)) / 1e3 if start and end else None,
                     "grid": _col(r, "Grid_Size", "Grid_Size_X"), "workgroup": _col(r, "Workgroup_Size", "Workgroup_Size_X"),
                     "lds": _col(r, "LDS_Block_Size", "Lds_Size"), "scratch": _col(r, "Scratch_Size", "Private_Segment_Size"),
                     "vgpr": _col(r, "VGPR_Count", "Arch_VGPR_Count"), "agpr": _col(r, "Accum_VGPR_Count"),
                     "sgpr": _col(r, "SGPR_Count")})
    return rows


def pick_kernel(dispatches: list[dict]) -> str | None:
    """The candidate kernel: the capture's last dispatch (input creation and warmup come first)."""
    return "^" + re.escape(dispatches[-1]["name"].split("(")[0]) if dispatches else None


def counters(out: Path, case: str, sub: Path, seed: int, kernel: str, avail: set[str]) -> dict:
    vals: dict[str, float] = {}
    skipped = []
    for group, wanted in COUNTER_GROUPS:
        use = [c for c in wanted if c in avail and c not in vals]
        skipped += [c for c in wanted if c not in avail]
        if not use:
            continue
        d = out / f"rocprof-pmc-{group}"
        cmd = [_rocprof(), "--pmc", *use, "--kernel-include-regex", kernel,
               "--kernel-iteration-range", "[2]", "-f", "csv", "-d", str(d), "-o", "run",
               "--"] + _capture_cmd(case, sub, seed)
        rc = _run(cmd, out / f"rocprof-pmc-{group}.log")
        rows = _csvs(d, "counter_collection")
        if (rc != 0 or not rows) and len(use) > 1:
            # the set may not fit in one hardware pass on this GPU: retry one counter per pass
            rows = []
            for one in use:
                d1 = out / f"rocprof-pmc-{group}-{one}"
                cmd1 = [_rocprof(), "--pmc", one, "--kernel-include-regex", kernel,
                        "--kernel-iteration-range", "[2]", "-f", "csv", "-d", str(d1), "-o", "run",
                        "--"] + _capture_cmd(case, sub, seed)
                if _run(cmd1, out / f"rocprof-pmc-{group}-{one}.log") == 0:
                    rows += _csvs(d1, "counter_collection")
        if not rows:
            skipped += use
            continue
        acc = defaultdict(float)
        for r in rows:
            acc[_col(r, "Counter_Name")] += float(_col(r, "Counter_Value") or 0)
        vals.update(acc)
    return {"values": dict(vals), "unavailable": sorted(set(skipped) - set(vals))}


def pc_sampling(out: Path, case: str, sub: Path, seed: int, kernel: str | None, interval_us: int) -> dict:
    d = out / "rocprof-pc"
    cmd = [_rocprof(), "--kernel-trace", "--pc-sampling-beta-enabled", "--pc-sampling-method", "host_trap",
           "--pc-sampling-unit", "time", "--pc-sampling-interval", str(interval_us),
           "-f", "csv", "-d", str(d), "-o", "run", "--"] + _capture_cmd(case, sub, seed)
    rc = _run(cmd, out / "rocprof-pc.log")
    rows = _csvs(d, "pc_sampling")
    if not rows:
        return {"error": f"no PC samples (rc={rc}); see rocprof-pc.log"}
    ids = {str(_col(r, "Dispatch_Id")) for r in _csvs(d, "kernel_trace")
           if not kernel or re.search(kernel, _col(r, "Kernel_Name") or "")}
    insts = Counter()
    undecoded = 0
    for r in rows:
        if ids and str(_col(r, "Dispatch_Id")) not in ids:
            continue
        ins = (_col(r, "Instruction") or "").strip()
        if ins:
            insts[ins] += 1
        else:
            undecoded += 1
    total = sum(insts.values())
    cls = Counter()
    for ins, n in insts.items():
        m = ins.split()[0] if ins else ""
        cls[next((c for c, rx in CLASSES if re.search(rx, m)), "other")] += n
    if not total:
        return {"error": f"{undecoded} candidate samples, none decoded to an instruction "
                         "(code object not recognized by rocprofv3)", "undecoded": undecoded}
    return {"samples": total, "undecoded": undecoded,
            "classes": {k: v / total for k, v in cls.most_common()} if total else {},
            "top": [{"instruction": i, "share": n / total} for i, n in insts.most_common(20)] if total else []}


def timeline(out: Path, case: str, sub: Path, seed: int) -> dict:
    d = out / "timeline"
    rc = _run([sys.executable, "-m", "megabench.integrations.timeline_profile_case", "--case", case,
               "--submission", str(sub), "--seed", str(seed), "--output", str(d)], out / "timeline.log")
    s = d / "summary.json"
    if rc != 0 or not s.exists():
        tail = (out / "timeline.log").read_text(errors="replace")[-400:]
        return {"error": f"timeline unavailable (rc={rc}): {tail.strip().splitlines()[-1] if tail.strip() else ''}"}
    return json.loads(s.read_text())


def findings(rep: dict, bw_tb_s: float, floor_ms: float | None) -> list[str]:
    f = []
    k = rep.get("kernel") or {}
    t_us = k.get("dur_us")
    c = (rep.get("counters") or {}).get("values", {})
    if t_us:
        line = f"Kernel {t_us/1e3:.3f} ms per call"
        if floor_ms:
            line += f"; HBM floor {floor_ms:.3f} ms -> {100*floor_ms/(t_us/1e3):.1f}% of SOL"
        f.append(line + ".")
        if "FETCH_SIZE" in c:
            gbps = c["FETCH_SIZE"] * 1024 / (t_us * 1e-6) / 1e9  # FETCH_SIZE is in KB
            f.append(f"HBM read {c['FETCH_SIZE']/1024:.1f} MB at {gbps/1e3:.2f} TB/s = {100*gbps/1e3/bw_tb_s:.1f}% of "
                     f"{bw_tb_s} TB/s peak (includes re-reads; compare with sol_info minimum_bytes).")
    if "TCC_HIT_sum" in c and "TCC_MISS_sum" in c and c["TCC_HIT_sum"] + c["TCC_MISS_sum"] > 0:
        f.append(f"L2 hit rate {100*c['TCC_HIT_sum']/(c['TCC_HIT_sum']+c['TCC_MISS_sum']):.1f}%.")
    if "SQ_WAVE_CYCLES" in c and "SQ_WAIT_INST_ANY" in c and c["SQ_WAVE_CYCLES"]:
        f.append(f"Waves waiting on dependencies {100*c['SQ_WAIT_INST_ANY']/c['SQ_WAVE_CYCLES']:.1f}% of wave cycles.")
    if c.get("SQ_LDS_BANK_CONFLICT") and c.get("SQ_LDS_IDX_ACTIVE"):
        f.append(f"LDS bank-conflict cycles {100*c['SQ_LDS_BANK_CONFLICT']/c['SQ_LDS_IDX_ACTIVE']:.1f}% of LDS active cycles.")
    pc = rep.get("pc_sampling") or {}
    if pc.get("classes"):
        cl = pc["classes"]
        top = ", ".join(f"{k} {100*v:.0f}%" for k, v in list(cl.items())[:6])
        f.append(f"PC samples ({pc['samples']}): {top}.")
        sync = cl.get("spin/sleep", 0) + cl.get("barrier", 0)
        if sync > 0.25:
            f.append(f"{100*sync:.0f}% of samples are in sleep/barrier: waves mostly wait for other blocks "
                     "(dependency/poll latency or load imbalance), not for their own work.")
        if cl.get("mem-wait", 0) > 0.25:
            f.append(f"{100*cl['mem-wait']:.0f}% of samples are in s_waitcnt: memory latency is exposed; "
                     "more loads in flight (prefetch depth, wider vector loads, more waves) is the lever.")
    tl = rep.get("timeline") or {}
    phases = tl.get("phases") or []
    if phases:
        tot = sum(p.get("wall_us", 0) for p in phases) or 1
        ranked = sorted(phases, key=lambda p: -p.get("wall_us", 0))[:5]
        f.append("Timeline phases by wall time (summed over occurrences): " + "; ".join(
            f"{p['event']} {p['wall_us']:.0f} us ({100*p['wall_us']/tot:.0f}%), imbalance {p.get('imbalance_us', 0):.0f}, "
            f"skew {p.get('begin_skew_us', 0):.0f}" for p in ranked) + ".")
        worst = max(phases, key=lambda p: p.get("imbalance_us", 0))
        if worst.get("imbalance_us", 0) > 0.3 * max(worst.get("mean_us", 1), 1):
            f.append(f"Largest imbalance: {worst['event']} (slowest block {worst.get('slowest_us', 0):.0f} us vs mean "
                     f"{worst.get('mean_us', 0):.0f} us): rebalance its work or overlap it.")
    elif tl.get("error"):
        f.append(f"No phase timeline: {tl['error']} (export build_timeline to get one).")
    return f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--bandwidth-tb-s", type=float, default=5.3)
    ap.add_argument("--kernel", help="regex of the candidate kernel (default: longest dispatch)")
    ap.add_argument("--skip", default="", help="comma list of timeline,counters,pc")
    ap.add_argument("--pc-interval-us", type=int, default=50)
    a = ap.parse_args(argv)
    skip = {s for s in a.skip.split(",") if s}
    sub = a.submission.resolve(strict=True)
    out = a.output or Path("reports") / f"trace-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    out.mkdir(parents=True, exist_ok=False)
    rep: dict = {"case": a.case, "submission": str(sub)}
    disp = kernel_dispatches(out, a.case, sub, a.seed, None)
    kre = a.kernel or pick_kernel(disp)
    rep["kernel_regex"] = kre
    mine = [r for r in disp if kre and re.search(kre, r["name"])]
    rep["kernel"] = mine[-1] if mine else None            # last = warmed call
    rep["other_dispatches"] = len(disp) - len(mine)
    try:
        sol = json.loads(subprocess.check_output(
            [sys.executable, "-m", "megabench", "sol", "--case", a.case, "--bandwidth-tb-s", str(a.bandwidth_tb_s),
             "--json"], text=True, stderr=subprocess.DEVNULL))
        floor = sol["cases"][0].get("floor_ms") or sol["cases"][0].get("hbm_floor_ms")
    except Exception:
        floor = None
    rep["floor_ms"] = floor
    if "counters" not in skip and kre:
        rep["counters"] = counters(out, a.case, sub, a.seed, kre, available_counters(out))
    if "pc" not in skip:
        rep["pc_sampling"] = pc_sampling(out, a.case, sub, a.seed, kre, a.pc_interval_us)
    if "timeline" not in skip:
        rep["timeline"] = timeline(out, a.case, sub, a.seed)
    rep["findings"] = findings(rep, a.bandwidth_tb_s, floor)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    md = [f"# Trace report: {a.case}", "", f"Output: `{out}`", "", "## Findings (signals, not proof)", ""]
    md += [f"- {x}" for x in rep["findings"]]
    k = rep.get("kernel") or {}
    md += ["", "## Kernel dispatch", "", "```", json.dumps(k, indent=1), "```",
           f"Other (non-candidate) dispatches in the capture: {rep['other_dispatches']}"]
    if rep.get("counters"):
        md += ["", "## Counters (warmed call)", "", "```", json.dumps(rep["counters"], indent=1), "```"]
    pc = rep.get("pc_sampling") or {}
    if pc.get("top"):
        md += ["", "## Top sampled instructions", ""] + [f"- {100*t['share']:.1f}%  `{t['instruction']}`" for t in pc["top"]]
    elif pc.get("error"):
        md += ["", f"PC sampling: {pc['error']}"]
    if (rep.get("timeline") or {}).get("phases"):
        md += ["", "## Phase timeline", "", "Perfetto trace: `timeline/trace.json` (ui.perfetto.dev)", "", "```"]
        md += [json.dumps(p) for p in rep["timeline"]["phases"]] + ["```"]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
