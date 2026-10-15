"""Method 1: one MPK decode step vs. our megakernel at the exact MegaBench KV point.

``python -m megabench.sota.method1 run [--smoke]`` runs every arm of the study:

* ours: the existing SOTA harness with the ``opus`` arm at b1-s128 (correctness
  grading, interleaved steady/isolated timing, bootstrap CIs);
* MPK offline (``mpk_worker --mode offline``): native MPK kernels at several
  ``max_seq_length`` values with a 128-token prompt. Because MPK stops when
  ``step + 1 >= msl``, T(msl=130) - T(msl=129) is exactly the decode step at
  position 128 over 129 KV positions. Extra configurations: mbt=8 (the paper's
  setting) and CUTLASS off (the PR #10 probe's setting), and a slope over
  msl in {129, 130, 146, 162, 178, 193}, which spreads round-to-round noise
  over 64 steps and is the less noisy estimate of the step;
* MPK online_notoken (``mpk_worker --mode notoken``): one call = one step at
  step 128, timed with the host sync the mode needs (the PR #10 probe).

``python -m megabench.sota.method1 report RUN_DIR [--from EARLIER_RUN ...]``
re-renders ``method1.md`` and ``method1.json`` from the records in ``RUN_DIR``,
taking any part RUN_DIR lacks (for example after a ``--parts`` rerun) from the
earlier runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

from . import timing
from .records import new_run_dir, read_records

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MPK_PYTHON = "/raid/garv901/mpk/venv/bin/python"
STEP_PAIR = (129, 130)
SLOPE_MSL = (129, 130, 146, 162, 178, 193)
# name -> worker arguments; "main" carries the slope set as well as the step pair.
# ``--stops`` compiles ONE kernel and sets each stop length at runtime, so every
# point shares code, weights and buffers. Separate builds per msl ("twobuild")
# carry a build-to-build offset of about +-1 ms (0.6% of a ~160 ms run), larger
# than the step itself; that config is kept only to document the offset.
MPK_CONFIGS = {
    "main": ["--mode", "offline", "--stops", ",".join(map(str, SLOPE_MSL)),
             "--mbt", "1", "--cutlass", "on"],
    "mbt8": ["--mode", "offline", "--stops", "129,130", "--mbt", "8", "--cutlass", "on"],
    "nocutlass": ["--mode", "offline", "--stops", "129,130", "--mbt", "1", "--cutlass", "off"],
    "main-nocutlass": ["--mode", "offline", "--stops", ",".join(map(str, SLOPE_MSL)),
                       "--mbt", "1", "--cutlass", "off"],
    "twobuild": ["--mode", "offline", "--msl", "129,130", "--mbt", "1", "--cutlass", "on"],
    # The PR #10 probe builds with PersistentKernel defaults, i.e. CUTLASS off.
    "notoken": ["--mode", "notoken"],
}
OFFLINE_CONFIGS = ("main", "main-nocutlass", "mbt8", "nocutlass", "twobuild")
NSYS = "/usr/local/cuda/bin/nsys"
NCU = "/usr/local/cuda/bin/ncu"
SEED = 7301


# --------------------------------------------------------------------------
# Running


def _mpk_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env.setdefault("MPK_COMPILE_ROOT", "/raid/garv901/.cache/megabench_sota/mpk_builds")
    return env


def _run(cmd: list[str], log: Path, env: dict | None = None, timeout: int = 3600) -> int:
    print("+", " ".join(cmd), flush=True)
    with log.open("a") as fh:
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                               cwd=str(log.parent), timeout=timeout)
            return r.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\nTIMEOUT after {timeout} s\n")
            return 124


def run(args) -> int:
    run_dir = args.out_dir or new_run_dir(args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    rounds, reps = (2, 5) if args.smoke else (args.rounds, args.reps)
    status = 0
    if "ours" in args.parts:
        cmd = [sys.executable, "-m", "megabench.sota", "run", "--arms", "opus",
               "--context", "128", "--no-profile", "--seeds", str(SEED),
               "--out-dir", str(run_dir / "ours"), "--tag", "method1-ours"]
        if args.smoke:
            cmd.append("--smoke")
        rc = _run(cmd, run_dir / "ours.log", env=dict(os.environ, PYTHONPATH=str(REPO)))
        print(f"ours: exit {rc}", flush=True)
        status |= rc != 0
    for name in [p for p in args.parts if p in MPK_CONFIGS]:
        work = run_dir / f"mpk-{name}"
        work.mkdir(exist_ok=True)  # online_notoken writes ./permanent_output_dir here
        cmd = [args.mpk_python, "-m", "megabench.sota.mpk_worker", *MPK_CONFIGS[name],
               "--seed", str(SEED), "--rounds", str(rounds), "--reps", str(reps),
               "--tag", name, "--out", str(run_dir / f"mpk-{name}.jsonl")]
        if args.smoke and name.startswith("main"):
            cmd[cmd.index("--stops") + 1] = "129,130"
        rc = _run(cmd, work / "worker.log", env=_mpk_env())
        print(f"mpk {name}: exit {rc}", flush=True)
        status |= rc != 0
    if "nsys" in args.parts:
        status |= run_nsys(args, run_dir)
    if "ncu" in args.parts:
        status |= run_ncu(args, run_dir)
    summary = analyze(run_dir)
    (run_dir / "method1.json").write_text(json.dumps(summary, indent=2, default=str))
    text = render(summary)
    (run_dir / "method1.md").write_text(text)
    print(text)
    return int(status)


def run_nsys(args, run_dir: Path) -> int:
    """Nsight Systems kernel timelines of both single-step paths (nsys_check.py)."""
    out_dir = run_dir / "nsys"
    out_dir.mkdir(exist_ok=True)
    reps = "10" if args.smoke else "50"
    sides = {"mpk": (args.mpk_python, _mpk_env()),
             "opus": (sys.executable, dict(os.environ, PYTHONPATH=str(REPO)))}
    status = 0
    for side, (python, env) in sides.items():
        work = out_dir / side
        work.mkdir(exist_ok=True)  # online_notoken writes ./permanent_output_dir here
        rep = work / f"{side}"
        cmd = [NSYS, "profile", "-t", "cuda", "--capture-range=cudaProfilerApi",
               "--capture-range-end=stop", "--force-overwrite=true", "-o", str(rep),
               python, "-m", "megabench.sota.nsys_check", side, "--reps", reps]
        rc = _run(cmd, work / "nsys.log", env=env)
        if rc == 0:
            rc = _run([NSYS, "stats", "--force-export=true", "--report", "cuda_gpu_trace",
                       "--format", "csv", "--output", str(rep), f"{rep}.nsys-rep"],
                      work / "nsys.log", env=env)
        if rc == 0 and not Path(f"{rep}.sqlite").exists():
            _run([NSYS, "export", "--type", "sqlite", "--force-overwrite=true",
                  "--output", f"{rep}.sqlite", f"{rep}.nsys-rep"], work / "nsys.log", env=env)
        csvs = sorted(work.glob(f"{side}*cuda_gpu_trace*.csv"))
        if rc == 0 and csvs:
            from .nsys_check import summarize
            (out_dir / f"{side}.json").write_text(json.dumps(summarize(csvs[-1]), indent=2))
        print(f"nsys {side}: exit {rc}", flush=True)
        status |= rc != 0 or not csvs
    return int(status)


def run_ncu(args, run_dir: Path) -> int:
    """Nsight Compute reports (.ncu-rep for the GUI, raw/details CSV for analysis).

    * opus: ``--set full`` on 3 ``qwen3_step`` launches inside the profiler range.
    * mpk: MPK's worker kernel spins on events from its concurrently running
      scheduler kernel, so per-kernel replay (which serializes kernels) cannot
      profile it. Range replay profiles the whole profiler range (one decode
      call) as one workload; ``range`` replays in-process, ``app-range`` re-runs
      the process per pass. Each attempt has a timeout; the log records hangs.
      Range replay takes its ranges from cudaProfilerStart/Stop itself and
      rejects ``--profile-from-start``.
    """
    out_dir = run_dir / "ncu"
    out_dir.mkdir(exist_ok=True)
    status = 0
    attempts = {
        "opus": [(sys.executable, dict(os.environ, PYTHONPATH=str(REPO)),
                  ["--set", "full", "--profile-from-start", "off", "--kernel-name", "qwen3_step",
                   "--launch-count", "3"], ["opus", "--reps", "5"], 1200)],
        "mpk": [(args.mpk_python, _mpk_env(),
                 ["--replay-mode", mode,
                  "--section", "SpeedOfLight", "--section", "MemoryWorkloadAnalysis",
                  "--section", "LaunchStats", "--section", "Occupancy"],
                 ["mpk", "--reps", "1", "--warmup", "3"], timeout)
                for mode, timeout in (("range", 480), ("app-range", 900))],
    }
    for side, tries in attempts.items():
        work = out_dir / side
        work.mkdir(exist_ok=True)
        ok = False
        for python, env, ncu_flags, script_args, timeout in tries:
            tag = side if side == "opus" else f"{side}-{ncu_flags[1]}"
            rep = work / tag
            # ncu locks SM clocks to base (1.12 GHz on B200) by default; the timings
            # ran at the boost clock, so profile there too.
            cmd = [NCU, "--clock-control", "none", *ncu_flags, "--target-processes", "all",
                   "-f", "-o", str(rep),
                   python, "-m", "megabench.sota.nsys_check", *script_args]
            rc = _run(cmd, work / "ncu.log", env=env, timeout=timeout)
            have = Path(f"{rep}.ncu-rep").exists()
            print(f"ncu {tag}: exit {rc}, report {'written' if have else 'missing'}", flush=True)
            if rc == 0 and have:
                for page in ("raw", "details"):
                    with open(work / f"{tag}_{page}.csv", "w") as fh:
                        subprocess.run([NCU, "--import", f"{rep}.ncu-rep", "--page", page,
                                        "--csv"], stdout=fh, stderr=subprocess.STDOUT)
                ok = True
                break
        status |= not ok
    return int(status)


# --------------------------------------------------------------------------
# Analysis


def _load(path: Path) -> list[dict]:
    return read_records(path) if path.exists() else []


def _stats(x: list[float]) -> dict:
    return timing.summarize(x, SEED) if x else {"n": 0}


def analyze_ours(records: list[dict]) -> dict:
    out: dict = {"correctness": {}, "timing": {}}
    for r in records:
        if r.get("record_type") == "correctness" and r.get("arm") == "opus":
            out["correctness"][r["variant"]] = {
                "grade": r.get("grade"), "grading": r.get("grading"),
                "megabench": r.get("megabench"), "mutated_inputs": r.get("mutated_inputs"),
                "bitexact_vs_submission": (r.get("config") or {}).get("bitexact_vs_submission")}
        elif r.get("record_type") == "timing" and r.get("arm") == "opus":
            key = f"{r['variant']}:{r['mode']}:{r.get('l2', 'warm')}"
            out["timing"][key] = {"stats": r.get("stats"), "n_steps": r.get("n_steps"),
                                  "throttled_rounds": r.get("throttled_rounds"),
                                  "clock": r.get("clock")}
        elif r.get("record_type") == "error" and r.get("arm") in ("opus", None):
            out.setdefault("errors", []).append(
                {k: r.get(k) for k in ("arm", "variant", "stage", "error")})
    return out


def _samples_by_msl(rows: list[dict]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for r in rows:
        out.setdefault(int(r["msl"]), []).extend(r["samples_ms"])
    return out


def _round_medians(rows: list[dict]) -> dict[int, dict[int, float]]:
    """round -> {msl: median T of that round's reps}."""
    by: dict[int, dict[int, float]] = {}
    for r in rows:
        by.setdefault(r["round"], {})[int(r["msl"])] = r["median_ms"]
    return dict(sorted(by.items()))


def _paired_rounds(rows: list[dict], a: int, b: int) -> list[float]:
    """Per-round median(T_b) - median(T_a): drift-robust step estimates."""
    return [v[b] - v[a] for v in _round_medians(rows).values() if a in v and b in v]


def _round_slopes(rows: list[dict], msls: list[int]) -> list[float]:
    """Least-squares slope of each round's medians over the full stop set."""
    return [timing.linear_fit(msls, [v[m] for m in msls])["slope"]
            for v in _round_medians(rows).values() if all(m in v for m in msls)]


def analyze_offline(records: list[dict]) -> dict:
    builds = [r for r in records if r.get("kind") == "mpk_build"]
    rows = [r for r in records if r.get("kind") == "mpk_timing"]
    errors = [r for r in records if r.get("kind") == "mpk_error"] + \
        [r for r in builds if not r.get("ok")]
    by_msl = _samples_by_msl(rows)
    out: dict = {
        "builds": [{k: b.get(k) for k in ("msl", "mbt", "cutlass", "ok", "compile_ms",
                                          "generated", "expected_generated", "final_step",
                                          "workers", "schedulers", "error")}
                   for b in builds],
        "per_msl": {m: _stats(x) for m, x in sorted(by_msl.items())},
        "errors": [{k: e.get(k) for k in ("msl", "error")} for e in errors],
    }
    gen_ok = all(r["generated"] == int(r["msl"]) - 128 for r in rows) and \
        all(b.get("generated") == b.get("expected_generated") for b in builds if b.get("ok"))
    out["generated_length_ok"] = bool(rows) and gen_ok
    heads: dict[int, set] = {}
    for r in rows:
        heads.setdefault(int(r["msl"]), set()).add(tuple(r.get("tokens_head") or ()))
    out["tokens_deterministic"] = bool(heads) and all(len(v) == 1 for v in heads.values())
    # Reps within a round share that round's drift, so the CIs are Student-t
    # intervals over per-round estimates; the rep-level bootstrap is kept for
    # reference only (it is several times too narrow here).
    a, b = STEP_PAIR
    if a in by_msl and b in by_msl:
        d = timing.diff_ci(by_msl[b], by_msl[a], SEED)
        paired = _paired_rounds(rows, a, b)
        tci = timing.t_interval(paired)
        ci = tci["ci95"]
        out["step"] = {"diff_ms": tci["mean"], "ci95": ci,
                       "paired_round_diffs_ms": paired,
                       "paired_median_ms": statistics.median(paired) if paired else None,
                       "pooled_diff_ms": d["diff"], "ci95_rep_bootstrap": d["ci95"],
                       "ci_half_width_pct": (100 * (ci[1] - ci[0]) / 2 / tci["mean"]
                                             if ci and tci["mean"] else None)}
    if len(by_msl) > 2:
        sl = timing.slope_ci(by_msl, SEED)
        per_round = _round_slopes(rows, sorted(by_msl))
        tci = timing.t_interval(per_round)
        # slope = mean of the per-round slopes (matches its CI); intercept and
        # residuals stay those of the fit to the pooled medians.
        sl.update({"pooled_slope_ms": sl["slope"], "ci95_rep_bootstrap": sl["ci95"],
                   "round_slopes_ms": per_round, "ci95": tci["ci95"]})
        if tci["mean"] is not None:
            sl["slope"] = tci["mean"]
        out["slope"] = sl
    clocks = [r.get("clock_after", {}).get("sm_mhz") for r in rows if r.get("clock_after")]
    clocks = [c for c in clocks if c]
    throttled = [r["round"] for r in rows
                 if (r.get("clock_after", {}).get("throttle_mask", 0) & ~0x3)]
    out["sm_mhz"] = {"min": min(clocks), "max": max(clocks)} if clocks else None
    out["throttled_rounds"] = sorted(set(throttled))
    if rows:
        out["meta"] = {k: rows[0].get(k) for k in ("gpu", "mpk_commit", "torch", "mbt", "cutlass",
                                                    "stop_method", "compile_msl")}
    return out


def analyze_notoken(records: list[dict]) -> dict:
    rows = [r for r in records if r.get("kind") == "mpk_timing"]
    checks = [r for r in records if r.get("kind") == "mpk_check"]
    x = [v for r in rows for v in r["samples_ms"]]
    return {"call": _stats(x), "check": checks[0] if checks else None,
            "errors": [r.get("error") for r in records if r.get("kind") == "mpk_error"]}


_DEMO_LINE = re.compile(r"Prompt length (\d+), generate length (\d+), "
                        r"per-token latency:? ([0-9.]+) ms")


def analyze_demo(demo_dir: Path) -> dict | None:
    """MPK's stock demo/qwen3/demo.py runs (slurm/mpk_demo_check.sbatch), if present.

    The demo prints total time / (prompt + generated); total T = that x msl. Each
    msl is its own build, so slopes between far-apart msl values are used (the
    build offset is small next to 64+ steps).
    """
    if not demo_dir.is_dir():
        return None
    runs = []
    for log in sorted(demo_dir.glob("demo-msl*-rep*/demo.log")):
        m = _DEMO_LINE.findall(log.read_text(errors="replace"))
        if not m:
            runs.append({"dir": log.parent.name, "error": "no latency line"})
            continue
        prompt, gen, per_tok = int(m[-1][0]), int(m[-1][1]), float(m[-1][2])
        runs.append({"dir": log.parent.name, "msl": prompt + gen, "prompt": prompt,
                     "generated": gen, "per_token_ms": per_tok,
                     "total_ms": per_tok * (prompt + gen)})
    by: dict[int, list[float]] = {}
    for r in runs:
        if "total_ms" in r:
            by.setdefault(r["msl"], []).append(r["total_ms"])
    med = {k: statistics.median(v) for k, v in sorted(by.items())}
    keys = list(med)
    slopes = [{"from": a, "to": b, "ms_per_step": (med[b] - med[a]) / (b - a)}
              for a, b in zip(keys, keys[1:])]
    return {"runs": runs, "median_total_ms": med, "slopes": slopes}


def analyze(run_dir: Path, fallback_dirs: tuple[Path, ...] = ()) -> dict:
    """Summarize RUN_DIR; parts it lacks are taken from ``fallback_dirs`` in order."""
    dirs = [Path(run_dir), *map(Path, fallback_dirs)]

    def find(rel: str) -> Path | None:
        return next((d / rel for d in dirs if (d / rel).exists()), None)

    sources = {}

    def take(key: str, rel: str) -> Path | None:
        p = find(rel)
        if p is not None:
            sources[key] = str(p)
        return p

    ours = take("ours", "ours/records.jsonl")
    demo = take("demo", "demo-check")
    out = {"run_dir": str(dirs[0]), "sources": sources,
           "ours": analyze_ours(_load(ours) if ours else []),
           "mpk": {}, "demo": analyze_demo(demo) if demo else None, "nsys": {}}
    for side in ("mpk", "opus"):
        p = take(f"nsys:{side}", f"nsys/{side}.json")
        if p:
            out["nsys"][side] = json.loads(p.read_text())
    for name in MPK_CONFIGS:
        p = take(f"mpk:{name}", f"mpk-{name}.jsonl")
        recs = _load(p) if p else []
        if not recs:
            continue
        out["mpk"][name] = analyze_notoken(recs) if name == "notoken" else analyze_offline(recs)
    return out


# --------------------------------------------------------------------------
# Rendering


def _ms(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def _ci(ci) -> str:
    return "—" if not ci else f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _worker_kernel(summary: dict | None) -> dict | None:
    """The longest-running MPK kernel per call: the persistent worker kernel."""
    if not summary:
        return None
    ks = summary.get("kernels", {})
    cands = {k: v for k, v in ks.items() if "prepare" not in k.lower()
             and "init" not in k.lower()}
    return max(cands.values(), key=lambda v: v["median_us"]) if cands else None


def _opus_kernel(summary: dict | None) -> dict | None:
    if not summary:
        return None
    return summary.get("kernels", {}).get("qwen3_step")


def _check_text(name: str, v: dict) -> str:
    err = v.get("max_abs_error")
    tail = "" if err is None else f" (max abs err {err:.3g})"
    return f"{name}: {v.get('mismatched')} mismatched{tail}"


def render(s: dict) -> str:
    ours, mpk = s["ours"], s["mpk"]
    t = ours.get("timing", {})
    steady = (t.get("direct:steady:warm") or {}).get("stats") or {}
    iso = (t.get("direct:isolated:warm") or {}).get("stats") or {}
    main = mpk.get("main", {})
    step = main.get("step", {})
    call = (mpk.get("notoken") or {}).get("call", {})
    lines = ["# Method 1: MPK vs. Opus megakernel at the MegaBench b1-s128 point", "",
             f"Run: `{s['run_dir']}`", ""]
    other = sorted({k: v for k, v in (s.get("sources") or {}).items()
                    if not v.startswith(s["run_dir"] + "/")}.items())
    if other:
        lines += ["Parts taken from other runs:", ""]
        lines += [f"- {k}: `{v}`" for k, v in other] + [""]
    lines += ["Both systems decode the token at position 128 over 129 KV positions "
              "(Qwen3-0.6B, batch 1, MegaBench synthetic BF16 weights, seed 7301).", "",
              "## Headline", "",
              "| View | MPK (ms) | MPK 95% CI | Opus (ms) | Opus 95% CI | MPK / Opus |",
              "|---|---|---|---|---|---|"]

    def row(view, a, aci, b, bci):
        ratio = f"{a / b:.2f}x" if a and b else "—"
        lines.append(f"| {view} | {_ms(a)} | {_ci(aci)} | {_ms(b)} | {_ci(bci)} | {ratio} |")

    for name, label in (("main", "CUTLASS on"), ("main-nocutlass", "CUTLASS off")):
        sl = mpk.get(name, {}).get("slope")
        if sl:
            row(f"Device step, MPK {label}: slope over positions 128–191 vs Opus steady",
                sl.get("slope"), sl.get("ci95"), steady.get("median"),
                steady.get("ci95_median"))
    row("Device step at position 128 only: MPK ΔT(msl 130−129) vs Opus steady",
        step.get("diff_ms"), step.get("ci95"), steady.get("median"), steady.get("ci95_median"))
    row("Call contract: MPK online_notoken call vs Opus isolated", call.get("median"),
        call.get("ci95_median"), iso.get("median"), iso.get("ci95_median"))
    ns = s.get("nsys") or {}
    mk = _worker_kernel(ns.get("mpk"))
    ok = _opus_kernel(ns.get("opus"))
    if mk or ok:
        row("GPU kernel only (nsys): MPK worker kernel vs Opus qwen3_step",
            mk and mk["median_us"] / 1e3, None, ok and ok["median_us"] / 1e3, None)
    lines += ["", "## MPK offline configurations", "",
              "CIs are Student-t 95% intervals over the rounds (one estimate per round).", "",
              "| Config | stops | ΔT (ms) | 95% CI | CI ± % | per-round ΔT (ms) | "
              "slope (ms/step) | slope 95% CI | gen. length ok | tokens deterministic | "
              "SM MHz | throttled rounds |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in OFFLINE_CONFIGS:
        m = mpk.get(name)
        if not m:
            continue
        st = m.get("step", {})
        sl = m.get("slope") or {}
        hw = st.get("ci_half_width_pct")
        clk = m.get("sm_mhz") or {}
        how = (m.get("meta") or {}).get("stop_method", "—")
        per = ", ".join(f"{v:.3f}" for v in st.get("paired_round_diffs_ms") or []) or "—"
        lines.append(
            f"| {name} | {how} | {_ms(st.get('diff_ms'))} | {_ci(st.get('ci95'))} | "
            f"{'—' if hw is None else f'{hw:.1f}'} | {per} | {_ms(sl.get('slope'))} | "
            f"{_ci(sl.get('ci95'))} | {m.get('generated_length_ok')} | "
            f"{m.get('tokens_deterministic')} | {clk.get('min', '—')}–{clk.get('max', '—')} | "
            f"{m.get('throttled_rounds')} |")
    for name in OFFLINE_CONFIGS:
        m = mpk.get(name) or {}
        sl = m.get("slope")
        if not sl:
            continue
        rs = ", ".join(f"{v:.4f}" for v in sl.get("round_slopes_ms") or [])
        lines += ["", f"## Slope ({name})", "",
                  f"Slope of median T over msl {sorted(sl['residuals'], key=int)}: "
                  f"**{_ms(sl['slope'])} ms/step** (95% CI {_ci(sl['ci95'])} over per-round "
                  f"slopes {rs}), intercept {_ms(sl['intercept'])} ms. It is the mean step "
                  "over positions 128–191.", "",
                  "| msl | median T (ms) | residual (ms) |", "|---|---|---|"]
        for m_, st in m.get("per_msl", {}).items():
            lines.append(f"| {m_} | {_ms(st.get('median'))} | "
                         f"{_ms(sl['residuals'].get(int(m_), sl['residuals'].get(str(m_))))} |")
    demo = s.get("demo")
    if demo:
        lines += ["", "## Cross-check: MPK's stock demo (paper artifact settings)", "",
                  "`demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-0.6B "
                  "--max-num-batched-requests 1 --ignore-eos` (real weights, its own graph, "
                  "mbt 8, page 4096, CUTLASS on), as `artifact_evaluation/B200/run_tgx.sh` runs "
                  "it. Per-token latency = total / (prompt + generated), the paper's metric.", "",
                  "| run | msl | prompt | generated | per-token (ms) | total (ms) |",
                  "|---|---|---|---|---|---|"]
        for r in demo["runs"]:
            lines.append(f"| {r['dir']} | {r.get('msl', '—')} | {r.get('prompt', '—')} | "
                         f"{r.get('generated', '—')} | {_ms(r.get('per_token_ms'))} | "
                         f"{_ms(r.get('total_ms'))} |")
        lines += [""] + [f"- Slope msl {sl['from']}→{sl['to']}: **{sl['ms_per_step']:.4f} ms/step**"
                         for sl in demo["slopes"]]
    ns = s.get("nsys") or {}
    if ns:
        lines += ["", "## Nsight Systems kernel timeline (one step at b1-s128)", "",
                  "| Side | kernel | n | median (us) | min (us) | max (us) |",
                  "|---|---|---|---|---|---|"]
        for side, summ in ns.items():
            for k, v in sorted(summ.get("kernels", {}).items(),
                               key=lambda kv: -kv[1]["median_us"]):
                lines.append(f"| {side} | `{k}` | {v['n']} | {v['median_us']:.1f} | "
                             f"{v['min_us']:.1f} | {v['max_us']:.1f} |")
            span = summ.get("mpk_call_span_us")
            if span:
                lines.append(f"| {side} | call span (prepare start → last kernel end) | "
                             f"{span['n']} | {span['median_us']:.1f} | {span['min_us']:.1f} | "
                             f"{span['max_us']:.1f} |")
    lines += ["", "## Our kernel (Opus run B) by variant and mode", "",
              "| Variant:mode:L2 | median (ms) | 95% CI | n |", "|---|---|---|---|"]
    for k, v in sorted(t.items()):
        st = v.get("stats") or {}
        lines.append(f"| {k} | {_ms(st.get('median'))} | {_ci(st.get('ci95_median'))} | "
                     f"{st.get('n', '—')} |")
    lines += ["", "## Correctness", ""]
    for v, c in sorted(ours.get("correctness", {}).items()):
        lines.append(f"- Opus `{v}`: grade **{c.get('grade')}** ({c.get('grading')}), "
                     f"bit-exact vs submission: {c.get('bitexact_vs_submission')}, "
                     f"mutated inputs: {c.get('mutated_inputs')}")
    chk = (mpk.get("notoken") or {}).get("check")
    if chk:
        outs = ", ".join(_check_text(k, v) for k, v in chk["outputs"].items())
        lines.append(f"- MPK online_notoken vs MegaBench reference (strict atol/rtol): "
                     f"strict pass **{chk.get('strict_pass')}**; {outs}")
    errs = [(n, e) for n, m in mpk.items() for e in (m.get("errors") or []) if e]
    errs += [("ours", e) for e in ours.get("errors", [])]
    if errs:
        lines += ["", "## Errors", ""]
        lines += [f"- {n}: `{str(e)[:300]}`" for n, e in errs]
    lines += ["", "## Notes", "",
              "- ΔT includes MPK's per-iteration `prepare_next_batch` scheduling; launch, "
              "prefill, printfs and shutdown cancel between the two stop lengths.",
              "- MPK CIs are Student-t intervals over rounds: reps in one round share that "
              "round's drift, so a bootstrap over pooled reps (kept in method1.json as "
              "`ci95_rep_bootstrap`) is too narrow. One ΔT moves by about ±0.15 ms between "
              "rounds; the slope divides that over 64 steps and is the better estimate.",
              "- MPK `GPU kernel only` is the `online_notoken` worker kernel: one launch, so it "
              "also holds the persistent kernel's start-up and shutdown.",
              "- `reinit` = one compiled kernel, stop length set at runtime (same code, weights "
              "and buffers for every point); `build` = one compile per stop length, which adds a "
              "build-to-build offset to every difference.",
              "- MPK greedy tokens vary between runs on these synthetic weights (near-tied "
              "logits); a dense step's work does not depend on token values.",
              "- Opus steady = back-to-back launches with device-side token feedback (no host "
              "sync); isolated = sync + one launch.",
              "- MPK online_notoken = 3 kernel launches + host sync per call (the mode does not "
              "join its streams to the caller)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="megabench.sota.method1", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out-dir", type=Path, default=None)
    r.add_argument("--tag", default="method1")
    r.add_argument("--parts", default="ours,main,main-nocutlass,mbt8,nocutlass,twobuild,"
                                      "notoken,nsys")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--reps", type=int, default=30)
    r.add_argument("--mpk-python", default=DEFAULT_MPK_PYTHON)
    r.add_argument("--smoke", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("run_dir", type=Path)
    rp.add_argument("--from", dest="fallback", type=Path, action="append", default=[],
                    help="earlier run dir for parts RUN_DIR lacks (repeatable, first wins)")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        summary = analyze(args.run_dir, tuple(args.fallback))
        (args.run_dir / "method1.json").write_text(json.dumps(summary, indent=2, default=str))
        text = render(summary)
        (args.run_dir / "method1.md").write_text(text)
        print(text)
        return 0
    args.parts = [p for p in args.parts.split(",") if p]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
