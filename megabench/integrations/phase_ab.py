"""Score a variant candidate against its parent in one GPU job.

``phase_ab PARENT_DIR VARIANT_DIR`` builds both candidates (each directory holds a complete
candidate: ``submission.py`` and the kernel files it loads next to it), checks the variant
against the reference, times parent and variant alternately (A B A B ...) with CUDA events
in ``--processes`` fresh processes, then records one per-block timeline of each. It prints
the end-to-end p50 of both and, for each phase, the critical-path window of both
(``timeline.phase_windows``), and writes ``result.json`` under the output directory. It
exits non-zero when the variant is not correct; the timings are then skipped.

A loaded candidate can land in a slow mode (``gpu_noise``): all its launches are slower by
a constant factor while a fresh load is not. Each side's p50 is therefore taken per
process; the reported p50 is the fast-cluster p50 over processes, and the result says
``unstable`` when the processes disagree by more than the tolerance.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from ..cases import load_regimes, select_cases
from ..heartbeat import beat
from . import gpu_noise


def interleave(reps: int) -> list[str]:
    """The timing order: A and B alternate, A first."""
    return ["A" if i % 2 == 0 else "B" for i in range(2 * reps)]


def time_alternating(fns: dict[str, object], reps: int, timer) -> dict[str, list[float]]:
    """Sample ``timer(fn)`` once per slot of :func:`interleave`; one list of samples per side."""
    samples: dict[str, list[float]] = {"A": [], "B": []}
    for side in interleave(reps):
        samples[side].append(timer(fns[side]))
    return samples


def phase_rows(a: dict, b: dict, names: list[str] | None) -> list[dict]:
    """Per phase: windows and bodies of A and B and their differences (B - A), largest A first."""
    rows = []
    for name in sorted(set(a) | set(b), key=lambda n: -float(a.get(n, {}).get("window_us", 0))):
        if names and name not in names:
            continue
        wa = float(a.get(name, {}).get("window_us", 0.0))
        wb = float(b.get(name, {}).get("window_us", 0.0))
        ba = float(a.get(name, {}).get("body_us", 0.0))
        bb = float(b.get(name, {}).get("body_us", 0.0))
        rows.append({"phase": name, "a_window_us": wa, "b_window_us": wb,
                     "diff_window_us": round(wb - wa, 3), "a_body_us": ba, "b_body_us": bb,
                     "diff_body_us": round(bb - ba, 3)})
    return rows


def format_result(result: dict) -> str:
    lines = [f"variant correct: {result['correct']}"
             + ("" if result["correct"] else f"  ({result['error']})")]
    e2e = result.get("e2e")
    if e2e:
        lines.append(f"end to end p50 (ms): A {e2e['a_p50_ms']:.4f}  B {e2e['b_p50_ms']:.4f}  "
                     f"B-A {e2e['diff_ms']:+.4f} ({e2e['diff_pct']:+.2f}%)  "
                     f"[{result.get('processes', 1)} processes x {result['reps']} pairs x "
                     f"{result['iters']} calls]")
        stab = result.get("stability")
        if stab:
            lines.append(f"timing stability: {stab['verdict'].upper()}; per-process B-A (ms): "
                         + ", ".join(f"{d:+.4f}" for d in stab["diffs_ms"]))
            for side, label in (("a", "A"), ("b", "B")):
                s = stab[side]
                lines.append(f"  {label} per-process p50 (ms): "
                             + ", ".join(f"{v:.4f}" for v in s["values_ms"])
                             + f"  spread {s['spread_pct']}%"
                             + (f"  slow process(es) {s['slow_processes']}: GPU noise, excluded"
                                if s["slow_processes"] else ""))
                if s.get("note"):
                    lines.append(f"    {s['note']}")
                if gpu_noise.score_line(s):
                    lines.append(f"  {label}" + gpu_noise.score_line(s))
            if stab["verdict"] != "stable":
                lines.append("  A and B above are scores: the slow mode only makes a run slower, "
                             "so the fast cluster and the best earlier run of the same code give "
                             "the time")
                lines.append("  to repeat it as a free stability rerun: add --rerun-of "
                             "<this result.json> to the same phase_ab command")
        lines.append(f"{'phase':<20} {'A window':>10} {'B window':>10} {'B-A':>10}   (us, summed over occurrences)")
        for row in result["phases"]:
            lines.append(f"{row['phase'][:20]:<20} {row['a_window_us']:>10.3f} "
                         f"{row['b_window_us']:>10.3f} {row['diff_window_us']:>+10.3f}")
    if result.get("rerun_of"):
        lines.append(gpu_noise.RERUN_NOTE + f" (of {result['rerun_of']})")
    return "\n".join(lines)


class GpuBackend:
    """The GPU parts: build, check, time and timeline. Tests replace it with a fake."""

    def __init__(self, case, device: str = "cuda:0") -> None:
        self.case, self.device = case, device
        self.case_args: list[str] = []  # how a worker process selects the same case

    def load(self, directory: Path):
        from ..harness.runner import _load_submission

        return _load_submission(directory / "submission.py", self.case)

    def check(self, candidate) -> None:
        """Raise when the candidate disagrees with the reference (one fresh seed)."""
        import torch
        from ..harness.correctness import check_trials

        with torch.inference_mode():
            check_trials(self.case, candidate, self.device, 1)

    def make_timer(self, iters: int, warmup: int):
        import secrets
        import torch
        from ..workloads import make_inputs

        with torch.inference_mode():
            inputs = make_inputs(self.case, secrets.randbits(32), self.device)

        def timer(fn) -> float:
            with torch.inference_mode():
                torch.cuda.synchronize(self.device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    fn(inputs)
                end.record()
                end.synchronize()
                beat("timed window")
            return start.elapsed_time(end) / iters

        def warm(fn) -> None:
            with torch.inference_mode():
                for _ in range(warmup):
                    fn(inputs)
                torch.cuda.synchronize(self.device)
                beat("warm-up")

        return timer, warm

    def time_process(self, parent: Path, variant: Path, reps: int, iters: int,
                     warmup: int) -> dict:
        """One fresh process: load both, time them alternately, time the canary."""
        cmd = [sys.executable, "-m", "megabench.integrations.phase_ab", str(parent),
               str(variant), *self.case_args, "--reps", str(reps), "--iters", str(iters),
               "--warmup", str(warmup), "--worker"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        lines = [x for x in out.stdout.splitlines() if x.startswith("{")]
        if out.returncode != 0 or not lines:
            raise RuntimeError(f"timing process failed: {(out.stderr or out.stdout)[-800:]}")
        return json.loads(lines[-1])

    def canary(self) -> dict:
        return gpu_noise.canary(self.device)

    def timeline(self, directory: Path, output: Path) -> dict:
        from . import timeline_profile_case as tpc

        with contextlib.redirect_stdout(io.StringIO()):
            tpc.capture(0, self.case.to_dict(), str(directory / "submission.py"), 7301, str(output))
        return json.loads((output / "summary.json").read_text()).get("windows", {})


def pair_identity(parent: Path, variant: Path) -> str:
    return gpu_noise.tree_digest(parent) + ":" + gpu_noise.tree_digest(variant)


def time_pair(backend, parent: Path, variant: Path, reps: int, iters: int, warmup: int) -> dict:
    """In this process: load both, warm both, time them alternately, time the canary."""
    cand = {"A": backend.load(parent), "B": backend.load(variant)}
    timer, warm = backend.make_timer(iters, warmup)
    for side in "AB":
        warm(cand[side])
    samples = time_alternating(cand, reps, timer)
    return {"a_ms": samples["A"], "b_ms": samples["B"], "canary": backend.canary()}


def stability(procs: list[dict], tolerance_pct: float) -> dict:
    """Per side, compare the per-process p50s; the fast-cluster p50 is the side's time."""
    out: dict = {"processes": len(procs)}
    for side in "ab":
        p50s = [statistics.median(p[f"{side}_ms"]) for p in procs]
        out[side] = gpu_noise.spread(p50s, tolerance_pct)
        out[side]["note"] = gpu_noise.canary_note([p.get("canary") for p in procs],
                                                  out[side]["slow_processes"], tolerance_pct)
    diffs = [statistics.median(p["b_ms"]) - statistics.median(p["a_ms"]) for p in procs]
    out["diffs_ms"] = [round(d, 4) for d in diffs]
    out["verdict"] = ("stable" if out["a"]["verdict"] == out["b"]["verdict"] == "stable"
                      else "unstable")
    return out


def run(parent: Path, variant: Path, *, output: Path, reps: int, iters: int, warmup: int,
        names: list[str] | None, backend, case_id: str = "", processes: int = 3,
        tolerance_pct: float = gpu_noise.DEFAULT_TOLERANCE_PCT,
        rerun_of: str | None = None, history: Path | None = None,
        setup: str = "") -> tuple[dict, int]:
    result: dict = {"case": case_id, "parent": str(parent), "variant": str(variant),
                    "reps": reps, "iters": iters, "order": interleave(reps),
                    "identity": pair_identity(parent, variant), "rerun_of": rerun_of}
    output.mkdir(parents=True, exist_ok=True)
    code = 0
    try:
        backend.check(backend.load(variant))
        result.update(correct=True, error=None)
    except Exception as exc:  # noqa: BLE001  # a wrong or crashing variant is a result, not a crash
        result.update(correct=False, error=f"{type(exc).__name__}: {exc}")
        code = 1
    if code == 0:
        procs = [backend.time_process(parent, variant, reps, iters, warmup)
                 for _ in range(processes)]
        stab = stability(procs, tolerance_pct)
        for side, tree in (("a", parent), ("b", variant)):
            key = gpu_noise.history_key("phase_ab", case_id, setup, gpu_noise.tree_digest(tree))
            gpu_noise.score(stab[side], gpu_noise.best_known(history, key), tolerance_pct)
            gpu_noise.add_history(history, key, stab[side], str(output / "result.json"))
        if "slow" in (stab["a"]["verdict"], stab["b"]["verdict"]):
            stab["verdict"] = "slow"
        # The score of each side (its lowest fast-cluster p50 over all its runs) is its time.
        a, b = stab["a"]["score_ms"], stab["b"]["score_ms"]
        result["e2e"] = {"a_ms": [x for p in procs for x in p["a_ms"]],
                         "b_ms": [x for p in procs for x in p["b_ms"]],
                         "a_p50_ms": a, "b_p50_ms": b,
                         "diff_ms": b - a, "diff_pct": 100.0 * (b - a) / a}
        result["stability"] = stab
        result["processes"] = processes
        windows = {side: backend.timeline(d, output / side) for side, d in
                   (("A", parent), ("B", variant))}
        result["phases"] = phase_rows(windows["A"], windows["B"], names)
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result, code


def main(argv: list[str] | None = None, backend_factory=GpuBackend) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parent", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--regimes", type=Path)
    parser.add_argument("--regime", help="id of the regime to score (default: the first)")
    parser.add_argument("--reps", type=int, default=5, help="A/B pairs")
    parser.add_argument("--iters", type=int, default=20, help="calls timed per sample")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--phase", action="append", default=[], help="report only this phase (repeat)")
    parser.add_argument("--processes", type=int, default=3,
                        help="fresh processes that each time A and B alternately")
    parser.add_argument("--tolerance-pct", type=float, default=gpu_noise.DEFAULT_TOLERANCE_PCT,
                        help="per-process p50s further than this from the fastest are noise")
    parser.add_argument("--rerun-of", type=Path,
                        help="an UNSTABLE or SLOW phase_ab result.json of the same two directories; "
                             "marks this run a stability rerun")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path,
                        default=Path(f"reports/phase-ab-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"))
    args = parser.parse_args(argv)
    if args.reps < 1 or args.iters < 1 or args.processes < 1:
        parser.error("--reps, --iters and --processes must be at least 1")
    for directory in (args.parent, args.variant):
        if not (directory / "submission.py").is_file():
            parser.error(f"no submission.py in {directory}")
    if args.parent.resolve() == args.variant.resolve():
        parser.error("PARENT_DIR and VARIANT_DIR are the same directory")
    case = select_cases("all", [args.case])[0]
    if args.regimes:
        case = replace(case, regimes=load_regimes(args.regimes))
    if case.regimes:
        pick = [sub for r, sub in case.regime_cases() if args.regime in (None, r["id"])]
        if not pick:
            parser.error(f"no regime {args.regime}")
        case = pick[0]
    if case.gpus != 1:
        parser.error("phase_ab scores single-GPU cases only")
    backend = backend_factory(case)
    backend.case_args = (["--case", args.case]
                         + (["--regimes", str(args.regimes.resolve())] if args.regimes else [])
                         + (["--regime", args.regime] if args.regime else []))
    if args.rerun_of is not None and not args.worker:
        ok, reason = gpu_noise.rerun_status(
            args.rerun_of, pair_identity(args.parent.resolve(), args.variant.resolve()))
        if not ok:
            parser.error(f"--rerun-of refused: {reason}")
        print(f"{gpu_noise.RERUN_NOTE} ({reason})")
    if args.worker:
        print(json.dumps(time_pair(backend, args.parent.resolve(), args.variant.resolve(),
                                   args.reps, args.iters, args.warmup)))
        return 0
    result, code = run(args.parent.resolve(), args.variant.resolve(), output=args.output,
                       reps=args.reps, iters=args.iters, warmup=args.warmup,
                       names=args.phase or None, backend=backend, case_id=args.case,
                       processes=args.processes, tolerance_pct=args.tolerance_pct,
                       rerun_of=str(args.rerun_of) if args.rerun_of else None,
                       # In a workspace, the history is the one ./check_candidate also keeps.
                       history=(Path("reports") if Path("reports").is_dir()
                                and Path("progress").is_dir() else args.output)
                       / gpu_noise.HISTORY_FILE,
                       setup="|".join(x for x in (args.regimes.name if args.regimes else "",
                                                   args.regime or "") if x))
    print(format_result(result))
    print(f"result: {args.output}/result.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
