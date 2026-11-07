"""Score a variant candidate against its parent in one GPU job.

``phase_ab PARENT_DIR VARIANT_DIR`` builds both candidates (each directory holds a complete
candidate: ``submission.py`` and the kernel files it loads next to it), checks the variant
against the reference, times parent and variant alternately (A B A B ...) with CUDA events,
then records one per-block timeline of each. It prints the end-to-end p50 of both and, for
each phase, the critical-path window of both (``timeline.phase_windows``), and writes
``result.json`` under the output directory. It exits non-zero when the variant is not
correct; the timings are then skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

from ..cases import load_regimes, select_cases


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
                     f"[{result['reps']} pairs x {result['iters']} calls]")
        lines.append(f"{'phase':<20} {'A window':>10} {'B window':>10} {'B-A':>10}   (us, summed over occurrences)")
        for row in result["phases"]:
            lines.append(f"{row['phase'][:20]:<20} {row['a_window_us']:>10.3f} "
                         f"{row['b_window_us']:>10.3f} {row['diff_window_us']:>+10.3f}")
    return "\n".join(lines)


class GpuBackend:
    """The GPU parts: build, check, time and timeline. Tests replace it with a fake."""

    def __init__(self, case, device: str = "cuda:0") -> None:
        self.case, self.device = case, device

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
            return start.elapsed_time(end) / iters

        def warm(fn) -> None:
            with torch.inference_mode():
                for _ in range(warmup):
                    fn(inputs)
                torch.cuda.synchronize(self.device)

        return timer, warm

    def timeline(self, directory: Path, output: Path) -> dict:
        from . import timeline_profile_case as tpc

        with contextlib.redirect_stdout(io.StringIO()):
            tpc.capture(0, self.case.to_dict(), str(directory / "submission.py"), 7301, str(output))
        return json.loads((output / "summary.json").read_text()).get("windows", {})


def run(parent: Path, variant: Path, *, output: Path, reps: int, iters: int, warmup: int,
        names: list[str] | None, backend, case_id: str = "") -> tuple[dict, int]:
    result: dict = {"case": case_id, "parent": str(parent), "variant": str(variant),
                    "reps": reps, "iters": iters, "order": interleave(reps)}
    output.mkdir(parents=True, exist_ok=True)
    code = 0
    cand = {"A": backend.load(parent), "B": backend.load(variant)}
    try:
        backend.check(cand["B"])
        result.update(correct=True, error=None)
    except Exception as exc:  # noqa: BLE001  # a wrong or crashing variant is a result, not a crash
        result.update(correct=False, error=f"{type(exc).__name__}: {exc}")
        code = 1
    if code == 0:
        timer, warm = backend.make_timer(iters, warmup)
        for side in "AB":
            warm(cand[side])
        samples = time_alternating(cand, reps, timer)
        a, b = statistics.median(samples["A"]), statistics.median(samples["B"])
        result["e2e"] = {"a_ms": samples["A"], "b_ms": samples["B"], "a_p50_ms": a, "b_p50_ms": b,
                         "diff_ms": b - a, "diff_pct": 100.0 * (b - a) / a}
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
    parser.add_argument("--output", type=Path,
                        default=Path(f"reports/phase-ab-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"))
    args = parser.parse_args(argv)
    if args.reps < 1 or args.iters < 1:
        parser.error("--reps and --iters must be at least 1")
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
    result, code = run(args.parent.resolve(), args.variant.resolve(), output=args.output,
                       reps=args.reps, iters=args.iters, warmup=args.warmup,
                       names=args.phase or None, backend=backend_factory(case), case_id=args.case)
    print(format_result(result))
    print(f"result: {args.output}/result.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
