"""Profile one warmed MegaBench candidate call with rocprofv3 (AMD ROCm).

The AMD counterpart of ``ncu_profile_case``. It runs that module (which warms the
candidate once, then makes one more call inside a roctx range) under ``rocprofv3``:

* always: kernel dispatch trace, per-kernel stats and a summary, so you see every
  launch, its grid/workgroup size, VGPR/SGPR/LDS/scratch use and duration;
* ``--pmc``: hardware counters for the kernels that match ``--kernel`` on their
  second dispatch (the warmed call). rocprofv3 fails when a counter set cannot be
  collected in one pass; split the counters over several runs. ``rocprofv3-avail``
  lists counters for the device.
* ``--att``: advanced thread trace (instruction-level timing) of the same dispatch.

Output goes to ``reports/rocprof-<timestamp>/``. Counters are diagnostic; correctness
and latency come only from ``check_candidate`` and the final grading.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--kernel", default=None,
                        help="regex of the candidate kernel name (filters counters and ATT)")
    parser.add_argument("--pmc", nargs="+", default=None,
                        help="counters, e.g. SQ_WAVES SQ_INSTS_VALU TCC_HIT_sum TCC_MISS_sum")
    parser.add_argument("--att", action="store_true", help="collect a thread trace")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="after --, extra rocprofv3 arguments")
    args = parser.parse_args(argv)

    tool = shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3"
    if not os.path.exists(tool):
        parser.error("rocprofv3 not found; load the rocm module")
    out = args.output or Path("reports") / f"rocprof-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    out.mkdir(parents=True, exist_ok=False)
    cmd = [tool, "--kernel-trace", "--marker-trace", "--stats", "--summary",
           "--summary-units", "usec", "-f", "csv", "-d", str(out), "-o", "run"]
    if args.kernel:
        cmd += ["--kernel-include-regex", args.kernel]
    if args.pmc or args.att:
        # The first matching dispatch is the warmup call; profile the warmed one.
        cmd += ["--kernel-iteration-range", "[2]"]
    if args.pmc:
        cmd += ["--pmc", *args.pmc]
    if args.att:
        cmd += ["--att"]
    extra = [a for a in args.extra if a != "--"]
    cmd += extra
    cmd += ["--", sys.executable, "-m", "megabench.integrations.ncu_profile_case",
            "--case", args.case, "--submission", str(args.submission.resolve()),
            "--seed", str(args.seed)]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    print(f"rocprofv3 output: {out}  (kernel_stats.csv, kernel_trace.csv, counter_collection.csv)")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
