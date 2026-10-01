"""Generate a protected VibeSys task for one MegaBench case."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ..cases import select_cases
from ..harness.runner import CONTRACT_FILES


TRUSTED = (*CONTRACT_FILES, "megabench/integrations/vibesys_case_evaluator.py")


def make_task(case_id: str, root: Path) -> Path:
    case = select_cases("all", [case_id])[0]
    if not case.ready:
        raise ValueError(f"case is not ready: {case_id}")
    task = root / ".vibesys" / "tasks" / case_id
    task.mkdir(parents=True, exist_ok=False)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in TRUSTED}
    checks = "\n".join(
        f"if hashlib.sha256((ROOT / {name!r}).read_bytes()).hexdigest() != {digest!r}:\n"
        f"    raise RuntimeError('trusted benchmark changed: {name}')"
        for name, digest in hashes.items())
    (task / "evaluator.py").write_text(
        "from pathlib import Path\nimport hashlib\nimport sys\n"
        "ROOT = Path(__file__).resolve().parents[3]\n"
        f"{checks}\n"
        "sys.path.insert(0, str(ROOT))\n"
        "from megabench.integrations.vibesys_case_evaluator import main\n"
        f"raise SystemExit(main({case_id!r}, sys.argv[1:]))\n")
    evaluator = f".vibesys/tasks/{case_id}/evaluator.py"
    (task / "vibesys.input.toml").write_text(
        "version = 1\n\n[agent]\ndomain = \"generic\"\n\n"
        f"[accuracy]\ncommand = [\"python\", \"{evaluator}\", \"accuracy\", "
        "\"--submission\", \"submission.py\"]\ntimeout_seconds = 1200\n\n"
        f"[benchmark]\ncommand = [\"python\", \"{evaluator}\", \"benchmark\", "
        "\"--submission\", \"submission.py\"]\ntimeout_seconds = 1800\n\n"
        "[benchmark.result]\njson_argument = \"--output-json\"\n"
        "metric = \"progress_score\"\n")
    (task / "OBJECTIVE.md").write_text(
        f"# One MegaBench case: {case_id}\n\n"
        f"Implement only `{case_id}`: {case.model}, {case.phase}, geometry "
        f"`{case.params}`. Read `megabench/docs/AGENT_TASK.md` and this case's "
        "PyTorch oracle. Edit only `submission.py` and candidate-owned helper "
        "files. Compute every required output from current inputs in at most "
        "one GPU kernel launch; preserve all inputs. Do not use prior candidate "
        "code, other checkouts, or `sandbox/` kernels. The protected evaluator "
        "checks only this assigned case. The shared `.venv` and "
        "`/usr/local/cuda` are mounted read-only for local build and checks.\n")
    return task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(make_task(args.case, args.root.resolve()))


if __name__ == "__main__":
    main()
