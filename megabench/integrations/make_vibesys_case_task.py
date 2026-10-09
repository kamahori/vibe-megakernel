"""Generate a protected VibeSys task for one MegaBench case."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

from ..cases import select_cases
from ..harness.runner import CONTRACT_FILES


TRUSTED = (*CONTRACT_FILES, "megabench/integrations/vibesys_case_evaluator.py")


def make_task(case_id: str, root: Path,
              benchmark_root: Path | None = None) -> Path:
    case = select_cases("all", [case_id])[0]
    if not case.ready:
        raise ValueError(f"case is not ready: {case_id}")
    root = root.resolve()
    trusted_root = (benchmark_root or root).resolve()
    task = root / ".vibesys" / "tasks" / case_id
    task.mkdir(parents=True, exist_ok=False)
    hashes = {name: hashlib.sha256((trusted_root / name).read_bytes()).hexdigest()
              for name in TRUSTED}
    checks = "\n".join(
        f"if hashlib.sha256((ROOT / {name!r}).read_bytes()).hexdigest() != {digest!r}:\n"
        f"    raise RuntimeError('trusted benchmark changed: {name}')"
        for name, digest in hashes.items())
    (task / "evaluator.py").write_text(
        "from pathlib import Path\nimport hashlib\nimport sys\n"
        f"ROOT = Path({str(trusted_root)!r})\n"
        f"{checks}\n"
        "sys.path.insert(0, str(ROOT))\n"
        "from megabench.integrations.vibesys_case_evaluator import main\n"
        # Multi-GPU cases evaluate through torch.multiprocessing.spawn, whose
        # children re-run this file as __mp_main__; only the parent evaluates.
        "if __name__ == '__main__':\n"
        f"    raise SystemExit(main({case_id!r}, sys.argv[1:]))\n")
    python = trusted_root / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)
    launcher = task / "evaluator.sh"
    launcher.write_text(
        "#!/bin/sh\nset -eu\n"
        f"exec {shlex.quote(str(python))} \"$(dirname \"$0\")/evaluator.py\" \"$@\"\n")
    launcher.chmod(0o755)
    command = lambda mode: json.dumps([
        f".vibesys/tasks/{case_id}/evaluator.sh", mode,
        "--submission", "submission.py",
    ])
    (task / "vibesys.input.toml").write_text(
        "version = 1\n\n[agent]\ndomain = \"kernel-writing\"\n\n"
        f"[accuracy]\ncommand = {command('accuracy')}\ntimeout_seconds = 1200\n\n"
        f"[benchmark]\ncommand = {command('benchmark')}\ntimeout_seconds = 1800\n\n"
        "[benchmark.result]\njson_argument = \"--output-json\"\n"
        "metric = \"progress_score\"\n")
    brief = "TASK.md" if (root / "TASK.md").is_file() else "megabench/docs/AGENT_TASK.md"
    reference = ("reference/megabench/tasks" if (root / "reference").is_dir()
                 else "megabench/tasks")
    module = {"dense_step": "dense", "moe_step": "moe",
              "quant_step": "gemma", "spec_target_step": "eagle3",
              "gptoss_step": "gptoss", "hybrid_step": "hybrid",
              "vl_decode_step": "vision", "spec_full_iteration": "speculative",
              "distributed_step": "distributed", "deepseek_v32_step": "frontier",
              "glm52_step": "frontier", "kimi_k3_step": "kimi",
              "glm53_flash_step": "glm53", "kimi_k3_layer": "kimi_layer",
              "tts_frame_step": "csm", "vla_action_step": "pi05",
              "world_frame_step": "waypoint", "megamoe_layer": "megamoe"}[case.family]
    (task / "OBJECTIVE.md").write_text(
        f"# One MegaBench case: {case_id}\n\n"
        f"Implement only `{case_id}`: {case.model}, {case.phase}, geometry "
        f"`{case.params}`. Read `{brief}` and `{reference}/{module}.py`, this case's "
        "PyTorch oracle. Edit only `submission.py` and candidate-owned helper "
        "files. Compute every required output from current inputs in at most "
        "one GPU kernel launch per rank, including communication; preserve all inputs. Do not use prior candidate "
        "code or other checkouts. The evaluator checks only this assigned "
        "case against a protected benchmark outside this workspace.\n")
    return task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-root", type=Path)
    args = parser.parse_args()
    print(make_task(args.case, args.root, args.benchmark_root))


if __name__ == "__main__":
    main()
