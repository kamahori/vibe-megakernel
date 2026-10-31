"""Microbench gate of a VibeSys workspace.

`items` reads the newest plan (progress/plans/round-*.json) and lists the plan claims that the
microbench phase must answer: each rejected alternative and each sentence that rules a design out,
tagged P1, P2, ... It writes them to progress/microbench/items/<round>.md.

`check` passes when, for the newest plan's round, progress/microbench/requests/<round>.md and
progress/microbench/answers/<round>.md exist, the answers name every item tag, and
progress/engine-map.md mentions the round (its status was reviewed this round). Without a plan
there is nothing to check. ./check_candidate runs `check` first and stops on a failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MICROBENCH = Path("progress/microbench")
ENGINE_MAP = Path("progress/engine-map.md")
FIELDS = ("reasoning", "hypothesis", "expected_effect")
# A sentence that rules a design out, in the words plans use for it.
RULES_OUT = re.compile(
    r"\b(reject\w*|premature|not worth|gains? nothing|does not pay|doesn't pay|no \w+ reuse|"
    r"ruled? out|too (?:complex|costly|risky|large)|cannot help|will not help|won't help)\b", re.I)
ROW_ID = re.compile(r"\b[0-9a-f]{10}\b")


def newest_plan(workspace: Path) -> Path | None:
    plans = sorted((workspace / "progress/plans").glob("round-*.json"))
    return plans[-1] if plans else None


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=\.)\s+|\n+\s*(?:[-*]|\d+[.)])\s+|\s+-\s+(?=\*{0,2}[A-Z])", text)
    return [p.strip(" -*\n") for p in parts if p.strip(" -*\n")]


def plan_items(plan: dict) -> list[str]:
    """Rejected alternatives and design-excluding sentences, in plan order, without repeats."""
    texts = [str(plan.get(name, "")) for name in FIELDS]
    texts += [str(u.get("reason", "")) for u in plan.get("hypothesis_updates", []) or []]
    items: list[str] = []
    for text in texts:
        in_rejected = False
        for sentence in _sentences(text):
            heading = re.match(r"(?i)\**rejected[^:]*:\**\s*(.*)", sentence)
            if heading:
                in_rejected = True
                sentence = heading.group(1)
                if not sentence:
                    continue
            elif in_rejected and re.match(r"(?i)(no (further )?profil|official evaluation|"
                                          r"keeping|the engine work|the parent|i am requesting|i request)", sentence):
                in_rejected = False
            if (in_rejected or RULES_OUT.search(sentence)) and sentence not in items:
                items.append(sentence)
    return items


def write_items(workspace: Path, plan_path: Path) -> list[str]:
    items = plan_items(json.loads(plan_path.read_text()))
    out = workspace / MICROBENCH / "items" / f"{plan_path.stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Plan claims to answer for {plan_path.stem}", "",
             "Answer each tag in the answers file. A claim that cites no primdb row or measurement "
             "is a guess until the microbench phase measures it.", ""]
    for i, item in enumerate(items, 1):
        cited = " (cites rows)" if ROW_ID.search(item) else ""
        lines.append(f"- P{i}{cited}: {item}")
    if not items:
        lines.append("(no rejected alternatives or design-excluding claims in this plan)")
    out.write_text("\n".join(lines) + "\n")
    return items


def check(workspace: Path) -> list[str]:
    plan = newest_plan(workspace)
    if plan is None:
        return []
    round_name = plan.stem
    problems = []
    request = workspace / MICROBENCH / "requests" / f"{round_name}.md"
    answer = workspace / MICROBENCH / "answers" / f"{round_name}.md"
    for path in (request, answer):
        if not path.is_file() or not path.read_text().strip():
            problems.append(f"{path.relative_to(workspace)} is missing")
    items = write_items(workspace, plan)
    if answer.is_file():
        text = answer.read_text()
        missing = [f"P{i}" for i in range(1, len(items) + 1)
                   if not re.search(rf"\bP{i}\b", text)]
        if missing:
            problems.append(f"{answer.relative_to(workspace)} does not answer plan claims "
                            f"{', '.join(missing)} (see {MICROBENCH}/items/{round_name}.md)")
    engine_map = workspace / ENGINE_MAP
    if not engine_map.is_file() or round_name not in engine_map.read_text():
        problems.append(f"{ENGINE_MAP} has no status entry for {round_name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microbench_gate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("items", "check"))
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    args = p.parse_args(argv)
    if args.command == "items":
        plan = newest_plan(args.workspace)
        if plan is None:
            print("no plan yet (progress/plans/round-*.json)")
            return 0
        write_items(args.workspace, plan)
        print((args.workspace / MICROBENCH / "items" / f"{plan.stem}.md").read_text(), end="")
        return 0
    problems = check(args.workspace)
    for problem in problems:
        print(f"check_candidate: {problem}. Do the microbench phase of TASK.md first.",
              file=sys.stderr)
    return 3 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
