"""Microbench gate of a VibeSys workspace.

`items` reads the newest plan (progress/plans/round-*.json) and lists the plan claims that the
microbench phase must answer: each rejected alternative and each sentence that rules a design out,
tagged P1, P2, ... It writes them to progress/microbench/items/<round>.md.

`check` passes when, for the newest plan's round, progress/microbench/requests/<round>.md and
progress/microbench/answers/<round>.md exist, the answers name every item tag, and
progress/engine-map.md mentions the round (its status was reviewed this round), and each question
that progress/microbench/open.md carries open from an earlier round was measured this round (rows
that `./primdb add` stamped with the round) or closed with row ids; one may be deferred. The
carried question that has gone longest without a matched measurement (rows added with
`--question Q<n> --conditions matched`) needs one this round. Without a
plan there is nothing to check. ./check_candidate runs `check` first and stops on a failure.
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
LEDGER = MICROBENCH / "open.md"
LEDGER_HEAD = re.compile(r"##\s+(Q\d+)\b[^\n]*?\b(open|closed|refuted)\b[^\n]*?\bopened\s+(round-\d+)",
                         re.I)


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
    open_questions = [e["tag"] for e in ledger(workspace) if e["status"] == "open"]
    if open_questions:
        items.append("The plan chose its target over the open questions "
                     f"{', '.join(open_questions)} in {LEDGER}: compare the plan's expected effect "
                     "with the largest open estimate, with rows for both")
    out =workspace / MICROBENCH / "items" / f"{plan_path.stem}.md"
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


def ledger(workspace: Path) -> list[dict]:
    """Entries of progress/microbench/open.md: `## Q<n> · <status> · opened <round>` plus body."""
    path = workspace / LEDGER
    if not path.is_file():
        return []
    entries = []
    for block in re.split(r"(?m)^(?=##\s+Q\d+\b)", path.read_text()):
        head = LEDGER_HEAD.match(block)
        if head:
            entries.append({"tag": head.group(1), "status": head.group(2).lower(),
                            "opened": head.group(3), "body": block})
    return entries


def _section(text: str, tag: str) -> str:
    """The part of an answers file under the heading that names `tag`."""
    match = re.search(rf"(?m)^(#+)[^\n]*\b{tag}\b[^\n]*\n", text)
    if not match:
        return ""
    end = re.search(rf"(?m)^#{{1,{len(match.group(1))}}}\s", text[match.end():])
    return text[match.end():match.end() + end.start()] if end else text[match.end():]


def local_sources(workspace: Path) -> list[tuple[str, dict]]:
    """(id, source) of each row that `./primdb add` put in the workspace."""
    path = workspace / MICROBENCH / "primdb_local.jsonl"
    out = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append((row.get("id"), row.get("source", {})))
    return out


def design_of(row: dict) -> str | None:
    """The design a row measures: its engine, else its arm, else its primitive.

    Rows that only load and synchronise (an oracle) set a floor; they are not a design.
    """
    params = row.get("params") or {}
    name = params.get("engine") or params.get("arm") or row.get("primitive")
    if not name or str(name).lower().startswith("oracle"):
        return None
    return str(name)


def matched_comparison(workspace: Path, question: str, round_name: str) -> set[str]:
    """Ids of the matched rows for *question* in *round_name*, if they compare two designs.

    A matched measurement compares designs under the same conditions, so the rows must
    cover at least two designs; matched rows of one design alone are not a comparison.
    """
    path = workspace / MICROBENCH / "primdb_local.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    picked = [r for r in rows if (r.get("source") or {}).get("question") == question
              and (r.get("source") or {}).get("conditions") == "matched"
              and (r.get("source") or {}).get("round") == round_name]
    if len({d for d in map(design_of, picked) if d}) < 2:
        return set()
    return {r.get("id") for r in picked}


SEEDED = "round-0000"  # questions written into the ledger when the workspace was made


def rows_of_round(workspace: Path, round_name: str) -> set[str]:
    """Ids of the rows that `./primdb add` stamped with this round."""
    return {i for i, s in local_sources(workspace) if s.get("round") == round_name}


def stalest_unmatched(workspace: Path, round_name: str) -> dict | None:
    """The carried open question that has gone longest without a matched measurement.

    A matched measurement is a row added with `./primdb add --question Q<n> --conditions matched`,
    or an earlier answers section for the question whose confidence says "under matching".
    Questions with none count from the round they were opened; ties go to the ledger order
    (largest estimate first).
    """
    last = {}
    for _, s in local_sources(workspace):
        if s.get("conditions") == "matched" and s.get("round", "") < round_name:
            tag, rnd = s.get("question"), s.get("round", "")
            if rnd > last.get(tag, "") and matched_comparison(workspace, tag, rnd):
                last[tag] = rnd
    for path in (workspace / MICROBENCH / "answers").glob("round-*.md"):
        if path.stem >= round_name:
            continue
        text = path.read_text()
        for entry in ledger(workspace):
            if re.search(r"(?im)confidence\W[^\n]*\bunder matching\b", _section(text, entry["tag"])):
                last[entry["tag"]] = max(last.get(entry["tag"], ""), path.stem)
    carried = [e for e in ledger(workspace) if e["status"] == "open" and e["opened"] < round_name]
    if not carried:
        return None
    return min(carried, key=lambda e: last.get(e["tag"], e["opened"]))


def check_ledger(workspace: Path, round_name: str, answers: str) -> list[str]:
    """Each question carried open from an earlier round is measured this round.

    The answers file has a section for each carried question that cites at least one row added
    this round; or the ledger closes or refutes the question with row ids. One carried question
    per round may instead be deferred, with `deferred:` and the reason in its section, but not
    the same question in two rounds in a row. The carried question that has gone longest without
    a matched measurement must get one this round, unless it is the deferred one.
    """
    problems = []
    new_rows = rows_of_round(workspace, round_name)
    earlier = sorted(p for p in (workspace / MICROBENCH / "answers").glob("round-*.md")
                     if p.stem < round_name)
    previous = earlier[-1].read_text() if earlier else ""
    deferred = []
    for entry in ledger(workspace):
        if entry["status"] != "open":
            if not ROW_ID.search(entry["body"].split("\n", 1)[-1]):
                problems.append(f"{LEDGER} closes {entry['tag']} without row ids")
            continue
        if entry["opened"] >= round_name:
            continue
        section = _section(answers, entry["tag"])
        if new_rows & set(ROW_ID.findall(section)):
            continue
        if re.search(r"(?im)^\W*deferred:\s*\S", section):
            if entry["opened"] == SEEDED:
                problems.append(f"open question {entry['tag']} was seeded with the workspace "
                                "and cannot be deferred; measure it")
                continue
            if re.search(r"(?im)^\W*deferred:\s*\S", _section(previous, entry["tag"])):
                problems.append(f"open question {entry['tag']} was deferred in the previous "
                                f"round too; measure it in {round_name}")
            deferred.append(entry["tag"])
            continue
        problems.append(
            f"open question {entry['tag']} (opened {entry['opened']}) has no rows measured in "
            f"{round_name}: give it a section in the answers that cites rows added this round "
            "with ./primdb add, or close or refute it in the ledger with row ids")
    stale = stalest_unmatched(workspace, round_name)
    if stale and stale["tag"] not in deferred:
        matched = matched_comparison(workspace, stale["tag"], round_name)
        if not matched & set(ROW_ID.findall(_section(answers, stale["tag"]))):
            problems.append(
                f"open question {stale['tag']} has gone longest without a matched measurement: "
                f"build every design its `settles it` line names under the same conditions "
                f"(matched rows of one design alone are not a comparison; an oracle is not a "
                f"design), add the rows with ./primdb add --question {stale['tag']} "
                "--conditions matched, and cite them in its section (or defer it, once)")
    if len(deferred) > 1:
        problems.append(f"only one carried open question may be deferred per round, not "
                        f"{', '.join(deferred)}")
    return problems


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
        problems += check_ledger(workspace, round_name, text)
    space = workspace / "progress/design-space.md"
    if space.is_file() and f"### {round_name}" not in space.read_text():
        problems.append(f"progress/design-space.md has no reconcile entry for {round_name}")
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
