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
plan there is nothing to check. ./check_candidate runs `check --build` first and stops on a failure:
--build also needs progress/microbench/build/<round>.md, the design each phase is built in; each
engine-map phase with matched rows (p.phase, m.us) is built in the family of its fastest row.
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


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def engine_families(workspace: Path) -> list[str]:
    """The engine names in the tables of `progress/engine-map.md`."""
    path = workspace / "progress" / "engine-map.md"
    if not path.is_file():
        return []
    names = {m.group(1).strip() for m in re.finditer(r"(?m)^\|\s*([^|]+?)\s*\|", path.read_text())}
    return sorted(n for n in names if _norm(n) and _norm(n) != "engine" and not set(n) <= set("-: "))


def family_of(design: str, families: list[str]) -> str | None:
    """The engine-map family whose name begins the design name, the longest match first."""
    d = _norm(design)
    for f in sorted(families, key=lambda f: -len(_norm(f))):
        if d.startswith(_norm(f)):
            return f
    return None


def matched_comparison(workspace: Path, question: str, round_name: str) -> set[str]:
    """Ids of the matched rows for *question* in *round_name*, if they compare two designs.

    A matched measurement compares designs under the same conditions, so the rows must
    cover at least two designs; matched rows of one design alone are not a comparison.
    The question seeded from the engine map compares engines, so its rows must cover every
    engine family that the engine map lists, as its `settles it` line says; variants of one
    engine are not an engine comparison.
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
    designs = {d for d in map(design_of, picked) if d}
    if len(designs) < 2:
        return set()
    seeded = any(e["tag"] == question and e["opened"] == SEEDED for e in ledger(workspace))
    families = engine_families(workspace)
    if seeded and families and not set(families) <= {family_of(d, families) for d in designs}:
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
                f"design; for a question seeded from the engine map, every engine family in its "
                f"tables, not variants of one family), add the rows with ./primdb add --question {stale['tag']} "
                "--conditions matched, and cite them in its section (or defer it, once)")
    if len(deferred) > 1:
        problems.append(f"only one carried open question may be deferred per round, not "
                        f"{', '.join(deferred)}")
    return problems


BUILD = MICROBENCH / "build"
BUILD_HELP = ("one table row per phase of the candidate: | phase | design built | notes |, where "
              "the phase cell names the engine-map tags (E1, E2, ...) that the phase computes")
PHASE_TAG = re.compile(r"\bE\d+\b")


def phase_ranking(workspace: Path) -> dict[str, tuple[str, float, str]]:
    """For each engine-map phase tag, the fastest matched row: (design, us, id).

    Rows count when `./primdb add` stored them with `--conditions matched`, a design
    (not an oracle), `p.phase` and the phase time `m.us`.
    """
    path = workspace / MICROBENCH / "primdb_local.jsonl"
    best: dict[str, tuple[str, float, str]] = {}
    if not path.is_file():
        return best
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get("source") or {}).get("conditions") != "matched":
            continue
        design = design_of(row)
        phase = (row.get("params") or {}).get("phase")
        us = (row.get("metrics") or {}).get("us")
        if not design or not phase or not isinstance(us, (int, float)) or us <= 0:
            continue
        for tag in PHASE_TAG.findall(str(phase)):
            if tag not in best or us < best[tag][1]:
                best[tag] = (design, float(us), row.get("id", "?"))
    return best


def check_build(workspace: Path, round_name: str) -> list[str]:
    """The build record of the round: the design each phase of the candidate is built in.

    Each engine-map phase with matched rows is built in the engine family of its fastest
    matched row. A design that the implementer prefers becomes the fastest by its own matched
    rows, not by a claim.
    """
    path = workspace / BUILD / f"{round_name}.md"
    rel = path.relative_to(workspace)
    if not path.is_file():
        return [f"{rel} is missing: write {BUILD_HELP}"]
    families = engine_families(workspace)
    ranking = phase_ranking(workspace)
    built: dict[str, str] = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.lstrip().startswith("|") or len(cells) < 2:
            continue
        if _norm(cells[0]) in ("phase", "") or set(cells[0]) <= set("-: "):
            continue
        for tag in PHASE_TAG.findall(cells[0]):
            built[tag] = cells[1]
    problems = []

    def family(name: str) -> str:
        return (family_of(name, families) if families else None) or _norm(name)

    for tag, (design, us, row_id) in sorted(ranking.items()):
        if tag not in built:
            problems.append(f"{rel} has no row for phase {tag}, which has matched rows: name the "
                            "design the candidate computes it in")
        elif family(built[tag]) != family(design):
            problems.append(
                f"{rel}: phase {tag} is built in {built[tag]}, but its fastest matched row is "
                f"{row_id} ({design}, {us:g} us). Build {tag} in that design's engine family (the "
                "verified proxies under progress/microbench/bench/ are a start). If you expect "
                "your design to be faster, measure it under matched conditions with p.phase="
                f"{tag}; its rows then decide")
    if not built:
        problems.append(f"{rel} has no phase rows: write {BUILD_HELP}")
    return problems


def check(workspace: Path, build: bool = False) -> list[str]:
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
    if build and not problems:
        problems += check_build(workspace, round_name)
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microbench_gate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("items", "check"))
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    p.add_argument("--build", action="store_true",
                   help="also check the round's build record (./check_candidate)")
    args = p.parse_args(argv)
    if args.command == "items":
        plan = newest_plan(args.workspace)
        if plan is None:
            print("no plan yet (progress/plans/round-*.json)")
            return 0
        write_items(args.workspace, plan)
        print((args.workspace / MICROBENCH / "items" / f"{plan.stem}.md").read_text(), end="")
        return 0
    problems = check(args.workspace, build=args.build)
    for problem in problems:
        step = ("Build each phase as TASK.md step 4 says" if "/build/" in problem
                else "Do the microbench phase of TASK.md first")
        print(f"check_candidate: {problem}. {step}.", file=sys.stderr)
    return 3 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
