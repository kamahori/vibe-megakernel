"""Microbench gate of a VibeSys workspace.

`items` reads the newest plan (progress/plans/round-*.json) and lists the plan claims that the
microbench phase must answer: each row of the plan's Designs table (the design that the plan
selects for a phase, which rejects the other designs of that phase), each rejected alternative
and each sentence that rules a design out, tagged P1, P2, ... It writes them to
progress/microbench/items/<round>.md. The answer to a design item cites a tuned matched row of
its phase or an open ledger question; a plan whose task has no Designs table gets an item that
asks for one.

`check` passes when, for the newest plan's round, progress/microbench/requests/<round>.md and
progress/microbench/answers/<round>.md exist, the answers name every item tag, and
progress/engine-map.md mentions the round (its status was reviewed this round), and each question
that progress/microbench/open.md carries open from an earlier round was measured this round (rows
that `./primdb add` stamped with the round) or closed with row ids; one may be deferred. The
carried question that has gone longest without a matched measurement (rows added with
`--question Q<n> --conditions matched`) needs one this round. Without a
plan there is nothing to check. A round whose phase the framework skipped
(progress/microbench/skipped/<round>.md: no question of the plan needed a new microbenchmark)
needs none of these files, and neither does a round whose phase the framework stopped at its
hard time limit (progress/microbench/deadline/<round>.md) or that ended with an error before
it wrote its files (progress/microbench/stopped/<round>.md). A claim that only the revised plan
makes (not its draft, progress/microbench/draft/<round>.json) and that cites no row goes to the
next round's phase (progress/microbench/carried/<round>.md). A plan claim that cites a row id
that the round's answers also cite counts as answered. ./check_candidate runs `check --build` first and stops on a failure:
--build also needs progress/microbench/build/<round>.md, the design each phase is built in; each
engine-map phase with matched rows (p.phase, m.us) is built in the family of its fastest row, and in
that family's fastest decomposition (p.decomp) when the rows measure two or more. A design is
its engine and its decomposition; the noise of a phase is the spread of its true repeats.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

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


DESIGNS_HELP = "| phase | design | decomposition | ingress | rows |"
DESIGN_ITEM = re.compile(r"^Design of (.*?): ")
NO_DESIGNS = ("The plan has no Designs table (" + DESIGNS_HELP + ") in its task: for each phase "
              "that the task writes, name the design that the task selects (engine, "
              "decomposition, ingress), and the tuned matched rows of that phase that rank it "
              "fastest or the open ledger question that prices the other designs")


def _cells(line: str) -> list[str] | None:
    """The cells of a markdown table row, or None when *line* is not one."""
    line = line.strip()
    if not line.startswith("|"):
        return None
    return [c.strip() for c in line.strip("|").split("|")]


def plan_designs(plan: dict) -> list[dict] | None:
    """The rows of the plan's Designs table: the design that the plan selects for each phase.

    The table is in the plan's `task` (the text the implementer receives), else in its
    `reasoning`: a header row whose cells include `phase` and `design`, then one row per
    phase. Each row gives {"phase", "design", "rest"}. None when the plan has no such table.
    """
    for field in ("task", "reasoning"):
        lines = str(plan.get(field, "")).splitlines()
        for i, line in enumerate(lines):
            head = _cells(line)
            if head is None:
                continue
            names = [_norm(c) for c in head]
            if "phase" not in names or not any(n.startswith("design") for n in names):
                continue
            p, d = names.index("phase"), next(k for k, n in enumerate(names)
                                             if n.startswith("design"))
            rows = []
            for row in lines[i + 1:]:
                cells = _cells(row)
                if cells is None:
                    break
                if all(set(c) <= set("-: ") for c in cells):
                    continue
                cells += [""] * (len(head) - len(cells))
                rest = [c for k, c in enumerate(cells) if k not in (p, d) and c]
                if cells[p] or cells[d]:
                    rows.append({"phase": cells[p], "design": cells[d], "rest": rest})
            return rows
    return None


def design_items(plan: dict) -> list[str]:
    """One claim for each row of the Designs table: a selection rejects the other designs."""
    designs = plan_designs(plan)
    if designs is None:
        return [NO_DESIGNS] if str(plan.get("task", "")).strip() else []
    items = []
    for row in designs:
        detail = f" ({'; '.join(row['rest'])})" if row["rest"] else ""
        items.append(f"Design of {row['phase']}: {row['design']}{detail}. The plan selects it "
                     "over the other designs of this phase")
    return items


def plan_items(plan: dict) -> list[str]:
    """Design selections, then rejected alternatives and design-excluding sentences, in plan
    order, without repeats."""
    texts = [str(plan.get(name, "")) for name in FIELDS]
    texts += [str(u.get("reason", "")) for u in plan.get("hypothesis_updates", []) or []]
    items: list[str] = [i for i in dict.fromkeys(design_items(plan))]
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


DRAFT = MICROBENCH / "draft"
CARRIED = MICROBENCH / "carried"


def revised_only(workspace: Path, plan_path: Path, items: list[str]) -> list[str]:
    """Claims of a revised plan that its draft did not make and that cite no row.

    The framework copies the draft to progress/microbench/draft/<round>.json before the
    revision. The microbench session of the round has closed by then, so such a claim goes to
    the next round's phase (progress/microbench/carried/<round>.md) and does not block this
    round.
    """
    draft = workspace / DRAFT / f"{plan_path.stem}.json"
    if not draft.is_file():
        return []
    try:
        before = set(plan_items(json.loads(draft.read_text())))
    except ValueError:
        return []
    return [item for item in items if item not in before and not ROW_ID.search(item)]


def carried_in(workspace: Path, round_name: str) -> list[str]:
    """Claims that the newest earlier round carried to this round's phase."""
    earlier = sorted(p for p in (workspace / CARRIED).glob("round-*.md") if p.stem < round_name)
    if not earlier:
        return []
    return [line[2:].strip() for line in earlier[-1].read_text().splitlines()
            if line.startswith("- ")]


def write_items(workspace: Path, plan_path: Path) -> list[str]:
    items = plan_items(json.loads(plan_path.read_text()))
    later = revised_only(workspace, plan_path, items)
    items = [item for item in items if item not in later]
    carried_file = workspace / CARRIED / f"{plan_path.stem}.md"
    if later:
        carried_file.parent.mkdir(parents=True, exist_ok=True)
        carried_file.write_text(
            f"# Claims of the revised plan of {plan_path.stem}, for the next microbench phase\n\n"
            + "".join(f"- {item}\n" for item in later))
    else:
        carried_file.unlink(missing_ok=True)
    items += [f"(carried from the previous revised plan) {item}"
              for item in carried_in(workspace, plan_path.stem) if item not in items]
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


DECOMP_KEYS = ("decomp", "partition")  # the param that names a row's decomposition, in order


def _design_name(row: dict) -> str | None:
    params = row.get("params") or {}
    name = params.get("engine") or params.get("arm") or row.get("primitive")
    return str(name) if name else None


def is_floor(row: dict) -> bool:
    """A row that only loads and synchronises (an oracle): it sets a floor, not a design."""
    name = _design_name(row)
    return bool(name) and name.lower().startswith("oracle")


def design_of(row: dict) -> str | None:
    """The design a row measures: its engine (else its arm, else its primitive), then
    `/<decomposition>` when `p.decomp` (else `p.partition`) names one.

    Two decompositions of one engine are two designs: at archC16 E2, mma.sync with a full-K
    output tile took 24.09 us and with warp split-K 6.67 us. Rows that only load and
    synchronise (an oracle) set a floor; they are not a design.
    """
    name = _design_name(row)
    if not name or is_floor(row):
        return None
    params = row.get("params") or {}
    decomp = next((params[k] for k in DECOMP_KEYS if params.get(k)), None)
    return f"{name}/{decomp}" if decomp else name


def engine_of(design: str) -> str:
    """The engine part of a design name (design_of): the text before `/`."""
    return design.split("/", 1)[0]


def decomp_of(design: str) -> str | None:
    """The decomposition part of a design name (design_of), if it has one."""
    return design.split("/", 1)[1] if "/" in design else None


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
    d = _norm(engine_of(design))
    for f in sorted(families, key=lambda f: -len(_norm(f))):
        if d.startswith(_norm(f)):
            return f
    return None


def tuned_rows(rows: list[dict]) -> list[dict]:
    """The rows among *rows* that may decide an engine comparison or a build.

    A row counts when its params say `tuned: true` and its design (engine and decomposition,
    design_of) has matched rows of at least two setups for the same question or phase (the
    variants of a sweep: another stage depth, ingress, tile or split size). Repeats of one
    setup are not a sweep. An untuned arm is an upper bound on its engine: it can show that the
    engine is fast, never that it is slow.
    """
    def key(r: dict) -> tuple:
        src, par = r.get("source") or {}, r.get("params") or {}
        return design_of(r), src.get("question") or par.get("phase")
    setups: dict[tuple, set[str]] = {}
    for r in rows:
        if (r.get("source") or {}).get("conditions") == "matched" and design_of(r):
            setups.setdefault(key(r), set()).add(setup_of(r))
    return [r for r in rows if (r.get("params") or {}).get("tuned") is True
            and design_of(r) and len(setups.get(key(r), ())) >= 2]


def matched_comparison(workspace: Path, question: str, round_name: str) -> set[str]:
    """Ids of the matched rows for *question* in *round_name*, if they compare two designs.

    A matched measurement compares tuned designs under the same conditions (see tuned_rows),
    so the tuned rows must cover at least two designs; matched rows of one design alone are not a comparison.
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
    # Only tuned designs compare: an untuned arm that loses does not show its engine loses.
    designs = {d for d in map(design_of, tuned_rows(picked)) if d}
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


def local_rows(workspace: Path) -> dict[str, dict]:
    """The rows that `./primdb add` put in the workspace, by id."""
    path = workspace / MICROBENCH / "primdb_local.jsonl"
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                out[row["id"]] = row
    return out


def round_number(round_name: str) -> int:
    match = re.search(r"(\d+)$", round_name)
    return int(match.group(1)) if match else 0


def earlier_designs(workspace: Path, tag: str, round_name: str) -> set[str]:
    """The designs measured for question *tag* before *round_name*.

    A row counts when `./primdb add --question` stamped it with the question, or when an
    earlier answers section for the question cites it.
    """
    rows = local_rows(workspace)
    ids = {i for i, r in rows.items() if (r.get("source") or {}).get("question") == tag
           and (r.get("source") or {}).get("round", "") < round_name}
    for path in (workspace / MICROBENCH / "answers").glob("round-*.md"):
        if path.stem < round_name:
            ids |= set(ROW_ID.findall(_section(path.read_text(), tag)))
    return {d for d in (design_of(rows[i]) for i in ids if i in rows) if d}


def last_matched(workspace: Path, round_name: str) -> dict[str, str]:
    """For each question, the last round before *round_name* with a matched comparison."""
    last: dict[str, str] = {}
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
    return last


OVERDUE_ROUNDS = 2  # rounds a carried question may go without a matched comparison


def stalest_unmatched(workspace: Path, round_name: str) -> dict | None:
    """The carried open question that has gone longest without a matched measurement.

    A matched measurement is a row added with `./primdb add --question Q<n> --conditions matched`,
    or an earlier answers section for the question whose confidence says "under matching".
    Questions with none count from the round they were opened; ties go to the ledger order
    (largest estimate first).
    """
    last = last_matched(workspace, round_name)
    carried = [e for e in ledger(workspace) if e["status"] == "open" and e["opened"] < round_name]
    if not carried:
        return None
    return min(carried, key=lambda e: last.get(e["tag"], e["opened"]))


def due_overdue(workspace: Path, round_name: str) -> dict | None:
    """The overdue question of the round: the first open question in ledger order (largest
    estimate first) that has gone OVERDUE_ROUNDS rounds without a matched comparison.

    One question per round, so that a ledger with many open questions cannot ask for more
    matched comparisons than one microbench phase can build.
    """
    last = last_matched(workspace, round_name)
    for entry in ledger(workspace):
        if entry["status"] != "open" or entry["opened"] >= round_name:
            continue
        since = last.get(entry["tag"], entry["opened"])
        if round_number(round_name) - round_number(since) >= OVERDUE_ROUNDS:
            return dict(entry, since=since)
    return None


def overdue_problem(entry: dict, round_name: str) -> str:
    return (f"open question {entry['tag']} has gone {OVERDUE_ROUNDS} or more rounds without a "
            f"matched comparison (last: {entry['since']}) and has the largest estimate of the "
            f"overdue questions: build the designs its `settles it` line names under the same "
            f"conditions, tuned, add the rows with ./primdb add --question {entry['tag']} "
            f"--conditions matched and cite them in its section of "
            f"{MICROBENCH}/answers/{round_name}.md. It cannot be deferred")


def overdue_questions(workspace: Path, round_name: str, answers: str = "") -> list[str]:
    """The problem for the round's overdue question, if *answers* cite no matched comparison."""
    due = due_overdue(workspace, round_name)
    if due is None:
        return []
    cited = rows_of_round(workspace, round_name) & set(ROW_ID.findall(_section(answers, due["tag"])))
    if matched_comparison(workspace, due["tag"], round_name) & cited:
        return []
    return [overdue_problem(due, round_name)]


def check_ledger(workspace: Path, round_name: str, answers: str) -> list[str]:
    """Each question carried open from an earlier round is measured this round.

    The answers file has a section for each carried question that cites at least one row added
    this round; or the ledger closes or refutes the question with row ids. One carried question
    per round may instead be deferred, with `deferred:` and the reason in its section, but not
    the same question in two rounds in a row. The carried question that has gone longest without
    a matched measurement must get one this round, unless it is the deferred one.

    Rows that only repeat designs already measured for the question do not count: the cited
    rows must form a matched comparison or add a design not measured for it before. A question
    that has gone OVERDUE_ROUNDS rounds without a matched comparison blocks until it gets one.
    """
    problems = []
    new_rows = rows_of_round(workspace, round_name)
    earlier = sorted(p for p in (workspace / MICROBENCH / "answers").glob("round-*.md")
                     if p.stem < round_name)
    previous = earlier[-1].read_text() if earlier else ""
    deferred, overdue = [], []
    rows = local_rows(workspace)
    last = last_matched(workspace, round_name)
    due = due_overdue(workspace, round_name)
    for entry in ledger(workspace):
        if entry["status"] != "open":
            if not ROW_ID.search(entry["body"].split("\n", 1)[-1]):
                problems.append(f"{LEDGER} closes {entry['tag']} without row ids")
            continue
        if entry["opened"] >= round_name:
            continue
        section = _section(answers, entry["tag"])
        cited = new_rows & set(ROW_ID.findall(section))
        matched = matched_comparison(workspace, entry["tag"], round_name) & cited
        since = last.get(entry["tag"], entry["opened"])
        if due and entry["tag"] == due["tag"] and not matched:
            overdue.append(entry["tag"])
            problems.append(overdue_problem(due, round_name))
            continue
        if cited:
            known = earlier_designs(workspace, entry["tag"], round_name)
            fresh = {design_of(rows[i]) for i in cited if i in rows} - known - {None}
            if matched or fresh:
                continue
            problems.append(
                f"open question {entry['tag']}: the rows cited this round ({', '.join(sorted(cited))}) "
                f"measure only designs measured for it before ({', '.join(sorted(known)) or '-'}). "
                "A run of the same benches again does not settle it: measure the comparison its "
                "`settles it` line names under matched conditions, or a design or arm not "
                "measured for it yet")
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
    if stale and stale["tag"] not in deferred + overdue:
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
              "the phase cell names the engine-map tags (E1, E2, ...) that the phase computes and "
              "the design cell names the engine, then the decomposition in the words of its "
              "rows' p.decomp (for example `tcgen05 swap-AB, stream_k_redadd`)")
PHASE_TAG = re.compile(r"\bE\d+\b")


RUN_KEYS = ("tuned", "rep", "repeat", "seed", "run", "trial", "note", "notes")
# Params that name a run of a setup, not the setup: rows equal in all other params are repeats.


class PhaseRow(NamedTuple):
    tag: str      # engine-map phase tag (E<n>)
    design: str   # design_of the row
    us: float     # phase time, m.us
    id: str
    setup: str    # the row's params without RUN_KEYS: equal setups are repeats


def setup_of(row: dict) -> str:
    """The setup a row measures: its family, primitive and params without RUN_KEYS."""
    params = {k: v for k, v in (row.get("params") or {}).items() if k not in RUN_KEYS}
    return json.dumps([row.get("family"), row.get("primitive"), params], sort_keys=True,
                      default=str)


def matched_phase_rows(workspace: Path) -> list[PhaseRow]:
    """(phase tag, design, us, id, setup) of each matched row with `p.phase` and `m.us`.

    Rows count when `./primdb add` stored them with `--conditions matched`, a design
    (not an oracle), `p.phase` and the phase time `m.us`, and they are tuned (tuned_rows).
    """
    out = []
    rows = [r for r in local_rows(workspace).values()
            if (r.get("source") or {}).get("conditions") == "matched"]
    for row in tuned_rows(rows):
        row_id = row.get("id", "?")
        design = design_of(row)
        phase = (row.get("params") or {}).get("phase")
        us = (row.get("metrics") or {}).get("us")
        if not design or not phase or not isinstance(us, (int, float)) or us <= 0:
            continue
        for tag in PHASE_TAG.findall(str(phase)):
            out.append(PhaseRow(tag, design, float(us), row_id, setup_of(row)))
    return out


def phase_ranking(workspace: Path) -> dict[str, tuple[str, float, str]]:
    """For each engine-map phase tag, the fastest matched row: (design, us, id)."""
    best: dict[str, tuple[str, float, str]] = {}
    for r in matched_phase_rows(workspace):
        if r.tag not in best or r.us < best[r.tag][1]:
            best[r.tag] = (r.design, r.us, r.id)
    return best


NOISE = 0.02  # relative gap treated as noise when no setup of the phase was measured twice


def phase_noise(rows: list[PhaseRow], tag: str) -> float:
    """The relative spread of repeated matched rows of one setup for phase *tag*.

    Repeats (rows equal in every param but RUN_KEYS) show how far two times of one setup
    differ; a gap between designs inside that spread is not a ranking. Rows of one design
    that differ in a knob (stage depth, ingress, tile, split) are a sweep, not repeats:
    their spread is the effect of the knob. With no repeats, NOISE.
    """
    by_setup: dict[str, list[float]] = {}
    for r in rows:
        if r.tag == tag:
            by_setup.setdefault(r.setup, []).append(r.us)
    spreads = [(max(v) - min(v)) / min(v) for v in by_setup.values() if len(v) > 1]
    return max(spreads + [NOISE])


DEFERRED = re.compile(r"(?i)\bdeferred:\s*(.*)")


def build_record(path: Path) -> dict[str, tuple[str, str]]:
    """Phase tag -> (design built, notes) from a build record."""
    built: dict[str, tuple[str, str]] = {}
    if not path.is_file():
        return built
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.lstrip().startswith("|") or len(cells) < 2:
            continue
        if _norm(cells[0]) in ("phase", "") or set(cells[0]) <= set("-: "):
            continue
        for tag in PHASE_TAG.findall(cells[0]):
            built[tag] = (cells[1], " | ".join(cells[2:]))
    return built


def check_build(workspace: Path, round_name: str) -> list[str]:
    """The build record of the round: the design each phase of the candidate is built in.

    Each engine-map phase with matched rows is built in the engine family of its fastest
    matched row, or in a family whose own fastest matched row is within the noise of it.
    Inside the family it is built in, when the rows measure two or more decompositions
    (design_of), the design cell names the decomposition (the words of its `p.decomp` or
    `p.partition` value) whose fastest row is within the noise of the family's fastest row:
    the engine alone does not make a design (archC16 E2: mma.sync full-K 24.09 us, split-K
    6.67 us). Otherwise the notes cell of the phase defers the rebuild with `deferred:`, the
    reason and the row ids it rests on. A phase deferred in the previous round's record
    cannot be deferred again. A design that the implementer prefers becomes the fastest by its
    own matched rows, not by a claim.
    """
    ranking = phase_ranking(workspace)
    path = workspace / BUILD / f"{round_name}.md"
    rel = path.relative_to(workspace)
    if not ranking:
        return []
    if not path.is_file():
        return [f"{rel} is missing: write {BUILD_HELP}"]
    families = engine_families(workspace)
    rows = matched_phase_rows(workspace)
    built = build_record(path)
    earlier = sorted(p for p in (workspace / BUILD).glob("round-*.md") if p.stem < round_name)
    previous = build_record(earlier[-1]) if earlier else {}
    problems = []

    def family(name: str) -> str:
        return (family_of(name, families) if families else None) or _norm(engine_of(name))

    def family_problem(tag: str, name: str, design: str, us: float, row_id: str,
                       noise: float) -> str | None:
        own = [r.us for r in rows if r.tag == tag and family(r.design) == family(name)]
        if family(name) == family(design) or (own and (min(own) - us) / us <= noise):
            return None
        gap = (f"its own fastest matched row is {min(own):g} us, {100 * (min(own) - us) / us:.1f}% "
               f"slower (noise {100 * noise:.1f}%)" if own else "it has no matched row for it")
        return (f"{rel}: phase {tag} is built in {name}, but its fastest matched row is "
                f"{row_id} ({design}, {us:g} us) and {gap}. Build {tag} in that design's engine "
                "family (the verified proxies under progress/microbench/bench/ are a start), or "
                "measure your design under matched conditions with p.phase="
                f"{tag} (its rows then decide), or write `deferred: <reason> <row ids>` in the "
                "notes cell of the phase (one round only)")

    def decomp_problem(tag: str, name: str, noise: float) -> str | None:
        best: dict[str, PhaseRow] = {}
        for r in rows:
            k = decomp_of(r.design)
            if r.tag == tag and k and family(r.design) == family(name):
                if k not in best or r.us < best[k].us:
                    best[k] = r
        if len(best) < 2:
            return None
        top = min(best.values(), key=lambda r: r.us)
        named = decomp_named(name, list(best))
        slugs = ", ".join(f"{k} ({best[k].us:g} us, {best[k].id})"
                          for k in sorted(best, key=lambda k: best[k].us))
        if named is None:
            return (f"{rel}: phase {tag} is built in {name}, which does not name its "
                    f"decomposition; the matched rows of its engine family measure {slugs}. Name "
                    f"the decomposition built (the words of its p.decomp) in the design cell")
        gap = (best[named].us - top.us) / top.us
        if gap <= noise:
            return None
        return (f"{rel}: phase {tag} is built in {name} ({named}, fastest matched row "
                f"{best[named].id}, {best[named].us:g} us), but its engine family's fastest "
                f"decomposition is {decomp_of(top.design)} ({top.id}, {top.us:g} us), "
                f"{100 * gap:.1f}% faster (noise {100 * noise:.1f}%). Build {tag} in that "
                "decomposition, or measure yours tuned under matched conditions with p.phase="
                f"{tag} and p.decomp (its rows then decide), or write `deferred: <reason> "
                "<row ids>` in the notes cell of the phase (one round only)")

    for tag, (design, us, row_id) in sorted(ranking.items()):
        ranked = len({family(r.design) for r in rows if r.tag == tag}) >= 2
        if tag not in built:
            if ranked:
                problems.append(f"{rel} has no row for phase {tag}, which has matched rows: name "
                                "the design the candidate computes it in")
            continue
        name, notes = built[tag]
        noise = phase_noise(rows, tag)
        # One engine family tuned at this phase: no family to rank it against yet, but the
        # decompositions of that family still rank.
        problem = (family_problem(tag, name, design, us, row_id, noise) if ranked else None) \
            or decomp_problem(tag, name, noise)
        if problem is None:
            continue
        deferral = DEFERRED.search(notes)
        if deferral and ROW_ID.search(deferral.group(1)) and len(ROW_ID.sub("", deferral.group(1)).strip(" ,;.:-")) > 0:
            if tag in previous and DEFERRED.search(previous[tag][1]):
                problems.append(
                    f"{rel}: phase {tag} was deferred in {earlier[-1].stem} too. Build it in the "
                    f"design of {row_id} ({design}, {us:g} us) this round, or measure your design "
                    f"under matched conditions with p.phase={tag}; its rows then decide")
            continue
        problems.append(problem)
    if not built:
        problems.append(f"{rel} has no phase rows: write {BUILD_HELP}")
    return problems


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def decomp_named(cell: str, decomps: list[str]) -> str | None:
    """The decomposition among *decomps* that a build-record cell names: every word of it
    appears in the cell (split_k_warp_redadd: "warp split-K, red.add"). With two, the one
    with the most words."""
    words = _norm(cell)
    hits = [k for k in decomps if _words(k) and all(w in words for w in _words(k))]
    return max(hits, key=lambda k: (len(_words(k)), len(k))) if hits else None


def _item_section(text: str, n: int) -> str:
    """The part of an answers file that answers plan item P<n>.

    A heading that names the tag, else the text from the first mention of the tag to the
    next mention of another P tag or the next heading.
    """
    section = _section(text, f"P{n}")
    if section:
        return section
    match = re.search(rf"\bP{n}\b", text)
    if not match:
        return ""
    end = re.search(rf"\bP(?!{n}\b)\d+\b|^#", text[match.end():], re.M)
    return text[match.start():match.end() + end.start()] if end else text[match.start():]


def design_problems(workspace: Path, items: list[str], answers: str) -> list[str]:
    """Each design item answered by its own section cites evidence that can rank the phase.

    The evidence is a tuned matched row of the item's phase (p.phase carries one of its
    engine-map tags; any tuned matched row when the phase cell names no tag), or an open
    ledger question. An item answered only through row ids that the plan and the answers both
    cite is not checked here: the answers to the draft were.
    """
    rows = [r for r in local_rows(workspace).values()
            if (r.get("source") or {}).get("conditions") == "matched"]
    tuned = tuned_rows(rows)
    open_tags = {e["tag"] for e in ledger(workspace) if e["status"] == "open"}
    problems = []
    for n, item in enumerate(items, 1):
        head = DESIGN_ITEM.match(item)
        if not head:
            continue
        section = _item_section(answers, n)
        if not section:
            continue
        tags = set(PHASE_TAG.findall(head.group(1)))
        ranked = {r.get("id") for r in tuned
                  if not tags or tags & set(PHASE_TAG.findall(str((r.get("params") or {})
                                                                  .get("phase", ""))))}
        if ranked & set(ROW_ID.findall(section)):
            continue
        if open_tags & set(re.findall(r"\bQ\d+\b", section)):
            continue
        phase = ", ".join(sorted(tags)) or head.group(1)
        problems.append(
            f"{MICROBENCH}/answers answers design item P{n} ({phase}) without a tuned matched "
            f"row of that phase or an open ledger question: cite the tuned matched rows "
            f"(p.phase={phase}, tuned, two or more variants) that rank the designs of the phase, "
            f"or open a question in {LEDGER} with a range for each other design and cite its tag")
    return problems


def check(workspace: Path, build: bool = False) -> list[str]:
    plan = newest_plan(workspace)
    if plan is None:
        return []
    round_name = plan.stem
    problems = []
    skipped = workspace / MICROBENCH / "skipped" / f"{round_name}.md"
    if skipped.is_file() and skipped.read_text().strip():
        # The framework skipped this round's phase: no question of the plan needed a new
        # microbenchmark. The design of each phase still follows the measured rankings, and
        # a question overdue for a matched comparison still blocks: run the microbench
        # subagent on it (steps 1-3) and cite its rows in the round's answers.
        answer = workspace / MICROBENCH / "answers" / f"{round_name}.md"
        problems = overdue_questions(workspace, round_name,
                                     answer.read_text() if answer.is_file() else "")
        if build and not problems:
            problems += check_build(workspace, round_name)
        return problems
    # The framework stopped this round's phase at its hard time limit: what was not answered
    # stays open, and the round goes on with the answers that exist.
    # The same holds when the phase ended with an error before it wrote its files
    # (progress/microbench/stopped/<round>.md).
    cut = any((workspace / MICROBENCH / marker / f"{round_name}.md").is_file()
              for marker in ("deadline", "stopped"))
    request = workspace / MICROBENCH / "requests" / f"{round_name}.md"
    answer = workspace / MICROBENCH / "answers" / f"{round_name}.md"
    for path in (request, answer):
        if not path.is_file() or not path.read_text().strip():
            problems.append(f"{path.relative_to(workspace)} is missing")
    items = write_items(workspace, plan)
    if cut:
        return check_build(workspace, round_name) if build else []
    if answer.is_file():
        text = answer.read_text()
        # A claim of a revised plan that cites rows which this round's answers cite is
        # answered by them: the orchestrator revised the plan from those answers.
        cited = set(ROW_ID.findall(text))
        missing = [f"P{i}" for i, item in enumerate(items, 1)
                   if not re.search(rf"\bP{i}\b", text)
                   and not (set(ROW_ID.findall(item)) & cited)]
        if missing:
            problems.append(f"{answer.relative_to(workspace)} does not answer plan claims "
                            f"{', '.join(missing)} (see {MICROBENCH}/items/{round_name}.md)")
        problems += design_problems(workspace, items, text)
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
