"""Create a small, single-case workspace for a kernel-writing agent."""

from __future__ import annotations

import argparse
import json
import re
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from ..cases import Case, load_regimes, select_cases
from ..sol import work_model
from .make_vibesys_case_task import make_task


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILES = {
    "dense_step": ("dense.py", "common.py", "references/qwen3.py"),
    "moe_step": ("moe.py", "common.py", "references/qwen3.py",
                 "references/moe.py"),
    "quant_step": ("gemma.py", "common.py"),
    "spec_target_step": ("eagle3.py", "common.py"),
    "gptoss_step": ("gptoss.py", "quantization.py", "common.py"),
    "hybrid_step": ("hybrid.py", "common.py"),
    "vl_decode_step": ("vision.py", "gemma.py", "common.py"),
    "spec_full_iteration": ("speculative.py", "eagle3.py", "common.py"),
    "distributed_step": ("distributed.py", "parallel.py", "common.py"),
    "deepseek_v32_step": ("frontier.py", "deepseek_checkpoint.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "glm52_step": ("frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "kimi_k3_step": ("kimi.py", "frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "kimi_k3_layer": ("kimi_layer.py", "kimi.py", "frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "glm53_flash_step": ("glm53.py", "frontier.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
    "tts_frame_step": ("csm.py", "common.py"),
    "vla_action_step": ("pi05.py", "vision.py", "gemma.py", "common.py"),
    "world_frame_step": ("waypoint.py", "common.py"),
    "megamoe_layer": ("megamoe.py", "distributed.py", "parallel.py", "quantization.py", "common.py"),
}
OUTPUT_KEYS = {
    "dense_step": ("logits", "next_token", "k_write", "v_write"),
    "moe_step": ("logits", "next_token", "k_write", "v_write"),
    "quant_step": ("logits", "next_token", "k_write", "v_write"),
    "spec_target_step": ("logits", "accepted_count", "committed_count",
                         "committed_tokens", "cache_length", "target_features",
                         "k_write", "v_write"),
    "gptoss_step": ("logits", "next_token", "k_write", "v_write", "expert_ids"),
    "hybrid_step": ("logits", "next_token", "k_write", "v_write",
                    "recurrent_state", "conv_state", "cache_length"),
    "vl_decode_step": ("logits", "next_token", "k_write", "v_write"),
    "distributed_step": ("logits", "next_token", "k_write", "v_write"),
    "deepseek_v32_step": ("logits", "next_token", "kv_write", "pe_write", "index_k_write", "sparse_indices", "expert_ids"),
    "glm52_step": ("logits", "next_token", "kv_write", "kv_scale_write", "pe_write", "index_k_write", "index_scale_write", "sparse_indices", "expert_ids"),
    "kimi_k3_step": ("logits", "next_token", "kv_write", "pe_write", "recurrent_state", "conv_state", "expert_ids", "cache_length"),
    "kimi_k3_layer": ("prefix", "expert_ids"),
    "glm53_flash_step": ("logits", "next_token", "kv_write", "index_k_write", "index_gate_write", "index_scores",
                         "sparse_mask", "recurrent_state", "conv_state", "expert_ids"),
    "tts_frame_step": ("codes", "logits", "k_write", "v_write"),
    "vla_action_step": ("actions",),
    "world_frame_step": ("latent", "k_write", "v_write"),
    "megamoe_layer": ("y",),
    "spec_full_iteration": ("logits", "proposed_tokens", "tree_parents", "accepted_count",
                            "proposed_count", "committed_count", "committed_tokens",
                            "cache_length", "draft_cache_length", "target_features",
                            "draft_hidden", "k_write", "v_write", "draft_k_write", "draft_v_write"),
}


def _python(benchmark_root: Path) -> Path:
    shared = benchmark_root / ".venv/bin/python"
    return shared if shared.is_file() else Path(sys.executable).resolve()


def _write_reference(project: Path, benchmark_root: Path, case: Case) -> str:
    target = project / "reference/megabench"
    files = REFERENCE_FILES[case.family]
    package_files = ["__init__.py", "tasks/__init__.py"]
    if any(name.startswith("references/") for name in files):
        package_files.append("tasks/references/__init__.py")
    for relative in package_files:
        source = benchmark_root / "megabench" / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative in files:
        source = benchmark_root / "megabench/tasks" / relative
        destination = target / "tasks" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # Keep the real dataclass: hybrid fixtures call dataclasses.replace().
    # Copy only its definition and the assigned geometry, never the catalog.
    definition = (benchmark_root / "megabench/cases.py").read_text().split(
        "CASES: tuple[Case, ...] = (", 1)[0]
    (target / "cases.py").write_text(definition + f"CASE = Case(**{case.to_dict()!r})\n")
    module = files[0].removesuffix(".py")
    (project / "reference/README.md").write_text(
        "# Assigned PyTorch reference\n\n"
        f"Read `megabench/tasks/{module}.py` for input generation and the "
        "PyTorch oracle. Its small helper imports are included here. "
        "`megabench.cases.CASE` contains only this case's public geometry.\n\n"
        "The reference snapshot is for inspection and local experiments. "
        "The check and profile scripts use the trusted benchmark outside "
        "this workspace.\n")
    return module


# The microbench phase is mandatory: before the implementer checks a candidate for the
# current VibeSys round, the round's requests and answers must exist, the answers must
# address each claim of the plan that rules a design out, and the engine map must carry a
# status for the round (megabench.integrations.microbench_gate).
ENGINE_MAP_TASK = (
    "\n## Engine map (read it before you choose or reject an engine)\n\n"
    "`progress/engine-map.md` lists each matrix-multiply phase of this case with every legal "
    "instruction mapping (tcgen05 with cta_group 1 or 2, normal or swap-AB; mma.sync; SIMT), "
    "the fraction of useful work after padding, the measured rate and row id from `./primdb`, "
    "and the measured GEMM or attention rows nearest the phase shape. Its status section records, "
    "each round, the engine that each dominant phase uses, the time it takes, and the gap to the "
    "best measured engine at that shape. `./engine_map M=.. N=.. K=..` prints the same table for "
    "another shape.\n\n"
    "A plan that rejects an engine, an instruction shape or a data path cites engine-map or "
    "`./primdb` row ids for the costs it compares. A rejection without rows is an open question: "
    "the microbench phase of the round measures it, and the result goes into the engine map.\n\n"
    "A plan's `task` names the design of each phase that the round writes, in a `Designs` "
    "table: `| phase | design | decomposition | ingress | rows |`, one row per phase, with the "
    "engine-map tags (E1, E2, ...) in the phase cell and the engine first in the design cell. "
    "A selection rejects the other designs of the phase, so each row is a plan claim like a "
    "rejection: `./microbench_items` lists it, and the microbench phase answers it with the "
    "tuned matched rows of that phase that rank the designs, or with an open ledger question "
    "that prices the other designs. The implementer receives the table in its task.\n\n"
    "Compare engines as whole designs. Each engine allows its own decomposition of a phase "
    "(where the accumulator lives, whether a CTA owns an output tile over the full reduction or "
    "the reduction is split and reduced in a separate pass, which neighbouring phase runs in its "
    "epilogue, how many grid barriers remain). A comparison that swaps the instruction inside "
    "one engine's decomposition, or that compares rows under different conditions, does not "
    "settle the choice. The share of profile samples in a phase's math does not either. "
    "An estimate made by swapping the instruction is a lower bound on what the other engine "
    "gains: it opens a ledger question for the whole design of each engine, and a plan ranks "
    "its targets by whole-design estimates, not by instruction-swap estimates.\n\n"
    "`progress/microbench/open.md` is the ledger of questions that measurements have not "
    "settled yet, each with an estimated end-to-end effect. When you plan a round, read it with "
    "the engine-map status. If the plan's target is not the open question with the largest "
    "estimate, the plan says why, with rows for both estimates.\n\n"
    "`progress/design-space.md` is the design tree of this case: for each dominant phase, every "
    "candidate design (engine, decomposition, data path) as alive, pruned (with the matched rows "
    "that prune it) or unknown (with an estimate range). The microbench phase keeps it. A plan "
    "names the branch it pursues and, for each phase it changes, a forecast of the phase time "
    "with the rows it uses; the next round reconciles the forecast with the measured time. Pursue "
    "the alive branch with the largest estimated gain, or say why not, with rows.\n\n"
    "Rank changes by what later rounds keep. A change of a phase's engine or decomposition "
    "replaces the code that every later tuning of that phase builds on, so tuning done before "
    "it is thrown away. When matched rows show a better design for a phase, the switch to that "
    "design comes before further tuning of the phase or of the work around it, unless the plan "
    "shows with rows that the switch does not change what that tuning would be. A profile's "
    "share of samples ranks the stalls of the current design, not the designs it could be.\n\n"
    "Rank the candidate changes of a round, across all phases, by their estimated end-to-end "
    "gain from rows. The size of a change, the number of files it touches, or how easy its gain "
    "is to attribute is not a reason to put a change with a larger estimated gain after one with "
    "a smaller gain: a switch of one phase to another design is one hypothesis, with the matched "
    "rows as its forecast and the phase time as its falsifier. If the plan targets a smaller "
    "change first, it shows with rows that the larger change gains less once the smaller one is "
    "in. Independent changes with no shared code can go in the same round.\n\n"
    "A change that rebuilds a phase's data path, decomposition or pipeline builds it inside the "
    "design that the matched rows rank best for that phase, not inside the design the candidate "
    "happens to use, unless the plan cites matched rows that show the current design gains more "
    "from the same rebuild. Work spent on a design that the rows rank lower is work that a later "
    "round throws away.\n")

MICROBENCH_TASK = (
    "\n## Microbench phase (mandatory in the opening rounds, then on open questions)\n\n"
    "A round plan starts as a draft. Before the implementer starts, the framework runs steps "
    "1-3 below on the draft, and the orchestrator revises the plan from the answers: the "
    "plan that the implementer receives already follows the measured rows. The implementer "
    "builds that plan, in the engine, decomposition and data path it names. When you "
    "implement, run `./microbench_check` first: if it passes, go to step 4. If it fails "
    "(the revised plan makes claims that the draft did not), do steps 1-3 for what it names.\n\n"
    "Before you write code for a round, do these steps. `<round>` is the name of the newest "
    "`progress/plans/round-*.json`.\n\n"
    "1. Run `./microbench_items`. It lists the claims of the plan that reject an alternative "
    "or rule a design out, as P1, P2, ... in `progress/microbench/items/<round>.md`.\n"
    "2. Write `progress/microbench/requests/<round>.md`. List what you need to know before you "
    "implement the plan: questions on the cost of a primitive, a shape or a data path; the "
    "assumptions that the plan depends on; the points where you are not sure how to implement "
    "an idea; and any performance regression from an earlier round that a microbenchmark could "
    "explain. Make each item specific: the shape, the data type, the occupancy and the metric "
    "that would settle it.\n"
    "3. Give the request file to the `microbench` subagent (Agent tool, "
    "`subagent_type: microbench`, `run_in_background: false`, so that you wait for its "
    "answers). It answers your items and each plan claim P1, P2, ... from "
    "`./primdb` rows that match the conditions (including rows that earlier runs of this case "
    "measured, origin `cache`) and from microbenchmarks of what no matched row covers, and it "
    "measures the "
    "largest question still open in `progress/microbench/open.md`. It writes "
    "`progress/microbench/answers/<round>.md`, adds new measurements to the database, updates "
    "the ledger, and adds a status entry for the round to `progress/engine-map.md`.\n"
    "4. Wait for complete answers before you write the code of a phase: `./microbench_check` "
    "passes only when they are complete. If it fails after the subagent returns, give the "
    "subagent its message and wait again. Then implement the plan. If later answers rank a "
    "phase's design differently from the code you already wrote, rebuild that phase in the "
    "design they rank best before you run `./check_candidate`. Before the check, write `progress/microbench/build/<round>.md`: one table row per phase of the candidate, `| phase | design built | notes |`, with the engine-map tags (E1, E2, ...) of the phase. `./check_candidate` refuses a phase built in another engine family than its fastest tuned matched row (rows with `tuned: true` from a sweep of two or more variants of the design; untuned rows and a phase with one tuned family force nothing), unless that family's own matched rows are within the run-to-run noise of it. To keep a slower design for one round, write `deferred: <reason> <row ids>` in the notes cell of the phase; the next round's record cannot defer the same phase. To keep a design for good, measure it under matched conditions; its rows then decide. In your notes, cite the answers that changed "
    "or confirmed a design choice. The plan's choice of design (engine, decomposition, data "
    "path) for a phase whose code this round writes or rewrites is a claim, not a constraint: "
    "if this round's matched rows rank another design best for that phase, build the phase in "
    "that design, apply the plan's change to it, and report the switch with its row ids. Every "
    "later round builds on the design you write now. If an answer contradicts another premise "
    "of the plan, still implement the plan, and report the contradiction with its row ids in "
    "your evidence and next step. Do not submit the round's result before `./check_candidate` has run on your "
    "candidate; a round without a checked candidate is lost.\n\n"
    "`./check_candidate` refuses to run until the request and answer files exist for the "
    "current round, the answers name every plan claim P1, P2, ..., the engine map has a "
    "status entry for the round, and each ledger question carried open from an earlier round "
    "has rows measured this round (one may be deferred with a reason). Rows that only run "
    "again the designs already measured for a question do not count: the rows must form a "
    "matched comparison or add a design or arm not measured for it before. A question that "
    "has gone two rounds without a matched comparison cannot be deferred. It also refuses a "
    "build record that is missing or that builds a phase in a slower family without a "
    "deferral. If a new question comes "
    "up during the round, append it to the "
    "request file and give it to the subagent again. Keep these files under `progress/`: a "
    "revert to an earlier round keeps them.\n\n"
    "After the opening rounds, the per-phase timeline of the current kernel "
    "(`./timeline_candidate`) is the default evidence for a plan, and the framework runs the "
    "phase only for the open questions of the draft that the timeline and the existing rows do "
    "not settle: a design that the kernel does not contain yet and that no matched row prices, "
    "or a cost that no phase of the kernel shows. Then the request names only those "
    "questions. When the draft has none, the framework writes "
    "`progress/microbench/skipped/<round>.md` with the reason, and the gate needs no request, "
    "answer or engine-map entry for that round. In such a round, cite the timeline rows of the "
    "parent for each design choice, and add a question to `progress/microbench/open.md` when "
    "the timeline cannot settle it, so that a later round measures it.\n")

CACHE_ROWS = "progress/microbench/primdb_cache.jsonl"

NO_FALLBACK_TASK = (
    "\n## No fallback to a design measured slower\n\n"
    "Each phase is built in the design that the measurements rank fastest. Two kinds of "
    "measurement rank designs, in this order:\n\n"
    "1. **In the kernel.** The per-phase time of the phase inside a correct candidate kernel, "
    "from `./timeline_candidate`, with each design built and checked in the same kernel.\n"
    "2. **In a proxy.** The matched rows of the microbenchmarks, when no in-kernel "
    "measurement compares the designs.\n\n"
    "An in-kernel measurement overrules the proxy rows for that phase: other phases, shared "
    "registers, cache and barriers change the cost of a phase, and only the kernel includes "
    "them. To keep a design that the rows rank slower, the design that the rows rank faster "
    "must first be built correctly in the same kernel and measured slower there; difficulty "
    "to build is never a reason. This rule binds every role:\n\n"
    "- **Orchestrator.** A plan names one design for each phase that it writes: the fastest "
    "by the in-kernel timeline where the parent's timeline compares the designs, else the "
    "fastest by the matched rows. It does not name a fallback to a slower design, and it "
    "does not make the faster design conditional on how hard it is to build.\n"
    "- **Implementer.** Build the phase in the design that the plan names. If you cannot make "
    "it correct, keep working on it inside the round; if the round ends first, return the "
    "failure: what you built, the test that fails, and the step that would fix it. If the "
    "named design is correct but the timeline shows the phase slower than the design it "
    "replaces, keep the faster build as the candidate and return both timelines and both "
    "checks as evidence.\n"
    "- **Judge.** Compare the design of each phase that the candidate builds or rewrites "
    "with the fastest design by the order above. Accept a design that the rows rank slower "
    "only when the round's evidence holds a correct build of the faster-ranked design in the "
    "same kernel and timelines of both builds that show it slower for that phase and slower "
    "end to end. Without that evidence, do not accept it, however correct or fast it is: "
    "send it back with the phase, the row ids and the design to build. Record each such "
    "gap between proxy and kernel as an open question in `progress/microbench/open.md`.\n")

GPU_JOBS_NOTE = (
    "- Every GPU job ends with a line that starts with the job name: the GPU it ran on, "
    "its SM and memory clocks, power, temperature, active throttle reasons, and how busy "
    "the other GPUs of the node were. `reports/gpu-jobs.jsonl` keeps one row for each "
    "job.\n")
TIMING_NOISE_TASK = (
    "\n## Timing noise on the GPU\n\n"
    "A process (or one loaded candidate inside it) can run in a slow mode: every launch "
    "in it is slower by a large, constant factor, and a fresh process of the same build "
    "is not. The cause is outside the candidate. So one timing from one process is not "
    "evidence.\n\n"
    "- `./check_candidate` times the candidate in the evaluate process and then in more "
    "fresh processes. It prints a `timing stability` block: the p50 of each process, the "
    "fast-cluster p50, the verdict, and the `score`. It also writes "
    "`reports/check-*.stability.json` and adds the run to `reports/timing-history.jsonl`.\n"
    "- `./phase_ab` times parent and variant in several fresh processes and gives the same "
    "block for each side, and the B-A difference of the two scores.\n"
    "{gpu_jobs}"
    "- A canary (a fixed GEMM and a fixed copy) runs in each fresh timing process. When a "
    "slow process also has a slow canary, the GPU or node was slow. When the canary is "
    "normal, the slow mode belongs to that loaded instance.\n\n"
    "The slow mode only makes a run slower, never faster. So:\n\n"
    "- The `score` of a code tree is the lowest fast-cluster p50 of all its runs with the same "
    "tool and setup, of any verdict. It is the tree's time. A new run can only lower it.\n"
    "- `STABLE`: all processes agree, and the run is not slower than an earlier run of the "
    "same code.\n"
    "- `UNSTABLE`: some processes were slow. The fast cluster is still the time of the code; "
    "the result is valid evidence.\n"
    "- `SLOW`: all processes agree, but they are slower than an earlier run of the same code. "
    "The whole run was in the slow mode; the score keeps the earlier best.\n\n"
    "How to use this:\n\n"
    "1. Compare candidates only by their scores, never by one process or one run.\n"
    "2. A `STABLE` result is not proof of a fast run: when every process of a run is slow, the "
    "processes agree. Compare a candidate with the score of its parent, not with a new "
    "parent run alone.\n"
    "3. When a decision depends on a result that is `UNSTABLE` or `SLOW`, or a candidate is "
    "slower than you expect, run the same command again as a stability rerun: "
    "`./check_candidate --rerun-of <report>` or `./phase_ab ... --rerun-of <result.json>`, "
    "as the output says. A stability rerun is not paid work. It does not count against a "
    "paid-call cap of the plan, in any role: the orchestrator does not count it when it "
    "sets a cap, the implementer and phase agents do not count it when they tally paid "
    "work, and the judge does not count it when it audits the tally. The tool accepts "
    "`--rerun-of` only when the earlier result was `UNSTABLE` or `SLOW` and the candidate "
    "files did not change, and it allows a small number of reruns after each such result. "
    "Each rerun report records `rerun_of`, so the judge can verify it. Change nothing "
    "between the runs: an edit makes the next run paid work again.\n"
    "   Every role, the judge included, settles a doubt about a timing with scores and these "
    "reruns, not with separate timing scripts or many extra processes.\n"
    "4. Do not change the kernel to chase a slow process. The same build was fast in "
    "another process, so the slowdown is not in the code.\n"
    "5. In your notes and in the round result, give the score, the verdict and the "
    "per-process values with each timing that you cite.\n"
    "6. When results stay `UNSTABLE` or `SLOW` over repeated runs, write it in your notes "
    "with the GPU state of those jobs, decide by the scores, and continue with the next "
    "item of the plan. Do not stop the round for it.\n"
)

PHASE_TIMING_TASK = (
    "\n## Per-phase timing and measured rounds\n\n"
    "A round acts on measured time per phase, not on estimates of it. This rule binds every "
    "role:\n\n"
    "- **Candidate contract.** A candidate with more than one phase exports "
    "`build_timeline(case)` next to `build` and `run`, with one event per phase. When a "
    "change adds, removes, splits or merges a phase, the same change updates the events.\n"
    "- **Implementer.** Run `./check_candidate` and `./timeline_candidate` in the foreground "
    "and read their results. Do not end your turn while a GPU command that you started still "
    "runs. Before you return a correct candidate, record its per-phase table from "
    "`./timeline_candidate` in the round notes.\n"
    "- **Orchestrator.** A plan that targets performance cites the parent's measured time for "
    "each phase from its latest timeline, next to the matched proxy row or the floor of that "
    "phase. It targets the phase with the largest gap between the two, or cites rows that "
    "show why another target recovers more.\n"
    "- **Forecast miss.** When the checked time is outside the range that the plan forecast, "
    "the next plan first finds the phases that missed their forecast and by how much, from "
    "the timeline, and only then names a design. A whole-kernel Nsight Compute capture adds "
    "all phases together, so it comes after the timeline, on the phase that the timeline "
    "ranks first.\n"
    "- **Judge.** A multi-phase candidate without `build_timeline`, or with events that do not "
    "match its phases, has a contract gap: send it back for the timeline, which does not "
    "change the candidate.\n")

PHASE_AGENTS_TASK = (
    "\n## Phase table and phase rounds\n\n"
    "By the end of the blueprint rounds (the opening rounds that run the full microbench "
    "phase), keep `progress/phases.md`: one row per phase of the kernel, with the timeline "
    "event name, the inputs and outputs with their layouts, the resources the phase may use "
    "(shared memory, registers, tensor-memory columns, blocks), the sync that starts and ends "
    "it, and its latest window from the timeline. `./timeline_candidate` writes the window of "
    "each phase (the critical-path time it adds) under `windows` in its `summary.json`. A row "
    "is the contract of its phase: other code relies on it. If a phase round finds no "
    "`progress/phases.md` (a run that starts from the kernel of an earlier run), the "
    "implementer first runs `./timeline_candidate` on the parent in the foreground and writes "
    "the table, then starts step 1. `progress/prior/` holds the round files of that earlier "
    "run: its plans, microbench requests and answers, and round notes.\n\n"
    "A later round is a phase round. The implementer coordinates and does not edit phase code "
    "before the subagents return:\n\n"
    "1. Give each target phase of the plan to a `phase` subagent (Agent tool, "
    "`subagent_type: phase`, `run_in_background: false`). Put all the Agent calls in one "
    "message: they run at the same time and your turn continues when all have returned. A "
    "subagent stops when your turn ends, so do not return the structured response while one "
    "is running. Give each subagent the round, the phase, "
    "its row of `progress/phases.md`, the plan and the parent timeline. Each subagent works in "
    "`variants/<round>/<phase>/<name>/` copies of the parent candidate and scores them with "
    "`./phase_ab PARENT_DIR VARIANT_DIR`, which in one GPU job builds both, checks the variant, "
    "times both alternately and compares the window of each phase.\n"
    "2. A boundary item (fuse two phases, replace a barrier) goes to one subagent that owns "
    "both phases.\n"
    "3. Wait for all subagents. Merge the winning diffs into the main candidate.\n"
    "4. Run `./check_candidate` and `./timeline_candidate` on the merged build in the "
    "foreground. Drop a diff whose gain does not hold in the merged build. Update "
    "`progress/phases.md` from the new timeline. Before the check, write the build record "
    "`progress/microbench/build/<round>.md` as the microbench section says.\n\n"
    "The orchestrator plan of a phase round names the target phases from the timeline windows, "
    "largest gaps first, and the boundary items.\n")

ROADMAP_ENGINES = (
    "# Engine selection per phase\n\n"
    "- [open] For each dominant phase, choose the engine, the instruction shape and the data path "
    "from measured rows (`progress/engine-map.md`, `./primdb`), not from general claims. The item "
    "stays open while the engine map shows a phase that runs slower than a measured engine at its "
    "shape by more than run-to-run noise, or a rejection that cites no rows.\n"
    "- [open] Work through `progress/microbench/open.md` in order of estimated effect; a plan "
    "that targets something else says why, with rows.\n")

DESIGN_SPACE = (
    "# Design space\n\n"
    "Kept by the microbench phase; see the `microbench` subagent for the format. One section per "
    "dominant phase with its candidate designs, then one `### round-NNNN` reconcile entry per "
    "round.\n")

LEDGER = (
    "# Open questions\n\n"
    "Questions that measurements have not settled under matching conditions, the largest "
    "estimated end-to-end effect first. Kept by the microbench phase; see the `microbench` "
    "subagent for the entry format. An entry closes only with the row ids that settle it.\n")


def seeded_ledger(engine_map: str) -> str:
    """The ledger with one question seeded from the engine map: which design wins each phase.

    The question names the phases of this case's engine map and asks for a matched comparison
    of every legal engine design. It carries no answer; the microbench phase of round 1 must
    measure it (``microbench_gate`` does not let a seeded question be deferred).
    """
    phases = re.findall(r"(?m)^### (E\d+): (.+)$", engine_map)
    listed = "; ".join(f"{tag} ({desc.strip()})" for tag, desc in phases) or (
        "the matrix-multiply phases of the reference")
    return LEDGER + (
        "\n## Q1 · open · opened round-0000\n"
        f"- phase: every phase in `progress/engine-map.md`: {listed}\n"
        "- question: for each phase that the reference profile or timeline shows as dominant, "
        "which engine design gives the lowest whole-phase time at its real shape, including the "
        "reductions, barriers and neighbouring phases that the design removes\n"
        "- settles it: per dominant phase, one proxy per engine family in its engine-map table, "
        "each built the way that engine allows (see Comparing engines and designs), all at the "
        "real shape in a persistent kernel with the same CTA count and ingress path\n"
        "- estimate: not yet measured; give the range with its arithmetic in round 1\n"
        "- history: seeded from the engine map when the workspace was made\n")


def _engine_map(project: Path, benchmark_root: Path, case: Case) -> str:
    """The seeded engine map of the case, or a stub when its phases were not traced."""
    result = subprocess.run(
        [str(_python(benchmark_root)), "-m", "megabench.primdb.engine_map", "--case", case.id],
        cwd=project, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(benchmark_root)})
    body = result.stdout if result.returncode == 0 else (
        f"# Engine map: {case.id}\n\nThe phases of this case were not traced. Read the "
        "reference, list its matrix multiplies, and add a section per phase with "
        "`./engine_map M=.. N=.. K=..`.\n")
    return body + (
        "\n## Status\n\n"
        "One entry per round, added by the microbench phase: `### round-NNNN`, then per dominant "
        "phase the current engine and instruction shape, its measured time (timeline or profile), "
        "the best measured engine at its shape with row ids, the gap, and open, closed or "
        "refuted (with rows).\n")


def _write_tools(project: Path, benchmark_root: Path, case: Case,
                 gpu_socket: Path | None = None, phase_agents: bool = False) -> None:
    python = shlex.quote(str(_python(benchmark_root)))
    trusted = shlex.quote(str(benchmark_root))
    submission = shlex.quote(str(project / "submission.py"))
    case_id = shlex.quote(case.id)
    common = ("#!/bin/sh\nset -eu\n"
              f"export PYTHONPATH={trusted}\n"
              f"cd {shlex.quote(str(project))}\n")
    regimes_flag = "--regimes regimes.json " if case.regimes else ""
    # With a GPU broker, the GPU commands run as Slurm jobs through ./gpu_run.
    gpu = ""
    if gpu_socket is not None:
        gpu = f"{shlex.quote(str(project / 'gpu_run'))} "
        gpu_run = project / "gpu_run"
        gpu_run.write_text(
            f"#!/bin/sh\nset -eu\nexport PYTHONPATH={trusted}\n"
            f"exec {python} -m megabench.integrations.gpu_broker run "
            f"--socket {shlex.quote(str(gpu_socket))} -- \"$@\"\n")
        gpu_run.chmod(0o755)
    check = project / "check_candidate"
    check.write_text(
        common + f"{python} -m megabench.integrations.microbench_gate check --build || exit 3\n"
        "run_id=$(date -u +%Y%m%dT%H%M%S)-$$\n"
        # --rerun-of REPORT (first): a stability rerun of an UNSTABLE or SLOW check of the same code.
        "rerun=\"\"\n"
        "if [ \"${1:-}\" = \"--rerun-of\" ]; then rerun=\"--rerun-of $2\"; shift 2; fi\n"
        # The evaluate process, then more fresh processes: gpu_noise reports their spread.
        f"exec {gpu}{python} -m megabench.integrations.gpu_noise check --case {case_id} "
        f"--submission {submission} $rerun " + regimes_flag
        + "--report \"reports/check-$run_id.jsonl\" -- "
        f"{python} -m megabench evaluate --case {case_id} "
        f"--submission {submission} --method local-check "
        "--session-id \"local-$run_id\" --device cuda:0 "
        "--trials 3 --warmup 1 --reps 3 --timeout 1200 "
        + regimes_flag + "--output \"reports/check-$run_id.jsonl\" \"$@\"\n")
    check.chmod(0o755)
    profile = project / "profile_candidate"
    profile.write_text(
        common + f"exec {gpu}{python} -m megabench.integrations.ncu_profile_case "
        f"--case {case_id} --submission {submission} \"$@\"\n")
    profile.chmod(0o755)
    timeline = project / "timeline_candidate"
    timeline.write_text(
        common + "run_id=$(date -u +%Y%m%dT%H%M%S)-$$\n"
        f"exec {gpu}{python} -m megabench.integrations.timeline_profile_case "
        f"--case {case_id} --submission {submission} "
        "--output \"reports/timeline-$run_id\" \"$@\"\n")
    timeline.chmod(0o755)
    if phase_agents:
        phase_ab = project / "phase_ab"
        phase_ab.write_text(
            common + "run_id=$(date -u +%Y%m%dT%H%M%S)-$$\n"
            f"exec {gpu}{python} -m megabench.integrations.phase_ab --case {case_id} "
            + regimes_flag + "--output \"reports/phase-ab-$run_id\" \"$@\"\n")
        phase_ab.chmod(0o755)
    shutil.copy2(benchmark_root / "megabench/integrations/timeline_probe.cuh",
                 project / "timeline_probe.cuh")
    shutil.copytree(benchmark_root / "megabench/skills", project / "megakernel_skills",
                    ignore=shutil.ignore_patterns("__pycache__"))
    sol = project / "sol_info"
    sol.write_text(
        common + f"exec {python} -m megabench sol --case {case_id} {regimes_flag}\"$@\"\n")
    sol.chmod(0o755)
    local = shlex.quote(str(project / "progress/microbench/primdb_local.jsonl"))
    cached = shlex.quote(str(project / CACHE_ROWS))
    for name, module, args in (("primdb", "megabench.primdb", ""),
                               ("engine_map", "megabench.primdb.engine_map", ""),
                               ("microbench_items", "megabench.integrations.microbench_gate",
                                "items "),
                               ("microbench_check", "megabench.integrations.microbench_gate",
                                "check ")):
        tool = project / name
        tool.write_text(common + f"export PRIMDB_LOCAL={local}\nexport PRIMDB_CACHE={cached}\n"
                        f"exec {python} -m {module} {args}\"$@\"\n")
        tool.chmod(0o755)
    # nvcc with a content cache and a time limit; it runs in the caller's directory.
    cache = shlex.quote(str(project / "progress/microbench/.build-cache"))
    mb_build = project / "mb_build"
    mb_build.write_text(
        f"#!/bin/sh\nset -eu\nexport PYTHONPATH={trusted}\n"
        f"export MB_BUILD_CACHE=\"${{MB_BUILD_CACHE:-{cache}}}\"\n"
        f"exec {python} -m megabench.integrations.mb_build \"$@\"\n")
    mb_build.chmod(0o755)
    shutil.copytree(benchmark_root / "megabench/primbench", project / "primbench",
                    ignore=shutil.ignore_patterns("__pycache__", "__init__.py", "*.sbatch"))
    shutil.copy2(benchmark_root / "megabench/primdb/schema.md", project / "primbench/schema.md")
    agents = project / ".claude/agents"
    agents.mkdir(parents=True)
    shutil.copy2(benchmark_root / "megabench/integrations/microbench-agent.md",
                 agents / "microbench.md")
    if phase_agents:
        shutil.copy2(benchmark_root / "megabench/integrations/phase-agent.md", agents / "phase.md")
    trace = project / "trace_regimes"
    trace.write_text(
        common + f"exec {python} -m megabench.integrations.trace_regimes "
        f"--case {case_id} \"$@\"\n")
    trace.chmod(0o755)


def _write_kernelagent_task(project: Path, benchmark_root: Path,
                            case: Case) -> None:
    (project / "problem.txt").write_text(
        (project / "TASK.md").read_text()
        + "\nFor KernelAgent, write `kernel.py` with "
        "`kernel_function(inputs: dict) -> dict`. Run `test.py` for local "
        "correctness. The campaign driver wraps this function as a MegaBench "
        "`build(case)` submission.\n")
    if case.gpus > 1:
        (project / "test.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(benchmark_root)!r})\n"
            "from megabench.cases import select_cases\n"
            "from megabench.harness.runner import evaluate_case\n"
            "if __name__ == '__main__':\n"
            f"    case = select_cases('all', [{case.id!r}])[0]\n"
            "    result = evaluate_case(case, Path(__file__).with_name('submission.py'), "
            "device='cuda:0', trials=3, warmup=1, reps=3)\n"
            "    assert result['status'] == 'ok_provisional', result\n")
        return
    scenarios = (("eagle3_raw", "accept_full", "accept_one")
                 if case.family in ("spec_target_step", "spec_full_iteration") else ("seeded",))
    spec_module = "speculative" if case.family == "spec_full_iteration" else "eagle3"
    (project / "test.py").write_text(
        "import sys\nimport torch\n"
        f"sys.path.insert(0, {str(benchmark_root)!r})\n"
        "from kernel import kernel_function\n"
        "from megabench.cases import select_cases\n"
        "from megabench.workloads import make_inputs, reference\n"
        "from megabench.runner import _compare, _audit_launches\n"
        f"case = select_cases('all', [{case.id!r}])[0]\n"
        "with torch.inference_mode():\n"
        f"    for index, scenario in enumerate({scenarios!r}):\n"
        "        inputs = make_inputs(case, 104729 + index, 'cuda:0')\n"
        "        if scenario in ('accept_full', 'accept_one'):\n"
        f"            from megabench.tasks.{spec_module} import set_acceptance_scenario\n"
        "            prefix = case.params['draft_depth'] if scenario == 'accept_full' else 1\n"
        "            set_acceptance_scenario(case, inputs, prefix)\n"
        "        originals = {name: value.detach().to('cpu', copy=True) "
        "for name, value in inputs.items()}\n"
        "        expected = reference(case, inputs)\n"
        "        actual = kernel_function(inputs)\n"
        "        torch.cuda.synchronize()\n"
        "        _compare(expected, actual, case, 'cuda:0')\n"
        "        assert all(torch.equal(inputs[name].cpu(), before) "
        "for name, before in originals.items())\n"
        "        audit = _audit_launches(kernel_function, inputs, 'cuda:0', 1)\n"
        "        assert audit['status'] == 'within_budget', audit\n")


def make_workspace(case_id: str, output: Path, *, agent: str = "vibesys",
                   benchmark_root: Path = BENCHMARK_ROOT,
                   regimes: Path | None = None, gpu_socket: Path | None = None,
                   gate_launcher: tuple[str, ...] = (), phase_agents: bool = False) -> Path:
    if agent not in ("vibesys", "kernelagent", "plain"):
        raise ValueError(f"unsupported agent: {agent}")
    case = select_cases("all", [case_id])[0]
    if not case.ready or case.family not in REFERENCE_FILES:
        raise ValueError(f"case has no minimal reference bundle: {case_id}")
    if regimes is not None:
        case = replace(case, regimes=load_regimes(regimes))
        if agent == "kernelagent":
            raise ValueError("regimes are not supported for the kernelagent workspace")
    benchmark_root = benchmark_root.resolve()
    project = output.resolve()
    project.mkdir(parents=True, exist_ok=False)
    (project / ".gitignore").write_text(
        "/.venv\nreports/\n.cache/\n.torch_extensions/\n"
        "__pycache__/\n*.log\n*.ncu-rep\n/agent.toml\n/variants/\n/.gpu_run/\n")
    shared_venv = benchmark_root / ".venv"
    if shared_venv.is_dir():
        (project / ".venv").symlink_to(shared_venv, target_is_directory=True)
    (project / "reports").mkdir()
    submission_source = (
        "def build(case: dict):\n"
        "    from kernel import kernel_function\n"
        "    return kernel_function\n"
        if agent == "kernelagent" else
        "def build(case: dict):\n    raise NotImplementedError(case['id'])\n"
    )
    (project / "submission.py").write_text(submission_source)
    module = _write_reference(project, benchmark_root, case)
    if regimes is not None:
        shutil.copyfile(regimes, project / "regimes.json")
    (project / "case.json").write_text(json.dumps(case.to_dict(), indent=2) + "\n")
    interface = (
        "Write `kernel.py` with `kernel_function(inputs: dict) -> dict`. "
        "The campaign driver wraps it as a MegaBench submission. "
        if agent == "kernelagent" else
        "Edit `submission.py` and your own helper files. Export "
        "`build(case: dict) -> run(inputs: dict) -> outputs: dict`. "
    )
    outputs = OUTPUT_KEYS[case.family]
    if case.family == "distributed_step" and "qwen" in case.model.lower():
        outputs += ("expert_ids",)
    if case.family == "kimi_k3_layer":
        from ..tasks.kimi_layer import attention_kind
        writes = ("kv_write", "pe_write") if attention_kind(case) == "mla" else ("recurrent_state", "conv_state")
        outputs = outputs[:1] + writes + outputs[1:]
    topology = (
        f"The evaluator starts {case.gpus} rank processes with data-parallel attention and EP={case.ep}. "
        "`build(case)` receives `case['execution']`: rank, world_size, tp_rank, ep_rank, "
        "device, backend, and live tp_group/ep_group handles. Each rank decodes its own "
        f"{case.params['batch']} sequences with replicated attention weights and owns "
        f"{case.params['experts'] // case.ep} contiguous experts. Routed latent rows must reach "
        "the ranks that own their experts and return to their source rank (the reference uses "
        "all-to-all). Complete all communication inside run. "
        "The budget includes NCCL kernels and is one GPU launch per rank. "
        "Use an allocation with all required GPUs visible. Nsight capture of spawned "
        "ranks requires `--target-processes all`.\n\n" if case.family == "kimi_k3_layer" else
        f"The evaluator starts {case.gpus} rank processes with EP={case.ep}. "
        "`build(case)` receives `case['execution']`: rank, world_size, tp_rank, ep_rank, "
        f"device, backend, and live tp_group/ep_group handles. Each rank holds {case.params['tokens']} "
        f"tokens with their routing and owns {case.params['experts'] // case.ep} contiguous experts. "
        "Token rows must reach the ranks that own their experts and the expert outputs must "
        "return to their source rank (the reference uses all-to-all). Complete all "
        "communication inside run. The budget includes NCCL kernels and is one GPU launch per rank. "
        "Use an allocation with all required GPUs visible. Nsight capture of spawned "
        "ranks requires `--target-processes all`.\n\n" if case.family == "megamoe_layer" else
        f"The evaluator starts {case.gpus} rank processes, with TP={case.tp}, EP={case.ep}. "
        "`build(case)` receives `case['execution']`: rank, world_size, tp_rank, ep_rank, "
        "device, backend, and live tp_group/ep_group handles. "
        "Weights and vocabulary logits use the reference's rank-local layouts. "
        "Compressed MLA KV and DSA index state are replicated; conventional KV and KDA heads are sharded. "
        "`next_token` is the global greedy token. Complete all communication inside run. "
        "The budget includes NCCL kernels and is one GPU launch per rank. "
        "Use an allocation with all required GPUs visible. Nsight capture of spawned "
        "ranks requires `--target-processes all`.\n\n" if case.gpus > 1 else "")
    (project / "TASK.md").write_text(
        f"# {case.id}\n\n" +
        (f"Implement one {case.model} decoder layer (layer {case.params['layer']}) {case.phase} step with the "
         if case.family == "kimi_k3_layer" else
         f"Implement the routed experts of one {case.model} MoE layer, dispatch through combine, with the "
         if case.family == "megamoe_layer" else
         f"Implement one complete {case.model} {case.phase} step with the ") +
        f"public geometry in `case.json`. Read `reference/megabench/tasks/{module}.py` "
        "and its included helpers for input layouts and exact PyTorch math.\n\n"
        + interface +
        f"Return {', '.join(f'`{key}`' for key in outputs)}. "
        "Compute from current runtime weights, token, KV cache, and any draft "
        "inputs; preserve all inputs. The complete step must use at most "
        "one GPU kernel launch per rank.\n\n" + topology
        + ("Use `test.py` for local correctness. " if agent == "kernelagent"
           else "") +
        "Use `./check_candidate` for local correctness, launch auditing, "
        "and timing. " +
        "Use `./profile_candidate` with Nsight Compute for optional "
        "kernel feedback. The capture command must set `TMPDIR=/tmp`, "
        "`TMP=/tmp`, and `TEMP=/tmp` inside the agent sandbox and pass "
        "`--profile-from-start off` to capture the warmed marker window. "
        "Use `./timeline_candidate` for per-block phase timing: export "
        "`build_timeline(case)` returning `(run, event_names)`, where `run(inputs)` "
        "launches a build with probes from the workspace's `timeline_probe.cuh` "
        "compiled with `-DMEGABENCH_TIMELINE` and returns the filled `uint64` buffer "
        "(see the header for its layout and macros). It writes a "
        "per-phase wall/slowest/mean/skew table and a Perfetto `trace.json` under "
        "`reports/`; probes slow the kernel, so confirm changes with `./check_candidate`. "
        "`megakernel_skills/` holds guidance for the method: split the kernel into "
        "phases and time them (`megakernel-phase-profiling`), study the slowest or "
        "furthest-from-floor phase in isolated microbenchmarks (`megakernel-microbench`; "
        "these do not use rounds), and evaluate Blackwell tensor-core designs "
        "(`blackwell-tensor-core-design`). "
        "When `regimes.json` (or the case) lists regimes, the score is the weighted "
        "result over all regimes, each shape is built and checked separately, and a "
        "submission may specialize per regime; `./trace_regimes` derives regimes from a "
        "production trace (see the `megakernel-regimes` skill). "
        "Use `./primdb` to look up measured B200 costs of hardware primitives (tensor-core "
        "and mma.sync instruction shapes, SIMT and dequant rates, TMEM moves, barriers, "
        "standalone and in-kernel GEMMs) before you choose or reject a design; "
        "`./primdb --help` lists the queries. "
        "Use `./sol_info --json` for an optimistic " +
        ("speed-of-light floor (the largest of the HBM, tensor-core and NVLink "
         "bounds) with byte and FLOP breakdowns" if work_model(case)
         else "HBM speed-of-light floor and byte breakdown") +
        "; pass `--result PATH` for "
        "the gap to a checked CUDA-event result where a traffic model is available. "
        "The default bandwidth is 8 TB/s per B200. These commands load the trusted benchmark outside "
        "this workspace; final grading is run separately.\n"
        + ("\nThis shell has no GPU. Run each GPU command, including microbenchmarks and "
           "Nsight captures, as `./gpu_run <command>`: it runs the command in this sandbox "
           "on one GPU as a separate Slurm job and can wait in the queue first. "
           "`./check_candidate`, `./profile_candidate` and `./timeline_candidate` already "
           "use it. Keep inputs and outputs inside this workspace, because the job does "
           "not see this shell's `/tmp`. Compile in this shell (`./mb_build` and nvcc need "
           "no GPU) and send only built programs to `./gpu_run`: a compile error inside a "
           "GPU job costs its wait in the queue.\n" if gpu_socket is not None else "")
        + (ENGINE_MAP_TASK + MICROBENCH_TASK + NO_FALLBACK_TASK
           + TIMING_NOISE_TASK.replace("{gpu_jobs}", GPU_JOBS_NOTE if gpu_socket is not None else "")
           + PHASE_TIMING_TASK
           + (PHASE_AGENTS_TASK if phase_agents else "") if agent == "vibesys" else ""))
    _write_tools(project, benchmark_root, case, gpu_socket, phase_agents)
    if agent == "vibesys":
        (project / "progress/microbench").mkdir(parents=True, exist_ok=True)
        engine_map = _engine_map(project, benchmark_root, case)
        (project / "progress/microbench/open.md").write_text(seeded_ledger(engine_map))
        (project / "progress/engine-map.md").write_text(engine_map)
        (project / "progress/design-space.md").write_text(DESIGN_SPACE)
        (project / "roadmap").mkdir(exist_ok=True)
        (project / "roadmap/engine-selection.md").write_text(ROADMAP_ENGINES)
        shutil.copy2(benchmark_root / "megabench/integrations/vibesys-agent.example.toml",
                     project / "agent.toml")
        make_task(case_id, project, benchmark_root,
                  project / "regimes.json" if regimes is not None else None,
                  gate_launcher)
    elif agent == "kernelagent":
        _write_kernelagent_task(project, benchmark_root, case)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=project, check=True)
    subprocess.run(["git", "add", "-A"], cwd=project, check=True)
    subprocess.run(["git", "-c", "user.name=MegaBench",
                    "-c", "user.email=megabench@example.invalid", "commit", "-q",
                    "-m", f"Prepare {case_id} agent workspace"],
                   cwd=project, check=True)
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", choices=("vibesys", "kernelagent", "plain"),
                        default="vibesys")
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT,
                        help="trusted benchmark snapshot used by workspace tools")
    parser.add_argument("--regimes", type=Path,
                        help="regime set (JSON) baked into the case and scoring")
    parser.add_argument("--gpu-socket", type=Path,
                        help="gpu_broker socket: GPU commands run through ./gpu_run")
    parser.add_argument("--phase-agents", action="store_true",
                        help="add the phase table section to TASK.md and install ./phase_ab and "
                             "the phase subagent")
    parser.add_argument("--gate-launcher", default="",
                        help="command prefix for the VibeSys accuracy and benchmark gates")
    args = parser.parse_args()
    print(make_workspace(args.case, args.output, agent=args.agent,
                         benchmark_root=args.benchmark_root, regimes=args.regimes,
                         gpu_socket=args.gpu_socket,
                         gate_launcher=tuple(shlex.split(args.gate_launcher)),
                         phase_agents=args.phase_agents))


if __name__ == "__main__":
    main()
