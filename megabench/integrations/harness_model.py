"""State-machine model of the VibeSys + megabench harness, with injected faults.

The model finds scenarios where the harness stops (a deadlock or a resume that can never
succeed) or ends with a kernel that is not the fastest approved one. Each part calls the
real decision code where that code is pure, so a change to the harness changes the model.

Parts:
  loop    The VibeSys round loop (multi/orchestration.py `_MultiRun`): rounds, paid attempts,
          judge, official evaluation, crash and resume, final winner. Explicit-state search
          over injected agent outcomes, judge verdicts, plan requests and crashes. Real code:
          cadence.review_due / official_due, compare_to_best + scalar_candidate_retained,
          provisional_candidate_retained, select_final_candidate, _MultiRun._first_attempt.
  timing  The decision "is the candidate faster than its parent" under the slow mode (a
          process can only be slower, by a fixed factor, with a probability). Exact
          enumeration of the process patterns, with reruns. Real code: gpu_noise.spread,
          gpu_noise.MAX_RERUNS.
  gate    microbench_gate.check on generated workspaces, one for each way the microbench
          phase can end (normal, hard time limit, failure, no summary, revised plan).
  broker  gpu_broker.choose_partition over a grid of queue windows, and the slot count of a
          real Broker after clients that leave in the middle of a job.

Run (needs an interpreter that imports both vibesys and megabench):
  <vibesys env>/bin/python -m megabench.integrations.harness_model [--json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace as NS

from . import gpu_noise


@dataclass
class Finding:
    part: str
    rule: str
    severity: str  # "stall", "wrong-winner", "lost-work", "cost", "ok"
    summary: str
    trace: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Part 1: the VibeSys round loop
# --------------------------------------------------------------------------------------------

# Agent outcomes of one implementer attempt (the tree is the workspace code).
#   win    the tree gets one level faster and stays correct
#   flat   no change of speed (the hypothesis was refuted)
#   break  the tree becomes incorrect (accuracy fails at the official gate)
#   fix    an incorrect tree becomes correct again
#   junk   no parseable response (the framework retries)
OUTCOMES = ("win", "flat", "break", "fix", "junk")


@dataclass(frozen=True)
class LoopState:
    phase: str                    # "round", "attempt", "done", "dead"
    rnd: int
    attempt: int = 0              # the attempt that runs next (0: not started)
    last_paid: tuple | None = None  # persisted (round, turn) of the latest paid attempt
    level: int = 0                # true speed level of the tree (higher is faster)
    ok: bool = True               # tree is correct
    approved: tuple = ()          # (round, level) of trees the judge passed while correct
    records: tuple = ()           # (round, passed, official, disposition, retained, level)
    requested: bool = False       # plan.request_official_evaluation of this round
    crashes: int = 0
    winner: int | None = None     # level of the final tree (baseline is 0)


def _vibesys():
    from vibesys.orchestration.hypothesis import cadence, transitions
    from vibesys.orchestration.hypothesis.config import HypothesisConfig
    from vibesys.orchestration.metrics import Measurement, MetricSpace
    from vibesys.orchestration.multi.orchestration import _MultiRun
    from vs_loop_state.agent import RoundRecord
    from vs_loop_state.api import CandidateDisposition, HypothesisOutcome
    return NS(cadence=cadence, transitions=transitions, HypothesisConfig=HypothesisConfig,
              Measurement=Measurement, MetricSpace=MetricSpace, MultiRun=_MultiRun,
              RoundRecord=RoundRecord, Disposition=CandidateDisposition,
              Outcome=HypothesisOutcome)


def _speed(level: int) -> float:
    """Headline metric (progress score, higher is better) of a tree level."""
    return round(1.0 * (1.03 ** level), 6)


class LoopModel:
    """Explicit-state search of `_MultiRun.execute` with injected agent behavior."""

    def __init__(self, max_rounds: int = 3, max_retries: int = 3, official_every: int = 3,
                 judge_every: int = 3, max_crashes: int = 1, outcomes=OUTCOMES):
        self.v = _vibesys()
        self.config = self.v.HypothesisConfig(
            max_rounds=max_rounds, judge_every=judge_every,
            official_eval_every=official_every, max_retries_per_round=max_retries)
        self.space = self.v.MetricSpace()
        self.max_crashes = max_crashes
        self.outcomes = outcomes

    # -- real records ---------------------------------------------------------------------
    def records(self, state: LoopState):
        out = []
        for rnd, passed, official, disp, retained, level in state.records:
            out.append(self.v.RoundRecord(
                round_number=rnd, commit=f"c{rnd}",
                perf_metric=_speed(level) if official else None,
                perf_unit="progress_score" if official else None, passed=passed,
                official_evaluation=official, candidate_disposition=disp,
                candidate_retained=retained, perf_direction="max",
                perf_provenance="framework" if official else None, legacy_reviewed=True))
        return out

    def first_attempt(self, state: LoopState) -> int:
        """Real `_MultiRun._first_attempt` on a stub of the persisted state."""
        marker = (NS(round_number=state.last_paid[0], member_id="h", turn_number=state.last_paid[1])
                  if state.last_paid else None)
        stub = NS(state=NS(usage_limited_attempt=None, last_paid_attempt=marker),
                  round_number=state.rnd)
        return self.v.MultiRun._first_attempt(stub, NS(request=NS(plan=NS(hypothesis_id="h"))))

    # -- transitions ----------------------------------------------------------------------
    def successors(self, s: LoopState):
        """Yield (event, next state)."""
        if s.phase in ("done", "dead"):
            return
        if s.phase == "round":
            if s.rnd > self.config.max_rounds:
                yield "finish", self.finish(s)
                return
            for requested in (False, True):
                first = self.first_attempt(s)
                if first > self.config.max_retries_per_round:
                    # _run_attempts raises "round N exhausted its paid attempts"; the run fails
                    # and every resume fails in the same place.
                    yield f"r{s.rnd}: resume -> exhausted paid attempts", replace(s, phase="dead")
                    return
                yield (f"r{s.rnd}: plan (official requested={requested})",
                       replace(s, phase="attempt", attempt=first, requested=requested))
            return
        # phase == "attempt": _mark_paid, then implement, review, official gate.
        retry = s.attempt
        if retry > self.config.max_retries_per_round:
            yield f"r{s.rnd}: attempts used up", self.close(s, passed=False, official=False)
            return
        paid = replace(s, last_paid=(s.rnd, retry))
        if s.crashes < self.max_crashes:
            # A crash (wall limit, node failure, kill) after the attempt was marked paid: the
            # run restarts from the persisted state at the same round.
            yield (f"r{s.rnd}a{retry}: crash after mark-paid, restart",
                   replace(paid, phase="round", attempt=0, crashes=s.crashes + 1))
        for outcome in self.outcomes:
            yield from self.attempt(paid, retry, outcome)

    def attempt(self, s: LoopState, retry: int, outcome: str):
        v, tag = self.v, f"r{s.rnd}a{retry}: implementer {outcome}"
        level, ok = s.level, s.ok
        if outcome == "win":
            level += 1
        elif outcome == "break":
            ok = False
        elif outcome == "fix":
            if ok:
                return
            ok = True
        nxt = replace(s, level=level, ok=ok, attempt=retry + 1)
        if outcome == "junk":
            yield tag + ", retry", nxt
            return
        declared = v.Outcome.DISPROVEN if outcome == "flat" else v.Outcome.SUPPORTED
        disposition = (v.Disposition.DISCARD if outcome == "flat"
                       else v.Disposition.PARETO_FRONTIER)
        due = v.cadence.review_due(self.config, round_number=s.rnd, outcome=declared,
                                   candidate_evidence_is_fresh=True)
        if not due:
            yield tag + ", review deferred", self.close(nxt, passed=False, official=False,
                                                       disposition=disposition)
            return
        yield tag + ", judge FAIL", nxt
        approved = nxt.approved + ((s.rnd, level),) if ok else nxt.approved
        passed = replace(nxt, approved=approved)
        ready = declared in {v.Outcome.SUPPORTED, v.Outcome.NOMINATED} or \
            disposition is v.Disposition.PARETO_FRONTIER
        reason = v.cadence.official_due(self.config, records=self.records(s), round_number=s.rnd,
                                        requested=s.requested, candidate_ready=ready)
        if reason is None:
            yield (tag + ", judge PASS, official not due",
                   self.close(passed, passed=True, official=False, disposition=disposition))
            return
        if not ok:
            # _evaluation_failed: revalidation, feedback, next attempt.
            yield tag + f", judge PASS, official ({reason}) accuracy FAIL", passed
            return
        yield (tag + f", judge PASS, official ({reason}) passed",
               self.close(passed, passed=True, official=True, disposition=disposition))

    def close(self, s: LoopState, passed: bool, official: bool, disposition=None) -> LoopState:
        v = self.v
        disposition = disposition or v.Disposition.UNASSESSED
        if official:
            prior = [v.Measurement(metric="progress_score", value=r.perf_metric, direction="max")
                     for r in self.records(s) if r.official_evaluation and r.perf_metric]
            reading = v.Measurement(metric="progress_score", value=_speed(s.level),
                                    direction="max")
            retained = v.transitions.scalar_candidate_retained(
                self.space.compare_to_best(reading, prior))
        else:
            retained = v.transitions.provisional_candidate_retained(disposition)
        record = (s.rnd, passed, official, disposition.value, retained, s.level)
        return replace(s, phase="round", rnd=s.rnd + 1, attempt=0, last_paid=None,
                       records=s.records + (record,), requested=False)

    def finish(self, s: LoopState) -> LoopState:
        winner = self.v.transitions.select_final_candidate(self.records(s), self.space)
        level = 0 if winner is None else next(r[5] for r in s.records if r[0] == winner.round_number)
        return replace(s, phase="done", winner=level)

    # -- search ---------------------------------------------------------------------------
    def explore(self, limit: int = 2_000_000):
        start = LoopState(phase="round", rnd=1)
        parent: dict[LoopState, tuple[LoopState | None, str]] = {start: (None, "start")}
        queue = deque([start])
        stuck, dead, lost = [], [], []
        while queue and len(parent) < limit:
            s = queue.popleft()
            nexts = list(self.successors(s))
            if not nexts and s.phase not in ("done", "dead"):
                stuck.append(s)
            if s.phase == "dead":
                dead.append(s)
            if s.phase == "done":
                best = max((lvl for _, lvl in s.approved), default=0)
                if s.winner < best:
                    lost.append(s)
            for event, n in nexts:
                if n not in parent:
                    parent[n] = (s, event)
                    queue.append(n)

        def trace(s):
            out = []
            while s is not None:
                prev, event = parent[s]
                out.append(event)
                s = prev
            return list(reversed(out))[1:]

        return NS(states=len(parent), stuck=stuck, dead=dead, lost=lost, trace=trace)

    def findings(self) -> list[Finding]:
        r = self.explore()
        out = []
        label = (f"max_rounds={self.config.max_rounds} retries={self.config.max_retries_per_round} "
                 f"official_every={self.config.official_eval_every}")
        if r.stuck:
            out.append(Finding("loop", "no state without a next step", "stall",
                               f"{len(r.stuck)} states have no next step ({label})",
                               r.trace(r.stuck[0])))
        if r.dead:
            shortest = min(r.dead, key=lambda s: len(r.trace(s)))
            out.append(Finding(
                "loop", "a restart always resumes", "stall",
                "a crash in the last paid attempt of a round makes every resume fail: "
                "_first_attempt returns last_paid.turn+1, which is above max_retries_per_round, "
                f"and _run_attempts raises 'exhausted its paid attempts' ({label}; "
                f"{len(r.dead)} reachable states)", r.trace(shortest)))
        if r.lost:
            worst = max(r.lost, key=lambda s: (max(l for _, l in s.approved) - s.winner,
                                               -len(r.trace(s))))
            best = max(lvl for _, lvl in worst.approved)
            baseline = sum(1 for s in r.lost if s.winner == 0)
            out.append(Finding(
                "loop", "the final tree is the fastest tree the judge approved", "wrong-winner",
                f"{len(r.lost)} end states keep a slower tree than the fastest approved one "
                f"({baseline} of them fall back to the input baseline). Only rounds with an "
                "official evaluation can win, and official evaluation runs only at the cadence, "
                "on request, or in the final round, so a failed final round loses the earlier "
                f"approved gains ({label}). Example: approved level {best}, kept level "
                f"{worst.winner}", r.trace(worst),
                {"lost": len(r.lost), "to_baseline": baseline}))
        if not out:
            out.append(Finding("loop", "all loop rules", "ok", f"no violation ({label}, "
                               f"{r.states} states)"))
        return out


# --------------------------------------------------------------------------------------------
# Part 2: timing decisions under the slow mode
# --------------------------------------------------------------------------------------------

def _patterns(p: float, n: int):
    """(slow flags, probability) for n processes, each slow with probability p."""
    for flags in itertools.product((False, True), repeat=n):
        k = sum(flags)
        yield flags, (p ** k) * ((1 - p) ** (n - k))


def _measure(true_ms: float, flags, slow_factor: float, tol: float) -> dict:
    return gpu_noise.spread([true_ms * (slow_factor if f else 1.0) for f in flags], tol)


def _outcomes(true_ms, p, n, slow_factor, tol, reruns):
    """Distribution of the final result of one check plus its stability reruns:
    [(fast_p50, verdict, jobs, min fast_p50 over all its runs, probability)]."""
    out = []

    def go(prob, jobs, best, depth):
        for flags, q in _patterns(p, n):
            res = _measure(true_ms, flags, slow_factor, tol)
            low = min(best, res["fast_p50_ms"])
            if res["verdict"] == "unstable" and depth < reruns:
                go(prob * q, jobs + 1, low, depth + 1)
            else:
                out.append((res["fast_p50_ms"], res["verdict"], jobs + 1, low, prob * q))

    go(1.0, 0, float("inf"), 0)
    merged: dict[tuple, float] = {}
    for *key, prob in out:  # identical results merge: the product enumerations stay small
        merged[tuple(key)] = merged.get(tuple(key), 0.0) + prob
    return [(*key, prob) for key, prob in merged.items()]


def timing_analysis(p_slow: float, delta_pct: float, slow_factor: float = 1.3,
                    tol: float = gpu_noise.DEFAULT_TOLERANCE_PCT, processes: int = 3,
                    parent_history: int = 1) -> dict:
    """Decision quality for a candidate that is delta_pct faster (negative: slower).

    current: accept when both final results are STABLE and the candidate's fast-cluster p50
             is lower; an UNSTABLE result after the reruns is inconclusive (keep the parent).
    digest_min: the slow mode only makes a run slower, so the best estimate of a code tree is
             the minimum fast-cluster p50 over all its runs, of any verdict; the parent also
             keeps the minimum of its earlier runs (parent_history of them).
    """
    parent_ms, cand_ms = 1.0, 1.0 * (1 - delta_pct / 100)
    rr = gpu_noise.MAX_RERUNS
    A = _outcomes(parent_ms, p_slow, processes, slow_factor, tol, rr)
    B = _outcomes(cand_ms, p_slow, processes, slow_factor, tol, rr)
    hist: dict[float, float] = {}
    for combo, prod in _history(A, parent_history):
        low = min(r[3] for r in combo)
        hist[low] = hist.get(low, 0.0) + prod
    hist = list(hist.items())
    faster = cand_ms < parent_ms
    cur = {"accept": 0.0, "reject": 0.0, "inconclusive": 0.0, "jobs": 0.0}
    dm = {"accept": 0.0, "reject": 0.0, "jobs": 0.0}
    for a in A:
        for b in B:
            q = a[4] * b[4]
            jobs = a[2] + b[2]
            cur["jobs"] += q * jobs
            if a[1] != "stable" or b[1] != "stable":
                cur["inconclusive"] += q
            elif b[0] < a[0]:
                cur["accept"] += q
            else:
                cur["reject"] += q
            for h, qh in hist:
                parent_best = min(a[3], h)
                dm["jobs"] += q * qh * jobs
                dm["accept" if b[3] < parent_best else "reject"] += q * qh
    wrong_cur = cur["reject"] + cur["inconclusive"] if faster else cur["accept"]
    wrong_dm = dm["reject"] if faster else dm["accept"]
    return {"p_slow": p_slow, "delta_pct": delta_pct, "faster": faster,
            "current": {k: round(v, 4) for k, v in cur.items()},
            "digest_min": {k: round(v, 4) for k, v in dm.items()},
            "wrong_current": round(wrong_cur, 4), "wrong_digest_min": round(wrong_dm, 4)}


def _history(A, n):
    """Combinations of n earlier parent runs, with their probabilities."""
    if n == 0:
        return [(((0, 0, 0, float("inf"), 1.0),), 1.0)]
    out = []
    for combo in itertools.product(A, repeat=n):
        prob = 1.0
        for r in combo:
            prob *= r[4]
        out.append((combo, prob))
    return out


def timing_findings(p_values=(0.1, 0.3, 0.5), deltas=(2.0, -2.0, 0.5)) -> list[Finding]:
    rows = [timing_analysis(p, d) for p in p_values for d in deltas]
    out = []
    false_accept = [r for r in rows if not r["faster"] and r["current"]["accept"] > 0]
    if false_accept:
        r = max(false_accept, key=lambda r: r["current"]["accept"])
        out.append(Finding(
            "timing", "a slower candidate is never accepted", "wrong-winner",
            "a STABLE verdict does not prove a fast measurement: when all processes of the "
            "parent's run are slow, the parent reads slow and STABLE, and a slower candidate "
            f"wins. Worst case p_slow={r['p_slow']}, candidate {-r['delta_pct']}% slower: "
            f"accepted with probability {r['current']['accept']} (digest_min: "
            f"{r['digest_min']['accept']})", [], {"rows": false_accept}))
    lost = [r for r in rows if r["faster"]]
    worst = max(lost, key=lambda r: r["wrong_current"])
    out.append(Finding(
        "timing", "a faster candidate is accepted", "cost",
        f"a faster candidate is rejected or left inconclusive with probability up to "
        f"{worst['wrong_current']} (p_slow={worst['p_slow']}, {worst['delta_pct']}% faster), "
        f"mostly because an UNSTABLE result counts as no evidence although its fast cluster "
        f"is the true time; digest_min: {worst['wrong_digest_min']}", [], {"rows": rows}))
    return out


# --------------------------------------------------------------------------------------------
# Part 3: the microbench gate after each way the phase can end
# --------------------------------------------------------------------------------------------

ROUND = "round-0002"


def _workspace(root: Path, plan_text: str, answers: str | None, request: bool = True,
               marker: str | None = None) -> Path:
    mb = root / "progress/microbench"
    (root / "progress/plans").mkdir(parents=True)
    (root / "progress/plans" / f"{ROUND}.json").write_text(json.dumps(
        {"reasoning": plan_text, "hypothesis": "h", "expected_effect": "e"}))
    (root / "progress/engine-map.md").write_text(f"status {ROUND}: reviewed\n")
    if request:
        (mb / "requests").mkdir(parents=True)
        (mb / "requests" / f"{ROUND}.md").write_text("deadline: x\nP1 measure\n")
    if answers is not None:
        (mb / "answers").mkdir(parents=True, exist_ok=True)
        (mb / "answers" / f"{ROUND}.md").write_text(answers)
    if marker:
        (mb / marker).mkdir(parents=True, exist_ok=True)
        (mb / marker / f"{ROUND}.md").write_text("# stopped\n")
    return root


def gate_findings() -> list[Finding]:
    from . import microbench_gate
    draft = "Rejected: a split design does not pay here."
    revised = draft + " The fused path is ruled out by register pressure."
    cases = [
        # (name, plan, answers, request file, marker, the run can go on without the phase)
        ("phase answered the draft", draft, "P1: measured, rows 0123456789\n", True, None, True),
        ("hard time limit, no answers", draft, None, True, "deadline", True),
        ("phase skipped by the framework", draft, None, False, "skipped", True),
        ("phase raised an error before any answer", draft, None, False, None, False),
        ("phase returned no summary, answers exist", draft, "P1: rows 0123456789\n", True,
         None, True),
        ("revised plan adds an uncited claim", revised, "P1: rows 0123456789\n", True, None,
         False),
    ]
    out, blocked = [], []
    for name, plan, answers, request, marker, _ in cases:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _workspace(Path(tmp), plan, answers, request, marker)
            problems = microbench_gate.check(ws)
        if problems:
            blocked.append((name, problems))
    for name, problems in blocked:
        if name == "phase raised an error before any answer":
            out.append(Finding(
                "gate", "the framework never leaves the gate unanswerable", "stall",
                "when the microbench phase raises an error (or its session dies) before it "
                "writes the request and answers, turns.py writes no marker, so check_candidate "
                "refuses every run of the round until the implementer writes the microbench "
                "files itself; the hard time limit writes progress/microbench/deadline/, this "
                "path does not", [f"gate: {p}" for p in problems]))
        elif name == "revised plan adds an uncited claim":
            out.append(Finding(
                "gate", "every claim the gate asks for has a role that answers it", "lost-work",
                "the revised plan replaces the draft after the microbench session closed; a new "
                "design-excluding sentence in it becomes a new P-tag that no answer names, so "
                "check_candidate refuses until the implementer answers it",
                [f"gate: {p}" for p in problems]))
        else:
            out.append(Finding("gate", name, "stall", "gate refused an expected pass",
                               [f"gate: {p}" for p in problems]))
    if not out:
        out.append(Finding("gate", "all phase endings", "ok", "the gate passes each ending"))
    return out


# --------------------------------------------------------------------------------------------
# Part 4: the GPU broker
# --------------------------------------------------------------------------------------------

def _windows(main_now: bool, prio_now: bool, main_max="02:00:00", prio_max="00:30:00"):
    later = "2099-01-01T00:00:00"
    return {"now": "2026-11-07T12:00:00", "partitions": [
        {"partition": "main", "groups": [{"gpus_min": 1, "gpus_max": 8, "options": [
            {"max_time": "00:20:00", "start": "now" if main_now else later},
            {"max_time": main_max, "start": later}]}]},
        {"partition": "priority", "groups": [{"gpus_min": 1, "gpus_max": 8, "options": [
            {"max_time": prio_max, "start": "now" if prio_now else later}]}]}]}


def broker_findings(jobs: int = 6, slots: int = 2) -> list[Finding]:
    from . import gpu_broker
    out = []
    bad = []
    for main_now, prio_now, minutes in itertools.product((True, False), (True, False),
                                                         (10, 30, 45, 90)):
        got = gpu_broker.choose_partition(_windows(main_now, prio_now), 1, minutes)
        main_ok = main_now and minutes <= 20
        prio_ok = prio_now and minutes <= 30
        want = "main" if main_ok else "priority" if prio_ok else None
        if got != want:
            bad.append(f"main_now={main_now} prio_now={prio_now} minutes={minutes}: "
                       f"got {got}, want {want}")
    if bad:
        out.append(Finding("broker", "partition choice follows slurm-windows", "cost",
                           f"{len(bad)} windows give the wrong partition", bad))
    # Slot leak: clients that leave in the middle of a job must give their slot back.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake = tmp / "bin"
        fake.mkdir()
        (fake / "scancel").write_text("#!/bin/sh\nexit 0\n")
        (fake / "scancel").chmod(0o755)
        old_path = os.environ["PATH"]
        os.environ["PATH"] = f"{fake}:{old_path}"
        try:
            args = argparse.Namespace(
                max_jobs=slots, sample_ms=0, spec=tmp / "none.json", requests=tmp,
                job_python=Path("/bin/true"), job_script=Path("/bin/true"),
                partition_policy="fixed", srun=["/bin/sh", "-c", "sleep 0.4", "sh"])
            broker = gpu_broker.Broker(args)
            broker.log = lambda message: None
            results, threads = [], []

            def client(leave: bool):
                ours, theirs = socket.socketpair()
                fds = [os.open(os.devnull, os.O_WRONLY), os.open(os.devnull, os.O_WRONLY)]
                if leave:
                    threading.Timer(0.15, ours.close).start()
                reply = broker.run(theirs, {"argv": ["/bin/true"], "cwd": str(tmp), "env": {}},
                                   fds)
                results.append(reply)
                for fd in fds:
                    os.close(fd)
                theirs.close()

            for i in range(jobs):
                t = threading.Thread(target=client, args=(i % 2 == 0,))
                t.start()
                threads.append(t)
            for t in threads:
                t.join(30)
            free = 0
            while broker.slots.acquire(blocking=False):
                free += 1
        finally:
            os.environ["PATH"] = old_path
    if free != slots:
        out.append(Finding("broker", "slots come back after clients leave", "stall",
                           f"{slots - free} of {slots} slots leaked", [json.dumps(results)]))
    if not out:
        out.append(Finding("broker", "partition choice and slot release", "ok",
                           f"16 windows correct; {jobs} jobs ({jobs // 2} clients left early) "
                           f"returned all {slots} slots"))
    return out


# --------------------------------------------------------------------------------------------

# Defects that the model found and that are not fixed yet, by (part, rule). A finding that is
# not here is new: main() then exits 1. Remove an entry when its fix lands.
KNOWN = {
    ("loop", "a restart always resumes"),
    ("loop", "the final tree is the fastest tree the judge approved"),
    ("timing", "a slower candidate is never accepted"),
    ("timing", "a faster candidate is accepted"),
    ("gate", "the framework never leaves the gate unanswerable"),
    ("gate", "every claim the gate asks for has a role that answers it"),
}


def new_problems(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity in ("stall", "wrong-winner")
            and (f.part, f.rule) not in KNOWN]


def run_all(official_every: int = 3) -> list[Finding]:
    findings = LoopModel(official_every=official_every).findings()
    findings += timing_findings()
    findings += gate_findings()
    findings += broker_findings()
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--official-every", type=int, default=3)
    args = parser.parse_args(argv)
    t0 = time.time()
    findings = run_all(args.official_every)
    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=1, default=str))
    else:
        for f in findings:
            known = " (known)" if (f.part, f.rule) in KNOWN else ""
            print(f"[{f.severity}{known}] {f.part}: {f.rule}\n  {f.summary}")
            for line in f.trace[:14]:
                print(f"    - {line}")
        print(f"({time.time() - t0:.1f}s)")
    new = new_problems(findings)
    for f in new:
        print(f"NEW: {f.part}: {f.rule}")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
