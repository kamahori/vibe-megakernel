from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from megabench import primdb
from unittest import mock

from megabench.integrations import microbench_gate
from megabench.integrations.microbench_gate import check, plan_items
from megabench.primdb import engine_map, phases

# Rejections of earlier VibeSys plans, in their words; none cites a measured row.
PAST_REJECTIONS = [
    {"reasoning": "No candidate exists. Rejected: starting directly with tcgen05/TMA designs, "
                  "because at batch 1 the GEMVs have no tensor-core reuse and that complexity "
                  "adds correctness risk. Rejected: aggressive phase fusion."},
    {"reasoning": "Rejected alternatives: - Tuning CS alone keeps the per-item syncs. - A "
                  "tensor-core QK/PV path is premature while latency and sync dominate. "
                  "Keeping the other phases fixed keeps the attribution clean."},
    {"reasoning": "The rejected alternatives are a tcgen05/tensor-core GEMM and per-phase "
                  "tuning. A tensor-core GEMM does not pay off at b=1, where the step is pure "
                  "GEMV and bandwidth-bound."},
    {"reasoning": "Rejected alternatives: - Tensor-core mma: needs a multi-row tile "
                  "restructure and still pays the same per-element dequant to bf16. - Dropping "
                  "per-element bf16 rounding: risks breaking reference numerics."},
]


DESIGNS_TASK = (
    "Rebuild the GEMM phases.\n\nDesigns:\n"
    "| phase | design | decomposition | ingress | rows |\n"
    "|---|---|---|---|---|\n"
    "| E2 | tcgen05 swap-AB | stream-K, TMEM accumulator | cp.async ring 4 | 6ddb22f9a6 |\n"
    "| E3/E5 | mma.sync swap-AB | warp split-K, red.add | ldg128 | |\n"
    "\nThen run the check.")


class PlanItemTests(unittest.TestCase):
    def test_past_tensor_core_rejections_become_items(self) -> None:
        for plan in PAST_REJECTIONS:
            with self.subTest(plan=plan["reasoning"][:40]):
                items = plan_items(plan)
                self.assertTrue(any("tensor-core" in i or "Tensor-core" in i for i in items), items)
                self.assertFalse(any(i.startswith("Keeping") for i in items), items)

    def test_plan_without_rejections_has_no_items(self) -> None:
        self.assertEqual(plan_items({"reasoning": "Profile first. The parent is HEAD."}), [])

    def test_designs_table_rows_become_items_first(self) -> None:
        plan = dict(PAST_REJECTIONS[2], task=DESIGNS_TASK)
        items = plan_items(plan)
        self.assertTrue(items[0].startswith("Design of E2: tcgen05 swap-AB (stream-K"), items)
        self.assertTrue(items[1].startswith("Design of E3/E5: mma.sync"), items)
        self.assertTrue(any("tcgen05/tensor-core GEMM" in i for i in items[2:]), items)
        self.assertEqual(len(microbench_gate.plan_designs(plan)), 2)

    def test_task_without_designs_table_asks_for_one(self) -> None:
        items = plan_items({"task": "Fuse the residual into the down epilogue."})
        self.assertEqual(items, [microbench_gate.NO_DESIGNS])
        self.assertIsNone(microbench_gate.plan_designs({"task": "| a | b |\n| 1 | 2 |"}))


class GateTests(unittest.TestCase):
    # These tests write one row per design; TunedTests covers which rows count as tuned.
    def setUp(self) -> None:
        patcher = mock.patch.object(microbench_gate, "tuned_rows", lambda rows: list(rows))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_gate_needs_files_answers_and_map_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            self.assertEqual(check(ws), [])
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0002.json").write_text(json.dumps(PAST_REJECTIONS[2]))
            problems = check(ws)
            self.assertEqual(len(problems), 3, problems)
            items = (ws / "progress/microbench/items/round-0002.md").read_text()
            self.assertIn("- P1:", items)
            mb = ws / "progress/microbench"
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0002.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("## Status\n### round-0002\n")
            self.assertIn("P1", check(ws)[0])
            n = items.count("- P")
            (mb / "answers/round-0002.md").write_text(" ".join(f"P{i}" for i in range(1, n + 1)))
            self.assertEqual(check(ws), [])

    def test_design_items_need_rows_of_their_phase_or_an_open_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            (ws / "progress/plans").mkdir(parents=True)
            plan = {"task": DESIGNS_TASK, "reasoning": "The parent is HEAD."}
            (ws / "progress/plans/round-0002.json").write_text(json.dumps(plan))
            mb = ws / "progress/microbench"
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0002.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("## Status\n### round-0002\n")
            answers = mb / "answers/round-0002.md"
            # Named, but with no evidence that can rank the phase.
            answers.write_text("## P1\nThe draft says so.\n## P2\nAlso.\n")
            problems = check(ws)
            self.assertEqual(len(problems), 2, problems)
            self.assertIn("design item P1 (E2)", problems[0])
            self.assertIn("design item P2 (E3, E5)", problems[1])
            # A matched row of another phase does not answer E2; one of E2 does.
            (mb / "primdb_local.jsonl").write_text("".join(json.dumps(
                {"id": i, "params": {"engine": "tcgen05_swapab", "phase": p},
                 "source": {"round": "round-0002", "conditions": "matched"}}) + "\n"
                for i, p in (("aaaaaaaaaa", "E4"), ("bbbbbbbbbb", "E2"))))
            answers.write_text("## P1\nRanked by aaaaaaaaaa.\n## P2\nOpen.\n")
            self.assertIn("design item P1", check(ws)[0])
            answers.write_text("## P1\nRanked by bbbbbbbbbb.\n## P2\nOpen, Q3.\n")
            self.assertIn("design item P2", " ".join(check(ws)), "Q3 is not in the ledger")
            (mb / "open.md").write_text("## Q3 · open · opened round-0002\n")
            # The open question adds its own item (P3).
            answers.write_text("## P1\nRanked by bbbbbbbbbb.\n## P2\nOpen, Q3.\n## P3\nQ3.\n")
            self.assertEqual(check(ws), [])
            # Answers without headings: the text after the tag answers it.
            answers.write_text("P1: bbbbbbbbbb ranks it. P2: see Q3. P3: Q3.\n")
            self.assertEqual(check(ws), [])

    def test_round_skipped_by_the_framework_needs_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0003.json").write_text(json.dumps(PAST_REJECTIONS[2]))
            self.assertTrue(check(ws))
            skipped = ws / "progress/microbench/skipped"
            skipped.mkdir(parents=True)
            (skipped / "round-0003.md").write_text("")
            self.assertTrue(check(ws), "an empty skip record does not count")
            (skipped / "round-0003.md").write_text("# round-0003: skipped\nreason\n")
            self.assertEqual(check(ws), [])
            (skipped / "round-0002.md").write_text("# other round\n")
            (skipped / "round-0003.md").unlink()
            self.assertTrue(check(ws), "a skip record of another round does not count")
            # A question overdue for a matched comparison blocks a skipped round too.
            (skipped / "round-0003.md").write_text("# round-0003: skipped\nreason\n")
            mb = ws / "progress/microbench"
            (mb / "open.md").write_text("## Q1 · open · opened round-0001\n")
            self.assertIn("Q1 has gone 2 or more rounds", check(ws)[0])
            (mb / "answers").mkdir()
            (mb / "answers/round-0003.md").write_text("## Q1\naaaaaaaaaa\n")
            (mb / "primdb_local.jsonl").write_text("".join(json.dumps(
                {"id": i, "params": {"engine": e},
                 "source": {"round": "round-0003", "question": "Q1", "conditions": "matched"}})
                + "\n" for i, e in (("aaaaaaaaaa", "one"), ("aaaaaaaaab", "two"))))
            self.assertEqual(check(ws), [])

    def test_round_cut_at_the_hard_limit_needs_no_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0003.json").write_text(json.dumps(PAST_REJECTIONS[2]))
            self.assertTrue(check(ws))
            cut = ws / "progress/microbench/deadline"
            cut.mkdir(parents=True)
            (cut / "round-0003.md").write_text("# stopped\n")
            self.assertEqual(check(ws), [])

    def test_phase_that_failed_before_its_answers_needs_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0003.json").write_text(json.dumps(PAST_REJECTIONS[2]))
            self.assertTrue(check(ws))
            stopped = ws / "progress/microbench/stopped"
            stopped.mkdir(parents=True)
            (stopped / "round-0003.md").write_text("# failed\n")
            self.assertEqual(check(ws), [])

    def test_claim_only_in_revised_plan_goes_to_the_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            draft = {"reasoning": "Rejected: a split-K GEMM does not pay."}
            revised = {"reasoning": draft["reasoning"] + " A fused epilogue is ruled out."}
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0002.json").write_text(json.dumps(revised))
            (ws / "progress/engine-map.md").write_text("round-0002 round-0003\n")
            mb = ws / "progress/microbench"
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0002.md").write_text("P1: measured\n")
            self.assertTrue(check(ws), "without the draft copy the new claim is P2")
            (mb / "draft").mkdir()
            (mb / "draft/round-0002.json").write_text(json.dumps(draft))
            self.assertEqual(check(ws), [])
            carried = (mb / "carried/round-0002.md").read_text()
            self.assertIn("fused epilogue", carried)
            (ws / "progress/plans/round-0003.json").write_text(json.dumps({"reasoning": "x"}))
            for kind in ("requests", "answers"):
                (mb / kind / "round-0003.md").write_text("nothing\n")
            problems = check(ws)
            self.assertTrue(any("does not answer plan claims P1" in p for p in problems))
            (mb / "answers/round-0003.md").write_text("P1: measured\n")
            self.assertEqual(check(ws), [])

    def test_revised_claim_citing_answer_rows_is_answered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            (ws / "progress/plans").mkdir(parents=True)
            plan = {"reasoning": "Rejected: a split-K GEMM, because row 0123456789 is slower. "
                                 "Rejected: SIMT attention is too costly."}
            (ws / "progress/plans/round-0002.json").write_text(json.dumps(plan))
            mb = ws / "progress/microbench"
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0002.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("### round-0002\n")
            (mb / "answers/round-0002.md").write_text("P2: measured. Row 0123456789 is 4 us.\n")
            self.assertEqual(check(ws), [])
            (mb / "answers/round-0002.md").write_text("P2: measured.\n")
            self.assertIn("P1", check(ws)[0])

    def test_ledger_needs_rows_measured_in_the_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            mb = ws / "progress/microbench"
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0003.json").write_text(json.dumps({"reasoning": "Go."}))
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0003.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("### round-0003\n")
            (mb / "open.md").write_text(
                "# Open questions\n\n## Q1 · open · opened round-0001\n- settle: matched pair\n\n"
                "## Q2 · open · opened round-0003\n- settle: new this round\n")
            answers = "P1\n## Q1\nstill estimated\n"
            (mb / "answers/round-0003.md").write_text(answers)
            problems = check(ws)
            self.assertIn("Q1 has gone 2 or more rounds without a matched comparison", problems[0])
            (mb / "open.md").write_text(
                "# Open questions\n\n## Q1 · open · opened round-0002\n- settle: matched pair\n\n"
                "## Q2 · open · opened round-0003\n- settle: new this round\n")
            self.assertIn("Q1 (opened round-0002) has no rows", check(ws)[0])
            # A row measured in an earlier round does not count; one stamped with this round does.
            local = mb / "primdb_local.jsonl"
            local.write_text(json.dumps({"id": "aaaaaaaaaa", "source": {"round": "round-0002"}}) + "\n"
                             + "".join(json.dumps({"id": i, "params": {"engine": e},
                                                   "source": {"round": "round-0003", "question": "Q1",
                                                              "conditions": "matched"}}) + "\n"
                                       for i, e in (("bbbbbbbbbb", "one"), ("bbbbbbbbbc", "two"))))
            (mb / "answers/round-0003.md").write_text("P1\n## Q1\nrow aaaaaaaaaa\n")
            self.assertIn("Q1", check(ws)[0])
            # A row of a design measured for Q1 before, run again, does not count.
            (mb / "answers/round-0002.md").write_text("## Q1\nbbbbbbbbbb\n")
            (mb / "answers/round-0003.md").write_text("P1\n## Q1\nrow bbbbbbbbbb\n## Q2\n")
            local.write_text(json.dumps({"id": "bbbbbbbbbb", "params": {"engine": "one"},
                                         "source": {"round": "round-0003", "question": "Q1"}}) + "\n"
                             + json.dumps({"id": "bbbbbbbbba", "params": {"engine": "one"},
                                           "source": {"round": "round-0002", "question": "Q1"}}) + "\n")
            self.assertIn("measure only designs measured for it before (one)", check(ws)[0])
            (mb / "answers/round-0002.md").unlink()
            local.write_text(json.dumps({"id": "aaaaaaaaaa", "source": {"round": "round-0002"}}) + "\n"
                             + "".join(json.dumps({"id": i, "params": {"engine": e},
                                                   "source": {"round": "round-0003", "question": "Q1",
                                                              "conditions": "matched"}}) + "\n"
                                       for i, e in (("bbbbbbbbbb", "one"), ("bbbbbbbbbc", "two"))))
            (mb / "answers/round-0003.md").write_text("P1\n## Q1\nrow bbbbbbbbbb\n## Q2\n")
            self.assertEqual(check(ws), [])
            # One carried question may be deferred; closing needs rows.
            (mb / "answers/round-0003.md").write_text("P1\n## Q1\ndeferred: Q2 first\n")
            self.assertEqual(check(ws), [])
            # Not the same question two rounds in a row.
            (mb / "answers/round-0002.md").write_text("## Q1\ndeferred: later\n")
            self.assertIn("deferred in the previous round", check(ws)[0])
            (mb / "answers/round-0002.md").unlink()
            (mb / "open.md").write_text("## Q1 · closed · opened round-0001\n- by reasoning\n")
            self.assertIn("without row ids", check(ws)[0])
            (mb / "open.md").write_text("## Q1 · refuted · opened round-0001\n- cccccccccc\n")
            self.assertEqual(check(ws), [])

    def test_stalest_question_needs_a_matched_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            mb = ws / "progress/microbench"
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0003.json").write_text(json.dumps({"reasoning": "Go."}))
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0003.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("### round-0003\n")
            (mb / "open.md").write_text("## Q1 · open · opened round-0001\n\n"
                                        "## Q2 · open · opened round-0001\n")

            def rows(*specs):
                # Each matched spec also gets a second design, so that it is a comparison.
                out = []
                for i, r, q, c in specs:
                    for k, engine in enumerate(("one", "two") if c == "matched" else ("three",)):
                        out.append(json.dumps({"id": i[:-1] + str(k), "params": {"engine": engine},
                                               "source": {"round": r, "question": q,
                                                          "conditions": c}}) + "\n")
                (mb / "primdb_local.jsonl").write_text("".join(out))

            # Q1 had a matched row in round 2; Q2 never did, so Q2 is due.
            rows(("aaaaaaaaaa", "round-0002", "Q1", "matched"),
                 ("bbbbbbbbbb", "round-0003", "Q1", "other"),
                 ("cccccccccc", "round-0003", "Q2", "other"))
            (mb / "answers/round-0003.md").write_text("P1\n## Q1\nbbbbbbbbb0\n## Q2\nccccccccc0\n")
            problems = check(ws)
            self.assertEqual(len(problems), 1, problems)
            # Two rounds without a matched comparison: the due question is also overdue.
            self.assertIn("Q2 has gone 2 or more rounds", problems[0])
            rows(("aaaaaaaaaa", "round-0002", "Q1", "matched"),
                 ("bbbbbbbbbb", "round-0003", "Q1", "other"),
                 ("cccccccccc", "round-0003", "Q2", "matched"))
            self.assertEqual(check(ws), [])
            # An earlier answer measured under matching conditions also counts.
            rows(("bbbbbbbbbb", "round-0003", "Q1", "other"), ("cccccccccc", "round-0003", "Q2", "other"))
            (mb / "answers/round-0002.md").write_text("## Q2\n- Confidence: measured under matching conditions.\n")
            self.assertIn("Q1 has gone 2 or more rounds", check(ws)[0])
            (mb / "answers/round-0002.md").unlink()
            rows(("aaaaaaaaaa", "round-0002", "Q1", "matched"),
                 ("bbbbbbbbbb", "round-0003", "Q1", "other"),
                 ("cccccccccc", "round-0003", "Q2", "matched"))
            # An overdue question cannot be deferred.
            (mb / "answers/round-0003.md").write_text("P1\n## Q1\nbbbbbbbbb0\n## Q2\ndeferred: x\n")
            self.assertIn("Q2 has gone 2 or more rounds", " ".join(check(ws)))

    def test_only_the_largest_overdue_question_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            mb = ws / "progress/microbench"
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0003.json").write_text(json.dumps({"reasoning": "Go."}))
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0003.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("### round-0003\n")
            # Ledger order is the estimate order: Q2 first.
            (mb / "open.md").write_text("## Q2 · open · opened round-0001\n\n"
                                        "## Q1 · open · opened round-0001\n")
            (mb / "primdb_local.jsonl").write_text("".join(json.dumps(
                {"id": i, "params": {"engine": e}, "source": {"round": "round-0003", "question": q}})
                + "\n" for i, e, q in (("aaaaaaaaaa", "one", "Q1"), ("bbbbbbbbbb", "two", "Q2"))))
            (mb / "answers/round-0003.md").write_text("P1\n## Q1\naaaaaaaaaa\n## Q2\nbbbbbbbbbb\n")
            problems = check(ws)
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("Q2 has gone 2 or more rounds", problems[0])

    def test_seeded_question_is_due_in_round_one_and_cannot_be_deferred(self) -> None:
        from megabench.integrations.make_agent_workspace import seeded_ledger
        ledger = seeded_ledger("# Engine map\n\n### E1: mm M=50 N=4096 K=1024\n\n"
                               "### E2: addmm M=1 N=3072 K=1024\n")
        self.assertIn("## Q1 · open · opened round-0000", ledger)
        self.assertIn("E1 (mm M=50 N=4096 K=1024); E2 (addmm M=1 N=3072 K=1024)", ledger)
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            mb = ws / "progress/microbench"
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0001.json").write_text(json.dumps({"reasoning": "Go."}))
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0001.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("### round-0001\n")
            (mb / "open.md").write_text(ledger)
            (mb / "answers/round-0001.md").write_text("P1\n## Q1\ndeferred: later\n")
            problems = check(ws)
            self.assertTrue(any("cannot be deferred" in p for p in problems), problems)
            (mb / "primdb_local.jsonl").write_text(json.dumps(
                {"id": "dddddddddd", "source": {"round": "round-0001", "question": "Q1",
                                                "conditions": "other"}}) + "\n")
            (mb / "answers/round-0001.md").write_text("P1\n## Q1\ndddddddddd\n")
            self.assertIn("Q1 has gone longest", " ".join(check(ws)))
            matched = [{"id": "dddddddddd", "params": {"engine": "one"},
                        "source": {"round": "round-0001", "question": "Q1", "conditions": "matched"}},
                       {"id": "eeeeeeeeee", "params": {"engine": "oracle_load_only"},
                        "source": {"round": "round-0001", "question": "Q1", "conditions": "matched"}}]
            (mb / "primdb_local.jsonl").write_text("".join(json.dumps(r) + "\n" for r in matched))
            # Matched rows of one design and an oracle are not a comparison.
            self.assertIn("Q1 has gone longest", " ".join(check(ws)))
            matched.append({"id": "ffffffffff", "params": {"engine": "two"},
                            "source": {"round": "round-0001", "question": "Q1",
                                       "conditions": "matched"}})
            (mb / "primdb_local.jsonl").write_text("".join(json.dumps(r) + "\n" for r in matched))
            self.assertEqual(check(ws), [])

    def test_seeded_question_needs_every_engine_family_of_the_engine_map(self) -> None:
        from megabench.integrations.make_agent_workspace import seeded_ledger
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            mb = ws / "progress/microbench"
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0001.json").write_text(json.dumps({"reasoning": "Go."}))
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0001.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text(
                "### E1: mm M=50 N=4096 K=1024\n\n| engine | mapping |\n|---|---|\n"
                "| alpha.one | x |\n| beta | y |\n\n### round-0001\n")
            (mb / "open.md").write_text(seeded_ledger("### E1: mm M=50 N=4096 K=1024\n"))

            def rows(*engines):
                (mb / "primdb_local.jsonl").write_text("".join(json.dumps(
                    {"id": f"{k:010d}", "params": {"engine": e},
                     "source": {"round": "round-0001", "question": "Q1", "conditions": "matched"}})
                    + "\n" for k, e in enumerate(engines)))
                (mb / "answers/round-0001.md").write_text(
                    "P1\n## Q1\n" + " ".join(f"{k:010d}" for k in range(len(engines))) + "\n")

            # Two variants of one engine family are not an engine comparison.
            rows("alpha_one_plain", "alpha_one_prefetch", "oracle_load")
            self.assertIn("Q1 has gone longest", check(ws)[0])
            rows("alpha_one_plain", "beta_split")
            self.assertEqual(check(ws), [])
            # Every family in the engine map, not two of them.
            (ws / "progress/engine-map.md").write_text(
                "| engine | mapping |\n|---|---|\n| alpha.one | x |\n| beta | y |\n| GAMMA | z |\n\n### round-0001\n")
            self.assertIn("Q1 has gone longest", check(ws)[0])
            rows("alpha_one_plain", "beta_split", "gamma_ffma")
            self.assertEqual(check(ws), [])
            # Without matched phase rows no build record is needed.
            self.assertEqual(check(ws, build=True), [])
            (mb / "build").mkdir()
            record = mb / "build/round-0001.md"
            head = "| phase | built | notes |\n|---|---|---|\n"
            record.write_text(head + "| E1 qkv | beta_split | |\n")
            # Rows without p.phase and m.us rank nothing.
            self.assertEqual(check(ws, build=True), [])

            def timed(*rows):
                (mb / "primdb_local.jsonl").write_text("".join(json.dumps(
                    {"id": f"{k:010d}", "params": {"engine": e, "phase": ph}, "metrics": {"us": us},
                     "source": {"round": "round-0001", "question": "Q1",
                                "conditions": c}}) + "\n"
                    for k, (e, ph, us, c) in enumerate(rows)))

            timed(("alpha_one_plain", "E1", 4.0, "matched"), ("beta_split", "E1", 4.1, "matched"),
                  ("gamma_ffma", "E1", 9.0, "matched"), ("oracle_load", "E1", 1.0, "matched"),
                  ("beta_split", "E1", 2.0, "other"))
            # The fastest matched row decides; an oracle and other-conditions rows do not.
            self.assertIn("fastest matched row is 0000000000", check(ws, build=True)[0])
            record.unlink()
            self.assertIn("build/round-0001.md is missing", check(ws, build=True)[0])
            record.write_text(head + "| E1 qkv | beta_split | |\n")
            record.write_text(head + "| E1 qkv | alpha_one_tuned | |\n")
            self.assertEqual(check(ws, build=True), [])
            # A faster matched row of the built design changes the order.
            timed(("alpha_one_plain", "E1", 4.0, "matched"), ("beta_split", "E1", 3.9, "matched"),
                  ("gamma_ffma", "E1", 9.0, "matched"))
            self.assertIn("(beta_split, 3.9 us)", check(ws, build=True)[0])
            # Every phase with matched rows needs a row in the record.
            timed(("alpha_one_plain", "E1", 4.0, "matched"), ("beta_split", "E2", 3.0, "matched"),
                  ("gamma_ffma", "E1", 9.0, "matched"))
            # A phase with one engine family measured has nothing to rank against.
            self.assertEqual(check(ws, build=True), [])
            timed(("alpha_one_plain", "E1", 4.0, "matched"), ("beta_split", "E2", 3.0, "matched"),
                  ("gamma_ffma", "E1", 9.0, "matched"), ("gamma_ffma", "E2", 5.0, "matched"))
            self.assertIn("no row for phase E2", " ".join(check(ws, build=True)))
            record.write_text(head)
            self.assertIn("no phase rows", " ".join(check(ws, build=True)))

    def test_open_questions_become_a_plan_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0002.json").write_text(json.dumps({"reasoning": "Go."}))
            (ws / "progress/microbench").mkdir(parents=True)
            (ws / "progress/microbench/open.md").write_text("## Q4 · open · opened round-0001\n")
            check(ws)
            items = (ws / "progress/microbench/items/round-0002.md").read_text()
            self.assertIn("- P1: The plan chose its target over the open questions Q4", items)


class TunedTests(unittest.TestCase):
    """Only tuned rows (tuned: true, a design with a sweep of two or more matched rows)
    settle an engine comparison or force the build of a phase."""

    def test_untuned_side_does_not_settle_or_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            mb = ws / "progress/microbench"
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0001.json").write_text(json.dumps({"reasoning": "Go."}))
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0001.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text(
                "| engine | mapping |\n|---|---|\n| alpha | x |\n| beta | y |\n\n### round-0001\n")
            (mb / "open.md").write_text("## Q1 · open · opened round-0000\n")
            (mb / "build").mkdir()
            (mb / "build/round-0001.md").write_text(
                "| phase | built | notes |\n|---|---|---|\n| E1 | beta_split | |\n")

            def rows(*specs):
                (mb / "primdb_local.jsonl").write_text("".join(json.dumps(
                    {"id": f"{k:010d}", "params": {"engine": e, "phase": "E1", "tuned": t},
                     "metrics": {"us": us},
                     "source": {"round": "round-0001", "question": "Q1", "conditions": "matched"}})
                    + "\n" for k, (e, us, t) in enumerate(specs)))
                (mb / "answers/round-0001.md").write_text(
                    "P1\n## Q1\n" + " ".join(f"{k:010d}" for k in range(len(specs))) + "\n")

            # alpha is tuned (a sweep of two), beta is not: no comparison, no forced build.
            rows(("alpha_a", 4.0, True), ("alpha_a", 4.5, True), ("beta_split", 6.0, False))
            problems = check(ws, build=True)
            self.assertIn("Q1 has gone longest", " ".join(problems))
            self.assertNotIn("fastest matched row", " ".join(problems))
            # A tuned flag on one row without a sweep does not count either.
            rows(("alpha_a", 4.0, True), ("alpha_a", 4.5, True), ("beta_split", 6.0, True))
            self.assertIn("Q1 has gone longest", " ".join(check(ws, build=True)))
            # Both tuned with sweeps: the comparison settles and the build follows it.
            rows(("alpha_a", 4.0, True), ("alpha_a", 4.5, True), ("beta_split", 6.0, True),
                 ("beta_split", 6.5, True))
            problems = check(ws, build=True)
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("fastest matched row is 0000000000", problems[0])


# The E2 rows of archC16-pi05-s1 (gate/up, M=50 N=4096 K=1024), in short: (engine, decomposition,
# knob, us, tuned). mma.sync full-K (D1) is 3.6x its split-K (D2); tcgen05 stream-K (D3) is fastest.
ARCH_E2 = [
    ("mma_sync_swapab", "output_tile", {"ksl": 1024}, 24.09, True),
    ("mma_sync_swapab", "split_k_warp_redadd", {"ksl": 128}, 6.67, True),
    ("mma_sync_swapab", "split_k_warp_redadd", {"ksl": 128}, 6.71, True),
    ("mma_sync_swapab", "split_k_warp_redadd", {"ksl": 128}, 6.62, True),
    ("mma_sync_swapab", "split_k_warp_redadd", {"ksl": 256}, 7.93, True),
    ("tcgen05_swapab", "stream_k_redadd", {"ring": 3}, 6.55, True),
    ("tcgen05_swapab", "stream_k_redadd", {"ring": 3}, 6.51, True),
    ("tcgen05_swapab", "stream_k_redadd", {"ring": 4}, 6.49, True),
    ("tcgen05_swapab", "stream_k_redadd", {"ring": 4}, 6.53, True),
    ("oracle_load_barrier_only", "split_k_warp_redadd", {"ksl": 128}, 3.01, False),
]


class RankingTests(unittest.TestCase):
    """check_build on the archC16 E2 pattern: a sweep of one engine is not noise, and two
    decompositions of one engine are two designs."""

    def workspace(self, directory: str, rows=ARCH_E2, round_name: str = "round-0001") -> Path:
        ws = Path(directory)
        mb = ws / "progress/microbench"
        (mb / "build").mkdir(parents=True)
        (ws / "progress/engine-map.md").write_text(
            "### E2: mm M=50 N=4096 K=1024 batch=1 bfloat16, 360 per step\n\n"
            "| engine | mapping |\n|---|---|\n| mma.sync | x |\n| tcgen05 | y |\n")
        (mb / "primdb_local.jsonl").write_text("".join(json.dumps(
            {"id": f"{k:010x}", "family": "gemm_phase", "primitive": "gemm_phase.persistent",
             "params": {"engine": e, "partition": d, "phase": "E2", "tuned": t, **knob},
             "metrics": {"us": us},
             "source": {"round": round_name, "question": "Q1", "conditions": "matched"}})
            + "\n" for k, (e, d, knob, us, t) in enumerate(rows)))
        return ws

    def build(self, ws: Path, cell: str, notes: str = "", round_name: str = "round-0001") -> list[str]:
        (ws / f"progress/microbench/build/{round_name}.md").write_text(
            f"| phase | design built | notes |\n|---|---|---|\n| E2 gate/up | {cell} | {notes} |\n")
        return microbench_gate.check_build(ws, round_name)

    def test_noise_is_the_spread_of_repeats_not_of_a_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = self.workspace(directory)
            rows = microbench_gate.matched_phase_rows(ws)
            # The mma.sync rows spread 264% (a sweep); its repeats spread 1.4%; the floor wins.
            self.assertAlmostEqual(microbench_gate.phase_noise(rows, "E2"), 0.02)
            problems = self.build(ws, "mma.sync swap-AB split-K")
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("2.0% slower (noise 2.0%)", problems[0])
            self.assertEqual(self.build(ws, "tcgen05 swap-AB stream-K"), [])


class EngineMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.seed = Path(self.tmp.name) / "seed.jsonl"
        rows = [primdb.normalize({"family": "tc_mma", "primitive": "tcgen05.mma", "p.kind": "f16",
                                  "p.a_src": "smem", "p.cta_group": 1, "p.M": 128, "p.N": n,
                                  "m.fma_per_clk_sm": 32.0 * n, "c.sm_clock_mhz": 1900.0})
                for n in (16, 64, 256)]
        rows.append(primdb.normalize({"family": "mma_sync", "primitive": "mma.sync",
                                      "p.shape": "m16n8k16", "m.fma_per_clk_sm": 1000.0}))
        rows.append(primdb.normalize({"family": "simt", "primitive": "simt.ffma",
                                      "m.flop_per_clk_sm": 256.0}))
        primdb.append(rows, self.seed)
        self.rows = primdb.load(self.seed, Path(self.tmp.name) / "none.jsonl")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mappings_pad_tokens_to_legal_shapes(self) -> None:
        by = {(m["engine"], m["orientation"], m["instr"]): m
              for m in engine_map.mappings(self.rows, 50, 4096, "bfloat16")}
        swap = by[("tcgen05", "cta_group 1, swap-AB (tokens as MMA N)", "M128 N64")]
        self.assertEqual(swap["padded_m"], 64)
        self.assertAlmostEqual(swap["useful"], 50 / 64)
        self.assertEqual(swap["rate"], 32.0 * 64)
        self.assertEqual(by[("tcgen05", "cta_group 1, tokens as MMA M", "M128 N256")]["padded_m"], 128)
        self.assertEqual(by[("SIMT", "FFMA, fp32 accumulate", "-")]["rate"], 128.0)
        # cta_group 2 has no rows here: the map says to measure it.
        self.assertTrue(any(m["rate"] is None for m in by.values()))

    def test_section_reports_bytes_and_rows(self) -> None:
        text = engine_map.section(self.rows, {"M": 50, "N": 4096, "K": 1024, "dtype": "bfloat16"})
        self.assertIn("8.39 MB per call", text)
        self.assertIn("| tcgen05 |", text)
        self.assertIn("no measured GEMM rows", text)

    def test_traced_cases_map_with_the_seed(self) -> None:
        table = phases.load()
        self.assertTrue(table, "case_phases.json is empty")
        rows = primdb.load()
        for case_id, case_phases in table.items():
            with self.subTest(case=case_id):
                self.assertTrue(all({"M", "N", "K", "count"} <= set(p) for p in case_phases))
                out = io.StringIO()
                self.assertEqual(engine_map.main(["--case", case_id], out), 0)
                self.assertIn("E1: ", out.getvalue())


if __name__ == "__main__":
    unittest.main()
