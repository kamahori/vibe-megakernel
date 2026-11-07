"""Rows that earlier runs of a case measured: harvest, lookup, reuse in a round, the gate."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from .. import primdb
from ..integrations import clean_vibesys_run as clean
from ..integrations.microbench_gate import check
from ..primdb import cache
from ..primdb.__main__ import main

CASE = "case-x"


def _row(engine: str, us: float, run: str, **extra) -> dict:
    flat = {"family": "phase", "primitive": "phase.mm", "p.M": 50, "p.N": 4096,
            "p.K": 1024, "p.engine": engine, "m.us": us, "c.sm_clock_mhz": 1900,
            "s.file": f"/runs/{run}/{CASE}/progress/microbench/bench/{engine}.cu",
            "s.round": "round-0001", "s.question": "Q1", "s.conditions": "matched"}
    return primdb.normalize(flat | extra)


def _local(root: Path, run: str) -> Path:
    return root / run / CASE / cache.LOCAL


class HarvestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_one_row_per_configuration_without_paths_into_other_runs(self) -> None:
        primdb.append([_row("one", 10.0, "r1", cv=0.05), _row("two", 8.0, "r1")],
                      _local(self.root, "r1"))
        # The same configuration at another clock, measured with less spread.
        primdb.append([_row("one", 9.5, "r2", cv=0.01, **{"c.sm_clock_mhz": 1950}),
                       _row("four", 1.0, "r2", kind="model")],
                      _local(self.root, "r2"))
        primdb.append([_row("three", 1.0, "r3")], _local(self.root, "r3"))
        rows = cache.harvest(self.root, CASE, exclude=("r3",))
        self.assertEqual(sorted(r["params"]["engine"] for r in rows), ["one", "two"])
        one = next(r for r in rows if r["params"]["engine"] == "one")
        self.assertEqual(one["metrics"]["us"], 9.5)
        self.assertEqual(one["source"]["cache_from"], "r2")
        self.assertEqual(one["source"]["cache_round"], "round-0001")
        self.assertEqual(one["source"]["file"], "one.cu")
        self.assertNotIn("question", one["source"])
        text = json.dumps(rows)
        self.assertNotIn("/runs/", text)

    def test_store_keeps_rows_after_the_run_directory_is_gone(self) -> None:
        primdb.append([_row("one", 10.0, "r1")], _local(self.root, "r1"))
        store = self.root / "store.jsonl"
        cache.write(cache.harvest(self.root, CASE), store)
        (_local(self.root, "r1")).unlink()
        rows = cache.harvest(self.root, CASE, store=store)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"]["cache_from"], "r1")

    def test_prepare_writes_the_cache_into_the_new_workspace(self) -> None:
        primdb.append([_row("one", 10.0, "r1")], _local(self.root, "r1"))
        primdb.append([_row("two", 10.0, "new")], _local(self.root, "new"))
        args = Namespace(case=CASE, microbench_cache_store=None)
        self.assertEqual(clean.cache_rows(args, self.root / "new"), 1)
        rows = list(primdb.read(self.root / "new" / CASE / "progress/microbench/primdb_cache.jsonl",
                                "cache"))
        self.assertEqual([r["params"]["engine"] for r in rows], ["one"])


class ReuseTests(unittest.TestCase):
    def test_reused_rows_count_as_this_rounds_matched_comparison(self) -> None:
        from ..integrations.make_agent_workspace import seeded_ledger
        with tempfile.TemporaryDirectory() as directory:
            ws = Path(directory)
            mb = ws / "progress/microbench"
            (ws / "progress/plans").mkdir(parents=True)
            (ws / "progress/plans/round-0001.json").write_text(json.dumps({"reasoning": "Go."}))
            for kind in ("requests", "answers"):
                (mb / kind).mkdir(parents=True)
                (mb / kind / "round-0001.md").write_text("- q\n")
            (ws / "progress/engine-map.md").write_text("### round-0001\n")
            (mb / "open.md").write_text(seeded_ledger("# Engine map\n\n"
                                                      "### E1: mm M=50 N=4096 K=1024\n"))
            rows = [cache._strip(_row(e, us, "old"), "old") for e, us in (("one", 9.0), ("two", 8.0))]
            for r in rows:
                r["id"] = primdb.row_id(r)
            cache.write(rows, mb / "primdb_cache.jsonl")
            ids = [r["id"] for r in rows]
            (mb / "answers/round-0001.md").write_text("P1\n## Q1\n" + " ".join(ids) + "\n")
            self.assertTrue(check(ws))
            out = io.StringIO()
            cwd = os.getcwd()
            try:
                os.chdir(ws)
                with patch.dict(os.environ, {"PRIMDB_CACHE": str(mb / "primdb_cache.jsonl")}):
                    self.assertEqual(main(["--local", str(mb / "primdb_local.jsonl"), "reuse",
                                           "zzzzzzzzzz"], io.StringIO()), 1)
                    self.assertEqual(main(["--local", str(mb / "primdb_local.jsonl"), "reuse",
                                           *ids, "--question", "Q1", "--conditions", "matched"],
                                          out), 0)
                    loaded = primdb.load(local=mb / "primdb_local.jsonl")
            finally:
                os.chdir(cwd)
            self.assertIn("reused 2 cached rows", out.getvalue())
            local = list(primdb.read(mb / "primdb_local.jsonl", "local"))
            self.assertEqual({r["source"]["round"] for r in local}, {"round-0001"})
            self.assertEqual({r["source"]["reused"] for r in local}, {"cache"})
            self.assertEqual([r["id"] for r in local], ids)
            # A reused row is listed once, as this run's row.
            self.assertEqual([r["origin"] for r in loaded if r["id"] in ids], ["local", "local"])
            self.assertEqual(check(ws), [])


if __name__ == "__main__":
    unittest.main()
