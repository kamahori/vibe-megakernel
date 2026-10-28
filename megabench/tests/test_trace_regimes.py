"""Trace-to-regimes replay: hand-checkable synthetic arrivals, bucketing, CLI exits."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..cases import load_regimes
from ..integrations import trace_regimes as tr

DENSE = "dense-step-qwen3-06b-b1-s128"
MOE = "moe-step-qwen3-30b-a3b-b1-s128"


def window(rows):
    t, i, o = zip(*rows)
    return tr.Window("syn", np.array(t, float), np.array(i), np.array(o), 60.0)


def run(rows, **kw):
    cfg = dict(max_seqs=8, max_batched_tokens=100, chunked_prefill=True, max_context=1000)
    cfg.update(kw)
    return tr.replay(window(rows), tr.Sched(**cfg), lambda d, k, p: 0.001)


class ReplayTests(unittest.TestCase):
    def test_two_requests_exact_steps(self) -> None:
        log = run([(0, 4, 3), (0, 4, 3)])
        self.assertEqual(log.d.tolist(), [0, 2, 2])       # prefill step, then two decode steps
        self.assertEqual(log.pf.tolist(), [8, 0, 0])
        self.assertEqual(log.kvmax.tolist(), [0, 4, 5])   # KV cached when the step runs
        self.assertEqual(log.counts["completed"], 2)

    def test_out_one_finishes_at_prefill(self) -> None:
        log = run([(0, 5, 1)])
        self.assertEqual(log.d.tolist(), [0])
        self.assertEqual(log.counts["completed"], 1)

    def test_max_seqs_limits_admission(self) -> None:
        log = run([(0, 2, 2)] * 3, max_seqs=2)
        self.assertEqual(log.d.tolist(), [0, 2, 0, 1])
        self.assertEqual(log.pf.tolist(), [4, 0, 2, 0])      # third request waits for a free slot
        self.assertTrue((log.d + 0 <= 2).all())

    def test_chunked_vs_unchunked(self) -> None:
        rows = [(0, 10, 2)]
        chunked = run(rows, max_batched_tokens=4)
        self.assertEqual(chunked.pf.tolist(), [4, 4, 2, 0])
        self.assertEqual(chunked.d.tolist(), [0, 0, 0, 1])
        # unchunked: a prompt must fit one step, so the second one waits for the next step
        un = run([(0, 3, 2), (0, 3, 2)], max_batched_tokens=4, chunked_prefill=False)
        self.assertEqual(un.pf.tolist(), [3, 3, 0])
        self.assertEqual(un.d.tolist(), [0, 1, 1])

    def test_load_scale_compresses_arrivals(self) -> None:
        rows = [(0, 2, 2), (10, 2, 2)]
        self.assertEqual(run(rows).counts["steps"], 4)                   # idle gap between requests
        fast = run(rows, load_scale=1e6)       # second request arrives during the first step
        self.assertEqual((fast.d.tolist(), fast.pf.tolist()), ([0, 1, 1], [2, 2, 0]))
        self.assertGreater(run(rows).idle_s, 9.9)

    def test_clip_shrinks_prompt_first(self) -> None:
        i, o, n = tr.clip_lengths(np.array([100, 5]), np.array([50, 0]), 120)
        self.assertEqual((i.tolist(), o.tolist(), n), ([70, 5], [50, 1], 1))

    def test_guards(self) -> None:
        log = run([(0, 2, 500)], max_steps=5)
        self.assertEqual(log.counts["steps"], 5)
        self.assertIn("max-steps", log.truncated)

    def test_step_timings_nearest_bucket(self) -> None:
        timings = {"regimes": [
            {"id": "a", "params": {"batch": 1, "context": 128}, "candidate_cuda_event_p50_ms": 1.0},
            {"id": "b", "params": {"batch": 32, "context": 4096}, "candidate_cuda_event_p50_ms": 5.0}]}
        f = tr.make_step_time(9.0, timings)
        self.assertAlmostEqual(f(2, 200, 0), 1e-3)
        self.assertAlmostEqual(f(30, 3000, 0), 5e-3)
        self.assertAlmostEqual(f(30, 3000, 16), 9e-3)     # prefill step falls back to --step-ms


class BucketTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = [(0.0, 20, 40), (0.0, 20, 40), (0.0, 300, 10), (5.0, 30, 30)]
        self.log = run(rows, max_context=1000)
        self.be, self.ke = tr.pow2_edges(8), tr.pow2_edges(1000)

    def test_edges_and_buckets(self) -> None:
        self.assertEqual(tr.pow2_edges(6), [1, 2, 4, 6])
        self.assertEqual(tr.bucket_of(np.array([1, 2, 3, 5]), [1, 2, 4, 8]).tolist(), [0, 1, 2, 3])

    def test_weights_sum_to_one(self) -> None:
        hist, buckets = tr.analyze(self.log, self.be, self.ke)
        self.assertAlmostEqual(sum(r["time_share"] for r in hist["buckets"]), 1.0)
        for family in ("dense_step", "moe_step"):
            cands, info = tr.pick_regimes(self.log, buckets, self.be, self.ke, family)
            self.assertAlmostEqual(sum(c["weight"] for c in cands), 1.0)
            self.assertLessEqual(info["scorable_time_share"], 1.0)

    def test_max_regimes_merges_weight(self) -> None:
        _, buckets = tr.analyze(self.log, self.be, self.ke)
        full, _ = tr.pick_regimes(self.log, buckets, self.be, self.ke, "dense_step")
        self.assertGreater(len(full), 2)
        two, _ = tr.pick_regimes(self.log, buckets, self.be, self.ke, "dense_step", max_regimes=2)
        self.assertEqual(len(two), 2)
        self.assertAlmostEqual(sum(c["weight"] for c in two), 1.0)


class CsvTests(unittest.TestCase):
    def test_parse_azure_csv(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                         "2024-05-10 00:00:00.009930+00:00,100,5\n"
                         "2024-05-10 00:00:01.509930+00:00,200,7\n")
            ts, inp, out = tr.parse_csv(p)
            self.assertEqual(inp.tolist(), [100, 200])
            self.assertEqual(out.tolist(), [5, 7])
            self.assertAlmostEqual(float((ts[1] - ts[0]) / np.timedelta64(1, "us")) / 1e6, 1.5)

    def test_pick_windows(self) -> None:
        counts = np.array([1, 1, 1, 9, 9, 9, 1, 1, 1, 5, 5, 5, 1, 1, 1])
        w = tr.pick_windows(counts, 3)
        self.assertEqual(w["peak"], 3)
        self.assertEqual(w["trough"], 0)


class CliTests(unittest.TestCase):
    def cli(self, *args, expect=0):
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            try:
                code = tr.main(list(args))
            except SystemExit as e:
                code = e.code
        self.assertEqual(code, expect, err.getvalue())
        return out.getvalue(), err.getvalue()

    def write_window(self, d: Path) -> Path:
        rows = [{"t": k * 0.05, "in": 50 + 40 * (k % 5), "out": 20 + 10 * (k % 3)} for k in range(200)]
        p = d / "w.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return p

    def base(self, d, case, extra=()):
        return ["--case", case, "--window-file", str(self.write_window(d)), "--max-seqs", "16",
                "--max-batched-tokens", "512", "--chunked-prefill", "--max-context", "1000",
                "--step-ms", "1", "--out", str(d / "r.json"), *extra]

    def test_dense_regimes_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.cli(*self.base(Path(d), DENSE))
            res = json.loads((Path(d) / "r.json").read_text())
            self.assertEqual(res["version"], 1)
            self.assertAlmostEqual(sum(r["weight"] for r in res["regimes"]), 1.0)
            for r in res["regimes"]:
                self.assertEqual(len(r["contexts"]), r["params"]["batch"])
                self.assertEqual(max(r["contexts"]), r["params"]["context"])
                self.assertGreaterEqual(min(r["contexts"]), 1)
            self.assertEqual(len(load_regimes(Path(d) / "r.json")), len(res["regimes"]))

    def test_non_dense_family_gets_batch_one(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.cli(*self.base(Path(d), MOE))
            res = json.loads((Path(d) / "r.json").read_text())
            self.assertTrue(res["regimes"])
            for r in res["regimes"]:
                self.assertEqual(list(r["params"]), ["context"])
                self.assertNotIn("contexts", r)
            self.assertGreater(res["not_scorable"]["time_share"], 0)
            self.assertEqual(len(load_regimes(Path(d) / "r.json")), len(res["regimes"]))

    def test_missing_data_source_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _, err = self.cli("--case", DENSE, "--max-seqs", "8", "--max-batched-tokens", "64", "--max-context", "1000",
                              "--step-ms", "1", "--out", f"{d}/r.json", expect=2)
            self.assertIn("MEGABENCH_TRACE_DIR", err)
            self.assertIn("AzurePublicDataset", err)

    def test_missing_step_model_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            args = self.base(Path(d), DENSE)
            i = args.index("--step-ms")
            _, err = self.cli(*args[:i], *args[i + 2:], expect=2)
            self.assertIn("step-time model", err)


if __name__ == "__main__":
    unittest.main()
