"""Ragged batched dense decode, Case contexts/regimes, and the batched HBM floor."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ..cases import CASES, Case, load_regimes
from ..sol import estimate_case
from ..tasks import dense
from ..tasks.oracle import fp32_reference
from ..tasks.references.qwen3 import RefDecoder

BASE = next(c for c in CASES if c.family == "dense_step")
TINY = {"batch": 1, "context": 8, "layers": 2, "hidden": 64, "q_heads": 4,
        "kv_heads": 2, "head_dim": 16, "intermediate": 96, "vocab": 200}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(**changes) -> Case:
    params = TINY | changes.pop("params", {})
    return replace(BASE, id="tiny", params=params, **changes)


class CaseTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        regimes = ({"id": "a", "weight": 2.0, "params": {"batch": 2}, "contexts": [3, 8]},
                   {"id": "b", "weight": 1, "params": {}})
        for case in (BASE, tiny(params={"batch": 3}, contexts=[5, 1, 8]),
                     tiny(params={"batch": 3}, contexts=(5, 1, 8), regimes=regimes)):
            with self.subTest(case=case.id):
                self.assertEqual(Case(**json.loads(json.dumps(case.to_dict()))), case)

    def test_default_dict_unchanged(self) -> None:
        keys = BASE.to_dict()
        self.assertNotIn("contexts", keys)
        self.assertNotIn("regimes", keys)
        self.assertIn("contexts", tiny(params={"batch": 2}, contexts=(1, 2)).to_dict())

    def test_validation(self) -> None:
        bad = [dict(params={"batch": 3}, contexts=(1, 2)),
               dict(params={"batch": 2}, contexts=(1, 9)),
               dict(params={"batch": 2}, contexts=(0, 2)),
               dict(regimes=({"id": "a", "weight": 1, "params": {}},) * 2),
               dict(regimes=({"id": "a", "weight": 0, "params": {}},)),
               dict(regimes=({"id": "a", "weight": float("inf"), "params": {}},)),
               dict(regimes=({"id": "a", "weight": 1, "params": {"nope": 1}},)),
               dict(regimes=({"id": "a", "weight": 1, "params": {"batch": 0}},)),
               dict(regimes=({"id": "a", "weight": 1, "params": {"batch": 2}, "contexts": [1]},))]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tiny(**changes)

    def test_regime_cases_and_load(self) -> None:
        doc = {"version": 1, "source": {"trace": "x"}, "regimes": [
            {"id": "a", "weight": 3, "params": {"batch": 2}, "contexts": [2, 7]},
            {"id": "b", "weight": 1, "params": {"context": 4}}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.json"
            path.write_text(json.dumps(doc))
            case = replace(tiny(), regimes=load_regimes(path))
        (ra, ca), (rb, cb) = case.regime_cases()
        self.assertEqual((ca.params["batch"], ca.contexts, ca.regimes), (2, (2, 7), ()))
        self.assertEqual((cb.params["context"], cb.params["batch"], cb.contexts), (4, 1, ()))
        self.assertEqual(ra["weight"], 3.0)


class DenseTests(unittest.TestCase):
    def test_batch_one_layout_unchanged(self) -> None:
        case = tiny()
        t = dense.make_inputs(case, 0, DEVICE)
        self.assertEqual(t["token"].shape, ())
        self.assertEqual(t["kcache"].shape, (2, 8, 2, 16))
        self.assertNotIn("context_lens", t)
        out = dense.reference(case, t)
        self.assertEqual(out["logits"].shape, (200,))
        self.assertEqual(out["k_write"].shape, (2, 2, 16))

    def test_batched_matches_per_sequence_loop(self) -> None:
        lens = (5, 1, 8)
        case = tiny(params={"batch": 3}, contexts=lens)
        t = dense.make_inputs(case, 1, DEVICE)
        self.assertEqual(t["token"].shape, (3,))
        self.assertEqual(t["kcache"].shape, (2, 3, 8, 2, 16))
        self.assertEqual(t["context_lens"].tolist(), list(lens))
        self.assertEqual(t["context_lens"].dtype, torch.int32)
        before = {k: v.clone() for k, v in t.items()}
        out = dense.reference(case, t)
        for k in t:
            self.assertTrue(torch.equal(t[k], before[k]), k)
        self.assertEqual(out["logits"].shape, (3, 200))
        self.assertEqual(out["next_token"].shape, (3,))
        self.assertEqual(out["next_token"].dtype, torch.int64)
        self.assertEqual(out["k_write"].shape, (2, 3, 2, 16))
        cfg = dense._qwen3_config(case)
        weights = {n: t[n] for n in dense.QWEN3_WEIGHT_NAMES}
        for b, n in enumerate(lens):
            ref = RefDecoder(cfg, weights, device=DEVICE)
            ref.k_cache[:, :n] = t["kcache"][:, b, :n]
            ref.v_cache[:, :n] = t["vcache"][:, b, :n]
            ref.pos = n
            logits = ref.step(int(t["token"][b]))
            torch.testing.assert_close(out["logits"][b], logits, atol=2e-2, rtol=2e-2)
            self.assertEqual(int(out["next_token"][b]), int(logits.argmax()))
            torch.testing.assert_close(out["k_write"][:, b], ref.k_cache[:, n], atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(out["v_write"][:, b], ref.v_cache[:, n], atol=2e-2, rtol=2e-2)

    def test_padding_ignored_and_oracle_runs(self) -> None:
        case = tiny(params={"batch": 3}, contexts=(5, 1, 8))
        t = dense.make_inputs(case, 2, DEVICE)
        want = dense.reference(case, t)
        for b, n in enumerate((5, 1, 8)):
            t["kcache"][:, b, n:] = 100.0
            t["vcache"][:, b, n:] = -100.0
        got = dense.reference(case, t)
        for k in want:
            self.assertTrue(torch.equal(want[k], got[k]), k)
        exact = fp32_reference(dense, case, t)
        self.assertEqual({k: (v.shape, v.dtype) for k, v in exact.items()}
                         .keys(), want.keys())
        self.assertEqual(exact["logits"].shape, want["logits"].shape)
        torch.testing.assert_close(exact["logits"], want["logits"], atol=5e-2, rtol=5e-2)


class SolTests(unittest.TestCase):
    def test_batched_floor(self) -> None:
        one = estimate_case(BASE)
        self.assertEqual(one["minimum_bytes"], 1207502352)
        four = estimate_case(replace(BASE, params=BASE.params | {"batch": 4}))
        ragged = estimate_case(replace(BASE, params=BASE.params | {"batch": 4},
                                       contexts=(128, 64, 32, 1)))
        self.assertGreater(four["minimum_bytes"], one["minimum_bytes"])
        self.assertLess(ragged["minimum_bytes"], four["minimum_bytes"])
        p = BASE.params
        kv = lambda n: 2 * p["layers"] * n * p["kv_heads"] * p["head_dim"] * 2
        by = {c["name"]: c["bytes"] for c in ragged["components"]}
        self.assertEqual(by["old_kv_cache_reads"], kv(128 + 64 + 32 + 1))
        self.assertEqual(by["logits_writes"], 4 * p["vocab"] * 4)

    def test_other_families_stay_batch_one(self) -> None:
        moe = next(c for c in CASES if c.family == "moe_step")
        row = estimate_case(replace(moe, params=moe.params | {"batch": 2}))
        self.assertEqual(row["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
