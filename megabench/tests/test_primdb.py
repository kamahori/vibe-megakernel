"""Check the primitive cost database loader, filters and CLI."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from .. import primdb
from ..primdb.__main__ import main


def _row(**flat) -> dict:
    return primdb.normalize({"family": "tc_mma", "primitive": "tcgen05.mma"} | flat)


class PrimdbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.seed = Path(self.tmp.name) / "seed.jsonl"
        self.local = Path(self.tmp.name) / "local.jsonl"
        rows = [_row(**{"p.M": 128, "p.N": n, "p.dtype": "bf16", "m.cyc_per_mma": c})
                for n, c in ((16, 54.4), (32, 54.4), (256, 128.0))]
        rows += [primdb.normalize({"family": "gemm", "primitive": "gemm.x", "p.M": m,
                                   "p.N": 4096, "p.K": 4096, "p.engine": e, "m.us": us,
                                   "c.l2": "cold"})
                 for m, e, us in ((8, "a", 10.0), (8, "b", 8.0), (64, "a", 12.0))]
        primdb.append(rows, self.seed)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *argv: str) -> str:
        out = io.StringIO()
        main(["--seed", str(self.seed), "--local", str(self.local), *argv], out)
        return out.getvalue()

    def test_normalize_routes_prefixes_and_validates(self) -> None:
        row = _row(**{"p.M": 64, "m.x_cyc": 1.0, "c.l2": "warm", "s.job": "1"})
        self.assertEqual((row["params"], row["metrics"], row["cond"], row["source"]),
                         ({"M": 64}, {"x_cyc": 1.0}, {"l2": "warm"}, {"job": "1"}))
        with self.assertRaises(ValueError):
            primdb.normalize({"family": "x", "primitive": "y"})  # a measure needs metrics
        with self.assertRaises(ValueError):
            _row(**{"m.x": float("nan")})

    def test_filters(self) -> None:
        rows = primdb.load(self.seed, self.local)
        pick = lambda **f: primdb.select(rows, "tcgen05.mma",
                                         {k: primdb.parse_filter(v) for k, v in f.items()})
        self.assertEqual(len(pick(N="16,32")), 2)
        self.assertEqual(len(pick(N="20..300")), 2)
        self.assertEqual(len(pick(dtype="bf16")), 3)
        self.assertEqual(len(primdb.select(rows, "tcgen05.")), 3)

    def test_nearest_is_log_distance(self) -> None:
        rows = primdb.select(primdb.load(self.seed, self.local), "tcgen05.mma")
        self.assertEqual(primdb.nearest(rows, {"N": 200.0}, 1)[0]["params"]["N"], 256)

    def test_add_then_query_marks_local_rows(self) -> None:
        bench = Path(self.tmp.name) / "bench.out"
        bench.write_text("noise\n" + json.dumps({"family": "tc_mma", "primitive": "tcgen05.mma",
                                                 "p.M": 64, "p.N": 8, "m.cyc_per_mma": 54.4}) + "\n")
        self.assertIn("appended 1", self._run("add", str(bench)))
        out = self._run("q", "tcgen05.mma", "M=64")
        self.assertIn("local", out)
        self.assertEqual(out.count("\n"), 2)

    def test_compare_ranks_engines_at_nearest_shape(self) -> None:
        out = self._run("compare", "gemm", "M=10", "N=4096", "K=4096")
        self.assertIn("nearest measured shape", out)
        lines = out.splitlines()
        head = next(i for i, l in enumerate(lines) if l.startswith("id "))
        col = lines[head].split().index("engine")
        engines = [l.split()[col] for l in lines[head + 1:] if l.strip() and not l.startswith("gemm")]
        self.assertEqual(engines, ["b", "a"])

    def test_seed_file_is_valid(self) -> None:
        rows = list(primdb.read(primdb.SEED, "seed"))
        self.assertTrue(rows)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
