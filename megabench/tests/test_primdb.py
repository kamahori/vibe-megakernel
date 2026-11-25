"""Check the primitive cost database loader, filters and CLI."""

from __future__ import annotations

import io
import json
import shutil
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


class AmdSeedTests(unittest.TestCase):
    def test_seed_rows_are_valid_and_cover_the_probes(self) -> None:
        from ..primdb import amd_seed
        rows = list(primdb.read(amd_seed.SEED_AMD, "seed"))
        families = {r["family"] for r in rows}
        self.assertTrue({"sync", "mem", "mfma", "gemv", "simt"} <= families)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        for r in rows:
            self.assertIn(r["cond"].get("arch"), ("gfx90a", "gfx942", "gfx950"))
        # every MFMA and GEMV row was checked against its reference
        for r in rows:
            if r["family"] in ("mfma", "gemv"):
                self.assertIs(r["verified"], True, r["primitive"])

    def test_seed_has_the_token_sweep_on_both_gpus(self) -> None:
        from ..primdb import amd_seed
        rows = list(primdb.read(amd_seed.SEED_AMD, "seed"))
        for arch in ("gfx942", "gfx90a"):
            lds = {r["params"]["M"] for r in rows if r["cond"]["arch"] == arch
                   and r["primitive"] in ("gemv.bf16.mfma_lds_sk", "gemv.bf16.mfma_lds_db")}
            self.assertEqual(lds, {1, 4, 8, 16, 32, 64, 128}, arch)
        self.assertFalse([r for r in rows if "mfma32" in r["primitive"]])

    def test_sweep_rows_that_repeat_the_full_probe_are_dropped(self) -> None:
        from ..primdb import amd_seed
        with tempfile.TemporaryDirectory() as tmp:
            full, sweep = Path(tmp) / "probe-x-1", Path(tmp) / "msweep-x-2"
            full.mkdir(); sweep.mkdir()
            base = {"family": "gemv", "primitive": "gemv.bf16.valu_r1", "dev": "gemv_amd.hip", "c.arch": "gfx942",
                    "c.cus": 304, "p.N": 4096, "p.K": 1024, "p.M": 4, "p.threads": 256, "m.us_single": 6.3,
                    "verified": True}
            (full / "gemv_amd.jsonl").write_text(json.dumps(base) + "\n")
            lines = [dict(base, **{"p.launch_bounds": 1024}),                     # repeats the probe
                     dict(base, **{"p.launch_bounds": 512}),                      # new kernel bound
                     dict(base, **{"p.launch_bounds": 1024, "p.N": 2048, "p.K": 768}),  # new shape
                     dict(base, **{"primitive": "gemv.bf16.mfma32_lds_sk", "p.launch_bounds": 512})]
            (sweep / "gemv_msweep.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
            rows = amd_seed.drop_repeats(amd_seed.rows_from(full) + amd_seed.rows_from(sweep))
            kept = sorted((r["source"]["file"].split("/")[0], r["params"]["N"], r["params"].get("launch_bounds"))
                          for r in rows)
            self.assertEqual(kept, [("msweep-x-2", 2048, 1024), ("msweep-x-2", 4096, 512), ("probe-x-1", 4096, None)])
            self.assertTrue(all(r["source"]["bench"].endswith("gemv_amd.hip") for r in rows))

    def test_skill_tables_match_the_seed(self) -> None:
        from ..primdb import amd_seed
        with tempfile.TemporaryDirectory() as directory:
            skills = Path(directory) / "skills"
            shutil.copytree(amd_seed.SKILLS, skills)
            amd_seed.skill_tables(amd_seed.SEED_AMD, skills)
            for name in ("cdna-sync-and-memory", "cdna-matrix-core-design"):
                self.assertEqual((skills / name / "SKILL.md").read_text(),
                                 (amd_seed.SKILLS / name / "SKILL.md").read_text(),
                                 f"{name}: run python -m megabench.primdb.amd_seed --skill-tables")

    def test_rows_from_a_probe_directory(self) -> None:
        from ..primdb import amd_seed
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "probe-x-1"
            out.mkdir()
            (out / "sync_amd.jsonl").write_text(json.dumps(
                {"family": "sync", "primitive": "fence.release.agent", "c.arch": "gfx942",
                 "c.cus": 304, "p.dirty_bytes_per_block": 0, "m.ns": 900.0, "verified": None}) + "\n")
            rows = amd_seed.rows_from(out)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cond"]["arch"], "gfx942")
            self.assertEqual(rows[0]["source"]["bench"], "megabench/primbench/amd/sync_amd.hip")
            self.assertEqual(rows[0]["source"]["file"], "probe-x-1/sync_amd.jsonl")
