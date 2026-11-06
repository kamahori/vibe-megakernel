"""mb_build: cache by sources and flags, and a compile time limit."""

import os
import tempfile
import time
import unittest
from pathlib import Path

from ..integrations.mb_build import build


def fake_nvcc(tmp: Path, sleep: float = 0.0) -> str:
    """A compiler that counts its runs and writes the -o file."""
    script = tmp / "nvcc"
    script.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --version ]; then echo fake 1.0; exit 0; fi\n"
        # -M: the make rule of each .cu input and the local headers it includes.
        "case \" $* \" in *\" -M \"*)\n"
        "  for a in \"$@\"; do case \"$a\" in *.cu) d=$(dirname \"$a\");\n"
        "    printf 'x.o: %s' \"$a\";\n"
        "    for h in $(sed -n 's/#include \"\\(.*\\)\"/\\1/p' \"$a\"); do printf ' \\\\\\n %s' \"$d/$h\"; done;\n"
        "    echo;; esac; done; exit 0;; esac\n"
        f"echo run >> {tmp}/runs\n"
        f"sleep {sleep}\n"
        # The output holds the contents of its inputs, so an object changes with its source.
        "out=; ins=; for a in \"$@\"; do [ \"$p\" = -o ] && out=$a;\n"
        "  case \"$a\" in *.cu|*.o) [ \"$p\" = -o ] || ins=\"$ins $a\";; esac; p=$a; done\n"
        "cat $ins > \"$out.tmp\"; echo bin >> \"$out.tmp\"; mv \"$out.tmp\" \"$out\"\n")
    script.chmod(0o755)
    return str(script)


def runs(tmp: Path) -> int:
    f = tmp / "runs"
    return len(f.read_text().splitlines()) if f.exists() else 0


class MbBuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        (self.src / "a.cu").write_text('#include "common.cuh"\nint main(){}\n')
        (self.src / "common.cuh").write_text("// v1\n")
        self.cache = self.tmp / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def args(self, *extra):
        return [str(self.src / "a.cu"), *extra, "-o", str(self.tmp / "a.out")]

    def test_same_sources_and_flags_do_not_run_nvcc_again(self):
        nvcc = fake_nvcc(self.tmp)
        self.assertEqual(build(self.args(), self.cache, 30, nvcc), 0)
        (self.tmp / "a.out").unlink()
        self.assertEqual(build(self.args(), self.cache, 30, nvcc), 0)
        self.assertEqual(runs(self.tmp), 1)
        self.assertTrue((self.tmp / "a.out").is_file())

    def test_a_changed_header_or_flag_builds_again(self):
        nvcc = fake_nvcc(self.tmp)
        build(self.args(), self.cache, 30, nvcc)
        (self.src / "common.cuh").write_text("// v2\n")
        build(self.args(), self.cache, 30, nvcc)
        build(self.args("-DSET=2"), self.cache, 30, nvcc)
        self.assertEqual(runs(self.tmp), 3)

    def test_a_compile_past_the_limit_is_stopped_and_not_cached(self):
        nvcc = fake_nvcc(self.tmp, sleep=30)
        start = time.monotonic()
        self.assertEqual(build(self.args(), self.cache, 1, nvcc), 124)
        self.assertLess(time.monotonic() - start, 10)
        self.assertFalse(self.cache.exists() and any(self.cache.rglob("a.out")))

    def test_an_edit_to_one_unit_rebuilds_only_that_unit_and_the_link(self):
        nvcc = fake_nvcc(self.tmp)
        (self.src / "a.cu").write_text('#include "common.cuh"\nint a;\n')
        (self.src / "b.cu").write_text('#include "common.cuh"\nint b;\n')
        (self.src / "other.cuh").write_text("// not included\n")
        a_o, b_o, exe = (str(self.tmp / n) for n in ("a.o", "b.o", "bench"))

        def all_steps():
            build(["-c", str(self.src / "a.cu"), "-o", a_o], self.cache, 30, nvcc)
            build(["-c", str(self.src / "b.cu"), "-o", b_o], self.cache, 30, nvcc)
            build([a_o, b_o, "-o", exe], self.cache, 30, nvcc)

        all_steps()
        self.assertEqual(runs(self.tmp), 3)
        all_steps()  # nothing changed
        self.assertEqual(runs(self.tmp), 3)
        (self.src / "other.cuh").write_text("// changed, still not included\n")
        all_steps()
        self.assertEqual(runs(self.tmp), 3)
        (self.src / "b.cu").write_text('#include "common.cuh"\nint b2;\n')
        all_steps()  # b.o and the link
        self.assertEqual(runs(self.tmp), 5)
        (self.src / "common.cuh").write_text("// v2\n")
        all_steps()  # both units; their objects are the same bytes, so the link is a hit
        self.assertEqual(runs(self.tmp), 7)

    def test_without_an_output_it_refuses(self):
        self.assertEqual(build([str(self.src / "a.cu")], self.cache, 30,
                               fake_nvcc(self.tmp)), 2)


if __name__ == "__main__":
    unittest.main()
