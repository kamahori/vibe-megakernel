"""The GPU job watchdog: a stalled harness command stops early, a busy one does not."""

from __future__ import annotations

import io
import sys
import tempfile
import time
import unittest
from pathlib import Path

from ..integrations import job_watchdog as wd

BEAT = ("import sys, time\n"
        "sys.path.insert(0, %r)\n"
        "from megabench.heartbeat import beat\n") % str(Path(__file__).resolve().parents[2])


class WatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.hb = Path(self.tmp.name) / ".gpu_run" / "1.hb"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_py(self, code: str, **kw) -> tuple[int, str, float]:
        err = io.StringIO()
        t0 = time.monotonic()
        rc = wd.run_watched([sys.executable, "-c", code], {"PATH": "/usr/bin:/bin"}, self.hb,
                            poll_s=0.2, stderr=err, **kw)
        return rc, err.getvalue(), time.monotonic() - t0

    def test_stall_after_a_beat_is_stopped(self) -> None:
        rc, err, took = self.run_py(BEAT + "beat('timed call', force=True)\ntime.sleep(60)",
                                    idle_s=1.5)
        self.assertEqual(rc, wd.HANG_EXIT)
        self.assertIn("HANG", err)
        self.assertIn("'timed call'", err)
        self.assertLess(took, 20)
        self.assertFalse(self.hb.parent.exists())  # cleaned up

    def test_longer_allowance_of_a_build_beat(self) -> None:
        rc, err, _ = self.run_py(BEAT + "beat('build', allow=4)\ntime.sleep(2.5)\n"
                                 "beat('done', force=True)", idle_s=1.0)
        self.assertEqual(rc, 0, err)

    def test_steady_beats_are_not_stopped(self) -> None:
        rc, err, _ = self.run_py(BEAT + "for i in range(8):\n    beat('call', force=True)\n"
                                 "    time.sleep(0.5)\nsys.exit(3)", idle_s=1.5)
        self.assertEqual(rc, 3, err)

    def test_no_beats_only_the_time_limit(self) -> None:
        rc, err, took = self.run_py("import time\ntime.sleep(3)", idle_s=0.5, limit_s=60)
        self.assertEqual(rc, 0, err)  # no heartbeat: the idle rule does not apply
        rc, err, took = self.run_py("import time\ntime.sleep(60)", idle_s=0.5,
                                    limit_s=wd.LIMIT_MARGIN_S + 1.5)
        self.assertEqual(rc, wd.HANG_EXIT)
        self.assertIn("time limit", err)
        self.assertLess(took, 20)


if __name__ == "__main__":
    unittest.main()
