import os
import tempfile
import unittest
from pathlib import Path

from megabench.integrations.gpu_broker import Broker, _bytes


def broker_with(sacct_output: str, directory: str) -> Broker:
    fake = Path(directory) / "sacct"
    fake.write_text("#!/bin/sh\ncat <<'END'\n" + sacct_output + "END\n")
    fake.chmod(0o755)
    b = Broker.__new__(Broker)
    b.sacct, b.env, b.log = str(fake), dict(os.environ), lambda message: None
    return b


class OomReportTests(unittest.TestCase):
    def read(self, b: Broker, name: str, jobid=None):
        r, w = os.pipe()
        hit = b.oom_report(name, w, jobid=jobid, tries=1)
        os.close(w)
        text = os.read(r, 4096).decode()
        os.close(r)
        return hit, text

    def test_out_of_memory_step_is_reported_with_limit_and_peak(self):
        with tempfile.TemporaryDirectory() as d:
            b = broker_with("vs-gpu-x-5|OUT_OF_MEMORY|48254344K|64G\n"
                            "vs-gpu-x-6|COMPLETED|1000K|64G\n", d)
            hit, text = self.read(b, "vs-gpu-x-5", jobid="1")
        self.assertTrue(hit)
        self.assertIn("OUT OF MEMORY", text)
        self.assertIn("64G", text)
        self.assertIn("48254344K", text)

    def test_other_failures_and_other_steps_are_not_reported(self):
        with tempfile.TemporaryDirectory() as d:
            b = broker_with("vs-gpu-x-5|FAILED|1000K|64G\n"
                            "vs-gpu-x-6|OUT_OF_MEMORY|9K|64G\n", d)
            self.assertEqual(self.read(b, "vs-gpu-x-5", jobid="1"), (False, ""))

    def test_memory_values(self):
        self.assertEqual(_bytes("64G"), 64 * 1024 ** 3)
        self.assertEqual(_bytes("1000K"), 1024000)
        self.assertEqual(_bytes("x"), 0)


if __name__ == "__main__":
    unittest.main()
