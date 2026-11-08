import importlib.util
import unittest

from megabench.integrations import harness_model as hm

HAS_VIBESYS = importlib.util.find_spec("vibesys.orchestration") is not None \
    if importlib.util.find_spec("vibesys") else False


class TimingTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for p in (0.0, 0.3, 0.5):
            t = hm.timing_analysis(p, 2.0)
            cur, dm = t["current"], t["digest_min"]
            self.assertAlmostEqual(cur["accept"] + cur["reject"] + cur["inconclusive"], 1, 3)
            self.assertAlmostEqual(dm["accept"] + dm["reject"], 1, 3)

    def test_no_slow_mode_decides_right(self):
        self.assertEqual(hm.timing_analysis(0.0, 2.0)["current"]["accept"], 1.0)
        self.assertEqual(hm.timing_analysis(0.0, -2.0)["current"]["accept"], 0.0)

    def test_parent_all_slow_accepts_a_slower_candidate(self):
        self.assertGreater(hm.timing_analysis(0.5, -2.0)["current"]["accept"], 0.05)


class PartTests(unittest.TestCase):
    def test_gate_and_broker_have_no_new_problem(self):
        findings = hm.gate_findings() + hm.broker_findings() + hm.timing_findings()
        self.assertEqual(hm.new_problems(findings), [])

    def test_broker_clean(self):
        self.assertEqual([f.severity for f in hm.broker_findings()], ["ok"])


@unittest.skipUnless(HAS_VIBESYS, "needs an interpreter that imports vibesys")
class LoopTests(unittest.TestCase):
    def test_loop_has_no_new_problem(self):
        findings = hm.LoopModel(max_rounds=2).findings()
        self.assertEqual(hm.new_problems(findings), [])

    def test_crash_in_last_attempt_blocks_resume(self):
        r = hm.LoopModel(max_rounds=2).explore()
        self.assertTrue(r.dead)
        self.assertIn("crash after mark-paid", r.trace(r.dead[0])[-2])

    def test_no_crash_no_dead_state(self):
        r = hm.LoopModel(max_rounds=2, max_crashes=0).explore()
        self.assertEqual(r.dead, [])
        self.assertEqual(r.stuck, [])


if __name__ == "__main__":
    unittest.main()
