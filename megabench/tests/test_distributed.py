"""Exercise real CPU rank communication against independent full references."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from ..cases import select_cases
from ..harness.distributed import evaluate_distributed, worst_rank_timing
from .distributed_probe import development
from .test_frontier import development as frontier_development
from .test_kimi import development as kimi_development


class DistributedReferenceTests(unittest.TestCase):
    def test_frontier_state_rollouts_match_serial_oracles(self):
        cases = select_cases('p3')
        checks = [(case,2) for case in cases] + [(case,8) for case in cases[:2]]
        for case,ranks in checks:
            with self.subTest(case=case.id,ranks=ranks), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                     f'--nproc-per-node={ranks}', '-m', 'megabench.verify_frontier',
                     '--case', case.id, '--device', 'cpu', '--trials', '2',
                     '--steps', '3', '--output', directory],
                    env=os.environ | {'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1'},
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, (result.stdout+result.stderr)[-12000:])
                for rank in range(ranks):
                    report = json.loads((Path(directory)/f'rank-{rank}.json').read_text())
                    self.assertEqual(report['status'], 'pass')
                    self.assertEqual(report['steps'], 3)
                    for trial in report['trials']:
                        self.assertEqual(len(trial['trajectory']), 3)
                        self.assertTrue(all(step['serial_comparison'] is not None
                                            for step in trial['trajectory']))

    def test_frontier_native_outputs_pass_the_submission_harness(self):
        submission = Path(__file__).parents[1] / 'examples/reference_submission.py'
        for original in select_cases('p3'):
            with self.subTest(case=original.id):
                tiny = (kimi_development(original) if original.family == 'kimi_k3_step'
                        else frontier_development(original))
                case = replace(tiny, ready=True, gpus=2, tp=2)
                report = evaluate_distributed(case, submission, device='cpu',
                                              trials=2, warmup=0, reps=1)
                self.assertEqual(report['status'], 'correctness_only', report)
                self.assertTrue(all(rank['correctness']['status'] == 'pass'
                                    for rank in report['ranks']))

    def test_tp_and_tp_ep_match_serial_oracles(self):
        for case, ranks in (("distributed-step-gemma3-27b-tp2", 2),
                            ("distributed-step-qwen3-30b-a3b-tp2-ep2", 4)):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
                result = subprocess.run(
                    [sys.executable, "-m", "torch.distributed.run", "--standalone",
                     "--nnodes=1", f"--nproc-per-node={ranks}",
                     "-m", "megabench.tests.distributed_probe", "--case", case,
                     "--trials", "2", "--output", directory],
                    env=environment, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-12000:])
                reports = [json.loads((Path(directory) / f"rank-{rank}.json").read_text())
                           for rank in range(ranks)]
                self.assertTrue(all(report["status"] == "pass" for report in reports))
                self.assertTrue(all(len(report["trials"]) == 2 for report in reports))

    def test_harness_propagates_rank_failure_and_times_slowest_rank(self):
        case = replace(development(select_cases('all', ['distributed-step-gemma3-27b-tp2'])[0]), ready=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'submission.py'
            base = (
                'from megabench.cases import Case\n'
                'from megabench.tasks.distributed import reference\n'
                'def build(contract):\n'
                '    execution = contract.pop("execution")\n'
                '    case = Case(**contract)\n'
                '    def run(inputs):\n'
                '        outputs = reference(case, inputs)\n')
            for mutation in (False, True):
                path.write_text(base +
                    ('        if execution["rank"] == 1: inputs["token"].add_(1)\n' if mutation else '') +
                    '        return outputs\n    return run\n')
                report = evaluate_distributed(case, path, device='cpu', trials=2, warmup=1, reps=2)
                self.assertEqual(report['status'], 'incorrect' if mutation else 'correctness_only')
                self.assertEqual(len(report['ranks']), 2)
                if mutation:
                    self.assertIn('rank 1', report['reason'])
                    self.assertIn('mutated input token', report['reason'])
                else:
                    self.assertTrue(all(rank['correctness']['status'] == 'pass' for rank in report['ranks']))
        ranks = [{'t': {'host_ms': [1, 5], 'cuda_event_ms': [2, 3]}},
                 {'t': {'host_ms': [4, 2], 'cuda_event_ms': [1, 6]}}]
        timing = worst_rank_timing(ranks, 't')
        self.assertEqual(timing['host_ms'], [4, 5])
        self.assertEqual(timing['cuda_event_ms'], [2, 6])


if __name__ == "__main__":
    unittest.main()
