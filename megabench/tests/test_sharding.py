"""Trainium TP4 cases: catalog selection and the tensor-parallel layout."""

import unittest
from dataclasses import replace

import torch

from megabench.cases import select_cases
from megabench.tasks.sharding import assemble, check_divisible, input_dim, shard, shard_outputs
from megabench.tasks.workloads import make_inputs, reference


def development(case):
    return replace(case, params=case.params | {"layers": 2})


class ShardingTests(unittest.TestCase):
    def test_neuron_cases_stay_out_of_default_suites(self):
        cuda = select_cases("p0")
        neuron = select_cases("p0", backend="neuron")
        self.assertTrue(cuda and all(case.backend == "cuda" for case in cuda))
        self.assertEqual([case.id for case in neuron],
                         [f"{case.id}-trn2-tp4" for case in cuda if case.family != "moe_step"])
        for case in neuron:
            self.assertEqual((case.gpus, case.tp, case.max_gpu_launches), (4, 4, 1))

    def test_shards_partition_inputs_and_outputs(self):
        for case in select_cases("p0", backend="neuron"):
            with self.subTest(case=case.id):
                case = development(case)
                single = replace(case, gpus=1, tp=1, backend="cuda")
                values = make_inputs(single, 3, "cpu")
                check_divisible(case, values, 4)
                shards = [shard(case, values, rank, 4) for rank in range(4)]
                for name, value in values.items():
                    dim = input_dim(case, name)
                    parts = [part[name] for part in shards]
                    if dim is None:
                        self.assertTrue(all(part is value for part in parts), name)
                    else:
                        self.assertTrue(torch.equal(torch.cat(parts, dim), value), name)
                outputs = reference(single, values)
                rebuilt = assemble([shard_outputs(outputs, rank, 4) for rank in range(4)])
                self.assertEqual(set(rebuilt), set(outputs))
                for name in outputs:
                    self.assertTrue(torch.equal(rebuilt[name], outputs[name]), name)

    def test_replicated_outputs_must_agree(self):
        ranks = [{"next_token": torch.tensor(5), "logits": torch.zeros(2)} for _ in range(4)]
        ranks[2]["next_token"] = torch.tensor(6)
        with self.assertRaises(AssertionError):
            assemble(ranks)


if __name__ == "__main__":
    unittest.main()
