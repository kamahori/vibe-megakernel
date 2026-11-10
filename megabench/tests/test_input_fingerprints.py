import unittest

import torch

from megabench.harness.correctness import input_fingerprints, mutated_inputs


class InputFingerprintTests(unittest.TestCase):
    def inputs(self):
        g = torch.Generator().manual_seed(1)
        return {"w": torch.randn(1000, 37, generator=g).to(torch.bfloat16),
                "ids": torch.arange(13, dtype=torch.int64),
                "mask": torch.zeros(5, dtype=torch.bool),
                "odd": torch.arange(7, dtype=torch.uint8),
                "empty": torch.empty(0), "scale": 2.5}

    def test_unchanged_inputs_pass(self):
        x = self.inputs()
        before = input_fingerprints(x)
        self.assertEqual(mutated_inputs(x, before), [])

    def test_one_changed_word_is_found(self):
        for name, change in [("w", lambda t: t.view(torch.int16)[123, 4].add_(1)),
                             ("ids", lambda t: t[3].add_(1)),
                             ("mask", lambda t: t[4].fill_(True)),
                             ("odd", lambda t: t[6].add_(1))]:
            x = self.inputs()
            before = input_fingerprints(x)
            change(x[name])
            self.assertEqual(mutated_inputs(x, before), [name])

    def test_bits_not_values(self):
        x = {"z": torch.zeros(4), "n": torch.full((3,), float("nan"))}
        before = input_fingerprints(x)
        self.assertEqual(mutated_inputs(x, before), [])  # NaN inputs are not a mutation
        x["z"][2] = -0.0
        self.assertEqual(mutated_inputs(x, before), ["z"])

    def test_swapped_words_and_new_shape_are_found(self):
        x = {"a": torch.arange(16, dtype=torch.float32), "b": torch.ones(4, 4)}
        before = input_fingerprints(x)
        x["a"][[0, 1]] = x["a"][[1, 0]]
        x["b"] = x["b"].t()
        self.assertEqual(sorted(mutated_inputs(x, before)), ["a", "b"])

    def test_chunked_hash_equals_one_pass(self):
        import megabench.harness.correctness as c
        x = {"w": torch.randn(4099)}
        key, one = input_fingerprints(x)
        old, c._PRINT_CHUNK = c._PRINT_CHUNK, 1000
        try:
            self.assertEqual(input_fingerprints(x, key)[1], one)
        finally:
            c._PRINT_CHUNK = old


if __name__ == "__main__":
    unittest.main()
