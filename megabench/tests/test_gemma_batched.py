"""Batched and prefill Gemma 3 quantized steps and their floors."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import torch

from .. import baselines
from ..baselines import build
from ..cases import CASES, Case
from ..harness.correctness import _compare
from ..sol import estimate_case
from ..tasks import gemma
from ..tasks.oracle import fp32_reference

BASE = next(c for c in CASES if c.id == "quant-step-gemma3-4b-w8-b1-s128")
# Six layers give five sliding-window layers and one global layer.
TINY = {"batch": 1, "context": 8, "layers": 6, "hidden": 64, "q_heads": 4,
        "kv_heads": 2, "head_dim": 16, "intermediate": 96, "vocab": 200,
        "bits": 8, "local_window": 4}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(**changes):
    params = TINY | changes.pop("params", {})
    return replace(BASE, id="tiny", params=params, **changes)


class GemmaBatchedTests(unittest.TestCase):
    def test_batch_one_layout_unchanged(self) -> None:
        t = gemma.make_inputs(tiny(), 0, DEVICE)
        self.assertEqual(t["token"].shape, ())
        self.assertEqual(t["kcache"].shape, (6, 8, 2, 16))
        self.assertNotIn("context_lens", t)
        self.assertEqual(gemma.reference(tiny(), t)["k_write"].shape, (6, 2, 16))

    def test_batched_layout(self) -> None:
        case = tiny(params={"batch": 3}, contexts=(2, 8, 5))
        t = gemma.make_inputs(case, 0, DEVICE)
        self.assertEqual(t["token"].shape, (3,))
        self.assertEqual(t["kcache"].shape, (6, 3, 8, 2, 16))
        self.assertEqual(t["context_lens"].tolist(), [2, 8, 5])
        out = gemma.reference(case, t)
        self.assertEqual(out["logits"].shape, (3, 200))
        self.assertEqual(out["next_token"].shape, (3,))
        self.assertEqual(out["k_write"].shape, (6, 3, 2, 16))
        uniform = gemma.make_inputs(tiny(params={"batch": 2}), 0, DEVICE)
        self.assertEqual(uniform["context_lens"].tolist(), [8, 8])

    def test_batched_matches_each_sequence_alone(self) -> None:
        for bits in (8, 4):
            with self.subTest(bits=bits):
                case = tiny(params={"batch": 3, "bits": bits}, contexts=(2, 8, 5))
                t = gemma.make_inputs(case, 1, DEVICE)
                original = {k: v.clone() for k, v in t.items()}
                got = fp32_reference(gemma, case, t)
                for name, value in original.items():
                    self.assertTrue(torch.equal(t[name], value), name)
                for b, length in enumerate(case.contexts):
                    one = tiny(params={"context": length, "bits": bits})
                    single = {k: v for k, v in t.items()
                              if k not in ("token", "kcache", "vcache", "context_lens")}
                    single["token"] = t["token"][b]
                    single["kcache"] = t["kcache"][:, b, :length].contiguous()
                    single["vcache"] = t["vcache"][:, b, :length].contiguous()
                    want = fp32_reference(gemma, one, single)
                    torch.testing.assert_close(got["logits"][b], want["logits"],
                                               rtol=1e-4, atol=1e-4)
                    self.assertEqual(int(got["next_token"][b]), int(want["next_token"]))
                    for key in ("k_write", "v_write"):
                        torch.testing.assert_close(got[key][:, b].float(), want[key].float(),
                                                   rtol=1e-2, atol=1e-2)

    def test_batched_floor_counts_each_sequence_cache(self) -> None:
        case = replace(BASE, params=BASE.params | {"batch": 3, "context": 4096},
                       contexts=(100, 4096, 2000))
        floor = {c["name"]: c["bytes"] for c in estimate_case(case)["components"]}
        p = case.params
        kdim, window = p["kv_heads"] * p["head_dim"], p["local_window"] - 1
        local, glob = p["layers"] - p["layers"] // 6, p["layers"] // 6
        positions = sum(local * min(n, window) + glob * n for n in case.contexts)
        self.assertEqual(floor["old_kv_cache_reads"], 2 * positions * kdim * 2)
        single = {c["name"]: c["bytes"] for c in estimate_case(BASE)["components"]}
        self.assertEqual(floor["quantized_projection_and_mlp_weights"],
                         single["quantized_projection_and_mlp_weights"])
        self.assertEqual(floor["new_kv_writes"], 3 * single["new_kv_writes"])


    def test_prefill_matches_successive_decode_steps(self) -> None:
        # Seven new tokens after three cached ones cross the window of four.
        case = tiny(params={"context": 3, "tokens": 7})
        t = gemma.make_inputs(case, 2, DEVICE)
        self.assertEqual(t["tokens"].shape, (7,))
        self.assertNotIn("token", t)
        got = fp32_reference(gemma, case, t)
        self.assertEqual(got["logits"].shape, (200,))
        self.assertEqual(got["k_write"].shape, (6, 7, 2, 16))
        kcache, vcache = t["kcache"], t["vcache"]
        weights = {k: v for k, v in t.items() if k not in ("tokens", "kcache", "vcache")}
        for i in range(7):
            one = tiny(params={"context": 3 + i})
            want = fp32_reference(gemma, one, weights | {
                "token": t["tokens"][i], "kcache": kcache, "vcache": vcache})
            for key in ("k_write", "v_write"):
                torch.testing.assert_close(got[key][:, i].float(), want[key].float(),
                                           rtol=1e-2, atol=1e-2)
            kcache = torch.cat((kcache, got["k_write"][:, i:i + 1].to(kcache.dtype)), 1)
            vcache = torch.cat((vcache, got["v_write"][:, i:i + 1].to(vcache.dtype)), 1)
        # The decode steps attend to BF16-rounded cached keys, so allow that rounding.
        torch.testing.assert_close(got["logits"], want["logits"], rtol=1e-2, atol=1e-2)
        self.assertEqual(int(got["next_token"]), int(got["logits"].argmax()))

    def test_prefill_floor_is_compute_bound(self) -> None:
        case = next(c for c in CASES if c.id == "quant-step-gemma3-4b-w8-prefill-t2048-c2048")
        floor = estimate_case(case)
        self.assertEqual(floor["bound"], "tensor_core")
        p = case.params
        weights = p["layers"] * (2 * 2048 * 2560 + 2 * 1024 * 2560 + 3 * 10240 * 2560)
        flops = {f["name"]: f["flops"] for f in floor["flop_components"]}
        self.assertEqual(flops["projections_and_mlp"], 2 * p["tokens"] * weights)
        decode = {c["name"]: c["bytes"] for c in estimate_case(BASE)["components"]}
        bytes_ = {c["name"]: c["bytes"] for c in floor["components"]}
        self.assertEqual(bytes_["quantized_projection_and_mlp_weights"],
                         decode["quantized_projection_and_mlp_weights"])


    def test_capture_safe_baseline_matches_batched_and_prefill(self) -> None:
        for case in (tiny(params={"batch": 3}, contexts=(2, 8, 5)),
                     tiny(params={"context": 3, "tokens": 7})):
            # FlashInfer has no kernels for the tiny head size; the CPU path
            # checks the projections, positions and output layout.
            with self.subTest(case=case.params), torch.inference_mode():
                t = gemma.make_inputs(case, 4, "cpu")
                expected = gemma.reference(case, t)
                actual = build(case, "cpu")(t)
                _compare(expected, actual, case, "cpu", fp32_reference(gemma, case, t))
                self.assertTrue(torch.equal(actual["next_token"], expected["next_token"]))

    def test_case_baselines_round_trip(self) -> None:
        case = tiny(baselines=["capture_safe_graph"])
        self.assertEqual(case.baselines, ("capture_safe_graph",))
        self.assertEqual(Case(**case.to_dict()), case)
        self.assertNotIn("baselines", tiny().to_dict())
        with self.assertRaises(ValueError):
            tiny(baselines=("vllm",))


    def test_baseline_timing_is_reused_for_the_same_contract(self) -> None:
        calls = []

        def fake(case, *args, **kwargs):
            calls.append(case.id)
            return {"variants": {}, "best": {"variant": "capture_safe_graph",
                                             "cuda_event_p50_ms": 1.0}}

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {"MEGABENCH_BASELINE_CACHE": directory}), \
                mock.patch.object(baselines, "timing_baselines", fake), \
                mock.patch.object(torch.cuda, "get_device_name", lambda *_: "fake"):
            run = lambda case: baselines.cached_timing_baselines(
                case, {}, "cuda:0", 1, 3, expected={}, exact={}, variants=("capture_safe_graph",))
            first, second = run(tiny()), run(tiny())
            self.assertNotIn("cached_from", first)
            self.assertEqual(second["best"], first["best"])
            self.assertIn("cached_from", second)
            run(tiny(params={"context": 9}))
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
