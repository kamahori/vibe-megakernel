"""Waypoint-1.5 world-model frame step on a development geometry (CPU)."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ..cases import select_cases
from ..harness.runner import evaluate_case
from ..tasks import waypoint
from ..tasks.oracle import fp32_reference

CASE_ID = "world-frame-step-waypoint15-1b-f128"
REFERENCE_SUBMISSION = Path(__file__).resolve().parents[1] / "examples" / "reference_submission.py"


def full_case():
    case = select_cases("all", [CASE_ID])[0]
    # The latent is the 16x32 patch grid times the 2x2 patch.
    return replace(case, ready=True, params=case.params | {"latent_height": 32, "latent_width": 64})


def development(frame_index: int = 8):
    """128 tokens per frame (one flex block). Of the four layers, 1 and 3 are
    global and 0 and 3 fuse the controller. The local ring has 3 slots and
    the global ring has 8 / 2 = 4."""
    return replace(full_case(), params=full_case().params | {
        "frame_index": frame_index, "layers": 4, "hidden": 64, "q_heads": 2,
        "kv_heads": 1, "head_dim": 32, "mlp_ratio": 2, "latent_channels": 4,
        "latent_height": 16, "latent_width": 32, "tokens_per_frame": 128,
        "local_window": 3, "global_window": 8, "global_period": 2,
        "global_dilation": 2, "buttons": 8})


def changed(a: dict, b: dict) -> bool:
    return any(not torch.equal(a[name], b[name]) for name in a)


class WaypointGeometryTest(unittest.TestCase):
    def test_full_layer_roles_and_windows(self) -> None:
        case = full_case()
        self.assertEqual(waypoint.geometry(case), {"grid_h": 16, "grid_w": 32, "slots": 16})
        layers = range(case.params["layers"])
        self.assertEqual([l for l in layers if waypoint.is_global(case, l)], [3, 7, 11, 15, 19, 23])
        self.assertEqual(waypoint.control_layers(case), [0, 3, 6, 9, 12, 15, 18, 21])
        local = waypoint.slot_frames(case, 0, 128)
        self.assertEqual(local, list(range(112, 128)))
        self.assertEqual([local[s] for s in waypoint.visible_slots(case, 0, 128)], list(range(113, 128)))
        glob = waypoint.slot_frames(case, 3, 128)
        self.assertEqual(glob, list(range(0, 128, 8)))
        self.assertEqual([glob[s] for s in waypoint.visible_slots(case, 3, 128)], list(range(8, 128, 8)))
        self.assertEqual((waypoint.write_slot(case, 0, 128), waypoint.write_slot(case, 3, 128)), (0, 0))
        # Off the dilation grid, global layers hide nothing and persist nothing.
        self.assertIsNone(waypoint.write_slot(case, 3, 129))
        self.assertEqual(len(waypoint.visible_slots(case, 3, 129)), 16)
        self.assertEqual(waypoint.slot_frames(case, 3, 129)[0], 128)

    def test_registered_latent_grid_is_validated(self) -> None:
        case = full_case()
        bad = replace(case, params=case.params | {"latent_height": 16, "latent_width": 32})
        with self.assertRaises(ValueError):
            waypoint.geometry(bad)

    def test_step_flops(self) -> None:
        flops = waypoint.step_flops(full_case())
        self.assertGreater(flops["total"], 9e12)
        self.assertLess(flops["total"], 11e12)


class WaypointReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = development()
        cls.values = waypoint.make_inputs(cls.case, 7, "cpu")
        cls.out = waypoint.reference(cls.case, cls.values)

    def run_with(self, case=None, **updates):
        return waypoint.reference(case or self.case, self.values | updates)

    def test_shapes_dtypes_and_finite(self) -> None:
        p = self.case.params
        self.assertEqual(self.out["latent"].shape, (p["latent_channels"], p["latent_height"], p["latent_width"]))
        kv = (p["layers"], p["tokens_per_frame"], p["kv_heads"], p["head_dim"])
        self.assertEqual(self.out["k_write"].shape, kv)
        self.assertEqual(self.out["v_write"].shape, kv)
        for value in self.out.values():
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertEqual(self.values["k_cache"].shape, (p["layers"], 4, p["tokens_per_frame"],
                                                         p["kv_heads"], p["head_dim"]))

    def test_deterministic(self) -> None:
        again = waypoint.make_inputs(self.case, 7, "cpu")
        self.assertFalse(changed({k: self.values[k] for k in again}, again))
        self.assertFalse(changed(self.out, waypoint.reference(self.case, again)))
        self.assertTrue(changed({"noise": self.values["noise"]},
                                {"noise": waypoint.make_inputs(self.case, 8, "cpu")["noise"]}))

    def test_does_not_mutate_inputs(self) -> None:
        snapshot = {k: v.clone() for k, v in self.values.items()}
        waypoint.reference(self.case, self.values)
        fp32_reference(waypoint, self.case, self.values)
        self.assertFalse(changed(snapshot, self.values))

    def test_depends_on_noise_controller_and_cache(self) -> None:
        for name in ("noise", "mouse", "scroll", "k_cache", "v_cache"):
            with self.subTest(name=name):
                value = self.values[name]
                bumped = value + 0.5 if name != "scroll" else -value + (value == 0)
                self.assertTrue(changed(self.out, self.run_with(**{name: bumped})))
        button = self.values["button"].clone()
        button[0] = 1 - button[0]
        self.assertTrue(changed(self.out, self.run_with(button=button)))
        # Each visible slot matters: local layer 0 slot 0, global layer 1 slot 1.
        for layer, slot in ((0, 0), (1, 1)):
            k = self.values["k_cache"].clone()
            k[layer, slot] = -k[layer, slot]
            self.assertTrue(changed(self.out, self.run_with(k_cache=k)))

    def test_hidden_and_unwritten_slots_do_not_matter(self) -> None:
        # f = 8: local ring slot 8 % 3 = 2 (frame 5) and global slot
        # (8 / 2) % 4 = 0 (frame 0) are being replaced. Local slot 3 is
        # outside the 3-slot ring.
        k, v = self.values["k_cache"].clone(), self.values["v_cache"].clone()
        for layer, slot in ((0, 2), (2, 2), (0, 3), (2, 3), (1, 0), (3, 0)):
            k[layer, slot] = 100.0
            v[layer, slot] = -100.0
        self.assertFalse(changed(self.out, self.run_with(k_cache=k, v_cache=v)))

    def test_early_frame_sees_only_written_history(self) -> None:
        case = development(frame_index=2)
        values = waypoint.make_inputs(case, 3, "cpu")
        out = waypoint.reference(case, values)
        # f = 2: local slots 0, 1 hold frames 0, 1; global slot 0 holds frame 0.
        self.assertEqual(waypoint.visible_slots(case, 0, 2), [0, 1])
        self.assertEqual(waypoint.visible_slots(case, 1, 2), [0])
        for layer, slot in ((0, 2), (1, 1), (1, 2), (1, 3)):
            self.assertFalse(bool(values["k_cache"][layer, slot].any()))
            k = values["k_cache"].clone()
            k[layer, slot] = 50.0
            self.assertFalse(changed(out, waypoint.reference(case, values | {"k_cache": k})))

    def test_frame_index_changes_rope_and_windows(self) -> None:
        other = replace(self.case, params=self.case.params | {"frame_index": 9})
        self.assertTrue(changed(self.out, waypoint.reference(other, self.values)))

    def test_cache_fixture_matches_committed_frames(self) -> None:
        # Rebuilding seven frames, then committing frame 7's K/V, must
        # reproduce the eight-frame fixture's slots for frame 7.
        case7 = development(frame_index=7)
        gen = torch.Generator().manual_seed(7)
        weights = waypoint.make_weights(case7, gen, "cpu")
        k7, v7 = waypoint.build_cache(case7, weights, gen, "cpu", 7)
        latent = torch.randn((4, 16, 32), generator=gen).to(torch.bfloat16)
        ctrl = waypoint._control_embedding(weights | waypoint._random_controller(gen, case7, "cpu"))
        _, ks, vs = waypoint._forward(case7, weights, latent, 0.0, 7, ctrl, k7, v7, kv_only=True)
        local_slot = waypoint.write_slot(case7, 0, 7)
        torch.testing.assert_close(self.values["k_cache"][0, local_slot], ks[0], rtol=0, atol=0)
        torch.testing.assert_close(self.values["v_cache"][2, local_slot], vs[2], rtol=0, atol=0)
        self.assertIsNone(waypoint.write_slot(case7, 1, 7))

    def test_reference_is_within_its_oracle_band(self) -> None:
        exact = fp32_reference(waypoint, self.case, self.values)
        for name, value in self.out.items():
            error = (value.double() - exact[name].double()).norm() / exact[name].double().norm()
            self.assertLess(float(error), 0.03, name)
            self.assertEqual(exact[name].dtype, torch.float32)

    def test_reference_submission_passes_harness_on_cpu(self) -> None:
        result = evaluate_case(self.case, REFERENCE_SUBMISSION, device="cpu",
                               trials=2, warmup=0, reps=1)
        self.assertEqual(result["status"], "correctness_only", result)
        self.assertEqual(len(result["correctness"]["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
