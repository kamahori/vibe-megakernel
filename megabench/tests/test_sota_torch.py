"""CPU tests for the graph-safe torch Qwen3 decode step (``sota/qwen3_torch.py``)."""

from __future__ import annotations

import dataclasses
import unittest

import torch

from ..cases import select_cases
from ..sota import qwen3_torch as qt
from ..sota.shapes import BASE_CASE_ID, ModelGeometry, Shape
from ..tasks.dense import make_inputs, reference

GEOM = ModelGeometry("tiny", hidden=64, layers=2, q_heads=4, kv_heads=2,
                     head_dim=16, intermediate=128, vocab=256)
CONTEXT = 8
SHAPE = Shape(1, CONTEXT, GEOM)
CASE = dataclasses.replace(
    select_cases("all", [BASE_CASE_ID])[0], id="dense-step-tiny-b1-s8",
    params=dict(batch=1, context=CONTEXT, layers=2, hidden=64, q_heads=4,
                kv_heads=2, head_dim=16, intermediate=128, vocab=256))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def run(inputs: dict, dtype, **kw):
    state = qt.build_state(SHAPE, inputs, dtype, "cpu")
    with torch.no_grad():
        return qt.decode_step(qt.weights_for(inputs, dtype), state, dtype, **kw), state


def float64_step(inputs: dict, ctx: int) -> dict:
    """Independent single-sequence float64 decode (plain loops, no shared code)."""
    g = GEOM
    w = {k: v.double() for k, v in inputs.items() if k not in ("token",)}
    pos = ctx
    inv = g.rope_theta ** (-torch.arange(0, g.head_dim, 2, dtype=torch.float64) / g.head_dim)
    ang = pos * inv
    cos, sin = torch.cat([ang, ang]).cos(), torch.cat([ang, ang]).sin()

    def norm(x, wt):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + g.eps) * wt

    def rot(x):
        h = x.shape[-1] // 2
        return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin

    x = w["embed"][int(inputs["token"])]
    ks, vs = [], []
    for l in range(g.layers):
        xn = norm(x, w["ln1"][l])
        q = rot(norm((w["wq"][l] @ xn).view(g.q_heads, g.head_dim), w["qn"][l]))
        k = rot(norm((w["wk"][l] @ xn).view(g.kv_heads, g.head_dim), w["kn"][l]))
        v = (w["wv"][l] @ xn).view(g.kv_heads, g.head_dim)
        ks.append(k)
        vs.append(v)
        kk = torch.cat([w["kcache"][l], k[None]])
        vv = torch.cat([w["vcache"][l], v[None]])
        o = torch.empty(g.q_heads, g.head_dim, dtype=torch.float64)
        for h in range(g.q_heads):
            kvh = h // (g.q_heads // g.kv_heads)
            p = torch.softmax(kk[:, kvh] @ q[h] * g.head_dim ** -0.5, 0)
            o[h] = p @ vv[:, kvh]
        x = x + w["wo"][l] @ o.reshape(-1)
        xn = norm(x, w["ln2"][l])
        x = x + w["wd"][l] @ (torch.nn.functional.silu(w["wg"][l] @ xn) * (w["wu"][l] @ xn))
    logits = w["embed"] @ norm(x, w["fnorm"])
    return {"logits": logits, "k": torch.stack(ks), "v": torch.stack(vs)}


class Qwen3TorchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = make_inputs(CASE, 7301, "cpu")

    def test_bf16_matches_megabench_reference(self) -> None:
        for seed in (1, 2, 3):
            inputs = make_inputs(CASE, seed, "cpu")
            ref = reference(CASE, inputs)
            out, _ = run(inputs, torch.bfloat16)
            rel = rel_l2(out["logits"][0], ref["logits"])
            print(f"bf16 vs reference seed {seed}: logits rel-L2 {rel:.3e}")
            self.assertLessEqual(rel, 1e-3)
            self.assertEqual(out["logits"].dtype, torch.float32)
            self.assertEqual(int(out["next_token"][0]), int(ref["next_token"]))
            for k in ("k_write", "v_write"):
                diff = (out[k][0].float() - ref[k].float()).abs().max().item()
                print(f"  {k} max-abs diff {diff:.3e}")
                self.assertLessEqual(diff, 2 ** -7 * ref[k].float().abs().max().item())

    def test_fp32_matches_float64(self) -> None:
        ref = float64_step(self.inputs, CONTEXT)
        out, _ = run(self.inputs, torch.float32)
        rel = rel_l2(out["logits"][0], ref["logits"])
        print(f"fp32 vs float64: logits rel-L2 {rel:.3e}, "
              f"k rel-L2 {rel_l2(out['k_write'][0], ref['k']):.3e}")
        self.assertLessEqual(rel, 1e-5)
        self.assertEqual(int(out["next_token"][0]), int(ref["logits"].argmax()))
        self.assertLessEqual(rel_l2(out["k_write"][0], ref["k"]), 1e-5)
        self.assertLessEqual(rel_l2(out["v_write"][0], ref["v"]), 1e-5)
        bf, _ = run(self.inputs, torch.bfloat16)
        gap = rel_l2(bf["logits"][0], out["logits"][0])
        print(f"bf16 vs fp32: logits rel-L2 {gap:.3e}")
        self.assertGreater(gap, 1e-4)
        self.assertLess(gap, 0.1)

    def test_chain_updates_token_and_is_reproducible(self) -> None:
        w = qt.weights_for(self.inputs, torch.float32)
        state = qt.build_state(SHAPE, self.inputs, torch.float32, "cpu")
        before = self.inputs["token"].clone()
        with torch.no_grad():
            first = qt.decode_step(w, state, torch.float32, chain=True)
            self.assertTrue(torch.equal(state.token, first["next_token"]))
            second = qt.decode_step(w, state, torch.float32)
        fresh_inputs = dict(self.inputs, token=first["next_token"][0].clone())
        fresh, _ = run(fresh_inputs, torch.float32)
        self.assertTrue(torch.equal(second["logits"], fresh["logits"]))
        self.assertTrue(torch.equal(self.inputs["token"], before))

    def test_batch_two_with_different_positions(self) -> None:
        dtype = torch.bfloat16
        w = qt.weights_for(self.inputs, dtype)
        s_max = CONTEXT + 1
        solo = []
        for p in (5, 8):
            shape = Shape(1, p, GEOM)
            ins = dict(self.inputs, kcache=self.inputs["kcache"][:, :p],
                       vcache=self.inputs["vcache"][:, :p])
            st = qt.build_state(shape, ins, dtype, "cpu", s_max=s_max)
            with torch.no_grad():
                solo.append(qt.decode_step(w, st, dtype))
        state = qt.build_state(Shape(2, CONTEXT, GEOM), self.inputs, dtype, "cpu")
        state.pos = torch.tensor([5, 8])
        state.token = torch.tensor([int(self.inputs["token"])] * 2)
        with torch.no_grad():
            out = qt.decode_step(w, state, dtype)
        for b in range(2):
            for k in out:
                self.assertTrue(torch.allclose(out[k][b].float(), solo[b][k][0].float(),
                                               atol=0, rtol=0) or k == "logits"
                                and rel_l2(out[k][b], solo[b][k][0]) < 1e-3, k)
        self.assertEqual(out["k_write"].shape, (2, 2, 2, 16))

    def test_inputs_not_mutated_and_eager_compile_smoke(self) -> None:
        before = {k: v.clone() for k, v in self.inputs.items()}
        out, _ = run(self.inputs, torch.bfloat16)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in self.inputs.items()))
        w = qt.weights_for(self.inputs, torch.bfloat16)
        state = qt.build_state(SHAPE, self.inputs, torch.bfloat16, "cpu")
        fn = torch.compile(lambda: qt.decode_step(w, state, torch.bfloat16, True),
                           backend="eager", fullgraph=True)
        with torch.no_grad():
            got = fn()
        self.assertTrue(torch.equal(got["logits"], out["logits"]))


if __name__ == "__main__":
    unittest.main()
