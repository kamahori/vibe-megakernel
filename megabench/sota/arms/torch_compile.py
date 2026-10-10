"""torch.compile (Inductor max-autotune) arm for the pure-torch Qwen3 decode step.

Variants:
  ``max-autotune``              ``torch.compile(mode="max-autotune")``: Inductor with
                                CUDA-graph trees; ``launch`` calls the compiled module.
  ``max-autotune-manualgraph``  ``mode="max-autotune-no-cudagraphs"``, then the compiled
                                callable is captured by hand in ``torch.cuda.CUDAGraph``
                                (one graph unchained, one chained); ``launch`` = replay.

Weights, KV cache, token, position and RoPE tables are buffers of one ``nn.Module``
marked static-address, so cudagraph trees neither copy them nor skip on the in-place
KV/token mutation (mutating static inputs is allowed). BF16 precision, B=1 fixture only.
Self-test (GPU): ``python -m megabench.sota.arms.torch_compile --selftest``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
from torch import nn

from .. import qwen3_torch as qt
from ..shapes import Shape

DTYPE = torch.bfloat16
VARIANTS = ("max-autotune", "max-autotune-manualgraph")
WARMUP = 4
_W_NAMES = ("ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2", "wg", "wu", "wd",
            "fnorm", "embed")


class _Decoder(nn.Module):
    """Decode step with every tensor held as a static-address buffer."""

    def __init__(self, w: dict, state: qt.DecodeState, chain: bool):
        super().__init__()
        self.chain = chain
        self.geometry = state.geometry
        for n in _W_NAMES:
            self.register_buffer("w_" + n, w[n], persistent=False)
        for n in ("token", "pos", "kcache", "vcache", "cos", "sin"):
            self.register_buffer(n, getattr(state, n), persistent=False)
        for t in self.buffers():
            torch._dynamo.mark_static_address(t)

    def forward(self) -> dict:
        w = {n: getattr(self, "w_" + n) for n in _W_NAMES}
        state = qt.DecodeState(self.token, self.pos, self.kcache, self.vcache,
                               self.cos, self.sin, self.geometry)
        return qt.decode_step(w, state, DTYPE, self.chain)


class TorchCompileStep:
    arm = "torch-compile"
    precision = "bf16"

    def __init__(self, variant: str, shape: Shape, inputs: dict, device: str):
        from torch._dynamo.utils import counters
        self.variant, self.shape, self.chained = variant, shape, False
        self.config: dict[str, Any] = {"pdl": "n/a", "graph": True}
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/raid/garv901/.cache/inductor")
        import torch._inductor.config as ic
        ic.coordinate_descent_tuning = True
        self.config["inductor_config"] = {"coordinate_descent_tuning": True}
        self.config["inductor_cache_dir"] = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        self.config["torch"] = torch.__version__

        with torch.no_grad():
            w = qt.weights_for(inputs, DTYPE)
            self.state = qt.build_state(shape, inputs, DTYPE, device)
            self._fixture_token = self.state.token.clone()
            mods = {c: _Decoder(w, self.state, c) for c in (False, True)}
            manual = variant == "max-autotune-manualgraph"
            mode = "max-autotune-no-cudagraphs" if manual else "max-autotune"
            self.config.update(compile_mode=mode, manual_cuda_graph=manual)
            self._fns = {c: torch.compile(m, mode=mode, fullgraph=True)
                         for c, m in mods.items()}
            skips0 = counters["inductor"]["cudagraph_skips"]
            t0 = time.perf_counter()
            if manual:
                self._graphs, self._static = {}, {}
                self._capture()
            else:
                for c in (False, True):
                    for _ in range(WARMUP):
                        torch.compiler.cudagraph_mark_step_begin()
                        self._fns[c]()
                torch.cuda.synchronize()
            self.config["compile_s"] = time.perf_counter() - t0
            skips = counters["inductor"]["cudagraph_skips"] - skips0
            self.config["cudagraph_skips"] = skips
            self.config["cudagraphs_active"] = bool(manual or skips == 0)
            self.state.token.copy_(self._fixture_token)
        self._last: dict | None = None

    def _capture(self) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for c in (False, True):
                for _ in range(WARMUP):
                    self._fns[c]()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for c in (False, True):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._static[c] = self._fns[c]()
            self._graphs[c] = g
        torch.cuda.synchronize()

    def launch(self) -> None:
        if "manualgraph" in self.variant:
            self._graphs[self.chained].replay()
        else:
            torch.compiler.cudagraph_mark_step_begin()
            self._last = self._fns[self.chained]()

    def outputs(self) -> dict[str, Any]:
        src = self._static[self.chained] if "manualgraph" in self.variant else self._last
        if src is None:
            raise RuntimeError("outputs() before launch()")
        return {k: v[0].clone() for k, v in src.items()}

    def set_chained(self, on: bool) -> None:
        self.chained = bool(on)
        if not on:
            self.state.token.copy_(self._fixture_token)

    def close(self) -> None:
        self._graphs = {}
        self._last = None


class TorchCompileArm:
    name = "torch-compile"

    def __init__(self, **options: Any):
        self.options = options

    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def supports(self, shape: Shape, variant: str) -> str | None:
        if variant not in VARIANTS:
            return f"unknown variant {variant!r}"
        if shape.batch != 1:
            return "batched fixtures not implemented"
        if not torch.cuda.is_available():
            return "CUDA is not available"
        return None

    def prepare(self, shape: Shape, inputs: dict, variant: str, device: str) -> TorchCompileStep:
        reason = self.supports(shape, variant)
        if reason:
            raise RuntimeError(reason)
        return TorchCompileStep(variant, shape, inputs, device)


def _selftest() -> None:
    from ..shapes import make_shape_inputs, reference_outputs
    shape = Shape(1, 128)
    inputs = make_shape_inputs(shape, 7301, "cuda")
    ref = reference_outputs(shape, inputs)
    arm = TorchCompileArm()
    for variant in arm.variants():
        step = arm.prepare(shape, inputs, variant, "cuda")
        step.set_chained(False)
        step.launch()
        torch.cuda.synchronize()
        out = step.outputs()
        print(f"[{variant}] compile_s={step.config['compile_s']:.1f} "
              f"cudagraph_skips={step.config['cudagraph_skips']} "
              f"active={step.config['cudagraphs_active']}")
        for k in ("logits", "k_write", "v_write"):
            a, b = out[k].double(), ref[k].double()
            print(f"  {k}: rel-L2={((a - b).norm() / b.norm()).item():.3e} "
                  f"max-abs={(a - b).abs().max().item():.3e}")
        print(f"  argmax: got={int(out['next_token'])} ref={int(ref['next_token'])}")
        step.set_chained(True)
        for _ in range(10):
            step.launch()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            step.launch()
        end.record()
        torch.cuda.synchronize()
        print(f"  chained: {start.elapsed_time(end) / 200:.4f} ms/step over 200 launches")
        step.close()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
