"""Minimal one-launch submission for the stencil challenge only."""

import torch
import triton
import triton.language as tl


@triton.jit
def _stencil(X, ALPHA, Y, WIDTH: tl.constexpr, STEPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    x = tl.load(X + row * WIDTH + col)
    alpha = tl.load(ALPHA)
    for _ in range(STEPS):
        left = tl.gather(x, (col + WIDTH - 1) % WIDTH, 0)
        right = tl.gather(x, (col + 1) % WIDTH, 0)
        x = (1.0 - 2.0 * alpha) * x + alpha * (left + right)
    tl.store(Y + row * WIDTH + col, x)


def build(case: dict):
    if case["family"] != "stencil":
        raise NotImplementedError("this example implements only stencil")
    params = case["params"]
    batch, width, steps = (params[k] for k in ("batch", "width", "steps"))

    def run(inputs: dict):
        y = torch.empty_like(inputs["x"])
        _stencil[(batch,)](inputs["x"], inputs["alpha"], y, width, steps,
                           num_warps=4)
        return {"y": y}

    return run
