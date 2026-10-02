"""Compare the reusable MPK projection pair with FP32 matmul."""

from __future__ import annotations

import json

import torch

from .mpk_hybrid_baseline import MPKPairLinear


def main() -> None:
    gen = torch.Generator(device="cuda").manual_seed(20261002)
    batch, hidden, intermediate = 1, 1024, 3072
    x = torch.randn((batch, hidden), generator=gen, device="cuda") * 0.02
    wg = (torch.randn((intermediate, hidden), generator=gen, device="cuda")
          * hidden ** -0.5).bfloat16()
    wu = (torch.randn((intermediate, hidden), generator=gen, device="cuda")
          * hidden ** -0.5).bfloat16()
    pair = MPKPairLinear(batch, hidden, intermediate)
    gate, up = pair(x, wg, wu)
    expected_gate = x @ wg.float().T
    expected_up = x @ wu.float().T
    first = {"gate_max_abs": (gate - expected_gate).abs().max().item(),
             "up_max_abs": (up - expected_up).abs().max().item()}
    x_new = x * 2
    gate, up = pair(x_new, wg, wu)
    expected_gate = x_new @ wg.float().T
    expected_up = x_new @ wu.float().T
    second = {"gate_max_abs": (gate - expected_gate).abs().max().item(),
              "up_max_abs": (up - expected_up).abs().max().item()}
    print(json.dumps({"first": first, "second": second}, indent=2))
    if max(*first.values(), *second.values()) > 0.01:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
