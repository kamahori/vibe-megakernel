"""Matrix-multiply phases of a case, traced from its PyTorch reference.

The reference runs once on CPU with every matrix multiply replaced by an
empty result of the right shape, so only the input generation costs time and
memory. Each distinct (op, M, N, K, batch, dtype) becomes one phase with its
count per step. A weight is `quantized` when it comes from an integer tensor
through shape-preserving ops (the reference dequantizes before the multiply).

  python -m megabench.primdb.phases --case ID [--case ID ...]   update case_phases.json

The workspace builder reads case_phases.json; tracing a large case needs tens
of GB of host memory, so it is done here once per case and not per workspace.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PHASES = Path(__file__).with_name("case_phases.json")
# A dequantized tensor has up to 8x the elements of its packed integer source.
_PACK = 8


def trace(case) -> list[dict]:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    from megabench.tasks import workloads

    aten = torch.ops.aten
    pairs = {aten.mm.default: (0, 1), aten.addmm.default: (1, 2),
             aten.bmm.default: (0, 1), aten.baddbmm.default: (1, 2)}
    vectors = {aten.mv.default: (0, 1), aten.addmv.default: (1, 2)}

    class Recorder(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.counts: Counter = Counter()

        def _origin(self, t):
            if not isinstance(t, torch.Tensor):
                return None
            if getattr(t, "_primdb_source", None):
                return t._primdb_source
            if not t.is_floating_point() and not t.is_complex() and t.dtype != torch.bool:
                return str(t.dtype).removeprefix("torch.")
            return None

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in pairs or func in vectors:
                if func in pairs:
                    i, j = pairs[func]
                    a, b = args[i], args[j]
                    batch = a.shape[0] if a.dim() == 3 else 1
                    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
                    weight, out_shape = b, (*a.shape[:-1], n)
                else:
                    i, j = vectors[func]
                    a, b = args[i], args[j]
                    batch, m, n, k = 1, 1, a.shape[0], a.shape[1]
                    weight, out_shape = a, (n,)
                key = (func.__name__.split(".")[0], int(m), int(n), int(k), int(batch),
                       str(a.dtype).removeprefix("torch."), self._origin(weight))
                self.counts[key] += 1
                return a.new_zeros(out_shape)
            out = func(*args, **kwargs)
            origins = [(self._origin(t), t) for t in args if isinstance(t, torch.Tensor)]
            origins = [(o, t) for o, t in origins if o is not None]
            if origins and isinstance(out, torch.Tensor) and out.is_floating_point():
                origin, src = origins[0]
                if src.numel() * _PACK >= out.numel() and out.numel() * _PACK >= src.numel():
                    out._primdb_source = origin
            return out

    inputs = workloads.make_inputs(case, 0, "cpu")
    recorder = Recorder()
    with recorder, torch.no_grad():
        workloads.reference(case, inputs)
    phases = []
    for (op, m, n, k, batch, dtype, origin), count in recorder.counts.most_common():
        phases.append({"op": op, "M": m, "N": n, "K": k, "batch": batch, "dtype": dtype,
                       "weight_source": origin, "count": count})
    return phases


def load(path: Path = PHASES) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def main(argv: list[str] | None = None) -> int:
    from megabench.cases import CASES

    p = argparse.ArgumentParser(prog="phases", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", required=True)
    p.add_argument("--output", type=Path, default=PHASES)
    args = p.parse_args(argv)
    table = load(args.output)
    by_id = {c.id: c for c in CASES}
    for case_id in args.case:
        table[case_id] = trace(by_id[case_id])
        print(f"{case_id}: {len(table[case_id])} phases")
    args.output.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
