"""Count native frontier input storage from meta shapes, without allocating GPUs.

This is a topology planning check. It does not verify GPU memory peaks,
collectives, execution, or checkpoint accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from .cases import select_cases
from .tasks import frontier, glm53, kimi, kimi_layer


def matrix(seed, name, rows, cols, fan_in, device, *, row_range=None, col_range=None):
    r0, r1 = row_range or (0, rows)
    c0, c1 = col_range or (0, cols)
    return torch.empty((r1-r0, c1-c0), device='meta', dtype=torch.bfloat16)


def fp8_matrix(seed, name, rows, cols, device, power_of_two, *, row_range=None, col_range=None):
    r0, r1 = row_range or (0, rows)
    c0, c1 = col_range or (0, cols)
    return (torch.empty((r1-r0, c1-c0), device='meta', dtype=torch.float8_e4m3fn),
            torch.empty((math.ceil((r1-r0)/128), math.ceil((c1-c0)/128)),
                        device='meta', dtype=torch.float32))


def mxfp4_matrix(seed, name, rows, cols, device, *, row_range=None, col_range=None):
    r0, r1 = row_range or (0, rows)
    c0, c1 = col_range or (0, cols)
    return (torch.empty((r1-r0, (c1-c0)//32, 16), device='meta', dtype=torch.uint8),
            torch.empty((r1-r0, (c1-c0)//32), device='meta', dtype=torch.uint8))


def audit(case):
    module = {'kimi_k3_step':kimi, 'glm53_flash_step':glm53, 'kimi_k3_layer':kimi_layer}.get(case.family, frontier)
    with patch.object(module, 'matrix', matrix), \
         patch.object(frontier, 'fp8_matrix', fp8_matrix), \
         patch.object(glm53, 'fp8_matrix', fp8_matrix), \
         patch.object(kimi, 'mxfp4_matrix', mxfp4_matrix), \
         patch.object(kimi_layer, 'mxfp4_matrix', mxfp4_matrix):
        values = module.make_inputs(case, 104729, 'meta', rank=0)
    by_dtype = Counter()
    for value in values.values():
        by_dtype[str(value.dtype)] += value.numel()*value.element_size()
    return {'case':case.id, 'tp':case.tp, 'ep':case.ep, 'tensor_count':len(values),
            'rank_input_bytes':sum(by_dtype.values()), 'bytes_by_dtype':dict(by_dtype),
            'method':'meta storage shapes only; not allocated GPU verification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = select_cases('p3')
    cases.append(replace(cases[-1], gpus=8, tp=8))
    results = [audit(case) for case in cases]
    with args.output.open('x') as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
