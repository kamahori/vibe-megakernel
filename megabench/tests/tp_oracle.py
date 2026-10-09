"""Full-weight CPU oracles with the same BF16 rounding as TP partitions.

An unpartitioned BF16 matmul rounds once. TP rounds each partial matmul
before the collective. Emulate only those boundaries in the full-weight
oracle; actual rank execution still exercises real communication/sharding.
"""

from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch

from ..tasks.quantization import pack_fp8_activation, unpack_fp8_activation, unpack_fp8_weight


def storage(tensor):
    return tensor.untyped_storage().data_ptr()


@contextmanager
def tp_rounding(case, values, module):
    tp = case.tp
    if tp == 1:
        yield
        return
    routed = {}
    plain = {}
    for key, value in values.items():
        if key.endswith('_latent_down'):
            continue
        if key in ('wo', 'wd') or key.endswith(('_o', '_down', '_shared_down')):
            if value.dtype == torch.bfloat16:
                destination = routed if key == 'wd' and 'experts' in case.params else plain
                destination[storage(value)] = key
    matmul = torch.Tensor.__matmul__

    def rounded_matmul(weight, vector):
        pointer = storage(weight)
        if weight.ndim != 2 or vector.ndim != 1 or pointer not in plain | routed:
            return matmul(weight, vector)
        parts = [matmul(w, x).float() for w, x in
                 zip(weight.chunk(tp, dim=1), vector.chunk(tp))]
        result = torch.stack(parts).sum(0)
        return result if pointer in routed else result.to(vector.dtype)

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, '__matmul__', rounded_matmul))
        if module.__name__.endswith('.frontier'):
            native_linear = module.linear
            row_weights = {}
            for layer in range(case.params['layers']):
                for name in ('o', 'down', 'shared_down'):
                    key = f'l{layer}_{name}'
                    if key in values:
                        is_routed = layer >= case.params['first_dense'] and name != 'o'
                        row_weights[storage(values[key])] = is_routed

            def fp8_linear(vector, weight, scale, power, *, quantize=True):
                pointer = storage(weight)
                if pointer not in row_weights:
                    return native_linear(vector, weight, scale, power, quantize=quantize)
                decoded = unpack_fp8_weight(weight, scale)
                if quantize:
                    payload, act_scale = pack_fp8_activation(vector.bfloat16(), power_of_two=power)
                    activation = unpack_fp8_activation(payload, act_scale)
                else:
                    activation = vector.float()
                parts = [matmul(w, x).bfloat16().float() for w, x in
                         zip(decoded.chunk(tp, dim=1), activation.chunk(tp))]
                result = torch.stack(parts).sum(0)
                return result if row_weights[pointer] else result.to(vector.dtype)

            stack.enter_context(patch.object(module, 'linear', fp8_linear))
        if module.__name__.endswith('.kimi'):
            unpack = module.unpack_mxfp4
            down = {storage(value) for key, value in values.items() if key.endswith('_down_blocks')}
            decoded_rows = []

            def unpack_expert(blocks, scales):
                result = unpack(blocks, scales)
                # The model immediately converts decoded weights to BF16;
                # return that dtype here so the registered storage survives.
                if storage(blocks) in down:
                    result = result.bfloat16()
                    routed[storage(result)] = 'expert_down'
                    decoded_rows.append(result)
                return result

            stack.enter_context(patch.object(module, 'unpack_mxfp4', unpack_expert))
        yield
