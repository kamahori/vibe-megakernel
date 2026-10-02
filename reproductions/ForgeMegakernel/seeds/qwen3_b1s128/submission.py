"""One persistent CTA Qwen3 decode; all model arithmetic stays on the GPU."""
from pathlib import Path


def build(case: dict):
    import torch
    import tempfile
    from torch.utils.cpp_extension import load

    expected = dict(batch=1, context=128, layers=28, hidden=1024,
                    q_heads=16, kv_heads=8, head_dim=128,
                    intermediate=3072, vocab=151936)
    if any(case['params'].get(k) != v for k, v in expected.items()):
        raise ValueError('This candidate specializes the public case geometry')
    root = Path(__file__).resolve().parent
    extension = load(name='dense_qwen3_single_cta',
                     sources=[str(root / 'dense_binding.cpp'),
                              str(root / 'dense_launch.cu')],
                     build_directory=tempfile.mkdtemp(prefix='dense_qwen3_build_'),
                     extra_cuda_cflags=['-O3', '--fmad=false'],
                     extra_cflags=['-O3'], verbose=False)
    names = ('token', 'ln1', 'wq', 'wk', 'wv', 'qn', 'kn', 'wo',
             'ln2', 'wg', 'wu', 'wd', 'fnorm', 'embed', 'kcache', 'vcache')
    shapes = ((), (28,1024), (28,2048,1024), (28,1024,1024),
              (28,1024,1024), (28,128), (28,128), (28,1024,2048),
              (28,1024), (28,3072,1024), (28,3072,1024),
              (28,1024,3072), (1024,), (151936,1024),
              (28,128,8,128), (28,128,8,128))

    def run(inputs: dict) -> dict:
        tensors = [inputs[name] for name in names]
        for name, tensor, shape in zip(names, tensors, shapes):
            if tuple(tensor.shape) != shape:
                raise ValueError(f'{name}: expected shape {shape}')
        device = tensors[0].device
        # empty allocations launch no initialization kernels; every value read
        # by the kernel is written in the same invocation before use.
        outputs = [torch.empty((151936,), device=device, dtype=torch.float32),
                   torch.empty((), device=device, dtype=torch.int64),
                   torch.empty((28,8,128), device=device, dtype=torch.bfloat16),
                   torch.empty((28,8,128), device=device, dtype=torch.bfloat16)]
        scratch = torch.empty((15360,), device=device, dtype=torch.float32)
        extension.run(tensors, outputs, scratch)
        return dict(zip(('logits', 'next_token', 'k_write', 'v_write'), outputs))

    return run
