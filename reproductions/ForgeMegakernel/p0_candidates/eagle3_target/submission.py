"""One-launch persistent CUDA implementation of the EAGLE3 target verifier."""
import ctypes
import hashlib
import pathlib
import subprocess
import tempfile
import torch

_NAMES = ("token", "draft_tokens", "ln1", "ln2", "fnorm", "wq", "wk", "wv",
          "wo", "wg", "wu", "wd", "embed", "lm_head", "kcache", "vcache",
          "logits", "accepted_count", "committed_count", "committed_tokens",
          "cache_length", "target_features", "k_write", "v_write", "scratch")
_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        source = pathlib.Path(__file__).with_name("kernel.cu")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = pathlib.Path(tempfile.gettempdir()) / f"eagle3_persistent_{digest}.so"
        if not target.exists():
            subprocess.run(["nvcc", "-O3", "--std=c++17", "-shared", "-Xcompiler", "-fPIC",
                            "-rdc=true", "-gencode", "arch=compute_90,code=sm_90",
                            "-gencode", "arch=compute_100,code=sm_100",
                            str(source), "-o", str(target)], check=True)
        _LIB = ctypes.CDLL(str(target))
        _LIB.launch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _LIB.launch.restype = ctypes.c_int
    return _LIB


def build(case: dict):
    p = case["params"]
    assert (p["batch"], p["context"], p["layers"], p["draft_depth"]) == (1, 128, 32, 4)
    lib = _library()
    device = "cuda"

    def run(inputs: dict) -> dict:
        outputs = {
            "logits": torch.empty((5, 128256), device=device, dtype=torch.float32),
            "accepted_count": torch.empty((), device=device, dtype=torch.int64),
            "committed_count": torch.empty((), device=device, dtype=torch.int64),
            "committed_tokens": torch.empty((5,), device=device, dtype=torch.int64),
            "cache_length": torch.empty((), device=device, dtype=torch.int64),
            "target_features": torch.empty((3, 4096), device=device, dtype=torch.bfloat16),
            "k_write": torch.empty((32, 5, 8, 128), device=device, dtype=torch.bfloat16),
            "v_write": torch.empty((32, 5, 8, 128), device=device, dtype=torch.bfloat16),
        }
        scratch = torch.empty((1_500_000,), device=device, dtype=torch.float32)
        values = [inputs.get(name, outputs.get(name, scratch if name == "scratch" else None))
                  for name in _NAMES]
        ptrs = (ctypes.c_void_p * len(values))(*(v.data_ptr() for v in values))
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        error = lib.launch(ptrs, stream)
        if error:
            raise RuntimeError(f"persistent kernel launch failed: CUDA error {error}")
        return outputs

    return run
