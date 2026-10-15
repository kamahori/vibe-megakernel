"""Compile CUDA sources into a cached shared library and load it with ctypes.

The cache key covers the sources, every loadstream header and the flags, so
an edited header always rebuilds. Kernels export ``extern "C"`` launchers
that take raw pointers and a ``cudaStream_t`` (pass
``torch.cuda.current_stream().cuda_stream``).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

INCLUDE = Path(__file__).resolve().parents[1] / "include"
DEFAULT_FLAGS = ("-O3", "-std=c++17", "-arch=sm_100a", "-lineinfo", "--expt-relaxed-constexpr",
                 "-Xcompiler", "-fPIC", "-shared")


def nvcc() -> str:
    found = shutil.which("nvcc")
    for candidate in (found, "/usr/local/cuda-13.1/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("nvcc not found; add the CUDA bin directory to PATH")


def compile_shared(sources: list[str | Path], *, defines: dict[str, object] | None = None,
                   flags: tuple[str, ...] = (), cache_dir: str | Path | None = None,
                   name: str = "kernel") -> ctypes.CDLL:
    """Build (or reuse) ``<cache>/<name>-<hash>.so`` and return it loaded.

    ``defines`` become ``-DKEY=VALUE`` (``-DKEY`` for ``None``/``True``), e.g.
    ``{"LS_PROFILE": None, "LS_WATCHDOG": None}`` for a profiling build.
    """
    sources = [Path(s).resolve() for s in sources]
    args = list(DEFAULT_FLAGS) + list(flags) + [f"-I{INCLUDE}"]
    for key, value in (defines or {}).items():
        args.append(f"-D{key}" if value in (None, True) else f"-D{key}={value}")
    digest = hashlib.sha256()
    for path in sources + sorted(INCLUDE.rglob("*.cuh")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(" ".join(args).encode())
    cache = Path(cache_dir or os.environ.get("LOADSTREAM_CACHE", INCLUDE.parent / ".cache"))
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name}-{digest.hexdigest()[:16]}.so"
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            out = Path(tmp) / target.name
            command = [nvcc(), *args, *map(str, sources), "-o", str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"nvcc failed ({' '.join(command)}):\n{result.stderr[-8000:]}")
            os.replace(out, target)
    return ctypes.CDLL(str(target))
