"""nvcc for microbenchmarks, with a content cache and a compile time limit.

usage: mb_build [--timeout SECONDS] -- <nvcc arguments ... -o OUTPUT ...>

The key of an output is the nvcc arguments, the nvcc version, and the contents
of every source and header next to each input file. A hit copies the cached
binary to OUTPUT and does not run nvcc. A change to a source, a local header,
a flag or a -D value is a new key. A compile that is longer than the limit is
stopped with its process group, and the tool says so: one translation unit per
measured arm compiles in seconds, a file that holds all arms behind -D sets
can take tens of minutes for each set.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT = 300
SOURCE_SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".inc"}


def _output(args: list[str]) -> Path | None:
    for i, arg in enumerate(args):
        if arg == "-o" and i + 1 < len(args):
            return Path(args[i + 1])
        if arg.startswith("-o") and len(arg) > 2 and not arg.startswith("-odir"):
            return Path(arg[2:])
    return None


def _inputs(args: list[str]) -> list[Path]:
    out = _output(args)
    return [Path(a) for a in args
            if not a.startswith("-") and Path(a).suffix in SOURCE_SUFFIXES
            and Path(a).is_file() and Path(a) != out]


def _include_dirs(args: list[str]) -> list[Path]:
    dirs = []
    for i, arg in enumerate(args):
        if arg == "-I" and i + 1 < len(args):
            dirs.append(Path(args[i + 1]))
        elif arg.startswith("-I") and len(arg) > 2:
            dirs.append(Path(arg[2:]))
    return [d for d in dirs if d.is_dir()]


def cache_key(args: list[str], nvcc_version: str) -> str:
    """Hash of the arguments, the compiler version and the local sources."""
    h = hashlib.sha256()
    h.update(nvcc_version.encode())
    h.update("\0".join(args).encode())
    dirs = {p.resolve().parent for p in _inputs(args)} | {d.resolve() for d in _include_dirs(args)}
    for d in sorted(dirs):
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix in SOURCE_SUFFIXES:
                h.update(str(f).encode())
                h.update(f.read_bytes())
    return h.hexdigest()[:24]


def _nvcc() -> str:
    return os.environ.get("MB_BUILD_NVCC") or shutil.which("nvcc") or "nvcc"


def _version(nvcc: str) -> str:
    try:
        return subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                              timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def build(args: list[str], cache_dir: Path, timeout: float, nvcc: str | None = None) -> int:
    nvcc = nvcc or _nvcc()
    out = _output(args)
    if out is None:
        print("mb_build: give the output with -o; the cache stores that file", file=sys.stderr)
        return 2
    key = cache_key(args, _version(nvcc))
    cached = cache_dir / key / out.name
    if cached.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, out)
        print(f"mb_build: cache hit {key}: {out} is unchanged sources and flags; nvcc not run")
        return 0
    start = time.monotonic()
    proc = subprocess.Popen([nvcc, *args], start_new_session=True)
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        print(f"mb_build: nvcc stopped after {timeout:.0f} s (limit). Split the source: one "
              "translation unit per measured arm, shared code in a header. A longer limit "
              "needs --timeout.", file=sys.stderr)
        return 124
    took = time.monotonic() - start
    if code != 0:
        return code
    if out.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
        shutil.copy2(out, tmp)
        os.replace(tmp, cached)
    print(f"mb_build: built {out} in {took:.1f} s (cached as {key})")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        split = argv.index("--")
        own, nvcc_args = argv[:split], argv[split + 1:]
    else:
        own, nvcc_args = [], argv
    parser = argparse.ArgumentParser(prog="mb_build", description=__doc__.split("\n")[0])
    parser.add_argument("--timeout", type=float,
                        default=float(os.environ.get("MB_BUILD_TIMEOUT", DEFAULT_TIMEOUT)))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("MB_BUILD_CACHE",
                                                    "progress/microbench/.build-cache")))
    opts = parser.parse_args(own)
    if not nvcc_args:
        parser.error("give the nvcc arguments after --")
    return build(nvcc_args, opts.cache_dir, opts.timeout)


if __name__ == "__main__":
    sys.exit(main())
