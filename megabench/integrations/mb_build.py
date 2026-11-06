"""nvcc for microbenchmarks, with a content cache and a compile time limit.

usage: mb_build [--timeout SECONDS] -- <nvcc arguments ... -o OUTPUT ...>

The key of an output is the nvcc arguments, the nvcc version, and the contents
of every file that the sources include (``nvcc -M``), plus the contents of
input objects for a link. A hit copies the cached output to OUTPUT and does not
run nvcc. A change to an included file, a flag or a -D value is a new key; an
edit to one translation unit does not rebuild another. With ``-c``, each
translation unit becomes its own cached object, and a link of objects is cached
by their contents. A compile that is longer than the limit is
stopped with its process group, and the tool says so: one translation unit per
measured arm compiles in seconds, a file that holds all arms behind -D sets
can take tens of minutes for each set.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
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


OBJECT_SUFFIXES = {".o", ".obj", ".a"}


def _objects(args: list[str]) -> list[Path]:
    out = _output(args)
    return [Path(a) for a in args
            if not a.startswith("-") and Path(a).suffix in OBJECT_SUFFIXES
            and Path(a).is_file() and Path(a) != out]


def _without_output(args: list[str]) -> list[str]:
    kept, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg == "-o":
            skip = True
        elif arg in ("-c", "--compile") or (arg.startswith("-o") and len(arg) > 2):
            continue
        else:
            kept.append(arg)
    return kept


def dependencies(args: list[str], nvcc: str) -> list[Path] | None:
    """The files that the sources of *args* include, from ``nvcc -M``; None on failure."""
    sources = _inputs(args)
    if not sources:
        return []
    flags = [a for a in _without_output(args) if Path(a).suffix not in OBJECT_SUFFIXES]
    try:
        res = subprocess.run([nvcc, *flags, "-M"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    files = []
    for rule in res.stdout.replace("\\\n", " ").splitlines():
        target, colon, prerequisites = rule.partition(": ")
        for token in prerequisites.split() if colon else []:
            path = Path(token)
            if token != "\\" and path.is_file():
                files.append(path.resolve())
    return sorted(set(files) | {p.resolve() for p in sources})


def cache_key(args: list[str], nvcc_version: str, nvcc: str | None = None) -> str:
    """Hash of the arguments, the compiler version, the included files and the objects.

    The included files come from the compiler's dependency list, so an edit to one
    translation unit does not change the key of another. If the list is not
    available, every source and header next to each input counts.
    """
    h = hashlib.sha256()
    h.update(nvcc_version.encode())
    h.update("\0".join(args).encode())
    deps = dependencies(args, nvcc) if nvcc else None
    if deps is None:
        dirs = {p.resolve().parent for p in _inputs(args)}
        deps = sorted(f.resolve() for d in dirs for f in d.iterdir()
                      if f.is_file() and f.suffix in SOURCE_SUFFIXES)
    for f in [*deps, *(o.resolve() for o in _objects(args))]:
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
    key = cache_key(args, _version(nvcc), nvcc)
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
    # A quoted flag string ("-O3 -std=c++17 ...") is split into its flags.
    nvcc_args = [part for arg in nvcc_args
                 for part in (shlex.split(arg) if arg.startswith("-") and " " in arg else [arg])]
    return build(nvcc_args, opts.cache_dir, opts.timeout)


if __name__ == "__main__":
    sys.exit(main())
