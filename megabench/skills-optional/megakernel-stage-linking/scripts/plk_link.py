#!/usr/bin/env python3
"""Check stage contracts of a megakernel pipeline, then compile packages separately and LTO-link one .so.

  plk_link.py --packages packages.json --pipeline pipeline.json --out build/k.so [--check-only]
              [--onetu build/k_onetu.so] [--arch gfx942] [-D NAME ...] [--hipcc-flag FLAG ...]

packages.json: {"name": {"src": "stages/a.cu", "defines": [], "exports": ["stage_a"], "calls": [],
                         "needs": ["x@l-1"], "provides": ["y@l"], "consumes": ["wq"]}, ...}
pipeline.json: {"slots": {"a": "name", ...},
                "sequence": ["init", {"repeat": ["a", "b"]}, "final"]}
Resources: "name@l" / "name@l-1" are per-layer (l = loop index; "@-1" before the loop, "@last" after);
plain names are register/LDS hand-offs and are removed when a stage lists them in "consumes".
--onetu also builds a one-TU control (all package sources #included into one TU) for the codegen gate.
"""
import argparse, json, os, subprocess, sys, tempfile

def resolve(res, l):
    return res.replace("@l-1", f"@{l-1}").replace("@l", f"@{l}")

def check(pkgs, pipe):
    slots, seq, errs, have = pipe["slots"], pipe["sequence"], [], set()
    for s, n in slots.items():
        if n not in pkgs: errs.append(f"L-PKG slot {s}: unknown package {n}")
    if errs: return errs
    exports = {e for n in slots.values() for e in pkgs[n].get("exports", [])}
    for n in slots.values():
        for f in pkgs[n].get("calls", []):
            if f not in exports: errs.append(f"L-CALL package {n} calls '{f}', which no linked package exports")
    def run(slot, l):
        p = pkgs[slots[slot]]
        for r in p.get("needs", []):
            rr = resolve(r, l) if l is not None else r
            if rr not in have: errs.append(f"L-NEED slot {slot} (package {slots[slot]})"
                                           f"{'' if l is None else f' layer {l}'}: needs '{r}' -> '{rr}' not provided")
        for r in p.get("consumes", []): have.discard(resolve(r, l) if l is not None else r)
        for r in p.get("provides", []): have.add(resolve(r, l) if l is not None else r.replace("@l", "@-1"))
    last = None
    for step in seq:
        if isinstance(step, dict):
            for l in range(2):
                for s in step["repeat"]: run(s, l)
            last = 1
            have.update({x.replace(f"@{last}", "@last") for x in list(have) if x.endswith(f"@{last}")})
        else:
            run(step, None)
    return sorted(set(errs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", required=True); ap.add_argument("--pipeline", required=True)
    ap.add_argument("--out"); ap.add_argument("--onetu"); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--arch", default="gfx942"); ap.add_argument("-D", action="append", default=[])
    ap.add_argument("--hipcc-flag", action="append", default=[])
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(a.packages))
    pkgs, pipe = json.load(open(a.packages)), json.load(open(a.pipeline))
    errs = check(pkgs, pipe)
    if errs:
        print("link REJECTED:"); [print("  " + e) for e in errs]; return 1
    print("contracts OK: " + ", ".join(f"{k}={v}" for k, v in pipe["slots"].items()))
    if a.check_only: return 0
    base = ["-O3", f"--offload-arch={a.arch}", "-fPIC", *[f"-D{d}" for d in a.D], *a.hipcc_flag]
    objdir = a.out + ".objs"; os.makedirs(objdir, exist_ok=True); objs = []
    for slot, n in pipe["slots"].items():
        p = pkgs[n]; obj = os.path.join(objdir, f"{slot}-{n}.o")
        subprocess.check_call(["hipcc", *base, "-fgpu-rdc", *[f"-D{d}" for d in p.get("defines", [])],
                               "-c", os.path.join(root, p["src"]), "-o", obj]); objs.append(obj)
    subprocess.check_call(["hipcc", *base, "-fgpu-rdc", "--hip-link", "-shared", *objs, "-o", a.out])
    print(f"linked {a.out} from {len(objs)} packages")
    if a.onetu:
        defs = {d for n in pipe["slots"].values() for d in pkgs[n].get("defines", [])}
        if len({tuple(pkgs[n].get("defines", [])) for n in pipe["slots"].values()} - {()}) > 0:
            print("note: package-specific defines are applied globally in the one-TU control")
        with tempfile.NamedTemporaryFile("w", suffix=".cu", dir=root, delete=False) as f:
            for n in pipe["slots"].values(): f.write(f'#include "{pkgs[n]["src"]}"\n')
            tu = f.name
        try:
            subprocess.check_call(["hipcc", *base, *[f"-D{d}" for d in sorted(defs)], "-shared", tu, "-o", a.onetu])
        finally:
            os.unlink(tu)
        print(f"one-TU control {a.onetu}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
