# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

- `megabench/` — **MegaBench**, a benchmark that grades agent-written GPU *megakernels* implementing a complete model inference step (e.g. a full 28-layer Qwen3 decode) against PyTorch references. `megabench/README.md` is the authoritative description; `megabench/docs/` holds the task catalog (`MODEL_STEP_TASKS.md`), precision contract (`REFERENCE_PRECISION.md`), agent brief and session protocol.
- `reproductions/` — `ForgeMegakernel/` (partial reproduction of the ForgeMegakernel agent edit/gate/review loop, adapted to MegaBench P0) and `mpk-megabench/` (notes for Mirage MPK probes in `megabench/probes/`; the Mirage checkout is an external dependency at `reproductions/mirage-mpk`, not vendored).
- `MEGAKERNEL_LANDSCAPE.md` — a literature/landscape survey document; prose only. Links in it to `sandbox/tirx/...` refer to local-only material not in this repo.

## Commands

Run everything from the repository root with the project venv (`.venv/bin/python`); there is no pyproject/requirements file — PyTorch (and CUDA for GPU runs) must already be installed. Tests use `unittest`.

```bash
.venv/bin/python -m unittest megabench.tests.test_suite            # core CPU protocol tests
.venv/bin/python -m unittest discover -s megabench/tests -t .      # all megabench tests
.venv/bin/python -m unittest megabench.tests.test_suite.SomeClass.test_name   # single test
.venv/bin/python -m megabench list --suite p0                      # suites: p0..p3, core, planned, all
.venv/bin/python -m megabench sol --suite p0 --json                # HBM/compute speed-of-light floors
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m megabench evaluate --submission /abs/path/solution.py \
  --case dense-step-qwen3-06b-b1-s128 --method M --session-id S --reps 20 --timeout 900
.venv/bin/python -m megabench aggregate --suite p0 --method M --input a.jsonl ... --output summary.json
.venv/bin/python -m megabench.make_agent_workspace --case CASE_ID --agent plain|vibesys|kernelagent --output NEW_PATH
```

`--device cpu` gives a correctness-only check (no score). Most tests run on CPU with tiny `Case` geometries; a few skip without CUDA. The `verify_*.py` scripts at `megabench/` root are full-geometry GPU verifications (multi-GPU for P2/P3) meant to run via a cluster scheduler. ForgeMegakernel tests: see `reproductions/ForgeMegakernel/README.md`.

## Architecture

- **Case catalog** — `megabench/cases.py` defines frozen `Case` dataclasses (`family`, `params` dims, `suite` p0–p3, `gpus/tp/ep`, `max_gpu_launches`, tolerances, `gate` = `band` or `tilert`, optional `regimes` and `checkpoint`). `select_cases(suite, ids)` picks cases. Only `ready=True` cases can be evaluated or scored.
- **Tasks** — `megabench/tasks/workloads.py` dispatches by `case.family` via `TASK_MODULES` to a per-family module (`dense.py`, `moe.py`, `gemma.py`, …). Each module exposes `make_inputs(case, seed, device)` and `reference(case, inputs)` (BF16 reference). `oracle()` is derived automatically by `tasks/oracle.py:fp32_reference` (same op graph without BF16 activation rounding). Shared math is in `tasks/common.py`; Qwen3 references live in `tasks/references/`. Adding a model = new task module + `TASK_MODULES` entry + `Case` in `cases.py` (+ SOL traffic model in `sol.py`, baselines in `baselines.py` as needed).
- **Harness** — `harness/runner.py` is the CLI (`list`/`sol`/`evaluate`/`aggregate`/internal `_worker`). `evaluate` runs each case in a fresh subprocess (or a locked-down Docker container with `--docker-image`) with a timeout. `harness/correctness.py` compares outputs against the oracle band and rejects input mutation; `harness/benchmark.py` times candidate vs eager and CUDA-Graph references and audits GPU launch counts with the PyTorch profiler (exceeding `max_gpu_launches` → `non_megakernel`). `harness/distributed.py` handles multi-GPU cases.
- **Compatibility shims** — root-level `megabench/runner.py` and `megabench/workloads.py` re-export harness/task symbols because existing submissions and agent workspaces import them; keep them working.
- **Contract digest** — `harness/runner.py:CONTRACT_FILES` lists the files whose hash is recorded in every result; aggregation rejects results with mismatched digests. Editing any of those files (cases, tasks, harness, baselines) invalidates previously recorded results, so update that list when adding a new task module.
- **Submission contract** — a Python file exporting `build(case: dict) -> run`, where `run(inputs) -> outputs` must not mutate inputs and must compute from current inputs. See `examples/reference_submission.py`.
- **Agent workspaces & integrations** — `make_agent_workspace.py` builds a minimal per-case workspace (TASK.md, case.json, only that case's reference and imports, `check_candidate`/`profile_candidate`/`sol_info`), copies `skills/` in as `megakernel_skills/`, and calls back into the trusted evaluator in this repo. `integrations/` holds the VibeSys adapter, NCU profiling, and the per-block phase timeline probe (`timeline_probe.cuh` + `timeline.py`).

## Conventions and rules

- **One case = one fresh agent session** (`docs/SESSION_PROTOCOL.md`). Aggregation rejects duplicate session IDs, duplicate cases, and shared candidate directories, and emits a geometric-mean score only if every ready case passes correctness and the launch gate.
- Results and workspaces are exclusive-create: never overwrite existing output files or experiment directories; always use a new path. `experiments/YYYY-MM-DD/HH-MM-SS-<campaign>/`, `runs/`, `archive/`, `*.jsonl`, `*.log`, `*.ncu-rep` are git-ignored local material.
- Precision: references compute in BF16 with FP32 sensitive reductions and native quantized weights; grading tolerances are relative to the BF16 reference's own error vs the FP32 oracle (`noise_factor`). Follow `docs/REFERENCE_PRECISION.md` when changing reference math.
- Experimental results are recorded in GitHub issues of `kamahori/vibe-megakernel`, not committed.
