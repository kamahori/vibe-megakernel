# MegaBench: whole-model megakernel challenges

MegaBench evaluates submitted implementations of a **complete model inference
step**. A decode task begins with a token and prior model state, runs every
decoder layer, and returns logits, the greedy next token, and new state. The
[task catalog](docs/MODEL_STEP_TASKS.md) describes each architecture and timed
boundary; [`cases.py`](cases.py) is the active machine-readable catalog.

## Repository layout

- [`cases.py`](cases.py) defines the case catalog.
- [`tasks/`](tasks/) contains one fixture and PyTorch reference per ready model,
  shared reference helpers, and the workload dispatcher. Its
  [`references/`](tasks/references/) package owns the Qwen3 decoder references
  used by MegaBench; it does not import the TIRx experiment copies.
- [`harness/`](harness/) contains the correctness checker, timing and launch
  audit, evaluation runner, and aggregation. `python -m megabench` remains the
  command-line entry point. Small root-level `runner.py` and `workloads.py`
  modules preserve imports used by existing submissions and agent workspaces.
- [`docs/`](docs/) holds the task catalog, the one-case agent brief, and the
  session protocol.
- [`integrations/`](integrations/) holds the VibeSys adapter and NCU profiling
  commands. The root-level NCU command remains available for agent workspaces.
- [`tests/`](tests/) contains the suite tests; [`examples/`](examples/) contains
  a reference submission.
- `experiments/YYYY-MM-DD/HH-MM-SS-<campaign>/`, `runs/`, and `archive/` hold
  local runs and historical material. They are ignored by Git. Existing
  experiment paths remain available through symlinks.

## Active catalog

| Priority | Model architecture and step | State |
| --- | --- | --- |
| P0 | Qwen3-0.6B dense GQA, full 28-layer decode | Ready with seeded synthetic BF16 weights |
| P0 | Qwen3-30B-A3B MoE, full decode | Ready with synthetic BF16 weights |
| P0 | Gemma 3 4B local/global attention, W8A16 and W4A16 full decode | Both ready with synthetic quantized weights |
| P0 | Llama 3.1 8B target with EAGLE3 proposal verification | Ready with synthetic weights and a linear proposal chain |
| P1 | GPT-OSS-20B native MXFP4 MoE decode | Planned |
| P1 | Qwen3.5-0.8B DeltaNet/attention hybrid decode | Planned |
| P1 | Gemma 3 4B image-conditioned text decode | Planned |
| P2 | Llama 3.1 8B plus EAGLE3 full speculative iteration | Planned |
| P2 | Gemma 3 27B TP decode; Qwen3-30B-A3B TP/EP decode | Planned |
| P3 | DeepSeek-V3.2, GLM-5.2-FP8, and Kimi-K3 full decode | Planned |

`list --suite core` and `list --suite p0` select the five ready P0 cells. `--suite p1`
through `p3`, `planned`, and `all` expose the rest of the catalog; an
unimplemented case returns `not_implemented` and cannot contribute to a score.
All P0 cells use full model layer counts and seeded synthetic weights. They
are a **shape/semantics tier**, not checkpoint accuracy. The Qwen3 MoE case
uses all 128 experts with top-8 routing and a separate LM head. Gemma W8 and
W4 use the same seeded
BF16 base stream, symmetric per-output-row scales, signed int8 or packed
signed int4 projection weights, and BF16 embedding/norms. Dequantization is
inside the timed step. The EAGLE3 target cell uses a feature-fused one-layer
synthetic draft head to generate a linear four-token proposal chain outside
timing; the timed task runs the full 32-layer Llama target and commit logic.
Correctness trials include raw draft proposals, full acceptance, and rollback
after one accepted token. Pinned paired checkpoints and branched trees remain
future tiers described in the [catalog](docs/MODEL_STEP_TASKS.md).

The former small fusion suite is retired. Its suite definition and results are
preserved locally in `megabench/archive/legacy-small-suite-2026-09-30/`,
which is ignored by Git. The old VibeSys smoke task and submitted kernels are
historical and do not implement the active cases.

## Submission contract

Provide a Python file exporting:

```python
def build(case: dict):
    # Compile or allocate reusable state here. Case metadata is public.
    def run(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Compute the complete inference step from the current inputs.
        ...
    return run
```

`run` may allocate outputs but must not mutate inputs. It must compute from
the current token, weights, and prior state; correctness trials change their
values. Every P0 cell currently has a one-GPU-kernel launch budget. See
[`tasks/`](tasks/) for tensor names, layouts, and exact reference math;
[`cases.py`](cases.py) specifies shapes
and tolerances. The included
[`reference_submission.py`](examples/reference_submission.py) demonstrates
the API and CPU correctness path. Its many PyTorch GPU launches fail the
megakernel launch gate.

Use the [agent task brief](docs/AGENT_TASK.md) when comparing implementations. One
case receives one fresh agent session and candidate checkout. The evaluator
accepts one case and records its method and session ID; aggregation reads five
separate case results. A suite-wide prompt or a shared candidate submission
does not define an agent benchmark run. The [session protocol](docs/SESSION_PROTOCOL.md)
specifies the isolation and aggregation rules.
Experimental results are recorded in GitHub issues: the [one-case P0
comparison](https://github.com/kamahori/vibe-megakernel/issues/1), the
[earlier suite-wide P0 trial](https://github.com/kamahori/vibe-megakernel/issues/2),
the [retired smoke-suite trial](https://github.com/kamahori/vibe-megakernel/issues/3),
and [MPK diagnostics](https://github.com/kamahori/vibe-megakernel/issues/4).
Raw logs, candidate checkouts, and result files remain local under ignored
directories.

## Run

```bash
.venv/bin/python -m unittest megabench.tests.test_suite
.venv/bin/python -m megabench list --suite p0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m megabench evaluate \
  --submission /absolute/path/to/solution.py \
  --case dense-step-qwen3-06b-b1-s128 \
  --method plain-codex --session-id dense-run-1 \
  --reps 20 --timeout 900
```

Repeat in four other fresh agent sessions, each with its own `--case`,
`--session-id`, candidate directory, and output JSONL. Then combine the five
results for one method:

```bash
.venv/bin/python -m megabench aggregate --suite p0 --method plain-codex \
  --input /path/to/dense.jsonl --input /path/to/moe.jsonl \
  --input /path/to/gemma-w8.jsonl --input /path/to/gemma-w4.jsonl \
  --input /path/to/eagle3.jsonl --output /path/to/p0-summary.json
```

Choose a visible GPU you are authorized to use. `--device cpu` performs a
correctness-only check; it does not create a GPU score. Each case runs in a
fresh subprocess with a timeout. Results are exclusive-create JSONL under
`megabench/runs/` by default, or at `--output`; existing files are never
overwritten. The full MoE cell needs roughly 60 GB for input weights plus
host snapshots for mutation checks. Both `runs/` and the legacy archive are
ignored by Git.

To isolate each case in a fresh Docker container, supply an existing local
image with Python, PyTorch, and the submission compiler/runtime:

```bash
.venv/bin/python -m megabench evaluate \
  --submission /absolute/path/to/solution.py \
  --case dense-step-qwen3-06b-b1-s128 --device cuda:0 \
  --method plain-codex --session-id dense-run-1 \
  --docker-image IMAGE_WITH_PYTORCH_AND_YOUR_KERNEL_DSL \
  --docker-gpus device=0 --reps 20
```

The selected host GPU appears as `cuda:0` inside the container. The image
must already be present (`--pull never`). The harness neither builds nor
removes images. The repository and submission directory are mounted read-only;
only a temporary result directory and `/tmp` are writable. The container has
no network, runs as the invoking UID/GID, and is removed after the run. On
timeout, cleanup checks the container's unique label before stopping it.
Docker isolation reduces accidental host changes; untrusted submissions
should run on a dedicated worker host.

## Evaluation

For each ready case, the harness generates fresh randomized inputs and checks
every output key, shape, dtype, device, and value against its PyTorch oracle.
Integer tokens must match exactly; floating outputs use the case tolerance.
BF16 state outputs use the separate `bf16_rtol` (0.008 by default) so a
rounding step from a different reduction order does not reject an otherwise
matching full-model computation. FP32 logits retain the tighter `rtol`.
Input mutation is rejected. Cold first-call time is reported separately.
Warm timing records synchronized host and CUDA-event latency for the candidate
and eager reference. Where capture succeeds, it also times a CUDA Graph
reference on the same static input. The provisional speedup divides the
faster available baseline by candidate median CUDA-event time.

The PyTorch profiler counts observed GPU kernel executions in one steady-state
`run`. A count outside the case launch budget yields `non_megakernel` even if
the host called one CUDA Graph replay. A passing trace is `ok_provisional`
until source and trace review verifies complete fused work and current-input
dependence. If profiling is unavailable, authenticity is unverified. A
geometric-mean speedup is emitted by `aggregate` only if **every ready case**
has a result from its own session and passes correctness and the launch gate.
Planned cases, CPU checks, and partially solved suites have no aggregate score.
The aggregator rejects duplicate session IDs, duplicate cases, and shared
candidate directories. Eager and graph references are local
baselines; a public ranking would additionally need optimized serving
baselines, repeated isolated regrades, and checkpoint-tier checks.
