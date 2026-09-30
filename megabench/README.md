# MegaBench: whole-model megakernel challenges

MegaBench evaluates submitted implementations of a **complete model inference
step**. A decode task begins with a token and prior model state, runs every
decoder layer, and returns logits, the greedy next token, and new state. The
[task catalog](MODEL_STEP_TASKS.md) describes each architecture and timed
boundary; [`cases.py`](cases.py) is the active machine-readable catalog.

## Active catalog

| Priority | Model architecture and step | State |
| --- | --- | --- |
| P0 | Qwen3-0.6B dense GQA, full 28-layer decode | Ready with seeded synthetic BF16 weights |
| P0 | Qwen3-30B-A3B MoE, full decode | Planned |
| P0 | Gemma 3 4B local/global attention, W8A16 and W4A16 full decode | Planned |
| P0 | Llama 3.1 8B target with EAGLE3 proposal verification | Planned |
| P1 | GPT-OSS-20B native MXFP4 MoE decode | Planned |
| P1 | Qwen3.5-0.8B DeltaNet/attention hybrid decode | Planned |
| P1 | Gemma 3 4B image-conditioned text decode | Planned |
| P2 | Llama 3.1 8B plus EAGLE3 full speculative iteration | Planned |
| P2 | Gemma 3 27B TP decode; Qwen3-30B-A3B TP/EP decode | Planned |
| P3 | DeepSeek-V3.2, GLM-5.2-FP8, and Kimi-K3 full decode | Planned |

`--suite core` selects ready cases only. `--suite p0` through `p3`, `planned`,
and `all` expose the rest of the catalog; an unimplemented case returns
`not_implemented` and cannot contribute to a score. The first ready case uses
the real Qwen3-0.6B geometry, including all 28 layers and the LM head, but
uses randomized weights and an initialized prior KV cache. It is a
**shape/semantics tier**, not checkpoint accuracy. Checkpoint validation and
the other architecture oracles are future promotion gates described in the
[catalog](MODEL_STEP_TASKS.md).

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
values. The current ready case has a one-GPU-kernel launch budget. See
[`workloads.py`](workloads.py) for tensor names, layouts, and exact reference
math; [`cases.py`](cases.py) specifies shapes and tolerances. The included
[`reference_submission.py`](examples/reference_submission.py) demonstrates
the API and CPU correctness path. Its many PyTorch GPU launches fail the
megakernel launch gate.

Use the [agent task brief](AGENT_TASK.md) when comparing implementations.

## Run

```bash
.venv/bin/python -m unittest megabench.test_suite
.venv/bin/python -m megabench list --suite all
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m megabench evaluate \
  --submission /absolute/path/to/solution.py \
  --case dense-step-qwen3-06b-b1-s128 --reps 20
```

Choose a visible GPU you are authorized to use. `--device cpu` performs a
correctness-only check; it does not create a GPU score. Each case runs in a
fresh subprocess with a timeout. Results are exclusive-create JSONL under
`megabench/runs/` by default, or at `--output`; existing files are never
overwritten. Both `runs/` and the legacy archive are ignored by Git.

To isolate each case in a fresh Docker container, supply an existing local
image with Python, PyTorch, and the submission compiler/runtime:

```bash
.venv/bin/python -m megabench evaluate \
  --submission /absolute/path/to/solution.py \
  --case dense-step-qwen3-06b-b1-s128 --device cuda:0 \
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
geometric-mean speedup is emitted only if **every selected case** passes
correctness and the launch gate. Planned cases, CPU checks, and partially
solved suites have no aggregate score. Eager and graph references are local
baselines; a public ranking would additionally need optimized serving
baselines, repeated isolated regrades, and checkpoint-tier checks.
