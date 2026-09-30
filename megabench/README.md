# MegaBench: benchmark for megakernel-writing agents

MegaBench evaluates an **agent's submitted implementation**, not any kernel
already in this repository. Each challenge supplies an input generator, an
independent PyTorch oracle, multiple hidden-at-submission-time correctness
trials, a launch budget, and a timed eager baseline. An agent can use Triton,
CUDA C++, TIRx, another DSL, or a Python loader for an extension as long as it
implements the submission protocol below. The suite imports no local TIRx
implementation, checkpoint, or reproduction checkout.

Use the [standard agent task brief](AGENT_TASK.md) when comparing agents.

## First challenge set

| Family | Cross-stage work | Swept dimensions |
|---|---|---|
| Dense decoder | RMSNorm, Q/K/V, KV append, attention, output projection, SwiGLU MLP, residual | B1/4, context 32/128 |
| Routed MoE | Router top-2, expert gather, up/activation/down, weighted combine | B4/16, experts 4/8 |
| Quantized MLP | RMSNorm, int8 or packed signed-int4 dequantization, SwiGLU, residual | W8A16/W4A16, B1/8 |
| Speculative verification | Draft argmax, target verification, accepted-prefix scan, fallback/bonus token commit | K2/4/8, B1/8 |
| Iterative stencil | Several dependent periodic diffusion steps | 4/8 steps, width 128/512 |

The speculative challenge is a **draft/verify/commit component**, not an
end-to-end speculative LLM: draft and target logits are inputs. It reports
proposed, accepted, and committed token counts, and never credits rejected
draft tokens as throughput. W4 uses two signed 4-bit weights per byte with a
per-output-row FP32 scale; W8 uses signed int8 with the same scale convention.
The model-like cases use synthetic weights, allowing agents to specialize
shape and data type but not answers. Two TP/EP cases are cataloged as
`planned`; they are not included in the runnable score until a multi-process
input/output and communication audit is implemented.

## Submission contract

Provide a Python file exporting:

```python
def build(case: dict):
    # Compile or allocate reusable state here. `case` gives only public metadata.
    def run(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Return exactly the keys, shapes, and dtypes specified by the oracle.
        ...
    return run
```

`run` may allocate output tensors but must not mutate inputs. It must compute
from **current** input values; the evaluator changes values across trials.
The callable is expected to perform all dependent stages within the case's
one-GPU-kernel launch budget. A launch may contain many CTAs and any valid
in-kernel synchronization scheme. Compilation/initialization is excluded from
steady-state timing and reported separately. See
[`workloads.py`](workloads.py) for tensor names, layouts, and precise math;
[`cases.py`](cases.py) fixes shapes, tolerances, and launch budgets.

The included [`reference_submission.py`](examples/reference_submission.py)
shows the API but is intentionally **not** a megakernel. Its PyTorch oracle
calls should pass correctness and fail the CUDA launch gate. The
[`triton_stencil.py`](examples/triton_stencil.py) example implements only the
stencil family with one Triton launch.

## Run on this server

```bash
.venv/bin/python -m unittest megabench.test_suite
.venv/bin/python -m megabench list --suite all
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m megabench evaluate \
  --submission megabench/examples/triton_stencil.py \
  --case stencil-b4-n128-t4 --reps 20
```

Use `--suite smoke` (five families), `--suite core` (all 13 runnable cells),
or repeat `--case ID` for a subset. `--device cpu` runs a correctness-only
protocol check; CUDA timing and launch audit require a visible GPU. Each cell
runs in a **fresh subprocess** with a timeout. Results are exclusive-create
JSONL under `megabench/runs/` by default, or at `--output`; existing files
are never overwritten. The old TIRx-specific JSONL traces remain locally in
`megabench/results/` but are ignored by Git and are **not agent scores**.

## Evaluation and score

For each case, the harness uses fresh randomized inputs and checks every
output key, shape, dtype, device, and value against the PyTorch oracle;
integer routing/token outputs must match exactly, floating outputs use the
case's explicit tolerance. It rejects input mutation. A first call (including
JIT) is timed separately. The warm path then records both synchronized host
wall time and CUDA-event latency for the candidate and the same-input eager
reference. Where capture succeeds, it also times a CUDA Graph replay of that
reference on the same static input; `--no-graph-baseline` disables this
diagnostic. The provisional speedup uses the **faster available** baseline
(eager or graph), divided by candidate median CUDA-event time. Per-case
p50/p95 and all raw samples are saved. Spec cases also save draft acceptance
and committed-token counts.

The [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler) counts
observed GPU kernel executions in one steady-state `run`. More than the case's
one-launch budget—or zero kernels—sets `non_megakernel`, regardless of speed.
CUDA Graph replay of many kernels therefore should not pass simply because
the host called `replay()` once. This failure mode is documented in the
[KernelBench-Mega devlog](https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/mega/DEVLOG.md).
The source file hash and advisory graph/compile/oracle hints are recorded;
string hints are **not** a secure authenticity test. Even a one-launch trace
is marked `ok_provisional`: source/trace review is still required to verify
the work is genuinely fused and not answer-cached or delegated. If profiling
is unavailable, authenticity is `unverified`, not assumed to pass.

A provisional geometric-mean speedup is emitted **only if every selected
cell** passes correctness and the launch gate. CPU runs, partially solved
suites, planned cases, and unverified authenticity have no aggregate score.
The eager/graph references are reproducible starting baselines, not claims
against best-in-class libraries. For public rankings, add a same-hardware
state-of-the-art implementation, repeated isolated regrades, and independent
source/trace review. This local process boundary limits crash propagation but
does **not** sandbox untrusted Python; run third-party agent submissions in a
container or another security boundary.
