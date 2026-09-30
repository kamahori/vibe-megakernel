# Standard task brief for a megakernel-writing agent

Implement one or more MegaBench cases in a Python submission file. Export
`build(case: dict) -> run(inputs: dict) -> outputs: dict`. Read
[`cases.py`](cases.py) for public shapes and [`workloads.py`](workloads.py)
for exact reference semantics. Your code may use Triton, CUDA C++/extensions,
TIRx, another GPU DSL, or a thin Python loader for a compiled kernel.

The timed `run` must compute the full challenge output from the current input
values within the case's GPU launch budget (one launch for the first challenge
set). Do not call the PyTorch oracle, cache answers, substitute a CUDA Graph
replay of multiple kernels, use `torch.compile` to hide multiple launches, or
perform substantial computation on the CPU. Specializing on public shapes,
data types, and case parameters is allowed; specializing on test seeds or
particular tensor contents is not. Do not mutate inputs.

Work is complete only if held-out correctness trials pass for every selected
case and a device trace shows the required fusion boundary. Report the
submission source, compiler/runtime versions, GPU, launch count, cold/JIT
time, warm host and CUDA-event latency, and both eager and CUDA Graph reference
baselines where available. The automatic one-launch result is provisional
until a separate source/trace audit checks that the work is genuinely fused.

To evaluate a case on this server:

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m megabench evaluate \
  --submission /absolute/path/to/your_solution.py \
  --case stencil-b4-n128-t4 --reps 20
```

For a multi-case run, use `--suite smoke` or `--suite core`. The two `planned`
TP/EP cells are not scoreable yet. Do not treat CPU correctness-only runs,
unverified profiler runs, or partially solved suites as a benchmark score.
