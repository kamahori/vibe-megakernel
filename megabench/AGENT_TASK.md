# Standard task brief for a whole-model megakernel agent

Implement a ready [MegaBench case](cases.py) in a Python submission file.
Export `build(case: dict) -> run(inputs: dict) -> outputs: dict`. Read
[`workloads.py`](workloads.py) for exact input and output semantics and the
[task catalog](MODEL_STEP_TASKS.md) for model architecture and timed boundary.
You may use Triton, CUDA C++/extensions, TIRx, another GPU DSL, or a Python
loader for compiled kernels.

The ready `dense-step-qwen3-06b-b1-s128` task is a complete 28-layer
Qwen3-0.6B decode step. The timed `run` must execute embedding, every attention
and MLP layer, final normalization, LM head, greedy next-token selection, and
new K/V generation for every layer, from the supplied token, BF16 weights,
and prior K/V state. The current task has a one-GPU-kernel launch budget.
Input weights and cache contents change across correctness trials. Do not
mutate inputs, call the PyTorch oracle, cache answers, replay a multi-kernel
CUDA Graph, or move model computation to the CPU. You may specialize on public
shapes, types, and case parameters.

Work is complete when held-out correctness trials pass and a device trace
shows the required fusion boundary. Report submission source, compiler and
runtime versions, GPU, launch count, cold/JIT time, warm host and CUDA-event
latency, and eager and CUDA Graph reference baselines where available. A
passing automatic launch audit is provisional until separate source and trace
review confirms that the full model ran inside the launch.

```bash
.venv/bin/python -m megabench list --suite all
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m megabench evaluate \
  --submission /absolute/path/to/your_solution.py \
  --case dense-step-qwen3-06b-b1-s128 --reps 20
```

`--suite core` selects all ready tasks. P0–P3 and `planned` list future tasks
but do not score them until their reference contracts are implemented. A
`--device cpu` run checks correctness only and has no megakernel score. For
containerized evaluation, add `--docker-image IMAGE` and
`--docker-gpus device=N`; the image must already be local and include PyTorch and your kernel
runtime.
