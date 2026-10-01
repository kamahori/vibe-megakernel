# Standard task brief for a whole-model megakernel agent

Implement exactly one ready [MegaBench case](../cases.py) in a Python submission
file. Start a fresh agent session and candidate checkout for each case; do not
carry candidate code or chat context from another case. The case ID must be
fixed in the session objective before the agent starts.
Export `build(case: dict) -> run(inputs: dict) -> outputs: dict`. Read
[`tasks/`](../tasks/) for exact input and output semantics and the
[task catalog](MODEL_STEP_TASKS.md) for model architecture and timed boundary.
You may use Triton, CUDA C++/extensions, TIRx, another GPU DSL, or a Python
loader for compiled kernels.

The five independent P0 cells cover full Qwen3-0.6B dense decode, Qwen3-30B-A3B
MoE decode, Gemma 3 4B W8A16 and W4A16 decode, and Llama 3.1 8B target
verification of a linear EAGLE3 proposal chain. Each timed `run` must execute
all stated model layers and return the oracle's logits, tokens, and state
writes. All current P0 cells have a one-GPU-kernel launch budget. Input
weights, token IDs, prior caches, and draft proposals change across trials;
speculative trials also force full acceptance and rollback. Do not
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
.venv/bin/python -m megabench list --suite p0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m megabench evaluate \
  --submission /absolute/path/to/your_solution.py \
  --case dense-step-qwen3-06b-b1-s128 \
  --method plain-codex --session-id unique-session-name \
  --reps 20 --timeout 900
```

Run one such command per independently produced candidate. Use `aggregate`
with one result JSONL per case only after all case sessions are finished.
P1–P3 and `planned` list future tasks that do not score until their reference
contracts are implemented. A
`--device cpu` run checks correctness only and has no megakernel score. For
containerized evaluation, add `--docker-image IMAGE` and
`--docker-gpus device=N`; the image must already be local and include PyTorch and your kernel
runtime.
