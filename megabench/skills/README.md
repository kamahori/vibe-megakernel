# Megakernel guidance skills

Method guidance for megakernel work, one skill per directory:

- `megakernel-phase-profiling`: split the kernel into phases, time them, and gate codegen
  changes (`scripts/codegen_gate.py`).
- `megakernel-microbench`: study the slowest or furthest-from-floor phase in isolated
  microbenchmarks; these do not use rounds.
- `blackwell-tensor-core-design`: evaluate Blackwell tensor-core designs.
- `megakernel-regimes`: decide whether to specialize the kernel per batch size or KV length, using
  trace-derived regimes (`./trace_regimes`).

`make_agent_workspace.py` copies this directory into every workspace as `megakernel_skills/`.

VibeSys loads the skills natively; `.vibesys.toml` routes them to the `cuda` backend and the
`kernel-writing` domain:

```
vibesys ... --skills-dir <vibesys>/resources/skills/<name> ...  # each preset skill you want
            --extra-skills <repo>/megabench/skills               # plus these skills
```

Name the preset skills explicitly. `--extra-skills` alone keeps the whole preset catalog,
which can include skills written from earlier runs. For an isolated run, use
`python -m megabench.integrations.clean_vibesys_run`: it builds a pruned trusted snapshot,
an empty agent `HOME`, a scrubbed environment and explicit skills, then checks the sandbox.
VibeSys then runs on the host without GPUs. Each GPU use is one Slurm job: the gates run
under `srun`, and the agent runs GPU commands with `./gpu_run`, in its own sandbox in the job.

`../integrations/vibesys-agent.example.toml` splits roles: orchestrator, profiler and judge
inherit `[model]`/`[thinking]`, while `[agent.roles.implementer]` uses a cheaper model.
