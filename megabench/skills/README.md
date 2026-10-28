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
vibesys ... --extra-skills <repo>/megabench/skills            # adds to the default kernel skills
vibesys ... --skills-dir <repo>/megabench/skills/<name>       # one skill at a time
```

`../integrations/vibesys-agent.example.toml` splits roles: orchestrator, profiler and judge
inherit `[model]`/`[thinking]`, while `[agent.roles.implementer]` uses a cheaper model.
