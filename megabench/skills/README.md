# Megakernel guidance skills

Method guidance for megakernel work, one skill per directory:

- `megakernel-design`: list the design options for synchronization, overlap, fusion, residency,
  GEMV and GEMM, attention, sequential inner loops and work distribution, with the questions that
  decide among them.
- `megakernel-phase-profiling`: split the kernel into phases, time them, and gate codegen
  changes (`scripts/codegen_gate.py` on cuda, `scripts/codegen_gate_amd.py` on rocm).
- `megakernel-microbench`: study the slowest or furthest-from-floor phase in isolated
  microbenchmarks; these do not use rounds.
- `blackwell-tensor-core-design`: evaluate Blackwell tensor-core designs (cuda backend only).
- `megakernel-regimes`: decide whether to specialize the kernel per batch size or KV length, using
  trace-derived regimes (`./trace_regimes`).

`make_agent_workspace.py` copies these skills into every workspace as `megakernel_skills/`, filtered by
backend (`--backend cuda|rocm`): a skill is copied only if its `backends` in `.vibesys.toml` lists the
backend, and files listed for one backend in `backend_files.toml` (for example `codegen_gate.py` for
cuda and `codegen_gate_amd.py` for rocm) are left out of other backends' workspaces.

Optional skills live in `../skills-optional/` and are copied only on request
(`--optional-skill NAME`). `megakernel-stage-linking` is optional: it splits a megakernel into
contract-checked stage packages (PipeLink-style, `scripts/plk_link.py`). On MI300X the split did not
improve performance: a separately compiled and linked seam cost 1.6-2.4%, and stages swapped between
independently written kernels were slower or neutral.

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

`../integrations/vibesys-agent.example.toml` splits roles: orchestrator, profiler and judge
inherit `[model]`/`[thinking]`, while `[agent.roles.implementer]` uses a cheaper model.
