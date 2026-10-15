# loadstream: a kit for load-centric decode megakernels

Header-only CUDA primitives plus Python helpers for writing a single-launch
megakernel as a statically known byte stream per SM: a producer warp streams
weights through a shared-memory TMA ring while consumer warps (taps) compute,
gated by event counters instead of grid barriers. Design notes:
`LOAD_CENTRIC_PLAN.md` at the repository root. Agent guidance:
`skill/load-centric-megakernel/SKILL.md`.

| Path | Contents |
|---|---|
| `include/loadstream/` | `ring.cuh`, `event.cuh`, `tap.cuh`, `stamp.cuh`, `plan.cuh`, `ptx.cuh` (`ls.cuh` includes all) |
| `loadstream/` | `build.py` (nvcc → cached `.so`), `plan.py` (byte ledger), `timeline.py` (stamp report) |
| `examples/mlp_chain/` | Worked example: L residual MLP blocks, verified against PyTorch, with SOL report and timeline |
| `tests/` | GPU unit tests for the ring and every tap |
| `bench/` | B200 calibration microbenchmarks; results in `CALIBRATION.md` |

Run from this directory on a GPU node (sm_100a, CUDA 13.x):

```bash
python -m tests.run_tests
python -m examples.mlp_chain.run [--profile]
python -m bench.run_micro
python -m bench.run_taps
```

`megabench.make_agent_workspace --kit loadstream` copies this directory into an
agent workspace as `loadstream_kit/`.
