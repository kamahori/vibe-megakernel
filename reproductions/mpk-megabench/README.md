# MPK on MegaBench

This addition contains the Mirage MPK integration probes and their result
reports. The probes live in `megabench/probes/mpk_*.py` so they can import the
MegaBench cases and correctness checker without changing the suite harness.

The experiments used Mirage's `mpk` branch at commit
`6ce3a6b836a69ddc91742798166cdae2ef498cfc`. The upstream checkout and
compiled MPK installation are dependencies, not vendored into this repository.
Place the source checkout at `reproductions/mirage-mpk` and install it using
its upstream `INSTALL.md`; the native Qwen3 probe reads the builder from that
source path. Use a CUDA-capable GPU and an MPK installation on `PYTHONPATH`.

From the repository root, choose new output paths for each run:

```bash
.venv/bin/python -m megabench.probes.mpk_relaxed_eval \
  --case dense-step-qwen3-06b-b1-s128 \
  --output /tmp/mpk-hybrid-dense.json --trials 3

.venv/bin/python -m megabench.probes.mpk_native_qwen3_probe \
  --output /tmp/mpk-native-dense.json --layers 28 --trials 3 --reps 5

.venv/bin/python -m megabench.probes.mpk_mlp_bench \
  --case dense-step-qwen3-06b-b1-s128 \
  --output-dir /tmp/mpk-mlp-dense
```

The native whole-model command exits nonzero when its strict MegaBench
numerical check fails; it still writes the diagnostic JSON. The hybrid command
uses a relaxed launch audit and does not qualify as a one-launch MegaBench
submission. Results and limits are documented in
`megabench/docs/MPK_NATIVE_QWEN3_TRIAL.md`,
`megabench/docs/MPK_FULL_BASELINE.md`, and
`megabench/docs/MPK_RELAXED_TRIAL.md`.
