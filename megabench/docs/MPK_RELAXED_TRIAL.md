# MPK with a relaxed launch limit on MegaBench P0

**Later work:** A full-step MPK-assisted hybrid now has passing MegaBench
checks; see `MPK_FULL_BASELINE.md` for its measured results. The findings below
describe the earlier native-adapter and component trial.

Experiment: 2026-10-01, one NVIDIA B200 (physical GPU 0). Mirage MPK is the
local `mpk` source checkout at `reproductions/mirage-mpk`, commit `6ce3a6b`.
The installation is under the ignored
`megabench/experiments/mpk-install.ehi7Pu/site` directory. MPK's native
offline call and request reset are allowed to use four GPU launches in this
trial.

## Outcome

**No complete P0 case has a valid MPK result yet.** Relaxing the launch limit
removes one barrier, but the installed MPK model demos do not implement the
MegaBench per-call input/output contract. In particular, every correctness
trial supplies fresh weights and KV tensors; `PersistentKernel.attach_input`
records the tensor pointers when the graph is built, and `__call__` only
launches it. A new tensor passed as a keyword argument does not rebind the
compiled graph. Copying values into the attached buffer works, but every
offline request must also reset the MPK request state. The native Qwen3 and
MoE demos process an MPK request lifecycle,
while MegaBench times one step and checks logits, token ID, and separate K/V
writes. A native demo's generation throughput is therefore not a MegaBench
latency. No full-case MPK/VibeSys speedup is claimed.

We compiled and measured the three-stage MPK gate/up, SiLU, down-plus-residual
MLP graph at three P0 model geometries. This tests a real MPK component and
the relaxed launch policy, **not** the full tasks. The inputs are new synthetic
BF16 tensors with the same dimensions as each case. The reference uses FP32
matmul/activation and a BF16 MPK output; the largest absolute errors were
below 0.003 in all three runs. The component probe asserts this tolerance on
both the first and a reset request before timing. Each CUDA-event median is
from 15 reset-and-run calls after three warmups; compilation and weight
shuffling are outside timing. The reset call synchronizes on the host and
launches `init_kernel`; it is included in these numbers. MPK's `compile()`
path does not expose the reset method on `PersistentKernel`, so the probe
accesses the compiled launcher's exported `init_request_func` through
`pk.init_func.__self__`. That private access is a trial workaround.

| P0 geometry | MPK MLP CUDA-event p50 | Host p50 | GPU launches | Max abs error |
| --- | ---: | ---: | ---: | ---: |
| Dense Qwen3 0.6B, B1/H1024/I3072 | 0.164 ms | 0.174 ms | 4 | 0.00176 |
| MoE Qwen3 30B-A3B, one B1/H2048/I768 expert MLP | 0.162 ms | 0.173 ms | 4 | 0.00196 |
| Llama 3.1 target, B5/H4096/I14336 MLP | 0.261 ms | 0.271 ms | 4 | 0.00205 |

All three profiled reset-and-run calls executed MPK's `init_kernel`,
`prepare_kernel`, `worker_kernel`, and `scheduler_kernel`. Raw measurements
and generated launchers are in the ignored
`megabench/experiments/2026-10-01/23-35-00-mpk-mlp-reset/` directory.
The initial `22-45-00-mpk-mlp` timings are **invalid and superseded**: they
called `pk()` repeatedly without resetting the offline request, so later
calls did not recompute. A corrected RMSNorm probe in
`megabench/experiments/2026-10-01/23-30-00-mpk-reset-probe/` confirmed that
resetting recomputes the output, copying into the attached buffer changes the
output, and passing a new tensor as a `pk()` keyword argument does not.

## Full-case compatibility findings

| MegaBench P0 case | Direct MPK path at this revision | Missing work for a valid comparison |
| --- | --- | --- |
| Dense Qwen3 0.6B | MPK Qwen3 model builder and native generation demo exist. | Bind fresh synthetic weights/token/KV for each `run(inputs)`; expose FP32 logits and separate K/V writes; check all 28 layers against the MegaBench oracle. |
| MoE Qwen3 30B-A3B | Native MoE demo and MLP/routing tasks exist. | Adapt 48-layer synthetic BF16 fixture and fresh pointers; expose the MegaBench outputs and verify its routing/cache semantics. The measured single-expert MLP omits routing and the other experts. |
| Gemma 3 W8A16 | No Gemma builder or exact signed-INT8 plus per-row-FP32-scale projection was found in the installed MPK Python API. | Implement exact quantized projection and Gemma attention/normalization; pre-dequantizing to BF16 would change the task. |
| Gemma 3 W4A16 | No Gemma builder or exact packed-INT4 projection was found. | Implement packed-nibble dequantization with the oracle's per-row scales and the complete Gemma step. |
| Llama 3.1 + EAGLE3 target verification | MPK has an EAGLE3 draft builder for a Qwen3 MoE model, and the common MLP graph runs at the target geometry. | Implement the Llama target verification call, five-token causal attention and outputs (including acceptance/commit), using the supplied proposals and fresh synthetic state. |

The corresponding MegaBench definitions are in `megabench/cases.py` and
`megabench/tasks/`. MPK APIs and builders inspected: `persistent_kernel.py`,
`models/qwen3/builder.py`, `models/eagle3/builder.py`, and
`demo/qwen3/demo_30B_A3B.py` in the local MPK checkout. These are adapter
and implementation gaps in this pinned build, not evidence that MPK cannot
support these tasks after extension.

## Comparison with the latest completed VibeSys grades

The completed ten-round VibeSys campaign measured the **full P0 calls** on
B200: dense 4.509 ms (separate verified M8 archive 3.518 ms), MoE 41.477 ms,
Gemma W8 15.864 ms, Gemma W4 8.687 ms, and EAGLE3 target 80.331 ms. All
five passed MegaBench correctness and its one-launch audit. Source:
`megabench/experiments/2026-10-01/23-02-20-vibesys-ncu-p0-10rounds/reports/campaign-summary.md`.
The rounds 11–30 continuation in
`megabench/experiments/2026-10-02/04-49-25-vibesys-ncu-p0-plus20/` had no
completed final grades at this report's write time. The MPK MLP numbers above
cannot be divided into those full-task values to infer a speedup.

## Reproduction

From the repository root, choose an idle authorized B200 and a new output
directory for each run:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/megabench/experiments/mpk-install.ehi7Pu/site" \
MIRAGE_HOME="$PWD/megabench/experiments/mpk-install.ehi7Pu/src" \
PATH="/usr/local/cuda/bin:$PATH" \
.venv/bin/python -m megabench.probes.mpk_mlp_bench \
  --case dense-step-qwen3-06b-b1-s128 \
  --output-dir megabench/experiments/my-new-mpk-dense-run
```

The script also accepts the MoE and EAGLE3 case IDs shown in `GEOMETRIES`.
Use `megabench/probes/mpk_launch_probe.py` to rerun the RMSNorm launch check.
No canonical MegaBench contract or VibeSys artifact was modified for this
trial.
