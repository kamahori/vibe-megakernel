# Accepted Forge P0 candidate snapshots

These source-only snapshots preserve the three Codex-generated candidates that
the Forge campaign kept after an independent MegaBench GPU gate and reviewer
pass. Each directory has a `submission.py` entry point and its CUDA source.
The files were copied byte-for-byte from the final accepted campaign
workspaces; compiled extensions, caches, private workspace Git histories,
and benchmark input data are excluded.

| Directory | MegaBench P0 case | Campaign round | CUDA-event p50 on B200 | Gate result |
| --- | --- | --- | ---: | --- |
| `dense_qwen3/` | `dense-step-qwen3-06b-b1-s128` | 2026-10-02 prompt campaign, round 2 | 3.686 ms | Correct, one launch; keep |
| `gemma_w4/` | `quant-step-gemma3-4b-w4-b1-s128` | 2026-10-02 five-case campaign, round 1 | 30.406 ms | Correct, one launch; keep |
| `eagle3_target/` | `spec-target-step-llama31-8b-k4` | 2026-10-02 five-case campaign, round 1 | 629.000 ms | Correct, one launch; keep |

These were synthetic-weight MegaBench checks, not the Forge paper's
checkpoint, 1000-token drift, MBU, or serving-level gates. Timing was recorded
on a shared B200 host and is provisional. The MoE and Gemma W8 candidates are
not included because the campaign reverted them or did not accept their
review. The original result JSONL files and review ledgers remain under the
ignored `megabench/experiments/2026-10-02/` campaign directories. See the
parent `README.md` for the full status and limitations.

To evaluate one snapshot, pass its `submission.py` to MegaBench from the
repository root on a CUDA host. For example:

```bash
.venv/bin/python -m megabench evaluate \
  --case dense-step-qwen3-06b-b1-s128 \
  --submission reproductions/ForgeMegakernel/p0_candidates/dense_qwen3/submission.py \
  --method forge-codex-gpt6sol --session-id forge-dense-snapshot \
  --output /tmp/forge-dense-snapshot.jsonl
```

Use a fresh output path for each run. The dense and Gemma W4 candidates build
PyTorch CUDA extensions at first use; the EAGLE3 candidate compiles its CUDA
library with `nvcc`.
