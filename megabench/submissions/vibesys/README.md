# VibeSys megakernels

Each folder holds one megakernel that a VibeSys run wrote, at the commit that the run selected as its winner.
Each folder contains only the source files that `submission.py` loads.
The build caches, dev scripts, and run notes stay in the run workspace.

| Case | Regime | Best (ms) | Rounds | Winner commit | Main files |
|---|---|---|---|---|---|
| `vla-action-step-pi05-b1` | pi05 action step, batch 1, KV 968 | 8.23 | 6 | `0935f63` (round 6) | `pi05_kernel.cu` |
| `quant-step-gemma3-4b-w8-b1-s128` | Gemma 3 4B, 34 layers, W8A16 decode, batch 1 | 1.66 | 5, plus 1 round done again | `06f0f47` (round 6) | `mk/megakernel.cu` |
| `moe-step-qwen3-30b-a3b-b1-s128` | Qwen3-30B-A3B, 48 layers, MoE decode, batch 1 | 2.25 | 5 | `974d4a3` (round 5) | `mk.cu` |
| `world-frame-step-waypoint15-1b-f128` | Waypoint-1.5-1B frame step, 512 tokens, frame 128 (KV 8192) | 22.06 | 5 | `4242e67` (code of round 4) | `mk/ext.cu`, `mk/*.cuh` |

The times are the p50 values of the official VibeSys gate on one B200.
Each candidate passed the framework accuracy gate in its run.
The Qwen3 gate jobs of rounds 1 to 3 stopped when the job had no more memory, so the first official Qwen3 time is from round 4.
The round 5 of Waypoint did not make a faster kernel, so the winner has the code of round 4.

## Engines and decompositions

- pi05: tcgen05 swap-AB with stream-K for the 50-token GEMMs, a fused flash attention, mma.sync tf32 split-K for the 1-row fp32 projection, SIMT for the small fp32 layers.
- Gemma 3: FFMA and mma.sync GEMVs on 16-column CTA tiles with a bulk-copy ring, stream-K for the down projection, SIMT split-KV attention on 32 CTAs with done counters in place of grid barriers.
- Qwen3: tcgen05 swap-AB with stream-K for the attention projections, tcgen05 whole tiles for each expert down projection, SIMT for gate/up and the LM head, SIMT attention on 32 CTAs, a split-K router with fixed-point reduction.
- Waypoint: tcgen05 for all GEMMs and for flash attention (S and O in TMEM), split-K 2 with red.v4 for wo and fc2, SIMT for the residual RMSNorm, the conditioning GEMV, and the output head.

## How to run

Run the MegaBench evaluator in a Slurm GPU job, with the folder name as the case ID:

```bash
.venv/bin/python -m megabench evaluate \
  --submission "$PWD/megabench/submissions/vibesys/moe-step-qwen3-30b-a3b-b1-s128/submission.py" \
  --case moe-step-qwen3-30b-a3b-b1-s128 --method vibesys --session-id check-1 \
  --reps 20 --timeout 900
```

The first call compiles the kernel into a cache folder next to `submission.py`.
