#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:25:00
#SBATCH --job-name=mpk-demo-check
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/runs/slurm-%j.log
# Cross-check: MPK's stock demo/qwen3/demo.py (its own hand-written graph, real
# Qwen3-0.6B weights, demo defaults mbt=8 page=4096 CUTLASS on), as the paper's
# B200 artifact runs it (artifact_evaluation/B200/run_tgx.sh), at several
# --max-seq-length values. Usage: sbatch mpk_demo_check.sbatch OUT_DIR
set -euo pipefail
OUT=${1:?usage: sbatch mpk_demo_check.sbatch OUT_DIR}
mkdir -p "$OUT"
source /raid/garv901/vibe-megakernel/megabench/sota/slurm/env.sh
MPK=/raid/garv901/vibe-megakernel/reproductions/mirage-mpk
PY=/raid/garv901/mpk/venv/bin/python
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
for MSL in 193 257 1088; do
  for REP in 1 2; do
    W="$OUT/demo-msl$MSL-rep$REP"; mkdir -p "$W"; cd "$W"
    echo "== msl=$MSL rep=$REP $(date -u +%FT%TZ)"
    "$PY" "$MPK/demo/qwen3/demo.py" --use-mirage --model Qwen/Qwen3-0.6B \
      --max-num-batched-requests 1 --max-seq-length "$MSL" --ignore-eos \
      --save-tokens "$W/tokens.json" > "$W/demo.log" 2>&1 || echo "FAILED msl=$MSL rep=$REP"
    grep -iE "per-token latency|Prompt length" "$W/demo.log" | tail -1
  done
done
