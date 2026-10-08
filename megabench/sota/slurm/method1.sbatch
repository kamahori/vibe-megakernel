#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=01:30:00
#SBATCH --job-name=sota-method1
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/runs/slurm-%j.log
# Method 1: one MPK decode step vs. the Opus megakernel at b1-s128.
# Extra arguments go to `python -m megabench.sota.method1 run`,
# e.g. `sbatch method1.sbatch --smoke --tag method1-smoke`.
set -euo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh
export MPK_COMPILE_ROOT=/raid/garv901/.cache/megabench_sota/mpk_builds
mkdir -p "$MPK_COMPILE_ROOT"

nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

.venv/bin/python -m megabench.sota.method1 run "$@"
