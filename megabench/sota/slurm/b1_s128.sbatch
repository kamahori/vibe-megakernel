#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:40:00
#SBATCH --job-name=sota-b1s128
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/runs/slurm-%j.log
# Batch 1, KV prefix 128. Extra arguments go to `python -m megabench.sota run`,
# e.g. `sbatch b1_s128.sbatch --smoke --tag smoke`.
set -euo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

.venv/bin/python -m megabench.sota run "$@"
