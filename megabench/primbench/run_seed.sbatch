#!/bin/bash
#SBATCH --job-name=primbench-seed
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=primbench-seed-%j.log
# Submit from the repo root: sbatch megabench/primbench/run_seed.sbatch [OUT_DIR]
cd "$SLURM_SUBMIT_DIR"
bash megabench/primbench/run_seed.sh "${1:-primbench-out/$SLURM_JOB_ID}"
