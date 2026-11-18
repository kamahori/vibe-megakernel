#!/bin/bash
# Build the AMD probes for the current GPU and run each once.
# Usage: bash megabench/primbench/amd/run_amd.sh OUT_DIR   (raw JSON lines per probe in OUT_DIR)
# Then:  python -m megabench.primdb add OUT_DIR/*.jsonl   (or the importer used for the seed)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: run_amd.sh OUT_DIR}
mkdir -p "$out/bin"
rocminfo > "$out/rocminfo.txt"
arch=$(grep -o 'gfx[0-9a-f]*' "$out/rocminfo.txt" | sort | uniq -c | sort -rn | awk 'NR==1{print $2}')
rocm-smi --showproductname --showclocks > "$out/gpu.txt" 2>&1 || true
hipcc --version > "$out/hipcc.txt" 2>&1 || true
echo "$arch" > "$out/arch.txt"
for b in sync_amd mem_amd mfma_amd gemv_amd; do
  hipcc -O3 -std=c++17 --offload-arch="$arch" -I"$here" -o "$out/bin/$b" "$here/$b.hip"
done
for b in sync_amd mfma_amd mem_amd gemv_amd; do
  echo "== $b $(date +%T)" >&2
  "$out/bin/$b" > "$out/$b.jsonl"
done
wc -l "$out"/*.jsonl
