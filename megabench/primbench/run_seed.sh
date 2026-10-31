#!/bin/bash
# Build every primbench family and run it once on the current GPU.
# Usage: bash megabench/primbench/run_seed.sh OUT_DIR   (raw JSON lines per family in OUT_DIR)
# Then: python -m megabench.primdb.importers ... --primbench OUT_DIR -o megabench/primdb/seed.jsonl
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: run_seed.sh OUT_DIR}
mkdir -p "$out/bin"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader > "$out/gpu.txt"
nvcc --version | tail -1 > "$out/nvcc.txt"
python3 "$here/gen_tmem.py" > "$out/tmem.cu"
for b in mma_sync simt tc_mma tc_mma2 dequant tmem sync ingress mem gemm_small gemm_small_tc; do
  src="$here/$b.cu"; [ "$b" = tmem ] && src="$out/tmem.cu"
  nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -I"$here" -o "$out/bin/$b" "$src" -lcuda
done
"$out/bin/simt" > "$out/simt.jsonl"
"$out/bin/mma_sync" > "$out/mma_sync.jsonl"
"$out/bin/tc_mma" > "$out/tc_mma.jsonl"
"$out/bin/tc_mma" overhead > "$out/tc_overhead.jsonl"
"$out/bin/tmem" > "$out/tmem.jsonl"
"$out/bin/dequant" > "$out/dequant.jsonl"
"$out/bin/tc_mma2" > "$out/tc_mma2.jsonl"
"$out/bin/sync" > "$out/sync.jsonl"
"$out/bin/ingress" > "$out/ingress.jsonl"
"$out/bin/mem" > "$out/mem.jsonl"
"$out/bin/gemm_small" > "$out/gemm_small.jsonl"
"$out/bin/gemm_small_tc" > "$out/gemm_small_tc.jsonl"
wc -l "$out"/*.jsonl
