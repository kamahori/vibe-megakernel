#!/bin/bash
# Prepare one isolated VibeSys run; prints the preflight and the launch command but does not
# start the run.
#
#   prep.sh NAME CASE REF TOKEN_FILE [ROUNDS]
#
# SEED_FROM=<workspace> [SEED_REF=<commit>]: start from an earlier run's candidate and
# progress files (with EXTRA_ARGS=--restart, ROUNDS counts the whole chain and the run does
# only the rounds that remain). EXTRA_ARGS: more clean_vibesys_run arguments.
# Paths come from env.sh. The agents cannot see: the other runs, this checkout (they get a
# snapshot of REF), the folder of the token file and every VS_FORBID path.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NAME=$1 CASE=$2 REF=$3 TOKEN=$4 ROUNDS=${5:-5}
RUN=$VS_RUNS/$NAME
forbid=(--forbid "$VS_REPO" --forbid "$(dirname "$(readlink -f "$TOKEN")")")
IFS=: read -ra extra_forbid <<< "$VS_FORBID"
for d in "${extra_forbid[@]}"; do [ -n "$d" ] && forbid+=(--forbid "$d"); done
for d in "$VS_RUNS"/*/; do
  [ -d "$d" ] || continue
  d=${d%/}; [ "$d" = "$RUN" ] || forbid+=(--forbid "$d")
done
cd "$VS_REPO"
"$VS_REPO/.venv/bin/python" -m megabench.integrations.clean_vibesys_run \
  --case "$CASE" --run-dir "$RUN" --oauth-token-file "$TOKEN" \
  --vibesys "$VS_VIBESYS" --vibesys-skills "$VS_SKILLS" \
  --rounds "$ROUNDS" --ref "$REF" --venv "$VS_VENV" --cuda "$VS_CUDA" \
  --max-gpu-jobs "${MAX_GPU_JOBS:-8}" \
  ${SEED_FROM:+--seed-from "$SEED_FROM" --seed-ref "${SEED_REF:-HEAD}"} \
  ${EXTRA_ARGS:-} "${forbid[@]}"
