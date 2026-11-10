#!/bin/bash
# Start a prepared VibeSys run (and its failover watcher) once Slurm can start the run's
# GPU jobs at once: a run's microbench phase has a hard time limit, and GPU jobs that wait
# in the queue use it up without measuring.
#
#   setsid nohup bash megabench/scripts/launch_when_free.sh RUN CASE REF TOTAL_ROUNDS \
#       [failover args...] >> $VS_LOOP_HOME/runs/launch_when_free.log 2>&1 < /dev/null &
#
# Needs slurm-windows (prints when Slurm can start a job of a given size). Paths: env.sh.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/env.sh"
RUN=$1 CASE=$2 REF=$3 TOTAL=$4; shift 4
R=$VS_RUNS/$RUN
limit=$(grep -oE -- "--time=[0-9:]+" "$R/launch.sh" | head -1 | cut -d= -f2)
limit=${limit:-00:20:00}
secs() { IFS=: read -r h m s <<< "$1"; echo $((10#$h * 3600 + 10#$m * 60 + 10#${s:-0})); }
need=$(secs "$limit")
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $RUN: $*"; }
log "waiting until main can start a 1-GPU job of $limit at once"
while true; do
  ok=$(slurm-windows --json 2>/dev/null | python3 -c "
import json, sys
need = int(sys.argv[1])
d = json.load(sys.stdin)
def secs(t):
    h, m, s = (int(x) for x in t.split(':'))
    return h * 3600 + m * 60 + s
for p in d.get('partitions', []):
    if p.get('partition') != 'main':
        continue
    for g in p.get('groups', []):
        if g.get('gpus_min', 1) > 1:
            continue
        for o in g.get('options', []):
            start = str(o.get('start', ''))
            if secs(o.get('max_time', '0:0:0')) >= need and (start == 'now' or start <= d['now'][:16]):
                print('yes'); sys.exit()
" "$need")
  if [ "$ok" = yes ]; then break; fi
  sleep 120
done
log "GPUs free: launching"
setsid nohup "$R/launch.sh" > "$R/vibesys.log" 2>&1 < /dev/null &
sleep 5
setsid nohup python3 "$HERE/failover.py" "$RUN" "$CASE" "$REF" "$TOTAL" "$@" \
  >> "$VS_RUNS/failover.log" 2>&1 < /dev/null &
log "launched run and failover watcher"
