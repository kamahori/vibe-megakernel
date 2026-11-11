#!/bin/bash
# Emit one line per new round artifact, error or exit for each run named.
# usage: watch.sh RUN_NAME...
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
B=$VS_RUNS
declare -A seen alive
for n in "$@"; do alive[$n]=1; done
while true; do
  for n in "$@"; do
    R=$B/$n; ws=$(ls -d $R/*/progress 2>/dev/null | head -1)
    for f in $ws/plans/round-*.json $ws/microbench/answers/round-*.md $ws/pareto-frontier.md; do
      [ -f "$f" ] || continue
      k="$n:$f:$(stat -c %Y "$f")"
      [ -n "${seen[$k]}" ] && continue
      seen[$k]=1
      echo "$n: $(basename $(dirname $f))/$(basename $f) updated"
    done
    if [ -f $R/vibesys.log ]; then
      c=$(grep -ciE "usage limit|rate.?limit|Traceback|out of extra usage|fatal" $R/vibesys.log)
      if [ "${seen[$n:err]:-0}" != "$c" ]; then
        seen[$n:err]=$c
        [ "$c" -gt 0 ] && echo "$n: log has $c error/limit lines; last: $(grep -iE 'usage limit|rate.?limit|Traceback|out of extra usage|fatal' $R/vibesys.log | tail -1 | cut -c1-200)"
      fi
    fi
    if [ "${alive[$n]}" = 1 ] && ! pgrep -f "^/bin/bash [^ ]*runs/$n/launch.sh" >/dev/null; then
      alive[$n]=0; echo "$n: EXITED; log tail: $(tail -2 $R/vibesys.log | tr '\n' ' ' | cut -c1-300)"
    fi
  done
  sleep 30
done
