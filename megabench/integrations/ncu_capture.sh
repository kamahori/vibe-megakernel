#!/bin/sh
# NCU_PATH wrapper for the linked ncu-mcp server. Keep report-import commands
# unchanged; only live captures need the CUDA profiler start/stop markers.
NCU_BIN=${MEGABENCH_NCU_BIN:-ncu}
case "$1" in
  -i|--import|--list-*|--query-*) exec "$NCU_BIN" "$@" ;;
  *) exec "$NCU_BIN" --profile-from-start off "$@" ;;
esac
