#!/bin/bash
# kill_on_stall.sh JOBID STALL_MIN LIVENESS_GLOB [LIVENESS_GLOB ...]
# SBU guard: scancel JOBID if its liveness files stop growing for STALL_MIN while the
# job is RUNNING. Liveness = the vLLM serve log(s) (grow on every completion/request) and
# the SLURM .out. Complements the in-job guards (serve-death exit, driver dead-search
# fail-fast) by catching the case they miss: a hung/stuck run silently burning walltime.
# Poll every 2 min; ignores PENDING (queue wait costs nothing).
set -uo pipefail
JOBID="$1"; STALL_MIN="$2"; shift 2; GLOBS=("$@")
INT=120; last=-1; stalled=0
echo "[watchdog $JOBID] armed: kill if no liveness growth for ${STALL_MIN}min. globs: ${GLOBS[*]}"
while :; do
  st=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)
  [ -z "$st" ] && { echo "[watchdog $JOBID] left queue — exiting"; break; }
  if [ "$st" = "RUNNING" ]; then
    cur=$(stat -c%s ${GLOBS[@]} 2>/dev/null | awk '{s+=$1} END{print s+0}')
    if [ "${cur:-0}" -gt "$last" ]; then last=$cur; stalled=0
    else
      stalled=$((stalled+INT))
      if [ "$stalled" -ge $((STALL_MIN*60)) ]; then
        echo "[watchdog $JOBID] STALLED ${STALL_MIN}min (no liveness growth) — SCANCEL to save SBU"
        scancel "$JOBID"
        break
      fi
    fi
  fi
  sleep "$INT"
done
