#!/bin/bash
# Launch the provenance-credit POC generation across N parallel shards on this node.
# Each shard: own MCP port + own output jsonl; SHARED trajectories dir (per-id filenames
# never collide). API-bound (gemini agent), so parallelism is just more in-flight items.
#
#   NSHARDS=4 OUT=runs/provenance_poc ./run_provenance_poc_sharded.sh
set -uo pipefail

DRRLM=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
HERE="$DRRLM/agent"
NSHARDS=${NSHARDS:-4}
OUT=${OUT:-$DRRLM/runs/provenance_poc}
BASE_PORT=${BASE_PORT:-8050}
LOGDIR=$DRRLM/runs/logs
mkdir -p "$OUT/trajectories" "$LOGDIR"

pids=()
for s in $(seq 0 $((NSHARDS-1))); do
  port=$((BASE_PORT + s))
  OUT="$OUT" \
  NUM_SHARDS="$NSHARDS" SHARD="$s" MCP_PORT="$port" \
  bash "$HERE/gen_provenance_poc.sh" > "$LOGDIR/poc_shard${s}.out" 2>&1 &
  pids+=($!)
  echo "[launch] shard $s -> port $port pid ${pids[-1]} log $LOGDIR/poc_shard${s}.out"
  sleep 3   # stagger MCP startups
done

echo "[launch] ${NSHARDS} shards running; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[done] all shards exited (failures=$fail)"

# merge per-shard outputs (trajectories already share the dir)
cat "$OUT"/drtulu_rl.shard*.jsonl > "$OUT/drtulu_rl.jsonl" 2>/dev/null || true
echo "[merge] $(wc -l < "$OUT/drtulu_rl.jsonl" 2>/dev/null) rows -> $OUT/drtulu_rl.jsonl"
echo "[trajectories] $(ls "$OUT/trajectories"/traj_*.jsonl 2>/dev/null | wc -l) trees captured"
