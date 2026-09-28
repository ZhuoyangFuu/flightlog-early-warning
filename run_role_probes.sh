#!/usr/bin/env bash
# Item 6/8 `emb_role` (8 roles x 512, 4096-d) probes -- the 18 runs that were OOM-killed on
# 15-16 Sept (old loader peaked at ~4x the 112 GiB matrix). Re-run with the preallocating
# loader (probe.load_embeddings, 2026-09-18) and X held in float16 (exact; up-cast at slice).
# Three views dirs sequentially: original encoder (tag vrole), channel-mask (chrole),
# random-mask control (rdrole). horizon.py skips arms whose json+curve+scores exist.
#   bash run_role_probes.sh start | status | tail | stop      (/mnt/wsssd is noexec -> bash)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/role_probes.log"; PIDF="$ROOT/state/role_probes.pid"
THREADS="${THREADS:-12}"
mkdir -p "$ROOT/logs" "$ROOT/state"
running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
work() {
  cd "$ROOT"
  export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
  export FLIGHTLOG_X_FP16=1
  echo "=== role probes start $(date) ==="
  for spec in "meta/eval_emb_views:vrole" "meta/eval_emb_views_channel:chrole" "meta/eval_emb_views_random4:rdrole"; do
    dir="${spec%%:*}"; tag="${spec#*:}"
    export FLIGHTLOG_EMB_DIR="$ROOT/$dir"
    for split in B_fixed C; do
      for head in lr gbt; do
        echo "=== $(date) tag=$tag split=$split head=$head shift120 ==="
        FLIGHTLOG_FEATURE_TAG=$tag $PY horizon.py --split $split --mode shift --h 120 \
          --views emb_role,emb_last --seeds 0,1,2 --head $head --dump-scores || echo "!! FAILED $tag $split $head shift120"
      done
      echo "=== $(date) tag=$tag split=$split lr widen10 ==="
      FLIGHTLOG_FEATURE_TAG=$tag $PY horizon.py --split $split --mode widen --h 10 \
        --views emb_role,emb_last --seeds 0,1,2 --head lr --dump-scores || echo "!! FAILED $tag $split lr widen10"
    done
  done
  echo "=== role probes done $(date) ==="
}
case "${1:-}" in
  start)
    running && { echo "already running (pid $(cat "$PIDF"))"; exit 1; }
    setsid nohup bash -c "$(declare -f work); ROOT='$ROOT'; PY='$PY'; THREADS='$THREADS'; work" \
      </dev/null >>"$LOG" 2>&1 &
    echo $! >"$PIDF"; echo "started pid $(cat "$PIDF"); log: $LOG" ;;
  status) running && echo "[role_probes] RUNNING (pid $(cat "$PIDF"))" || echo "[role_probes] not running"
    echo "done arms: $(ls $ROOT/meta/horizon_pt_*role.json 2>/dev/null | wc -l) / 54 json"
    grep -E "^=== |!!|Killed|det@" "$LOG" 2>/dev/null | tail -n 6 ;;
  tail) tail -f "$LOG" ;;
  stop) running && { pkill -P "$(cat "$PIDF")"; kill "$(cat "$PIDF")"; echo stopped; } || echo "not running" ;;
  *) echo "usage: bash run_role_probes.sh start|status|tail|stop" ;;
esac
