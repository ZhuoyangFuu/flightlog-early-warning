#!/usr/bin/env bash
# Item 5, detached: build flight-relative embeddings, then score the horizon arms on them.
#   bash run_flightrel.sh start | status | tail | stop
# /mnt/wsssd is noexec -- invoke with bash. CPU only; OMP capped (shared box).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/flightrel.log"; PIDF="$ROOT/state/flightrel.pid"
THREADS="${THREADS:-8}"
mkdir -p "$ROOT/logs" "$ROOT/state"
running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
work() {
  cd "$ROOT"
  export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
  echo "=== flightrel start $(date) ==="
  $PY flight_relative.py || { echo "BUILD FAILED"; exit 1; }
  export FLIGHTLOG_EMB_DIR="$ROOT/meta/eval_emb_rel"
  # rel-only view, then the concatenated view; lr first (minutes), gbt after (tens of min)
  for view in emb_rel emb_mean,emb_rel; do
    tag=rel; [[ "$view" == *,* ]] && tag=relcat
    for split in B_fixed C; do
      for head in lr gbt; do
        echo "=== $(date) tag=$tag split=$split head=$head shift120 ==="
        FLIGHTLOG_FEATURE_TAG=$tag $PY horizon.py --split $split --mode shift --h 120 \
          --views $view --seeds 0,1,2 --head $head --dump-scores
      done
      echo "=== $(date) tag=$tag split=$split lr widen10 (original-band reproduction) ==="
      FLIGHTLOG_FEATURE_TAG=$tag $PY horizon.py --split $split --mode widen --h 10 \
        --views $view --seeds 0,1,2 --head lr --dump-scores
    done
  done
  echo "=== flightrel done $(date) ==="
}
case "${1:-}" in
  start)
    running && { echo "already running (pid $(cat "$PIDF"))"; exit 1; }
    setsid nohup bash -c "$(declare -f work); ROOT='$ROOT'; PY='$PY'; THREADS='$THREADS'; work" \
      </dev/null >>"$LOG" 2>&1 &
    echo $! >"$PIDF"; echo "started pid $(cat "$PIDF"); log: $LOG" ;;
  status) running && echo "RUNNING (pid $(cat "$PIDF"))" || echo "not running"
          grep -E '^===|det@|cached|FAILED|Error' "$LOG" 2>/dev/null | tail -n 12 ;;
  tail) tail -n "${N:-40}" "$LOG" ;;
  stop) running && pkill -P "$(cat "$PIDF")"; kill "$(cat "$PIDF")" 2>/dev/null; echo stopped ;;
  *) echo "usage: bash $0 start|status|tail|stop"; exit 1 ;;
esac
