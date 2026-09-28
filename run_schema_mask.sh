#!/usr/bin/env bash
# Detached runner for build_schema_mask.py -- same discipline as the other launchers here.
# setsid so it outlives the shell/SSH session; stdin /dev/null; stdout to an append-only log.
# The work is idempotent (it rewrites its outputs from scratch), so `start` doubles as retry.
#
# /mnt/wsssd is noexec -- invoke as `bash run_schema_mask.sh`, never ./
#
#   bash run_schema_mask.sh start | status | tail | stop

set -u
cd "$(dirname "$0")"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG=logs/schema_mask.log
PIDFILE=logs/schema_mask.pid
mkdir -p logs

case "${1:-status}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    echo "=== schema_mask start $(date) ===" >>"$LOG"
    # OMP capped: the runoff training jobs own the GPUs and 32 dataloader workers, and
    # this is pure pandas -- it has no business taking 64 cores.
    OMP_NUM_THREADS=4 setsid nohup "$PY" build_schema_mask.py \
      >>"$LOG" 2>&1 </dev/null &
    echo $! >"$PIDFILE"
    echo "launched detached (pid $(cat $PIDFILE)); survives disconnect."
    echo "  log:    tail -f $(pwd)/$LOG"
    echo "  status: bash run_schema_mask.sh status"
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "RUNNING (pid $(cat "$PIDFILE"))"
    else
      echo "not running"
    fi
    for f in meta/schema_mask.parquet meta/schema_mask_report.json; do
      [ -f "$f" ] && echo "  $f  $(stat -c '%y' "$f" | cut -d. -f1)  $(stat -c %s "$f") bytes" \
                  || echo "  $f  MISSING"
    done
    echo "--- last log lines ---"
    tail -n 12 "$LOG" 2>/dev/null || echo "  (no log yet)"
    ;;
  tail) tail -f "$LOG" ;;
  stop)
    [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && echo "stopped" || echo "not running"
    ;;
  *) echo "usage: bash run_schema_mask.sh start|status|tail|stop"; exit 2 ;;
esac
