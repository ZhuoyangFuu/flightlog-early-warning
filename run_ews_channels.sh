#!/usr/bin/env bash
# Detached launcher for the TIER-0 AGGREGATION-WIDTH DIAGNOSTIC (ews_channels.py).
#
#   bash run_ews_channels.sh start     # 14 arms, resumable
#   bash run_ews_channels.sh status
#   bash run_ews_channels.sh tail
#   bash run_ews_channels.sh list      # the grid, runs nothing
#   bash run_ews_channels.sh stop
#
# /mnt/wsssd is noexec -- always invoke as `bash run_ews_channels.sh`, never ./
#
# The question: van Beers et al. get a POSITIVE classical-EWS result from rotor
# speed alone; we get 18.6% aggregating top-k over 87 channels. Is our negative
# about EWS, or about our aggregation? This re-runs the frozen indicator over
# physically-chosen channel subsets, changing exactly one thing.
#
# ⚠ POST-HOC DIAGNOSTIC. It never touches meta/ews_curve_FROZEN.csv or
# meta/ews_frozen_config.json; the pre-registered Tier-0 headline is unchanged.
#
# `start` is also resume: each arm caches meta/ews_traces_chan-*.parquet +
# curves + a per-split json, and a cached arm is skipped, so a killed run loses
# at most one arm (~20 min at 87 channels, less on the small subsets).
# setsid-detached; SSH disconnect and logout do not touch it.
#
# CPU-only -- both GPUs stay free. Pool of EWS_WORKERS processes over one shard
# at a time; peak RSS is per-worker and small (one flight's (T, C) matrix), far
# lighter than the horizon jobs. This box is shared, so workers default to 12
# and OMP is pinned to 1 (the parallelism is the Pool, not BLAS).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/ews_channels.log"
PIDF="$ROOT/state/ews_channels.pid"
WORKERS="${EWS_WORKERS:-12}"
MIN_RAM_GIB="${MIN_RAM_GIB:-20}"

mkdir -p "$ROOT/logs" "$ROOT/state"

running() {
  [[ -f "$PIDF" ]] || return 1
  kill -0 "$(cat "$PIDF")" 2>/dev/null
}

case "${1:-}" in
start)
  if running; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi
  if [[ ! -f "$ROOT/meta/ews_traces_w10k3ext.parquet" ]]; then
    echo "ERROR: meta/ews_traces_w10k3ext.parquet missing -- that is the control"
    echo "  arm this diagnostic is compared against. Run ews_baseline.py with the"
    echo "  extended grid first (RESULTS.md 2.1)."
    exit 1
  fi
  for other in horizon probe finetune h1; do
    if [[ -f "$ROOT/state/$other.pid" ]] && kill -0 "$(cat "$ROOT/state/$other.pid")" 2>/dev/null; then
      echo "WARNING: run_$other.sh is running; this job is CPU-heavy ($WORKERS procs)."
      if [[ "${FORCE:-0}" != "1" ]]; then echo "  Override with FORCE=1."; exit 1; fi
    fi
  done
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if (( avail < MIN_RAM_GIB )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "REFUSING TO START: only ${avail} GiB RAM available, need ${MIN_RAM_GIB}."
    exit 1
  fi
  shift || true
  args=("$@")
  # DUMP=1 bash run_ews_channels.sh start  -> also persist the per-bin z-series
  # (~0.9 GB/arm, ~12 GB total) so sequential.py can cross the channel grid with
  # the 11-rule grid. Needs ~20 GB free; checked below.
  if [[ "${DUMP:-0}" == "1" ]]; then
    args+=(--dump-scores)
    free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
    if (( free_gb < 20 )); then
      echo "REFUSING TO START: --dump-scores needs ~12 GB, only ${free_gb} GB free."
      exit 1
    fi
  fi
  echo "=== start $(date '+%F %T')  workers=$WORKERS  args=${args[*]:-none} ===" >> "$LOG"
  setsid nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    EWS_WORKERS="$WORKERS" "$PY" "$ROOT/ews_channels.py" "${args[@]}" \
    >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 2
  echo "started pid $(cat "$PIDF")"
  echo "  log:    bash run_ews_channels.sh tail"
  echo "  status: bash run_ews_channels.sh status"
  ;;

status)
  if running; then
    echo "RUNNING: $(cat "$PIDF")"
    ps -o pid,sid,tty,etime,rss,args --no-headers -p "$(cat "$PIDF")" 2>/dev/null
    echo "  workers: $(pgrep -c -P "$(cat "$PIDF")" 2>/dev/null || echo 0) child procs"
  else
    echo "not running"
  fi
  echo
  OMP_NUM_THREADS=1 "$PY" "$ROOT/ews_channels.py" --status
  echo
  echo "free RAM: $(free -g | awk '/^Mem:/{print $7" GiB available"}')"
  ;;

list) OMP_NUM_THREADS=1 "$PY" "$ROOT/ews_channels.py" --list ;;

tail) tail -f "$LOG" ;;

stop)
  if ! running; then echo "not running"; exit 0; fi
  pkill -TERM -P "$(cat "$PIDF")" 2>/dev/null
  kill "$(cat "$PIDF")" 2>/dev/null && echo "stopped $(cat "$PIDF")"
  rm -f "$PIDF" ;;

*)
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1 ;;
esac
