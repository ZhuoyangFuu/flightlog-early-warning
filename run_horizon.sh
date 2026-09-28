#!/usr/bin/env bash
# Detached launcher for item 2, the training-horizon sweep (PROJECT_PLAN §7.5, §11.0.1).
#
#   bash run_horizon.sh start [args...]   # default: the full grid (14 jobs)
#   bash run_horizon.sh status
#   bash run_horizon.sh tail
#   bash run_horizon.sh stop
#   bash run_horizon.sh list              # what the grid is, without running it
#
# /mnt/wsssd is noexec -- always invoke as `bash run_horizon.sh`, never ./
#
# `start` is also resume: horizon.py caches every (mode, h, split, seed) to
# meta/horizon_pt_*.json + meta/horizon_curve_*.csv and skips any arm that
# already has both, so a killed run re-reads the cache and continues.
# setsid-detached; SSH disconnect and logout do not touch it.
#
# CPU-only -- it reads the cached embeddings, so the GPUs stay free. Same memory
# profile as run_probe.sh: the full grid is 6,087,288 x 1024 float32 = 23 GiB
# resident and the load concatenates, so peak is ~50 GiB. This box is shared: it
# refuses to start below MIN_RAM_GIB, and OMP is capped so it does not crowd out
# other users. Do not run it alongside run_probe.sh -- they would each hold their
# own 23 GiB copy of the same matrix.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/horizon.log"
PIDF="$ROOT/state/horizon.pid"
THREADS="${THREADS:-8}"
MIN_RAM_GIB="${MIN_RAM_GIB:-80}"

mkdir -p "$ROOT/logs" "$ROOT/state"

running() {
  [[ -f "$PIDF" ]] || return 1
  kill -0 "$(cat "$PIDF")" 2>/dev/null
}

case "${1:-}" in

start)
  if running; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi
  if [[ ! -d "$ROOT/meta/eval_emb" ]]; then
    echo "ERROR: meta/eval_emb missing -- the embedding pass must finish first."
    exit 1
  fi
  if [[ -f "$ROOT/state/probe.pid" ]] && kill -0 "$(cat "$ROOT/state/probe.pid")" 2>/dev/null; then
    echo "REFUSING TO START: run_probe.sh is running and would double the 23 GiB"
    echo "  embedding matrix in RAM. Wait for it, or override with FORCE=1."
    [[ "${FORCE:-0}" != "1" ]] && exit 1
  fi
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if (( avail < MIN_RAM_GIB )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "REFUSING TO START: only ${avail} GiB RAM available, need ${MIN_RAM_GIB}."
    echo "  Override with FORCE=1."
    exit 1
  fi
  shift
  args=("$@")
  echo "=== start $(date '+%F %T')  threads=$THREADS  args=${args[*]:-<full grid>} ===" >> "$LOG"
  setsid nohup env OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    MKL_NUM_THREADS="$THREADS" "$PY" "$ROOT/horizon.py" "${args[@]}" \
    >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 2
  echo "started pid $(cat "$PIDF")"
  echo "  log:    bash run_horizon.sh tail"
  echo "  status: bash run_horizon.sh status"
  ;;

status)
  if running; then
    echo "RUNNING: $(cat "$PIDF")"
    ps -eo pid,sid,tty,etime,rss,args --no-headers -p "$(cat "$PIDF")" 2>/dev/null
  else
    echo "not running"
  fi
  echo
  OMP_NUM_THREADS=1 "$PY" "$ROOT/horizon.py" --status
  echo
  echo "free RAM: $(free -g | awk '/^Mem:/{print $7" GiB available"}')"
  ;;

list)  OMP_NUM_THREADS=1 "$PY" "$ROOT/horizon.py" --list ;;

tail)  tail -f "$LOG" ;;

stop)
  if ! running; then echo "not running"; exit 0; fi
  kill "$(cat "$PIDF")" 2>/dev/null && echo "stopped $(cat "$PIDF")"
  rm -f "$PIDF" ;;

*)
  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1 ;;
esac
