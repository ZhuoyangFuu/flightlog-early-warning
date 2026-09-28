#!/usr/bin/env bash
# Detached launcher for the SHIFTED-BAND TIER-1 CONTROL (horizon_tier1.py).
#
#   bash run_h1.sh start [args...]   # default: the 14-arm control grid
#   bash run_h1.sh status
#   bash run_h1.sh tail
#   bash run_h1.sh stop
#   bash run_h1.sh list              # what the grid is, without running it
#
# /mnt/wsssd is noexec -- always invoke as `bash run_h1.sh`, never ./
#
# The question: RESULTS.md 4.5/4.6's dominance arm changed BOTH the features
# (hand-built stats -> pretrained embeddings) AND the training band (1-10 s ->
# 111-150 s). This grid varies only the band, holding the features at Tier-1's,
# so a win here would mean the result is the band and not the representation.
#
# `start` is also resume: every arm caches to meta/h1_pt_*.json +
# meta/h1_curve_*.csv (+ meta/scores/h1_scores_*.parquet with --dump-scores) and
# a cached arm is skipped, so a killed run re-reads the cache and continues.
# setsid-detached; SSH disconnect and logout do not touch it.
#
# CPU-only -- the GPUs stay free. Feature matrix is 6,087,288 x 435 float32 =
# 9.9 GiB resident, preallocated (no concat spike), so the profile is far lighter
# than run_horizon.sh's ~50 GiB. This box is shared: it refuses to start below
# MIN_RAM_GIB and caps OMP so it does not crowd out other users.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/h1.log"
PIDF="$ROOT/state/h1.pid"
THREADS="${THREADS:-8}"
MIN_RAM_GIB="${MIN_RAM_GIB:-40}"

mkdir -p "$ROOT/logs" "$ROOT/state"

running() {
  [[ -f "$PIDF" ]] || return 1
  kill -0 "$(cat "$PIDF")" 2>/dev/null
}

case "${1:-}" in
start)
  if running; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi
  if [[ ! -d "$ROOT/meta/tier1_feats" ]]; then
    echo "ERROR: meta/tier1_feats missing -- run tier1_baseline.py first."
    exit 1
  fi
  for other in horizon probe finetune; do
    if [[ -f "$ROOT/state/$other.pid" ]] && kill -0 "$(cat "$ROOT/state/$other.pid")" 2>/dev/null; then
      echo "REFUSING TO START: run_$other.sh is running; both would hold a large"
      echo "  feature matrix in RAM. Wait for it, or override with FORCE=1."
      [[ "${FORCE:-0}" != "1" ]] && exit 1
    fi
  done
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if (( avail < MIN_RAM_GIB )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "REFUSING TO START: only ${avail} GiB RAM available, need ${MIN_RAM_GIB}."
    echo "  Override with FORCE=1."
    exit 1
  fi
  shift
  args=("$@")
  if [[ ${#args[@]} -eq 0 ]]; then args=(--dump-scores); fi
  echo "=== start $(date '+%F %T')  threads=$THREADS  args=${args[*]} ===" >> "$LOG"
  setsid nohup env OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    MKL_NUM_THREADS="$THREADS" "$PY" "$ROOT/horizon_tier1.py" "${args[@]}" \
    >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 2
  echo "started pid $(cat "$PIDF")"
  echo "  log:    bash run_h1.sh tail"
  echo "  status: bash run_h1.sh status"
  ;;

status)
  if running; then
    echo "RUNNING: $(cat "$PIDF")"
    ps -eo pid,sid,tty,etime,rss,args --no-headers -p "$(cat "$PIDF")" 2>/dev/null
  else
    echo "not running"
  fi
  echo
  OMP_NUM_THREADS=1 "$PY" "$ROOT/horizon_tier1.py" --status
  echo
  echo "--- control gate (widen 10s must reproduce Tier-1 lr 0.202 / gbt 0.364) ---"
  grep -h "CONTROL vs frozen" "$LOG" 2>/dev/null | tail -4 || echo "  not reached yet"
  echo
  echo "free RAM: $(free -g | awk '/^Mem:/{print $7" GiB available"}')"
  ;;

list)  OMP_NUM_THREADS=1 "$PY" "$ROOT/horizon_tier1.py" --list ;;

tail)  tail -f "$LOG" ;;

stop)
  if ! running; then echo "not running"; exit 0; fi
  kill "$(cat "$PIDF")" 2>/dev/null && echo "stopped $(cat "$PIDF")"
  rm -f "$PIDF" ;;

*)
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1 ;;
esac
