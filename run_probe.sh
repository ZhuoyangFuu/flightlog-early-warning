#!/usr/bin/env bash
# Detached launcher for Tier-2 stage 2, the frozen-encoder probes (PROJECT_PLAN §7.4).
#
#   bash run_probe.sh start [args...]   # default: --all (12 jobs, lr+mlp x 6 splits)
#   bash run_probe.sh status
#   bash run_probe.sh tail
#   bash run_probe.sh stop
#
# /mnt/wsssd is noexec -- always invoke as `bash run_probe.sh`, never ./
#
# `start` is also resume: probe.py caches every (head, split, seed) to
# meta/tier2_pt_*.json + meta/tier2_curve_*.csv and skips any pair that already
# has both, so a killed run re-reads the cache and continues at the next job.
# setsid-detached; SSH disconnect and logout do not touch it.
#
# CPU-only -- it reads the cached embeddings, the GPUs stay free for fine-tuning.
# Memory: the full grid is 6,087,288 x 1024 float32 = 23 GiB resident, and the
# load concatenates, so peak is ~50 GiB. This box is shared: it refuses to start
# below MIN_RAM_GIB, and OMP is capped so it does not crowd out other users.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/probe.log"
PIDF="$ROOT/state/probe.pid"
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
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if (( avail < MIN_RAM_GIB )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "REFUSING TO START: only ${avail} GiB RAM available, need ${MIN_RAM_GIB}."
    echo "  Override with FORCE=1."
    exit 1
  fi
  shift
  args=("$@"); [[ ${#args[@]} -eq 0 ]] && args=(--all)
  echo "=== start $(date '+%F %T')  threads=$THREADS  args=${args[*]} ===" >> "$LOG"
  setsid nohup env OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    MKL_NUM_THREADS="$THREADS" "$PY" "$ROOT/probe.py" "${args[@]}" \
    >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 2
  echo "started pid $(cat "$PIDF")"
  echo "  log:    bash run_probe.sh tail"
  echo "  status: bash run_probe.sh status"
  ;;

status)
  if running; then
    echo "RUNNING: $(cat "$PIDF")"
    ps -eo pid,sid,tty,etime,rss,args --no-headers -p "$(cat "$PIDF")" 2>/dev/null
  else
    echo "not running"
  fi
  echo
  echo "--- finished probes (meta/tier2_pt_*.json) ---"
  shopt -s nullglob
  # The base grid is 12 jobs (2 heads x 6 split arms); `--max-iter` refits add
  # their own `_it<N>` entries on top, so count those separately rather than
  # letting them inflate the grid's completion count.
  files=("$ROOT"/meta/tier2_pt_*.json)
  base=0; refit=0
  for f in "${files[@]}"; do
    [[ -e "$f" ]] || continue
    if [[ "$f" == *_it[0-9]*.json ]]; then refit=$((refit+1)); else base=$((base+1)); fi
  done
  if (( ${#files[@]} == 0 )); then
    echo "  (none yet)  0 of 12 base jobs"
  else
    for f in "${files[@]}"; do
      "$PY" -c "
import json,os
d=json.load(open('$f'))
k=[x for x in d if x.startswith('det_at_')]
det=d.get(k[0],float('nan')) if k else float('nan')
print('  %-40s det %.3f  auprc %.3f' % (os.path.basename('$f'), det, d.get('window_auprc',float('nan'))))"
    done
    echo "  ${base} of 12 base jobs cached; ${refit} max-iter refit(s)"
  fi
  echo
  echo "free RAM: $(free -g | awk '/^Mem:/{print $7" GiB available"}')"
  ;;

tail)  tail -f "$LOG" ;;

stop)
  if ! running; then echo "not running"; exit 0; fi
  kill "$(cat "$PIDF")" 2>/dev/null && echo "stopped $(cat "$PIDF")"
  rm -f "$PIDF" ;;

*)
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1 ;;
esac
