#!/usr/bin/env bash
# Detached launcher for masked-reconstruction pretraining (PROJECT_PLAN §7).
#
#   bash run_pretrain.sh start [extra pretrain.py args...]
#   bash run_pretrain.sh status
#   bash run_pretrain.sh tail
#   bash run_pretrain.sh stop
#   bash run_pretrain.sh gpucheck
#
# /mnt/wsssd is noexec -- always invoke as `bash run_pretrain.sh`, never ./
#
# `start` is also resume: pretrain.py loads checkpoints/pretrain_last.pt if it
# exists. The job is setsid-detached, so SSH disconnect and logout do not touch
# it; only a reboot or `stop` ends it. It refuses to start if the GPUs are
# already busy -- override with FORCE=1 (see gpucheck).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/pretrain.log"
PIDF="$ROOT/state/pretrain.pid"
NGPU="${NGPU:-2}"
FREE_VRAM_GIB="${FREE_VRAM_GIB:-20}"   # a GPU counts as free above this
MAX_UTIL="${MAX_UTIL:-15}"             # ...and below this % utilisation

mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/checkpoints"

running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

gpu_free_count() {
  local n=0
  while IFS=, read -r idx util used total; do
    util="${util// /}"; used="${used// /}"; total="${total// /}"
    local freeg=$(( (total - used) / 1024 ))
    if (( util <= MAX_UTIL && freeg >= FREE_VRAM_GIB )); then n=$((n+1)); fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
             --format=csv,noheader,nounits)
  echo "$n"
}

case "${1:-}" in

start)
  shift || true
  if running; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi

  if [[ ! -f "$ROOT/meta/channel_stats.json" ]]; then
    echo "ERROR: meta/channel_stats.json missing."
    echo "  run:  $PY pretrain_data.py --stats"
    exit 1
  fi

  free=$(gpu_free_count)
  if (( free < NGPU )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "REFUSING TO START: only $free of $NGPU GPUs are free."
    echo "  (a GPU counts as free at <=${MAX_UTIL}% util and >=${FREE_VRAM_GIB} GiB spare)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader
    echo "  Start anyway with:  FORCE=1 bash run_pretrain.sh start"
    echo "  Or fewer GPUs with: NGPU=1 bash run_pretrain.sh start"
    exit 1
  fi

  echo "=== start $(date '+%F %T')  ngpu=$NGPU  args: $* ===" >> "$LOG"
  if (( NGPU > 1 )); then
    setsid nohup "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" \
      --master_port="${MASTER_PORT:-29517}" "$ROOT/pretrain.py" "$@" \
      >> "$LOG" 2>&1 < /dev/null &
  else
    setsid nohup "$PY" "$ROOT/pretrain.py" "$@" >> "$LOG" 2>&1 < /dev/null &
  fi
  echo $! > "$PIDF"
  sleep 2
  if running; then
    echo "started pid $(cat "$PIDF")  (detached; log: $LOG)"
  else
    echo "FAILED to start -- last log lines:"; tail -20 "$LOG"; exit 1
  fi
  ;;

status)
  echo "=== pretrain $(date '+%F %T') ==="
  if running; then
    pid=$(cat "$PIDF")
    sid=$(ps -o sid= -p "$pid" | tr -d ' ')
    echo "  RUNNING pid $pid  sid $sid  $([[ "$pid" == "$sid" ]] && echo detached || echo '*** NOT detached ***')"
    ps -o etime=,pcpu=,rss= -p "$pid" | awk '{print "  uptime "$1"  cpu "$2"%  rss "int($3/1048576)"GiB"}'
  else
    echo "  not running"
  fi
  "$PY" "$ROOT/pretrain.py" --status
  echo "  --- checkpoints ---"
  ls -lh "$ROOT/checkpoints"/*.pt 2>/dev/null | awk '{print "  "$9" "$5" "$6" "$7" "$8}' || echo "  none yet"
  echo "  --- gpus ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader | sed 's/^/  /'
  ;;

tail)   tail -f "$LOG" ;;

stop)
  if ! running; then echo "not running"; exit 0; fi
  pid=$(cat "$PIDF")
  echo "SIGTERM -> $pid (it checkpoints, then exits)"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  for _ in $(seq 1 60); do running || break; sleep 2; done
  running && { echo "still up after 120s, SIGKILL"; kill -9 -- "-$pid" 2>/dev/null; }
  rm -f "$PIDF"; echo "stopped"
  ;;

gpucheck)
  echo "free GPUs by the start rule (<=${MAX_UTIL}% util, >=${FREE_VRAM_GIB} GiB spare): $(gpu_free_count) of $NGPU needed"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader
  ;;

*) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
