#!/usr/bin/env bash
# Detached launcher for the Tier-2 embedding pass (PROJECT_PLAN §7.4).
#
#   bash run_embed.sh start          # both GPUs, full 6,087,288-window grid (~7 h)
#   bash run_embed.sh pilot [N]      # N flights per (shard,worker), one GPU (~25 min)
#   bash run_embed.sh status
#   bash run_embed.sh tail
#   bash run_embed.sh stop
#   bash run_embed.sh gpucheck
#
# /mnt/wsssd is noexec -- always invoke as `bash run_embed.sh`, never ./
#
# `start` is also resume: embed_eval.py skips any (shard, part) whose parquet
# already exists, and each part is written atomically via .tmp + rename, so a
# killed job never leaves a half-written part behind. Both processes are
# setsid-detached; SSH disconnect and logout do not touch them.
#
# Memory: a curated shard is ONE ~6.4 GiB Parquet row group, so each dataloader
# worker costs a fixed ~6.4 GiB resident. WORKERS=3 per process (2 processes) is
# ~40 GiB. Do not raise it without checking `free -g` -- this box is shared.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/embed_eval.log"
PIDF="$ROOT/state/embed_eval.pids"
NGPU="${NGPU:-2}"
WORKERS="${WORKERS:-3}"
BATCH="${BATCH:-96}"
FREE_VRAM_GIB="${FREE_VRAM_GIB:-8}"    # inference peak is ~3.5 GiB at batch 96
MAX_UTIL="${MAX_UTIL:-15}"

mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/meta/eval_emb"

running() {
  [[ -f "$PIDF" ]] || return 1
  while read -r p; do kill -0 "$p" 2>/dev/null && return 0; done < "$PIDF"
  return 1
}

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

require_ckpt() {
  if [[ ! -f "$ROOT/checkpoints/pretrain_last.pt" ]]; then
    echo "ERROR: checkpoints/pretrain_last.pt missing -- pretraining must finish first."
    exit 1
  fi
}

case "${1:-}" in

start)
  if running; then echo "already running (pids: $(tr '\n' ' ' < "$PIDF"))"; exit 0; fi
  require_ckpt
  free=$(gpu_free_count)
  if (( free < NGPU )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "REFUSING TO START: only $free of $NGPU GPUs are free."
    echo "  (free = <=${MAX_UTIL}% util and >=${FREE_VRAM_GIB} GiB spare)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader
    echo "  Start anyway with:  FORCE=1 bash run_embed.sh start"
    echo "  Or one GPU with:    NGPU=1 bash run_embed.sh start"
    exit 1
  fi
  : > "$PIDF"
  echo "=== start $(date '+%F %T')  ngpu=$NGPU workers=$WORKERS batch=$BATCH ===" >> "$LOG"
  for ((i=0; i<NGPU; i++)); do
    setsid nohup env OMP_NUM_THREADS=4 "$PY" "$ROOT/embed_eval.py" \
      --part "$i" --of "$NGPU" --gpu "$i" \
      --workers "$WORKERS" --batch "$BATCH" \
      >> "$LOG" 2>&1 < /dev/null &
    echo $! >> "$PIDF"
  done
  sleep 2
  echo "started pids: $(tr '\n' ' ' < "$PIDF")"
  echo "  log:    bash run_embed.sh tail"
  echo "  status: bash run_embed.sh status"
  ;;

pilot)
  require_ckpt
  n="${2:-400}"
  echo "=== pilot $(date '+%F %T')  flights/shard/worker=$n ===" >> "$LOG"
  setsid nohup env OMP_NUM_THREADS=4 "$PY" "$ROOT/embed_eval.py" \
    --part 0 --of 1 --gpu "${GPU:-0}" --workers "$WORKERS" --batch "$BATCH" \
    --pilot "$n" >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 2
  echo "pilot started (pid $(cat "$PIDF")); tail with: bash run_embed.sh tail"
  ;;

status)
  if running; then
    echo "RUNNING: $(tr '\n' ' ' < "$PIDF")"
    ps -eo pid,sid,tty,etime,rss,args --no-headers -p "$(tr '\n' ',' < "$PIDF" | sed 's/,$//')" 2>/dev/null
  else
    echo "not running"
  fi
  echo
  "$PY" "$ROOT/embed_eval.py" --status
  echo
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  echo "free RAM: $(free -g | awk '/^Mem:/{print $7" GiB available"}')"
  ;;

tail)    tail -f "$LOG" ;;

stop)
  if ! running; then echo "not running"; exit 0; fi
  while read -r p; do kill "$p" 2>/dev/null && echo "stopped $p"; done < "$PIDF"
  rm -f "$PIDF"
  ;;

gpucheck)
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
  echo "free by policy (<=${MAX_UTIL}% util, >=${FREE_VRAM_GIB} GiB spare): $(gpu_free_count) of $NGPU needed"
  ;;

*)
  sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1 ;;
esac
