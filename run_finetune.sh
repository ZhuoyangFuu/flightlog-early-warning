#!/usr/bin/env bash
# Detached launcher for Tier-2 fine-tuning (PROJECT_PLAN §7.4.1).
#
#   bash run_finetune.sh train  <split> <seed> <init> [gpu]   # e.g. train A 0 pretrained 0
#   bash run_finetune.sh score  <split> <seed> <init> [gpu]
#   bash run_finetune.sh ablate <split> <seed>                # pretrained + scratch, one per GPU
#   bash run_finetune.sh ablate-score <split> <seed>          # the scoring half, both arms at once
#   bash run_finetune.sh status
#   bash run_finetune.sh tail   [tag]
#   bash run_finetune.sh stop
#   bash run_finetune.sh gpucheck
#
# /mnt/wsssd is noexec -- always invoke as `bash run_finetune.sh`, never ./
#
# `train` is also resume (per-epoch checkpoints); `score` resumes per shard.
# Everything is setsid-detached: SSH disconnect and logout do not touch it.
#
# `ablate` is the one that matters: it runs --init pretrained on GPU 0 and
# --init scratch on GPU 1 with identical everything else. A fine-tuned number
# quoted without its scratch arm is not evidence that pretraining did anything.
#
# COST: training is cheap next to scoring. A split-A score pass covers ~20-25%
# of the 6,087,288-window grid (~1.5 h); a B_fixed pass covers ~49% (~3.5 h).
# Memory: curated shards are ONE ~6.4 GiB row group each, so every dataloader
# worker costs ~6.4 GiB. WORKERS=3 per process. Do not raise it blindly.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
PIDF="$ROOT/state/finetune.pids"
WORKERS="${WORKERS:-3}"
FREE_VRAM_GIB="${FREE_VRAM_GIB:-10}"
MAX_UTIL="${MAX_UTIL:-15}"

mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/checkpoints"

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

launch() { # mode split seed init gpu
  local mode="$1" split="$2" seed="$3" init="$4" gpu="$5"
  local tag="${split}_${init}_seed${seed}"
  local log="$ROOT/logs/ft_${tag}.log"
  local extra=""
  [[ "$mode" == "score" ]] && extra="--score"
  echo "=== $mode $tag gpu=$gpu $(date '+%F %T') ===" >> "$log"
  setsid nohup env OMP_NUM_THREADS=4 "$PY" "$ROOT/finetune.py" \
    --split "$split" --seed "$seed" --init "$init" --gpu "$gpu" \
    --workers "$WORKERS" $extra >> "$log" 2>&1 < /dev/null &
  echo $! >> "$PIDF"
  echo "  $mode $tag -> gpu $gpu, log logs/ft_${tag}.log"
}

require_free() {
  local need="$1"
  local free; free=$(gpu_free_count)
  if (( free < need )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "REFUSING TO START: only $free of $need GPUs free (<=${MAX_UTIL}% util, >=${FREE_VRAM_GIB} GiB spare)."
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo "  Override with FORCE=1."
    exit 1
  fi
}

case "${1:-}" in

train|score)
  mode="$1"; split="${2:-A}"; seed="${3:-0}"; init="${4:-pretrained}"; gpu="${5:-0}"
  if running; then echo "already running: $(tr '\n' ' ' < "$PIDF")"; exit 1; fi
  require_free 1
  : > "$PIDF"
  launch "$mode" "$split" "$seed" "$init" "$gpu"
  ;;

ablate)
  split="${2:-A}"; seed="${3:-0}"
  if running; then echo "already running: $(tr '\n' ' ' < "$PIDF")"; exit 1; fi
  require_free 2
  : > "$PIDF"
  launch train "$split" "$seed" pretrained 0
  launch train "$split" "$seed" scratch    1
  echo "ablation running; when BOTH finish, score each arm:"
  echo "  bash run_finetune.sh score $split $seed pretrained 0"
  echo "  bash run_finetune.sh score $split $seed scratch 1"
  ;;

ablate-score)
  # The scoring half of `ablate`: both arms at once, one per GPU. Scoring is the
  # expensive half (B_fixed ~4.4 h per arm), and the two arms are independent, so
  # running them serially would double the wall clock for no reason.
  split="${2:-A}"; seed="${3:-0}"
  if running; then echo "already running: $(tr '\n' ' ' < "$PIDF")"; exit 1; fi
  for init in pretrained scratch; do
    tag="${split}_${init}"; [[ "$split" == "A" ]] && tag="${split}_${init}_seed${seed}"
    if [[ ! -f "$ROOT/checkpoints/ft_${tag}_best.pt" ]]; then
      echo "REFUSING: checkpoints/ft_${tag}_best.pt missing -- train that arm first."
      exit 1
    fi
  done
  require_free 2
  : > "$PIDF"
  launch score "$split" "$seed" pretrained 0
  launch score "$split" "$seed" scratch    1
  ;;

status)
  if running; then
    echo "RUNNING: $(tr '\n' ' ' < "$PIDF")"
    ps -o pid,sid,tty,etime,rss,args --no-headers \
       -p "$(tr '\n' ',' < "$PIDF" | sed 's/,$//')" 2>/dev/null
  else
    echo "not running"
  fi
  echo
  echo "--- finished runs (meta/ft_pt_*.json) ---"
  for f in "$ROOT"/meta/ft_pt_*.json; do
    [[ -e "$f" ]] || { echo "  (none yet)"; break; }
    "$PY" -c "
import json,sys
d=json.load(open('$f'))
k=[x for x in d if x.startswith('det_at_')]
print(f\"  {'$(basename "$f")':44s} det {d.get(k[0],float('nan')) if k else float('nan'):.3f}  auprc {d.get('window_auprc',float('nan')):.3f}  valAP {d.get('val_ap',float('nan')):.3f}\")"
  done
  echo
  echo "--- checkpoints ---"
  ls -1 "$ROOT"/checkpoints/ft_*.pt 2>/dev/null | sed 's|.*/|  |' || echo "  (none yet)"
  echo
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  ;;

tail)
  if [[ -n "${2:-}" ]]; then tail -f "$ROOT/logs/ft_$2.log"
  else tail -f $(ls -t "$ROOT"/logs/ft_*.log 2>/dev/null | head -2); fi ;;

stop)
  if ! running; then echo "not running"; exit 0; fi
  while read -r p; do kill "$p" 2>/dev/null && echo "stopped $p"; done < "$PIDF"
  rm -f "$PIDF" ;;

gpucheck)
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
  echo "free by policy: $(gpu_free_count) of 2" ;;

*)
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1 ;;
esac
