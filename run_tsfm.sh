#!/usr/bin/env bash
# Tier-1.5 TSFM embedding pass -- detached, resumable, one-command status.
#
# Same shape as run_embed.sh: setsid-detached so it survives a disconnect, one
# process per GPU dealing flights round-robin, per-(shard, part) parquet output so a
# killed run resumes by skipping what is already on disk.
#
#   bash run_tsfm.sh pilot        # throughput probe, writes NO cache, ~minutes
#   bash run_tsfm.sh start        # full grid, both GPUs
#   bash run_tsfm.sh status
#   bash run_tsfm.sh tail
#   bash run_tsfm.sh stop
#
# ALWAYS run `pilot` first and read the extrapolation before `start`.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-/home/zhuoyang/venvs/tsfm/bin/python}"   # NOT the flightlog venv: the TSFM
                                                   # stack pins its own torch/transformers
MODEL="${MODEL:-moment-small}"
NGPU="${NGPU:-2}"
WORKERS="${WORKERS:-3}"        # a curated shard is ONE ~6.4 GiB row group; >6 will OOM
BATCH="${BATCH:-64}"
PILOT_FLIGHTS="${PILOT_FLIGHTS:-40}"
FREE_VRAM_GIB="${FREE_VRAM_GIB:-8}"

LOG="$ROOT/logs/tsfm.log"
PIDF="$ROOT/state/tsfm.pids"
mkdir -p "$ROOT/logs" "$ROOT/state"

running() { [[ -s "$PIDF" ]] && xargs -a "$PIDF" -r ps -p >/dev/null 2>&1; }

guard_gpu() {
    # Refuse to start on a card that is already busy -- this box is shared, and the
    # hazard is VRAM headroom, not utilisation.
    local bad=0
    for g in $(seq 0 $((NGPU - 1))); do
        local free
        free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits)
        if (( free / 1024 < FREE_VRAM_GIB )); then
            echo "  GPU $g has only $((free / 1024)) GiB free (need $FREE_VRAM_GIB)"; bad=1
        fi
    done
    (( bad == 0 )) || { echo "refusing to start; free VRAM or lower NGPU"; exit 1; }
}

case "${1:-status}" in
pilot)
    echo "=== pilot: $MODEL, $PILOT_FLIGHTS flights, no cache written ==="
    guard_gpu
    "$PY" tsfm_baseline.py --model "$MODEL" --gpu 0 --batch "$BATCH" \
        --workers "$WORKERS" --pilot "$PILOT_FLIGHTS" 2>&1 | tee -a "$LOG"
    ;;
start)
    if running; then echo "already running (pids $(cat "$PIDF"))"; exit 1; fi
    guard_gpu
    : > "$PIDF"
    echo "=== start $(date '+%F %T')  model=$MODEL ngpu=$NGPU workers=$WORKERS batch=$BATCH ===" >> "$LOG"
    for g in $(seq 0 $((NGPU - 1))); do
        setsid nohup "$PY" tsfm_baseline.py --model "$MODEL" \
            --gpu "$g" --part "$g" --of "$NGPU" \
            --batch "$BATCH" --workers "$WORKERS" \
            >> "$LOG" 2>&1 < /dev/null &
        echo $! >> "$PIDF"
        sleep 20      # stagger: two processes opening the same shard at once doubles
    done              # the peak RSS for no throughput gain
    echo "launched $(wc -l < "$PIDF") process(es); pids $(tr '\n' ' ' < "$PIDF")"
    echo "watch:  bash run_tsfm.sh status"
    ;;
status)
    echo "=== tsfm status $(date '+%F %T') ==="
    if running; then
        echo "  RUNNING"
        ps -o pid,etime,rss,args --no-headers -p "$(tr '\n' ',' < "$PIDF" | sed 's/,$//')" 2>/dev/null \
            | sed 's/^/    /'
    else
        echo "  not running"
    fi
    "$PY" tsfm_baseline.py --status
    echo "  last log lines:"; tail -4 "$LOG" 2>/dev/null | sed 's/^/    /'
    ;;
tail) tail -f "$LOG" ;;
stop)
    if running; then xargs -a "$PIDF" -r kill; echo "stopped"; else echo "not running"; fi
    ;;
*) sed -n '2,16p' "$0" ;;
esac
