#!/usr/bin/env bash
# Autonomous GPU chain for FlightLog items 6 + 8, armed to start when the runoff v3 queue
# has drained and both cards are free. Detached; survives logout.
#   bash run_flightlog_chain.sh start|status|stop        log: logs/chain.log
# Steps (each verified before the next):
#   1 item-6 pilot + reproduction gate     2 item-6 full pass (original encoder)   -> CPU probes (bg)
#   3 item-8 A: continued pretraining, channel mask, 4 epochs -> checkpoints/pt_channel
#   4 item-8 B: continued pretraining, random mask,  4 epochs -> checkpoints/pt_random4 (control)
#   5 views pass on A   -> CPU probes (bg)        6 views pass on B -> CPU probes (bg)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
RUNOFF=/home/zhuoyang/runoff_st_pretrain
LOG="$ROOT/logs/chain.log"; PIDF="$ROOT/state/chain.pid"; mkdir -p "$ROOT/logs" "$ROOT/state"
log(){ echo "=== $(date '+%F %T') $*" >>"$LOG"; }
runoff_busy(){
  [[ -s "$RUNOFF/logs/queue_v3.txt" ]] && return 0
  for pf in "$RUNOFF"/logs/phase3_v3_*.pid; do [[ -e "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null && return 0; done
  return 1
}
gpus_free(){
  local n=0
  while IFS=, read -r idx used; do used="${used// /}"; (( used < 1500 )) && n=$((n+1)); done \
    < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  (( n >= 2 ))
}
any_alive(){ # $1 = pidfile; true if any listed pid is alive
  [[ -f "$1" ]] || return 1
  while read -r p; do [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && return 0; done <"$1"
  return 1
}
wait_pids(){ while any_alive "$1"; do sleep 120; done; }
wait_pid(){ while [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; do sleep 120; done; }
chain(){
  cd "$ROOT"
  log "chain armed; waiting for the runoff queue to drain and both GPUs to free"
  while runoff_busy || ! gpus_free; do sleep 300; done
  sleep 120; log "GPUs free -> step 1: item-6 pilot + reproduction gate"
  bash run_embed_views.sh pilot 200 >>"$LOG" 2>&1; sleep 5
  wait_pids "$ROOT/state/embed_views_eval_emb_views.pids"
  if ! grep -q "check: PASS" "$ROOT/logs/embed_views_eval_emb_views.log"; then log "GATE FAILED (no 'check: PASS'); chain aborted"; exit 1; fi
  rm -f "$ROOT/meta/eval_emb_views/"*pilot*.parquet
  log "gate PASS -> step 2: item-6 full pass on the original encoder"
  FORCE=1 bash run_embed_views.sh start >>"$LOG" 2>&1; sleep 5
  wait_pids "$ROOT/state/embed_views_eval_emb_views.pids"
  log "step 2 done; launching CPU probes (bg) -> step 3: item-8 A (channel mask)"
  setsid nohup bash run_views_probes.sh meta/eval_emb_views "v" </dev/null >/dev/null 2>&1 &
  sleep 60
  FORCE=1 bash run_pretrain.sh start --init-from "$ROOT/checkpoints/pretrain_last.pt" --mask-mode channel \
    --epochs 4 --out "$ROOT/checkpoints/pt_channel" --workers 8 >>"$LOG" 2>&1; sleep 5
  wait_pid "$ROOT/state/pretrain.pid"
  [[ -f "$ROOT/checkpoints/pt_channel/pretrain_last.pt" ]] || { log "step 3 produced no checkpoint; aborted"; exit 1; }
  log "step 3 done -> step 4: item-8 B (random mask, extra-steps control)"
  FORCE=1 bash run_pretrain.sh start --init-from "$ROOT/checkpoints/pretrain_last.pt" --mask-mode random \
    --epochs 4 --out "$ROOT/checkpoints/pt_random4" --workers 8 >>"$LOG" 2>&1; sleep 5
  wait_pid "$ROOT/state/pretrain.pid"
  [[ -f "$ROOT/checkpoints/pt_random4/pretrain_last.pt" ]] || { log "step 4 produced no checkpoint; aborted"; exit 1; }
  log "step 4 done -> step 5: views pass on A"
  FORCE=1 CKPT="$ROOT/checkpoints/pt_channel/pretrain_last.pt" OUT_DIR="$ROOT/meta/eval_emb_views_channel" \
    bash run_embed_views.sh start >>"$LOG" 2>&1; sleep 5
  wait_pids "$ROOT/state/embed_views_eval_emb_views_channel.pids"
  setsid nohup bash run_views_probes.sh meta/eval_emb_views_channel "ch" </dev/null >/dev/null 2>&1 &
  log "step 5 done (probes bg) -> step 6: views pass on B"
  FORCE=1 CKPT="$ROOT/checkpoints/pt_random4/pretrain_last.pt" OUT_DIR="$ROOT/meta/eval_emb_views_random4" \
    bash run_embed_views.sh start >>"$LOG" 2>&1; sleep 5
  wait_pids "$ROOT/state/embed_views_eval_emb_views_random4.pids"
  setsid nohup bash run_views_probes.sh meta/eval_emb_views_random4 "rd" </dev/null >/dev/null 2>&1 &
  log "step 6 done (probes bg). GPU chain complete; probes finish on CPU, then: python paper_tables.py"
}
case "${1:-}" in
  start) [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null && { echo "chain already armed"; exit 1; }
         setsid nohup bash -c "$(declare -f log runoff_busy gpus_free any_alive wait_pids wait_pid chain); ROOT='$ROOT'; PY='$PY'; RUNOFF='$RUNOFF'; LOG='$LOG'; chain" </dev/null >/dev/null 2>&1 &
         echo $! >"$PIDF"; echo "chain armed (pid $(cat "$PIDF")); log $LOG" ;;
  status) [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null && echo "[chain] ARMED (pid $(cat "$PIDF"))" || echo "[chain] not running"; tail -n 6 "$LOG" 2>/dev/null ;;
  stop) [[ -f "$PIDF" ]] && kill "$(cat "$PIDF")" 2>/dev/null && echo stopped ;;
  *) echo "usage: bash $0 start|status|stop"; exit 1 ;;
esac
