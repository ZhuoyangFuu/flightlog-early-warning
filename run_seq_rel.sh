#!/usr/bin/env bash
# Rule sweep (sequential.py) over the item-5 flight-relative score dumps (rel / relcat),
# so their (band, rule) dominance counts appear in paper_tables.py alongside the other arms.
#   bash run_seq_rel.sh start | status | tail
ROOT=/mnt/wsssd/FlightLog; PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG=$ROOT/logs/seq_rel.log; PIDF=$ROOT/state/seq_rel.pid
run(){ cd $ROOT; for f in meta/scores/horizon_scores_shift120_*_rel.parquet meta/scores/horizon_scores_shift120_*_relcat.parquet; do
    a=$(basename "$f" .parquet); echo "=== $(date '+%F %T') $a ===" >>"$LOG"
    nice -n 19 $PY sequential.py --arm "$a" >>"$LOG" 2>&1 || echo "!! $a failed" >>"$LOG"; done
  echo "=== seq_rel done $(date '+%F %T') ===" >>"$LOG"; }
case "${1:-status}" in
  start) [[ -f $PIDF ]] && kill -0 "$(cat $PIDF)" 2>/dev/null && { echo running; exit 1; }
         export OMP_NUM_THREADS=8; setsid nohup bash -c "$(declare -f run); ROOT=$ROOT; PY=$PY; LOG=$LOG; run" </dev/null >/dev/null 2>&1 & echo $! >$PIDF; echo "started (pid $(cat $PIDF)); log $LOG" ;;
  status) [[ -f $PIDF ]] && kill -0 "$(cat $PIDF)" 2>/dev/null && echo "[seq_rel] RUNNING (pid $(cat $PIDF))" || echo "[seq_rel] not running"; grep -c "^=== 2026" $LOG 2>/dev/null; tail -n 3 $LOG 2>/dev/null ;;
  tail) tail -f $LOG ;;
esac
