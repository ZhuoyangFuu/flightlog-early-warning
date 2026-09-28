#!/usr/bin/env bash
# Rebuild meta/sequential.csv from EVERY score dump in meta/scores, one detached parallel
# pass (4 workers x 4 threads). Each arm writes meta/sequential_parts/<arm>.csv; the
# combined table is the union of the parts (sequential.py, 2026-09-14 fix).
#   bash run_seq_all.sh start | status | tail
ROOT=/mnt/wsssd/FlightLog; PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG=$ROOT/logs/seq_all.log; PIDF=$ROOT/state/seq_all.pid; PARTS=$ROOT/meta/sequential_parts
run(){ cd $ROOT; mkdir -p $PARTS
  ls meta/scores/*.parquet | xargs -n1 basename | sed 's/\.parquet$//' | while read -r a; do
    [[ -s $PARTS/$a.csv ]] && continue; echo "$a"; done \
  | OMP_NUM_THREADS=4 xargs -P 4 -I{} bash -c 'echo "=== $(date "+%F %T") {} ===" >>'"$LOG"'; nice -n 19 '"$PY"' sequential.py --arm {} >>'"$LOG"' 2>&1 || echo "!! {} failed" >>'"$LOG"
  echo "=== seq_all done $(date '+%F %T'): $(ls $PARTS/*.csv | wc -l) parts ===" >>"$LOG"; }
case "${1:-status}" in
  start) [[ -f $PIDF ]] && kill -0 "$(cat $PIDF)" 2>/dev/null && { echo running; exit 1; }
         setsid nohup bash -c "$(declare -f run); ROOT=$ROOT; PY=$PY; LOG=$LOG; PARTS=$PARTS; run" </dev/null >/dev/null 2>&1 & echo $! >$PIDF; echo "started (pid $(cat $PIDF)); log $LOG" ;;
  status) [[ -f $PIDF ]] && kill -0 "$(cat $PIDF)" 2>/dev/null && echo "[seq_all] RUNNING (pid $(cat $PIDF))" || echo "[seq_all] not running"; echo "parts: $(ls $PARTS/*.csv 2>/dev/null | wc -l) / $(ls $ROOT/meta/scores/*.parquet | wc -l)"; tail -n 2 $LOG 2>/dev/null ;;
  tail) tail -f $LOG ;;
esac
