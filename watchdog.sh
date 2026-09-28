#!/bin/bash
# FlightLog stall watchdog -- detached, survives SSH disconnect.
#
# Alerts and hourly heartbeats append to logs/watchdog.log. Nothing here talks
# to a terminal, so it must be run as its own session leader:
#
#   cd /mnt/wsssd/FlightLog
#   setsid nohup bash watchdog.sh >> logs/watchdog.log 2>&1 < /dev/null &
#
# Healthy backpressure (producer paused while the consumer drains) shows a flat
# `done` count, so a flat `done` ALONE is not a stall -- that is what made the
# Aug 8 deadlock invisible for two days. A real deadlock is both sides frozen:
# `done` flat AND the staging file count flat.
cd /mnt/wsssd/FlightLog || { echo "WATCHDOG-ERROR: cannot cd to project"; exit 1; }
PY=/home/zhuoyang/venvs/flightlog/bin/python
INTERVAL=300          # 5 min between polls
STALL_POLLS=3         # 3 flat polls = 15 min frozen = stall
HEARTBEAT=12          # every 12 polls = 1 h

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

dbcount() {
  $PY -c "
import sqlite3
try:
    c=sqlite3.connect('file:state/bulk.db?mode=ro', uri=True)
    print(c.execute('select count(*) from dl where status $1').fetchone()[0])
except Exception: print(-1)" 2>/dev/null
}

prev_done=-1; prev_stag=-1; flat=0; warned=0; tick=0
say "watchdog up (pid $$, interval ${INTERVAL}s)"

while true; do
  done_n=$(dbcount '="done"')
  pend_n=$(dbcount 'in ("pending","running")')
  stag_n=$(find staging -name '*.ulg' 2>/dev/null | wc -l)
  cons_n=$(ps -eo args --no-headers | grep -F 'tensorize_loop.py' | grep -cv 'bin/bash')
  prod_n=$(ps -eo args --no-headers | grep -F 'download.py --bulk-only' | grep -cv 'bin/bash')
  free_g=$(df -BG --output=avail /mnt/wsssd | tail -1 | tr -dc '0-9')
  shards=$(ls proc/bulk-*.parquet 2>/dev/null | wc -l)

  if [ "$pend_n" = "0" ]; then
    say "COMPLETE: all bulk logs downloaded. staging=$stag_n shards=$shards."
    say "  Follow-up: 938-log top-up from meta/lost_unflushed_logs.csv (reset those rows in state/bulk.db to pending)."
    exit 0
  fi
  if [ "$prod_n" -eq 0 ]; then
    say "STALL: producer GONE with $pend_n pending. Restart per RESUME.md section 2."
    exit 1
  fi
  if [ "$cons_n" -eq 0 ]; then
    say "STALL: consumer GONE with $stag_n .ulg staged, $pend_n pending. Restart per RESUME.md section 2."
    exit 1
  fi
  if [ -n "$free_g" ] && [ "$free_g" -lt 450 ]; then
    say "DISK WARNING: /mnt/wsssd free ${free_g}G, nearing the 400G guard that hard-stops both jobs."
  fi

  if [ "$done_n" = "$prev_done" ] && [ "$stag_n" = "$prev_stag" ]; then
    flat=$((flat+1))
    if [ "$flat" -ge "$STALL_POLLS" ] && [ "$warned" -eq 0 ]; then
      say "STALL: no progress for $((flat*INTERVAL/60)) min. done=$done_n staging=$stag_n pending=$pend_n (both frozen -- the Aug 8 signature). Check logs/bulk_consumer.log and logs/bulk_download.log."
      warned=1
    fi
  else
    [ "$warned" -eq 1 ] && say "RECOVERED: done=$done_n staging=$stag_n pending=$pend_n"
    flat=0; warned=0
  fi

  tick=$((tick+1))
  if [ $((tick % HEARTBEAT)) -eq 0 ]; then
    say "ok: done=$done_n pending=$pend_n staging=$stag_n shards=$shards free=${free_g}G"
  fi

  prev_done=$done_n; prev_stag=$stag_n
  sleep $INTERVAL
done
