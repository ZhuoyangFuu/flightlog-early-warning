#!/bin/bash
# Unattended pipeline: wait for curated download -> tensorize curated ->
# stream the 429k-log bulk corpus (download + tensorize + discard raw).
#
# Every stage is resumable. Killing this script mid-way loses no completed work:
# the downloader keeps state in SQLite, the consumer writes shards incrementally.
#
# Launch:  setsid nohup bash run_bulk.sh > logs/bulk_pipeline.log 2>&1 < /dev/null &
# Status:  bash status.sh

set -u
cd /mnt/wsssd/FlightLog
PY=/home/zhuoyang/venvs/flightlog/bin/python

echo "=== [$(date +%H:%M:%S)] stage 0: waiting for curated download to finish ==="
while pgrep -f "download.py --curated-only" > /dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] curated download process has exited"
$PY download.py --status

# Guard: only proceed if the curated set actually completed. A guard-stop or a
# crash should NOT silently roll into a 36-hour bulk run on a partial dataset.
PENDING=$($PY - <<'EOF'
import sqlite3
c = sqlite3.connect("state/download.db")
print(c.execute("SELECT COUNT(*) FROM dl WHERE status != 'done'").fetchone()[0])
EOF
)
echo "[$(date +%H:%M:%S)] curated rows not done: $PENDING"
if [ "$PENDING" -gt 200 ]; then
    echo "ABORT: curated download incomplete ($PENDING pending). Fix before bulk run."
    exit 1
fi

echo
echo "=== [$(date +%H:%M:%S)] stage 1: tensorize curated set (raw retained) ==="
$PY tensorize.py \
    --raw-dir raw --out-dir proc --tag curated \
    --workers 24 --shard-size 2000
echo "[$(date +%H:%M:%S)] curated tensorization done"

echo
echo "=== [$(date +%H:%M:%S)] stage 2: bulk stream (429k logs) ==="
mkdir -p staging
# Consumer first, so staging never accumulates unattended.
setsid nohup $PY tensorize_loop.py \
    --staging staging --out-dir proc --tag bulk \
    --workers 24 --shard-size 2000 --batch 600 --idle-exit-s 2400 \
    > logs/bulk_consumer.log 2>&1 < /dev/null &
echo "[$(date +%H:%M:%S)] consumer started (pid $!)"
sleep 10

# Producer: pauses when staging fills, resumes as the consumer drains it.
setsid nohup $PY download.py \
    --bulk-only --raw-dir /mnt/wsssd/FlightLog/staging \
    --db /mnt/wsssd/FlightLog/state/bulk.db \
    --workers 10 --wait-on-cap --max-raw-gb 60 --min-free-gb 400 \
    > logs/bulk_download.log 2>&1 < /dev/null &
echo "[$(date +%H:%M:%S)] producer started (pid $!)"

echo
echo "Both stages detached. Monitor with: bash status.sh"
