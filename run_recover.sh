#!/bin/bash
# Recovery + EWS runner, after the float16 overflow fix.
#
#   1. restart the bulk consumer (drains the 59 GB staging backlog)
#   2. re-tensorize the curated set with float32 + plausibility masking
#   3. run the Tier-0 EWS baseline on the corrected curated tensors
#
# Detached and resumable. Launch:
#   setsid nohup bash run_recover.sh > logs/recover.log 2>&1 < /dev/null &
set -u
cd /mnt/wsssd/FlightLog
PY=/home/zhuoyang/venvs/flightlog/bin/python

echo "=== [$(date +%H:%M:%S)] stage A: restart bulk consumer ==="
if pgrep -f "tensorize_loop.py" > /dev/null; then
    echo "consumer already running"
else
    setsid nohup $PY tensorize_loop.py \
        --staging staging --out-dir proc --tag bulk \
        --workers 16 --shard-size 2000 --batch 600 --idle-exit-s 2400 \
        > logs/bulk_consumer.log 2>&1 < /dev/null &
    echo "consumer started (pid $!)"
fi

echo
echo "=== [$(date +%H:%M:%S)] stage B: re-tensorize curated (float32) ==="
rm -f proc/curated-*.parquet
$PY tensorize.py --raw-dir raw --out-dir proc --tag curated \
    --workers 20 --shard-size 2000
echo "[$(date +%H:%M:%S)] curated re-tensorization done"

echo
echo "=== [$(date +%H:%M:%S)] stage C: validate before trusting it ==="
$PY validate_tensors.py --shard proc/curated-00000.parquet --n 300 > logs/validate_curated.log 2>&1
grep -E "PASS|FAIL" logs/validate_curated.log
if grep -q FAIL logs/validate_curated.log; then
    echo "ABORT: validation failed, not running EWS on bad tensors"
    exit 1
fi

echo
echo "=== [$(date +%H:%M:%S)] stage D: Tier-0 EWS baseline ==="
$PY ews_baseline.py --shards 'proc/curated-*.parquet' --tag curated --workers 14
echo "[$(date +%H:%M:%S)] EWS baseline complete"
