#!/bin/bash
# Wait for the curated re-tensorization to finish, then trim raw.
# Guarded: never deletes while tensorize.py is still reading raw/.
set -u
cd /mnt/wsssd/FlightLog
PY=/home/zhuoyang/venvs/flightlog/bin/python
echo "[$(date +%H:%M:%S)] waiting for curated tensorize to finish..."
while pgrep -f "tensorize.py --raw-dir raw" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] tensorize done; verifying before trim"
if [ ! -f meta/quality_curated.parquet ]; then
    echo "ABORT: no quality_curated.parquet"; exit 1
fi
echo "--- DRY RUN ---"
$PY trim_raw.py
echo "--- EXECUTING ---"
$PY trim_raw.py --execute
df -h /mnt/wsssd | tail -1
