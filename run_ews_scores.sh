#!/usr/bin/env bash
# Detached launcher for item 3b(i): dump Tier-0's per-bin z-series so
# sequential.py can sweep the SAME rule grid over the baseline that it sweeps
# over the learned tiers (PROJECT_PLAN §11.0.1, RESULTS.md §4.6).
#
#   bash run_ews_scores.sh start
#   bash run_ews_scores.sh status
#   bash run_ews_scores.sh verify     # the control -- run this before using the dump
#   bash run_ews_scores.sh tail
#   bash run_ews_scores.sh stop
#
# /mnt/wsssd is noexec -- always invoke as `bash run_ews_scores.sh`, never ./
#
# WHY THIS EXISTS. §4.6 gave the learned tiers ten sequential alarm rules and
# gave Tier-0 none, which is an asymmetry pointing our way. (The pre-existing
# asymmetry pointed the other way: Tier-0 has always alarmed only after 0.5 s of
# CONSECUTIVE threshold crossings, while every learned tier alarmed on a single
# window.) Neither is a fair comparison. The cached ews_traces_*.parquet keeps
# only first-CROSSING bins per threshold, which throws the series away, so the
# rules cannot be applied after the fact -- hence a re-run with --dump-scores.
#
# CONFIG MUST MATCH THE FROZEN TIER-0 POINT: w10k3, i.e. --window-s 10 --topk 3.
# Anything else produces a series that is not the baseline being compared to.
#
# COST. ~13,668 curated logs, ~8,000 bins each => ~109M rows, ~0.6-1.0 GB of
# Parquet. Roughly 25-40 min at 14 workers on an idle box; it was 20.5 min for
# the same scan without the dump. CPU-only, the GPUs stay free.
#
# THIS BOX IS SHARED. It refuses to start while run_horizon.sh or run_probe.sh
# is running -- those hold a 23 GiB embedding matrix and 8 threads, and three
# heavy CPU jobs at once helps nobody. Override with FORCE=1 only deliberately.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
TAG="${TAG:-w10k3}"
LOG="$ROOT/logs/ews_scores.log"
PIDF="$ROOT/state/ews_scores.pid"
OUT="$ROOT/meta/scores/ews_scores_${TAG}.parquet"
WORKERS="${WORKERS:-14}"

mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/meta/scores"

running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

busy() {
  local p
  for p in horizon probe; do
    if [[ -f "$ROOT/state/$p.pid" ]] && kill -0 "$(cat "$ROOT/state/$p.pid")" 2>/dev/null; then
      echo "$p"; return 0
    fi
  done
  return 1
}

case "${1:-}" in

start)
  if running; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi
  if other=$(busy); then
    echo "REFUSING TO START: run_${other}.sh is running (pid $(cat "$ROOT/state/$other.pid"))."
    echo "  It holds ~23 GiB and 8 threads; wait for it, or override with FORCE=1."
    [[ "${FORCE:-0}" != "1" ]] && exit 1
  fi
  if [[ -f "$OUT" ]]; then
    echo "$OUT already exists ($(du -h "$OUT" | cut -f1)). Delete it to re-dump."
    exit 0
  fi
  echo "=== start $(date '+%F %T')  workers=$WORKERS  tag=$TAG ===" >> "$LOG"
  # setsid + nohup: survives SSH disconnect and logout.
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  setsid nohup "$PY" "$ROOT/ews_baseline.py" \
      --tag "$TAG" --workers "$WORKERS" \
      --window-s 10 --topk 3 --dump-scores \
      >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 2
  echo "started pid $(cat "$PIDF")"
  echo "  log:    bash run_ews_scores.sh tail"
  echo "  status: bash run_ews_scores.sh status"
  echo "  THEN:   bash run_ews_scores.sh verify   <-- do not skip"
  ;;

status)
  if running; then echo "RUNNING (pid $(cat "$PIDF"))"; else echo "not running"; fi
  echo "ews scores status $(date '+%F %T')"
  [[ -f "$OUT" ]] && echo "  dump: $OUT ($(du -h "$OUT" | cut -f1))" \
                  || echo "  dump: not written yet"
  grep -E "^\[shard" "$LOG" 2>/dev/null | tail -3 | sed 's/^/  /'
  echo "  free RAM: $(free -g | awk '/^Mem:/{print $7}') GiB"
  ;;

verify)
  # THE CONTROL. The dumped series must reproduce the cached first-crossing bins
  # in ews_traces_<tag>.parquet exactly, for every log and every threshold. If it
  # does not, the dump is not the baseline and nothing built on it is valid.
  "$PY" - "$TAG" <<'EOF'
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/wsssd/FlightLog")
from ews_baseline import _first_persistent
tag = sys.argv[1]
tr = pd.read_parquet(f"/mnt/wsssd/FlightLog/meta/ews_traces_{tag}.parquet")
sc = pd.read_parquet(f"/mnt/wsssd/FlightLog/meta/scores/ews_scores_{tag}.parquet")
PERSIST = 10          # cfg["persist_bins"] = 0.5 s at 20 Hz
bad = tot = 0
for lid, d in sc.groupby("log_id", sort=False):
    row = tr[tr.log_id == lid]
    if not len(row) or row.iloc[0]["status"] != "ok":
        continue
    d = d.sort_values("t_bin")
    s = d.score.to_numpy(np.float64); t0 = int(d.t_bin.iloc[0])
    for th, want in json.loads(row.iloc[0]["cross"]).items():
        got = _first_persistent(s, float(th), PERSIST)
        got = got + t0 if got >= 0 else -1
        tot += 1
        if got != want:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH {lid} th={th}: dump {got} vs cached {want}")
print(f"first-crossing comparisons: {tot:,}  mismatches: {bad}")
print("CONTROL PASS" if bad == 0 else "CONTROL FAIL -- do not use this dump")
sys.exit(0 if bad == 0 else 1)
EOF
  ;;

tail) tail -f "$LOG" ;;

stop)
  if running; then kill "$(cat "$PIDF")" && echo "stopped"; else echo "not running"; fi
  ;;

*)
  sed -n '2,30p' "$0"
  echo
  echo "usage: bash run_ews_scores.sh {start|status|verify|tail|stop}"
  ;;
esac
