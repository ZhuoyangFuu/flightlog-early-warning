#!/usr/bin/env bash
# CPU probes on one item-6 views directory. Usage:
#   bash run_views_probes.sh <views_dir> <tag_prefix>     e.g.  meta/eval_emb_views ""  |  meta/eval_emb_views_channel ch
# Tags written: {prefix}role, {prefix}dev, {prefix}mean -- all registered in paper_tables.FEATURE_TAGS.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
DIR="$1"; PRE="${2:-}"
LOG="$ROOT/logs/views_probes_$(basename "$DIR").log"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export FLIGHTLOG_EMB_DIR="$ROOT/$DIR"
cd "$ROOT"
{
echo "=== views probes $(date) dir=$DIR prefix=$PRE ==="
for spec in "role:emb_role,emb_last" "dev:emb_dev,emb_last,dev_norm" "mean:emb_mean,emb_last"; do
  tag="${PRE}${spec%%:*}"; views="${spec#*:}"
  for split in B_fixed C; do
    for head in lr gbt; do
      echo "=== $(date) tag=$tag split=$split head=$head shift120 ==="
      FLIGHTLOG_FEATURE_TAG=$tag $PY horizon.py --split $split --mode shift --h 120 --views $views --seeds 0,1,2 --head $head --dump-scores
    done
    echo "=== $(date) tag=$tag split=$split lr widen10 ==="
    FLIGHTLOG_FEATURE_TAG=$tag $PY horizon.py --split $split --mode widen --h 10 --views $views --seeds 0,1,2 --head lr --dump-scores
  done
done
echo "=== views probes done $(date) ==="
} >>"$LOG" 2>&1
