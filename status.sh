#!/bin/bash
# One-command status for the whole FlightLog pipeline.
cd /mnt/wsssd/FlightLog
PY=/home/zhuoyang/venvs/flightlog/bin/python

echo "############ FlightLog pipeline  $(date '+%Y-%m-%d %H:%M:%S') ############"
echo "results: RESULTS.md   design: PROJECT_PLAN.md   live state: RESUME.md"
echo
echo "--- processes ---"
pgrep -af "download.py|tensorize_loop.py|tensorize.py|run_bulk.sh|pretrain.py|embed_eval.py|probe.py" \
    | sed 's/\/home\/zhuoyang\/venvs\/flightlog\/bin\///' || echo "  (none running)"

echo
echo "--- curated download ---"
$PY download.py --status 2>/dev/null | sed 's/^/  /'

if [ -f state/bulk.db ]; then
    echo
    echo "--- bulk download ---"
    $PY download.py --status --db /mnt/wsssd/FlightLog/state/bulk.db \
        --raw-dir /mnt/wsssd/FlightLog/staging 2>/dev/null | sed 's/^/  /'
fi

echo
echo "--- tensorized output ---"
$PY tensorize_loop.py --status --tag bulk --out-dir proc 2>/dev/null | sed 's/^/  /'
if ls proc/curated-*.parquet >/dev/null 2>&1; then
    n=$(ls proc/curated-*.parquet | wc -l)
    sz=$(du -sm proc/curated-*.parquet 2>/dev/null | awk '{s+=$1} END {print s}')
    echo "  curated shards  : $n  (${sz} MB)"
fi

echo
echo "--- EWS sweep (Q1 freeze) ---"
if pgrep -f "ews_sweep.py" >/dev/null 2>&1; then
    echo "  RUNNING (pid $(pgrep -of 'ews_sweep.py'))"
else
    echo "  not running"
fi
$PY ews_sweep.py --status 2>/dev/null | sed 's/^/  /'

echo
echo "--- Tier-1 baselines ---"
if pgrep -f "tier1_baseline.py" >/dev/null 2>&1; then
    echo "  RUNNING (pid $(pgrep -of 'tier1_baseline.py'))"
else
    echo "  not running"
fi
$PY tier1_baseline.py --status 2>/dev/null | sed 's/^/  /'

echo
echo "--- pretraining (§7.2) ---"
if [ -f meta/pretrain_status.json ]; then
    $PY -c "import json;d=json.load(open('meta/pretrain_status.json'));print(f\"  {d['state']}  step {d['step']}/{d['total_steps']}  epoch {d['epoch']}  ({d['updated']})\")"
else
    echo "  not started"
fi

echo
echo "--- Tier-2 embedding pass (§7.4) ---"
$PY embed_eval.py --status 2>/dev/null | tail -1 | sed 's/^/  /' || echo "  not started"

echo
echo "--- Tier-2 probes (§7.4.2) ---"
if pgrep -f "bin/python .*probe\.py" >/dev/null 2>&1; then
    echo "  RUNNING (pid $(pgrep -of 'bin/python .*probe\.py'))"
else
    echo "  not running"
fi
if ls meta/tier2_pt_*.json >/dev/null 2>&1; then
    for f in meta/tier2_pt_*.json; do
        $PY -c "
import json;d=json.load(open('$f'))
k=[x for x in d if x.startswith('det_at_')]
conv='' if d.get('converged',True) else '  UNCONVERGED'
print('  $(basename $f .json)'.ljust(34), 'det', round(d.get(k[0],float('nan')),3) if k else 'n/a',
      ' auprc', round(d.get('window_auprc',float('nan')),4), conv)"
    done
    # `_it<N>` entries are max-iter refits, not extra grid jobs -- count apart.
    echo "  $(ls meta/tier2_pt_*.json | grep -cv '_it[0-9]*\.json$') of 12 base jobs cached; $(ls meta/tier2_pt_*.json | grep -c '_it[0-9]*\.json$') max-iter refit(s)"
else
    echo "  no probes finished"
fi

echo
echo "--- Tier-2 fine-tuning (§7.4.1) ---"
if ls meta/ft_pt_*.json >/dev/null 2>&1; then
    for f in meta/ft_pt_*.json; do
        $PY -c "
import json;d=json.load(open('$f'))
k=[x for x in d if x.startswith('det_at_')]
print('  $(basename $f .json)'.ljust(40), 'det', round(d.get(k[0],float('nan')),3) if k else 'n/a')"
    done
else
    echo "  no runs finished"
fi
# Match the python invocation only, and exclude --smoke: a bare `pgrep -f
# finetune.py` also matches any shell whose command line mentions the script
# (a wait-loop, a tail, this file), which reads as a phantom running job.
if pgrep -af "bin/python .*finetune\.py" 2>/dev/null | grep -qv -- "--smoke"; then
    echo "  RUNNING"; pgrep -af "bin/python .*finetune\.py" | grep -v -- "--smoke" | sed 's/^/    /'
else
    echo "  not running"
fi

echo
echo "--- detection-lead frontier (§7.5, item 1) ---"
# Read-only view over the cached curves; the interesting line is whether ANY
# model has entered the dominance quadrant, so print that rather than the table.
if [[ -f "meta/frontier.csv" ]]; then
    "$PY" -c "
import pandas as pd
t = pd.read_csv('meta/frontier.csv')
learned = t[t.tier != 'ews']
dom = learned[learned.dominates] if 'dominates' in learned else learned.iloc[:0]
print('  %d arms scored; DOMINATING TIER-0 (per split): %s'
      % (len(learned), ', '.join(dom.model + ' ' + dom.split) if len(dom) else 'NONE'))
# Per-split bars only -- the corpus-wide 0.186/66.0s is a different flight
# population (RESULTS.md §0, §2.1) and must never be printed as the bar.
for sp in ('A', 'B_fixed', 'C'):
    g = learned[learned.split == sp]
    if not len(g):
        continue
    b = g.loc[g['lead@det_ews'].idxmax()]
    print('  %-8s bar det %.3f / lead %5.1fs%s | best arm: %s (%.1fs at matched det)'
          % (sp, b.ews_det, b.ews_lead,
             '' if b.ews_admissible else ' [FAR>cap]', b.model, b['lead@det_ews']))" \
    2>/dev/null || echo "  (unreadable)"
else
    echo "  not built -- run: python frontier.py"
fi

echo
echo "--- training-horizon sweep (§7.5, item 2) ---"
if pgrep -af "bin/python .*horizon\.py" 2>/dev/null | grep -q .; then
    echo "  RUNNING"; pgrep -af "bin/python .*horizon\.py" | sed 's/^/    /'
else
    echo "  not running"
fi
n=$(ls meta/horizon_pt_*.json 2>/dev/null | wc -l)
echo "  $n arms cached   (detail: bash run_horizon.sh status)"
echo "  score dumps: $(ls meta/scores/horizon_scores_*.parquet 2>/dev/null | wc -l) (needed by items 3/3b/4)"

echo
echo "--- sequential alarm rules (§4.6, item 3) ---"
if pgrep -af "bin/python .*sequential\.py" 2>/dev/null | grep -q .; then
    echo "  RUNNING"; pgrep -af "bin/python .*sequential\.py" | sed 's/^/    /'
else
    echo "  not running"
fi
if [[ -f meta/sequential.csv ]]; then
    # The `raw` control must reproduce each arm's cached curve; surface the best
    # rule per split rather than the full 10-rule table, which is too wide here.
    "$PY" -c "
import pandas as pd, numpy as np
t = pd.read_csv('meta/sequential.csv')
d = [c for c in t.columns if c.startswith('det@')][0]
l = [c for c in t.columns if c.startswith('lead@')][0]
print('  %d rows over %d arms' % (len(t), t.arm.nunique()))
tier0 = t[t.arm.str.startswith('ews_scores_')]
print('  Tier-0 swept under the same grid: %s'
      % ('YES (%d rows)' % len(tier0) if len(tier0) else 'NO -- item 3b(i), claims are NOT fair yet'))
for arm, g in t.groupby('arm'):
    raw = g[g.rule == 'raw']
    if not len(raw): continue
    floor = raw.iloc[0][d]
    ok = g[g[d] >= floor]
    if not len(ok): continue
    b = ok.loc[ok[l].idxmax()]
    print('  %-38s best %s %-9s det %.3f lead %6.1fs  (raw %.3f / %.1fs)'
          % (arm[:38], b.rule, str(b.param), b[d], b[l], floor, raw.iloc[0][l]))" \
    2>/dev/null | head -20 || echo "  (unreadable)"
    # This table is PER SEED and PER ARM. Two of the project's standing rules say
    # it must not be read as a result, and both were learned by publishing a wrong
    # one -- so the warning ships next to the numbers, not in a doc nobody opens.
    echo "  ⚠ per-seed, best-of-N: NOT a result. Quote only n=3 means, and only"
    echo "    against Tier-0's BEST-of-11 under the same rule (RESULTS.md §4.6)."
    echo "    FINAL: split A 0/30 dominate; B_fixed 14/20 vs the strict bar."
else
    echo "  not run -- python sequential.py --all"
fi

echo
echo "--- Tier-0 score dump (item 3b(i)) ---"
if [[ -f state/ews_scores.pid ]] && kill -0 "$(cat state/ews_scores.pid)" 2>/dev/null; then
    echo "  RUNNING (pid $(cat state/ews_scores.pid))"
else
    echo "  not running"
fi
if ls meta/scores/ews_scores_*.parquet >/dev/null 2>&1; then
    du -h meta/scores/ews_scores_*.parquet | sed 's/^/  dump: /'
    echo "  VERIFY BEFORE USE: bash run_ews_scores.sh verify"
else
    echo "  not dumped -- bash run_ews_scores.sh start"
fi

echo
echo "--- shifted-band Tier-1 control (§4.7) ---"
if [[ -f state/h1.pid ]] && kill -0 "$(cat state/h1.pid)" 2>/dev/null; then
    echo "  RUNNING (pid $(cat state/h1.pid))"
else
    echo "  not running"
fi
echo "  $(ls meta/h1_pt_*.json 2>/dev/null | wc -l) of 14 arms cached"
grep -h "CONTROL vs frozen" logs/h1.log 2>/dev/null | tail -2 | sed 's/^ */  gate: /'
OMP_NUM_THREADS=1 "$PY" horizon_tier1.py --status 2>/dev/null \
  | sed -n '/vs Tier-0 best-of-11/,/^$/p' | sed 's/^/  /'
echo "  ⚠ 3 of 4 Tier-1 shifted configs DOMINATE Tier-0 with no pretraining."
echo "    Dominance is a LABEL-BAND result, not a representation result (RESULTS.md §4.7)."

echo
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /'

echo
echo "--- disk ---"
df -h /mnt/wsssd | tail -1 | awk '{print "  /mnt/wsssd  used "$3"  free "$4"  ("$5" full)"}'
du -sh raw staging proc 2>/dev/null | sed 's/^/  /'
