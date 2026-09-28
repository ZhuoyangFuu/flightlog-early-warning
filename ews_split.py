#!/usr/bin/env python
"""Re-score Tier-0 on each SPLIT'S TEST ARM (PROJECT_PLAN §7.5).

Why this exists
---------------
Tier-0's frozen headline -- detection 0.186, median lead 66.0 s at FAR 1.617/h --
is computed over the **whole curated tier: 1,777 positive flights**. The learned
tiers are scored on their split's *test arm*: **354 positives on split A**, 1,439
on `B_fixed`. Those are different populations, and on split A the Tier-0 number
is measured over flights that split A *trains* on.

So the comparison that `RESULTS.md` §3 and §4.3 make -- Tier-0's 0.186/66.0 s in
the same table as a split-A model's 0.708 -- is **not like-for-like**, and a
dominance claim (§7.5) cannot rest on it. This file fixes that: it recomputes
Tier-0's operating point restricted to exactly the flights each split tests on.

Nothing about Tier-0 is refitted or retuned. The frozen configuration (w=10 s,
top-3, v2, `meta/ews_frozen_config.json`) is untouched; `meta/ews_traces_*.parquet`
already stores, per flight, the **first bin at which the EWS score crosses each of
16 fixed thresholds**. Restricting to a flight subset is therefore a pure
re-aggregation of cached crossings -- the indicator itself never runs again.

The curve is rebuilt with `tier1_baseline.flight_curve`'s exact survival-estimator
semantics (alarmed hours accrue to the first crossing on an alarmed negative and
to end-of-flight otherwise; a positive counts as detected only if its crossing
precedes `t_e`), against the roster from `splits_eval.parquet` -- the same roster
`eval_common.roster_for` hands the learned tiers.

Known limitation, stated rather than hidden: the traces carry 16 fixed thresholds,
where the learned tiers use a 130-point log-spaced grid. The low-FAR tail is
therefore coarser for Tier-0 than for the tiers it is being compared against, and
`det_at_1.6h` is interpolated across a wider gap. It is enough to place the
operating point; it is not enough to draw a smooth Tier-0 frontier.

Usage
-----
    python ews_split.py                 # all splits -> meta/ews_bysplit.json
    python ews_split.py --split A
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier1_baseline import (WARMUP, HZ, NEG_MIN_BINS, REF_FAR, FAR_CAP, op_point)

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")

SPLIT_COL = {"A": "split_A_s0", "B": "split_B", "B_fixed": "split_B_fixed",
             "C": "split_C_cond"}


def frozen():
    with open(os.path.join(META, "ews_frozen_config.json")) as f:
        return json.load(f)


def load_traces(tag):
    """-> DataFrame(log_id, cross) where `cross` maps threshold(float) -> first bin."""
    p = os.path.join(META, f"ews_traces_{tag}.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    t = pd.read_parquet(p, columns=["log_id", "status", "cross"])
    t = t[t.status == "ok"].copy()
    t["cross"] = t["cross"].map(
        lambda s: {float(k): int(v) for k, v in json.loads(s).items()})
    return t[["log_id", "cross"]]


def curve_for(ids, splits, cross_by_id, thresholds):
    """`tier1_baseline.flight_curve` semantics, driven by cached crossings.

    Restricted to `ids`. A flight with no trace is a MISS on the positive side and
    contributes its full duration to alarmed hours on the negative side -- the same
    conservative treatment the learned tiers get for a flight with no scoreable
    window (RESULTS.md §4.1, "integrity checks").
    """
    r = splits[splits.log_id.isin(ids)]
    pos = r[r.t_e_bin >= 0]
    neg = r[(r.t_e_bin < 0) & (r.n_bins >= NEG_MIN_BINS)]
    rows = []
    for th in thresholds:
        alarmed_h, n_alarm = 0.0, 0
        for lid, nb in zip(neg.log_id, neg.n_bins):
            c = cross_by_id.get(lid, {}).get(th, -1)
            if c is not None and c >= 0:
                n_alarm += 1
                alarmed_h += (c - WARMUP) / HZ / 3600
            else:
                alarmed_h += (nb - WARMUP) / HZ / 3600
        far = n_alarm / alarmed_h if alarmed_h > 0 else np.nan
        det, leads = 0, []
        for lid, te in zip(pos.log_id, pos.t_e_bin):
            c = cross_by_id.get(lid, {}).get(th, -1)
            if c is not None and c >= 0 and c < te:
                det += 1
                leads.append((te - c) / HZ)
        rows.append({"threshold": float(th), "far_per_hour": far,
                     "n_neg_alarmed": n_alarm,
                     "detection_rate": det / len(pos) if len(pos) else np.nan,
                     "n_detected": det,
                     "median_lead_s": float(np.median(leads)) if leads else np.nan,
                     "p25_lead_s": float(np.percentile(leads, 25)) if leads else np.nan,
                     "p75_lead_s": float(np.percentile(leads, 75)) if leads else np.nan})
    return pd.DataFrame(rows).sort_values("far_per_hour").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_COL))
    # The frozen config names tag `w10k3`, whose grid stops at 80 -- the edge Tier-0
    # sits on for `B_fixed`. `--traces w10k3ext` swaps in the extended-grid re-run of
    # the SAME configuration (w=10 s, top-3); only the threshold list differs, so the
    # indicator is unchanged and the frozen Q1 headline is untouched. Note the freeze
    # rule is "max detection subject to FAR <= cap", and every ADDED threshold is
    # higher (hence lower detection), so extending the grid cannot move the
    # corpus-wide frozen point -- it can only add operating points below it.
    ap.add_argument("--traces", default=None,
                    help="trace tag to read instead of the frozen config's "
                         "(e.g. w10k3ext). Output tags follow it.")
    ap.add_argument("--out-suffix", default="", help="suffix for output filenames")
    args = ap.parse_args()

    cfg = frozen()
    if args.traces:
        cfg = dict(cfg, tag=args.traces)
    print(f"frozen Tier-0 config: {cfg['tag']} / {cfg['labelset']} "
          f"(w={cfg['window_s']}s, top-{cfg['topk']})")
    print(f"corpus-wide headline: det {cfg['detection_rate']:.3f}  "
          f"lead {cfg['median_lead_s']:.1f}s  FAR {cfg['far_per_hour']:.3f}/h\n")

    tr = load_traces(cfg["tag"])
    cross_by_id = dict(zip(tr.log_id, tr["cross"]))
    thresholds = sorted({t for d in tr["cross"].head(1) for t in d})
    print(f"{len(cross_by_id):,} traces, {len(thresholds)} cached thresholds: "
          f"{thresholds}\n")

    splits = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))
    out, rows = {}, []
    for name in ([args.split] if args.split else list(SPLIT_COL)):
        col = SPLIT_COL[name]
        if col not in splits.columns:
            print(f"  {name}: no column {col} -- skipped")
            continue
        ids = set(splits.loc[splits[col] == "test", "log_id"])
        c = curve_for(ids, splits, cross_by_id, thresholds)
        c.to_csv(os.path.join(META, f"ews_curve_split_{name}{args.out_suffix}.csv"),
                 index=False)
        pt = op_point(c)
        npos = int(((splits[col] == "test") & (splits.t_e_bin >= 0)).sum())
        pt.update(split=name, n_test_flights=len(ids), n_test_pos=npos)
        out[name] = pt
        rows.append(pt)
        print(f"  {name:8s} n_test {len(ids):6,}  pos {npos:5,}  "
              f"det@{REF_FAR}h {pt.get(f'det_at_{REF_FAR}h', float('nan')):.3f}  "
              f"lead {pt.get(f'lead_at_{REF_FAR}h', float('nan')):.1f}s  "
              f"| freeze-rule det {pt.get('detection_rate', float('nan')):.3f} "
              f"lead {pt.get('median_lead_s', float('nan')):.1f}s "
              f"@ FAR {pt.get('far_per_hour', float('nan')):.3f}")

    outp = os.path.join(META, f"ews_bysplit{args.out_suffix}.json")
    with open(outp, "w") as f:
        json.dump({"config": cfg, "by_split": out}, f, default=float, indent=1)
    print(f"\nwrote {outp}")
    print("\nThese, not the corpus-wide 0.186/66.0s, are the Tier-0 reference points\n"
          "a split-A or split-B_fixed model must be compared against.")


if __name__ == "__main__":
    main()
