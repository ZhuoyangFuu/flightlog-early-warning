#!/usr/bin/env python
"""Alarm curves for the HotMobile figure: F(tau) = fraction of loss-of-control events
alarmed at least tau seconds before the failsafe, with the threshold set for FAR 1.6/h.

Uses the SAME harness as everything else (tier1_baseline.flight_curve / threshold_grid,
sequential.transform for the alarm rules, sequential.rosters_for_arm for the roster) so
F(0) equals det@1.6/h and the tau at which F = F(0)/2 equals the reported median lead.
The threshold is interpolated at exactly 1.6/h the way op_point does it (linear in FAR
between the two bracketing grid thresholds).

  python alarm_curves.py learned          # the learned arms (small dumps, ~10 min)
  python alarm_curves.py tier0 --workers 6  # the 15 classical dumps x 11 rules x 2 splits
Writes HotMobile/curves/<dump>__<rule>__<param>__<split>.csv with columns tau, F.
"""
import argparse, os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/wsssd/FlightLog")
from tier1_baseline import REF_FAR, HZ, threshold_grid, flight_curve
from sequential import (rosters_for_arm, rules_for, transform, DT_LEARNED, DT_TIER0, SCORES, META)

OUT = "/mnt/wsssd/FlightLog/HotMobile/curves"
TAU = np.arange(0, 401, 1.0)
SPLITS = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))

def leads_at(sbf, roster, th):
    pos = roster[roster.t_e_bin >= 0]
    out = np.full(len(pos), -1.0)
    for i, (lid, te) in enumerate(zip(pos.log_id, pos.t_e_bin)):
        t, s = sbf.get(lid, (None, None))
        if t is None: continue
        hit = np.flatnonzero(s >= th)
        if len(hit) and t[hit[0]] < te:
            out[i] = (te - t[hit[0]]) / HZ
    return out   # -1 = missed

def F_of(leads):
    return np.array([(leads >= tau).mean() for tau in TAU])

def curve_at_ref(sbf, roster, grid):
    c = flight_curve(sbf, roster, grid).dropna(subset=["far_per_hour"]).sort_values("far_per_hour")
    f = c.far_per_hour.to_numpy(float); th = c.threshold.to_numpy(float)
    if not (f.min() <= REF_FAR <= f.max()):
        return None, None
    j = np.searchsorted(f, REF_FAR)          # f[j-1] <= REF <= f[j]
    if j == 0: return F_of(leads_at(sbf, roster, th[0])), th[0]
    lo, hi = j - 1, j
    w = 0.0 if f[hi] == f[lo] else (REF_FAR - f[lo]) / (f[hi] - f[lo])
    Flo, Fhi = F_of(leads_at(sbf, roster, th[lo])), F_of(leads_at(sbf, roster, th[hi]))
    return (1 - w) * Flo + w * Fhi, (1 - w) * th[lo] + w * th[hi]

def do_dump(name, rules="all"):
    path = os.path.join(SCORES, name + ".parquet")
    df = pd.read_parquet(path).sort_values(["log_id", "t_bin"]).reset_index(drop=True)
    rosters = rosters_for_arm(name, SPLITS, df)
    tier0 = name.startswith("ews_scores_")
    dt = DT_TIER0 if tier0 else DT_LEARNED
    scores = df.score.to_numpy(np.float64); lid = df.log_id.to_numpy(); tb = df.t_bin.to_numpy()
    isneg = (df.y == 0).to_numpy()
    mu, sd = float(np.nanmean(scores[isneg])), float(np.nanstd(scores[isneg]))
    starts = np.concatenate(([0], np.flatnonzero(lid[1:] != lid[:-1]) + 1, [len(lid)]))
    filled = np.where(np.isfinite(scores), scores, mu)
    for rule, param, spec in rules_for(dt):
        if rules == "raw" and rule != "raw": continue
        v = transform(rule, spec, mu, sd, starts, scores if rule == "raw" else filled)
        neg = v[isneg & np.isfinite(v)]
        if neg.size < 4: continue
        sbf = {lid[a]: (tb[a:b], v[a:b]) for a, b in zip(starts[:-1], starts[1:])}
        grid = threshold_grid(neg)
        for label, roster in rosters:
            if tier0 and label not in ("B_fixed", "C"): continue
            sp0 = label if label is not None else "own"
            if os.path.exists(os.path.join(OUT, f"{name}__{rule}__{str(param).replace('/', 'of')}__{sp0}.csv")):
                continue
            F, th = curve_at_ref(sbf, roster, grid)
            if F is None: continue
            sp = label if label is not None else "own"
            fn = os.path.join(OUT, f"{name}__{rule}__{str(param).replace('/', 'of')}__{sp}.csv")
            pd.DataFrame({"tau": TAU, "F": F}).to_csv(fn, index=False)
            with open(fn + ".th", "w") as fh: fh.write(f"{th}\n")
            print(f"  {name} {rule} {param} [{sp}] F0={F[0]:.3f} th={th:.4g}", flush=True)

LEARNED = ([f"horizon_scores_shift120_{sp}{s}{t}" for sp in ("B_fixed", "C")
            for s in ("", "_seed1", "_seed2") for t in ("", "_relcat")]
           + [f"horizon_scores_shift120_C{s}_tsfm" for s in ("", "_seed1", "_seed2")]
           + [f"h1_scores_lr_shift120_{sp}{s}" for sp in ("B_fixed", "C") for s in ("", "_seed1", "_seed2")])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["learned", "tier0", "one"])
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--name")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.what == "one": do_dump(a.name); sys.exit()
    names = LEARNED if a.what == "learned" else sorted(
        os.path.basename(p)[:-8] for p in glob.glob(os.path.join(SCORES, "ews_scores_*.parquet"))
        if "smoke" not in p)
    rules = "raw" if a.what == "learned" else "all"
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(do_dump, names, [rules] * len(names)))
    else:
        for n in names: do_dump(n, rules)
    open(os.path.join(OUT, f"{a.what}.done"), "w").write("done\n")
