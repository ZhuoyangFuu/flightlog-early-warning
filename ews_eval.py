#!/usr/bin/env python
"""Turn EWS threshold crossings into the Section 8 evaluation.

Primary metric: median lead time at a fixed false-alarm rate of 1 per hour of
flight. Reports the full lead-time versus false-alarm curve, not one operating
point.

FALSE-ALARM RATE ESTIMATOR. Traces record the FIRST crossing per threshold, not
every crossing, so counting alarms directly would undercount flights that alarm
repeatedly. We instead estimate the alarm hazard rate the standard survival way:

    lambda = (# negative flights that ever alarm) / (total time at risk)

where time at risk is the time until first alarm for flights that alarm, and the
full scoreable duration for flights that never do. For a roughly constant hazard
this is the maximum-likelihood rate in alarms per hour, and it does not depend on
having logged repeat crossings.

LEAD TIME. For a positive flight with event bin t_e, an alarm at bin c counts as
a detection only when 0 <= c < t_e; lead time is (t_e - c) / 20 Hz seconds. An
alarm at or after t_e is not a warning and is discarded, not credited.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
HZ = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(ROOT, "meta", "ews_traces_curated.parquet"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "meta", "manifest.parquet"))
    ap.add_argument("--target-far", type=float, default=1.0, help="false alarms per hour")
    ap.add_argument("--out", default=os.path.join(ROOT, "meta", "ews_curve_curated.csv"))
    ap.add_argument("--exclude-csv", default=None,
                    help="csv with a log_id column; those logs are dropped entirely "
                         "(trigger-set-v2 demotions: t_e unreliable, not negatives)")
    args = ap.parse_args()

    df = pd.read_parquet(args.traces)
    df = df[df["status"] == "ok"].copy()
    if args.exclude_csv:
        ex = set(pd.read_csv(args.exclude_csv)["log_id"])
        n0 = len(df)
        df = df[~df["log_id"].isin(ex)].copy()
        print(f"trigger-v2 exclusion: dropped {n0 - len(df)} of {n0} logs")
    man = pd.read_parquet(args.manifest)[["log_id", "mav_type", "vehicle_key", "airframe_type"]]
    df = df.merge(man, on="log_id", how="left")

    df["cross"] = df["cross"].apply(json.loads)
    pos = df[df["t_e_bin"] >= 0].copy()
    neg = df[df["t_e_bin"] < 0].copy()
    print(f"scoreable logs: {len(df)}   positives {len(pos)}   negatives {len(neg)}")
    neg_hours = neg["n_bins"].sum() / HZ / 3600
    print(f"negative flight time: {neg_hours:.1f} h")
    print(f"positive flight time: {pos['n_bins'].sum()/HZ/3600:.1f} h\n")

    thresholds = sorted({float(k) for c in df["cross"] for k in c})
    rows = []
    for th in thresholds:
        key = f"{th:g}"
        # ---- false alarms on negatives (survival hazard estimator) --------
        nc = neg["cross"].apply(lambda c: c.get(key, -1)).to_numpy()
        nb = neg["n_bins"].to_numpy()
        alarmed = nc >= 0
        at_risk_bins = np.where(alarmed, nc, nb)
        at_risk_h = at_risk_bins.sum() / HZ / 3600
        far = alarmed.sum() / at_risk_h if at_risk_h > 0 else np.nan

        # ---- detection and lead time on positives -------------------------
        pc = pos["cross"].apply(lambda c: c.get(key, -1)).to_numpy()
        te = pos["t_e_bin"].to_numpy()
        detected = (pc >= 0) & (pc < te)
        lead = (te[detected] - pc[detected]) / HZ
        rows.append({
            "threshold": th,
            "far_per_hour": far,
            "n_neg_alarmed": int(alarmed.sum()),
            "detection_rate": detected.mean() if len(pos) else np.nan,
            "n_detected": int(detected.sum()),
            "median_lead_s": float(np.median(lead)) if len(lead) else np.nan,
            "p25_lead_s": float(np.percentile(lead, 25)) if len(lead) else np.nan,
            "p75_lead_s": float(np.percentile(lead, 75)) if len(lead) else np.nan,
        })

    curve = pd.DataFrame(rows).sort_values("threshold")
    curve.to_csv(args.out, index=False)

    print("--- LEAD TIME vs FALSE ALARM RATE ---")
    print(f"{'thresh':>7} {'FAR/h':>9} {'detect':>8} {'n_det':>7} "
          f"{'median lead':>12} {'IQR lead':>18}")
    for _, r in curve.iterrows():
        print(f"{r.threshold:7.0f} {r.far_per_hour:9.3f} {r.detection_rate:8.3f} "
              f"{int(r.n_detected):7d} {r.median_lead_s:11.1f}s "
              f"{r.p25_lead_s:7.1f}-{r.p75_lead_s:.1f}s")

    # ---- primary operating point: interpolate to the target FAR ----------
    c = curve.dropna(subset=["far_per_hour"]).sort_values("far_per_hour")
    print(f"\n--- PRIMARY METRIC (FAR = {args.target_far:g}/hour) ---")
    if c["far_per_hour"].min() <= args.target_far <= c["far_per_hour"].max():
        lead = np.interp(args.target_far, c["far_per_hour"], c["median_lead_s"])
        det = np.interp(args.target_far, c["far_per_hour"], c["detection_rate"])
        thr = np.interp(args.target_far, c["far_per_hour"], c["threshold"])
        print(f"  threshold      : {thr:.2f}")
        print(f"  median lead    : {lead:.1f} s")
        print(f"  detection rate : {det:.3f}")
    else:
        print(f"  target FAR outside the achievable range "
              f"[{c['far_per_hour'].min():.3f}, {c['far_per_hour'].max():.3f}]")
        near = c.iloc[(c["far_per_hour"] - args.target_far).abs().argmin()]
        print(f"  closest point: threshold {near.threshold:g}, "
              f"FAR {near.far_per_hour:.3f}/h, lead {near.median_lead_s:.1f}s, "
              f"detection {near.detection_rate:.3f}")

    # ---- per-airframe breakdown at the closest operating point -----------
    near = c.iloc[(c["far_per_hour"] - args.target_far).abs().argmin()]
    key = f"{near.threshold:g}"
    print(f"\n--- PER-AIRFRAME at threshold {key} ---")
    pos = pos.assign(_c=pos["cross"].apply(lambda x: x.get(key, -1)))
    pos = pos.assign(_det=(pos["_c"] >= 0) & (pos["_c"] < pos["t_e_bin"]),
                     _lead=(pos["t_e_bin"] - pos["_c"]) / HZ)
    for m, g in pos.groupby("mav_type"):
        if len(g) >= 10:
            d = g[g["_det"]]
            print(f"  {str(m):32s} n={len(g):4d} detect={g['_det'].mean():.3f} "
                  f"median lead={np.median(d['_lead']) if len(d) else float('nan'):6.1f}s")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
