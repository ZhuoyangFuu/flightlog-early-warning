#!/usr/bin/env python
"""Build the prioritized download manifest from the PX4 dbinfo snapshot.

Priority tiers (Section 4 of PROJECT_PLAN.md, revised for measured download rate):
  0  pilot        - 500 logs stratified by mav_type, for the go/no-go gate
  1  positive     - metadata suggests a failure event (errors / error_labels / short+errors)
  2  negative     - clean long flights, matched to the positive pool
  3  bulk         - everything else, for masked-reconstruction pretraining

Tiers 0-2 are the CURATED set: raw .ulg is retained permanently.
Tier 3 is stream-and-discard: raw is deleted once tensorized.
"""
import argparse
import gzip
import json
import os
import random
from collections import Counter, defaultdict

import pandas as pd

# Durations above this are corrupt (observed: up to 1.8e13 s).
MAX_SANE_DURATION_S = 86400
MIN_FLIGHT_S = 30          # below this it is a bench/ground test
MIN_NEGATIVE_S = 120       # Section 5 negative-sampling floor


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        return json.loads(f.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbinfo", default="meta/dbinfo_20260807.json.gz")
    ap.add_argument("--out", default="meta/manifest.parquet")
    ap.add_argument("--pilot-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--curated-gb", type=float, default=250.0,
                    help="disk budget for retained raw .ulg files")
    args = ap.parse_args()

    rows = load(args.dbinfo)
    print(f"loaded {len(rows)} rows")
    df = pd.DataFrame(rows)

    # ---- sanitize -------------------------------------------------------
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce").fillna(0)
    bad_dur = (df["duration_s"] > MAX_SANE_DURATION_S) | (df["duration_s"] < 0)
    print(f"corrupt duration_s: {int(bad_dur.sum())} rows -> set to 0 (kept, flagged)")
    df.loc[bad_dur, "duration_s"] = 0
    df["duration_corrupt"] = bad_dur

    df["num_logged_errors"] = pd.to_numeric(df["num_logged_errors"], errors="coerce").fillna(0).astype(int)
    df["num_logged_warnings"] = pd.to_numeric(df["num_logged_warnings"], errors="coerce").fillna(0).astype(int)
    df["n_error_labels"] = df["error_labels"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # de-duplicate on log_id (defensive; ids should be unique)
    n0 = len(df)
    df = df.drop_duplicates(subset="log_id").reset_index(drop=True)
    if len(df) != n0:
        print(f"dropped {n0-len(df)} duplicate log_id rows")

    # ---- vehicle grouping key for leak-free splits ----------------------
    # vehicle_uuid is the strongest identity; fall back to sys_hw+airframe.
    df["vehicle_key"] = df["vehicle_uuid"].fillna("").replace("", pd.NA)
    df["vehicle_key"] = df["vehicle_key"].fillna(
        "hw:" + df["sys_hw"].fillna("?") + "|af:" + df["airframe_name"].fillna("?")
    )

    # ---- tiering --------------------------------------------------------
    df["tier"] = 3
    df["tier_reason"] = "bulk"

    flyable = df["duration_s"] >= MIN_FLIGHT_S

    is_pos = flyable & (
        (df["num_logged_errors"] > 0)
        | (df["n_error_labels"] > 0)
        # short flight that logged errors: classic crash-ends-log signature
        | ((df["duration_s"] < 120) & (df["num_logged_errors"] > 0))
    )
    df.loc[is_pos, ["tier", "tier_reason"]] = [1, "positive_candidate"]

    is_neg = (
        (df["duration_s"] >= MIN_NEGATIVE_S)
        & (df["num_logged_errors"] == 0)
        & (df["n_error_labels"] == 0)
        & (~df["duration_corrupt"])
    )
    df.loc[is_neg & (df["tier"] == 3), ["tier", "tier_reason"]] = [2, "negative_candidate"]

    # ---- pilot: stratified by mav_type, drawn from flyable logs ---------
    rng = random.Random(args.seed)
    pilot_ids = []
    pool = df[flyable]
    by_type = defaultdict(list)
    for lid, mt in zip(pool["log_id"], pool["mav_type"]):
        by_type[mt].append(lid)
    types = sorted(by_type)
    per = max(1, args.pilot_n // len(types))
    for t in types:
        ids = by_type[t]
        rng.shuffle(ids)
        pilot_ids.extend(ids[:per])
    rng.shuffle(pilot_ids)
    pilot_ids = set(pilot_ids[: args.pilot_n])
    df.loc[df["log_id"].isin(pilot_ids), ["tier", "tier_reason"]] = [0, "pilot"]

    # ---- ordering: tier, then most-informative first --------------------
    df["order_key"] = (
        df["tier"] * 1_000_000
        - df["num_logged_errors"].clip(upper=50) * 1000
        - df["n_error_labels"].clip(upper=10) * 100
    )
    df = df.sort_values(["order_key", "log_id"]).reset_index(drop=True)
    df["rank"] = range(len(df))

    # ---- curated raw-retention set, capped by disk budget ---------------
    # Every log is downloaded and tensorized; only `curated` keeps its .ulg.
    # Retention is insurance + human inspection, not a necessity: re-fetching
    # the curated set from the CDN costs ~80 min at the measured 288 logs/min.
    MEAN_MB = 18.29
    budget_n = int(args.curated_gb * 1000 / MEAN_MB)
    df["curated"] = False
    df.loc[df["tier"] == 0, "curated"] = True          # pilot always retained
    remaining = budget_n - int(df["curated"].sum())

    # Split the remainder between positives and matched negatives, and
    # stratify by mav_type so the cross-airframe claim survives the cap.
    for tier, share in ((1, 0.6), (2, 0.4)):
        quota = int(remaining * share)
        sub = df[(df["tier"] == tier)]
        by_t = defaultdict(list)
        for idx, mt in zip(sub.index, sub["mav_type"]):
            by_t[str(mt)].append(idx)
        # round-robin across airframe types, richest-signal first within type
        for t in by_t:
            by_t[t].sort(key=lambda i: df.at[i, "order_key"])
        picked, types = [], sorted(by_t)
        while len(picked) < quota and any(by_t[t] for t in types):
            for t in types:
                if by_t[t] and len(picked) < quota:
                    picked.append(by_t[t].pop(0))
        df.loc[picked, "curated"] = True

    keep = [
        "rank", "log_id", "download_url", "tier", "tier_reason", "curated",
        "log_date", "duration_s", "duration_corrupt", "mav_type", "airframe_type",
        "airframe_name", "sys_hw", "ver_sw_release", "vehicle_key", "vehicle_uuid",
        "num_logged_errors", "num_logged_warnings", "n_error_labels", "estimator",
    ]
    out = df[keep].copy()
    # dbinfo mixes int and str in these columns (e.g. airframe_type "4011" vs 4011)
    for c in ["mav_type", "airframe_type", "airframe_name", "sys_hw",
              "ver_sw_release", "vehicle_key", "vehicle_uuid", "estimator",
              "log_date", "tier_reason", "log_id", "download_url"]:
        out[c] = out[c].astype(str)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)

    print("\n--- TIERS ---")
    for t in sorted(out["tier"].unique()):
        sub = out[out["tier"] == t]
        print(f"  tier {t} ({sub['tier_reason'].iloc[0]:19s}): {len(sub):7d} logs  "
              f"est {len(sub)*18.29/1e3:7.1f} GB raw")
    print(f"  TOTAL: {len(out)} logs, est {len(out)*18.29/1e3:.1f} GB raw")
    nc = int(out["curated"].sum())
    print(f"  curated (raw retained): {nc} logs, est {nc*18.29/1e3:.1f} GB "
          f"(budget {args.curated_gb:.0f} GB)")
    print(f"  stream-and-discard: {len(out)-nc} logs")

    print("\n--- CURATED SET COMPOSITION (mav_type) ---")
    cur = out[out["curated"]]
    for k, v in Counter(cur["mav_type"]).most_common(12):
        print(f"  {str(k):34s} {v:6d}")
    print(f"\n  distinct vehicle_key in curated: {cur['vehicle_key'].nunique()}")
    print(f"  distinct ver_sw_release in curated: {cur['ver_sw_release'].nunique()}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
