#!/usr/bin/env python
"""Materialise the A/B/C split tables once, so every downstream model scores the same rows.

TWO TABLES, and confusing them would be a silent disaster:

  meta/splits_eval.parquet      CURATED tier, 13,398 flights. This is the evaluation set.
                                Tier-1 splits exactly these rows, so a probe that joins
                                here is directly comparable to the frozen GBT 0.708 / 0.360.
  meta/splits_pretrain.parquet  BULK tier, 399,434 flights. The pretraining corpus. It
                                shares no purpose with the eval table and its split-A
                                seeds are NOT the same partition.

tier1_baseline.py globs proc/curated-*.parquet -- 7 shards, not the 201 bulk shards -- and
rebuilds its splits inline from meta/tier1_feats/roster-*.parquet. The eval table below
replays that code path operation for operation (same concat order, same drop_duplicates,
same manifest merge, same vehicle_key fillna, same GroupShuffleSplit seeds) so membership
is identical by construction rather than by coincidence.

Sources:
  meta/tier1_feats/roster-*.parquet   curated roster + t_e_bin (post v2 exclusion)
  meta/qparts_bulk/*.parquet          bulk status, t_e_bin, channel/topic presence
  meta/manifest.parquet               vehicle_key, mav_type, ver_sw_release, curated

The quality table is the authority for bulk labels -- t_e_bin is NOT in the shards (they
carry only the label grid), and primary_events does not exist in the bulk tier at all. Its
parts have three schema generations and inconsistent column ORDER; they are read by name
and deduplicated on log_id, preferring the row that actually carries a label.

Run:  OMP_NUM_THREADS=4 /home/zhuoyang/venvs/flightlog/bin/python build_splits.py
CPU only, no GPU, reads ~90 MB of metadata -- it never touches the 230 GB of shards.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "meta")
FEATS = os.path.join(META, "tier1_feats")
OUT = os.path.join(META, "splits_pretrain.parquet")
OUT_EVAL = os.path.join(META, "splits_eval.parquet")
SUMMARY = os.path.join(META, "splits_summary.json")

# Kept byte-identical to tier1_baseline.py so split B reproduces the frozen 0.360.
MULTI = {"Quadrotor", "Hexarotor", "Octorotor", "Tricopter", "Coaxial helicopter"}
SEEDS = (0, 1, 2)
TEST_FRAC = 0.2
# Split C holds out the two newest release lines; everything at or below 1.15 trains.
C_HELD_OUT = {"1.16", "1.17"}


def is_winged_asbuilt(m):
    """tier1_baseline._is_winged, reproduced exactly -- including its dead string."""
    m = str(m)
    return "VTOL" in m or m in ("Fixed Wing", "Helicopter (tail rotor)")


def is_winged_fixed(m):
    """What that function was evidently meant to say. 'Helicopter (tail rotor)' matches no
    mav_type in the corpus; the real value is 'Normal helicopter with tail rotor', so 4,350
    tail-rotor helicopters currently fall into neither arm of split B."""
    m = str(m)
    return "VTOL" in m or m in ("Fixed Wing", "Normal helicopter with tail rotor")


def major_minor(s):
    if not isinstance(s, str):
        return None
    g = re.match(r"v(\d+)\.(\d+)", s)
    return f"{g.group(1)}.{g.group(2)}" if g else None


def load_quality():
    """Union the quality-table parts. Column order differs across schema generations, so
    select by name; prefer rows carrying t_e_bin when a log_id appears more than once."""
    want = ["log_id", "status", "t_e_bin", "n_bins", "n_topics_present",
            "n_channels_present", "valid_frac"]
    frames = []
    for f in sorted(glob.glob(os.path.join(META, "qparts_bulk", "*.parquet"))):
        df = pd.read_parquet(f)
        for c in want:
            if c not in df.columns:
                df[c] = np.nan
        frames.append(df[want])
    q = pd.concat(frames, ignore_index=True)
    # 1,692 log_ids appear in both the consolidated file and a part. Sorting labelled rows
    # first makes the dedup deterministic and keeps the informative copy.
    q["_has_label"] = q.t_e_bin.notna()
    q = (q.sort_values("_has_label", ascending=False)
           .drop_duplicates("log_id")
           .drop(columns="_has_label"))
    return q


def load_lost():
    """The 938 logs of the 2026-08-08 staging deadlock: parsed successfully, marked `ok`
    in the quality table, but lost before their shard was flushed. Status `ok` therefore
    over-counts the corpus by exactly this many, and a split table built from status alone
    would hand downstream code log_ids that resolve to nothing in the shards."""
    p = os.path.join(META, "lost_unflushed_logs.csv")
    if not os.path.exists(p):
        return set()
    lost = pd.read_csv(p)
    col = "log_id" if "log_id" in lost.columns else lost.columns[0]
    return set(lost[col].astype(str))


def assign_splits(df):
    """A/B/C on an already-ordered frame carrying log_id, vehicle_key, mav_type, ver_mm."""
    for seed in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
        tr_i, te_i = next(gss.split(df, groups=df.vehicle_key))
        col = np.array(["train"] * len(df), dtype=object)
        col[te_i] = "test"
        df[f"split_A_s{seed}"] = col

    multi = df.mav_type.isin(MULTI)
    df["split_B"] = np.where(multi, "train",
                             np.where(df.mav_type.map(is_winged_asbuilt),
                                      "test", "excluded"))
    df["split_B_fixed"] = np.where(multi, "train",
                                   np.where(df.mav_type.map(is_winged_fixed),
                                            "test", "excluded"))
    df["split_C"] = np.where(df.ver_mm.isna(), "excluded",
                             np.where(df.ver_mm.isin(C_HELD_OUT), "test", "train"))
    return df


def counts(df, col):
    out = {k: int(v) for k, v in df[col].value_counts().items()}
    out["test_positives"] = int(((df[col] == "test") & df.is_positive).sum())
    return out


def build_eval():
    """The curated tier -- the set Tier-1 actually scored. Every step here mirrors
    tier1_baseline.main() so split membership is identical, not merely similar."""
    roster = pd.concat(
        [pd.read_parquet(f) for f in
         sorted(glob.glob(os.path.join(FEATS, "roster-*.parquet")))],
        ignore_index=True).drop_duplicates("log_id")
    man = pd.read_parquet(os.path.join(META, "manifest.parquet"),
                          columns=["log_id", "mav_type", "vehicle_key",
                                   "ver_sw_release", "airframe_name", "sys_hw"])
    df = roster.merge(man, on="log_id", how="left")
    df["vehicle_key"] = df.vehicle_key.fillna(df.log_id)
    df["ver_mm"] = df.ver_sw_release.map(major_minor)
    df["is_positive"] = df.t_e_bin.fillna(-1) >= 0
    df = assign_splits(df)
    df.to_parquet(OUT_EVAL, index=False)

    for s in SEEDS:
        tr = set(df.vehicle_key[df[f"split_A_s{s}"] == "train"])
        te = set(df.vehicle_key[df[f"split_A_s{s}"] == "test"])
        assert not (tr & te), f"eval split_A_s{s} leaks {len(tr & te)} vehicles"

    return df, {
        "n_flights": int(len(df)),
        "n_positive": int(df.is_positive.sum()),
        "n_vehicles": int(df.vehicle_key.nunique()),
        "split_A": {f"seed{s}": counts(df, f"split_A_s{s}") for s in SEEDS},
        "split_B": counts(df, "split_B"),
        "split_B_fixed": counts(df, "split_B_fixed"),
        "split_C": counts(df, "split_C"),
    }


def main():
    q = load_quality()
    n_all = len(q)
    q = q[q.status == "ok"].copy()
    lost = load_lost()
    n_ok = len(q)
    q = q[~q.log_id.isin(lost)].copy()
    n_lost_dropped = n_ok - len(q)

    man = pd.read_parquet(os.path.join(META, "manifest.parquet"),
                          columns=["log_id", "mav_type", "vehicle_key", "curated",
                                   "ver_sw_release", "airframe_name", "sys_hw"])
    df = q.merge(man, on="log_id", how="left")
    # Same fallback tier1 uses: a log with no vehicle_key is its own vehicle, never pooled.
    df["vehicle_key"] = df.vehicle_key.fillna(df.log_id)
    df["ver_mm"] = df.ver_sw_release.map(major_minor)
    df["is_positive"] = df.t_e_bin.fillna(-1) >= 0

    # ---- Split A: vehicle-grouped 80/20, three seeds. Grouping on vehicle_key is the
    # whole point -- a random window split leaks the same aircraft into both arms.
    for seed in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
        tr_i, te_i = next(gss.split(df, groups=df.vehicle_key))
        col = np.array(["train"] * len(df), dtype=object)
        col[te_i] = "test"
        df[f"split_A_s{seed}"] = col

    # ---- Split B: multirotor -> winged. Anything in neither set is excluded, not trained on.
    winged_ab = df.mav_type.map(is_winged_asbuilt)
    winged_fx = df.mav_type.map(is_winged_fixed)
    multi = df.mav_type.isin(MULTI)
    df["split_B"] = np.where(multi, "train", np.where(winged_ab, "test", "excluded"))
    df["split_B_fixed"] = np.where(multi, "train", np.where(winged_fx, "test", "excluded"))

    # ---- Split C: cross-firmware, forward in time. Unparseable versions are excluded
    # rather than guessed at.
    df["split_C"] = np.where(df.ver_mm.isna(), "excluded",
                             np.where(df.ver_mm.isin(C_HELD_OUT), "test", "train"))

    df.to_parquet(OUT, index=False)

    ev, ev_summary = build_eval()

    summary = {
        "eval_table": {
            "path": "meta/splits_eval.parquet",
            "tier": "curated -- THE EVALUATION SET, the rows Tier-1 scored",
            **ev_summary,
        },
        "pretrain_table_path": "meta/splits_pretrain.parquet",
        "pretrain_tier": "bulk -- the pretraining corpus, NOT comparable to Tier-1",
        "n_quality_rows": int(n_all),
        "n_status_ok": int(n_ok),
        "n_lost_unflushed_dropped": int(n_lost_dropped),
        "n_usable": int(len(df)),
        "n_positive": int(df.is_positive.sum()),
        "n_vehicles": int(df.vehicle_key.nunique()),
        "n_versions_major_minor": int(df.ver_mm.nunique()),
        "split_A": {f"seed{s}": {k: int(v) for k, v in
                                 df[f"split_A_s{s}"].value_counts().items()}
                    for s in SEEDS},
        "split_A_positives": {f"seed{s}": {k: int(v) for k, v in
                                           df[df.is_positive][f"split_A_s{s}"]
                                           .value_counts().items()} for s in SEEDS},
        "split_B": {k: int(v) for k, v in df.split_B.value_counts().items()},
        "split_B_fixed": {k: int(v) for k, v in df.split_B_fixed.value_counts().items()},
        "split_B_positives": {k: int(v) for k, v in
                              df[df.is_positive].split_B.value_counts().items()},
        "split_C": {k: int(v) for k, v in df.split_C.value_counts().items()},
        "split_C_positives": {k: int(v) for k, v in
                              df[df.is_positive].split_C.value_counts().items()},
        "split_C_held_out": sorted(C_HELD_OUT),
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {OUT}  ({len(df)} flights in shards; {n_ok} status=ok minus "
          f"{n_lost_dropped} lost-unflushed, of {n_all} quality rows)")
    print(json.dumps(summary, indent=2))

    # Vehicle-level leakage is the one failure that would invalidate every number, so
    # assert it rather than trust it.
    for s in SEEDS:
        tr = set(df.vehicle_key[df[f"split_A_s{s}"] == "train"])
        te = set(df.vehicle_key[df[f"split_A_s{s}"] == "test"])
        assert not (tr & te), f"split_A_s{s}: {len(tr & te)} vehicles in both arms"
    for col in ("split_B", "split_B_fixed", "split_C"):
        tr = set(df.log_id[df[col] == "train"])
        te = set(df.log_id[df[col] == "test"])
        assert not (tr & te), f"{col}: log_id in both arms"
    print("\nleakage assertions passed (no vehicle_key spans both arms of any split A "
          "seed; no log_id spans both arms of B/B_fixed/C)")


if __name__ == "__main__":
    main()
