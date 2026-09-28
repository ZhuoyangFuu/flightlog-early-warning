#!/usr/bin/env python
"""Aggregate every scored arm into ONE tidy table, so the figures and the paper
document read from a single source instead of from prose.

Nothing is refitted and nothing is recomputed here: this walks the cached result
JSONs written by `probe.py` / `horizon.py` (Tier-2, pretrained embeddings) and
`horizon_tier1.py` (Tier-1, hand-built window statistics), plus `ews_split.py`'s
per-split Tier-0 reference and `sequential.py`'s rule sweep. If a number here
disagrees with RESULTS.md, RESULTS.md is the one to trust and this script has a bug.

Emits
  meta/paper/arms.csv        one row per (features, head, band, split, seed)
  meta/paper/summary.csv     the same collapsed to n=3 mean +/- sd, with a
                             `dominates` flag against that split's Tier-0 per-axis best over the channel × rule cross (176 configs since 2026-09-11)
  meta/paper/tier0.csv       Tier-0 per split: frozen point and per-axis best over the channel × rule cross
  meta/paper/rules.csv       per (arm-family, split) rule-sweep dominance counts

Definitions, fixed here so every figure and table agrees:
  det   = det_at_1.6h    detection rate interpolated at exactly FAR 1.6/h
  lead  = lead_at_1.6h   median lead in seconds at that same operating point
  bar   = Tier-0's BEST configuration over the 11-rule grid on that split, on
          BOTH axes independently -- the strict bar. Dominance means beating both.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/wsssd/FlightLog")
META = ROOT / "meta"
OUT = META / "paper"

# Band label -> (short name, nominal seconds before the event the band starts).
# `widen` 10 s is the ORIGINAL band (1-10 s); the `shift` bands move it back.
BANDS = {"widen10": ("1-10 s", 10), "shift120": ("111-120 s", 120),
         "shift150": ("141-150 s", 150), "shift240": ("231-240 s", 240),
         "shift300": ("291-300 s", 300), "shift360": ("351-360 s", 360)}

SPLITS = ["A", "B_fixed", "C"]


def _seed_of(stem: str) -> int:
    m = re.search(r"_seed(\d+)", stem)
    return int(m.group(1)) if m else 0


def load_arms() -> pd.DataFrame:
    """Every horizon_pt_* (embeddings) and h1_pt_* (statistics) result JSON."""
    rows = []
    for p in sorted(META.glob("h1_pt_*.json")) + sorted(META.glob("horizon_pt_*.json")):
        d = json.loads(p.read_text())
        if "det_at_1.6h" not in d:
            continue
        band = f"{d['mode']}{int(d['h_s'])}" if d["mode"] != "widen" else f"widen{int(d['h_s'])}"
        # arms scored under a FLIGHTLOG_FEATURE_TAG carry the tag as a stem suffix
        tag = re.search(r"_(" + "|".join(FEATURE_TAGS) + r")$", p.stem)
        rows.append({
            "artifact": p.stem,
            "features": ("stats" if d.get("features") == "tier1_stats" or p.stem.startswith("h1_")
                         else (tag.group(1) if tag else "emb")),
            "head": d.get("head", "lr"),
            "band": band,
            # band WIDTH is a separate axis (the 4.4 width experiment ran
            # `shift` 120 s at 30 s wide); without it that arm collides with the
            # 9 s one and looks like a duplicate seed.
            "width_s": d.get("width_s"),
            "split": d["split"],
            "seed": d.get("seed", _seed_of(p.stem)),
            "det": d["det_at_1.6h"],
            # gbt arms with too few alarms at the operating point store no lead
            "lead": d.get("lead_at_1.6h", float("nan")),
            "auprc": d["window_auprc"],
            "n_train_pos": d.get("n_train_pos"),
            "n_test_flights": d.get("n_test_flights"),
        })
    df = pd.DataFrame(rows)
    # one arm was dumped twice under different stems in the same (config, seed);
    # keep the first and report, rather than silently averaging a duplicate in
    key = ["features", "head", "band", "width_s", "split", "seed"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        print(f"  ! {dup.sum()} duplicate (config, seed) rows:")
        for _, g in df[dup].groupby(key):
            print("   ", g["artifact"].tolist(), g["det"].round(4).tolist())
        df = df.drop_duplicates(key, keep="first")
    return df


def load_tier0() -> pd.DataFrame:
    """Tier-0 per split: the frozen `raw` point, and its BEST over the 11-rule grid.

    The strict bar takes the best detection and the best lead over the grid
    independently, which is the hardest honest reading of "the baseline, operated
    as well as it can be".
    """
    by = json.loads((META / "ews_bysplit.json").read_text())["by_split"]
    seq = pd.read_csv(META / "sequential.csv")
    # 15 real channel-set dumps (14 `chan-*` + the frozen 87-ch `w10k3`); the 200-log
    # `smoke_dump` pilot is not a configuration and must not be counted in the cross.
    t0 = seq[seq["arm"].str.startswith("ews_scores") & (seq["arm"] != "ews_scores_smoke_dump")]
    rows = []
    for sp in SPLITS:
        b = by.get(sp, {})
        g = t0[t0["eval_split"] == sp]
        rows.append({
            "split": sp,
            "n_test_flights": b.get("n_test_flights"),
            "n_test_pos": b.get("n_test_pos"),
            "raw_det": b.get("det_at_1.6h"),
            "raw_lead": b.get("lead_at_1.6h", b.get("median_lead_s")),
            "best_det": g["det@1.6h"].max() if len(g) else np.nan,
            "best_lead": g["lead@1.6h"].max() if len(g) else np.nan,
            "n_rules": len(g),
        })
    return pd.DataFrame(rows)


def summarise(arms: pd.DataFrame, tier0: pd.DataFrame) -> pd.DataFrame:
    g = (arms[arms["width_s"].isin([None, 9.0]) | arms["width_s"].isna()]
         .groupby(["features", "head", "band", "split"])
         .agg(n=("seed", "size"),
              det=("det", "mean"), det_sd=("det", "std"),
              lead=("lead", "mean"), lead_sd=("lead", "std"),
              auprc=("auprc", "mean"))
         .reset_index())
    bar = tier0.set_index("split")[["best_det", "best_lead"]]
    g = g.join(bar, on="split")
    # Dominance is on the MEAN, and only claimed at n = 3 -- a single-seed
    # HistGBT number is not a bar (RESULTS.md 4.8 finding 4).
    g["dominates"] = (g["n"] >= 3) & (g["det"] > g["best_det"]) & (g["lead"] > g["best_lead"])
    g["band_s"] = g["band"].map(lambda b: BANDS.get(b, (b, np.nan))[1])
    g["band_label"] = g["band"].map(lambda b: BANDS.get(b, (b, np.nan))[0])
    return g.sort_values(["split", "band_s", "features", "head"])


FEATURE_TAGS = ("tsfm", "rel", "relcat", "vrole", "vdev",
                "role", "dev", "vmean",          # item 6 views, original encoder
                "chrole", "chdev", "chmean",     # item 8 A: channel-mask continued pretraining
                "rdrole", "rddev", "rdmean")     # item 8 B: random-mask continued (control)


def rule_counts() -> pd.DataFrame:
    """How many (arm x rule) configurations clear the strict bar, per family.

    This is a ROBUSTNESS count over the whole rule grid, not a best-of-N
    selection -- the rule was never chosen on held-out data, so a selected best
    is not reportable (RESULTS.md 4.6 caveat 2).
    """
    seq = pd.read_csv(META / "sequential.csv")
    seq = seq[~seq["arm"].str.startswith("ews_scores")].copy()
    t0 = load_tier0().set_index("split")
    # `eval_split` is only populated for the Tier-0 rows (one dump, four splits);
    # a learned arm carries its split in the artifact name instead.
    fam = []
    for a in seq["arm"]:
        # 2026-09-11: arms scored under a FLIGHTLOG_FEATURE_TAG carry a trailing
        # feature suffix (`_tsfm`, `_rel`, `_relcat`, ...). Strip it for the split
        # parse and record it as the feature family, so the TSFM and the derived
        # embedding arms count in their own rows rather than as `emb`.
        # Known FLIGHTLOG_FEATURE_TAG values only: `B_fixed` itself ends in
        # `_fixed`, so a generic "strip the last token" parse mislabels seed 0.
        tag = re.search(r"_(" + "|".join(FEATURE_TAGS) + r")$", a)
        ftag = tag.group(1) if tag else None
        core = a[: -len("_" + ftag)] if ftag else a
        feats = "stats" if a.startswith("h1_") else (ftag or "emb")
        head = "gbt" if "_gbt" in core else "lr"
        m = re.search(r"(widen\d+|shift\d+)", core)
        sp = next((s for s in ("B_fixed", "A", "C", "B")
                   if re.search(rf"_{s}(_seed\d+)?(_gbt)?$", core)), None)
        fam.append((feats, head, m.group(1) if m else "?", sp))
    seq[["features", "head", "band", "split"]] = pd.DataFrame(fam, index=seq.index)
    assert seq["split"].notna().all(), seq.loc[seq["split"].isna(), "arm"].unique()
    seq["bar_det"] = seq["split"].map(t0["best_det"])
    seq["bar_lead"] = seq["split"].map(t0["best_lead"])
    # Collapse seeds FIRST: a (band, rule) configuration dominates if its n = 3
    # MEAN clears the bar on both axes. Counting per-seed rows instead would let
    # a lucky seed carry a configuration, which is the failure mode this project
    # hit four times (RESULTS.md 4.8 finding 3).
    sub = seq[seq["band"].str.startswith("shift") & seq["split"].isin(SPLITS)]
    cfg = (sub.groupby(["split", "features", "head", "band", "rule", "param"], dropna=False)
           .agg(n=("det@1.6h", "size"), det=("det@1.6h", "mean"),
                lead=("lead@1.6h", "mean"),
                bar_det=("bar_det", "first"), bar_lead=("bar_lead", "first"))
           .reset_index())
    cfg["dom"] = (cfg["n"] >= 3) & (cfg["det"] > cfg["bar_det"]) & (cfg["lead"] > cfg["bar_lead"])
    out = (cfg.groupby(["split", "features", "head"])
           .agg(n_configs=("dom", "size"), n_dominating=("dom", "sum"))
           .reset_index())
    out["frac"] = out["n_dominating"] / out["n_configs"]
    cfg.to_csv(OUT / "rules_per_config.csv", index=False)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = load_arms()
    tier0 = load_tier0()
    summ = summarise(arms, tier0)
    rules = rule_counts()

    arms.to_csv(OUT / "arms.csv", index=False)
    summ.to_csv(OUT / "summary.csv", index=False)
    tier0.to_csv(OUT / "tier0.csv", index=False)
    rules.to_csv(OUT / "rules.csv", index=False)

    print(f"arms      {len(arms):4d} rows  ({arms['artifact'].nunique()} artifacts)")
    print(f"summary   {len(summ):4d} configs, {(summ['n'] >= 3).sum()} at n>=3")
    print(f"\nTier-0 per split (raw point, and best over {tier0['n_rules'].max()} rules):")
    print(tier0.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nn=3 configs, by split:")
    cols = ["split", "band_label", "features", "head", "n", "det", "det_sd",
            "lead", "lead_sd", "auprc", "dominates"]
    print(summ[summ["n"] >= 3][cols].to_string(index=False,
                                               float_format=lambda v: f"{v:.4f}"))
    print("\nrule-grid dominance counts (shifted bands only):")
    print(rules.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nwrote {OUT}/{{arms,summary,tier0,rules}}.csv")


if __name__ == "__main__":
    main()
