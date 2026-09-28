#!/usr/bin/env python
"""Second half of item 3b(iii): apply the pre-declared selection rule, report on the
half that did not choose it.

`rule_select.py` scored every (arm, rule) on both halves. This file does the choosing,
and it is kept separate so the criterion is visible in one place and cannot be quietly
tuned after seeing a table.

THE CRITERION, declared once
----------------------------
    argmax det@1.6/h on the SELECT half; ties broken by longer lead@1.6/h

which is `ews_sweep.py`'s frozen Tier-0 criterion ("max detection s.t. FAR <= cap,
tie-break longer median lead") reused verbatim. Seeds are averaged on the SELECT half
BEFORE the argmax, so a rule is chosen for a *configuration*, never for a lucky seed.

Tier-0 is selected the same way on the same half and reported on the same half, so the
comparison is selection-matched as well as rule-matched.

WHAT TO QUOTE
-------------
The REPORT-half numbers. The SELECT-half numbers are printed only so the optimism of
selection is visible -- the gap between them is the thing this whole exercise exists to
expose.

Usage:  python rule_select_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

META = Path(__file__).resolve().parent / "meta"
OUT = META / "paper"
SPLITS = ["A", "B_fixed", "C"]
BANDS = {"widen10": "1-10 s", "shift120": "111-120 s", "shift150": "141-150 s",
         "shift240": "231-240 s", "shift300": "291-300 s", "shift360": "351-360 s"}


def parse_arm(a: str) -> dict:
    """(features, head, band, seed) from a dump name; split comes from the CSV."""
    m = re.search(r"(widen\d+|shift\d+)", a)
    s = re.search(r"_seed(\d+)", a)
    return {"features": "stats" if a.startswith("h1_") else "emb",
            "head": "gbt" if "_gbt" in a else "lr",
            "band": m.group(1) if m else "?",
            "seed": int(s.group(1)) if s else 0}


def main():
    src = OUT / "rule_select.csv"
    df = pd.read_csv(src)
    df = df[np.isfinite(df["det_select"]) & np.isfinite(df["det_report"])].copy()
    meta = pd.DataFrame([parse_arm(a) for a in df["arm"]], index=df.index)
    df = pd.concat([df, meta], axis=1)
    df["is_t0"] = df["arm"].str.startswith("ews_scores_")

    # ---- average the seeds on BOTH halves, per (config, rule) ----
    keys = ["split", "features", "head", "band", "rule", "param", "is_t0"]
    cfg = (df.groupby(keys, dropna=False)
           .agg(n=("det_select", "size"),
                det_sel=("det_select", "mean"), lead_sel=("lead_select", "mean"),
                det_rep=("det_report", "mean"), lead_rep=("lead_report", "mean"))
           .reset_index())

    # ---- the choice, made on the SELECT half only ----
    def choose(g):
        return g.sort_values(["det_sel", "lead_sel"], ascending=False).iloc[0]

    chosen = (cfg.groupby(["split", "features", "head", "band", "is_t0"], dropna=False)
              .apply(choose, include_groups=False).reset_index())

    rows = []
    for sp in SPLITS:
        t0 = chosen[(chosen["split"] == sp) & chosen["is_t0"]]
        if t0.empty:
            continue
        b = t0.iloc[0]
        for _, r in chosen[(chosen["split"] == sp) & ~chosen["is_t0"]].iterrows():
            if not r["band"].startswith("shift") or r["n"] < 3:
                continue
            rows.append({
                "split": sp, "features": r["features"], "head": r["head"],
                "band": BANDS.get(r["band"], r["band"]), "n": int(r["n"]),
                "rule": f"{r['rule']} {r['param']}" if str(r["param"]) != "None" else "raw",
                "det_sel": r["det_sel"], "det_rep": r["det_rep"],
                "lead_sel": r["lead_sel"], "lead_rep": r["lead_rep"],
                "t0_rule": f"{b['rule']} {b['param']}" if str(b["param"]) != "None" else "raw",
                "t0_det_rep": b["det_rep"], "t0_lead_rep": b["lead_rep"],
                "dominates_report": (r["det_rep"] > b["det_rep"]
                                     and r["lead_rep"] > b["lead_rep"]),
            })
    out = pd.DataFrame(rows)
    out["det_optimism"] = out["det_sel"] - out["det_rep"]
    out.to_csv(OUT / "rule_select_report.csv", index=False)

    print("Selection criterion: argmax det@1.6/h on the SELECT half, tie-break longer "
          "lead. Seeds averaged before choosing. Tier-0 selected the same way.\n")
    cols = ["split", "band", "features", "head", "rule", "det_sel", "det_rep",
            "lead_sel", "lead_rep", "t0_det_rep", "t0_lead_rep", "dominates_report"]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nTier-0's own selected rule, per split (this is the bar on the report half):")
    for sp in SPLITS:
        b = chosen[(chosen["split"] == sp) & chosen["is_t0"]]
        if not b.empty:
            r = b.iloc[0]
            rule = f"{r['rule']} {r['param']}" if str(r["param"]) != "None" else "raw"
            print(f"  {sp:8s} {rule:16s} select {r['det_sel']:.3f}/{r['lead_sel']:6.1f}s"
                  f"   report {r['det_rep']:.3f}/{r['lead_rep']:6.1f}s")

    print("\nDominance on the REPORT half, by family (this is the quotable count):")
    g = (out.groupby(["split", "features", "head"])["dominates_report"]
         .agg(["sum", "size"]).reset_index())
    print(g.to_string(index=False))

    print(f"\nSelection optimism (det_select - det_report), mean "
          f"{out['det_optimism'].mean():+.4f}, max {out['det_optimism'].max():+.4f}")
    print(f"\nwrote {OUT / 'rule_select_report.csv'}")


if __name__ == "__main__":
    main()
