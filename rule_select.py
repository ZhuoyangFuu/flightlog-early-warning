#!/usr/bin/env python
"""Item 3b(iii): choose the sequential alarm rule on HELD-OUT data, not the test table.

The problem this closes
-----------------------
`sequential.py` sweeps 11 alarm rules and reports all of them. RESULTS.md 4.6 caveat 2
is honest about what that leaves open: `ewma` 15 and `cusum` 0.5 were *named* in the
write-up after looking at a table computed on the test split. Reporting the whole grid
(as `paper_tables.rule_counts` does) is defensible; naming a winner off that table is not.

Selecting on a validation split is the fix, and no split has one -- `splits_eval.parquet`
carries train/test only, on all three. Refitting every arm against a smaller train set to
create one would cost days and would change every number in the paper.

So the validation arm is carved out of the TEST flights instead, by vehicle
-------------------------------------------------------------------------
Test flights are partitioned into two halves, **grouped by `vehicle_key`** so no vehicle
appears in both. Neither half was ever seen in training, so both are honest held-out
data; the split is only about which half is allowed to influence the choice of rule.

  SELECT half  the rule is chosen here, for every arm AND for Tier-0
  REPORT half  the chosen rule is scored here, and this is the number to quote

The partition is deterministic (`blake2b(vehicle_key, salt)`) and greedily balanced on
positive-flight count, so the two halves carry comparable event counts. It does not
depend on any score, so it cannot be tuned.

The selection criterion is the project's OWN pre-registered one
--------------------------------------------------------------
`ews_sweep.py` froze Tier-0 with "max detection s.t. FAR <= cap, tie-break longer median
lead", written before that sweep ran. The same criterion is reused here rather than
inventing one, which is the difference between a pre-registered rule and a convenient one:

    argmax det@1.6/h on the SELECT half, ties broken by longer lead@1.6/h

Seeds are averaged BEFORE selecting, so a rule is chosen for a configuration rather than
for a lucky seed -- the failure mode that fired four times on this project.

Fairness
--------
**Tier-0 gets the same privilege.** Its rule is selected on the SELECT half by the same
criterion and scored on the REPORT half, so the comparison is rule-matched and
selection-matched. Comparing a rule-selected learned arm against a fixed baseline would
be the 2026-08-24 protocol error again, in a new costume.

Usage
-----
    python rule_select.py                 # all splits, all arms
    python rule_select.py --split B_fixed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sequential as seq                                            # noqa: E402
from eval_common import roster_for                                  # noqa: E402
from tier1_baseline import REF_FAR, threshold_grid, flight_curve, op_point  # noqa: E402

META = Path(HERE) / "meta"
OUT = META / "paper"
SPLITS_PATH = META / "splits_eval.parquet"
SALT = b"fl-rule-sel-v1"                    # fixed (blake2b caps salt at 16 B)
SPLITS = ["A", "B_fixed", "C"]


# --------------------------------------------------------------------- partition
def halve_by_vehicle(splits: pd.DataFrame, test_ids: set) -> dict[str, set]:
    """Partition test flights into SELECT / REPORT halves, grouped by vehicle.

    Deterministic and score-independent: vehicles are ordered by a fixed hash, then
    dealt greedily to whichever half currently holds fewer positive flights. Greedy
    balancing matters because vehicles are very unequal -- a few contribute dozens of
    events and most contribute none.
    """
    t = splits[splits["log_id"].isin(test_ids)]
    veh = (t.groupby("vehicle_key")
           .agg(n=("log_id", "size"), pos=("is_positive", "sum"))
           .reset_index())
    veh["h"] = veh["vehicle_key"].map(
        lambda v: int.from_bytes(hashlib.blake2b(str(v).encode(), salt=SALT,
                                                 digest_size=8).digest(), "big"))
    veh = veh.sort_values(["pos", "h"], ascending=[False, True])
    load = {"select": [0, 0], "report": [0, 0]}          # [positives, flights]
    assign = {}
    for _, r in veh.iterrows():
        side = min(load, key=lambda k: (load[k][0], load[k][1]))
        assign[r["vehicle_key"]] = side
        load[side][0] += int(r["pos"]); load[side][1] += int(r["n"])
    out = {}
    for side in ("select", "report"):
        vs = {v for v, s in assign.items() if s == side}
        out[side] = set(t.loc[t["vehicle_key"].isin(vs), "log_id"])
    return out


# --------------------------------------------------------------------- scoring
def score_dump(path: Path, rosters: dict[str, pd.DataFrame], dt: float) -> list[dict]:
    """Every rule in the grid, scored separately on each roster half.

    This mirrors `sequential.score_arm` line for line -- same negative-window mu/sd,
    same non-finite fill, same `raw`-on-untouched-array rule, same threshold grid
    built from the transformed NEGATIVES -- and changes exactly one thing: the
    rosters are passed in rather than derived. The threshold grid is built once per
    rule from the whole dump's negatives and reused for both halves, so a half's
    numbers differ only by which flights are scored, never by where the thresholds
    sit. That is the same invariant `score_arm` maintains across splits.
    """
    df = pd.read_parquet(path).sort_values(["log_id", "t_bin"]).reset_index(drop=True)
    scores = df.score.to_numpy(dtype=np.float64)
    lid = df.log_id.to_numpy()
    tb = df.t_bin.to_numpy()
    isneg = (df.y == 0).to_numpy()
    mu = float(np.nanmean(scores[isneg]))
    sd = float(np.nanstd(scores[isneg]))
    starts = np.concatenate(([0], np.flatnonzero(lid[1:] != lid[:-1]) + 1, [len(lid)]))
    filled = np.where(np.isfinite(scores), scores, mu)

    rows = []
    for rule, param, spec in seq.rules_for(dt):
        v = seq.transform(rule, spec, mu, sd, starts,
                          scores if rule == "raw" else filled)
        neg = v[isneg & np.isfinite(v)]
        if neg.size < 4 or not np.isfinite(v).any():
            continue
        sbf = {lid[a]: (tb[a:b], v[a:b]) for a, b in zip(starts[:-1], starts[1:])}
        grid = threshold_grid(neg)
        rec = {"arm": path.stem, "rule": rule, "param": str(param)}
        for half, roster in rosters.items():
            pt = op_point(flight_curve(sbf, roster, grid))
            rec[f"det_{half}"] = pt.get(f"det_at_{REF_FAR}h", np.nan)
            rec[f"lead_{half}"] = pt.get(f"lead_at_{REF_FAR}h", np.nan)
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, action="append")
    ap.add_argument("--out", default=str(OUT / "rule_select.csv"))
    args = ap.parse_args()
    want = args.split or SPLITS

    OUT.mkdir(parents=True, exist_ok=True)
    splits = pd.read_parquet(SPLITS_PATH)
    dumps = sorted((META / "scores").glob("*.parquet"))
    print(f"{len(dumps)} score dumps, splits {want}")

    all_rows = []
    for sp in want:
        col = seq.SPLIT_COL[sp].format(seed=0)
        test_ids = set(splits.loc[splits[col] == "test", "log_id"])
        halves = halve_by_vehicle(splits, test_ids)
        rosters = {h: roster_for(splits, ids) for h, ids in halves.items()}
        npos = {h: int(splits.loc[splits["log_id"].isin(ids), "is_positive"].sum())
                for h, ids in halves.items()}
        print(f"\n=== {sp}: {len(test_ids)} test flights -> "
              f"select {len(halves['select'])} ({npos['select']} pos) / "
              f"report {len(halves['report'])} ({npos['report']} pos)")

        for d in dumps:
            name = d.stem
            is_t0 = name.startswith("ews_scores_")
            # a learned arm belongs to exactly one split; Tier-0 is scored on all
            if not is_t0:
                tag, art = seq._artifact_tag(name)
                p = META / f"{art}_pt_{tag}.json"
                if not p.exists() or json.loads(p.read_text())["split"] != sp:
                    continue
            dt = seq.DT_TIER0 if is_t0 else seq.DT_LEARNED
            rows = score_dump(d, rosters, dt)
            for r in rows:
                r["split"] = sp
            all_rows += rows
            print(f"  {name:44s} {len(rows)} rules", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
