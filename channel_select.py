#!/usr/bin/env python
"""Choose Tier-0's CHANNEL SET on held-out data — is the +0.055 real, or optimism?

The problem this closes
-----------------------
`ews_channels.py` (2026-09-02) found that restricting Tier-0 to the 8 commanded-rotor
channels at top-1 beats the frozen 87-channel top-3 configuration by **+0.055 detection**
at a matched FAR of 1.6/h. That number is **best-of-14, chosen by looking at the table it
is reported in**, while the arm it beats was chosen by a *pre-registered* sweep. The two
are not comparable, and this project has already measured what the difference is worth:
`rule_select.py` / RESULTS.md §6.1 found selection optimism of **+0.009 on average and
+0.171 at worst** when a sequential rule was picked off the test table. +0.055 sits inside
that range, so the finding is *unfalsified*, not *established*.

This file applies §6.1's protocol to the channel set instead of the alarm rule.

The design is §6.1's, reused rather than reinvented
---------------------------------------------------
`halve_by_vehicle` is imported from `rule_select.py` — the SAME deterministic,
score-independent, positives-balanced `blake2b(vehicle_key)` partition, with the same
salt. `op_point` and `REF_FAR` come from `tier1_baseline`, and the curve is built by
`ews_split.curve_for`, so the metric is the same object the per-split Tier-0 bars in
RESULTS.md §2.1 are computed with. Nothing here defines a number of its own.

    SELECT half   the channel set is chosen here
    REPORT half   the chosen set is scored here, and this is the number to quote

The selection criterion is the project's own pre-registered one, third reuse:
**argmax det@1.6/h on the SELECT half, ties broken by longer lead@1.6/h.**

Fairness
--------
**The frozen 87-channel arm is in the candidate pool**, so "all channels, top-3" can win
the selection on its own merits. It is also scored on the REPORT half regardless, because
the quantity that matters is not the winner's score but

    selected-arm REPORT det  −  frozen-arm REPORT det

which is what a pre-registered channel sweep would actually have bought. Reporting the
winner alone would smuggle the selection back in.

No rescoring happens. Every arm's per-flight first-crossing bins are already cached in
`meta/ews_traces_*.parquet`; restricting to a flight subset is pure re-aggregation.

Cost: **seconds**, CPU-only. Measured 0.2 s per curve, 120 curves. No launcher, no
detachment — it re-aggregates cached crossings and fits nothing.

Usage
-----
    python channel_select.py              # -> meta/paper/channel_select.csv
    python channel_select.py --status     # re-print the verdict from that CSV
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ews_channels as EC                                     # noqa: E402
from ews_split import load_traces, curve_for, SPLIT_COL       # noqa: E402
from rule_select import halve_by_vehicle                      # noqa: E402
from tier1_baseline import REF_FAR, op_point                  # noqa: E402

META = Path(HERE) / "meta"
OUT = META / "paper"
CONTROL = EC.CONTROL_TAG            # w10k3ext = the frozen config, extended grid

# Populations the selection is run on. "corpus" is the whole curated tier -- the
# population RESULTS.md §2's headline 0.186 is measured over; the three splits are
# the populations the learned tiers are actually compared against (§2.1).
POPULATIONS = ["corpus", "A", "B_fixed", "C"]


def arms():
    """(label, tag) for every candidate, frozen arm FIRST so ties favour it."""
    out = [("all-87ch-k3 (FROZEN)", CONTROL)]
    out += [(f"{g}-{EC.n_channels(g)}ch-k{k}", EC.tag_of(g, k)) for g, k in EC.GRID]
    return out


def population_ids(splits, pop):
    if pop == "corpus":
        return set(splits["log_id"])
    return set(splits.loc[splits[SPLIT_COL[pop]] == "test", "log_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--out", default=str(OUT / "channel_select.csv"))
    args = ap.parse_args()

    if args.status:
        p = Path(args.out)
        if not p.exists():
            print("not run yet")
            return
        report(pd.read_csv(p))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    splits = pd.read_parquet(META / "splits_eval.parquet")
    if "is_positive" not in splits.columns:
        splits = splits.assign(is_positive=(splits.t_e_bin >= 0).astype(int))

    # Load every arm's cached crossings once.
    cross, thresholds = {}, {}
    for label, tag in arms():
        tr = load_traces(tag)
        cross[tag] = dict(zip(tr.log_id, tr["cross"]))
        thresholds[tag] = sorted({t for d in tr["cross"].head(1) for t in d})
        print(f"{label:28s} {tag:22s} {len(cross[tag]):,} traces, "
              f"{len(thresholds[tag])} thresholds", flush=True)

    rows = []
    for pop in POPULATIONS:
        ids = population_ids(splits, pop)
        halves = halve_by_vehicle(splits, ids)
        npos = {h: int(splits.loc[splits.log_id.isin(v), "is_positive"].sum())
                for h, v in halves.items()}
        print(f"\n=== {pop}: {len(ids):,} flights -> "
              f"select {len(halves['select']):,} ({npos['select']} pos) / "
              f"report {len(halves['report']):,} ({npos['report']} pos)", flush=True)
        for label, tag in arms():
            rec = {"population": pop, "arm": label, "tag": tag,
                   "is_frozen": tag == CONTROL,
                   "n_pos_select": npos["select"], "n_pos_report": npos["report"]}
            for half in ("select", "report"):
                c = curve_for(halves[half], splits, cross[tag], thresholds[tag])
                pt = op_point(c)
                rec[f"det_{half}"] = pt.get(f"det_at_{REF_FAR}h", np.nan)
                rec[f"lead_{half}"] = pt.get(f"lead_at_{REF_FAR}h", np.nan)
            rows.append(rec)
            print(f"  {label:28s} select {rec['det_select']:.3f} / "
                  f"report {rec['det_report']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    report(df)


def report(df):
    print("\n" + "=" * 78)
    print("HELD-OUT CHANNEL-SET SELECTION — is the aggregation-width gain real?")
    print("=" * 78)
    print("Criterion: argmax det@1.6/h on SELECT, tie-break longer lead. "
          "REPORT is the quotable number.\n")
    verdicts = []
    for pop, g in df.groupby("population", sort=False):
        g = g.copy()
        # pre-registered criterion, seeds/ties resolved exactly as ews_sweep.py did
        g = g.sort_values(["det_select", "lead_select"], ascending=False)
        sel = g.iloc[0]
        fro = g[g.is_frozen].iloc[0]
        gain = sel.det_report - fro.det_report
        optimism = sel.det_select - sel.det_report
        print(f"--- {pop} "
              f"({int(sel.n_pos_select)} pos select / {int(sel.n_pos_report)} pos report)")
        print(f"  selected on SELECT : {sel.arm:28s} "
              f"det {sel.det_select:.3f} / lead {sel.lead_select:.1f}s")
        print(f"  its REPORT score   : {'':28s} "
              f"det {sel.det_report:.3f} / lead {sel.lead_report:.1f}s"
              f"   (selection optimism {optimism:+.3f})")
        print(f"  FROZEN on REPORT   : {fro.arm:28s} "
              f"det {fro.det_report:.3f} / lead {fro.lead_report:.1f}s")
        print(f"  >> HONEST GAIN     : {gain:+.3f} detection, "
              f"{sel.lead_report - fro.lead_report:+.1f}s lead")
        if sel.is_frozen:
            print("     the frozen arm WON the selection — no restricted set is better")
        verdicts.append((pop, gain, bool(sel.is_frozen)))
        print()

    print("=" * 78)
    real = [v for v in verdicts if v[1] > 0.02]
    if not real:
        print("✅ VERDICT: the +0.055 does NOT survive held-out selection on any\n"
              "   population. The §2 negative stands as written, and the restricted\n"
              "   arms are reported as a diagnostic that FAILED to overturn it —\n"
              "   which is a stronger answer to van Beers than never having looked.")
    else:
        pops = ", ".join(f"{p} {g:+.3f}" for p, g, _ in real)
        print(f"⚠ VERDICT: the gain SURVIVES held-out selection on: {pops}.\n"
              "   The §2 negative is partly an aggregation artifact and must be\n"
              "   restated. Every margin measured against the affected bar needs\n"
              "   re-checking, and the paper should report the restricted arm as\n"
              "   Tier-0's honest best.")
    print("=" * 78)


if __name__ == "__main__":
    main()
