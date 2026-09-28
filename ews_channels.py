#!/usr/bin/env python
"""Tier-0 AGGREGATION-WIDTH DIAGNOSTIC — is the negative about EWS, or about us?

Why this exists
---------------
van Beers, Scheffer et al. (arXiv:2512.20868 / PNAS 2026) report a POSITIVE
classical-EWS result for quadrotor loss of control using lag-1 autocorrelation
on ROTOR SPEED ALONE — one signal, chosen by physics, on 247 lab flights with
induced blade damage. Our Tier-0 aggregates a top-k z-score over all 87 channels
and detects 18.6% at FAR 1.62/h. Ditlevsen & Ditlevsen (arXiv:2609.01164) show
that an early-warning signal living in a subspace can be *destroyed* by
aggregating over the full state.

So a reviewer will ask: is `RESULTS.md` §2's negative a property of CLASSICAL
EWS, or of OUR AGGREGATION? This file answers it by re-running the frozen Tier-0
indicator over physically-chosen channel subsets, changing exactly one thing.

⚠⚠ THIS DOES NOT RE-SELECT TIER-0, AND MUST NEVER BE ALLOWED TO.
`meta/ews_curve_FROZEN.csv` and `meta/ews_frozen_config.json` were fixed by a
pre-registered 12-configuration sweep and stay the headline. Everything here is
a POST-HOC DIAGNOSTIC, written to its own tags and its own summary, and must be
reported as one. If a restricted arm beats the frozen arm, that is a finding to
publish beside the headline — not a licence to swap the headline for it.

What is held fixed (single-variable control)
--------------------------------------------
Window 10 s, warm-up 20 s, the v2 label set, the persistence rule, the
standardisation, and the 30-point EXTENDED threshold grid — the same grid
`w10k3ext` uses, because RESULTS.md §2.1 established the 16-point grid truncates
at 80 exactly where Tier-0's `B_fixed` point lands. Only the channel set and
top-k vary.

The grid, declared before any number is read from it
----------------------------------------------------
    rotor_out (8 ch)   x  topk 1,2,3       the direct van Beers analogue
    rotor     (16 ch)  x  topk 1,2,3,5     + normalised motor commands
    rates     (6 ch)   x  topk 1,2,3       the fast inner loop
    rotor_rates (22)   x  topk 1,2,3,5     both, and nothing else
    all       (87 ch)  x  topk 3           CONTROL — reuses cached w10k3ext
                                           traces, must reproduce 0.186 / 66.0 s

Usage
-----
    bash run_ews_channels.sh start     # detached; see that launcher
    python ews_channels.py --list      # the grid, runs nothing
    python ews_channels.py --status    # progress + results so far
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import CHANNEL_GROUPS, channel_indices, N_CHANNELS

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
META = os.path.join(ROOT, "meta")
EXCLUDE = os.path.join(META, "trigger_v2_exclude.csv")

WINDOW_S = 10.0        # the frozen window — held fixed on purpose
FAR_CAP = 2.0
REF_FAR = 1.6

# The extended 30-point grid, copied from the w10k3ext run (logs/ews_ext.log) so
# the diagnostic arms and the reference arm share a threshold set exactly, plus
# FOUR POINTS BELOW 2.0 as a SUPERSET.
#
# Why the extension, decided before the run: a restricted arm takes the k-th
# largest z over far fewer channels, so its score distribution is compressed —
# measured on a 200-log pilot, median `score_max` is 5.06 on 8 rotor channels
# against 19.14 on all 87. The frozen grid starts at 2.0, and an arm whose
# optimum lands there would be sitting on the edge of its own grid: exactly the
# failure RESULTS.md §2.1 had to correct at the top end. A superset is safe by
# the same argument used there — the shared thresholds are unchanged, so the
# added points can only add operating points, never move an existing one.
THRESHOLDS = [1.0, 1.25, 1.5, 1.75,
              2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0,
              20.0, 30.0, 50.0, 65.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0,
              110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 175.0, 200.0]

# (group, topk). Declared here, before the run. Do not extend after reading
# results without saying so in RESULTS.md.
GRID = ([("rotor_out", k) for k in (1, 2, 3)]
        + [("rotor", k) for k in (1, 2, 3, 5)]
        + [("rates", k) for k in (1, 2, 3)]
        + [("rotor_rates", k) for k in (1, 2, 3, 5)])

# The control arm is not re-run: `w10k3ext` IS all-channels / top-3 / window 10 s
# on this exact threshold grid.
CONTROL_TAG = "w10k3ext"


def tag_of(group, k):
    return f"chan-{group}-k{k}"


def traces_path(tag):
    return os.path.join(META, f"ews_traces_{tag}.parquet")


def curve_path(tag, v2):
    return os.path.join(META, f"ews_curve_{tag}{'_v2' if v2 else ''}.csv")


def n_channels(group):
    ci = channel_indices(group)
    return N_CHANNELS if ci is None else len(ci)


def op_from_curve(path):
    """Freeze-rule operating point + the reference-FAR point, from a curve CSV."""
    c = pd.read_csv(path).dropna(subset=["far_per_hour"]).sort_values("far_per_hour")
    if c.empty:
        return {}
    capped = c[c.far_per_hour <= FAR_CAP]
    best = capped.loc[capped.detection_rate.idxmax()] if len(capped) else None
    ref = c.iloc[(c.far_per_hour - REF_FAR).abs().argmin()]
    out = {
        "min_far": float(c.iloc[0].far_per_hour),
        "det_at_min_far": float(c.iloc[0].detection_rate),
        f"det_near_{REF_FAR}h": float(ref.detection_rate),
        f"lead_near_{REF_FAR}h": float(ref.median_lead_s),
        "far_of_ref_pt": float(ref.far_per_hour),
    }
    if best is not None:
        out.update(det_at_cap=float(best.detection_rate),
                   lead_at_cap=float(best.median_lead_s),
                   far_at_cap=float(best.far_per_hour),
                   thresh_at_cap=float(best.threshold))
        # Standing rule (RESULTS.md §2.1): never let an operating point sit at
        # the edge of its own grid. Flag it rather than silently reporting it.
        edge = ""
        if best.threshold <= min(THRESHOLDS):
            edge = "LOW"
        elif best.threshold >= max(THRESHOLDS):
            edge = "HIGH"
        out["grid_edge"] = edge
    return out


def run(cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def score_and_eval(group, k, dump_scores=False):
    tag = tag_of(group, k)
    tp = traces_path(tag)
    sp = os.path.join(META, "scores", f"ews_scores_{tag}.parquet")
    # With --dump-scores the per-bin z-series is what we are after, and the
    # traces alone do not contain it -- so an arm whose traces exist but whose
    # dump does not must still be rescored. Without it, cached traces are enough.
    need = not os.path.exists(tp) or (dump_scores and not os.path.exists(sp))
    if not need:
        print(f"cached, skipping score: {os.path.basename(tp)}", flush=True)
    else:
        cmd = [PY, os.path.join(ROOT, "ews_baseline.py"),
               "--shards", os.path.join(ROOT, "proc", "curated-*.parquet"),
               "--tag", tag, "--workers", os.environ.get("EWS_WORKERS", "12"),
               "--window-s", WINDOW_S, "--topk", k, "--channels", group,
               "--thresholds", ",".join(f"{t:g}" for t in THRESHOLDS)]
        if dump_scores:
            cmd.append("--dump-scores")
        run(cmd)
    for v2 in (False, True):
        cp = curve_path(tag, v2)
        if os.path.exists(cp):
            continue
        cmd = [PY, os.path.join(ROOT, "ews_eval.py"), "--traces", tp, "--out", cp]
        if v2:
            cmd += ["--exclude-csv", EXCLUDE]
        run(cmd)
    # per-split points, so the diagnostic can be compared against the per-split
    # Tier-0 bars the learned tiers actually face (RESULTS.md §2.1), not only
    # against the corpus-wide headline.
    bs = os.path.join(META, f"ews_bysplit_{tag}.json")
    if not os.path.exists(bs):
        run([PY, os.path.join(ROOT, "ews_split.py"),
             "--traces", tag, "--out-suffix", f"_{tag}"])
    return tag


def summarise():
    rows = []
    for group, k in [("all", 3)] + GRID:
        tag = CONTROL_TAG if group == "all" else tag_of(group, k)
        cp = curve_path(tag, v2=True)
        if not os.path.exists(cp):
            continue
        r = {"group": group, "n_channels": n_channels(group), "topk": k,
             "tag": tag, "is_control": group == "all"}
        r.update(op_from_curve(cp))
        bs = os.path.join(META, f"ews_bysplit_{tag}.json")
        if group == "all":
            bs = os.path.join(META, "ews_bysplit_ext.json")
        if os.path.exists(bs):
            with open(bs) as f:
                by = json.load(f).get("by_split", {})
            for s in ("A", "B_fixed", "C"):
                if s in by:
                    r[f"det_{s}"] = by[s].get(f"det_at_{REF_FAR}h")
                    r[f"lead_{s}"] = by[s].get(f"lead_at_{REF_FAR}h")
        rows.append(r)
    if not rows:
        print("nothing scored yet")
        return None
    df = pd.DataFrame(rows)
    out = os.path.join(META, "ews_channels_summary.csv")
    df.to_csv(out, index=False)

    cols = ["group", "n_channels", "topk", "det_at_cap", "lead_at_cap",
            "far_at_cap", "grid_edge", f"det_near_{REF_FAR}h",
            f"lead_near_{REF_FAR}h", "det_A", "det_B_fixed", "det_C"]
    cols = [c for c in cols if c in df.columns]
    print("\n===== AGGREGATION-WIDTH DIAGNOSTIC "
          "(post-hoc; the frozen Tier-0 headline is unchanged) =====")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {out}")

    ctrl = df[df.is_control]
    if len(ctrl):
        c = ctrl.iloc[0]
        print(f"\nCONTROL (all 87 ch, top-3): det {c.get('det_at_cap', float('nan')):.3f} "
              f"lead {c.get('lead_at_cap', float('nan')):.1f}s "
              f"@ FAR {c.get('far_at_cap', float('nan')):.3f}/h "
              f"— must match RESULTS.md §2's 0.186 / 66.0 s / 1.617")
        best = df[~df.is_control]
        if len(best) and "det_at_cap" in best:
            b = best.loc[best.det_at_cap.idxmax()]
            print(f"BEST RESTRICTED: {b.group} ({int(b.n_channels)} ch) top-{int(b.topk)} "
                  f"-> det {b.det_at_cap:.3f} lead {b.lead_at_cap:.1f}s "
                  f"@ FAR {b.far_at_cap:.3f}/h")
            d = b.det_at_cap - c.get("det_at_cap", np.nan)
            print(f"\nVERDICT: restricted − frozen = {d:+.3f} detection.")
            if d > 0.02:
                print("  ⚠ THE RESTRICTED ARM IS MATERIALLY BETTER. The §2 negative is\n"
                      "    partly an AGGREGATION artifact. Report both, weaken the\n"
                      "    'classical EWS fails' claim to name the aggregation, and\n"
                      "    re-check every margin measured against the 0.186 bar.")
            elif d < -0.02:
                print("  ✅ Restricting to the physics-chosen channels is WORSE. The\n"
                      "    negative is about the indicators, not our aggregation, and\n"
                      "    top-k over 87 channels is the stronger baseline — say so.")
            else:
                print("  ✅ Within noise of the frozen arm: the negative is robust to\n"
                      "    aggregation width. This is the cleanest answer to van Beers.")
    edges = df[df.get("grid_edge", "").astype(str).isin(["LOW", "HIGH"])]
    if len(edges):
        print(f"\n⚠ {len(edges)} arm(s) have an operating point AT A GRID EDGE "
              f"({', '.join(edges.tag)}). Extend --thresholds for those before "
              f"quoting them (standing rule, RESULTS.md §2.1).")
    return df


def status():
    print(f"Tier-0 channel diagnostic  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    done = 0
    for group, k in GRID:
        tag = tag_of(group, k)
        t = os.path.exists(traces_path(tag))
        c = os.path.exists(curve_path(tag, v2=True))
        s = os.path.exists(os.path.join(META, f"ews_bysplit_{tag}.json"))
        done += t and c and s
        print(f"  {group:12s} ({n_channels(group):2d} ch) top-{k}   "
              f"traces={'DONE' if t else '....'}  curves={'DONE' if c else '....'}  "
              f"split={'DONE' if s else '....'}")
    print(f"  {done}/{len(GRID)} arms complete")
    summarise()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    # §4.5 caveat 7: the channel grid and the 11-rule grid were swept separately
    # and never crossed, so the best-of-N bar is the max over their UNION (25
    # configurations), not over the 15 x 11 product. Dumping the per-bin z-series
    # lets `sequential.py` sweep the rules on every channel arm and close that.
    # ~0.9 GB per arm, ~12 GB for the grid.
    ap.add_argument("--dump-scores", action="store_true",
                    help="also persist per-bin z-series to meta/scores/ "
                         "(needed for the channel x rule cross)")
    args = ap.parse_args()

    if args.list:
        print(f"window {WINDOW_S:g}s (frozen), v2 labels, "
              f"{len(THRESHOLDS)}-point extended threshold grid\n")
        for group, k in GRID:
            ci = channel_indices(group)
            print(f"  {group:12s} top-{k}  {len(ci):2d} channels: "
                  f"{', '.join(CHANNEL_GROUPS[group])}")
        print(f"\n{len(GRID)} arms + 1 cached control ({CONTROL_TAG})")
        return
    if args.status:
        status()
        return
    if args.summarise:
        summarise()
        return

    if not os.path.exists(EXCLUDE):
        sys.exit(f"missing {EXCLUDE}")
    if not os.path.exists(traces_path(CONTROL_TAG)):
        sys.exit(f"missing control traces {traces_path(CONTROL_TAG)} — the "
                 f"diagnostic is meaningless without the arm it is compared to")

    t0 = time.time()
    for i, (group, k) in enumerate(GRID, 1):
        print(f"\n===== [{i}/{len(GRID)}] {group} ({n_channels(group)} ch) top-{k}  "
              f"(elapsed {(time.time()-t0)/60:.0f} min) =====", flush=True)
        try:
            score_and_eval(group, k, dump_scores=args.dump_scores)
        except subprocess.CalledProcessError as e:
            print(f"!! ARM FAILED ({group}, k={k}): {e}. Continuing.", flush=True)
    summarise()
    print(f"\ntotal elapsed {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
