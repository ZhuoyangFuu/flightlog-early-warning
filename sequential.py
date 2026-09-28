#!/usr/bin/env python
"""Item 3: sequential alarm rules (PROJECT_PLAN §7.5, §11.0.1).

Why this exists
---------------
`tier1_baseline.flight_curve` alarms on the **first window whose instantaneous
score crosses a threshold**. There is no persistence, no smoothing, no memory.
Over ~450 windows per flight at FAR ~1.6/h, one noisy window destroys a negative
flight, and the only way to buy a lower FAR is to raise the threshold -- which
pushes the crossing *later* and costs lead time. That is the mechanism behind
§4.3's finding that Tier-1 GBT, thresholded to Tier-0's detection rate, fires
5.4 s before the event.

Tier-0 does not have this problem: critical slowing down *accumulates*, so its
indicator drifts slowly and crosses early. This file gives the learned tiers the
same property, using the standard sequential-change-detection rules:

  raw    the existing instantaneous rule; the control, and it must reproduce the
         cached curve exactly or the harness is wrong
  ewma   exponentially weighted moving average of the score, span `k`
  kofm   alarm when k of the last m windows cross -- persistence, the cheapest
         way to reject isolated spikes
  cusum  Page's CUSUM on the standardised score: S_t = max(0, S_{t-1} + z_t - d).
         The classical change detector. Accumulates weak evidence rather than
         waiting for one strong window, which is exactly the property that buys
         lead time.

FAIRNESS IS THE WHOLE POINT
---------------------------
Every rule is applied **identically to every tier**, Tier-0 included, and scored
through the same `flight_curve` / `op_point` on the same roster. A rule that helps
the learned model also helps the baselines, and the comparison stays honest. A
rule applied only to the proposed model would be worthless.

The rules transform a score SERIES into another score SERIES; the alarm logic,
threshold grid, survival estimator and operating point are untouched. So this is
a decision-rule change, not a metric change.

Input: the per-window score dumps written by `horizon.py --dump-scores` and
`probe.py --dump-scores` into `meta/scores/`. Tier-0 is handled separately
because its cached artifact is first-crossing bins, not a score series.

Usage
-----
    python sequential.py --list
    python sequential.py --arm horizon_scores_shift120_B_fixed
    python sequential.py --all
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier1_baseline import (REF_FAR, HZ, STRIDE, threshold_grid, flight_curve,
                            op_point)
from eval_common import roster_for

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")
SCORES = os.path.join(META, "scores")

SPLIT_COL = {"A": "split_A_s{seed}", "B": "split_B", "B_fixed": "split_B_fixed",
             "C": "split_C_cond"}

# Score-dump basename prefix -> the prefix of that producer's cached artifacts.
# `horizon.py` writes horizon_scores_* / horizon_pt_* / horizon_curve_*;
# `horizon_tier1.py` (the RESULTS.md 4.7 control) writes h1_*. Both must resolve,
# because an arm whose metadata cannot be found silently falls back to the
# dump-derived roster -- the +9.9% denominator bug documented in roster_for_arm.
ARM_PREFIX = {"horizon_scores_": "horizon", "h1_scores_": "h1"}


def _artifact_tag(name):
    """(tag, artifact prefix) for a score-dump basename."""
    for pre, art in ARM_PREFIX.items():
        if name.startswith(pre):
            return name[len(pre):], art
    return name, "horizon"


def rosters_for_arm(name, splits, df):
    """[(label, roster)] to score `name` against.

    A learned arm was fit on one split and is scored on that split's test set, so
    it gets exactly one roster. Tier-0 is not fit on anything: its z-series covers
    the whole curated tier, and RESULTS.md 2.1 already established that its
    operating point is strongly SPLIT-DEPENDENT (det/lead 0.143/217.3 s on A vs
    0.153/51.7 s on B_fixed). So an `ews_scores_*` dump is scored once per split,
    against each split's own test roster -- which is what makes the resulting rows
    comparable to the learned arms one for one.
    """
    if name.startswith("ews_scores_"):
        out = []
        for sp, tmpl in SPLIT_COL.items():
            col = tmpl.format(seed=0)
            if col in splits.columns:
                te = set(splits.loc[splits[col] == "test", "log_id"])
                out.append((sp, roster_for(splits, te)))
        return out
    return [(None, roster_for_arm(name, splits, df)[0])]


def roster_for_arm(name, splits, df):
    """The roster the arm was ORIGINALLY scored against -- not the dump's ids.

    This is subtle and it silently inflates every number if you get it wrong.
    `flight_curve` divides by `len(pos)`, the positive flights on the roster, and
    counts a flight with no scoreable window as a MISS (`t is None -> continue`).
    The dump only contains windows that were scored, so deriving the roster from
    `df.log_id.unique()` drops exactly those flights out of the DENOMINATOR
    instead of counting them as misses.

    Measured on `shift240_A_seed1`: 31 of 344 positive test flights contribute no
    window at all, so the dump-derived roster reported det 0.338 where the arm's
    own cached curve says 0.307 -- a free +9.9%. Thresholds, FAR and median lead
    all matched to 1e-14, so nothing else flagged it; only the denominator moved.

    So: rebuild the roster from the split column, exactly as `horizon.py` does.
    """
    tag, art = _artifact_tag(name)
    ppath = os.path.join(META, f"{art}_pt_{tag}.json")
    if os.path.exists(ppath):
        with open(ppath) as f:
            pt = json.load(f)
        col = SPLIT_COL[pt["split"]].format(seed=pt.get("seed", 0))
        if col in splits.columns:
            te = set(splits.loc[splits[col] == "test", "log_id"])
            return roster_for(splits, te), col
    print(f"    WARNING: no split metadata for {name}; falling back to the "
          f"dump's own ids. Detection rate will read HIGH -- do not quote it.")
    return roster_for(splits, set(df.log_id.unique())), None

# Rule grids. Kept small on purpose: this is a decision rule, not a model, and a
# wide sweep over it on the test arm would be threshold-tuning by another name.
# Each is reported for every arm so the choice is visible rather than selected.
#
# EVERY GRID IS IN SECONDS, NOT SAMPLES, AND THAT IS LOAD-BEARING.
# ----------------------------------------------------------------
# The learned tiers score windows at `tier1_baseline.STRIDE = 20` bins = 1 s per
# sample. Tier-0's `zmax` is one value PER BIN = 0.05 s per sample at 20 Hz. So a
# grid written in samples is 20x different in time between the two tiers: `ewma`
# span 15 would be 15 s for a probe and 0.75 s for Tier-0, and `kofm` (3,5) would
# be "3 of the last 5 seconds" against "3 of the last 0.25 s". That comparison
# would look fair and be nothing of the sort -- and the whole point of this file
# is that a rule applied to only one tier is worthless.
#
# So the grid is declared in seconds and converted per arm using that arm's own
# sample period. At dt = 1 s these reproduce the original sample-based grid
# exactly ([5,15,40], [(2,3),(3,5),(5,10)]), so the learned-tier numbers in
# RESULTS.md 4.6 are unchanged by this refactor.
EWMA_SPANS_S = [5.0, 15.0, 40.0]
# (fraction of the window that must cross, window length in seconds)
KOFM_S = [(2 / 3, 3.0), (3 / 5, 5.0), (5 / 10, 10.0)]
CUSUM_DRIFT = [0.25, 0.5, 1.0]

# Sample period of a score series, in seconds. Learned dumps are on Tier-1's
# 1 s window stride; Tier-0 is one sample per 20 Hz bin.
DT_LEARNED = STRIDE / HZ
DT_TIER0 = 1.0 / HZ

# Tier-0's own persistence requirement: ews_baseline cfg["persist_bins"] = 0.5 s.
TIER0_PERSIST_S = 0.5


def rules_for(dt):
    """The rule grid realised in SAMPLES for a series sampled every `dt` seconds.

    `param` is carried through to the output as the human-readable spec (seconds),
    so `sequential.csv` rows stay comparable across tiers even though the sample
    counts differ by 20x.
    """
    yield ("raw", None, None)
    for sec in EWMA_SPANS_S:
        yield ("ewma", f"{sec:g}s", max(1, int(round(sec / dt))))
    for frac, sec in KOFM_S:
        m = max(2, int(round(sec / dt)))
        k = max(1, min(m, int(round(frac * m))))
        yield ("kofm", f"{frac * sec:g}/{sec:g}s", (k, m))
    for d in CUSUM_DRIFT:
        # CUSUM accumulates once per SAMPLE, so at 20x the sample rate it would
        # accumulate 20x as fast for the same drift. Scaling the increment by dt
        # makes S an integral in units of sd-seconds for either tier.
        yield ("cusum", f"{d:g}", (d, dt))
    # Tier-0's NATIVE rule, so its own operating point appears in the table it is
    # being compared against. `ews_baseline._first_persistent` requires the
    # indicator to hold above threshold for `persist_bins` = 0.5 s CONSECUTIVE,
    # which is k-of-m with k = m. At the learned tiers' dt = 1 s this collapses to
    # m = 1, i.e. `raw`, so it is only emitted where it means something.
    m = int(round(TIER0_PERSIST_S / dt))
    if m >= 2:
        yield ("kofm", f"{TIER0_PERSIST_S:g}s-consec", (m, m))


# All three rules are vectorised. A per-window Python loop over the 3.09M-row
# B_fixed dump, repeated for 10 rules, is minutes of pure interpreter overhead per
# arm; these run in seconds and are exact, not approximations.

def ewma(s, span):
    a = 2.0 / (span + 1.0)
    return pd.Series(s).ewm(alpha=a, adjust=False).mean().to_numpy()


def kofm(s, k, m):
    """Score whose crossing of `th` means 'k of the last m raw scores crossed th'.

    The k-th largest of the trailing window crosses `th` exactly when at least k
    of the last m do, so the rule composes with the existing threshold sweep
    instead of needing its own. Vectorised with a sliding window view; the first
    m-1 positions are handled separately since their window is short.
    """
    n = len(s)
    out = np.full(n, -np.inf, dtype=np.float64)
    if n >= m:
        w = np.lib.stride_tricks.sliding_window_view(s, m)      # (n-m+1, m)
        out[m - 1:] = np.partition(w, m - k, axis=1)[:, m - k]
    # ramp-up: window shorter than m, needs at least k samples to alarm
    for i in range(k - 1, min(m - 1, n)):
        w = s[:i + 1]
        out[i] = np.partition(w, len(w) - k)[len(w) - k]
    return out


def cusum(s, drift, dt, mu, sd):
    """Page's CUSUM on the standardised score: S_t = max(0, S_{t-1} + (z_t - d)*dt).

    Closed form rather than a loop, using the reflected-random-walk identity
        S_t = C_t - min(0, min_{j<=t} C_j),   C = cumsum((z - d) * dt)
    which is exact for the non-negative-reset recursion above.

    Standardisation uses the NEGATIVE-window mean/sd of this arm, passed in, so
    the drift term `d` means the same thing across arms and tiers. The `* dt`
    makes S an integral in sd-SECONDS rather than sd-samples, so a tier sampled
    20x faster does not accumulate 20x as quickly for the same `d`. Scaling by a
    constant cannot change `raw`, and for a fixed tier it only rescales S, which
    the threshold grid absorbs -- it matters only when comparing across tiers.
    """
    z = (s - mu) / (sd if sd > 0 else 1.0)
    c = np.cumsum((z - drift) * dt)
    return c - np.minimum(0.0, np.minimum.accumulate(c))


def transform(rule, spec, mu, sd, starts, scores):
    """Apply `rule` within each flight. `starts` are the flight boundaries.

    The frame is pre-sorted by (log_id, t_bin), so each flight is one contiguous
    slice and no gather is needed -- which matters, because a fancy-index copy of
    a 3M-row array per flight per rule is the other way this gets slow.
    """
    if rule == "raw":
        return scores
    out = np.empty(len(scores), dtype=np.float64)
    fn = {"ewma": lambda s: ewma(s, spec),
          "kofm": lambda s: kofm(s, *spec),
          "cusum": lambda s: cusum(s, spec[0], spec[1], mu, sd)}.get(rule)
    if fn is None:
        raise SystemExit(f"unknown rule {rule}")
    for a, b in zip(starts[:-1], starts[1:]):
        out[a:b] = fn(scores[a:b])
    return out


def score_arm(path, splits, verbose=True, dt=DT_LEARNED):
    name = os.path.basename(path)[:-8]
    df = pd.read_parquet(path).sort_values(["log_id", "t_bin"]).reset_index(drop=True)
    rosters = rosters_for_arm(name, splits, df)
    scores = df.score.to_numpy(dtype=np.float64)
    lid = df.log_id.to_numpy()
    tb = df.t_bin.to_numpy()
    isneg = (df.y == 0).to_numpy()
    # nan-aware: the learned dumps are all finite, but Tier-0's z-series is ~2.7%
    # non-finite (bins whose expanding baseline is still shorter than
    # `min_baseline`). A plain mean/std would be NaN and take every rule with it.
    mu = float(np.nanmean(scores[isneg]))
    sd = float(np.nanstd(scores[isneg]))
    # Flight boundaries in the sorted frame: computed once, reused by every rule.
    starts = np.concatenate(([0], np.flatnonzero(lid[1:] != lid[:-1]) + 1,
                             [len(lid)]))

    # A non-finite score means "no evidence available at this bin", NOT "no
    # anomaly" and NOT "extreme anomaly". `raw` already treats it correctly:
    # `flight_curve` tests `s >= th`, and NaN fails every comparison, so the bin
    # simply cannot alarm. The other rules have memory, and a NaN inside a
    # cumsum or an ewm destroys everything after it -- so they run on a series
    # with non-finite bins filled at the negative-window MEAN, which contributes
    # exactly zero to CUSUM's standardised increment. `raw` is deliberately left
    # on the untouched array so the control still reproduces the cached curve.
    filled = np.where(np.isfinite(scores), scores, mu)
    n_bad = int(scores.size - np.isfinite(scores).sum())
    if n_bad and verbose:
        print(f"    {n_bad:,} non-finite scores ({100 * n_bad / scores.size:.2f}%) "
              f"filled at the negative mean for the memory-bearing rules")

    rows = []
    for rule, param, spec in rules_for(dt):
        v = transform(rule, spec, mu, sd, starts,
                      scores if rule == "raw" else filled)
        neg = v[isneg & np.isfinite(v)]
        if neg.size < 4 or not np.isfinite(v).any():
            continue
        sbf = {lid[a]: (tb[a:b], v[a:b]) for a, b in zip(starts[:-1], starts[1:])}
        # The threshold grid is built once per rule from the transformed
        # negatives, then reused across splits, so a split's rows differ only by
        # which flights are scored -- never by where the thresholds sit.
        grid = threshold_grid(neg)
        for label, roster in rosters:
            curve = flight_curve(sbf, roster, grid)
            pt = op_point(curve)
            rows.append({"rule": rule, "param": str(param), "eval_split": label,
                         f"det@{REF_FAR}h": pt.get(f"det_at_{REF_FAR}h", np.nan),
                         f"lead@{REF_FAR}h": pt.get(f"lead_at_{REF_FAR}h", np.nan),
                         "det_freeze": pt.get("detection_rate", np.nan),
                         "lead_freeze": pt.get("median_lead_s", np.nan),
                         "far_freeze": pt.get("far_per_hour", np.nan)})
            if verbose and label is not None:
                r = rows[-1]
                print(f"    {rule:6s} {str(param):9s} [{label:8s}] "
                      f"det {r[f'det@{REF_FAR}h']:.3f}  "
                      f"lead {r[f'lead@{REF_FAR}h']:6.1f}s", flush=True)
        if verbose and rosters[0][0] is None:
            r = rows[-1]
            print(f"    {rule:6s} {str(param):9s} det {r[f'det@{REF_FAR}h']:.3f}  "
                  f"lead {r[f'lead@{REF_FAR}h']:6.1f}s", flush=True)
        if rule == "raw":
            if name.startswith("ews_scores_"):
                check_raw_tier0(rows, verbose, name)
            else:
                check_raw(name, curve, verbose)
    return pd.DataFrame(rows)


def tier0_ref_path(name):
    """Which per-split reference a Tier-0 dump must reproduce.

    Added 2026-09-02 with the channel-set arms (RESULTS.md 2.2). Each restricted
    arm has its OWN crossing-derived per-split points, written by
    `ews_channels.py` as `ews_bysplit_chan-<group>-k<k>.json`; checking it against
    the frozen arm's `ews_bysplit_ext.json` would fail by construction and would
    say nothing. Resolving per arm keeps the control meaningful: every dump is
    verified against the independent code path for THAT configuration.
    """
    tag = name[len("ews_scores_"):]
    if tag.startswith("chan-"):
        return os.path.join(META, f"ews_bysplit_{tag}.json")
    return os.path.join(META, "ews_bysplit_ext.json")


def check_raw_tier0(raw_rows, verbose=True, name="ews_scores_w10k3"):
    """Tier-0's control: `raw` must reproduce RESULTS.md 2.1's per-split bars.

    `check_raw` cannot be used here. It diffs against a cached `horizon_curve_*`
    keyed on threshold, but Tier-0's cached curves were swept over a FIXED
    threshold list (2, 2.5, ... 200) while this file derives thresholds from
    negative quantiles. The two share no thresholds, so that diff would be
    vacuous rather than reassuring.

    The right control is the operating point itself. `meta/ews_bysplit_ext.json`
    holds det@REF_FAR and lead@REF_FAR per split, computed by `ews_split.py` from
    the ORIGINAL first-crossing traces -- a completely independent code path from
    the dumped z-series this file reads. Agreement across all four splits means
    the dump, the roster construction and the threshold grid are all sound.

    Tolerances are loose on purpose: the two paths interpolate to FAR = 1.6/h
    along differently-placed grids, so exact equality is not expected and would
    indicate the comparison is not actually independent. Measured 2026-08-24:
    every split within 0.005 detection.
    """
    ref_path = tier0_ref_path(name)
    if not os.path.exists(ref_path):
        if verbose:
            print(f"    control: {os.path.basename(ref_path)} missing -- NOT verified")
        return None
    with open(ref_path) as f:
        ref = json.load(f)["by_split"]
    ok, lines = True, []
    for r in raw_rows:
        sp = r.get("eval_split")
        if sp not in ref:
            continue
        want_d = ref[sp].get(f"det_at_{REF_FAR}h", np.nan)
        want_l = ref[sp].get(f"lead_at_{REF_FAR}h", np.nan)
        dd = abs(r[f"det@{REF_FAR}h"] - want_d)
        dl = abs(r[f"lead@{REF_FAR}h"] - want_l)
        good = dd <= 0.01 and dl <= 10.0
        ok &= good
        lines.append(f"      {sp:8s} det {r[f'det@{REF_FAR}h']:.3f} vs {want_d:.3f} "
                     f"(d {dd:.3f})  lead {r[f'lead@{REF_FAR}h']:6.1f} vs {want_l:6.1f} "
                     f"(d {dl:.1f}s)  {'ok' if good else 'OUT OF TOLERANCE'}")
    if verbose or not ok:
        print(f"    control: {'PASS' if ok else 'FAIL'} vs RESULTS.md 2.1 per-split "
              f"bars (ews_bysplit_ext.json)")
        for ln in lines:
            print(ln)
    return ok


def check_raw(name, curve, verbose=True):
    """The control: `raw` must reproduce the arm's cached curve to the digit.

    This is the whole reason `raw` is in the rule list. It is what caught the
    roster-denominator bug documented in `roster_for_arm`, and it is cheap, so it
    runs on every arm every time rather than being a one-off.
    """
    tag, art = _artifact_tag(name)
    cpath = os.path.join(META, f"{art}_curve_{tag}.csv")
    if not os.path.exists(cpath):
        if verbose:
            print(f"    control: no cached curve for {tag} -- NOT verified")
        return None
    old = pd.read_csv(cpath)
    m = curve.merge(old, on="threshold", suffixes=("_new", "_old"))
    if not len(m):
        # No shared thresholds. This is NOT automatically a failure: the dump
        # stores float32 while the cached curve was built from float64
        # `predict_proba` output, and a BOOSTED TREE's scores are heavily tied,
        # so float32 rounding merges values that were distinct and every
        # log-spaced rank in `threshold_grid` lands somewhere else. Observed on
        # every `gbt` arm and on none of the `lr` arms.
        #
        # Fall back to the control Tier-0 already uses for the same reason
        # (run_ews_scores.sh verify): compare the thing that actually gets
        # quoted -- the interpolated operating point -- instead of a
        # threshold-by-threshold diff that is vacuous when the grids differ.
        ppath = os.path.join(META, f"{art}_pt_{tag}.json")
        if not os.path.exists(ppath):
            print(f"    control: FAIL -- no shared thresholds and no {art}_pt "
                  f"for {tag}")
            return False
        with open(ppath) as f:
            pt = json.load(f)
        got = op_point(curve)
        dd = abs(got.get(f"det_at_{REF_FAR}h", np.nan)
                 - pt.get(f"det_at_{REF_FAR}h", np.nan))
        dl = abs(got.get(f"lead_at_{REF_FAR}h", np.nan)
                 - pt.get(f"lead_at_{REF_FAR}h", np.nan))
        ok = bool(dd < 5e-3 and dl < 2.0)
        print(f"    control: {'PASS' if ok else 'FAIL'} vs cached OPERATING POINT "
              f"(thresholds disjoint -- float32 dump of tied {pt.get('head','?')} "
              f"scores); |ddet| {dd:.2g}, |dlead| {dl:.2g}s")
        return ok
    worst = {c: float(np.nanmax(np.abs(m[f"{c}_new"] - m[f"{c}_old"])))
             for c in ("far_per_hour", "detection_rate", "median_lead_s")}
    ok = worst["far_per_hour"] < 1e-9 and worst["detection_rate"] < 1e-9 \
        and worst["median_lead_s"] < 1e-9
    if verbose or not ok:
        tick = "PASS" if ok else "FAIL"
        print(f"    control: {tick} vs cached curve ({len(m)} shared thresholds; "
              f"max|dfar| {worst['far_per_hour']:.2g}, "
              f"max|ddet| {worst['detection_rate']:.2g}, "
              f"max|dlead| {worst['median_lead_s']:.2g})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", help="basename (no .parquet) under meta/scores/")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    avail = sorted(glob.glob(os.path.join(SCORES, "*.parquet")))
    if args.list or not (args.arm or args.all):
        print(f"{len(avail)} score dumps in {SCORES}:")
        for a in avail:
            print("  ", os.path.basename(a)[:-8])
        if not avail:
            print("\n  none yet -- produce them with:")
            print("    python horizon.py --grid 2c --dump-scores")
            print("  (an arm that lacks scores is refitted; cached arms with "
                  "scores are reused)")
        return

    splits = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))
    targets = avail if args.all else [os.path.join(SCORES, args.arm + ".parquet")]
    out = {}
    for p in targets:
        name = os.path.basename(p)[:-8]
        if not os.path.exists(p):
            print(f"missing {p}"); continue
        print(f"\n=== {name} ===", flush=True)
        dt = DT_TIER0 if name.startswith("ews_scores_") else DT_LEARNED
        t = score_arm(p, splits, dt=dt)
        t.insert(0, "dt_s", dt)
        t.insert(0, "arm", name)
        out[name] = t
    if out:
        # 2026-09-14: one part file per arm, and the combined table is the UNION of every
        # part on disk. Before this, `--arm X` overwrote sequential.csv with X's rows alone
        # and silently destroyed the 1,715-row sweep; the table is rebuilt from the parts.
        parts = os.path.join(META, "sequential_parts")
        os.makedirs(parts, exist_ok=True)
        for name, t in out.items():
            t.to_csv(os.path.join(parts, name + ".csv"), index=False)
        allt = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(parts, "*.csv")))],
                         ignore_index=True)
        o = os.path.join(META, "sequential.csv")
        allt.to_csv(o + ".tmp", index=False)
        os.replace(o + ".tmp", o)
        pd.set_option("display.width", 200)
        print(f"\nwrote {o}")
        print("\nBest rule per arm by lead at the reference FAR "
              "(det must not fall below the raw rule's):")
        for name, t in out.items():
            raw = t[t.rule == "raw"]
            if not len(raw):
                continue
            floor = raw.iloc[0][f"det@{REF_FAR}h"]
            ok = t[t[f"det@{REF_FAR}h"] >= floor]
            if not len(ok):
                print(f"  {name}: no rule holds detection"); continue
            b = ok.loc[ok[f"lead@{REF_FAR}h"].idxmax()]
            print(f"  {name}: {b.rule} {b.param} -> det {b[f'det@{REF_FAR}h']:.3f} "
                  f"lead {b[f'lead@{REF_FAR}h']:.1f}s "
                  f"(raw: {floor:.3f} / {raw.iloc[0][f'lead@{REF_FAR}h']:.1f}s)")


if __name__ == "__main__":
    main()
