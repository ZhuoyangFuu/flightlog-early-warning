#!/usr/bin/env python
"""Tier 0 classical early-warning-signal baseline (PROJECT_PLAN.md Section 7).

Computes the textbook EWS indicators -- rolling variance, lag-1 autocorrelation
(critical slowing down), rolling skewness, and optionally DFA -- causally over
each flight, and evaluates them the way Section 8 requires: median lead time at
a fixed false-alarm rate of 1 per hour of flight, plus the full lead-time
versus false-alarm curve.

CAUSALITY. Every indicator at time t uses only samples in (t-W, t]. The
per-flight baseline used for standardisation is an EXPANDING window that also
ends before t, so no future information reaches the alarm decision. This is the
single easiest thing to get wrong in an EWS evaluation and it inflates results
enormously when it happens.

MASKING. Channels absent from a log are masked, and indicators are computed over
valid samples only with a minimum-count requirement. A channel that is mostly
masked contributes nothing rather than contributing zeros.

Outputs
    meta/ews_traces_<tag>.parquet   per-log alarm times at each threshold
    meta/ews_curve_<tag>.csv        lead time vs false-alarm-rate curve
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import (CHANNELS, N_CHANNELS, TARGET_HZ,
                      CHANNEL_GROUPS, channel_indices, channel_names)
from shard_io import decode
from eval_common import window_labels

ROOT = os.path.dirname(os.path.abspath(__file__))
DT = 1.0 / TARGET_HZ


# --------------------------------------------------------------------------
# rolling statistics over masked, irregular-validity series
# --------------------------------------------------------------------------
def _csum(a):
    """Prepend a zero row so windowed sums are S[t+W] - S[t]."""
    return np.concatenate([np.zeros((1, a.shape[1]), dtype=np.float64),
                           np.cumsum(a, axis=0)], axis=0)


def rolling_stats(x, v, W, min_count):
    """Causal rolling mean/var/skew and lag-1 autocorrelation.

    x : (T, C) float32, zeros where invalid
    v : (T, C) bool
    Returns dict of (T, C) arrays; NaN where fewer than min_count valid samples.
    """
    T, C = x.shape
    xv = np.where(v, x, 0.0).astype(np.float64)
    vv = v.astype(np.float64)

    S0 = _csum(vv)
    S1 = _csum(xv)
    S2 = _csum(xv * xv)
    S3 = _csum(xv * xv * xv)

    # lag-1 cross product, valid only where BOTH samples are valid
    pair = np.zeros_like(xv)
    pv = np.zeros_like(vv)
    pair[1:] = xv[1:] * xv[:-1]
    pv[1:] = vv[1:] * vv[:-1]
    P = _csum(pair)
    PV = _csum(pv)

    def win(S):
        return S[W:] - S[:-W]

    n = win(S0)
    s1 = win(S1)
    s2 = win(S2)
    s3 = win(S3)
    pn = win(PV)
    ps = win(P)

    ok = n >= min_count
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, s1 / np.maximum(n, 1), np.nan)
        var = np.where(ok, s2 / np.maximum(n, 1) - mean ** 2, np.nan)
        var = np.maximum(var, 0.0)
        sd = np.sqrt(var)
        m3 = s3 / np.maximum(n, 1) - 3 * mean * (s2 / np.maximum(n, 1)) + 2 * mean ** 3
        skew = np.where(ok & (sd > 1e-12), m3 / np.maximum(sd ** 3, 1e-30), np.nan)
        # Skewness is m3/sd^3 and explodes when sd is near zero on a quiet
        # channel. Unclipped it reached ~1e8 and dominated the aggregate score.
        # Beyond +/-20 the statistic is numerically degenerate, not informative.
        skew = np.clip(skew, -20.0, 20.0)
        # lag-1 autocorrelation via the same windowed moments
        pmean = np.where(pn >= min_count, ps / np.maximum(pn, 1), np.nan)
        ac1 = np.where(ok & (var > 1e-12), (pmean - mean ** 2) / np.maximum(var, 1e-30), np.nan)
        ac1 = np.clip(ac1, -1.5, 1.5)

    # pad so index t corresponds to a window ENDING at t (causal)
    pad = np.full((W, C), np.nan)
    return {
        "var": np.vstack([pad, var]),
        "ac1": np.vstack([pad, ac1]),
        "skew": np.vstack([pad, skew]),
    }


def dfa_alpha(x, v, W, scales=(4, 8, 16, 32)):
    """Detrended fluctuation analysis exponent on the trailing window.

    Evaluated on a coarse time grid because DFA is the expensive indicator;
    values are held between evaluations.
    """
    T, C = x.shape
    out = np.full((T, C), np.nan)
    step = max(1, W // 4)
    logs = np.log(np.asarray(scales, dtype=np.float64))
    for t in range(W, T, step):
        seg = x[t - W:t]
        m = v[t - W:t]
        if m.sum() < W * 0.5 * C / max(C, 1):
            pass
        y = np.where(m, seg, np.nan)
        # integrate the mean-removed profile, NaNs carried as gaps
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(y, axis=0)
            prof = np.nancumsum(np.where(np.isnan(y), 0.0, y - mu), axis=0)
        F = []
        for s in scales:
            nseg = W // s
            if nseg < 2:
                F.append(np.full(C, np.nan))
                continue
            p = prof[: nseg * s].reshape(nseg, s, C)
            idx = np.arange(s, dtype=np.float64)
            idx = (idx - idx.mean()) / max(idx.std(), 1e-12)
            # per-segment linear detrend via projection onto [1, idx]
            mean_p = p.mean(axis=1, keepdims=True)
            slope = (p * idx[None, :, None]).mean(axis=1, keepdims=True)
            resid = p - mean_p - slope * idx[None, :, None]
            F.append(np.sqrt((resid ** 2).mean(axis=(0, 1))))
        Fm = np.vstack(F)
        with np.errstate(invalid="ignore", divide="ignore"):
            lf = np.log(np.maximum(Fm, 1e-30))
            lm = logs[:, None]
            a = ((lm - lm.mean()) * (lf - np.nanmean(lf, axis=0))).sum(axis=0)
            b = ((lm - lm.mean()) ** 2).sum()
            alpha = a / max(b, 1e-30)
        out[t: t + step] = alpha
    return out


def _first_persistent(z, th, k):
    """First bin where z exceeds th for k CONSECUTIVE bins, else -1.

    A single-sample spike is noise, not an early warning. Requiring persistence
    is the cheapest way to suppress the isolated excursions that dominate the
    false-alarm rate.
    """
    e = np.nan_to_num(z, nan=-np.inf) > th
    if not e.any():
        return -1
    # run-length: index i qualifies if e[i-k+1..i] all True
    c = np.zeros(len(e), dtype=np.int32)
    run = 0
    for i, val in enumerate(e):
        run = run + 1 if val else 0
        if run >= k:
            return int(i - k + 1)
    return -1


# --------------------------------------------------------------------------
def score_one(args_tuple):
    """Return per-log alarm evidence: the running aggregate anomaly score."""
    row, cfg = args_tuple
    try:
        x, v = decode(row)
    except Exception as e:
        return {"log_id": row["log_id"], "status": f"decode_error: {e}"[:120]}

    # Channel restriction (aggregation-width diagnostic). Slicing here, before
    # any statistic is computed, means the restricted arm differs from the
    # frozen one in EXACTLY ONE THING: which columns exist. The window,
    # warm-up, standardisation, persistence rule, threshold grid and top-k
    # aggregation are the same objects.
    ci = cfg.get("channel_idx")
    if ci is not None:
        if x.shape[1] != N_CHANNELS:
            return {"log_id": row["log_id"],
                    "status": f"channel_mismatch: {x.shape[1]} != {N_CHANNELS}"}
        x = x[:, ci]
        v = v[:, ci]

    T, C = x.shape
    W = cfg["window_bins"]
    warm = cfg["warmup_bins"]
    if T < warm + 2 * W:
        return {"log_id": row["log_id"], "status": "too_short", "n_bins": T}

    st = rolling_stats(x, v, W, cfg["min_count"])
    # Variance is positive and heavy-tailed over orders of magnitude, and its
    # baseline spread collapses to ~0 whenever a channel sits constant (idle
    # actuator, steady cpuload). Standardising it directly produced z-scores up
    # to 1e18. Work in log space so the indicator is scale-free and its
    # dispersion cannot degenerate.
    with np.errstate(divide="ignore", invalid="ignore"):
        logvar = np.log(st["var"] + 1e-12)
    inds = [logvar, st["ac1"], st["skew"]]
    names = ["logvar", "ac1", "skew"]
    # Per-indicator floor on the baseline dispersion, in the indicator's own
    # units. A single global 1e-9 floor is what let the degeneracy through.
    floors = [cfg["floor_logvar"], cfg["floor_ac1"], cfg["floor_skew"]]
    if cfg["dfa"]:
        inds.append(dfa_alpha(x, v, W))
        names.append("dfa")
        floors.append(cfg["floor_skew"])

    # ---- causal standardisation against an EXPANDING trailing baseline ----
    # baseline for time t uses [warmup, t - W] only: strictly the past, and
    # lagged by one window so the alarm window itself never enters its own
    # reference distribution.
    zmax = np.full(T, np.nan)
    for ind, floor in zip(inds, floors):
        a = ind.copy()
        a[~np.isfinite(a)] = np.nan
        good = np.isfinite(a)
        ag = np.where(good, a, 0.0)
        n = np.cumsum(good, axis=0).astype(np.float64)
        s1 = np.cumsum(ag, axis=0)
        s2 = np.cumsum(ag * ag, axis=0)
        lag = W
        nb = np.full_like(n, np.nan)
        s1b = np.full_like(s1, np.nan)
        s2b = np.full_like(s2, np.nan)
        nb[lag:] = n[:-lag]
        s1b[lag:] = s1[:-lag]
        s2b[lag:] = s2[:-lag]
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = s1b / np.maximum(nb, 1)
            sd = np.sqrt(np.maximum(s2b / np.maximum(nb, 1) - mu ** 2, 0.0))
            z = (a - mu) / np.maximum(sd, floor)
        z[nb < cfg["min_baseline"]] = np.nan
        # channel aggregation: k-th largest z, robust to a single noisy channel
        zs = np.sort(np.where(np.isfinite(z), z, -np.inf), axis=1)
        k = min(cfg["topk"], C)
        agg = zs[:, -k]
        agg[~np.isfinite(agg)] = np.nan
        zmax = np.fmax(zmax, agg) if np.any(np.isfinite(zmax)) else agg

    zmax[:warm] = np.nan
    finite = np.isfinite(zmax)
    out = {
        "log_id": row["log_id"],
        "status": "ok",
        "n_bins": T,
        "t_e_bin": int(row["t_e_bin"]),
        "score_max": float(np.nanmax(zmax)) if finite.any() else np.nan,
        # first crossing bin for each threshold in the sweep
        "cross": json.dumps({
            f"{th:g}": _first_persistent(zmax, th, cfg["persist_bins"])
            for th in cfg["thresholds"]}),
        "valid_bins": int(finite.sum()),
    }
    if cfg.get("dump_scores"):
        # Item 3b(i): persist the per-bin score SERIES so Tier-0 can be swept
        # over the same sequential-rule grid as the learned tiers. The cached
        # artifact above is first-CROSSING bins, which throws the series away and
        # so cannot be smoothed, accumulated or made persistent after the fact.
        #
        # Warm-up bins are dropped rather than emitted as NaN: they can never
        # alarm (`_first_persistent` maps NaN to -inf) and they would otherwise
        # poison CUSUM's cumsum. Interior NaNs -- bins whose expanding baseline
        # has fewer than `min_baseline` samples -- are kept and handled at the
        # consumer, which fills them with the negative-window mean, i.e. "no
        # evidence", before accumulating.
        out["scores"] = zmax[warm:].astype(np.float32)
        out["score_t0"] = int(warm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.join(ROOT, "proc", "curated-*.parquet"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "meta", "manifest.parquet"))
    ap.add_argument("--tag", default="curated")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--warmup-s", type=float, default=20.0)
    ap.add_argument("--topk", type=int, default=3)
    # Aggregation-width diagnostic (2026-09-02). See channels.CHANNEL_GROUPS for
    # why: van Beers et al. get a positive EWS result from ROTOR SPEED ALONE,
    # so a reviewer will ask whether our negative is a property of classical EWS
    # or of aggregating over all 87 channels. Restricting the channel set is the
    # single-variable control that answers it.
    #
    # ⚠ This does NOT re-select Tier-0. The frozen configuration (w=10 s, top-3,
    # all channels) was chosen by a pre-registered sweep and stays the headline;
    # any restricted arm is a POST-HOC DIAGNOSTIC and must be reported as one.
    # Always write it to a NEW --tag.
    ap.add_argument("--channels", default="all",
                    help="channel subset: a group name (%s) or a comma-separated "
                         "list of 'topic.field' prefixes. Default 'all' = the "
                         "frozen 87-channel vocabulary."
                         % ", ".join(CHANNEL_GROUPS))
    ap.add_argument("--dfa", action="store_true", help="include DFA (slow)")
    ap.add_argument("--dump-scores", action="store_true",
                    help="also persist the per-bin z-series to "
                         "meta/scores/ews_scores_<tag>.parquet (item 3b(i): lets "
                         "sequential.py sweep Tier-0 over the same rule grid). "
                         "~122M rows / ~1 GB for the full curated tier.")
    ap.add_argument("--limit", type=int, default=None)
    # The default grid stops at 80, and on `split_B_fixed` Tier-0's best operating
    # point lands exactly there -- at FAR 2.251/h, above FAR_CAP, i.e. ON THE EDGE
    # OF THE GRID (RESULTS.md §2.1). Two headline claims rest on that edge, so the
    # grid has to be extendable. Measured 2026-08-21: the aggregated z-score
    # saturates near 158 (99th pct of both arms), and above 200 essentially no
    # flight alarms at all, so there is no point going past ~200.
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated; default is the original 16-point grid. "
                         "Extending it changes `cross` and therefore every curve "
                         "derived from the traces -- write to a NEW --tag.")
    args = ap.parse_args()

    ci = channel_indices(args.channels)
    cfg = {
        "channel_idx": ci,
        "window_bins": int(args.window_s * TARGET_HZ),
        "warmup_bins": int(args.warmup_s * TARGET_HZ),
        "min_count": max(4, int(args.window_s * TARGET_HZ * 0.5)),
        "min_baseline": int(10 * TARGET_HZ),
        "floor_logvar": 0.25,   # nats; below this a log-variance baseline is degenerate
        "floor_ac1": 0.02,      # autocorrelation units
        "floor_skew": 0.05,
        "persist_bins": int(0.5 * TARGET_HZ),   # alarm must hold 0.5 s
        "topk": args.topk,
        "dfa": args.dfa,
        "dump_scores": args.dump_scores,
        "thresholds": ([float(x) for x in args.thresholds.split(",")]
                       if args.thresholds else
                       [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 80]),
    }
    print(f"config: {cfg}", flush=True)
    nch = N_CHANNELS if ci is None else len(ci)
    print(f"channels: {args.channels!r} -> {nch}/{N_CHANNELS}", flush=True)
    if ci is not None:
        print("  " + ", ".join(channel_names(ci)), flush=True)
        if args.topk > nch:
            sys.exit(f"--topk {args.topk} exceeds the {nch}-channel subset; "
                     f"the aggregate would silently fall back to the minimum "
                     f"z-score and would not be comparable to the frozen arm.")

    files = sorted(glob.glob(args.shards))
    print(f"{len(files)} shards", flush=True)

    recs = []
    t0 = time.time()
    for fi, f in enumerate(files, 1):
        rows = pq.read_table(f).to_pylist()
        if args.limit:
            rows = rows[: max(0, args.limit - len(recs))]
            if not rows:
                break
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(score_one, [(x, cfg) for x in rows], chunksize=4):
                recs.append(r)
        print(f"[shard {fi}/{len(files)}] {len(recs)} logs "
              f"{len(recs)/(time.time()-t0):.1f}/s", flush=True)

    if args.dump_scores:
        sp = os.path.join(ROOT, "meta", "scores", f"ews_scores_{args.tag}.parquet")
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        keep = [r for r in recs if r.get("status") == "ok" and "scores" in r]
        n = sum(len(r["scores"]) for r in keep)
        print(f"\nwriting {n:,} score rows for {len(keep):,} flights -> {sp}",
              flush=True)
        sdf = pd.DataFrame({
            "log_id": np.repeat([r["log_id"] for r in keep],
                                [len(r["scores"]) for r in keep]),
            "t_bin": np.concatenate([np.arange(r["score_t0"],
                                               r["score_t0"] + len(r["scores"]),
                                               dtype=np.int32) for r in keep]),
            "score": np.concatenate([r["scores"] for r in keep]),
        })
        # `y` is the FROZEN Tier-1 window label, so the threshold grid and the
        # CUSUM standardisation are built from the same definition of "negative"
        # the learned arms use. Anything outside the eval table falls to -1.
        ros = pd.read_parquet(
            os.path.join(ROOT, "meta", "splits_eval.parquet")
        ).set_index("log_id")
        sdf["y"] = window_labels(sdf.log_id.to_numpy(), sdf.t_bin.to_numpy(),
                                 ros).astype(np.int8)
        sdf.to_parquet(sp, index=False)
        print(f"wrote {sp}  ({os.path.getsize(sp)/1e9:.2f} GB)")
        for r in recs:
            r.pop("scores", None)

    df = pd.DataFrame(recs)
    out = os.path.join(ROOT, "meta", f"ews_traces_{args.tag}.parquet")
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    ok = df[df["status"] == "ok"]
    print(f"scored {len(ok)}/{len(df)}  "
          f"positives {int((ok['t_e_bin']>=0).sum())}  "
          f"negatives {int((ok['t_e_bin']<0).sum())}")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
