#!/usr/bin/env python
"""Tier 1 shallow baselines (PROJECT_PLAN §7): logistic regression and gradient
boosting on hand-crafted window statistics. Establishes how much of the problem
is trivially separable before any deep model is trained.

PROTOCOL — deliberately mirrors the Tier-0 EWS evaluation so numbers compare:
  - features: per-channel causal window stats (mean, log-variance, lag-1
    autocorrelation, skewness, valid-fraction) over a trailing 5 s window,
    computed at 1 s stride from the same shards, same 20 s warmup.
  - labels: trigger set v2 (meta/trigger_v2_exclude.csv logs dropped entirely).
    Positive training windows end 1-10 s before t_e; windows 10-30 s before are
    ambiguous and excluded; negative windows come only from no-trigger flights
    >= 120 s. Eval scores EVERY window of a test flight (t < t_e for positives).
  - flight-level eval: first score crossing per threshold, survival-hazard FAR
    on negatives (time at risk = time to first alarm, else scoreable duration),
    detection requires the crossing strictly before t_e, lead = (t_e - c)/20 Hz.
  - splits: A = GroupShuffleSplit on vehicle_key, 80/20, seeds 0/1/2.
            B = train multirotor mav_types, test fixed-wing/VTOL/helicopter.
  - operating point mirrors the sweep freeze rule: max detection s.t. FAR<=2/h.
    Reported alongside it is detection INTERPOLATED at exactly REF_FAR=1.6/h,
    which is the number to quote when comparing models -- see threshold_grid()
    for why the FAR<=2 point alone is not trustworthy for boosted trees.

RESUMABLE: feature extraction checkpoints per shard in meta/tier1_feats/; the
train/eval stage reruns from those in minutes. One-command progress:

    python tier1_baseline.py --status
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
from shard_io import decode
from channels import N_CHANNELS, TARGET_HZ

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")
FEATS = os.path.join(META, "tier1_feats")
EXCLUDE = os.path.join(META, "trigger_v2_exclude.csv")
HZ = TARGET_HZ

W = 100                 # 5 s feature window
STRIDE = 20             # 1 s
WARMUP = 400            # 20 s, matches EWS
MIN_COUNT = 50
MIN_BINS = 600          # matches EWS too_short rule (warmup + 2W)
NEG_MIN_BINS = 2400     # plan §5: negatives fly >= 120 s
POS_LO, POS_HI = 200, 20    # positive train windows end 1-10 s before t_e
AMBIG = 600                 # 10-30 s before t_e: excluded from training
FAR_CAP, REF_FAR = 2.0, 1.6
N_THRESH_TAIL = 90          # log-spaced ranks into the tail of negative scores
N_THRESH_BULK = 40          # linear quantiles over the bulk (see threshold_grid)
STATS = ["mean", "logvar", "ac1", "skew", "vfrac"]
N_FEAT = N_CHANNELS * len(STATS)

MULTI = {"Quadrotor", "Hexarotor", "Octorotor", "Tricopter", "Coaxial helicopter"}


def _is_winged(m):
    """Split B as originally built. KEPT VERBATIM so the frozen 0.360 reproduces.

    The string "Helicopter (tail rotor)" does not exist in this corpus -- the
    actual value is "Normal helicopter with tail rotor" (769 flights) -- so
    tail-rotor helicopters land in NEITHER arm of this split. That is a defect,
    not a design choice; see _is_winged_fixed. Do not "fix" this function: the
    frozen Tier-1 split-B number is defined by it.
    """
    m = str(m)
    return "VTOL" in m or m in ("Fixed Wing", "Helicopter (tail rotor)")


def _is_winged_fixed(m):
    """Split B as intended: non-multirotor test arm, helicopters included.

    Moves the eval tier's test arm 5,825 -> 6,594 flights and its positives
    1,409 -> 1,439. This is the paper's headline cross-airframe split as of
    2026-08-18 (PROJECT_PLAN §8); `B` is retained alongside it so nothing already
    computed silently moves.
    """
    m = str(m)
    return ("VTOL" in m
            or m in ("Fixed Wing", "Normal helicopter with tail rotor"))


# ---------------------------------------------------------------- features ----
def _csum(a):
    return np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)], axis=0)


def window_feats(x, v, ts):
    """Stats of the window (t-W, t] for each t in ts. Returns (len(ts), N_FEAT)."""
    xv = np.where(v, x, 0.0).astype(np.float64)
    vv = v.astype(np.float64)
    S0, S1, S2, S3 = _csum(vv), _csum(xv), _csum(xv * xv), _csum(xv ** 3)
    pair = np.zeros_like(xv); pv = np.zeros_like(vv)
    pair[1:] = xv[1:] * xv[:-1]; pv[1:] = vv[1:] * vv[:-1]
    P, PV = _csum(pair), _csum(pv)

    hi, lo = ts + 1, ts + 1 - W
    n = S0[hi] - S0[lo]
    s1, s2, s3 = S1[hi] - S1[lo], S2[hi] - S2[lo], S3[hi] - S3[lo]
    ps, pn = P[hi] - P[lo], PV[hi] - PV[lo]

    ok = n >= MIN_COUNT
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, s1 / np.maximum(n, 1), np.nan)
        var = np.maximum(np.where(ok, s2 / np.maximum(n, 1) - mean ** 2, np.nan), 0.0)
        sd = np.sqrt(var)
        logvar = np.log(var + 1e-12)
        m3 = s3 / np.maximum(n, 1) - 3 * mean * (s2 / np.maximum(n, 1)) + 2 * mean ** 3
        skew = np.clip(np.where(ok & (sd > 1e-12), m3 / np.maximum(sd ** 3, 1e-30),
                                np.nan), -20, 20)
        pmean = np.where(pn >= MIN_COUNT, ps / np.maximum(pn, 1), np.nan)
        ac1 = np.clip(np.where(ok & (var > 1e-12),
                               (pmean - mean ** 2) / np.maximum(var, 1e-30), np.nan),
                      -1.5, 1.5)
        vfrac = n / W
    return np.stack([mean, logvar, ac1, skew, vfrac], axis=2).reshape(
        len(ts), N_FEAT).astype(np.float32)


def process_shard(args_t):
    path, exclude = args_t
    base = os.path.basename(path).replace(".parquet", "")
    fout = os.path.join(FEATS, f"feat-{base}.parquet")
    rout = os.path.join(FEATS, f"roster-{base}.parquet")
    if os.path.exists(fout) and os.path.exists(rout):
        return base, -1
    frames, roster = [], []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64):
        for r in batch.to_pylist():
            if r["log_id"] in exclude or r["n_bins"] < MIN_BINS:
                continue
            t_e, n = int(r["t_e_bin"]), int(r["n_bins"])
            roster.append({"log_id": r["log_id"], "t_e_bin": t_e, "n_bins": n})
            end = min(t_e, n) if t_e >= 0 else n     # positives: strictly pre-event
            ts = np.arange(WARMUP, end, STRIDE)
            if not len(ts):
                continue
            try:
                x, v = decode(r)
            except Exception:
                roster.pop()
                continue
            f = window_feats(x, v, ts)
            df = pd.DataFrame(f, columns=[f"f{i:03d}" for i in range(N_FEAT)])
            df.insert(0, "t_bin", ts.astype(np.int32))
            df.insert(0, "log_id", r["log_id"])
            frames.append(df)
    pd.concat(frames, ignore_index=True).to_parquet(fout, index=False)
    pd.DataFrame(roster).to_parquet(rout, index=False)
    return base, len(roster)


# ------------------------------------------------------------------- eval ----
def threshold_grid(neg_scores):
    """Thresholds that actually resolve the LOW-FAR end of the curve.

    The original grid was 60 linearly-spaced quantiles over [0.5, 0.99995]. That
    leaves a hole exactly where the operating point lives: boosted-tree scores
    pile up in the tail, so consecutive quantiles jumped from FAR 2.10/h straight
    to 0.05/h and the "best point under FAR<=2" snapped to a useless corner
    (gbt_A_seed2 read 0.112 @ 0.053/h next to 0.747 @ 2.10/h). That is grid
    resolution masquerading as seed variance.

    Fix: place thresholds at ranks spaced logarithmically down from the top of
    the sorted negative scores, so the density is highest where few negatives
    alarm — i.e. where FAR is small. Keep a coarse quantile sweep over the bulk
    so the high-FAR end of the curve is still drawn.
    """
    s = np.sort(neg_scores)
    n = s.size
    if n < 4:
        return np.unique(s)
    ranks = np.unique(np.round(
        np.logspace(0, np.log10(max(n // 2, 3)), N_THRESH_TAIL)).astype(int))
    tail = s[n - ranks]                       # rank 1 = the single highest negative
    bulk = np.quantile(s, np.linspace(0.5, 0.99, N_THRESH_BULK))
    return np.unique(np.concatenate([bulk, tail]))


def flight_curve(scores_by_flight, roster, thresholds):
    """Lead-time-vs-FAR rows from per-flight score series (survival estimator)."""
    rows = []
    pos = roster[roster.t_e_bin >= 0]
    neg = roster[(roster.t_e_bin < 0) & (roster.n_bins >= NEG_MIN_BINS)]
    for th in thresholds:
        alarmed_h, n_alarm = 0.0, 0
        for lid, nb in zip(neg.log_id, neg.n_bins):
            t, s = scores_by_flight.get(lid, (None, None))
            c = -1
            if t is not None:
                hit = np.flatnonzero(s >= th)
                if len(hit):
                    c = t[hit[0]]
            if c >= 0:
                n_alarm += 1
                alarmed_h += (c - WARMUP) / HZ / 3600
            else:
                alarmed_h += (nb - WARMUP) / HZ / 3600
        far = n_alarm / alarmed_h if alarmed_h > 0 else np.nan
        det, leads = 0, []
        for lid, te in zip(pos.log_id, pos.t_e_bin):
            t, s = scores_by_flight.get(lid, (None, None))
            if t is None:
                continue                       # no scoreable window -> miss
            hit = np.flatnonzero(s >= th)
            if len(hit) and t[hit[0]] < te:
                det += 1
                leads.append((te - t[hit[0]]) / HZ)
        rows.append({"threshold": float(th), "far_per_hour": far,
                     "n_neg_alarmed": n_alarm,
                     "detection_rate": det / len(pos) if len(pos) else np.nan,
                     "n_detected": det,
                     "median_lead_s": float(np.median(leads)) if leads else np.nan,
                     "p25_lead_s": float(np.percentile(leads, 25)) if leads else np.nan,
                     "p75_lead_s": float(np.percentile(leads, 75)) if leads else np.nan})
    return pd.DataFrame(rows)


def op_point(curve):
    c = curve.dropna(subset=["far_per_hour"]).sort_values("far_per_hour")
    out = {}
    if not len(c):
        return out

    # (a) the freeze rule, mirroring Tier-0: max detection subject to FAR <= cap
    capped = c[c.far_per_hour <= FAR_CAP]
    if len(capped):
        b = capped.loc[capped.detection_rate.idxmax()]
        out = {"far_per_hour": float(b.far_per_hour),
               "detection_rate": float(b.detection_rate),
               "median_lead_s": float(b.median_lead_s), "threshold": float(b.threshold)}

    # (b) the cross-model comparable: detection at EXACTLY the reference FAR,
    # linearly interpolated along the curve. The old version snapped to the
    # nearest grid point, so the headline moved with grid placement rather than
    # with the model. Interpolating makes lr/gbt/Tier-0 directly comparable.
    f = c.far_per_hour.to_numpy(dtype=float)
    d = c.detection_rate.to_numpy(dtype=float)
    if f.min() <= REF_FAR <= f.max():
        out[f"det_at_{REF_FAR}h"] = float(np.interp(REF_FAR, f, d))
        lead = c.median_lead_s.to_numpy(dtype=float)
        m = np.isfinite(lead)
        if m.sum() >= 2 and f[m].min() <= REF_FAR <= f[m].max():
            out[f"lead_at_{REF_FAR}h"] = float(np.interp(REF_FAR, f[m], lead[m]))
    else:
        # curve never reaches the reference FAR; report the closest and say so
        out[f"det_at_{REF_FAR}h"] = float(d[np.abs(f - REF_FAR).argmin()])
        out["ref_far_extrapolated"] = True
    return out


def train_eval(df, roster, train_ids, test_ids, model_name, seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score

    fcols = [c for c in df.columns if c.startswith("f")]
    ros = roster.set_index("log_id")
    te = df.log_id.map(ros.t_e_bin)
    nb = df.log_id.map(ros.n_bins)
    is_pos_flight = te >= 0
    y = np.where(is_pos_flight & (df.t_bin >= te - POS_LO) & (df.t_bin <= te - POS_HI), 1,
                 np.where(~is_pos_flight & (nb >= NEG_MIN_BINS), 0, -1))

    tr = df.log_id.isin(train_ids) & (y >= 0)
    # cap negatives so LR stays tractable; positives are all kept
    rng = np.random.default_rng(seed)
    neg_idx = np.flatnonzero(tr & (y == 0))
    if len(neg_idx) > 400_000:
        drop = rng.choice(neg_idx, len(neg_idx) - 400_000, replace=False)
        tr = tr.copy(); tr.iloc[drop] = False
    Xtr, ytr = df.loc[tr, fcols].to_numpy(), y[tr.to_numpy()]

    # HistGBT's binner raises "window shape cannot be larger than input array
    # shape" on a column it sees as entirely NaN: _find_binning_thresholds
    # guards len(distinct_values) == 1 but not == 0, and then calls
    # sliding_window_view(distinct_values, 2) on an empty array. Two ways a
    # column gets there, and the second is why the first fix was not enough:
    #   1. the column is all-NaN in the fold (a channel absent from every
    #      multirotor, i.e. every training vehicle under split B);
    #   2. the column has only a handful of non-NaN rows, and _BinMapper's
    #      200k-row subsample -- drawn WITH REPLACEMENT -- misses all of them.
    # (2) is intermittent, which is why split A survived and split B did not.
    # P(miss) = (1 - k/n)^200000 ~ exp(-k/2) at n = 400k, so a floor of 200
    # non-NaN rows makes it impossible in practice. Columns that sparse carry
    # no usable signal anyway. Applied to both models so lr and gbt keep an
    # identical feature set.
    MIN_FINITE = 200
    finite = np.isfinite(Xtr)
    keep = np.array([finite[:, j].sum() >= MIN_FINITE
                     and np.unique(Xtr[finite[:, j], j]).size >= 2
                     for j in range(Xtr.shape[1])])
    if not keep.all():
        print(f"    dropped {int((~keep).sum())} degenerate feature columns",
              flush=True)
    Xtr = Xtr[:, keep]

    if model_name == "lr":
        clf = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                            StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                             random_state=seed)
    clf.fit(Xtr, ytr)

    ev = df.log_id.isin(test_ids)
    sc = clf.predict_proba(df.loc[ev, fcols].to_numpy()[:, keep])[:, 1]
    sub = df.loc[ev, ["log_id", "t_bin"]].assign(score=sc)
    yw = y[ev.to_numpy()]
    auprc = average_precision_score(yw[yw >= 0], sc[yw >= 0]) if (yw == 1).any() else np.nan

    scores_by_flight = {lid: (g.t_bin.to_numpy(), g.score.to_numpy())
                        for lid, g in sub.groupby("log_id", sort=False)}
    curve = flight_curve(scores_by_flight, roster[roster.log_id.isin(test_ids)],
                         threshold_grid(sc[(yw == 0)]))
    return curve, {"window_auprc": float(auprc), **op_point(curve)}


def cached_run(df, roster, tr_ids, te_ids, model, split, seed):
    """One fit/eval, checkpointed. A restart re-uses any run that finished, so a
    crash late in the grid costs one run, not the whole hour."""
    tag = f"{model}_{split}" + (f"_seed{seed}" if split == "A" else "")
    cpath = os.path.join(META, f"tier1_curve_{tag}.csv")
    ppath = os.path.join(META, f"tier1_pt_{tag}.json")
    if os.path.exists(ppath) and os.path.exists(cpath):
        with open(ppath) as f:
            pt = json.load(f)
        print(f"  {tag}: cached {pt}", flush=True)
        return pt
    t1 = time.time()
    curve, pt = train_eval(df, roster, tr_ids, te_ids, model, seed)
    curve.to_csv(cpath, index=False)
    pt.update(model=model, split=split, seed=seed,
              minutes=round((time.time() - t1) / 60, 1))
    with open(ppath, "w") as f:
        json.dump(pt, f, default=float)
    print(f"  {tag}: {pt}", flush=True)
    return pt


# ------------------------------------------------------------------- main ----
def status():
    n = len(glob.glob(os.path.join(FEATS, "feat-*.parquet")))
    total = len(glob.glob(os.path.join(ROOT, "proc", "curated-*.parquet")))
    print(f"tier1 status  {time.strftime('%H:%M:%S')}")
    print(f"  features: {n}/{total} shards")
    sp = os.path.join(META, "tier1_summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            s = json.load(f)
        for k, v in s["headline"].items():
            print(f"  {k}: {v}")
    else:
        print("  train/eval: not finished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    if args.status:
        status()
        return

    os.makedirs(FEATS, exist_ok=True)
    exclude = set(pd.read_csv(EXCLUDE)["log_id"])
    shards = sorted(glob.glob(os.path.join(ROOT, "proc", "curated-*.parquet")))
    print(f"{len(shards)} shards, {len(exclude)} v2-excluded logs", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for base, nlog in pool.imap_unordered(
                process_shard, [(s, exclude) for s in shards]):
            print(f"  [{(time.time()-t0)/60:5.1f} min] {base}: "
                  f"{'cached' if nlog < 0 else f'{nlog} logs'}", flush=True)

    print("loading features...", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob(os.path.join(FEATS, "feat-*.parquet")))],
                   ignore_index=True)
    roster = pd.concat([pd.read_parquet(f) for f in
                        sorted(glob.glob(os.path.join(FEATS, "roster-*.parquet")))],
                       ignore_index=True).drop_duplicates("log_id")
    man = pd.read_parquet(os.path.join(META, "manifest.parquet"),
                          columns=["log_id", "mav_type", "vehicle_key"])
    roster = roster.merge(man, on="log_id", how="left")
    roster["vehicle_key"] = roster.vehicle_key.fillna(roster.log_id)
    print(f"{len(df)} windows from {len(roster)} flights "
          f"({(roster.t_e_bin >= 0).sum()} positives)", flush=True)

    from sklearn.model_selection import GroupShuffleSplit
    summary = {"headline": {}, "runs": []}
    winged = roster.mav_type.map(_is_winged)
    winged_fixed = roster.mav_type.map(_is_winged_fixed)

    for model in ("lr", "gbt"):
        # Split A: vehicle-grouped 80/20, 3 seeds
        pts = []
        for seed in (0, 1, 2):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr_i, te_i = next(gss.split(roster, groups=roster.vehicle_key))
            tr_ids = set(roster.log_id.iloc[tr_i])
            te_ids = set(roster.log_id.iloc[te_i])
            pt = cached_run(df, roster, tr_ids, te_ids, model, "A", seed)
            summary["runs"].append(pt)
            pts.append(pt)
        for key in (f"det_at_{REF_FAR}h", f"lead_at_{REF_FAR}h",
                    "detection_rate", "median_lead_s", "window_auprc"):
            vals = [p[key] for p in pts if key in p and np.isfinite(p.get(key, np.nan))]
            if vals:
                summary["headline"][f"{model}_A_{key}"] = (
                    f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}")

        # Split B: multirotor -> winged. Both definitions are run: "B" is the
        # as-built split the frozen 0.360 belongs to, "B_fixed" is the headline
        # (adds tail-rotor helicopters to the test arm; PROJECT_PLAN §8).
        tr_ids = set(roster.log_id[roster.mav_type.isin(MULTI)])
        for bname, mask in (("B", winged), ("B_fixed", winged_fixed)):
            te_ids = set(roster.log_id[mask])
            pt = cached_run(df, roster, tr_ids, te_ids, model, bname, 0)
            summary["runs"].append(pt)
            for key in (f"det_at_{REF_FAR}h", f"lead_at_{REF_FAR}h",
                        "detection_rate", "median_lead_s", "window_auprc"):
                if key in pt and np.isfinite(pt.get(key, np.nan)):
                    summary["headline"][f"{model}_{bname}_{key}"] = f"{pt[key]:.3f}"

    summary["config"] = {"W": W, "stride": STRIDE, "warmup": WARMUP,
                         "pos_window_s": [POS_HI / HZ, POS_LO / HZ],
                         "labelset": "v2", "far_cap": FAR_CAP,
                         "note": "detection_rate/median_lead_s are at the best "
                                 f"operating point with FAR<={FAR_CAP}/h, "
                                 "mirroring the Tier-0 freeze rule"}
    with open(os.path.join(META, "tier1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote meta/tier1_summary.json  (total {(time.time()-t0)/60:.0f} min)")
    for k, v in summary["headline"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
