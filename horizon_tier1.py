#!/usr/bin/env python
"""THE SHIFTED-BAND TIER-1 CONTROL. Does the dominance come from the learned
representation, or just from moving the training label band?

WHY THIS FILE EXISTS
--------------------
`RESULTS.md` §4.5/§4.6 report the project's result: cross-airframe, a frozen-encoder
linear probe trained on a band 111-150 s before the event dominates classical EWS on
BOTH axes (e.g. `shift` 120 s + `kofm` 2/3s: det 0.234 / lead 92.5 s against Tier-0's
best-of-11 at 0.155 / 54.4 s).

But that arm changed TWO things at once relative to the Tier-1 bar:

  1. the FEATURES        hand-built window statistics  ->  pretrained embeddings
  2. the TRAINING BAND   1-10 s before t_e             ->  111-120 s before t_e

Nothing in the project has ever varied (2) while holding (1) fixed. So the reviewer's
question -- "would hand-built statistics trained on the same shifted band dominate
Tier-0 too, with no pretraining at all?" -- is currently unanswered, and the
representation claim rests on that answer.

This file answers it. It is `horizon.py` with exactly one substitution: the feature
matrix. Everything else is imported rather than restated --

    band_bins, train_labels   from horizon.py    (identical band semantics)
    window_labels, score_report, roster_for
                              from eval_common   (identical frozen eval + alarm grid)
    splits_eval.parquet                          (identical train/test partitions)
    NEG_CAP = 400_000                            (identical negative subsample)

-- so a difference in the output is a difference in the features and nothing else.

READ THE OUTCOME THIS WAY
-------------------------
  Tier-1 + shifted band DOES dominate Tier-0  ->  the result is the LABEL BAND, and
      §4.5/§4.6 must be reframed: the finding is "train further back", not "pretrain".
      That is still a real and publishable finding, but it is a different paper.
  Tier-1 + shifted band does NOT dominate     ->  the band is necessary but not
      sufficient, and the representation is doing the work. §4.1's linear-vs-linear
      claim then extends to the dominance result, which is what the paper needs.

Either way it is the cheapest decisive experiment available (CPU, ~1 h) and it closes
the largest hole in the headline claim.

THE CONTROL, AND WHY IT IS NOT OPTIONAL
---------------------------------------
`widen` h = 10 s IS the original Tier-1 band, so those two arms must land on §3's
frozen numbers: lr 0.202, gbt 0.364 on `B_fixed`. That is the same reproduction gate
`horizon.py` uses (`widen` 10 must return the frozen probe's 0.334) and the same one
`sequential.check_raw` uses. If the control misses, the partition or the roster moved
and NOTHING else in this file may be quoted. The delta is computed, printed and
persisted into every arm's JSON.

One deliberate difference from `horizon.py`'s head, forced by the features: Tier-1
statistics contain NaN (a channel absent from a flight) where embeddings do not, so
the `lr` pipeline carries `SimpleImputer(constant 0)` ahead of the scaler, exactly as
`tier1_baseline.train_eval` does. The degenerate-column drop is ported for the same
reason -- HistGBT's binner crashes on an all-NaN or near-empty column.

Usage
-----
    bash run_h1.sh start          # the default grid (14 arms), detached
    bash run_h1.sh status
    python horizon_tier1.py --list
    python horizon_tier1.py --split B_fixed --mode shift --h 120 --head gbt
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier1_baseline import REF_FAR, HZ, FEATS, N_FEAT
from eval_common import window_labels, score_report, roster_for
from horizon import band_bins, train_labels, MAX_ITER
from probe import NEG_CAP

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")

# Ported verbatim from tier1_baseline.train_eval. HistGBT's _BinMapper raises
# "window shape cannot be larger than input array shape" on a column it sees as
# all-NaN, and its 200k-row subsample is drawn WITH REPLACEMENT so a very sparse
# column can vanish intermittently. A floor of 200 finite rows makes that
# impossible in practice; such columns carry no signal anyway. Applied to BOTH
# heads so lr and gbt keep an identical feature set, as Tier-1 does.
MIN_FINITE = 200

# The frozen Tier-1 `B_fixed` numbers the `widen` 10 s control must reproduce
# (RESULTS.md §3, meta/tier1_pt_{lr,gbt}_B_fixed.json).
CONTROL_REF = {"lr": 0.2022307030976344, "gbt": 0.36398595492299574}
CONTROL_TOL = 0.02      # absolute det@1.6/h; far inside seed noise, wide enough
                        # to pass a partition that differs only in flight count


def load_tier1_features():
    """-> (log_id, t_bin, X float32 (n, N_FEAT)) on Tier-1's own window grid.

    Shard order is `sorted(glob(...))`, matching `tier1_baseline.main`, so the row
    order -- and therefore the negative subsample RNG draw -- is reproducible.

    Built column-wise into a preallocated array rather than by `pd.concat` of seven
    frames: the concat would transiently hold two copies of a 10.6 GiB matrix, and
    this box is shared.
    """
    parts = sorted(glob.glob(os.path.join(FEATS, "feat-*.parquet")))
    if not parts:
        raise SystemExit(f"no Tier-1 features in {FEATS}\n"
                         f"  run:  python tier1_baseline.py")
    n = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    fcols = [f"f{i:03d}" for i in range(N_FEAT)]
    X = np.empty((n, N_FEAT), dtype=np.float32)
    lids, tbs, off = [], [], 0
    for p in parts:
        t = pq.read_table(p)
        k = t.num_rows
        lids.append(np.asarray(t["log_id"].to_pylist(), dtype=object))
        tbs.append(t["t_bin"].to_numpy())
        for j, c in enumerate(fcols):
            X[off:off + k, j] = t[c].to_numpy(zero_copy_only=False)
        off += k
        del t
        print(f"  loaded {os.path.basename(p)}  {k:,}", flush=True)
    assert off == n
    return np.concatenate(lids), np.concatenate(tbs), X


def fit_tier1_head(Xtr, ytr, head, seed):
    """Tier-1's own two models, unchanged except for the lr iteration cap.

    `max_iter=MAX_ITER` (5000) rather than Tier-1's 1000 so these are converged fits
    and therefore comparable to the `_it5000` probe arms in §4.1 and to every
    `horizon.py` arm, which all use 5000. See probe.LR_MAX_ITER for why that
    distinction cost a whole grid once already.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier

    if head == "lr":
        clf = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                            StandardScaler(),
                            LogisticRegression(max_iter=MAX_ITER,
                                               class_weight="balanced",
                                               random_state=seed))
    elif head == "gbt":
        clf = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                             random_state=seed)
    else:
        raise SystemExit(f"unknown head {head}")
    return clf.fit(Xtr, ytr)


def arm_tag(mode, h_s, split, seed, head, width_s=None):
    t = f"{head}_{mode}{h_s}"
    if width_s is not None:
        t += f"w{width_s:g}"
    t += f"_{split}"
    if split == "A" or seed != 0:
        t += f"_seed{seed}"
    return t


def run_one(log_id, t_bin, X, y_eval, splits, ros, split, seed, mode, h_s, head,
            width_s=None, dump_scores=False):
    tag = arm_tag(mode, h_s, split, seed, head, width_s)
    ppath = os.path.join(META, f"h1_pt_{tag}.json")
    cpath = os.path.join(META, f"h1_curve_{tag}.csv")
    spath = os.path.join(META, "scores", f"h1_scores_{tag}.parquet")
    have_scores = (not dump_scores) or os.path.exists(spath)
    if os.path.exists(ppath) and os.path.exists(cpath) and have_scores:
        with open(ppath) as f:
            pt = json.load(f)
        print(f"  {tag}: cached  det@{REF_FAR}h "
              f"{pt.get(f'det_at_{REF_FAR}h', float('nan')):.3f}", flush=True)
        return pt

    lo, hi = band_bins(mode, h_s, width_s)
    t0 = time.time()
    y_tr = train_labels(log_id, t_bin, ros, lo, hi)

    col = {"A": f"split_A_s{seed}", "B": "split_B", "B_fixed": "split_B_fixed",
           "C": "split_C_cond"}[split]
    tr_ids = set(splits.loc[splits[col] == "train", "log_id"])
    te_ids = set(splits.loc[splits[col] == "test", "log_id"])
    in_tr = pd.Series(log_id).isin(tr_ids).to_numpy()
    in_te = pd.Series(log_id).isin(te_ids).to_numpy()

    tr = in_tr & (y_tr >= 0)
    rng = np.random.default_rng(seed)
    neg_idx = np.flatnonzero(tr & (y_tr == 0))
    if len(neg_idx) > NEG_CAP:
        tr[rng.choice(neg_idx, len(neg_idx) - NEG_CAP, replace=False)] = False
    n_pos = int((y_tr[tr] == 1).sum())
    if n_pos == 0:
        print(f"  {tag}: no positive training windows -- skipped", flush=True)
        return None

    Xtr = X[tr]
    finite = np.isfinite(Xtr)
    keep = np.array([finite[:, j].sum() >= MIN_FINITE
                     and np.unique(Xtr[finite[:, j], j]).size >= 2
                     for j in range(Xtr.shape[1])])
    del finite
    if not keep.all():
        print(f"    dropped {int((~keep).sum())} degenerate feature columns",
              flush=True)

    clf = fit_tier1_head(Xtr[:, keep], y_tr[tr], head, seed)
    del Xtr
    sc = clf.predict_proba(X[in_te][:, keep])[:, 1]

    # SCORED AGAINST THE FROZEN EVALUATION LABELS, never y_tr -- the rule
    # horizon.py exists to respect, restated here because it is the one thing that
    # would silently make this control meaningless.
    pt = score_report(log_id[in_te], t_bin[in_te], sc, y_eval[in_te],
                      roster_for(splits, te_ids), curve_path=cpath)
    if dump_scores:
        os.makedirs(os.path.dirname(spath), exist_ok=True)
        pd.DataFrame({"log_id": log_id[in_te], "t_bin": t_bin[in_te],
                      "score": sc.astype(np.float32),
                      "y": y_eval[in_te].astype(np.int8)}).to_parquet(spath,
                                                                     index=False)
        print(f"    scores -> {os.path.basename(spath)} "
              f"({int(in_te.sum()):,} rows)", flush=True)

    est = clf[-1] if head == "lr" else clf
    n_it = getattr(est, "n_iter_", None)
    if n_it is not None:
        n_it = int(np.max(n_it))
    pt.update(mode=mode, h_s=h_s, width_s=(lo - hi) / HZ,
              band_lo_bins=lo, band_hi_bins=hi,
              band_lo_s=lo / HZ, band_hi_s=hi / HZ,
              split=split, seed=seed, head=head, features="tier1_stats",
              n_features_kept=int(keep.sum()), n_features=int(len(keep)),
              n_train_windows=int(tr.sum()), n_train_pos=n_pos,
              n_train_pos_flights=int(pd.Series(log_id[tr & (y_tr == 1)]).nunique()),
              n_iter=n_it, max_iter=(MAX_ITER if head == "lr" else 300),
              minutes=round((time.time() - t0) / 60, 1))

    # The reproduction gate travels with the number, not with the log.
    if mode == "widen" and h_s == 10 and split == "B_fixed" and head in CONTROL_REF:
        d = pt.get(f"det_at_{REF_FAR}h", float("nan")) - CONTROL_REF[head]
        pt["control_delta_det"] = float(d)
        pt["control_pass"] = bool(abs(d) <= CONTROL_TOL)
        print(f"    CONTROL vs frozen Tier-1 {head} {CONTROL_REF[head]:.3f}: "
              f"delta {d:+.4f}  {'PASS' if abs(d) <= CONTROL_TOL else 'FAIL'}",
              flush=True)

    with open(ppath, "w") as f:
        json.dump(pt, f, default=float, indent=1)
    print(f"  {tag}: det@{REF_FAR}h {pt.get(f'det_at_{REF_FAR}h', float('nan')):.3f}  "
          f"lead {pt.get(f'lead_at_{REF_FAR}h', float('nan')):.1f}s  "
          f"auprc {pt.get('window_auprc', float('nan')):.4f}  "
          f"pos {n_pos:,}  {pt['minutes']}min", flush=True)
    return pt


def grid():
    """The control grid: the two dominating `B_fixed` arms, both heads, n = 3.

    Mirrors exactly the arms RESULTS.md §4.5/§4.6 quote (`shift` 120 s and
    `shift` 150 s on B_fixed at seeds 0/1/2), plus the two `widen` 10 s
    reproduction gates. Ordered so the gates run FIRST -- if the partition moved,
    that is visible in the log within a few minutes rather than after an hour.
    """
    jobs = [("B_fixed", 0, "widen", 10, "lr", None),      # gate vs 0.202
            ("B_fixed", 0, "widen", 10, "gbt", None)]     # gate vs 0.364
    for head in ("lr", "gbt"):
        for h in (120, 150):
            for seed in (0, 1, 2):
                jobs.append(("B_fixed", seed, "shift", h, head, None))
    return jobs


def grid_A():
    """Secondary: the same question in-distribution, on split A's `shift` 240 s.

    Split A is closed for the learned model (0/30 vs Tier-0), so this cannot
    produce a win -- it tests whether the split-A DETECTION doubling (0.287 vs
    0.143) is also band-driven rather than representation-driven. Cheap, and it
    makes the reframing argument symmetric if the B_fixed control comes back hot.
    """
    return [("A", sd, "shift", 240, head, None)
            for head in ("lr", "gbt") for sd in (0, 1, 2)]


def grid_C():
    """The feature control for split C, mirroring `horizon.grid_C` arm for arm.

    Same purpose as the B_fixed control: C's dominance rows must not be allowed to
    read as a representation result until hand-built statistics have been run
    under the identical band. Tier-0's best-of-11 bar on C is plain `raw` at
    det 0.180 / lead 51.5 s -- the hardest detection bar of the three splits.

    Note `widen` 10 s here is NOT a reproduction gate, because `tier1_baseline`
    never ran split C at all (RESULTS.md 4.1 shows "--" for Tier-1 on C). These
    arms ESTABLISH the missing Tier-1 baseline for that split rather than
    reproducing one, which is worth having on its own.
    """
    jobs = [("C", sd, "widen", 10, hd, None)
            for hd in ("lr", "gbt") for sd in (0, 1, 2)]
    for hd in ("lr", "gbt"):
        for h in (120, 150):
            for sd in (0, 1, 2):
                jobs.append(("C", sd, "shift", h, hd, None))
    return jobs


def summarise(df):
    keep = [c for c in ("head", "mode", "h_s", "split", "seed", "n_train_pos",
                        f"det_at_{REF_FAR}h", f"lead_at_{REF_FAR}h", "window_auprc",
                        "control_delta_det", "control_pass") if c in df.columns]
    pd.set_option("display.width", 220)
    print("\n=== Tier-1 shifted-band control ===")
    print(df.sort_values(["split", "head", "mode", "h_s", "seed"])[keep]
            .round(4).to_string(index=False))

    # The comparison the file was written for, laid out so it cannot be misread.
    # Tier-0's `B_fixed` best-of-11 bar (RESULTS.md §4.6, meta/sequential.csv).
    BAR_DET, BAR_LEAD = 0.154523, 54.423349
    # NB: df["mode"] not df.mode -- the latter is DataFrame.mode(), the method,
    # and comparing it to a string silently yields False for every row, which
    # emptied this whole block on the first run.
    sub = df[(df["split"] == "B_fixed") & (df["mode"] == "shift")]
    if len(sub):
        print(f"\nvs Tier-0 best-of-11 on B_fixed (det {BAR_DET:.3f} / "
              f"lead {BAR_LEAD:.1f} s), raw alarm rule:")
        g = sub.groupby(["head", "h_s"]).agg(
            n=(f"det_at_{REF_FAR}h", "size"),
            det=(f"det_at_{REF_FAR}h", "mean"), det_sd=(f"det_at_{REF_FAR}h", "std"),
            lead=(f"lead_at_{REF_FAR}h", "mean"),
            lead_sd=(f"lead_at_{REF_FAR}h", "std")).reset_index()
        g["dominates"] = (g.det > BAR_DET) & (g.lead > BAR_LEAD)
        print(g.round(4).to_string(index=False))
        print("\nTier-2 frozen-probe reference, same bands, same bar (RESULTS.md §4.6):")
        print("  shift 120 s  det 0.239 +/- 0.002   lead 78.8 +/- 4.7 s   DOMINATES")
        print("  shift 150 s  det 0.175 +/- 0.007   lead 90.7 +/- 0.0 s   DOMINATES")
        print("\nIf a `dominates` row above is True, the dominance result is the BAND,")
        print("not the representation, and RESULTS.md 4.5/4.6 must be reframed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["A", "B", "B_fixed", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    ap.add_argument("--mode", choices=["widen", "shift"])
    ap.add_argument("--h", type=int, help="horizon in seconds")
    ap.add_argument("--head", default="lr", choices=["lr", "gbt"])
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--grid", choices=["default", "A", "C"], default="default")
    ap.add_argument("--dump-scores", action="store_true",
                    help="persist per-window test scores to meta/scores/ so the "
                         "sequential rules of 4.6 can be run on the control too")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.grid == "A":
        jobs = grid_A()
    elif args.grid == "C":
        jobs = grid_C()
    elif args.split is None:
        jobs = grid()
    elif args.seeds:
        jobs = [(args.split, int(sd), args.mode, args.h, args.head, args.width)
                for sd in args.seeds.split(",")]
    else:
        jobs = [(args.split, args.seed, args.mode, args.h, args.head, args.width)]

    if args.list:
        print(f"{len(jobs)} jobs")
        for s, sd, m, h, hd, w in jobs:
            lo, hi = band_bins(m, h, w)
            print(f"  {hd:3s}  {m:5s} h={h:3d}s  split {s:8s} seed {sd}  "
                  f"band = t_e-{lo/HZ:.0f}s .. t_e-{hi/HZ:.0f}s")
        return

    if args.status:
        rows = []
        for s, sd, m, h, hd, w in jobs:
            p = os.path.join(META, f"h1_pt_{arm_tag(m, h, s, sd, hd, w)}.json")
            if os.path.exists(p):
                with open(p) as f:
                    rows.append(json.load(f))
        print(f"{len(rows)} of {len(jobs)} arms cached")
        if rows:
            summarise(pd.DataFrame(rows))
        return

    splits = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))
    ros = splits.set_index("log_id")[["t_e_bin", "n_bins"]]

    print("loading Tier-1 features...", flush=True)
    log_id, t_bin, X = load_tier1_features()
    print(f"  X = {X.shape}  ({X.nbytes/2**30:.1f} GiB)", flush=True)
    y_eval = window_labels(log_id, t_bin, ros)     # FROZEN -- scoring only
    print(f"  eval windows: {len(y_eval):,}  pos {int((y_eval==1).sum()):,}  "
          f"neg {int((y_eval==0).sum()):,}  dropped {int((y_eval==-1).sum()):,}",
          flush=True)

    out = []
    for split, seed, mode, h, head, width in jobs:
        pt = run_one(log_id, t_bin, X, y_eval, splits, ros, split, seed, mode, h,
                     head, width_s=width, dump_scores=args.dump_scores)
        if pt:
            out.append(pt)
    if out:
        summarise(pd.DataFrame(out))


if __name__ == "__main__":
    main()
