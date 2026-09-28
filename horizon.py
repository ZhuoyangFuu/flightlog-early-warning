#!/usr/bin/env python
"""Item 2: does the TRAINING label band determine lead time? (PROJECT_PLAN §7.5, §11.0.1)

§7.5 finding 2: Tier-1 GBT on split A, thresholded down to Tier-0's 18.6%
detection rate, fires **5.5 s** before the event against Tier-0's 66.0 s. It was
trained on `POS_LO, POS_HI = 200, 20` -- windows ending **1-10 s before t_e** --
and is doing exactly what it was taught. Every supervised tier in this project is
optimised for imminence and then scored on earliness.

This file tests that directly: refit the frozen-encoder linear probe on cached
embeddings under **training** bands that reach further back, and read the lead
time off the same frontier. If lead is band-determined, it moves. If it does not
move, the signal genuinely is not there at long horizons and §7.5's whole
premise is wrong -- which is equally worth knowing, and cheaply.

THE PROTOCOL RULE THIS FILE EXISTS TO RESPECT
---------------------------------------------
**The training band varies. The EVALUATION band and alarm grid do not.**
Scoring goes through `eval_common.score_report` with `window_labels` untouched,
so every arm here stays directly comparable to Tier-0, Tier-1, and every number
already in `RESULTS.md`. Only which windows the head is *fitted* on changes.
Changing the eval band would make every arm incomparable to everything and is
the single easiest way to fake a win on this task.

Two band families, because they answer different questions
----------------------------------------------------------
  widen  positives = every window ending in (t_e - h, t_e - 1 s].
         "Fire any time in the last h seconds." Practical: it is what you would
         actually deploy, and it multiplies the scarce positive count.
  shift  positives = windows ending in (t_e - h, t_e - (h - 10 s)] only.
         A narrow band at *distance* h. Scientific: is there separable signal
         60 s out at all, independent of what happens at 5 s?

`widen` at h = 10 s reproduces the existing band exactly, so it is the built-in
reproduction control: split A seed 0 must return det@1.6 = 0.672 and B_fixed
0.334 (frontier.py / RESULTS.md §4.1). **If the control does not reproduce,
nothing else in this file means anything.**

Usage
-----
    bash run_horizon.sh start          # the default grid, detached
    bash run_horizon.sh status
    python horizon.py --list           # what the grid is, without running it
    python horizon.py --mode widen --h 60 --split B_fixed
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier1_baseline import REF_FAR, POS_LO, POS_HI, NEG_MIN_BINS, HZ
from eval_common import window_labels, score_report, roster_for
from probe import load_embeddings, fit_head, NEG_CAP

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")

# lbfgs cap. The 2026-08-19 grid at 1000 hit the cap unconverged everywhere and
# had to be refit; start where that refit ended so these are converged fits and
# comparable to the `_it5000` probe numbers rather than to the superseded ones.
MAX_ITER = 5000

# Horizons in seconds. The §7.2 ablation proposed h in {1,2,5,10}s, which is
# under-ranged for this question: Tier-0's median lead is 66.0 s, so a grid that
# stops at 10 s cannot possibly reach it. These bracket it on both sides.
HORIZONS_WIDEN = [10, 30, 60, 120]
HORIZONS_SHIFT = [30, 60, 120]
SPLITS = [("A", 0), ("B_fixed", 0)]


BASE_WIDTH = POS_LO - POS_HI        # 180 bins = 9 s, the original band's width


def band_bins(mode, h_s, width_s=None):
    """(lo, hi) in BINS: positive iff t_e - lo <= t_bin <= t_e - hi.

    Mirrors `eval_common.window_labels`' convention exactly -- lo is the FAR edge
    (further before the event), hi the NEAR edge -- so `widen` at h=10 returns
    (POS_LO, POS_HI) unchanged and the control is a true reproduction.

    `width_s` applies to `shift` only and defaults to the original band's width
    (BASE_WIDTH = 9 s). Widening the shifted band is item 2c's second question:
    `shift` trades detection for lead (0.334 -> 0.239 on B_fixed), and a wider band
    at the same distance is the obvious way to buy some of that detection back
    without giving up the horizon.
    """
    lo = int(round(h_s * HZ))
    if mode == "widen":
        return lo, POS_HI
    if mode == "shift":
        w = BASE_WIDTH if width_s is None else int(round(width_s * HZ))
        hi = lo - w
        if hi <= 0:
            raise SystemExit(f"shift h={h_s}s width={w/HZ}s reaches past t_e")
        return lo, hi
    raise SystemExit(f"unknown mode {mode}")


def train_labels(log_id, t_bin, ros, lo, hi):
    """`window_labels` with a parameterised POSITIVE band, for FITTING only.

    Identical in every other respect, and deliberately so: 1 = inside the band,
    0 = a window of a negative flight that flew long enough to count, -1 =
    everything else and it must be DROPPED. Windows of a positive flight outside
    the band stay ambiguous-not-safe at every horizon -- widening the band must
    not quietly reclassify them as negatives, which would change the negative
    distribution between arms and make the sweep uninterpretable.
    """
    te = pd.Series(log_id).map(ros.t_e_bin).to_numpy()
    nb = pd.Series(log_id).map(ros.n_bins).to_numpy()
    is_pos = te >= 0
    return np.where(is_pos & (t_bin >= te - lo) & (t_bin <= te - hi), 1,
                    np.where(~is_pos & (nb >= NEG_MIN_BINS), 0, -1))


FEATURE_TAG = os.environ.get("FLIGHTLOG_FEATURE_TAG", "")


def arm_tag(mode, h_s, split, seed, width_s=None, head="lr"):
    """Cache tag for one arm.

    Split A materialises a different train/test partition per seed
    (`split_A_s{seed}`), so its seed has always been in the tag. B_fixed/B/C have
    ONE fixed partition, but the seed still changes the fit -- it drives the
    negative subsample down to NEG_CAP and the head's `random_state`. Seed 0 keeps
    its historical bare tag so the 2026-08-21 grid stays cached; any other seed
    gets its own entry rather than silently overwriting it.

    `head` is suffixed ONLY when it is not the historical `lr`, for the same
    reason: every arm scored before 2026-08-31 was linear and its tag must not
    move, or the whole cached grid silently refits.
    """
    t = f"{mode}{h_s}"
    if width_s is not None:
        t += f"w{width_s:g}"
    t += f"_{split}"
    if split == "A" or seed != 0:
        t += f"_seed{seed}"
    if head != "lr":
        t += f"_{head}"
    # Same discipline as `head`: a non-default FEATURE SET gets its own tag so the
    # Tier-1.5 TSFM arms cannot overwrite the pretrained-embedding grid. Empty by
    # default, so every tag scored before 2026-09-01 is byte-identical.
    if FEATURE_TAG:
        t += f"_{FEATURE_TAG}"
    return t


def run_one(log_id, t_bin, X, y_eval, splits, ros, split, seed, mode, h_s,
            width_s=None, dump_scores=False, head="lr"):
    tag = arm_tag(mode, h_s, split, seed, width_s, head)
    ppath = os.path.join(META, f"horizon_pt_{tag}.json")
    cpath = os.path.join(META, f"horizon_curve_{tag}.csv")
    spath = os.path.join(META, "scores", f"horizon_scores_{tag}.parquet")
    # A cached arm is only fully cached if it also has the scores, when asked for.
    # Items 3 (sequential alarm rules) and 4 (matched-event lead) both need the
    # per-window test scores, and the curve alone throws them away.
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

    clf = fit_head(X[tr].astype(np.float32, copy=False), y_tr[tr], head, seed, max_iter=MAX_ITER)
    sc = clf.predict_proba(X[in_te].astype(np.float32, copy=False))[:, 1]

    # SCORED AGAINST THE FROZEN EVALUATION LABELS, never y_tr.
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
    est = clf[-1]
    n_it = int(np.max(est.n_iter_)) if getattr(est, "n_iter_", None) is not None else None
    pt.update(mode=mode, h_s=h_s, width_s=(hi and (lo - hi) / HZ),
              band_lo_bins=lo, band_hi_bins=hi,
              band_lo_s=lo / HZ, band_hi_s=hi / HZ,
              split=split, seed=seed, head=head,
              n_train_windows=int(tr.sum()), n_train_pos=n_pos,
              n_train_pos_flights=int(pd.Series(log_id[tr & (y_tr == 1)]).nunique()),
              n_iter=n_it, max_iter=(MAX_ITER if head == "lr" else 300),
              # `converged` is an lbfgs notion. A boosted tree stopping at its
              # max_iter is fully trained, not truncated, so never flag it.
              converged=((n_it < MAX_ITER) if n_it is not None else None)
                        if head == "lr" else None,
              minutes=round((time.time() - t0) / 60, 1))
    with open(ppath, "w") as f:
        json.dump(pt, f, default=float, indent=1)
    print(f"  {tag}: det@{REF_FAR}h {pt.get(f'det_at_{REF_FAR}h', float('nan')):.3f}  "
          f"lead {pt.get(f'lead_at_{REF_FAR}h', float('nan')):.1f}s  "
          f"auprc {pt.get('window_auprc', float('nan')):.4f}  "
          f"pos {n_pos:,}  {pt['minutes']}min", flush=True)
    return pt


def grid():
    jobs = []
    for split, seed in SPLITS:
        for h in HORIZONS_WIDEN:
            jobs.append((split, seed, "widen", h, None))
        for h in HORIZONS_SHIFT:
            jobs.append((split, seed, "shift", h, None))
    return jobs


def grid_gbt():
    """Item 4.7 follow-up: the BEST-vs-BEST arm the control exposed.

    RESULTS.md 4.7 found that hand-built statistics under the shifted band
    dominate Tier-0 on both axes, and that the strongest control arm is a
    BOOSTED TREE (`shift` 120 s: det 0.181 / lead 152.9 s) -- while every Tier-2
    horizon arm ever run is LINEAR. So the two tiers had never been compared at
    the same model class, and the representation's loss on the lead axis may be
    a head effect rather than a feature effect.

    This grid closes that: the same HistGBT hyper-parameters Tier-1 uses, on the
    embeddings, at the two bands and three seeds the control ran. Read each arm
    against `meta/h1_pt_gbt_shift{120,150}_B_fixed*.json` -- same head, same band,
    same split, same seeds, same alarm grid, so the ONLY difference is the
    features. `widen` 10 s is included as the reproduction gate: gbt-on-embeddings
    at the ORIGINAL band has never been run either, and it is the number that says
    what the representation is worth to a boosted tree before any band shift.
    """
    jobs = [("B_fixed", 0, "widen", 10, None, "gbt")]
    for h in (120, 150):
        for seed in (0, 1, 2):
            jobs.append(("B_fixed", seed, "shift", h, None, "gbt"))
    return jobs


def grid_gbt2():
    """Seeds + splits for the ONE arm that came back a positive: `widen` 10 s + gbt.

    The 2026-08-31 gbt grid returned det **0.376** for a boosted tree on the
    embeddings at the ORIGINAL 1-10 s band, against Tier-1 GBT's frozen **0.364**
    on the same split. **That is the first Tier-2 arm on this project ever to
    clear the Tier-1 bar** -- RESULTS.md 4.1 states plainly that the bar is not
    cleared, and 4.2 that neither fine-tuned arm reaches it either. It reframes
    4.7's negative: the representation was never the problem, the LINEAR HEAD was.

    But it is n = 1 with a margin of 0.012, which this project has been burned by
    three times (4.4 finding 6, the two 08-21 retractions). So: seeds 1-2 on
    B_fixed to make it quotable, and A / C to test whether it generalises -- split
    A is the hard case, where Tier-1 GBT reaches 0.708 and the linear probe only
    0.598, so it is the sharpest test of "the head was the bottleneck".
    """
    return ([("B_fixed", sd, "widen", 10, None, "gbt") for sd in (1, 2)]
            + [("A", sd, "widen", 10, None, "gbt") for sd in (0, 1, 2)]
            + [("C", 0, "widen", 10, None, "gbt")])


def grid_C():
    """Split C, the last untested surface in the dominance claim (4.5 caveat 4).

    Cross-airframe dominance is replicated across seeds, rules, heads and feature
    sets -- but on ONE of three splits. C's "NONE" in 4.5 has always meant
    *untested*, not refuted: no horizon arm had ever been run on it.

    C is schema-conditioned (`split_C_cond`), so it controls for logged-field
    availability rather than airframe. Its Tier-0 bar is the HARDEST detection bar
    of the three -- best-of-11 is plain `raw` at det 0.180 / lead 51.5 s, against
    B_fixed's 0.155 -- so this is not a soft replication.

    Both heads at both dominating bands, n = 3, plus `widen` 10 s at n = 3 as the
    reproduction gate: seed 0 must return the 4.1 probe number for C (det 0.484).
    Mirrors `horizon_tier1.grid_C` arm for arm, so every C row has its
    single-variable feature control.
    """
    jobs = [("C", sd, "widen", 10, None, hd)
            for hd in ("lr", "gbt") for sd in (0, 1, 2)]
    for hd in ("lr", "gbt"):
        for h in (120, 150):
            for sd in (0, 1, 2):
                jobs.append(("C", sd, "shift", h, None, hd))
    return jobs


def grid_2c():
    """Item 2b + 2c in one process, because the 15-min embedding load dominates.

    2b -- replicate the dominance arm (`shift` 120 s, B_fixed) at seeds 1 and 2.
    2c -- `shift` 120 s is the EDGE of the horizon grid, so the optimum may be
          further out and has never been tested: push to 150/180/240 s. And
          `shift` trades detection for lead, so try a 30 s-wide band at the same
          distance to buy detection back without giving up the horizon.
    Split A gets 180/240 s because its Tier-0 bar is 217.3 s -- 120 s reached only
    76.3 s there, so if split A is reachable at all it is from much further out.
    """
    return [
        ("B_fixed", 1, "shift", 120, None),      # 2b
        ("B_fixed", 2, "shift", 120, None),      # 2b
        ("B_fixed", 0, "shift", 150, None),      # 2c: past the edge
        ("B_fixed", 0, "shift", 180, None),
        ("B_fixed", 0, "shift", 240, None),
        ("B_fixed", 0, "shift", 120, 30.0),      # 2c: wider band, same distance
        ("A",       0, "shift", 180, None),      # 2c: can split A be reached?
        ("A",       0, "shift", 240, None),
    ]


def grid_2d():
    """Item 2d: get split A's optimum INTERIOR, and replicate both winners.

    Split A's dominating arm (`shift` 240 s) is the last horizon that was run --
    the same grid-edge failure mode that §2.1's threshold grid had, and that one
    turned out to be hiding a real effect. 300/360 s settle whether 240 s is the
    optimum or just the edge. Note the positive count falls fast with h (a flight
    needs h seconds of pre-event data to contribute at all), so 360 s may be
    skipped for want of positives -- which is itself the answer.

    The last two entries re-run arms that are already scored, purely to persist
    per-window scores (`--dump-scores`) for items 3 and 4; their metrics must come
    out identical to the cached ones, which is a free regression check.
    """
    return [
        ("A",       0, "shift", 300, None),
        ("A",       0, "shift", 360, None),
        ("A",       1, "shift", 240, None),
        ("A",       2, "shift", 240, None),
        ("B_fixed", 1, "shift", 150, None),
        ("B_fixed", 2, "shift", 150, None),
        ("B_fixed", 0, "shift", 120, None),   # refit only to dump scores
        ("A",       0, "shift", 240, None),   # refit only to dump scores
    ]


def grid_3b():
    """Item 3b(ii): bring the SEQUENTIAL-RULE results (RESULTS.md 4.6) to n = 3.

    Item 3 measured the alarm rules on whatever score dumps grid 2d happened to
    leave behind, which is 1 seed for `shift` 120 s, 2 for `shift` 150 s and 1
    each for `shift` 300/360 s. Two findings ride on those counts -- the free
    +18% detection `ewma` 15 buys cross-airframe, and the `cusum` 0.5 arm that is
    the first thing to move split A's 217.3 s bar -- and neither may be quoted
    below n = 3 (the lead-stability protocol, 11.0.1).

    Every arm here EXCEPT the split-A 300/360 s seeds is already scored and is
    refit only to persist `--dump-scores`. Their metrics must reproduce the
    cached values exactly, so most of this grid is also a regression check.

    Run with --dump-scores or it is a no-op on the cached arms.
    """
    return [
        ("B_fixed", 1, "shift", 120, None),   # refit only to dump scores
        ("B_fixed", 2, "shift", 120, None),   # refit only to dump scores
        ("B_fixed", 0, "shift", 150, None),   # refit only to dump scores
        ("A",       1, "shift", 300, None),
        ("A",       2, "shift", 300, None),
        ("A",       1, "shift", 360, None),
        ("A",       2, "shift", 360, None),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["A", "B", "B_fixed", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["widen", "shift"])
    ap.add_argument("--h", type=int, help="horizon in seconds")
    ap.add_argument("--views", default="emb_mean,emb_last")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; with --split/--mode/--h runs that "
                         "arm at each. Item 2b: the dominance claim is n=1.")
    ap.add_argument("--width", type=float, default=None,
                    help="shift-band width in seconds (default: the original 9 s)")
    ap.add_argument("--head", default="lr", choices=["lr", "gbt", "mlp"],
                    help="read-out head. `lr` is every arm before 2026-08-31 and "
                         "keeps the historical cache tags; `gbt` is the best-vs-best "
                         "arm against the RESULTS.md 4.7 Tier-1 control")
    ap.add_argument("--grid", choices=["default", "2c", "2d", "3b", "gbt", "gbt2", "C"],
                    default="default")
    ap.add_argument("--dump-scores", action="store_true",
                    help="also persist per-window test scores to meta/scores/ "
                         "(needed by items 3 and 4; refits an arm that lacks them)")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    ap.add_argument("--status", action="store_true", help="summarise cached arms")
    args = ap.parse_args()

    if args.grid == "2c":
        jobs = grid_2c()
    elif args.grid == "2d":
        jobs = grid_2d()
    elif args.grid == "3b":
        jobs = grid_3b()
    elif args.grid == "gbt":
        jobs = grid_gbt()
    elif args.grid == "gbt2":
        jobs = grid_gbt2()
    elif args.grid == "C":
        jobs = grid_C()
    elif args.split is None:
        jobs = grid()
    elif args.seeds:
        jobs = [(args.split, int(sd), args.mode, args.h, args.width, args.head)
                for sd in args.seeds.split(",")]
    else:
        jobs = [(args.split, args.seed, args.mode, args.h, args.width, args.head)]

    # The pre-2026-08-31 grids predate the `head` field and yield 5-tuples; they
    # were all linear, so pad them rather than editing four grid functions.
    jobs = [j if len(j) == 6 else (*j, "lr") for j in jobs]

    if args.list:
        print(f"{len(jobs)} jobs")
        for s, sd, m, h, w, hd in jobs:
            lo, hi = band_bins(m, h, w)
            print(f"  {hd:3s}  {m:5s} h={h:3d}s w={(lo-hi)/HZ:4.0f}s  split {s:8s} "
                  f"seed {sd}  band = t_e-{lo/HZ:.0f}s .. t_e-{hi/HZ:.0f}s")
        return

    if args.status:
        rows = []
        for s, sd, m, h, w, hd in jobs:
            p = os.path.join(META, f"horizon_pt_{arm_tag(m, h, s, sd, w, hd)}.json")
            if os.path.exists(p):
                with open(p) as f:
                    rows.append(json.load(f))
        print(f"{len(rows)} of {len(jobs)} arms cached")
        if rows:
            summarise(pd.DataFrame(rows))
        return

    splits = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))
    ros = splits.set_index("log_id")[["t_e_bin", "n_bins"]]

    print("loading embeddings...", flush=True)
    log_id, t_bin, X = load_embeddings(views=tuple(args.views.split(",")))
    print(f"  X = {X.shape}  ({X.nbytes/2**30:.1f} GiB)", flush=True)
    y_eval = window_labels(log_id, t_bin, ros)     # FROZEN -- scoring only
    print(f"  eval windows: {len(y_eval):,}  pos {int((y_eval==1).sum()):,}  "
          f"neg {int((y_eval==0).sum()):,}", flush=True)

    out = []
    for split, seed, mode, h, width, hd in jobs:
        pt = run_one(log_id, t_bin, X, y_eval, splits, ros, split, seed, mode, h,
                     width_s=width, dump_scores=args.dump_scores, head=hd)
        if pt:
            out.append(pt)
    if out:
        summarise(pd.DataFrame(out))


def summarise(df):
    keep = [c for c in ("mode", "h_s", "split", "n_train_pos", "n_train_pos_flights",
                        f"det_at_{REF_FAR}h", f"lead_at_{REF_FAR}h", "window_auprc",
                        "converged") if c in df.columns]
    pd.set_option("display.width", 200)
    print("\n=== horizon sweep ===")
    print(df.sort_values(["split", "mode", "h_s"])[keep].round(4).to_string(index=False))
    print("\nControl check: widen h=10 must reproduce the frozen probe "
          "(A seed0 det 0.672, B_fixed det 0.334).")


if __name__ == "__main__":
    main()
