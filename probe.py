#!/usr/bin/env python
"""Tier-2, stage 2: probes on the frozen encoder representation.

PROJECT_PLAN §7 ("Proposed": ... then linear probe and fine-tune"). Stage 1
(`embed_eval.py`) cached one pooled vector per scored window; this stage fits a
head on those vectors and scores it with **Tier-1's own evaluation functions**.

The comparison this is built to support
--------------------------------------
Tier-1 (§7.1), detection interpolated at exactly FAR 1.6/h:

    Tier-0 classical EWS   0.186 (lead 66.0 s)      --        --
    Tier-1 logistic reg.   0.572 +- 0.054            0.246     0.202
    Tier-1 grad boosting   0.708 +- 0.045            0.360     0.364
                            split A                  split B   split B_fixed

The bar is GBT at 0.708 / 0.364, not EWS at 0.186 (§7.1, consequence 3).
`B_fixed` is the headline cross-airframe split as of 2026-08-18 (§8).

Everything that could make the numbers incomparable lives in `eval_common`,
which in turn imports it from `tier1_baseline` rather than restating it: the
label bands (`POS_LO`, `POS_HI`), the negative-flight floor (`NEG_MIN_BINS`),
the threshold grid, the survival-estimator curve and the operating point. The
400k negative cap is here because it is a *fitting* choice, not a metric one.
`finetune.py` scores through the same `eval_common` path, so probe and
fine-tune numbers are directly comparable to each other as well as to Tier-1.

What differs from Tier-1, deliberately: the *features*. Tier-1 uses 5 statistics
x 87 channels over a 5 s window; this uses the encoder's pooled representation of
a 25.6 s window. Same flights, same alarm grid, same metric.

Heads
-----
  lr   -- logistic regression on standardised embeddings. THE linear probe: it
          is the honest "is the representation linearly separable" test, and it
          is the number that should be compared against Tier-1 lr (0.572/0.246).
  mlp  -- one hidden layer (512), the cheap non-linear read-out. Compare against
          Tier-1 gbt (0.708/0.360).

Usage
-----
    python probe.py --split A --seed 0 --head lr
    python probe.py --all                    # every head x split, cached
    python probe.py --pilot                  # use the pilot embeddings
    python probe.py --all --max-iter 5000     # converged-lr refit; the mlp jobs
                                              # are unaffected and return cached
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
from tier1_baseline import REF_FAR
from eval_common import window_labels, score_report, roster_for

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")
# Overridable so a DIFFERENT frozen feature extractor can be dropped in without
# touching any of the fitting or scoring code -- the Tier-1.5 TSFM baseline
# (`tsfm_baseline.py`) writes this schema to `meta/tsfm_emb`. Unset, it is exactly
# the historical path, so every cached arm is unaffected.
EMB = os.environ.get("FLIGHTLOG_EMB_DIR", os.path.join(META, "eval_emb"))
NEG_CAP = 400_000          # matches tier1_baseline.train_eval
# All six lr fits of the 2026-08-19 grid hit this cap WITHOUT converging (lbfgs
# status=1), so those numbers are a lower bound on the linear probe. --max-iter
# raises it and is part of the cache tag, so a refit does not collide with them.
LR_MAX_ITER = 1000


def load_embeddings(pilot=False, views=("emb_mean", "emb_last")):
    """-> (log_id array, t_bin array, X (n, d*len(views))).

    2026-09-18: preallocates X once and fills it per part straight from the
    flat fixed-size-list buffer. The old path (object array -> np.stack ->
    astype -> concatenate per part, then one more concatenate over parts)
    peaked at ~4x the final matrix and was OOM-killed on every 4096-d
    `emb_role` probe (18/18, 15-16 Sept) while the box also held the
    pretraining loader. Values are bit-identical to the old path (the source
    is float16; float16 -> float32 is exact). Set FLIGHTLOG_X_FP16=1 to keep X
    in float16 (half the RAM); every consumer up-casts the rows it slices with
    `.astype(np.float32, copy=False)`, so results are unchanged either way.
    """
    parts = sorted(glob.glob(os.path.join(EMB, "emb-*.parquet")))
    parts = [p for p in parts if ("pilot" in p) == bool(pilot)]
    if not parts:
        raise SystemExit(f"no {'pilot ' if pilot else ''}embeddings in {EMB}\n"
                         f"  run:  bash run_embed.sh {'pilot' if pilot else 'start'}")
    dtype = np.float16 if os.environ.get("FLIGHTLOG_X_FP16") else np.float32
    metas = [pq.read_metadata(p) for p in parts]
    n_tot = sum(m.num_rows for m in metas)
    schema = pq.read_schema(parts[0])
    dims = [schema.field(v).type.list_size for v in views]
    X = np.empty((n_tot, sum(dims)), dtype=dtype)
    lids, tbs, r0 = [], [], 0
    for p, m in zip(parts, metas):
        t = pq.read_table(p, columns=["log_id", "t_bin", *views])
        n = m.num_rows
        lids.append(np.asarray(t["log_id"].to_pylist(), dtype=object))
        tbs.append(t["t_bin"].to_numpy())
        c0 = 0
        for v, d in zip(views, dims):
            arr = t[v].combine_chunks()
            assert arr.null_count == 0, f"{p}: nulls in {v}"
            flat = arr.flatten().to_numpy(zero_copy_only=False)
            X[r0:r0 + n, c0:c0 + d] = flat.reshape(n, d)
            c0 += d
        r0 += n
        del t
        print(f"  loaded {os.path.basename(p)}  {n:,}", flush=True)
    assert r0 == n_tot
    return np.concatenate(lids), np.concatenate(tbs), X


def fit_head(Xtr, ytr, head, seed, max_iter=LR_MAX_ITER):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if head == "lr":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=max_iter,
                                                class_weight="balanced",
                                                random_state=seed)).fit(Xtr, ytr)
    if head == "gbt":
        # Added 2026-08-31 for the "best vs best" arm of RESULTS.md 4.7. The
        # shifted-band Tier-1 control's strongest model is a boosted tree, but
        # every Tier-2 horizon arm was linear, so the two tiers had never been
        # compared at the same model class. Hyper-parameters are Tier-1's
        # verbatim (tier1_baseline.train_eval) so the ONLY difference between
        # `h1_pt_gbt_*` and `horizon_pt_*_gbt` is the feature matrix.
        # Wrapped in a pipeline purely so callers can keep indexing clf[-1].
        from sklearn.ensemble import HistGradientBoostingClassifier
        return make_pipeline(
            HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                           random_state=seed)).fit(Xtr, ytr)
    if head == "mlp":
        from sklearn.neural_network import MLPClassifier
        # class_weight has no MLP equivalent in sklearn; balance by resampling
        # the positives instead, so the head sees the same prior as lr does.
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(ytr == 1)
        neg = np.flatnonzero(ytr == 0)
        if len(pos) and len(pos) < len(neg):
            pos = rng.choice(pos, len(neg), replace=True)
        idx = rng.permutation(np.concatenate([pos, neg]))
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(512,), max_iter=60,
                                           early_stopping=True, random_state=seed)
                             ).fit(Xtr[idx], ytr[idx])
    raise SystemExit(f"unknown head {head}")


def run_one(log_id, t_bin, X, y, splits, ros, split, seed, head, pilot=False,
            max_iter=LR_MAX_ITER):
    tag = f"{head}_{split}" + (f"_seed{seed}" if split == "A" else "")
    # A non-default max_iter is a different fit, so it gets a different cache
    # entry -- otherwise a refit would silently return the unconverged result.
    if head == "lr" and max_iter != LR_MAX_ITER:
        tag += f"_it{max_iter}"
    if pilot:
        tag += "_pilot"
    ppath = os.path.join(META, f"tier2_pt_{tag}.json")
    cpath = os.path.join(META, f"tier2_curve_{tag}.csv")
    if os.path.exists(ppath) and os.path.exists(cpath):
        with open(ppath) as f:
            pt = json.load(f)
        print(f"  {tag}: cached {pt}", flush=True)
        return pt

    col = {"A": f"split_A_s{seed}", "B": "split_B", "B_fixed": "split_B_fixed",
           "C": "split_C_cond"}[split]
    tr_ids = set(splits.loc[splits[col] == "train", "log_id"])
    te_ids = set(splits.loc[splits[col] == "test", "log_id"])

    t0 = time.time()
    in_tr = pd.Series(log_id).isin(tr_ids).to_numpy()
    in_te = pd.Series(log_id).isin(te_ids).to_numpy()

    tr = in_tr & (y >= 0)
    rng = np.random.default_rng(seed)
    neg_idx = np.flatnonzero(tr & (y == 0))
    if len(neg_idx) > NEG_CAP:
        tr[rng.choice(neg_idx, len(neg_idx) - NEG_CAP, replace=False)] = False
    if not (y[tr] == 1).any():
        print(f"  {tag}: no positive training windows -- skipped", flush=True)
        return None

    clf = fit_head(X[tr].astype(np.float32, copy=False), y[tr], head, seed, max_iter=max_iter)
    sc = clf.predict_proba(X[in_te].astype(np.float32, copy=False))[:, 1]

    pt = score_report(log_id[in_te], t_bin[in_te], sc, y[in_te],
                      roster_for(splits, te_ids), curve_path=cpath)
    pt.update(n_train_windows=int(tr.sum()), n_train_pos=int((y[tr] == 1).sum()),
              head=head, split=split, seed=seed,
              minutes=round((time.time() - t0) / 60, 1))
    # Record whether the fit actually converged -- an unconverged head is a lower
    # bound on itself, and that has to travel with the number, not with the log.
    est = clf[-1]
    n_it = getattr(est, "n_iter_", None)
    if n_it is not None:
        n_it = int(np.max(n_it))
        cap = max_iter if head == "lr" else est.get_params().get("max_iter")
        pt.update(n_iter=n_it, max_iter=int(cap), converged=bool(n_it < cap))
    with open(ppath, "w") as f:
        json.dump(pt, f, default=float, indent=1)
    print(f"  {tag}: {pt}", flush=True)
    return pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A", choices=["A", "B", "B_fixed", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", default="lr", choices=["lr", "mlp"])
    ap.add_argument("--views", default="emb_mean,emb_last")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-iter", type=int, default=LR_MAX_ITER,
                    help="lbfgs iteration cap for the lr head; a non-default "
                         "value caches under its own tag (see LR_MAX_ITER)")
    args = ap.parse_args()

    splits = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))
    ros = splits.set_index("log_id")[["t_e_bin", "n_bins"]]

    print("loading embeddings...", flush=True)
    log_id, t_bin, X = load_embeddings(pilot=args.pilot,
                                       views=tuple(args.views.split(",")))
    print(f"  X = {X.shape}  ({X.nbytes/2**30:.1f} GiB)", flush=True)
    y = window_labels(log_id, t_bin, ros)
    print(f"  windows: {len(y):,}  pos {int((y==1).sum()):,}  "
          f"neg {int((y==0).sum()):,}  dropped {int((y==-1).sum()):,}", flush=True)

    # B_fixed is the headline cross-airframe split (PROJECT_PLAN §8, decided
    # 2026-08-18); B is kept in the grid so the frozen Tier-1 0.360 has a
    # like-for-like Tier-2 counterpart.
    jobs = ([(s, sd, h) for h in ("lr", "mlp")
             for s, sd in [("A", 0), ("A", 1), ("A", 2),
                           ("B_fixed", 0), ("B", 0), ("C", 0)]]
            if args.all else [(args.split, args.seed, args.head)])
    out = []
    for split, seed, head in jobs:
        pt = run_one(log_id, t_bin, X, y, splits, ros, split, seed, head,
                     pilot=args.pilot, max_iter=args.max_iter)
        if pt:
            out.append(pt)

    if out:
        df = pd.DataFrame(out)
        keep = [c for c in (f"det_at_{REF_FAR}h", f"lead_at_{REF_FAR}h",
                            "window_auprc", "detection_rate", "far_per_hour")
                if c in df.columns]
        print("\n=== Tier-2 probe summary ===")
        print(df[["head", "split", "seed"] + keep].to_string(index=False))


if __name__ == "__main__":
    main()
