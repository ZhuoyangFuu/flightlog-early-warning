#!/usr/bin/env python
"""Shared scoring path for every Tier-2 model (probe and fine-tune alike).

This file exists so that "score a set of windows and report detection at
FAR 1.6/h" is written down exactly once. `probe.py` and `finetune.py` both call
it, and everything it needs is imported from `tier1_baseline` rather than
restated, so a Tier-1 protocol change propagates to all of them.

The contract is deliberately narrow: give it per-window (log_id, t_bin, score)
on **Tier-1's grid** plus the roster of the flights those windows came from, and
it returns the same dict shape Tier-1 reports.
"""
import os

import numpy as np
import pandas as pd

from tier1_baseline import (POS_LO, POS_HI, NEG_MIN_BINS, REF_FAR,
                            threshold_grid, flight_curve, op_point)


def window_labels(log_id, t_bin, ros):
    """Tier-1's window labels, unchanged.

    1  = inside the pre-event band (t_e - POS_LO, t_e - POS_HI], i.e. 1-10 s before
    0  = a window of a negative flight that flew long enough to count
    -1 = everything else, and it must be DROPPED, not treated as negative:
         windows of a positive flight outside the band are ambiguous, not safe.

    `ros` is indexed by log_id; ids missing from it map to NaN and fall to -1,
    which is what should happen to a flight that is not in the evaluation table.
    """
    te = pd.Series(log_id).map(ros.t_e_bin).to_numpy()
    nb = pd.Series(log_id).map(ros.n_bins).to_numpy()
    is_pos = te >= 0
    return np.where(is_pos & (t_bin >= te - POS_LO) & (t_bin <= te - POS_HI), 1,
                    np.where(~is_pos & (nb >= NEG_MIN_BINS), 0, -1))


def score_report(log_id, t_bin, score, y, roster_test, curve_path=None):
    """Per-window scores -> Tier-1's reported metrics.

    `roster_test` must carry log_id / t_e_bin / n_bins for the test flights ONLY:
    `flight_curve` derives both the false-alarm rate and the detection rate from
    it, so passing the full roster would silently score against the wrong
    denominator.
    """
    from sklearn.metrics import average_precision_score

    auprc = (average_precision_score(y[y >= 0], score[y >= 0])
             if (y == 1).any() and (y == 0).any() else np.nan)

    sub = pd.DataFrame({"log_id": log_id, "t_bin": t_bin, "score": score})
    scores_by_flight = {lid: (g.t_bin.to_numpy(), g.score.to_numpy())
                        for lid, g in sub.groupby("log_id", sort=False)}
    # The threshold grid is built from NEGATIVE scores only -- that is what makes
    # the low-FAR end of the curve resolvable (tier1_baseline.threshold_grid).
    neg = score[y == 0]
    if neg.size < 4:
        return {"window_auprc": float(auprc), "error": "too few negative windows"}
    curve = flight_curve(scores_by_flight, roster_test, threshold_grid(neg))
    if curve_path:
        curve.to_csv(curve_path, index=False)
    return {"window_auprc": float(auprc), **op_point(curve),
            "n_test_windows": int(len(score)),
            "n_test_flights": len(scores_by_flight)}


def roster_for(splits, ids):
    """log_id / t_e_bin / n_bins for exactly `ids`, in the shape flight_curve wants."""
    r = splits[["log_id", "t_e_bin", "n_bins"]]
    return r[r.log_id.isin(ids)].reset_index(drop=True)


def summarise(rows, keys=None):
    """Mean +/- sd across seeds, the way §8 asks for it."""
    if not rows:
        return {}
    keys = keys or (f"det_at_{REF_FAR}h", f"lead_at_{REF_FAR}h",
                    "window_auprc", "detection_rate", "median_lead_s")
    out = {}
    for k in keys:
        v = [r[k] for r in rows if k in r and np.isfinite(r.get(k, np.nan))]
        if v:
            out[k] = f"{np.mean(v):.3f} +/- {np.std(v):.3f}" if len(v) > 1 \
                else f"{v[0]:.3f}"
    return out
