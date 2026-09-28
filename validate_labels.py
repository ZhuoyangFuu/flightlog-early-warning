#!/usr/bin/env python
"""Gate 3' -- validate trigger labels against independent flight physics.

Replaces the 240-log human review. The labels come from `failsafe_flags` and
the failure-detector topics, which are EXCLUDED from X, so checking them
against the attitude/rate/position/actuator channels is not circular: different
topics, different sensors.

Four diagnostics:

  D1 concurrence  -- at t_e, do the physics indicators depart from the log's
                     OWN earlier-flight baseline? A trigger that marks nothing
                     physical is firmware bookkeeping, not a failure.
  D2 control      -- same test at a matched pseudo-event time in no-trigger
                     logs. This is the automated form of the blinded controls,
                     over thousands of logs and immune to being fit against.
  D3 per-trigger  -- concurrence broken down by primary field. This is what
                     catches the next mode_req_* before it costs a corpus.
  D4 timing       -- anomaly onset relative to t_e. Validates t_e as an event
                     marker, which every EWS lead-time number rests on.

Indicators are built only from channels independent of the label source:
  attitude tracking error (q vs q_d), body-rate magnitude, descent rate,
  and motor-command spread.

Usage:
    python validate_labels.py --shards 'proc/bulk-*.parquet' --max-shards 30
"""
import argparse
import glob
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import BIN_S, CHANNEL_ID
from shard_io import decode

ROOT = os.path.dirname(os.path.abspath(__file__))

IQ = {
    "q":  [CHANNEL_ID[("vehicle_attitude", f"q[{i}]")] for i in range(4)],
    "qd": [CHANNEL_ID[("vehicle_attitude_setpoint", f"q_d[{i}]")] for i in range(4)],
    "w":  [CHANNEL_ID[("vehicle_angular_velocity", f"xyz[{i}]")] for i in range(3)],
    "vz": CHANNEL_ID[("vehicle_local_position", "vz")],
    "mot": [CHANNEL_ID[("actuator_motors", f"control[{i}]")] for i in range(8)],
}

# Per-indicator dispersion floors, each in its OWN units. A single global floor
# is what produced 1e18 z-scores in the first EWS run; never do that again.
FLOOR = {"att_err_deg": 2.0, "rate_mag": 0.05, "descent": 0.30, "motor_spread": 0.05}

PRE_S, POST_S = 1.0, 2.0        # event window around t_e
GAP_S = 10.0                    # baseline must end this long before t_e
MIN_BASE_S = 5.0                # minimum usable baseline
ONSET_LOOKBACK_S = 20.0         # D4 search window before t_e
PERSIST_BINS = 5                # 0.25 s, so a single spike is not an onset
Z_THRESH = 6.0


def indicators(x, v):
    """Per-bin physics indicators. Returns {name: (values, valid_mask)}."""
    out = {}

    q, qv = x[:, IQ["q"]], v[:, IQ["q"]].all(axis=1)
    qd, qdv = x[:, IQ["qd"]], v[:, IQ["qd"]].all(axis=1)
    nq, nqd = np.linalg.norm(q, axis=1), np.linalg.norm(qd, axis=1)
    ok = qv & qdv & (nq > 0.5) & (nqd > 0.5)
    dot = np.abs(np.einsum("ij,ij->i", q, qd, optimize=True)) / np.where(ok, nq * nqd, 1.0)
    out["att_err_deg"] = (np.degrees(2 * np.arccos(np.clip(dot, 0, 1))), ok)

    w, wv = x[:, IQ["w"]], v[:, IQ["w"]].all(axis=1)
    out["rate_mag"] = (np.linalg.norm(w, axis=1), wv)

    # NED: +vz is downward, so descent rate is vz itself.
    out["descent"] = (x[:, IQ["vz"]], v[:, IQ["vz"]])

    m, mv = x[:, IQ["mot"]], v[:, IQ["mot"]]
    n_mot = mv.sum(axis=1)
    keep = n_mot >= 2                       # >=2 valid motors guarantees no all-NaN row
    spread = np.zeros(len(m), dtype=np.float64)
    if keep.any():
        sub = np.where(mv[keep], m[keep], np.nan)
        spread[keep] = np.nanmax(sub, axis=1) - np.nanmin(sub, axis=1)
    out["motor_spread"] = (spread, keep)
    return out


def auc_mw(a, b):
    """P(random a > random b) via tie-aware ranks, plus a normal-approx p.

    Hand-rolled because scipy is not in this venv and one AUC is not worth a
    dependency on the machine that runs the pipeline.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan, np.nan
    ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    u = ranks[:na].sum() - na * (na + 1) / 2.0
    auc = u / (na * nb)
    mu, sd = na * nb / 2.0, np.sqrt(na * nb * (na + nb + 1) / 12.0)
    z = (u - mu) / sd if sd > 0 else np.nan
    p = math.erfc(abs(z) / np.sqrt(2)) / 2 if np.isfinite(z) else np.nan
    return auc, p


def robust_z(series, ok, base_lo, base_hi, ev_lo, ev_hi, floor):
    """Peak robust z of the event window against the log's own baseline."""
    b = series[base_lo:base_hi][ok[base_lo:base_hi]]
    e = series[ev_lo:ev_hi][ok[ev_lo:ev_hi]]
    if len(b) < int(MIN_BASE_S / BIN_S) or len(e) < 3:
        return np.nan, np.nan, np.nan
    med = np.median(b)
    iqr = np.subtract(*np.percentile(b, [75, 25]))
    sd = max(iqr / 1.349, floor)
    return float(np.clip((np.max(e) - med) / sd, -1e3, 1e3)), float(med), float(sd)


def onset_offset_s(series, ok, med, sd, t_bin, lookback, ev_hi):
    """First persistent threshold crossing near t_e, in seconds relative to it."""
    lo = max(0, t_bin - lookback)
    seg, segok = series[lo:ev_hi], ok[lo:ev_hi]
    if not segok.any():
        return np.nan
    hot = ((seg - med) / sd > Z_THRESH) & segok
    if not hot.any():
        return np.nan
    run = 0
    for i, h in enumerate(hot):
        run = run + 1 if h else 0
        if run >= PERSIST_BINS:
            return ((lo + i - PERSIST_BINS + 1) - t_bin) * BIN_S
    return np.nan


def process_shard(a):
    path, fracs, seed = a
    rng = np.random.default_rng(seed)
    pre, post = int(PRE_S / BIN_S), int(POST_S / BIN_S)
    gap, look = int(GAP_S / BIN_S), int(ONSET_LOOKBACK_S / BIN_S)
    rows = []
    try:
        pf = pq.ParquetFile(path)
    except Exception:
        return rows
    for batch in pf.iter_batches(batch_size=64):
        for r in batch.to_pylist():
            n = r["n_bins"]
            t_e = r["t_e_bin"]
            pos = t_e >= 0
            if pos:
                t = t_e
            else:
                # Matched pseudo-event: same relative-position distribution as
                # real t_e, so the control is not systematically earlier in flight.
                t = int(np.clip(rng.choice(fracs) * n, 0, n - 1))
            base_hi = t - gap
            if base_hi < int(MIN_BASE_S / BIN_S) or t + post > n:
                continue
            try:
                x, v = decode(r)
            except Exception:
                continue
            ind = indicators(x, v)
            rec = {"log_id": r["log_id"], "positive": pos, "n_bins": n,
                   "t_bin": t, "frac": t / n}
            zs, onsets = [], []
            for name, (s, ok) in ind.items():
                z, med, sd = robust_z(s, ok, 0, base_hi, max(0, t - pre),
                                      min(n, t + post), FLOOR[name])
                rec[f"z_{name}"] = z
                if np.isfinite(z):
                    zs.append(z)
                    o = onset_offset_s(s, ok, med, sd, t, look, min(n, t + post))
                    if np.isfinite(o):
                        onsets.append(o)
            if not zs:
                continue
            rec["z_max"] = max(zs)
            rec["n_ind"] = len(zs)
            rec["anomaly"] = rec["z_max"] > Z_THRESH
            rec["onset_s"] = min(onsets) if onsets else np.nan
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.join(ROOT, "proc", "bulk-*.parquet"))
    ap.add_argument("--max-shards", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quality", default=os.path.join(
        ROOT, "meta", "qparts_bulk", "q-consolidated.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "meta", "label_validation.parquet"))
    args = ap.parse_args()

    q = pd.read_parquet(args.quality, columns=["log_id", "status", "t_e_bin",
                                               "n_bins", "label"])
    q = q[q.status == "ok"]
    p = q[q.t_e_bin.notna() & (q.t_e_bin >= 0)]
    fracs = (p.t_e_bin / p.n_bins).clip(0.02, 0.98).to_numpy(dtype=float)
    print(f"t_e position prior from {len(fracs)} positives "
          f"(median {np.median(fracs):.2f} of flight)")

    files = sorted(glob.glob(args.shards))[: args.max_shards]
    print(f"scanning {len(files)} shards with {args.workers} workers")
    tasks = [(f, fracs, 1000 + i) for i, f in enumerate(files)]
    rows = []
    with Pool(args.workers) as pool:
        for i, rr in enumerate(pool.imap_unordered(process_shard, tasks), 1):
            rows.extend(rr)
            print(f"  [{i}/{len(files)}] {len(rows)} logs", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no usable logs")
        return
    lab = q.set_index("log_id")["label"].to_dict()
    df["primary_fields"] = df.log_id.map(
        lambda k: json.loads(lab[k]).get("primary_fields", []) if k in lab else [])
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} logs)\n")
    report(df)


def report(df):
    pos, neg = df[df.positive], df[~df.positive]
    line = "=" * 66

    print(line + f"\nD1  CONCURRENCE AT t_e   (anomaly = robust z > {Z_THRESH})\n" + line)
    print(f"  positives analysed      : {len(pos)}")
    if len(pos):
        print(f"  with physical anomaly   : {pos.anomaly.sum()} "
              f"({100*pos.anomaly.mean():.1f}%)")
        print(f"  median z_max            : {pos.z_max.median():.1f}")
        print("\n  per-indicator median z at t_e:")
        for c in [c for c in df.columns if c.startswith("z_") and c != "z_max"]:
            cov = pos[c].notna().mean()
            print(f"    {c[2:]:16s} median={pos[c].median():7.2f}  coverage={100*cov:5.1f}%")

    print("\n" + line + "\nD2  MATCHED CONTROLS (no trigger fired)\n" + line)
    print(f"  controls analysed       : {len(neg)}")
    if len(neg):
        print(f"  with physical anomaly   : {neg.anomaly.sum()} "
              f"({100*neg.anomaly.mean():.1f}%)")
        print(f"  median z_max            : {neg.z_max.median():.1f}")
    if len(pos) and len(neg):
        a, b = pos.anomaly.mean(), neg.anomaly.mean()
        lift = f"   lift x{a/b:.2f}" if b > 0 else ""
        print(f"\n  SEPARATION: {100*a:.1f}% of positives vs "
              f"{100*b:.1f}% of controls{lift}")
        # Rank-based effect size: P(random positive scores above random control).
        auc, p = auc_mw(pos.z_max, neg.z_max)
        print(f"  AUC(z_max, positive vs control) = {auc:.3f}   p={p:.1e}")

    print("\n" + line + "\nD3  PER-TRIGGER CONCURRENCE\n" + line)
    ex = pos.explode("primary_fields")
    rows = [(f, len(g), g.anomaly.mean())
            for f, g in ex.groupby("primary_fields") if f and len(g) >= 5]
    for f, n, r in sorted(rows, key=lambda t: t[2]):
        flag = "  <-- SUSPECT" if r < 0.5 else ""
        print(f"  {str(f):32s} n={n:5d}  concurrence={r:.3f}{flag}")

    print("\n" + line + "\nD4  ANOMALY ONSET RELATIVE TO t_e\n" + line)
    o = pos.onset_s.dropna()
    if len(o):
        print(f"  positives with an onset : {len(o)} ({100*len(o)/len(pos):.1f}%)")
        print(f"  median onset            : {o.median():+.2f} s "
              f"(negative = physics leads the trigger)")
        for p_ in [10, 25, 50, 75, 90]:
            print(f"    p{p_:<3d} {np.percentile(o, p_):+7.2f} s")
        print(f"  onset BEFORE t_e        : {100*(o < 0).mean():.1f}% of events")


if __name__ == "__main__":
    main()
