#!/usr/bin/env python
"""Parse .ulg -> masked 20 Hz tensor shards, with quality accounting.

Resampling method
-----------------
Source rates vary from 1 Hz (cpuload) to 250 Hz (sensor_combined), and vary for
the SAME topic across logs (vehicle_attitude: p50 20 Hz, p90 227 Hz). Naive
subsampling would alias the fast streams -- which is fatal here, because the
Q1 hypothesis is about rising variance and autocorrelation, exactly the
statistics aliasing corrupts.

We therefore bin-average onto the 20 Hz grid: every source sample falling in a
50 ms bin is averaged. A boxcar average IS a low-pass filter, it handles the
irregular and non-monotonic timestamps that scipy.decimate cannot, and it
degrades gracefully when a topic is slower than the grid (bins with no sample
are forward-filled up to MAX_STALE_S, then masked invalid).

Mask semantics (never zero-fill -- 0 is physically meaningful for most channels)
  valid[t, c] = True  iff a real, finite measurement supports that cell
  Sources of invalid: channel absent from log, no sample in bin and stale,
  all-NaN in bin, or inside a logger dropout interval.
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import (BIN_S, BOUNDS, CHANNEL_ID, CHANNELS, MAX_STALE_S,
                      N_CHANNELS, TARGET_HZ, TOPIC_FIELDS)
from labels import LEAKAGE_EXCLUDE, extract_events, summarize

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))

from shard_io import STORE_DTYPE  # single source of truth for the dtype

# Per-channel plausibility bounds as an array; inf where the channel is unbounded.
BOUNDS_ARR = np.array([b if b is not None else np.inf for b in BOUNDS], dtype=np.float64)

# Fail loudly if the channel vocabulary ever overlaps the leakage exclusion.
_bad = {t for t, _ in CHANNELS} & LEAKAGE_EXCLUDE
if _bad:
    raise RuntimeError(f"LEAKAGE: model-input channels drawn from excluded topics: {_bad}")


def _clean_series(ts, val):
    """Sort by time, drop non-monotonic duplicates, coerce to float64."""
    ts = np.asarray(ts, dtype=np.float64) / 1e6
    val = np.asarray(val, dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    ts, val = ts[order], val[order]
    keep = np.ones(len(ts), dtype=bool)
    if len(ts) > 1:
        keep[1:] = np.diff(ts) > 0          # drop exact-duplicate timestamps
    return ts[keep], val[keep]


def _bin_average(ts, val, t0, n_bins):
    """Average samples into 50 ms bins. Returns (sum, count) per bin."""
    idx = np.floor((ts - t0) / BIN_S).astype(np.int64)
    ok = (idx >= 0) & (idx < n_bins) & np.isfinite(val)
    idx, v = idx[ok], val[ok]
    s = np.bincount(idx, weights=v, minlength=n_bins)
    c = np.bincount(idx, minlength=n_bins)
    return s[:n_bins], c[:n_bins]


def _ffill_limited(x, valid, max_bins):
    """Forward-fill invalid cells from the last valid one, up to max_bins."""
    n = len(x)
    out = x.copy()
    ov = valid.copy()
    last_i = -1
    for i in range(n):
        if valid[i]:
            last_i = i
        elif last_i >= 0 and (i - last_i) <= max_bins:
            out[i] = x[last_i]
            ov[i] = True
    return out, ov


def _reject(rec, path, keep_raw):
    """Record a terminal rejection and free the raw .ulg.

    Every rejection below is a deterministic property of the file, so
    reprocessing can never change the verdict. Retaining these in staging
    deadlocked the stream: rejects accumulated to the producer's backpressure
    cap, and the consumer re-parsed the same ones forever.
    """
    if not keep_raw:
        try:
            os.remove(path)
            rec["raw_deleted"] = True
        except OSError:
            rec["raw_deleted"] = False
    return rec, None, None


def process_one(path, keep_raw):
    from pyulog import ULog

    rec = {"path": path, "log_id": os.path.basename(path)[:-4]}
    t0w = time.time()
    try:
        ulog = ULog(path)
    except Exception as e:
        rec["status"] = "parse_error"
        rec["err"] = f"{type(e).__name__}: {e}"[:300]
        return _reject(rec, path, keep_raw)

    # ---- time base: union of wanted topics -------------------------------
    tmin, tmax = np.inf, -np.inf
    have = {}
    for d in ulog.data_list:
        if d.name in TOPIC_FIELDS and "timestamp" in d.data:
            ts = np.asarray(d.data["timestamp"], dtype=np.float64) / 1e6
            if len(ts) < 2:
                continue
            # keep the largest instance if a topic is multi-instance
            if d.name in have and len(have[d.name].data["timestamp"]) >= len(ts):
                continue
            have[d.name] = d
            tmin = min(tmin, float(np.nanmin(ts)))
            tmax = max(tmax, float(np.nanmax(ts)))
    if not have or not np.isfinite(tmin) or tmax <= tmin:
        rec["status"] = "no_usable_topics"
        return _reject(rec, path, keep_raw)

    dur = tmax - tmin
    if dur > 86400:
        rec["status"] = "duration_corrupt"
        rec["dur_s"] = dur
        return _reject(rec, path, keep_raw)
    n_bins = int(np.floor(dur / BIN_S)) + 1
    if n_bins < int(TARGET_HZ * 10):        # <10 s of data is not a flight
        rec["status"] = "too_short"
        rec["dur_s"] = dur
        return _reject(rec, path, keep_raw)

    X = np.zeros((n_bins, N_CHANNELS), dtype=np.float32)
    V = np.zeros((n_bins, N_CHANNELS), dtype=bool)
    max_stale_bins = int(round(MAX_STALE_S / BIN_S))
    src_hz = {}

    for topic, fields in TOPIC_FIELDS.items():
        d = have.get(topic)
        if d is None:
            continue                        # channel absent -> stays masked
        present = {f.field_name for f in d.field_data}
        raw_ts = np.asarray(d.data["timestamp"], dtype=np.float64) / 1e6
        src_hz[topic] = round(len(raw_ts) / max(dur, 1e-9), 2)
        for fname in fields:
            cid = CHANNEL_ID[(topic, fname)]
            if fname not in present:
                continue                    # field absent in this firmware
            ts, val = _clean_series(d.data["timestamp"], d.data[fname])
            if len(ts) == 0:
                continue
            s, c = _bin_average(ts, val, tmin, n_bins)
            valid = c > 0
            x = np.zeros(n_bins, dtype=np.float64)
            np.divide(s, c, out=x, where=valid)
            x, valid = _ffill_limited(x, valid, max_stale_bins)
            X[:, cid] = x.astype(np.float32)
            V[:, cid] = valid

    # ---- mask out logger dropout intervals -------------------------------
    n_drop_bins = 0
    for dp in ulog.dropouts:
        try:
            ds = dp.timestamp / 1e6 - tmin
            de = ds + dp.duration / 1e3
        except Exception:
            continue
        b0 = max(0, int(np.floor(ds / BIN_S)))
        b1 = min(n_bins, int(np.ceil(de / BIN_S)) + 1)
        if b1 > b0:
            V[b0:b1, :] = False
            n_drop_bins += b1 - b0

    # ---- physical plausibility masking -----------------------------------
    # Corrupt source samples (measured up to 4.2e32 m/s^2) are masked, not
    # clipped: a nonsensical value is missing data, not a large measurement.
    n_implausible = 0
    for cid, b in enumerate(BOUNDS_ARR):
        if not np.isfinite(b):
            continue
        bad = (np.abs(X[:, cid]) > b) & V[:, cid]
        k = int(bad.sum())
        if k:
            n_implausible += k
            V[bad, cid] = False

    # ---- storage dtype + overflow guard ----------------------------------
    # float16 CANNOT hold this data: vehicle_air_data.baro_pressure_pa is
    # ~101325 Pa against an fp16 ceiling of 65504, so 98% of that channel
    # became +inf, and vehicle_local_position.x/y saturated at exactly 65504.
    # We store float32 and then verify: anything non-finite AFTER the cast is
    # masked invalid and counted, so a range problem can never again pass
    # silently as data.
    Xs = X.astype(STORE_DTYPE)
    overflow = ~np.isfinite(Xs) & V
    n_overflow = int(overflow.sum())
    if n_overflow:
        V &= np.isfinite(Xs)
    X = np.where(np.isfinite(Xs), Xs, 0).astype(STORE_DTYPE)

    # Defense in depth: force every invalid cell to exactly zero. The mask is
    # authoritative, but if a downstream consumer ever forgets to apply it,
    # this makes the bug produce obvious zeros instead of plausible stale
    # values silently leaking across dropout intervals.
    X[~V] = 0.0

    # ---- labels (from EXCLUDED topics; never enter X) --------------------
    try:
        ev = extract_events(ulog)
        lab = summarize(ev)
        t_e_bin = None
        if ev["t_e"] is not None:
            t_e_bin = int(np.floor((ev["t_e"] - tmin) / BIN_S))
            if not (0 <= t_e_bin < n_bins):
                t_e_bin = None
                lab["notes"] = list(lab.get("notes", [])) + ["t_e_outside_grid"]
    except Exception as e:
        lab, t_e_bin = {"error": f"{type(e).__name__}: {e}"[:200]}, None

    # ---- quality record ---------------------------------------------------
    per_ch_valid = V.mean(axis=0)
    rec.update({
        "status": "ok",
        "dur_s": round(dur, 2),
        "n_bins": n_bins,
        "n_topics_present": len(have),
        "n_channels_present": int((per_ch_valid > 0).sum()),
        "valid_frac": float(V.mean()),
        "n_overflow": n_overflow,
        "n_implausible": n_implausible,
        "dropout_bins": n_drop_bins,
        "src_hz": json.dumps(src_hz),
        "t_e_bin": t_e_bin,
        "label": json.dumps(lab, default=str),
        "parse_s": round(time.time() - t0w, 3),
        "raw_bytes": os.path.getsize(path),
    })

    blob = {
        "log_id": rec["log_id"],
        "n_bins": n_bins,
        "n_ch": N_CHANNELS,
        "t0_unix": tmin,
        "x": X.astype(STORE_DTYPE).tobytes(),
        "mask": np.packbits(V, axis=None).tobytes(),
        "t_e_bin": -1 if t_e_bin is None else t_e_bin,
    }
    if not keep_raw:
        try:
            os.remove(path)
            rec["raw_deleted"] = True
        except OSError:
            rec["raw_deleted"] = False
    return rec, blob, per_ch_valid


def _worker(a):
    path, keep_raw = a
    try:
        return process_one(path, keep_raw)
    except Exception as e:
        return ({"path": path, "log_id": os.path.basename(path)[:-4],
                 "status": "worker_crash",
                 "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"},
                None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "raw"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "proc"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "meta", "manifest.parquet"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delete-raw", action="store_true",
                    help="delete non-curated raw .ulg after tensorizing")
    ap.add_argument("--tag", default="proc")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    man = pd.read_parquet(args.manifest)
    curated = set(man.loc[man["curated"], "log_id"])

    files = []
    for dp, _, fs in os.walk(args.raw_dir):
        for f in fs:
            if f.endswith(".ulg"):
                files.append(os.path.join(dp, f))
    files.sort()
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} .ulg files to process, {args.workers} workers", flush=True)

    tasks = [(p, (os.path.basename(p)[:-4] in curated) or not args.delete_raw)
             for p in files]

    recs, blobs = [], []
    shard_i = 0
    ch_valid_acc = np.zeros(N_CHANNELS)
    n_ok = 0
    t0 = time.time()

    def flush():
        nonlocal shard_i, blobs
        if not blobs:
            return
        tbl = pa.table({k: [b[k] for b in blobs] for k in blobs[0]})
        out = os.path.join(args.out_dir, f"{args.tag}-{shard_i:05d}.parquet")
        pq.write_table(tbl, out, compression="zstd", compression_level=6)
        print(f"  wrote {out} ({len(blobs)} logs, "
              f"{os.path.getsize(out)/1e6:.1f} MB)", flush=True)
        shard_i += 1
        blobs = []

    with Pool(args.workers) as pool:
        for i, (rec, blob, chv) in enumerate(
                pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            recs.append(rec)
            if blob is not None:
                blobs.append(blob)
                ch_valid_acc += chv
                n_ok += 1
                if len(blobs) >= args.shard_size:
                    flush()
            if i % 500 == 0:
                el = time.time() - t0
                print(f"[{i}/{len(tasks)}] ok={n_ok} {i/el:.1f} logs/s "
                      f"eta={(len(tasks)-i)/max(i/el,1e-9)/60:.1f} min", flush=True)
    flush()

    qdf = pd.DataFrame(recs)
    qpath = os.path.join(ROOT, "meta", f"quality_{args.tag}.parquet")
    qdf.to_parquet(qpath, index=False)

    print("\n" + "=" * 70)
    print(f"processed {len(recs)} logs in {(time.time()-t0)/60:.1f} min")
    print("\nSTATUS BREAKDOWN:")
    for k, v in qdf["status"].value_counts().items():
        print(f"  {k:20s} {v:7d}  {100*v/len(qdf):5.1f}%")
    ok = qdf[qdf["status"] == "ok"]
    if len(ok):
        print(f"\nvalid_frac (fraction of tensor cells backed by real data):")
        print(f"  mean {ok['valid_frac'].mean():.3f}  p10 {ok['valid_frac'].quantile(.1):.3f}"
              f"  p50 {ok['valid_frac'].median():.3f}  p90 {ok['valid_frac'].quantile(.9):.3f}")
        print(f"channels present per log: mean {ok['n_channels_present'].mean():.1f}"
              f" / {N_CHANNELS}  p10 {ok['n_channels_present'].quantile(.1):.0f}"
              f"  p90 {ok['n_channels_present'].quantile(.9):.0f}")
        print(f"positives (t_e found): {int((ok['t_e_bin']>=0).sum())} "
              f"= {100*(ok['t_e_bin']>=0).mean():.1f}%")
        tot_mb = sum(os.path.getsize(os.path.join(args.out_dir, f))
                     for f in os.listdir(args.out_dir) if f.startswith(args.tag))
        print(f"\nshard bytes/log: {tot_mb/max(n_ok,1)/1e3:.1f} KB")
        print(f"projected for 442,689 logs: {tot_mb/max(n_ok,1)*442689/1e12:.2f} TB")
        print("\nPER-CHANNEL COVERAGE (worst 12):")
        cv = ch_valid_acc / max(n_ok, 1)
        for j in np.argsort(cv)[:12]:
            print(f"  {CHANNELS[j][0]+'.'+CHANNELS[j][1]:52s} {cv[j]:.3f}")
    print(f"\nwrote {qpath}")


if __name__ == "__main__":
    main()
