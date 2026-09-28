#!/usr/bin/env python
"""Fast label-only scan of downloaded .ulg files.

Parses ONLY the four label topics rather than the whole log. sensor_combined
alone is often >80% of a log's bytes, so filtering makes this roughly an order
of magnitude faster than a full parse -- fast enough to run the gate-3 sample
before the tensorizer has caught up.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import extract_events, summarize

ROOT = os.path.dirname(os.path.abspath(__file__))

LABEL_TOPICS = ["failure_detector_status", "vehicle_status", "failsafe_flags",
                "vehicle_land_detected"]


def scan(path):
    from pyulog import ULog
    log_id = os.path.basename(path)[:-4]
    try:
        u = ULog(path, message_name_filter_list=LABEL_TOPICS)
    except Exception as e:
        return {"log_id": log_id, "status": f"parse_error: {type(e).__name__}"}
    try:
        ev = extract_events(u)
        s = summarize(ev)
    except Exception as e:
        return {"log_id": log_id, "status": f"label_error: {type(e).__name__}: {e}"[:150]}
    return {
        "log_id": log_id,
        "status": "ok",
        "t_e": ev["t_e"],
        "is_positive": ev["t_e"] is not None,
        "n_primary": s["n_primary"],
        "n_secondary": s["n_secondary"],
        "primary_fields": ",".join(s["primary_fields"]),
        "secondary_fields": ",".join(s["secondary_fields"]),
        "armed_s": s["armed_s"],
        "n_ff_fields": s["n_ff_fields"],
        "n_fds_fields": s["n_fds_fields"],
        # t_e relative to the first armed moment: a trigger firing in the first
        # second of arming is usually a bench test, not a flight event
        "t_e_after_arm_s": (round(ev["t_e"] - ev["armed"][0][0], 1)
                            if ev["t_e"] is not None and ev["armed"] else None),
        "bytes": os.path.getsize(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "meta", "label_scan.parquet"))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = []
    for dp, _, fs in os.walk(args.raw_dir):
        for f in fs:
            if f.endswith(".ulg"):
                files.append(os.path.join(dp, f))
    files.sort()
    if args.limit:
        files = files[: args.limit]
    print(f"scanning {len(files)} logs with {args.workers} workers", flush=True)

    recs, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(scan, files, chunksize=8), 1):
            recs.append(r)
            if i % 1000 == 0:
                el = time.time() - t0
                print(f"  [{i}/{len(files)}] {i/el:.1f} logs/s "
                      f"eta {(len(files)-i)/(i/el)/60:.1f} min", flush=True)

    df = pd.DataFrame(recs)
    df.to_parquet(args.out, index=False)
    ok = df[df["status"] == "ok"]
    print(f"\nscanned {len(df)} in {(time.time()-t0)/60:.1f} min")
    print(f"  parse/label failures: {len(df)-len(ok)}")
    if len(ok):
        print(f"  primary positives: {int(ok['is_positive'].sum())} "
              f"({100*ok['is_positive'].mean():.1f}%)")
        pf = ok[ok["is_positive"]]["primary_fields"].str.split(",").explode().value_counts()
        print("\n  positives by trigger field:")
        for k, v in pf.items():
            print(f"    {k:28s} {v:6d}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
