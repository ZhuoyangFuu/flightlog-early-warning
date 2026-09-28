#!/usr/bin/env python
"""Reclaim disk by deleting curated raw .ulg that is no longer needed.

Raw retention was insurance, not a necessity: gate-3 inspection happens through
logs.px4.io URLs and never touches local files, and re-downloading the whole
curated set costs ~70 minutes at the measured CDN rate.

KEEP (never deleted):
  * the tier-0 pilot          -- the reference sample the gates were measured on
  * the gate-3 review sample  -- so a rater can open a local copy if needed
  * any log that did NOT tensorize successfully -- deleting raw for a log we
    failed to process would lose the only evidence of WHY it failed

DELETE: everything else that has a verified-good tensor in a shard.

Dry-run by default. Pass --execute to actually unlink.
"""
import argparse
import glob
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.abspath(__file__))


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "raw"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "meta", "manifest.parquet"))
    ap.add_argument("--quality", default=os.path.join(ROOT, "meta", "quality_curated.parquet"))
    ap.add_argument("--gate3-key", default=os.path.join(ROOT, "meta", "gate3_answer_key.csv"))
    ap.add_argument("--shards", default=os.path.join(ROOT, "proc", "curated-*.parquet"))
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    keep = set(man.loc[man["tier"] == 0, "log_id"])
    print(f"keep: {len(keep)} pilot logs")

    if os.path.exists(args.gate3_key):
        g = set(pd.read_csv(args.gate3_key)["log_id"])
        keep |= g
        print(f"keep: +{len(g)} gate-3 review logs -> {len(keep)} total")

    # A log is only safe to delete if it tensorized OK.
    if not os.path.exists(args.quality):
        sys.exit(f"missing {args.quality}; run tensorize.py first")
    q = pd.read_parquet(args.quality)
    ok_ids = set(q.loc[q["status"] == "ok", "log_id"])
    bad = q[q["status"] != "ok"]
    print(f"tensorized ok: {len(ok_ids)}   not ok: {len(bad)} (raw retained)")
    if len(bad):
        for k, v in bad["status"].value_counts().items():
            print(f"    {k}: {v}")

    # Cross-check against what is actually IN the shards, not just the quality
    # log -- the shard is the artifact we are trading the raw file for.
    in_shards = set()
    for f in sorted(glob.glob(args.shards)):
        in_shards |= set(pq.read_table(f, columns=["log_id"]).column("log_id").to_pylist())
    print(f"log_ids present in curated shards: {len(in_shards)}")
    safe = ok_ids & in_shards
    if len(safe) < len(ok_ids):
        print(f"  WARNING {len(ok_ids)-len(safe)} logs marked ok but absent from shards; retained")

    files = []
    for dp, _, fs in os.walk(args.raw_dir):
        for f in fs:
            if f.endswith(".ulg"):
                files.append(os.path.join(dp, f))
    print(f"\nraw .ulg on disk: {len(files)}")

    to_del, del_bytes, keep_bytes = [], 0, 0
    for p in files:
        lid = os.path.basename(p)[:-4]
        sz = os.path.getsize(p)
        if lid in keep or lid not in safe:
            keep_bytes += sz
        else:
            to_del.append(p)
            del_bytes += sz

    print(f"  DELETE : {len(to_del):6d} files  {human(del_bytes)}")
    print(f"  RETAIN : {len(files)-len(to_del):6d} files  {human(keep_bytes)}")

    if not args.execute:
        print("\nDRY RUN. Pass --execute to delete.")
        return

    n = 0
    freed = 0
    for p in to_del:
        try:
            sz = os.path.getsize(p)
            os.remove(p)
            freed += sz
            n += 1
        except OSError as e:
            print(f"  failed {p}: {e}")
    # prune empty shard dirs
    for dp, dirs, fs in os.walk(args.raw_dir, topdown=False):
        if dp != args.raw_dir and not os.listdir(dp):
            os.rmdir(dp)
    print(f"\ndeleted {n} files, freed {human(freed)}")


if __name__ == "__main__":
    main()
