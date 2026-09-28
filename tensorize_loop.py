#!/usr/bin/env python
"""Streaming consumer: drain a staging dir of .ulg into tensor shards, delete raw.

Runs alongside `download.py --wait-on-cap`, which pauses when staging fills.
Together they form a bounded-memory pipeline over the 429k-log bulk corpus
without ever materialising 8 TB of raw .ulg on a shared disk.

Files are only claimed once they are stable (size unchanged across two polls
and no .part sibling), so a file still being written is never parsed.

Usage:
    python tensorize_loop.py --staging staging --out-dir proc --tag bulk
    python tensorize_loop.py --status
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import N_CHANNELS
from tensorize import _worker

ROOT = os.path.dirname(os.path.abspath(__file__))


def count_ulg(staging):
    """Raw .ulg count in staging, stable or not — the real backlog."""
    return sum(1 for _, _, fs in os.walk(staging)
               for f in fs if f.endswith(".ulg"))


def stable_files(staging, seen_sizes, min_age_s=5.0):
    """Return .ulg paths that are fully written (stable size, no .part sibling)."""
    out = []
    now = time.time()
    for dp, _, fs in os.walk(staging):
        for f in fs:
            if not f.endswith(".ulg"):
                continue
            p = os.path.join(dp, f)
            if os.path.exists(p + ".part"):
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            if now - st.st_mtime < min_age_s:
                continue
            if seen_sizes.get(p) == st.st_size:
                out.append(p)
            seen_sizes[p] = st.st_size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=os.path.join(ROOT, "staging"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "proc"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "meta", "manifest.parquet"))
    ap.add_argument("--tag", default="bulk")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--idle-exit-s", type=float, default=1800,
                    help="exit after this long with nothing to do")
    ap.add_argument("--flush-s", type=float, default=600,
                    help="force a partial shard if blobs are older than this")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    qdir = os.path.join(ROOT, "meta", f"qparts_{args.tag}")
    if args.status:
        n_shard = len([f for f in os.listdir(args.out_dir)
                       if f.startswith(args.tag)]) if os.path.isdir(args.out_dir) else 0
        sz = sum(os.path.getsize(os.path.join(args.out_dir, f))
                 for f in os.listdir(args.out_dir)
                 if f.startswith(args.tag)) if os.path.isdir(args.out_dir) else 0
        nq = 0
        if os.path.isdir(qdir):
            for f in os.listdir(qdir):
                try:
                    nq += pq.read_metadata(os.path.join(qdir, f)).num_rows
                except Exception:
                    pass
        stag = sum(1 for dp, _, fs in os.walk(args.staging) for f in fs
                   if f.endswith(".ulg")) if os.path.isdir(args.staging) else 0
        print(f"=== tensorize_loop [{args.tag}] {time.strftime('%H:%M:%S')} ===")
        print(f"  logs tensorized : {nq}")
        print(f"  shards written  : {n_shard}  ({sz/1e9:.1f} GB)")
        print(f"  staging backlog : {stag} .ulg")
        if nq:
            print(f"  bytes/log       : {sz/nq/1e3:.1f} KB")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(qdir, exist_ok=True)
    man = pd.read_parquet(args.manifest)
    curated = set(man.loc[man["curated"], "log_id"])
    print(f"consumer up: staging={args.staging} out={args.out_dir} "
          f"workers={args.workers} curated_retained={len(curated)}", flush=True)

    seen_sizes = {}
    blobs = []
    shard_i = len([f for f in os.listdir(args.out_dir) if f.startswith(args.tag)])
    n_total = 0
    idle_since = None
    t0 = time.time()
    last_flush = time.time()

    def flush():
        nonlocal shard_i, blobs, last_flush
        last_flush = time.time()
        if not blobs:
            return
        tbl = pa.table({k: [b[k] for b in blobs] for k in blobs[0]})
        out = os.path.join(args.out_dir, f"{args.tag}-{shard_i:05d}.parquet")
        pq.write_table(tbl, out, compression="zstd", compression_level=6)
        print(f"  shard {out} ({len(blobs)} logs, {os.path.getsize(out)/1e6:.1f} MB)",
              flush=True)
        shard_i += 1
        blobs = []

    pool = Pool(args.workers)
    try:
        while True:
            files = stable_files(args.staging, seen_sizes)
            if not files:
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since > args.idle_exit_s:
                    print("idle timeout; flushing and exiting", flush=True)
                    break
                time.sleep(10)
                continue
            idle_since = None
            files = files[: args.batch]
            # curated logs keep their raw; bulk logs are deleted after tensorizing
            tasks = [(p, os.path.basename(p)[:-4] in curated) for p in files]
            recs = []
            for rec, blob, _ in pool.imap_unordered(_worker, tasks, chunksize=4):
                recs.append(rec)
                if blob is not None:
                    blobs.append(blob)
                    if len(blobs) >= args.shard_size:
                        flush()
            # Time-based flush: raw is deleted as soon as a log tensorizes, so
            # anything sitting in `blobs` when the process dies is gone for good
            # (938 logs were lost that way on 2026-08-08). Bounds the window.
            if blobs and time.time() - last_flush > args.flush_s:
                flush()
            n_total += len(recs)
            pd.DataFrame(recs).to_parquet(
                os.path.join(qdir, f"q-{int(time.time()*1000)}.parquet"), index=False)
            for p in files:
                seen_sizes.pop(p, None)
            el = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] tensorized {n_total} "
                  f"({n_total/el*60:.0f}/min) backlog~{count_ulg(args.staging)}",
                  flush=True)
    finally:
        flush()
        pool.close()
        pool.join()
    print(f"consumer exit: {n_total} logs in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
