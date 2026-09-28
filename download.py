#!/usr/bin/env python
"""Detached, resumable PX4 log downloader with hard disk guards.

Design notes:
  * State lives in SQLite (state/download.db) and is committed continuously,
    so the job resumes exactly where it stopped after a kill or reboot.
  * Disk is guarded twice: an absolute floor of free space on the filesystem
    (other lab members share /mnt/wsssd) and a cap on this project's own
    raw/ directory. Hitting either stops the job cleanly rather than filling
    the disk.
  * Concurrency is modest and fixed. The CDN is a bulk-serving endpoint, but
    there is no reason to be rude about it.

Usage:
    python download.py --curated-only          # start / resume
    python download.py --status                # one-command status
"""
import argparse
import os
import queue
import shutil
import signal
import sqlite3
import sys
import threading
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "state", "download.db")
RAW = os.path.join(ROOT, "raw")
UA = "px4-flightlog-research/0.1 (zhuoyang@udel.edu; academic early-warning study)"

STOP = threading.Event()


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def init_db(manifest, curated_only, tiers, bulk_only=False):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS dl (
        log_id TEXT PRIMARY KEY, url TEXT, tier INT, curated INT, rank INT,
        status TEXT DEFAULT 'pending', bytes INT DEFAULT 0,
        attempts INT DEFAULT 0, err TEXT, ts REAL)""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_status ON dl(status, rank)")
    n_before = con.execute("SELECT COUNT(*) FROM dl").fetchone()[0]
    if n_before == 0:
        df = pd.read_parquet(manifest)
        if curated_only:
            df = df[df["curated"]]
        if bulk_only:
            df = df[~df["curated"]]
        if tiers:
            df = df[df["tier"].isin(tiers)]
        df = df.sort_values("rank")
        con.executemany(
            "INSERT OR IGNORE INTO dl(log_id,url,tier,curated,rank) VALUES (?,?,?,?,?)",
            list(zip(df["log_id"], df["download_url"], df["tier"].astype(int),
                     df["curated"].astype(int), df["rank"].astype(int))))
        con.commit()
        print(f"seeded {len(df)} rows into {DB}", flush=True)
    else:
        print(f"resuming existing state: {n_before} rows", flush=True)
    return con


def dir_bytes(path):
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def worker(wid, con_lock, q, results, args):
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    while not STOP.is_set():
        try:
            item = q.get(timeout=2)
        except queue.Empty:
            # Keep waiting. Returning here was a deadlock: during a multi-minute
            # backpressure pause the queue goes empty, every worker exits, and
            # the main loop then blocks forever filling a queue with no
            # consumers. Workers now only leave on STOP or the None sentinel.
            continue
        if item is None:          # shutdown sentinel
            q.task_done()
            return
        log_id, url = item
        shard = log_id[:2]
        outdir = os.path.join(RAW, shard)
        os.makedirs(outdir, exist_ok=True)
        dest = os.path.join(outdir, log_id + ".ulg")
        tmp = dest + ".part"
        if os.path.exists(dest):
            results.put((log_id, "done", os.path.getsize(dest), None))
            q.task_done()
            continue
        try:
            with sess.get(url, stream=True, timeout=(15, 180)) as r:
                if r.status_code != 200:
                    results.put((log_id, "http_error", 0, f"HTTP {r.status_code}"))
                    q.task_done()
                    continue
                n = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if STOP.is_set():
                            break
                        fh.write(chunk)
                        n += len(chunk)
            if STOP.is_set():
                if os.path.exists(tmp):
                    os.remove(tmp)
                q.task_done()
                return
            os.rename(tmp, dest)
            results.put((log_id, "done", n, None))
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            results.put((log_id, "error", 0, f"{type(e).__name__}: {e}"[:200]))
        q.task_done()


def status(args):
    if not os.path.exists(DB):
        print("no state db yet; download has not been started")
        return
    con = sqlite3.connect(DB, timeout=60)
    rows = con.execute("SELECT status, COUNT(*), COALESCE(SUM(bytes),0) FROM dl GROUP BY status").fetchall()
    tot = con.execute("SELECT COUNT(*) FROM dl").fetchone()[0]
    done = dict((r[0], r[1]) for r in rows).get("done", 0)
    print(f"=== FlightLog download status  {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    for s, c, b in rows:
        print(f"  {s:12s} {c:8d}  {human(b)}")
    print(f"  {'TOTAL':12s} {tot:8d}   ({100*done/tot if tot else 0:.2f}% done)")
    st = con.execute("SELECT MIN(ts), MAX(ts) FROM dl WHERE status='done' AND ts IS NOT NULL").fetchone()
    if st and st[0]:
        el = st[1] - st[0]
        rate = done / el * 60 if el > 0 else 0
        rem = tot - done
        print(f"  elapsed {el/3600:.2f} h   rate {rate:.0f} logs/min")
        if rate > 0:
            print(f"  remaining {rem} logs -> ETA {rem/rate/60:.2f} h")
    du = shutil.disk_usage(ROOT)
    print(f"  raw/ on disk: {human(dir_bytes(RAW))}")
    print(f"  filesystem free: {human(du.free)} of {human(du.total)}")
    err = con.execute("SELECT err, COUNT(*) FROM dl WHERE status NOT IN ('done','pending') "
                      "GROUP BY err ORDER BY 2 DESC LIMIT 8").fetchall()
    if err:
        print("  top errors:")
        for e, c in err:
            print(f"    {c:6d}  {str(e)[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "meta", "manifest.parquet"))
    ap.add_argument("--curated-only", action="store_true")
    ap.add_argument("--bulk-only", action="store_true",
                    help="everything NOT in the curated set (stream-and-discard tier)")
    ap.add_argument("--tiers", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--raw-dir", default=None, help="override raw/staging dir")
    ap.add_argument("--min-free-gb", type=float, default=400.0,
                    help="stop if filesystem free space drops below this (shared disk)")
    ap.add_argument("--max-raw-gb", type=float, default=300.0,
                    help="stop if this project's raw/ exceeds this")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--wait-on-cap", action="store_true",
                    help="pause for the tensorizer consumer instead of exiting "
                         "when the raw/ cap is hit (streaming pipeline mode)")
    ap.add_argument("--db", default=None, help="override state db path")
    args = ap.parse_args()

    global DB, RAW
    if args.db:
        DB = args.db
    if args.raw_dir:
        RAW = args.raw_dir

    if args.status:
        status(args)
        return

    con = init_db(args.manifest, args.curated_only, args.tiers, args.bulk_only)
    con_lock = threading.Lock()

    def handle(sig, frm):
        print("\n[signal] draining and shutting down cleanly...", flush=True)
        STOP.set()
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    raw_now = dir_bytes(RAW)
    print(f"raw/ currently {human(raw_now)}; caps: raw<={args.max_raw_gb}GB, "
          f"free>={args.min_free_gb}GB", flush=True)

    q = queue.Queue(maxsize=args.workers * 4)
    results = queue.Queue()
    threads = [threading.Thread(target=worker, args=(i, con_lock, q, results, args),
                               daemon=True) for i in range(args.workers)]
    for t in threads:
        t.start()

    t0 = time.time()
    n_done = n_err = 0
    bytes_done = 0
    last_report = last_guard = time.time()

    def drain(block=False):
        nonlocal n_done, n_err, bytes_done
        got = 0
        while True:
            try:
                log_id, st, nb, err = results.get(timeout=1 if block and got == 0 else 0)
            except queue.Empty:
                return got
            got += 1
            if st == "done":
                n_done += 1
                bytes_done += nb
                con.execute("UPDATE dl SET status='done', bytes=?, ts=? WHERE log_id=?",
                            (nb, time.time(), log_id))
            else:
                n_err += 1
                con.execute("UPDATE dl SET status=?, attempts=attempts+1, err=?, ts=? "
                            "WHERE log_id=?", (st, err, time.time(), log_id))

    while not STOP.is_set():
        batch = con.execute(
            "SELECT log_id, url FROM dl WHERE status='pending' "
            "OR (status IN ('error','http_error') AND attempts < ?) "
            "ORDER BY rank LIMIT 2000", (args.max_retries,)).fetchall()
        if not batch:
            break
        for log_id, url in batch:
            if STOP.is_set():
                break
            # mark in-flight so a crash does not lose the row's identity
            con.execute("UPDATE dl SET status='running' WHERE log_id=?", (log_id,))
            q.put((log_id, url))
            drain()

            now = time.time()
            if now - last_guard > 20:
                last_guard = now
                # Liveness: respawn any worker that died. A run this long must
                # not silently lose its consumers and stall with a full queue.
                for wi, t in enumerate(threads):
                    if not t.is_alive():
                        nt = threading.Thread(
                            target=worker, args=(wi, con_lock, q, results, args),
                            daemon=True)
                        nt.start()
                        threads[wi] = nt
                        print(f"[liveness] respawned dead worker {wi}", flush=True)
                du = shutil.disk_usage(ROOT)
                cur_raw = raw_now + bytes_done
                if du.free < args.min_free_gb * 1e9:
                    print(f"[GUARD] free {human(du.free)} < {args.min_free_gb}GB floor. "
                          "Stopping to protect shared disk.", flush=True)
                    STOP.set()
                    break
                if cur_raw > args.max_raw_gb * 1e9:
                    if args.wait_on_cap:
                        # Backpressure: the tensorizer consumer drains staging.
                        # Pause instead of exiting so the pipeline self-regulates.
                        waited = 0
                        while (not STOP.is_set()
                               and dir_bytes(RAW) > args.max_raw_gb * 1e9 * 0.8):
                            time.sleep(15)
                            waited += 15
                            if waited % 300 == 0:
                                print(f"[backpressure] staging full ({human(dir_bytes(RAW))}), "
                                      f"waited {waited}s for consumer", flush=True)
                        # staging drained: recompute our byte baseline
                        raw_now = dir_bytes(RAW)
                        bytes_done = 0
                    else:
                        print(f"[GUARD] raw/ {human(cur_raw)} > {args.max_raw_gb}GB cap. Stopping.",
                              flush=True)
                        STOP.set()
                        break
            if now - last_report > 30:
                last_report = now
                con.commit()
                el = now - t0
                print(f"[{time.strftime('%H:%M:%S')}] done={n_done} err={n_err} "
                      f"{human(bytes_done)} rate={n_done/el*60:.0f}/min "
                      f"free={human(shutil.disk_usage(ROOT).free)}", flush=True)
        con.commit()

    # Retire workers with sentinels rather than relying on an empty-queue
    # timeout, which is what deadlocked the first bulk run.
    for _ in threads:
        try:
            q.put(None, timeout=5)
        except Exception:
            pass
    for t in threads:
        t.join(timeout=30)
    for _ in range(20):
        if drain() == 0:
            break
        time.sleep(0.5)
    # any row still 'running' at exit is genuinely unfinished -> re-queue later
    con.execute("UPDATE dl SET status='pending' WHERE status='running'")
    con.commit()
    el = time.time() - t0
    print(f"\nEXIT after {el/3600:.2f}h: done={n_done} err={n_err} {human(bytes_done)} "
          f"rate={n_done/el*60 if el else 0:.0f}/min", flush=True)


if __name__ == "__main__":
    main()
