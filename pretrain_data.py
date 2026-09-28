#!/usr/bin/env python
"""Window sampler over the tensor shards + per-channel statistics.

PROJECT_PLAN §7 ("Proposed"): masked-reconstruction pretraining over the full
bulk corpus. This module supplies the windows.

Design notes that are load-bearing:

*   **Streaming, shard-at-a-time.** Random access into 215 GB of Parquet would
    be dominated by row-group decompression. Instead each dataloader worker
    takes a disjoint slice of the shard list, streams row batches, and samples
    a few windows per flight. Shard order and window offsets are reshuffled
    every epoch from a seed, so the sample is not fixed across epochs.
*   **Masks are never zero-filled.** Invalid cells arrive as exactly 0.0 from
    `shard_io.decode` and stay 0.0 after normalisation, but the validity mask
    travels with every window and the loss only ever reads valid cells.
    See PROJECT_PLAN §6.1 -- zero is a physically meaningful value here.
*   **Channel presence is per-window, not per-log.** A channel that exists in
    the log but is dropped out over this particular 25.6 s is treated as absent
    for that window: excluded from channel attention and from the loss. That is
    the variable-channel-set problem the encoder is meant to solve, so it must
    be visible in the data.
*   **Eval-vehicle exclusion (default ON).** Any `vehicle_key` appearing in the
    curated tier -- the downstream evaluation set -- is held out of pretraining.
    Self-supervised pretraining on eval vehicles is defensible in principle, but
    it is exactly the kind of thing a reviewer will ask about, and the corpus is
    large enough that the exclusion costs nothing.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_io import decode
from channels import N_CHANNELS, TARGET_HZ

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")
PROC = os.path.join(ROOT, "proc")
STATS_PATH = os.path.join(META, "channel_stats.json")

WINDOW = 512                # 25.6 s at 20 Hz
PATCH = 16                  # 0.8 s -> 32 time patches per window
MIN_VALID_FRAC = 0.20       # skip windows that are mostly dropout
MIN_PRESENT_CH = 8          # skip windows with almost no channels
WINDOWS_PER_FLIGHT = 4      # cap, so long flights do not dominate a shard


# --------------------------------------------------------------- rosters ----
def eval_vehicle_keys():
    """vehicle_keys used by the downstream (curated-tier) evaluation."""
    man_path = os.path.join(META, "manifest.parquet")
    if not os.path.exists(man_path):
        return set()
    import pandas as pd
    cur_ids = set()
    for f in sorted(glob.glob(os.path.join(PROC, "curated-*.parquet"))):
        cur_ids |= set(pq.read_table(f, columns=["log_id"])["log_id"].to_pylist())
    man = pd.read_parquet(man_path, columns=["log_id", "vehicle_key"])
    keys = set(man[man.log_id.isin(cur_ids)].vehicle_key.dropna())
    keys.discard("")
    return keys


def bulk_shards():
    return sorted(glob.glob(os.path.join(PROC, "bulk-*.parquet")))


# ------------------------------------------------------------ statistics ----
def compute_channel_stats(shards, max_flights=20000, seed=0, per_shard=None):
    """Per-channel mean/std over valid cells, streamed (Welford, not a big load).

    Spread across shards rather than draining them in order: a shard is roughly
    a contiguous slice of upload history, so taking 20k flights from the first
    ten shards would normalise the corpus by whatever was uploaded that week.
    """
    n = np.zeros(N_CHANNELS, dtype=np.int64)
    mean = np.zeros(N_CHANNELS, dtype=np.float64)
    m2 = np.zeros(N_CHANNELS, dtype=np.float64)
    seen = 0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(shards))
    if per_shard is None:
        per_shard = max(1, int(np.ceil(max_flights / max(len(shards), 1))))
    for si in order:
        pf = pq.ParquetFile(shards[si])
        taken = 0
        for rb in pf.iter_batches(batch_size=32):
            for row in rb.to_pylist():
                if row["n_bins"] is None or row["n_bins"] < WINDOW:
                    continue
                x, v = decode(row)
                taken += 1
                # chunked Welford per channel over valid cells
                for c in range(N_CHANNELS):
                    vc = v[:, c]
                    k = int(vc.sum())
                    if k == 0:
                        continue
                    xs = x[vc, c].astype(np.float64)
                    bm = xs.mean()
                    bv = xs.var()
                    delta = bm - mean[c]
                    tot = n[c] + k
                    mean[c] += delta * k / tot
                    m2[c] += bv * k + delta * delta * n[c] * k / tot
                    n[c] = tot
                seen += 1
            if seen >= max_flights or taken >= per_shard:
                break
        if seen >= max_flights:
            break
    std = np.sqrt(np.where(n > 1, m2 / np.maximum(n - 1, 1), 1.0))
    # A channel with no observations or no spread must not scale anything to
    # infinity; 1.0 leaves it untouched and the mask keeps it out of the loss.
    std = np.where((n < 100) | (std < 1e-6), 1.0, std)
    mean = np.where(n < 100, 0.0, mean)
    return {"mean": mean.tolist(), "std": std.tolist(), "count": n.tolist(),
            "n_flights": seen, "window": WINDOW}


def load_channel_stats():
    if not os.path.exists(STATS_PATH):
        raise SystemExit(
            f"missing {STATS_PATH}\nRun:  python pretrain_data.py --stats")
    with open(STATS_PATH) as f:
        s = json.load(f)
    return (np.asarray(s["mean"], dtype=np.float32),
            np.asarray(s["std"], dtype=np.float32))


# --------------------------------------------------------------- dataset ----
class ShardWindows(IterableDataset):
    """Yields (x, valid) windows: x float32 (WINDOW, C) z-scored, valid bool."""

    def __init__(self, shards, mean, std, epoch=0, seed=1234,
                 exclude_log_ids=None, window=WINDOW,
                 windows_per_flight=WINDOWS_PER_FLIGHT, limit_flights=None):
        self.shards = list(shards)
        self.mean, self.std = mean, std
        self.epoch, self.seed = epoch, seed
        self.exclude = exclude_log_ids or set()
        self.window = window
        self.wpf = windows_per_flight
        self.limit_flights = limit_flights

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng((self.seed, self.epoch, wid))
        order = rng.permutation(len(self.shards))
        mine = [self.shards[i] for i in order[wid::nw]]
        produced = 0
        for path in mine:
            try:
                pf = pq.ParquetFile(path)
            except Exception:
                continue                      # a truncated shard must not kill training
            for rb in pf.iter_batches(batch_size=32):
                for row in rb.to_pylist():
                    if row["log_id"] in self.exclude:
                        continue
                    nb = row["n_bins"]
                    if nb is None or nb < self.window:
                        continue
                    try:
                        x, v = decode(row)
                    except Exception:
                        continue
                    k = min(self.wpf, max(1, nb // self.window))
                    starts = rng.choice(nb - self.window + 1, size=k, replace=False) \
                        if nb - self.window + 1 >= k else [0]
                    for s0 in starts:
                        xs = x[s0:s0 + self.window]
                        vs = v[s0:s0 + self.window]
                        if vs.mean() < MIN_VALID_FRAC:
                            continue
                        if int(vs.any(axis=0).sum()) < MIN_PRESENT_CH:
                            continue
                        xn = (xs - self.mean) / self.std
                        xn = np.where(vs, xn, 0.0).astype(np.float32)
                        yield torch.from_numpy(xn), torch.from_numpy(vs)
                        produced += 1
                        if self.limit_flights and produced >= self.limit_flights:
                            return


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    vs = torch.stack([b[1] for b in batch])
    return xs, vs


def make_loader(batch_size=32, workers=8, epoch=0, exclude_eval_vehicles=True,
                limit_flights=None, seed=1234):
    mean, std = load_channel_stats()
    shards = bulk_shards()
    if not shards:
        raise SystemExit(f"no bulk shards under {PROC}")
    exclude = set()
    if exclude_eval_vehicles:
        p = os.path.join(META, "pretrain_excluded_logs.json")
        if os.path.exists(p):
            with open(p) as f:
                exclude = set(json.load(f))
        else:
            print("  (no pretrain_excluded_logs.json; run --build-exclude)",
                  flush=True)
    ds = ShardWindows(shards, mean, std, epoch=epoch, seed=seed,
                      exclude_log_ids=exclude, limit_flights=limit_flights)
    # persistent_workers MUST stay False: workers get a *copy* of the dataset at
    # spawn, so a persistent worker would never see ds.set_epoch() and every
    # epoch would replay the same shard order and window offsets. Respawning
    # workers per epoch costs a few seconds and keeps the reshuffle real.
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=workers, collate_fn=collate,
        pin_memory=True, prefetch_factor=4 if workers else None,
        persistent_workers=False), ds


# ------------------------------------------------------------------ main ----
def build_exclude():
    """Materialise the list of bulk log_ids sharing a vehicle with the eval tier."""
    import pandas as pd
    keys = eval_vehicle_keys()
    man = pd.read_parquet(os.path.join(META, "manifest.parquet"),
                          columns=["log_id", "vehicle_key"])
    bulk_ids = set()
    for f in bulk_shards():
        bulk_ids |= set(pq.read_table(f, columns=["log_id"])["log_id"].to_pylist())
    hit = man[man.vehicle_key.isin(keys) & man.log_id.isin(bulk_ids)]
    out = sorted(set(hit.log_id))
    p = os.path.join(META, "pretrain_excluded_logs.json")
    with open(p, "w") as f:
        json.dump(out, f)
    print(f"{len(keys)} eval vehicle_keys -> {len(out)} bulk logs excluded "
          f"({100*len(out)/max(len(bulk_ids),1):.2f}% of {len(bulk_ids)}) -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="compute channel stats")
    ap.add_argument("--build-exclude", action="store_true")
    ap.add_argument("--max-flights", type=int, default=20000)
    ap.add_argument("--peek", type=int, default=0, help="yield N windows and report")
    args = ap.parse_args()

    if args.build_exclude:
        build_exclude()
        return
    if args.stats:
        shards = bulk_shards()
        print(f"computing channel stats over <= {args.max_flights} flights "
              f"from {len(shards)} shards ...", flush=True)
        s = compute_channel_stats(shards, max_flights=args.max_flights)
        with open(STATS_PATH, "w") as f:
            json.dump(s, f, indent=1)
        cnt = np.asarray(s["count"])
        print(f"wrote {STATS_PATH}  ({s['n_flights']} flights)")
        print(f"  channels with >=100 valid cells: {(cnt >= 100).sum()}/{N_CHANNELS}")
        return
    if args.peek:
        loader, _ = make_loader(batch_size=8, workers=2, limit_flights=args.peek)
        for i, (x, v) in enumerate(loader):
            print(f"batch {i}: x{tuple(x.shape)} {x.dtype}  valid_frac "
                  f"{v.float().mean():.3f}  present_ch "
                  f"{v.any(dim=1).sum(dim=1).float().mean():.1f}")
            if i >= 2:
                break
        return
    ap.print_help()


if __name__ == "__main__":
    main()
