#!/usr/bin/env python
"""Tier-2, stage 1: pooled encoder embeddings on the Tier-1 scoring grid.

PROJECT_PLAN §7 ("Proposed": pretrain, then linear probe and fine-tune). This
module turns the frozen masked-reconstruction encoder into a feature extractor
and runs it over the *evaluation* tier (curated shards).

Why this exists as a separate, cached stage
-------------------------------------------
The encoder costs ~118 windows/s per 3090, and the evaluation grid is 6,087,288
windows -- ~7.2 h on both GPUs. That is far too expensive to repeat per split,
per seed, per head. It is also *fixed*: the representation does not depend on
which split a flight lands in. So it is computed once, cached, and every probe,
ablation and seed afterwards is a cheap fit on the cached vectors.

Comparability is the whole point, so the grid is Tier-1's grid, exactly
----------------------------------------------------------------------
`tier1_baseline.process_shard` scores each flight at `t in arange(WARMUP, end,
STRIDE)` where `end = min(t_e, n_bins)` on positives (strictly pre-event) and
`n_bins` on negatives, over flights that pass the trigger-v2 exclusion and
`n_bins >= MIN_BINS`. Those constants are *imported* rather than restated, so
the two tiers cannot drift apart. Every downstream number therefore lands on
the same alarm grid as GBT 0.708 / 0.360 (§7.1), and `flight_curve` / `op_point`
can be reused unmodified.

Tier-1's feature window is (t-100, t]; the encoder's is (t-512, t] -- 25.6 s
rather than 5 s. Those are different *models*, not different protocols: the
alarm opportunities, the time-at-risk accounting and the reference FAR are
identical. For the ~6 positions per flight where t+1 < 512 the window is
LEFT-PADDED with cells marked invalid, rather than dropped, so the grid stays
byte-for-byte Tier-1's. The encoder was trained with a validity mask and drops
absent cells from attention, so a partly-empty window is in-distribution for it.

Output
------
One parquet per (shard, part): `log_id`, `t_bin`, and two pooled vectors,
    emb_mean : mean over all valid (channel, patch) tokens
    emb_last : mean over valid tokens in the final 4 patches (last 3.2 s)
stored float16. fp16 is safe *here* and nowhere else in this codebase: these are
unit-scale post-LayerNorm activations, not physical units. The standing rule
about fp16 (`shard_io.py`: baro_pressure_pa overflows at 65504) is about raw
channel storage and still holds.

Memory
------
A curated shard is ONE Parquet row group of ~6.4 GiB, so each dataloader worker
that touches a shard costs a fixed ~6.4 GiB resident -- the same trap that OOM'd
pretraining twice on the bulk shards. Size as `workers x 6.4 GiB + ~4 GiB` and
do not raise `--workers` past 6 per process on this box. The GPU is the
bottleneck anyway (window construction is ~40 MB/s), so 3-4 is plenty.

Usage
-----
    python embed_eval.py --part 0 --of 2 --gpu 0     # one process per GPU
    python embed_eval.py --pilot 400                 # ~25 min smoke run
    python embed_eval.py --status
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_io import decode
from pretrain_model import build_model
from pretrain_data import load_channel_stats
# Grid constants come from Tier-1 itself; never restate them here.
from tier1_baseline import STRIDE, WARMUP, MIN_BINS, EXCLUDE

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")
PROC = os.path.join(ROOT, "proc")
OUT = os.path.join(META, "eval_emb")
CKPT = os.path.join(ROOT, "checkpoints", "pretrain_last.pt")

LAST_K = 4          # final 4 patches = last 3.2 s, the "recent dynamics" pool


def curated_shards():
    return sorted(glob.glob(os.path.join(PROC, "curated-*.parquet")))


def load_exclude():
    if not os.path.exists(EXCLUDE):
        return set()
    import pandas as pd
    return set(pd.read_csv(EXCLUDE).log_id)


class ShardEvalWindows(IterableDataset):
    """Yields ready-made batches of windows for one shard, on Tier-1's grid.

    Batches are assembled in the worker (not by a collate_fn) because a flight
    produces ~450 consecutive windows: building them as one strided gather and
    slicing costs a fraction of what per-window Python would.
    """

    def __init__(self, shard, mean, std, exclude, window, batch,
                 part=0, nparts=1, limit_flights=None, keep_ids=None):
        self.shard, self.mean, self.std = shard, mean, std
        self.exclude = exclude
        self.window, self.batch = window, batch
        self.part, self.nparts = part, nparts
        self.limit_flights = limit_flights
        # keep_ids restricts the stream to one split arm (finetune.py scores only
        # the test arm). None = the whole shard, which is what the embedding pass
        # wants since its cache is split-agnostic.
        self.keep_ids = set(keep_ids) if keep_ids is not None else None

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        # Flights are dealt round-robin: first across parts (GPUs), then across
        # workers. Both arms of every split therefore land in every part, so a
        # partial run is still a usable (if smaller) evaluation set.
        pf = pq.ParquetFile(self.shard)
        idx = -1
        done = 0
        for rb in pf.iter_batches(batch_size=16):
            for row in rb.to_pylist():
                if row["log_id"] in self.exclude:
                    continue
                if self.keep_ids is not None and row["log_id"] not in self.keep_ids:
                    continue
                nb = row["n_bins"]
                if nb is None or nb < MIN_BINS:
                    continue
                idx += 1
                if idx % self.nparts != self.part:
                    continue
                if (idx // self.nparts) % nw != wid:
                    continue
                if self.limit_flights and done >= self.limit_flights:
                    return
                done += 1
                try:
                    x, v = decode(row)
                except Exception:
                    continue
                yield from self._flight(row, x, v)

    def _flight(self, row, x, v):
        t_e, nb = int(row["t_e_bin"]), int(row["n_bins"])
        end = min(t_e, nb) if t_e >= 0 else nb
        ts = np.arange(WARMUP, end, STRIDE)
        if not len(ts):
            return
        W = self.window
        xn = np.where(v, (x - self.mean) / self.std, 0.0).astype(np.float32)

        lo = ts + 1 - W                       # window is (t-W, t]
        for i in range(0, len(ts), self.batch):
            tb, lb = ts[i:i + self.batch], lo[i:i + self.batch]
            xb = np.zeros((len(tb), W, xn.shape[1]), dtype=np.float32)
            vb = np.zeros((len(tb), W, xn.shape[1]), dtype=bool)
            for j, l in enumerate(lb):
                if l >= 0:
                    xb[j] = xn[l:l + W]
                    vb[j] = v[l:l + W]
                else:
                    # short prefix: LEFT-pad, cells stay invalid (see docstring)
                    k = l + W
                    xb[j, W - k:] = xn[:k]
                    vb[j, W - k:] = v[:k]
            yield (row["log_id"], torch.from_numpy(tb.astype(np.int32)),
                   torch.from_numpy(xb), torch.from_numpy(vb))


@torch.no_grad()
def pool(model, x, v, last_k=LAST_K):
    """(B,T,C) -> two (B,d) pooled vectors over VALID tokens only."""
    _, vp = model.to_patches(x, v)                  # (B,C,P,patch)
    tok = vp.any(dim=-1).unsqueeze(-1).float()      # (B,C,P,1) 1 where token has data
    h = model.encode(x, v).float()                  # (B,C,P,d)
    n = tok.sum(dim=(1, 2)).clamp(min=1.0)
    emb_mean = (h * tok).sum(dim=(1, 2)) / n
    tl = tok.clone()
    tl[:, :, :-last_k] = 0.0
    nl = tl.sum(dim=(1, 2)).clamp(min=1.0)
    emb_last = (h * tl).sum(dim=(1, 2)) / nl
    return emb_mean, emb_last


def write_part(path, log_ids, t_bins, em, el, d):
    tbl = pa.table({
        "log_id": pa.array(log_ids, type=pa.string()),
        "t_bin": pa.array(np.concatenate(t_bins), type=pa.int32()),
        "emb_mean": pa.FixedSizeListArray.from_arrays(
            pa.array(np.concatenate(em).reshape(-1), type=pa.float16()), d),
        "emb_last": pa.FixedSizeListArray.from_arrays(
            pa.array(np.concatenate(el).reshape(-1), type=pa.float16()), d),
    })
    tmp = path + ".tmp"
    pq.write_table(tbl, tmp, compression="zstd")
    os.replace(tmp, path)                      # atomic: a killed job leaves no half-part


def run(args):
    os.makedirs(OUT, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg)
    sd = {k.replace("module.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model = model.to(dev).eval()
    d = cfg["d"]
    print(f"encoder: step {ck['step']} epoch {ck['epoch']} d={d} window={cfg['window']}",
          flush=True)

    mean, std = load_channel_stats()
    exclude = load_exclude()
    shards = curated_shards()
    t_start = time.time()
    n_done = 0

    for shard in shards:
        base = os.path.basename(shard).replace(".parquet", "")
        tag = f"{base}.p{args.part}" + (f".pilot{args.pilot}" if args.pilot else "")
        out = os.path.join(OUT, f"emb-{tag}.parquet")
        if os.path.exists(out):
            print(f"  {tag}: cached", flush=True)
            continue
        ds = ShardEvalWindows(shard, mean, std, exclude, cfg["window"], args.batch,
                              part=args.part, nparts=args.of,
                              limit_flights=args.pilot)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=args.workers,
            pin_memory=True, prefetch_factor=4 if args.workers else None)
        log_ids, t_bins, em, el = [], [], [], []
        t0, n = time.time(), 0
        for lid, tb, xb, vb in dl:
            xb = xb.to(dev, non_blocking=True)
            vb = vb.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = pool(model, xb, vb)
            log_ids.extend([lid] * len(tb))
            t_bins.append(tb.numpy())
            em.append(a.half().cpu().numpy())
            el.append(b.half().cpu().numpy())
            n += len(tb)
            if n % 20000 < args.batch:
                r = n / max(time.time() - t0, 1e-9)
                print(f"  {tag}: {n:,} windows  {r:.0f}/s", flush=True)
        if not t_bins:
            print(f"  {tag}: no windows", flush=True)
            continue
        write_part(out, log_ids, t_bins, em, el, d)
        n_done += n
        print(f"  {tag}: DONE {n:,} windows in {(time.time()-t0)/60:.1f} min",
              flush=True)

    print(f"\npart {args.part}: {n_done:,} windows in "
          f"{(time.time()-t_start)/3600:.2f} h", flush=True)


GRID = 6_087_288        # tier1_baseline's window count -- the denominator


def status():
    """Pilot parts are counted separately: they are a *subsample* of the same
    grid, so folding them into the total would overstate progress."""
    tot = {False: 0, True: 0}
    cnt = {False: 0, True: 0}
    for p in sorted(glob.glob(os.path.join(OUT, "emb-*.parquet"))):
        pilot = "pilot" in p
        n = pq.ParquetFile(p).metadata.num_rows
        tot[pilot] += n
        cnt[pilot] += 1
        print(f"  {os.path.basename(p):48s} {n:>10,}{'  (pilot)' if pilot else ''}")
    if cnt[True]:
        print(f"{cnt[True]} pilot parts, {tot[True]:,} windows (smoke run, not progress)")
    print(f"{cnt[False]} parts, {tot[False]:,} windows "
          f"({100*tot[False]/GRID:.1f}% of the {GRID:,}-window grid)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pilot", type=int, default=0,
                    help="flights per (shard,worker); smoke run, separate output")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        return status()
    run(args)


if __name__ == "__main__":
    main()
