#!/usr/bin/env python
"""Tier-1.5: an open time-series foundation model as a frozen feature extractor.

Why this exists (PROJECT_PLAN 7.3, "REQUIRED")
---------------------------------------------
The first question a reviewer asks is *why train a 50.5 M-parameter encoder from
scratch instead of using an existing TSFM?* The project's answer -- none of the open
models saw 20 Hz vehicle telemetry with ragged channel sets -- has been an assertion
with no experiment behind it. This file supplies the experiment.

The design is deliberately the SAME SUBSTITUTION as `horizon_tier1.py`
---------------------------------------------------------------------
`horizon_tier1.py` established the control pattern for this project: take the working
pipeline and change exactly one thing -- the feature matrix. It imports the label
bands and the eval path rather than restating them, so the swap is exact.

This file does the same at the stage before: it imports `ShardEvalWindows` from
`embed_eval.py`, so the window grid, the (t-512, t] window, the left-padding of short
prefixes, the exclusion list and the normalisation are *the same objects*, not
reimplementations. Only the encoder changes. The output parquet has `embed_eval`'s
schema, so `probe.py` / `horizon.py` consume it through `--emb-dir` with no other
change, and every downstream head, seed, band and split is then identical by
construction.

Frozen, not fine-tuned, and that is the right first comparison
--------------------------------------------------------------
RESULTS.md 4.2 found that fine-tuning this project's own encoder end-to-end is a NULL
result and lands *below* the frozen probe -- freezing wins at this label budget. So the
apples-to-apples question is "frozen TSFM features vs frozen our-encoder features vs
hand-built statistics, same heads, same grid", which is what this runs. Fine-tuning the
TSFM is a strictly later question, and a more expensive one.

Backends
--------
  moment   AutonLab/MOMENT-1-{small,base,large}. Primary. Its native context length is
           **512**, which is exactly this project's window, so no resampling or
           re-chunking is needed -- a like-for-like comparison rather than an adapted
           one. It is patch-based with channel independence, which is the interesting
           contrast: our encoder uses factorised channel attention over the same grid.
  moirai   Salesforce/moirai-1.1-R-{small,base}. Optional. Any-variate attention, the
           closest architectural match to the ragged-channel problem, but it flattens
           variates into one sequence (87 x 16 patches = 1,392 tokens) and needs uni2ts.

Both are run FROZEN in eval mode under `torch.no_grad()`, bf16 autocast.

Channels that are absent from a flight
--------------------------------------
`ShardEvalWindows` yields `x` already zero-filled where `valid` is False (it z-scores
with the training statistics and writes 0 elsewhere), plus the boolean mask. Our own
encoder consumes the mask and drops absent cells from attention. A TSFM has no such
input, so an absent channel is fed to it as a constant-zero series -- which is exactly
the handicap the paper is claiming these models have on ragged telemetry, and it must be
reported as the TSFM's real operating condition rather than engineered away. The
per-window valid fraction is recorded so the effect can be quantified afterwards.

Cost
----
Measure it, do not guess it: `--pilot N` runs N flights, prints windows/s and the
extrapolated wall clock for the full 6,087,288-window grid on 1 and 2 GPUs, and exits
without writing a cache. Run that BEFORE launching the full pass.

Usage
-----
    python tsfm_baseline.py --pilot 40                     # throughput probe, ~minutes
    python tsfm_baseline.py --part 0 --of 2 --gpu 0        # one half of the grid
    python tsfm_baseline.py --status
"""
from __future__ import annotations

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_eval import (ShardEvalWindows, curated_shards, load_exclude,   # noqa: E402
                        META, LAST_K)
from pretrain_data import load_channel_stats                              # noqa: E402
from pretrain_data import WINDOW                                          # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(META, "tsfm_emb")
STATE = os.path.join(ROOT, "state")

MODELS = {
    "moment-small": "AutonLab/MOMENT-1-small",
    "moment-base": "AutonLab/MOMENT-1-base",
    "moment-large": "AutonLab/MOMENT-1-large",
    "moirai-small": "Salesforce/moirai-1.1-R-small",
    "moirai-base": "Salesforce/moirai-1.1-R-base",
}


# --------------------------------------------------------------------- backends
class MomentBackend:
    """MOMENT in `embedding` mode, run CHANNEL-INDEPENDENTLY so we can mask.

    Fed a multivariate (B, C, 512) tensor, MOMENT returns (B, D): it pools the
    channels itself, which would silently average in the ~60% of the 87 channels a
    given flight does not log. Those arrive as constant zeros, so that pooling is a
    dilution the model never asked for and a reviewer would rightly object to.

    MOMENT is channel-independent internally, so reshaping to (B*C, 1, 512) and
    pooling ourselves costs THE SAME -- measured on a 3090: 83.0 windows/s
    multivariate against 80.5 channel-independent, i.e. within noise. We therefore
    take the channel-independent path and mean-pool over the channels that are
    actually present, which is exactly what `embed_eval`'s `emb_mean` does over
    valid (channel, patch) tokens. Same summary, computed the same way, so the two
    feature vectors are comparable rather than merely both 512-dimensional.
    """

    name = "moment"

    def __init__(self, repo, device, dtype=torch.bfloat16):
        import warnings
        from momentfm import MOMENTPipeline
        with warnings.catch_warnings():
            # "only the reconstruction head is pre-trained" -- expected and correct:
            # we want the encoder, and no head is used in embedding mode.
            warnings.simplefilter("ignore")
            self.model = MOMENTPipeline.from_pretrained(
                repo, model_kwargs={"task_name": "embedding"})
            self.model.init()
        self.model = self.model.to(device).eval()
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def embed(self, xb, vb):
        B, W, C = xb.shape
        x = xb.permute(0, 2, 1).reshape(B * C, 1, W).contiguous()
        x = x.to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=self.dtype):
            out = self.model(x_enc=x)
        e = out.embeddings.float().view(B, C, -1)
        present = vb.any(dim=1).to(e.device).float().unsqueeze(-1)   # (B, C, 1)
        return (e * present).sum(dim=1) / present.sum(dim=1).clamp(min=1.0)


class MoiraiBackend:
    """Moirai's encoder hidden states, mean-pooled over (variate, patch) tokens."""

    name = "moirai"

    def __init__(self, repo, device, dtype=torch.bfloat16, patch=32):
        from uni2ts.model.moirai import MoiraiModule
        self.module = MoiraiModule.from_pretrained(repo).to(device).eval()
        self.device, self.dtype, self.patch = device, dtype, patch

    @torch.no_grad()
    def embed(self, xb, vb):
        raise SystemExit(
            "moirai backend is a stub: uni2ts exposes a forecasting head, and pulling "
            "encoder states out of MoiraiModule needs a version-specific path. Use "
            "--model moment-* unless that has been implemented and CONTROLLED against "
            "a known output.")


def build_backend(key, device):
    repo = MODELS[key]
    if key.startswith("moment"):
        return MomentBackend(repo, device)
    return MoiraiBackend(repo, device)


# --------------------------------------------------------------------- driver
def run(args):
    device = f"cuda:{args.gpu}"
    torch.backends.cuda.matmul.allow_tf32 = True
    mean, std = load_channel_stats()
    exclude = load_exclude()
    shards = curated_shards()
    print(f"{len(shards)} curated shards, {len(exclude)} excluded flights", flush=True)

    t0 = time.time()
    be = build_backend(args.model, device)
    print(f"backend {args.model} ready in {time.time() - t0:.1f}s", flush=True)

    os.makedirs(OUT, exist_ok=True)
    n_done, n_valid_sum, n_cells = 0, 0.0, 0
    t_start = time.time()

    for si, shard in enumerate(shards):
        tag = os.path.basename(shard).replace(".parquet", "")
        suffix = f".pilot{args.pilot}" if args.pilot else ""
        # Named `emb-*` on purpose: `probe.load_embeddings` globs that pattern, so
        # pointing FLIGHTLOG_EMB_DIR here makes every downstream head, split and seed
        # work with no code change at all. The model is recorded in the directory
        # name (`meta/tsfm_emb`) and in the manifest, not in the file name.
        outp = os.path.join(OUT, f"emb-{tag}.p{args.part}{suffix}.parquet")
        if os.path.exists(outp) and not args.pilot:
            print(f"  {tag} p{args.part}: cached, skip", flush=True)
            continue

        ds = ShardEvalWindows(shard, mean, std, exclude, WINDOW, args.batch,
                              part=args.part, nparts=args.of,
                              limit_flights=args.pilot or None)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=args.workers,
            pin_memory=True, prefetch_factor=2 if args.workers else None)

        lids, tbs, embs = [], [], []
        for log_id, tb, xb, vb in loader:
            e = be.embed(xb, vb).cpu().numpy().astype(np.float16)
            lids += [log_id] * len(tb)
            tbs.append(tb.numpy())
            embs.append(e)
            n_done += len(tb)
            n_valid_sum += float(vb.float().mean()) * len(tb)
            n_cells += len(tb)
            if n_done % (args.batch * 50) < args.batch:
                el = time.time() - t_start
                print(f"    {n_done:,} windows  {n_done / el:.1f}/s  "
                      f"valid {n_valid_sum / max(n_cells, 1) * 100:.1f}%", flush=True)
            if args.pilot and n_done >= args.pilot_windows:
                break

        if args.pilot:
            break

        if not lids:
            print(f"  {tag} p{args.part}: no windows", flush=True)
            continue
        tbl = pa.table({"log_id": pa.array(lids),
                        "t_bin": pa.array(np.concatenate(tbs)),
                        "emb_mean": pa.array(list(np.concatenate(embs)))})
        pq.write_table(tbl, outp, compression="zstd")
        with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
            json.dump({"model": args.model, "repo": MODELS[args.model],
                       "window": WINDOW, "pooling": "mean over present channels",
                       "schema": "log_id, t_bin, emb_mean (float16)",
                       "views": ["emb_mean"],
                       "note": "consume with FLIGHTLOG_EMB_DIR=meta/tsfm_emb and "
                               "--views emb_mean"}, f, indent=2)
        print(f"  wrote {os.path.basename(outp)}  {len(lids):,} windows", flush=True)

    el = time.time() - t_start
    rate = n_done / max(el, 1e-9)
    print(f"\n{n_done:,} windows in {el / 60:.1f} min  =  {rate:.1f} windows/s")
    print(f"mean valid-cell fraction {n_valid_sum / max(n_cells, 1) * 100:.1f}% "
          f"(absent channels are fed as constant zeros -- see the module docstring)")
    if args.pilot:
        grid = 6_087_288
        print(f"\nEXTRAPOLATION to the full {grid:,}-window grid:")
        for ng in (1, 2):
            h = grid / (rate * ng) / 3600
            print(f"  {ng} GPU: {h:6.1f} h  ({h / 24:.1f} days)")
        print("\nOur own encoder ran this grid at ~118 windows/s per 3090 (~7.2 h on "
              "both). Compare before committing.")
        print("Pilot mode wrote NO cache.")


def status():
    parts = sorted(glob.glob(os.path.join(OUT, "emb-*.parquet")))
    if not parts:
        print("  no TSFM embeddings yet")
        return
    n = 0
    for p in parts:
        m = pq.ParquetFile(p).metadata
        n += m.num_rows
        print(f"  {os.path.basename(p):58s} {m.num_rows:>10,}")
    print(f"  {len(parts)} parts, {n:,} windows "
          f"({n / 6_087_288 * 100:.1f}% of the 6,087,288-window grid)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="moment-small", choices=sorted(MODELS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=3,
                    help="a curated shard is ONE ~6.4 GiB row group, so each worker "
                         "costs a fixed ~6.4 GiB resident. Do not exceed 6.")
    ap.add_argument("--pilot", type=int, default=0,
                    help="flights per shard to probe throughput; writes no cache")
    ap.add_argument("--pilot-windows", type=int, default=6000)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        status()
        return
    run(args)


if __name__ == "__main__":
    main()
