#!/usr/bin/env python
"""Item 6: richer pooled read-outs from the SAME frozen checkpoint (PROJECT_PLAN §11.0.1).

The pooling diagnosis: `embed_eval.py` averages 87 x 32 tokens into one vector. Loss of
control is 3-5 channels diverging while ~70 stay nominal, so the mean divides the contrast
by ~20 -- and the r ~ 0.98 between `emb_mean` and `emb_last` says pooling has erased both
channel and temporal localisation. Tier-1 GBT wins at a boosted head because its 5 x 87
statistics are NOT pooled and a tree can pick the diverging channel. This file emits views
that keep that localisation, from the identical encoder pass, so the probe-vs-GBT gap
(RESULTS.md §4.8, §4.10 finding 3) can be tested as a read-out artifact. No retraining.

Views per window (all float16, d = 512 unless stated):
    emb_mean      mean over all valid tokens              -- CONTROL, must reproduce eval_emb
    emb_last      mean over valid tokens, last 4 patches  -- as before
    emb_role      per control-ROLE mean over the last 4 patches, 8 roles concatenated (8 x d)
    emb_chmax     element-wise MAX over channels of the per-channel last-4-patch mean
    emb_dev       the last-4-patch mean of the ONE channel that deviates most from its own
                  window-long mean (the "most diverging channel" token)
    dev_norm      per-channel deviation magnitudes, 87 values (0 for absent channels)

`emb_mean`/`emb_last` are computed with the same code path as embed_eval.pool and are the
reproduction gate: `--check` compares them against the cached `meta/eval_emb` part.

Consumed through the verified env hooks, e.g.
    FLIGHTLOG_EMB_DIR=meta/eval_emb_views FLIGHTLOG_FEATURE_TAG=role \
        python horizon.py --split C --mode shift --h 120 --views emb_role,emb_last --head lr

Usage:
    python embed_views.py --part 0 --of 2 --gpu 0        # one process per GPU
    python embed_views.py --pilot 200 --gpu 1 --check    # smoke + reproduction gate
    python embed_views.py --status
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import CHANNELS, N_CHANNELS                              # noqa: E402
from embed_eval import (ShardEvalWindows, curated_shards, load_exclude,   # noqa: E402
                        LAST_K, CKPT, META)
from pretrain_model import build_model                                  # noqa: E402
from pretrain_data import load_channel_stats                            # noqa: E402

OUT = os.path.join(META, "eval_emb_views")

ROLES = {
    "attitude": ("vehicle_attitude.", "vehicle_attitude_setpoint."),
    "rates": ("vehicle_angular_velocity.", "vehicle_rates_setpoint."),
    "position": ("vehicle_local_position.", "vehicle_local_position_setpoint."),
    "imu": ("sensor_combined.",),
    "actuator": ("actuator_motors.", "actuator_outputs."),
    "est_health": ("estimator_status.",),
    "innovation": ("estimator_innovations.",),
    "exogenous": ("battery_status.", "vehicle_air_data.", "cpuload."),
}


def role_matrix():
    """(n_roles, C) 0/1 membership; asserts every channel has exactly one role."""
    names = [f"{t}.{f}" for t, f in CHANNELS]
    M = np.zeros((len(ROLES), N_CHANNELS), np.float32)
    for r, (_, prefixes) in enumerate(ROLES.items()):
        for c, n in enumerate(names):
            if n.startswith(prefixes):
                M[r, c] = 1.0
    assert (M.sum(0) == 1).all(), "every channel must belong to exactly one role"
    return M


@torch.no_grad()
def views(model, x, v, R, last_k=LAST_K):
    """(B,T,C) -> dict of pooled views. R: (n_roles, C) role membership on device."""
    _, vp = model.to_patches(x, v)                      # (B,C,P,patch)
    tok = vp.any(dim=-1).float()                        # (B,C,P)
    h = model.encode(x, v).float()                      # (B,C,P,d)
    B, C, P, d = h.shape
    n_all = tok.sum(dim=(1, 2)).clamp(min=1.0)
    emb_mean = (h * tok[..., None]).sum(dim=(1, 2)) / n_all[:, None]
    tl = tok.clone()
    tl[:, :, :-last_k] = 0.0
    n_last = tl.sum(dim=(1, 2)).clamp(min=1.0)
    emb_last = (h * tl[..., None]).sum(dim=(1, 2)) / n_last[:, None]

    # per-channel means: over the whole window and over the last k patches
    ch_n = tok.sum(2).clamp(min=1.0)                    # (B,C)
    ch_mean = (h * tok[..., None]).sum(2) / ch_n[..., None]          # (B,C,d)
    chl_n = tl.sum(2)                                   # (B,C) 0 if channel absent in tail
    ch_last = (h * tl[..., None]).sum(2) / chl_n.clamp(min=1.0)[..., None]
    present = (chl_n > 0).float()                       # (B,C)

    # role means over the last k patches (channels absent in the tail contribute 0)
    Rn = (R[None] * present[:, None, :]).sum(-1).clamp(min=1.0)     # (B,n_roles)
    emb_role = torch.einsum("rc,bcd->brd", R, ch_last * present[..., None]) / Rn[..., None]
    emb_role = emb_role.reshape(B, -1)

    # element-wise max over present channels of the last-k per-channel mean
    neg = torch.finfo(h.dtype).min
    emb_chmax = torch.where(present[..., None] > 0, ch_last, torch.full_like(ch_last, neg))
    emb_chmax = emb_chmax.max(1).values
    emb_chmax = torch.where(torch.isfinite(emb_chmax) & (emb_chmax > neg / 2), emb_chmax,
                            torch.zeros_like(emb_chmax))

    # most-diverging channel: largest ||last-k mean - window mean|| among present channels
    dev = ((ch_last - ch_mean) ** 2).sum(-1).sqrt() * present               # (B,C)
    top = dev.argmax(1)                                                    # (B,)
    emb_dev = ch_last[torch.arange(B, device=h.device), top]               # (B,d)
    return {"emb_mean": emb_mean, "emb_last": emb_last, "emb_role": emb_role,
            "emb_chmax": emb_chmax, "emb_dev": emb_dev, "dev_norm": dev}


def write_part(path, log_ids, t_bins, cols):
    fields = {"log_id": pa.array(log_ids, type=pa.string()),
              "t_bin": pa.array(np.concatenate(t_bins), type=pa.int32())}
    for k, chunks in cols.items():
        arr = np.concatenate(chunks)
        fields[k] = pa.FixedSizeListArray.from_arrays(
            pa.array(arr.reshape(-1), type=pa.float16()), arr.shape[1])
    tmp = path + ".tmp"
    pq.write_table(pa.table(fields), tmp, compression="zstd")
    os.replace(tmp, path)


def check_against_cache(out, tag):
    """Reproduction gate: emb_mean/emb_last must match the cached eval_emb part."""
    ref = os.path.join(META, "eval_emb", f"emb-{tag}.parquet")
    if not os.path.exists(ref):
        # a pilot subset has no cached twin; the full part covers its windows, so
        # compare on the shared (log_id, t_bin) keys of the full part instead
        full = os.path.join(META, "eval_emb", f"emb-{tag.split('.pilot')[0]}.parquet")
        if os.path.exists(full):
            ref = full
        else:
            print(f"  check: no cached {os.path.basename(ref)} -- skipped")
            return None
    a, b = pq.read_table(out), pq.read_table(ref)
    ka = list(zip(a["log_id"].to_pylist(), a["t_bin"].to_pylist()))
    kb = list(zip(b["log_id"].to_pylist(), b["t_bin"].to_pylist()))
    common = set(ka) & set(kb)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    idx = [(ia[k], ib[k]) for k in sorted(common)]
    worst = {}
    for col in ("emb_mean", "emb_last"):
        A = np.stack(a[col].to_numpy(zero_copy_only=False)).astype(np.float32)
        Bm = np.stack(b[col].to_numpy(zero_copy_only=False)).astype(np.float32)
        d = np.abs(A[[i for i, _ in idx]] - Bm[[j for _, j in idx]])
        worst[col] = float(d.max())
    ok = all(w < 2e-2 for w in worst.values())     # fp16 storage + bf16 autocast jitter
    print(f"  check: {'PASS' if ok else 'FAIL'} on {len(idx):,} shared windows; "
          f"max|d| {worst}")
    return ok


def run(args):
    global OUT
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})
    model = model.to(dev).eval()
    R = torch.from_numpy(role_matrix()).to(dev)
    print(f"encoder: {args.ckpt} step {ck['step']} d={cfg['d']}; roles "
          f"{[(r, int(n)) for r, n in zip(ROLES, role_matrix().sum(1))]}", flush=True)
    mean, std = load_channel_stats()
    exclude = load_exclude()
    t_start, n_done = time.time(), 0
    for shard in curated_shards():
        base = os.path.basename(shard).replace(".parquet", "")
        tag = f"{base}.p{args.part}" + (f".pilot{args.pilot}" if args.pilot else "")
        out = os.path.join(OUT, f"emb-{tag}.parquet")
        if os.path.exists(out):
            print(f"  {tag}: cached", flush=True)
            if args.check:
                check_against_cache(out, tag)
            if args.pilot:
                break
            continue
        ds = ShardEvalWindows(shard, mean, std, exclude, cfg["window"], args.batch,
                              part=args.part, nparts=args.of, limit_flights=args.pilot)
        dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=args.workers,
                                         pin_memory=True,
                                         prefetch_factor=4 if args.workers else None)
        log_ids, t_bins, cols = [], [], {}
        t0, n = time.time(), 0
        for lid, tb, xb, vb in dl:
            xb = xb.to(dev, non_blocking=True)
            vb = vb.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vw = views(model, xb, vb, R)
            log_ids.extend([lid] * len(tb))
            t_bins.append(tb.numpy())
            for k, val in vw.items():
                cols.setdefault(k, []).append(val.half().cpu().numpy())
            n += len(tb)
            if n % 20000 < args.batch:
                print(f"  {tag}: {n:,} windows  {n/max(time.time()-t0,1e-9):.0f}/s",
                      flush=True)
        if not t_bins:
            print(f"  {tag}: no windows", flush=True)
            continue
        write_part(out, log_ids, t_bins, cols)
        n_done += n
        print(f"  {tag}: DONE {n:,} windows in {(time.time()-t0)/60:.1f} min", flush=True)
        if args.check:
            check_against_cache(out, tag)
        if args.pilot:
            break
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "views": ["emb_mean", "emb_last", "emb_role",
                                                "emb_chmax", "emb_dev", "dev_norm"],
                   "roles": list(ROLES), "last_k": LAST_K}, f, indent=1)
    print(f"\npart {args.part}: {n_done:,} windows in {(time.time()-t_start)/3600:.2f} h",
          flush=True)


def status():
    tot, cnt = 0, 0
    for p in sorted(glob.glob(os.path.join(OUT, "emb-*.parquet"))):
        n = pq.ParquetFile(p).metadata.num_rows
        pil = "pilot" in p
        print(f"  {os.path.basename(p):48s} {n:>10,}{'  (pilot)' if pil else ''}")
        if not pil:
            tot += n
            cnt += 1
    print(f"{cnt} parts, {tot:,} windows ({100*tot/6_087_288:.1f}% of the grid)")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--pilot", type=int, default=0, help="flights per shard, smoke only")
    ap.add_argument("--check", action="store_true", help="reproduction gate vs eval_emb")
    ap.add_argument("--out", default=OUT, help="output dir (one per encoder checkpoint)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        OUT = args.out
        status()
        return
    run(args)


if __name__ == "__main__":
    main()
