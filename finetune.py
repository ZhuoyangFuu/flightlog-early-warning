#!/usr/bin/env python
"""Tier-2, stage 3: end-to-end fine-tuning of the pretrained encoder.

PROJECT_PLAN §7 ("Proposed": ... then linear probe and fine-tune"), §7.4.

Why this is a separate stage from `probe.py`, and why the probe cannot stand in
--------------------------------------------------------------------------------
The probe asks "is the frozen representation linearly separable?". Only this
stage asks the question the paper is actually about: **does the proposed model
clear the Tier-1 bar** of GBT 0.708 (split A) / 0.364 (split B_fixed)? A probe
number cannot answer that, because the encoder never saw a label. Fine-tuning
also invalidates the cached embeddings -- the weights move -- so this stage does
its own forward passes for both training and scoring.

The ablation that carries the claim
-----------------------------------
`--init pretrained` against `--init scratch` **is the experiment**. Identical
architecture, identical data, identical schedule; the only difference is whether
the encoder starts from `checkpoints/pretrain_last.pt` or from random init. If
scratch matches pretrained, the 6.25 h of pretraining bought nothing and the
paper must say so. Run both, always, before quoting a fine-tuned number.
`--freeze-encoder` gives the third arm (head-only, i.e. the probe trained
end-to-end) and is the control that separates "the representation is good" from
"the optimiser had 50M parameters to play with".

Training windows are NOT the evaluation grid, deliberately
----------------------------------------------------------
Scoring must happen on Tier-1's stride-20 grid or the FAR is not comparable
(§7.4). Training must not: the pre-event band is only 180 bins wide, so the grid
yields ~9 positive windows per positive flight and roughly 14k in a split-A train
arm -- far too few for a 50.5M-parameter model. Training therefore samples window
offsets **continuously** inside the band, which is a legitimate ~20x
augmentation: the windows are real data at real offsets, and it touches nothing
the evaluation sees. Negative windows are sampled continuously too.

Class balance comes from *how many windows each flight contributes*, not from
reweighting a stream that is 99% negative: a positive flight yields
`--pos-per-flight` windows from its band, a negative flight yields
`--neg-per-flight`. At the defaults (8 and 1) and the corpus's 14% positive-flight
rate that lands near 55% positive windows, which is why the loss needs no
`pos_weight`.

A **reservoir shuffle buffer** sits between the sampler and the batcher. The
shard stream hands out one flight's windows consecutively, so without it a batch
would be a handful of flights seen from 8 angles each -- the batch-diversity
weakness flagged in §7.2. Here it would be actively harmful, because those
correlated windows share a label.

Held-out validation is carved from the TRAIN arm by vehicle
-----------------------------------------------------------
`--val-frac` of training *vehicles* (never flights, never the test arm) are held
out to pick the best epoch. Selecting on test would be the exact leak this
project's split design exists to prevent.

Usage
-----
    python finetune.py --split A --seed 0 --init pretrained     # train
    python finetune.py --split A --seed 0 --init pretrained --score
    python finetune.py --smoke                                  # CPU wiring test
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_io import decode
from pretrain_model import build_model
from pretrain_data import load_channel_stats
from embed_eval import curated_shards, load_exclude, LAST_K
from eval_common import window_labels, score_report, roster_for
from tier1_baseline import (POS_LO, POS_HI, NEG_MIN_BINS, STRIDE, WARMUP,
                            MIN_BINS)

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
PRETRAINED = os.path.join(CKPT_DIR, "pretrain_last.pt")
SCORES = os.path.join(META, "ft_scores")

SPLIT_COL = {"A": "split_A_s{seed}", "B": "split_B", "B_fixed": "split_B_fixed",
             "C": "split_C_cond"}


# ------------------------------------------------------------------ model ----
class WindowClassifier(nn.Module):
    """Pretrained encoder + the same two-view pooling the probe uses, + a head.

    Pooling is deliberately identical to `embed_eval.pool` (mean over all valid
    tokens, concatenated with mean over the last LAST_K patches) so that
    "probe on frozen features" and "fine-tune" differ only in which weights move.
    """

    def __init__(self, enc, d, last_k=LAST_K, dropout=0.1):
        super().__init__()
        self.enc = enc
        self.last_k = last_k
        self.norm = nn.LayerNorm(2 * d)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * d, 1)
        nn.init.zeros_(self.fc.bias)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)

    def forward(self, x, v):
        _, vp = self.enc.to_patches(x, v)
        tok = vp.any(dim=-1).unsqueeze(-1).to(torch.float32)   # (B,C,P,1)
        h = self.enc.encode(x, v).to(torch.float32)            # (B,C,P,d)
        n = tok.sum(dim=(1, 2)).clamp(min=1.0)
        mean = (h * tok).sum(dim=(1, 2)) / n
        tl = tok.clone()
        tl[:, :, :-self.last_k] = 0.0
        last = (h * tl).sum(dim=(1, 2)) / tl.sum(dim=(1, 2)).clamp(min=1.0)
        z = self.drop(self.norm(torch.cat([mean, last], dim=-1)))
        return self.fc(z).squeeze(-1)                          # logits (B,)


def build_classifier(init, device, dropout=0.1, freeze=False):
    if init == "pretrained":
        ck = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        enc = build_model(cfg)
        enc.load_state_dict({k.replace("module.", ""): v
                             for k, v in ck["model"].items()})
        print(f"  encoder <- {PRETRAINED} (step {ck['step']})", flush=True)
    else:
        # Same architecture, random init. This is the ablation, so the config
        # MUST come from the pretrained checkpoint rather than from defaults.
        cfg = torch.load(PRETRAINED, map_location="cpu",
                         weights_only=False)["cfg"]
        enc = build_model(cfg)
        print("  encoder <- random init (scratch arm)", flush=True)
    m = WindowClassifier(enc, cfg["d"], dropout=dropout)
    if freeze:
        for p in m.enc.parameters():
            p.requires_grad = False
        print("  encoder FROZEN (head-only arm)", flush=True)
    return m.to(device), cfg


# ------------------------------------------------------------------- data ----
class LabeledWindows(IterableDataset):
    """Streams labelled windows from the curated shards for one split arm.

    Yields (x, valid, y). Window offsets are continuous, not on the stride-20
    grid -- see the module docstring.
    """

    def __init__(self, shards, keep_ids, mean, std, exclude, window,
                 pos_per_flight=8, neg_per_flight=1, seed=0, epoch=0,
                 buffer=512, limit_flights=None):
        self.shards, self.keep = list(shards), set(keep_ids)
        self.mean, self.std, self.exclude = mean, std, exclude
        self.window = window
        self.ppf, self.npf = pos_per_flight, neg_per_flight
        self.seed, self.epoch = seed, epoch
        self.buffer = buffer
        self.limit_flights = limit_flights

    def set_epoch(self, e):
        self.epoch = e

    def _windows(self, row, rng):
        t_e, nb = int(row["t_e_bin"]), int(row["n_bins"])
        W = self.window
        if t_e >= 0:
            # Positive band, in window-END coordinates: the label is defined on
            # t (the last bin of the window), so sample t in the band and take
            # (t-W, t]. Windows that would start before bin 0 are skipped rather
            # than padded -- padding is an evaluation-grid concession (§7.4), and
            # training on mostly-empty windows would just teach the head that
            # "empty" means "positive".
            lo_t, hi_t = t_e - POS_LO, min(t_e - POS_HI, nb - 1)
            if hi_t < lo_t or hi_t < W:
                return
            k, label = self.ppf, 1.0
        else:
            if nb < NEG_MIN_BINS:
                return
            lo_t, hi_t = WARMUP, nb - 1
            if hi_t < max(lo_t, W):
                return
            k, label = self.npf, 0.0
        lo_t = max(lo_t, W - 1)
        if hi_t < lo_t:
            return
        try:
            x, v = decode(row)
        except Exception:
            return
        xn = np.where(v, (x - self.mean) / self.std, 0.0).astype(np.float32)
        for t in rng.integers(lo_t, hi_t + 1, size=k):
            s = int(t) + 1 - W
            yield (torch.from_numpy(xn[s:s + W].copy()),
                   torch.from_numpy(v[s:s + W].copy()),
                   torch.tensor(label))

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng((self.seed, self.epoch, wid))
        order = rng.permutation(len(self.shards))
        buf, seen = [], 0
        for si in order:
            try:
                pf = pq.ParquetFile(self.shards[si])
            except Exception:
                continue
            for rb in pf.iter_batches(batch_size=16):
                for row in rb.to_pylist():
                    if row["log_id"] not in self.keep or row["log_id"] in self.exclude:
                        continue
                    if row["n_bins"] is None or row["n_bins"] < MIN_BINS:
                        continue
                    seen += 1
                    if seen % nw != wid:
                        continue
                    if self.limit_flights and seen > self.limit_flights * nw:
                        # `return`, not `break`: breaking would only leave the
                        # row-batch loop and then decompress every remaining
                        # ~6.4 GiB row group to no purpose. Debug path only.
                        rng.shuffle(buf)
                        yield from buf
                        return
                    for item in self._windows(row, rng):
                        buf.append(item)
                        # Reservoir shuffle: emit a RANDOM element once full, so
                        # a batch mixes flights instead of replaying one flight.
                        if len(buf) >= self.buffer:
                            j = int(rng.integers(len(buf)))
                            buf[j], buf[-1] = buf[-1], buf[j]
                            yield buf.pop()
        rng.shuffle(buf)
        yield from buf


def make_loader(ids, mean, std, exclude, window, args, epoch, seed,
                limit_flights=None):
    ds = LabeledWindows(curated_shards(), ids, mean, std, exclude, window,
                        pos_per_flight=args.pos_per_flight,
                        neg_per_flight=args.neg_per_flight,
                        seed=seed, epoch=epoch, buffer=args.shuffle_buffer,
                        limit_flights=limit_flights)
    return torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers, drop_last=True,
        pin_memory=True, prefetch_factor=4 if args.workers else None,
        persistent_workers=False), ds


# -------------------------------------------------------------- train/eval ----
def windows_per_pass(splits, ids, args, window):
    """How many windows one pass over `ids` yields — counted, not guessed.

    This matters because the arms differ by an order of magnitude. The B_fixed
    train arm has only ~313 usable positive flights, so at the defaults one pass
    is ~5.9k windows = ~246 batches, far short of `--steps-per-epoch` 1500. Left
    unaccounted, the loader would simply run dry each epoch, the cosine schedule
    (built over `epochs * steps_per_epoch`) would never complete, and the run
    would spend its whole life at a high LR. So the effective epoch length is
    `min(steps_per_epoch, this)` and the schedule is built from that.
    """
    d = splits[splits.log_id.isin(ids) & (splits.n_bins >= MIN_BINS)]
    pos = d[d.t_e_bin >= 0]
    hi = np.minimum(pos.t_e_bin - POS_HI, pos.n_bins - 1)
    lo = np.maximum(pos.t_e_bin - POS_LO, window - 1)
    n_pos = int((hi >= lo).sum())
    neg = d[(d.t_e_bin < 0) & (d.n_bins >= NEG_MIN_BINS)]
    n_neg = int((neg.n_bins - 1 >= max(WARMUP, window - 1)).sum())
    n_win = n_pos * args.pos_per_flight + n_neg * args.neg_per_flight
    frac = n_pos * args.pos_per_flight / max(n_win, 1)
    return n_win, n_pos, n_neg, frac


def split_ids(splits, split, seed, val_frac, rng_seed=12345):
    col = SPLIT_COL[split].format(seed=seed)
    tr = splits[splits[col] == "train"]
    te = splits[splits[col] == "test"]
    # Validation is carved by VEHICLE, never by flight: two flights of the same
    # airframe are not independent, and a flight-level carve would leak.
    veh = tr.vehicle_key.fillna(tr.log_id).unique()
    rng = np.random.default_rng(rng_seed + seed)
    val_veh = set(rng.choice(veh, max(1, int(len(veh) * val_frac)), replace=False))
    key = tr.vehicle_key.fillna(tr.log_id)
    return (set(tr.log_id[~key.isin(val_veh)]), set(tr.log_id[key.isin(val_veh)]),
            set(te.log_id))


@torch.no_grad()
def val_auprc(model, loader, device, max_batches=60):
    from sklearn.metrics import average_precision_score
    model.eval()
    ys, ps = [], []
    for i, (x, v, y) in enumerate(loader):
        if i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            lo = model(x.to(device), v.to(device))
        ps.append(torch.sigmoid(lo.float()).cpu().numpy())
        ys.append(y.numpy())
    model.train()
    if not ys:
        return float("nan")
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float(average_precision_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")


def tag_of(args):
    t = f"{args.split}_{args.init}"
    if args.split == "A":
        t += f"_seed{args.seed}"
    if args.freeze_encoder:
        t += "_frozen"
    return t


def train(args, device):
    # Seed torch explicitly: the head is randomly initialised and dropout is
    # active, so without this two runs of the "same" configuration differ, and
    # the pretrained-vs-scratch ablation would be reading init noise.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    splits = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))
    mean, std = load_channel_stats()
    exclude = load_exclude()
    tr_ids, va_ids, _ = split_ids(splits, args.split, args.seed, args.val_frac)
    print(f"  train {len(tr_ids)} flights / val {len(va_ids)} flights", flush=True)

    model, cfg = build_classifier(args.init, device, args.dropout,
                                  args.freeze_encoder)
    W = cfg["window"]

    n_win, n_pos, n_neg, frac = windows_per_pass(splits, tr_ids, args, W)
    steps_per_pass = max(1, n_win // args.batch)
    eff_spe = min(args.steps_per_epoch, steps_per_pass)
    print(f"  one pass: {n_win:,} windows ({n_pos} pos / {n_neg} neg flights, "
          f"{frac:.3f} positive) = {steps_per_pass} steps at batch {args.batch}",
          flush=True)
    if steps_per_pass < args.steps_per_epoch:
        print(f"  NOTE: --steps-per-epoch {args.steps_per_epoch} exceeds one pass; "
              f"using {eff_spe} so the LR schedule completes", flush=True)

    # Two parameter groups: the encoder is already good and must not be wrecked
    # in the first hundred steps, the head is random and needs to move fast.
    enc_p = [p for p in model.enc.parameters() if p.requires_grad]
    head_p = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith("enc.")]
    groups = ([{"params": enc_p, "lr": args.lr_encoder}] if enc_p else []) + \
             [{"params": head_p, "lr": args.lr_head}]
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss()

    ck_path = os.path.join(CKPT_DIR, f"ft_{tag_of(args)}.pt")
    best_path = os.path.join(CKPT_DIR, f"ft_{tag_of(args)}_best.pt")
    step, ep0, best = 0, 0, -1.0
    if args.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, ep0, best = ck["step"], ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"  resumed at epoch {ep0}, step {step}, best val AP {best:.4f}",
              flush=True)

    total = args.epochs * eff_spe
    for ep in range(ep0, args.epochs):
        dl, _ = make_loader(tr_ids, mean, std, exclude, W, args, ep, args.seed,
                            args.limit_flights)
        t0, run, n = time.time(), 0.0, 0
        for x, v, y in dl:
            # Count steps WITHIN the epoch: `step` can fall out of sync with
            # ep * steps_per_epoch whenever a pass runs dry (see
            # windows_per_pass), and a global-step bound would then be wrong.
            if n >= eff_spe:
                break
            lr_scale = min(1.0, (step + 1) / max(args.warmup, 1)) * \
                0.5 * (1 + math.cos(math.pi * min(step / total, 1.0)))
            for g, base in zip(opt.param_groups,
                               ([args.lr_encoder] if enc_p else []) + [args.lr_head]):
                g["lr"] = base * lr_scale
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logit = model(x.to(device, non_blocking=True),
                              v.to(device, non_blocking=True))
            loss = lossf(logit.float(), y.to(device).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += float(loss.detach()); n += 1; step += 1
            if n % 50 == 0:
                print(f"  e{ep:02d} step {step:6d} loss {run/n:.4f} "
                      f"{n/(time.time()-t0):.2f} st/s", flush=True)
        vdl, _ = make_loader(va_ids, mean, std, exclude, W, args, ep, args.seed + 777)
        ap = val_auprc(model, vdl, device)
        print(f"  e{ep:02d} DONE  train_loss {run/max(n,1):.4f}  val_AP {ap:.4f}"
              f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "step": step, "epoch": ep, "best": best, "cfg": cfg,
                 "args": vars(args)}
        torch.save(state, ck_path + ".tmp"); os.replace(ck_path + ".tmp", ck_path)
        if np.isfinite(ap) and ap > best:
            best = ap
            state["best"] = best
            torch.save(state, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
            print(f"  e{ep:02d} new best val AP {best:.4f} -> {best_path}", flush=True)
    print(f"finetune {tag_of(args)} done, best val AP {best:.4f}", flush=True)


# ------------------------------------------------------------------ score ----
@torch.no_grad()
def score(args, device):
    """Score the TEST arm on Tier-1's grid and report the same metrics Tier-1 does."""
    from embed_eval import ShardEvalWindows

    splits = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))
    mean, std = load_channel_stats()
    exclude = load_exclude()
    _, _, te_ids = split_ids(splits, args.split, args.seed, args.val_frac)

    best_path = os.path.join(CKPT_DIR, f"ft_{tag_of(args)}_best.pt")
    if not os.path.exists(best_path):
        raise SystemExit(f"no checkpoint {best_path} -- train first")
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = WindowClassifier(build_model(cfg), cfg["d"], dropout=0.0)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    print(f"  scoring {len(te_ids)} test flights with {best_path} "
          f"(val AP {ck.get('best', float('nan')):.4f})", flush=True)

    os.makedirs(SCORES, exist_ok=True)
    lids, tbs, scs = [], [], []
    t0 = time.time()
    for shard in curated_shards():
        base = os.path.basename(shard).replace(".parquet", "")
        part = os.path.join(SCORES, f"score-{tag_of(args)}-{base}.parquet")
        if os.path.exists(part):
            t = pq.read_table(part)
            lids.append(np.asarray(t["log_id"].to_pylist(), dtype=object))
            tbs.append(t["t_bin"].to_numpy()); scs.append(t["score"].to_numpy())
            print(f"  {base}: cached", flush=True)
            continue
        # Reuse the embedding pass's sampler so the grid is identical by
        # construction; keep_ids restricts it to the test arm.
        ds = ShardEvalWindows(shard, mean, std, exclude, cfg["window"],
                              args.score_batch, part=0, nparts=1,
                              keep_ids=te_ids)
        dl = torch.utils.data.DataLoader(ds, batch_size=None,
                                         num_workers=args.workers,
                                         pin_memory=True,
                                         prefetch_factor=4 if args.workers else None)
        L, T, S = [], [], []
        for lid, tb, xb, vb in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                lo = model(xb.to(device, non_blocking=True),
                           vb.to(device, non_blocking=True))
            L.extend([lid] * len(tb)); T.append(tb.numpy())
            S.append(torch.sigmoid(lo.float()).cpu().numpy())
        if not T:
            print(f"  {base}: no test windows", flush=True)
            continue
        tbl = pa.table({"log_id": pa.array(L, type=pa.string()),
                        "t_bin": pa.array(np.concatenate(T), type=pa.int32()),
                        "score": pa.array(np.concatenate(S), type=pa.float32())})
        pq.write_table(tbl, part + ".tmp", compression="zstd")
        os.replace(part + ".tmp", part)
        lids.append(np.asarray(L, dtype=object)); tbs.append(np.concatenate(T))
        scs.append(np.concatenate(S))
        print(f"  {base}: {len(L):,} windows ({(time.time()-t0)/60:.1f} min)",
              flush=True)

    log_id = np.concatenate(lids); t_bin = np.concatenate(tbs)
    sc = np.concatenate(scs)
    ros = splits.set_index("log_id")[["t_e_bin", "n_bins"]]
    y = window_labels(log_id, t_bin, ros)
    rep = score_report(log_id, t_bin, sc, y, roster_for(splits, te_ids),
                       curve_path=os.path.join(META, f"ft_curve_{tag_of(args)}.csv"))
    rep.update(split=args.split, seed=args.seed, init=args.init,
               frozen=bool(args.freeze_encoder), val_ap=ck.get("best"))
    with open(os.path.join(META, f"ft_pt_{tag_of(args)}.json"), "w") as f:
        json.dump(rep, f, default=float, indent=1)
    print(f"\n=== finetune {tag_of(args)} ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    return rep


def smoke(args):
    """CPU wiring test: a handful of real windows through the real classes."""
    device = "cpu"
    args.workers, args.batch, args.epochs = 0, 2, 1
    args.steps_per_epoch, args.limit_flights, args.warmup = 2, 4, 1
    print("smoke: building classifier (pretrained)...", flush=True)
    model, cfg = build_classifier("pretrained", device, 0.1, False)
    n = sum(p.numel() for p in model.parameters())
    print(f"  {n/1e6:.2f}M params (encoder {cfg['d']}d/{cfg['depth']}L)", flush=True)
    splits = pd.read_parquet(os.path.join(META, "splits_eval.parquet"))
    mean, std = load_channel_stats()
    tr_ids, va_ids, te_ids = split_ids(splits, "A", 0, args.val_frac)
    print(f"  split A seed0: train {len(tr_ids)} val {len(va_ids)} test {len(te_ids)}",
          flush=True)
    assert not (tr_ids & va_ids) and not (tr_ids & te_ids) and not (va_ids & te_ids), \
        "split arms overlap"
    dl, _ = make_loader(tr_ids, mean, std, load_exclude(), cfg["window"], args,
                        0, 0, limit_flights=3)
    x, v, y = next(iter(dl))
    print(f"  batch x{tuple(x.shape)} v{tuple(v.shape)} y={y.tolist()} "
          f"valid_frac {float(v.float().mean()):.3f} "
          f"present_ch {float(v.any(1).sum(1).float().mean()):.1f}", flush=True)
    assert torch.isfinite(x).all(), "non-finite input"
    logit = model(x, v)
    loss = nn.BCEWithLogitsLoss()(logit.float(), y.float())
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in model.parameters()
            if p.grad is not None)
    lv = float(loss.detach())
    print(f"  logits {logit.detach().tolist()}  loss {lv:.4f}  grad_sum {g:.3e}",
          flush=True)
    assert math.isfinite(lv) and g > 0, "no gradient flow"

    # The class-balance claim in the module docstring is checked, not assumed:
    # balance comes from pos/neg windows-per-flight against the 14% positive
    # flight rate, which is why the loss carries no pos_weight.
    dl2, _ = make_loader(tr_ids, mean, std, load_exclude(), cfg["window"], args,
                         1, 1, limit_flights=60)
    ys = []
    for i, (_, _, yy) in enumerate(torch.utils.data.DataLoader(
            dl2.dataset, batch_size=32, num_workers=0)):
        ys.append(yy)
        if i >= 20:
            break
    ys = torch.cat(ys)
    print(f"  label balance over {len(ys)} sampled windows: "
          f"{float(ys.mean()):.3f} positive", flush=True)
    assert 0.25 < float(ys.mean()) < 0.85, "class balance far off the documented range"
    print("SMOKE OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A", choices=list(SPLIT_COL))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "scratch"])
    ap.add_argument("--freeze-encoder", action="store_true")
    # One pass is short (see windows_per_pass): at the defaults it is 688 steps
    # for split A, 490 for C and only 217 for B_fixed. Because window offsets
    # are re-sampled continuously each epoch, an extra epoch is genuinely fresh
    # data rather than a repeat, so epochs are the right knob -- 8 gives ~5.5k
    # steps on A, ~3.9k on C, ~1.7k on B_fixed. Raise --epochs for B_fixed if it
    # underfits; steps-per-epoch only ever acts as a cap.
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--score-batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pos-per-flight", type=int, default=8)
    ap.add_argument("--neg-per-flight", type=int, default=1)
    ap.add_argument("--shuffle-buffer", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit-flights", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.limit_flights = args.limit_flights or None

    if args.smoke:
        return smoke(args)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"=== finetune {tag_of(args)} on {device} ===", flush=True)
    if args.score:
        return score(args, device)
    train(args, device)


if __name__ == "__main__":
    main()
