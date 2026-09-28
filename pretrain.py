#!/usr/bin/env python
"""Masked-reconstruction pretraining over the bulk corpus (PROJECT_PLAN §7).

Run it through `run_pretrain.sh`, never directly -- the launcher is what makes
it detached, restartable and inspectable.

    bash run_pretrain.sh start          # setsid-detached, resumes automatically
    bash run_pretrain.sh status
    bash run_pretrain.sh tail
    bash run_pretrain.sh stop

Estimate before committing GPU time:

    python pretrain.py --scaling-test           # steps/s, epoch ETA, peak VRAM

Resume safety: optimizer, scaler, step, epoch and RNG all live in the
checkpoint, written to a .tmp and renamed into place, so a kill at any instant
leaves a loadable file. `start` doubles as resume.
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import N_CHANNELS
from pretrain_data import WINDOW, PATCH, make_loader
from pretrain_model import build_model, count_params

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
META = os.path.join(ROOT, "meta")
STATUS = os.path.join(META, "pretrain_status.json")

_STOP = False


def _sigterm(_s, _f):
    global _STOP
    _STOP = True
    print("\n[signal] finishing the current step, then checkpointing", flush=True)


def default_cfg():
    return {"n_channels": N_CHANNELS, "window": WINDOW, "patch": PATCH,
            "d": 512, "depth": 12, "heads": 8, "mlp_ratio": 4.0,
            "dropout": 0.0, "mask_ratio": 0.5}


def save_ckpt(path, model, opt, scaler, step, epoch, cfg, best=None):
    obj = {"model": (model.module if hasattr(model, "module") else model).state_dict(),
           "opt": opt.state_dict(),
           "scaler": scaler.state_dict() if scaler is not None else None,
           "step": step, "epoch": epoch, "cfg": cfg, "best": best,
           "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()}
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)               # atomic: a kill mid-write cannot corrupt


def load_ckpt(path, model, opt, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    (model.module if hasattr(model, "module") else model).load_state_dict(ck["model"])
    if opt is not None and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck.get("step", 0), ck.get("epoch", 0), ck.get("best")


def write_status(**kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f, indent=1, default=float)
    os.replace(tmp, STATUS)


def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))


# ------------------------------------------------------------------ train ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16, help="per GPU")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=6000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--mask-mode", default="random",
                    choices=["random", "channel", "span", "mixed"],
                    help="pretext (2026-09-11): whole-channel / span / mixed masking")
    ap.add_argument("--init-from", default=None,
                    help="continued pretraining: initialise WEIGHTS from this checkpoint "
                         "(fresh optimiser/schedule) when --out has no pretrain_last.pt")
    ap.add_argument("--no-exclude-eval-vehicles", action="store_true")
    ap.add_argument("--scaling-test", action="store_true",
                    help="measure throughput and exit without training")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model + few steps, verifies the whole path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=CKPT_DIR)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        if os.path.exists(STATUS):
            with open(STATUS) as f:
                for k, v in json.load(f).items():
                    print(f"  {k}: {v}")
        else:
            print("  pretrain: no status file yet (never started)")
        return

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    # ---- distributed (torchrun) or single process
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world > 1
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(args.device or
                          (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    is_main = rank == 0

    cfg = default_cfg()
    cfg["mask_ratio"] = args.mask_ratio
    cfg["mask_mode"] = args.mask_mode
    if args.smoke:
        cfg.update(d=128, depth=2, heads=4)

    torch.manual_seed(1234 + rank)
    model = build_model(cfg).to(device)
    nparam = count_params(model)
    if is_main:
        print(f"model: d={cfg['d']} depth={cfg['depth']} heads={cfg['heads']} "
              f"-> {nparam/1e6:.1f}M params, {N_CHANNELS}x{WINDOW//PATCH} tokens",
              flush=True)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out, exist_ok=True)
    last = os.path.join(args.out, "pretrain_last.pt")
    step, start_epoch, best = 0, 0, None
    if os.path.exists(last):
        step, start_epoch, best = load_ckpt(last, model, opt, scaler, device)
        if is_main:
            print(f"--resume: from step {step}, epoch {start_epoch}", flush=True)
    elif args.init_from:
        ck0 = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd0 = {k.replace("module.", ""): v for k, v in ck0["model"].items()}
        (model.module if ddp else model).load_state_dict(sd0)
        if is_main:
            print(f"--init-from: weights from {args.init_from} (step {ck0.get('step')}); "
                  f"fresh optimiser, mask_mode={args.mask_mode}", flush=True)
    elif is_main:
        print("--resume: no checkpoint yet, starting from scratch", flush=True)

    total_steps = args.epochs * args.steps_per_epoch
    loader, ds = make_loader(batch_size=args.batch_size, workers=args.workers,
                             epoch=start_epoch,
                             exclude_eval_vehicles=not args.no_exclude_eval_vehicles)

    # ---- scaling test: measure, report, exit. No checkpoint, no state change.
    if args.scaling_test or args.smoke:
        n_probe = 12 if args.smoke else 40
        model.train()
        t_data = t_step = 0.0
        t0 = time.time()
        it = iter(loader)
        done = 0
        for i in range(n_probe):
            td = time.time()
            try:
                x, v = next(it)
            except StopIteration:
                break
            t_data += time.time() - td
            x, v = x.to(device, non_blocking=True), v.to(device, non_blocking=True)
            ts = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, st = model(x, v)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_step += time.time() - ts
            done = i + 1
        el = time.time() - t0
        if done == 0:
            print("scaling test: the loader yielded nothing -- check shards/stats")
            return
        sps = done / el
        win_s = args.batch_size * sps
        print(f"\nscaling test ({done} steps, batch {args.batch_size}/GPU):")
        print(f"  {sps:.2f} steps/s   {win_s:.0f} windows/s   "
              f"data {100*t_data/el:.0f}% / compute {100*t_step/el:.0f}%")
        if device.type == "cuda":
            print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        eff = sps * world
        print(f"  epoch of {args.steps_per_epoch} steps: "
              f"{args.steps_per_epoch/eff/60:.1f} min on {world} GPU(s)")
        print(f"  {args.epochs} epochs: {args.epochs*args.steps_per_epoch/eff/3600:.1f} h")
        if t_data > 0.5 * el:
            print("  NOTE: data-bound. Raise --workers before adding GPUs.")
        return

    # ---- real training loop
    model.train()
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    for epoch in range(start_epoch, args.epochs):
        ds.set_epoch(epoch)
        # `steps_per_epoch` has to bound the inner loop explicitly. The dataset
        # yields ~1M windows against a budget of `total_steps`, so the loader is
        # never exhausted and without this break the epoch loop would iterate
        # exactly once -- which is what happened on the 14 Aug run: 48 dataloader
        # workers spawned once, were never recycled, grew for ~2 h, and the OOM
        # killer took one at step 9,500. Bounding the epoch bounds worker memory.
        epoch_start = step
        it = iter(loader)                 # fresh worker pool for this epoch
        for x, v in it:
            if _STOP:
                break
            lr = lr_at(step, args.lr, args.warmup, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, st = model(x, v)
                loss = loss / args.accum
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run_loss += float(loss) * args.accum
            run_n += 1
            step += 1

            if is_main and step % args.log_every == 0:
                m = run_loss / max(run_n, 1)
                el = time.time() - t0
                print(f"  e{epoch:02d} step {step:>7} loss {m:.4f} lr {lr:.2e} "
                      f"{step/max(el,1e-9):.2f} st/s", flush=True)
                write_status(step=step, epoch=epoch, loss=m, lr=lr,
                             steps_per_s=step / max(el, 1e-9),
                             total_steps=total_steps, params_m=nparam / 1e6,
                             state="running")
                run_loss, run_n = 0.0, 0
            if is_main and step % args.ckpt_every == 0:
                save_ckpt(last, model, opt, scaler, step, epoch, cfg, best)
            if step >= total_steps or step - epoch_start >= args.steps_per_epoch:
                break
        # Drop the iterator now rather than waiting for GC: __del__ shuts the
        # worker processes down, which is the whole point of rolling the epoch.
        del it
        if is_main:
            save_ckpt(os.path.join(args.out, f"pretrain_e{epoch:02d}.pt"),
                      model, opt, scaler, step, epoch, cfg, best)
            save_ckpt(last, model, opt, scaler, step, epoch + 1, cfg, best)
        if _STOP or step >= total_steps:
            break

    if is_main:
        save_ckpt(last, model, opt, scaler, step, epoch, cfg, best)
        write_status(step=step, epoch=epoch, total_steps=total_steps,
                     state="stopped" if _STOP else "done")
        print(f"\n{'stopped' if _STOP else 'done'} at step {step} "
              f"({(time.time()-t0)/3600:.2f} h)", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
