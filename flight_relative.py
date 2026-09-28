#!/usr/bin/env python
"""Item 5: flight-relative embeddings -- deviation from the flight's own nominal state.

PROJECT_PLAN §7.6 piece 4 and §11.0.1 item 5. The pretrained encoder is fed windows
standardised with ONE corpus-global mean/std per channel (`pretrain_data.load_channel_stats`),
so a fixed-wing in nominal cruise and a quadrotor in trouble can sit in the same region of
embedding space. That is the mechanism behind the cross-airframe collapse (GBT 0.708 -> 0.397).
The cheapest test of "detection is deviation-from-self" needs NO GPU: subtract, from every
cached window embedding, the mean embedding of the SAME flight's opening windows.

Causality. The reference for window i is the mean of the flight's windows whose end bin is
<= min(t_i, WARMUP + REF_S seconds). So a window never sees anything after itself, and after
the opening REF_S seconds the reference is frozen. Positive flights whose fault is already
present at take-off will have it in the reference and lose it -- that is a property of the
design, disclosed, not a bug; the eval label band (1-10 s before t_e) and the shifted training
bands (111-150 s) are far from the opening window for any flight long enough to be scored.

Output. One parquet per input part in `meta/eval_emb_rel/`, same `log_id`, `t_bin`, with
columns `emb_rel` (emb_mean - reference) and `emb_mean` (the original, so a concatenated
view is one `--views emb_mean,emb_rel` away). float16 like the source. Consumed with no code
change through the two env hooks §4.10 verified:

    FLIGHTLOG_EMB_DIR=meta/eval_emb_rel FLIGHTLOG_FEATURE_TAG=rel \
        python horizon.py --split B_fixed --mode shift --h 120 --views emb_rel --seeds 0,1,2

Usage:
    python flight_relative.py            # build all parts (resumable; skips existing)
    python flight_relative.py --status
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier1_baseline import HZ, WARMUP  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")
SRC = os.path.join(META, "eval_emb")
DST = os.path.join(META, "eval_emb_rel")
REF_S = 30.0                       # reference = opening 30 s after the 20 s warm-up
REF_END_BIN = int(WARMUP + REF_S * HZ)
MIN_REF = 3                        # windows; below this the reference is too noisy -> zeros


def build_part(src, dst):
    t = pq.read_table(src)
    log_id = np.asarray(t["log_id"].to_pylist(), dtype=object)
    t_bin = t["t_bin"].to_numpy()
    E = np.stack(t["emb_mean"].to_numpy(zero_copy_only=False)).astype(np.float32)
    n, d = E.shape
    rel = np.zeros_like(E)
    order = np.lexsort((t_bin, log_id))            # by flight, then time
    lid_s, tb_s = log_id[order], t_bin[order]
    bounds = np.flatnonzero(np.r_[True, lid_s[1:] != lid_s[:-1], True])
    n_short = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        tb = tb_s[a:b]
        e = E[idx]
        in_ref = tb <= REF_END_BIN
        if in_ref.sum() < MIN_REF:
            # fall back to the first MIN_REF windows so every flight gets a reference
            in_ref = np.zeros(len(tb), bool)
            in_ref[:MIN_REF] = True
            n_short += 1
        # causal running mean over reference windows: cumulative up to each window,
        # frozen once the reference period ends
        cum = np.cumsum(e * in_ref[:, None], axis=0)
        cnt = np.cumsum(in_ref).astype(np.float32)
        last = int(np.flatnonzero(in_ref)[-1])
        ref = cum / np.maximum(cnt, 1.0)[:, None]
        ref[last + 1:] = ref[last]
        rel[idx] = e - ref
    out = pa.table({"log_id": pa.array(log_id.tolist(), type=pa.string()),
                    "t_bin": pa.array(t_bin, type=pa.int32()),
                    "emb_mean": t["emb_mean"],
                    # FixedSizeListArray.from_arrays, as embed_eval.write_part does: a
                    # python list of 300k float16 rows is minutes; this is instant
                    "emb_rel": pa.FixedSizeListArray.from_arrays(
                        pa.array(rel.astype(np.float16).reshape(-1), type=pa.float16()), d)})
    tmp = dst + ".tmp"
    pq.write_table(out, tmp, compression="zstd")
    os.replace(tmp, dst)
    return n, len(bounds) - 1, n_short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    srcs = [p for p in sorted(glob.glob(os.path.join(SRC, "emb-*.parquet")))
            if "pilot" not in p]
    os.makedirs(DST, exist_ok=True)
    if args.status:
        done = sum(os.path.exists(os.path.join(DST, os.path.basename(p))) for p in srcs)
        print(f"{done}/{len(srcs)} parts built in {DST}")
        return
    t0 = time.time()
    for i, src in enumerate(srcs, 1):
        dst = os.path.join(DST, os.path.basename(src))
        if os.path.exists(dst):
            print(f"[{i}/{len(srcs)}] cached {os.path.basename(dst)}", flush=True)
            continue
        n, nf, ns = build_part(src, dst)
        print(f"[{i}/{len(srcs)}] {os.path.basename(dst)}  {n:,} windows  {nf:,} flights  "
              f"{ns} short-reference  ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(os.path.join(DST, "MANIFEST.json"), "w") as f:
        f.write('{"source": "meta/eval_emb", "ref_s": %g, "ref_end_bin": %d, "min_ref": %d, '
                '"columns": ["emb_mean", "emb_rel"], "causal": true}\n'
                % (REF_S, REF_END_BIN, MIN_REF))
    print("done")


if __name__ == "__main__":
    main()
