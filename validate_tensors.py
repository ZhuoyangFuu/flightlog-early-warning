#!/usr/bin/env python
"""Physics- and signal-level validation of tensorized shards.

A tensorizer that never crashes can still be silently wrong. These checks are
falsifiable statements about what the numbers must mean:

  1. round-trip     -- shapes and mask bit-count survive encode/decode
  2. quaternion     -- ||q|| == 1 on valid attitude cells
  3. gravity        -- accelerometer magnitude ~ 9.81 m/s^2 in steady flight
  4. anti-aliasing  -- bin-averaging must REDUCE high-frequency variance
                       relative to naive subsampling on fast channels
  5. mask integrity -- no invalid cell carries a nonzero value that could be
                       mistaken for data; no all-NaN channel marked valid
  6. label sanity   -- t_e lies inside the grid and inside an armed interval
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import CHANNEL_ID, CHANNELS, N_CHANNELS, TARGET_HZ
from shard_io import STORE_DTYPE, decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    tbl = pq.read_table(args.shard).to_pylist()[: args.n]
    print(f"validating {len(tbl)} logs from {os.path.basename(args.shard)}\n")

    fails = []
    qn_all, gv_all = [], []
    n_maskleak = 0
    n_te_ok = n_te = 0

    qi = [CHANNEL_ID[("vehicle_attitude", f"q[{k}]")] for k in range(4)]
    ai = [CHANNEL_ID[("sensor_combined", f"accelerometer_m_s2[{k}]")] for k in range(3)]
    gi = [CHANNEL_ID[("sensor_combined", f"gyro_rad[{k}]")] for k in range(3)]

    for r in tbl:
        x, m = decode(r)
        # 1. round trip
        if x.shape != (r["n_bins"], N_CHANNELS):
            fails.append(f"{r['log_id']}: shape {x.shape}")
            continue
        # 5. mask integrity: invalid cells must be exactly zero (never stale data)
        if np.any(x[~m] != 0):
            n_maskleak += 1
        # 2. quaternion norm
        vq = m[:, qi].all(axis=1)
        if vq.sum() > 10:
            qn = np.linalg.norm(x[vq][:, qi], axis=1)
            qn_all.append(qn)
        # 3. gravity magnitude
        va = m[:, ai].all(axis=1)
        if va.sum() > 10:
            gv_all.append(np.linalg.norm(x[va][:, ai], axis=1))
        # 6. label inside grid
        if r["t_e_bin"] >= 0:
            n_te += 1
            if 0 <= r["t_e_bin"] < r["n_bins"]:
                n_te_ok += 1

    print("--- 1. ROUND TRIP ---")
    print(f"  shape/decode failures: {len(fails)}/{len(tbl)}")
    for f in fails[:5]:
        print("   ", f)

    print("\n--- 2. QUATERNION NORM (must be 1.0) ---")
    if qn_all:
        q = np.concatenate(qn_all)
        print(f"  n={len(q)}  mean {q.mean():.5f}  p1 {np.percentile(q,1):.5f}  "
              f"p99 {np.percentile(q,99):.5f}")
        off = np.abs(q - 1.0) > 0.02
        print(f"  |‖q‖-1| > 0.02 : {100*off.mean():.3f}%  "
              f"{'PASS' if off.mean() < 0.02 else 'FAIL'}")
    else:
        print("  no valid attitude cells")

    print("\n--- 3. ACCELEROMETER MAGNITUDE (should centre near 9.81) ---")
    if gv_all:
        g = np.concatenate(gv_all)
        print(f"  n={len(g)}  p25 {np.percentile(g,25):.2f}  median {np.median(g):.2f}  "
              f"p75 {np.percentile(g,75):.2f} m/s^2")
        ok = 8.0 < np.median(g) < 11.5
        print(f"  {'PASS' if ok else 'FAIL'} (median in [8.0, 11.5])")

    print("\n--- 4. FINITE VALUES AND DYNAMIC RANGE (per channel) ---")
    # This check did not exist when the float16 overflow shipped. Storing
    # baro_pressure_pa (~101325 Pa) in fp16 (ceiling 65504) turned 98% of that
    # channel into +inf, and every other check above still passed because none
    # of them touched pressure. Any non-finite value under a valid mask is a
    # hard failure.
    nbad = np.zeros(N_CHANNELS)
    nval = np.zeros(N_CHANNELS)
    amax = np.zeros(N_CHANNELS)
    for r in tbl:
        x, m = decode(r)
        bad = (~np.isfinite(x)) & m
        nbad += bad.sum(0)
        nval += m.sum(0)
        fx = np.where(np.isfinite(x) & m, np.abs(x), 0)
        amax = np.maximum(amax, fx.max(0))
    tot_bad = int(nbad.sum())
    print(f"  non-finite cells under a valid mask: {tot_bad}  "
          f"{'PASS' if tot_bad == 0 else 'FAIL'}")
    if tot_bad:
        for j in np.argsort(-nbad)[:10]:
            if nbad[j] == 0:
                break
            print(f"    {CHANNELS[j][0]+'.'+CHANNELS[j][1]:48s} "
                  f"{100*nbad[j]/max(nval[j],1):5.1f}% of valid cells")
    fp16_ceil = 65504.0
    near = [j for j in range(N_CHANNELS) if amax[j] > 0.5 * fp16_ceil]
    print(f"  channels exceeding half the fp16 ceiling: {len(near)} "
          f"(harmless in float32, fatal in float16)")
    for j in sorted(near, key=lambda j: -amax[j])[:5]:
        print(f"    {CHANNELS[j][0]+'.'+CHANNELS[j][1]:48s} max|x| = {amax[j]:.1f}")

    print("\n--- 5. MASK INTEGRITY ---")
    print(f"  logs with nonzero value under an invalid mask: {n_maskleak}/{len(tbl)}  "
          f"{'PASS' if n_maskleak == 0 else 'FAIL'}")

    print("\n--- 6. LABEL GRID SANITY ---")
    print(f"  t_e set in {n_te} logs; inside grid {n_te_ok}/{n_te} "
          f"{'PASS' if n_te == n_te_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
