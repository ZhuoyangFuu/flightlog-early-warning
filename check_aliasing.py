#!/usr/bin/env python
"""Quantify why bin-averaging (not subsampling) is required at 20 Hz.

Q1 tests whether variance and lag-1 autocorrelation RISE before loss of
control. If decimation aliases high-frequency gyro energy into the 0-10 Hz
band, that inflation is indistinguishable from the effect being measured.
This script measures the size of that contamination on real logs.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyulog import ULog

BIN = 0.05  # 20 Hz


def main():
    files = sorted(glob.glob("/mnt/wsssd/FlightLog/raw/*/*.ulg"))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    rows = []
    for p in files:
        if len(rows) >= 60:
            break
        if os.path.getsize(p) > 60e6:      # keep the scan fast
            continue
        try:
            u = ULog(p, message_name_filter_list=["sensor_combined"])
        except Exception:
            continue
        d = next((x for x in u.data_list if x.name == "sensor_combined"), None)
        if d is None:
            continue
        ts = np.asarray(d.data["timestamp"], float) / 1e6
        v = np.asarray(d.data["gyro_rad[0]"], float)
        o = np.argsort(ts)
        ts, v = ts[o], v[o]
        k = np.ones(len(ts), bool)
        k[1:] = np.diff(ts) > 0
        ts, v = ts[k], v[k]
        if len(ts) < 4000:
            continue
        hz = len(ts) / (ts[-1] - ts[0])
        if hz < 100:
            continue
        t0 = ts[0]
        n = int((ts[-1] - t0) / BIN)
        idx = np.floor((ts - t0) / BIN).astype(int)
        ok = (idx >= 0) & (idx < n) & np.isfinite(v)
        idx, vv = idx[ok], v[ok]
        s = np.bincount(idx, weights=vv, minlength=n)
        c = np.bincount(idx, minlength=n)
        good = c > 0
        binned = np.divide(s, c, out=np.zeros(n), where=good)
        # naive decimation == take the first sample landing in each bin
        uniq, first_pos = np.unique(idx, return_index=True)
        naive = np.zeros(n)
        naive[uniq] = vv[first_pos]
        m = good
        rows.append((hz, np.var(binned[m]), np.var(naive[m]),
                     np.var(np.diff(binned[m])), np.var(np.diff(naive[m]))))

    r = np.array(rows)
    print("ANTI-ALIASING CHECK  (sensor_combined gyro_rad[0] resampled to 20 Hz)")
    print(f"n logs = {len(r)}   median source rate = {np.median(r[:, 0]):.0f} Hz\n")
    print(f"  in-band variance    binned {np.median(r[:, 1]):.5f}   "
          f"naive {np.median(r[:, 2]):.5f}   ratio {np.median(r[:, 2]/r[:, 1]):.2f}x")
    print(f"  1st-diff variance   binned {np.median(r[:, 3]):.5f}   "
          f"naive {np.median(r[:, 4]):.5f}   ratio {np.median(r[:, 4]/r[:, 3]):.2f}x")
    infl = np.median(r[:, 4] / r[:, 3])
    print(f"\n  naive decimation inflates sample-to-sample variance {infl:.1f}x (median)")
    print("  -> that aliased energy would masquerade as the rising-variance EWS signal.")


if __name__ == "__main__":
    main()
