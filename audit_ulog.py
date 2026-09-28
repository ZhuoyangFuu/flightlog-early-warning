#!/usr/bin/env python
"""Audit parsed ulog files: topic availability, sample rates, and data-quality defects.

Run: python audit_ulog.py <dir_of_ulg> [--out report.json]
"""
import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")

# Channels the plan wants as model input (Section 6).
WANTED = [
    "vehicle_attitude", "vehicle_attitude_setpoint",
    "vehicle_angular_velocity", "vehicle_rates_setpoint",
    "vehicle_local_position", "vehicle_local_position_setpoint",
    "sensor_combined", "actuator_motors", "actuator_outputs",
    "estimator_status", "estimator_innovations",
    "battery_status", "vehicle_air_data", "wind", "cpuload",
]
# Autopilot-output topics used ONLY for labels; excluded from model input (Section 5).
LABEL_ONLY = [
    "failure_detector_status", "vehicle_status", "failsafe_flags",
    "estimator_status_flags", "vehicle_command",
]


def audit_one(path):
    from pyulog import ULog

    rec = {"file": os.path.basename(path), "bytes": os.path.getsize(path)}
    t0 = time.time()
    try:
        ulog = ULog(path)
    except Exception as e:
        rec["parse_error"] = f"{type(e).__name__}: {e}"
        rec["parse_s"] = time.time() - t0
        return rec
    rec["parse_s"] = time.time() - t0

    rec["dropouts"] = len(ulog.dropouts)
    rec["dropout_ms"] = sum(d.duration for d in ulog.dropouts)
    rec["msg_count"] = len(ulog.data_list)

    topics = {}
    for d in ulog.data_list:
        name = d.name
        ts = d.data.get("timestamp")
        if ts is None or len(ts) < 2:
            continue
        ts = np.asarray(ts, dtype=np.float64)
        dur = (ts[-1] - ts[0]) / 1e6
        dt = np.diff(ts) / 1e6
        entry = {
            "n": int(len(ts)),
            "dur_s": round(dur, 2),
            "hz": round(len(ts) / dur, 2) if dur > 0 else None,
            "n_fields": len(d.field_data),
            "nonmono": int((dt <= 0).sum()),
        }
        if len(dt):
            entry["dt_p50_ms"] = round(float(np.median(dt)) * 1e3, 3)
            entry["dt_p99_ms"] = round(float(np.percentile(dt, 99)) * 1e3, 3)
            entry["max_gap_s"] = round(float(dt.max()), 3)
            # fraction of time sitting in gaps > 5x the median cadence
            med = float(np.median(dt))
            if med > 0:
                big = dt[dt > 5 * med]
                entry["gap_frac"] = round(float(big.sum() / dur), 4) if dur > 0 else None
        # NaN / inf audit over numeric fields
        bad = tot = 0
        for f in d.field_data:
            if f.field_name == "timestamp":
                continue
            arr = np.asarray(d.data[f.field_name], dtype=np.float64)
            if arr.dtype.kind not in "fiu":
                continue
            tot += arr.size
            bad += int((~np.isfinite(arr)).sum())
        entry["nonfinite_frac"] = round(bad / tot, 6) if tot else 0.0
        # multi-instance topics (e.g. battery_status_0/_1) keep the worst case
        if name in topics:
            prev = topics[name]
            entry["instances"] = prev.get("instances", 1) + 1
            if prev["n"] > entry["n"]:
                entry.update({k: prev[k] for k in ("n", "dur_s", "hz") if k in prev})
        topics[name] = entry
    rec["topics"] = topics

    # armed duration from vehicle_status (label-only topic, used for filtering)
    rec["armed_s"] = None
    for d in ulog.data_list:
        if d.name == "vehicle_status" and "arming_state" in d.data:
            ts = np.asarray(d.data["timestamp"], dtype=np.float64) / 1e6
            arm = np.asarray(d.data["arming_state"])
            armed = arm == 2  # ARMING_STATE_ARMED
            if armed.any() and len(ts) > 1:
                dt = np.diff(ts, prepend=ts[0])
                rec["armed_s"] = round(float(dt[armed].sum()), 1)
            break

    # weak-label trigger presence (Section 5 triggers 1-3)
    trig = {}
    for d in ulog.data_list:
        if d.name == "failure_detector_status":
            for f in d.field_data:
                fn = f.field_name
                if fn.startswith("fd_") or fn in ("failure_detector_status",):
                    arr = np.asarray(d.data[fn])
                    if arr.dtype.kind in "biu" and arr.max() > 0:
                        trig[f"fds.{fn}"] = int((arr > 0).sum())
        if d.name == "vehicle_status" and "nav_state" in d.data:
            ns = np.asarray(d.data["nav_state"])
            # PX4 failsafe-ish nav states: RTL(5), LAND(18/19), DESCEND(20), TERMINATION(21)
            fs = np.isin(ns, [5, 18, 19, 20, 21])
            if fs.any():
                trig["nav_state_failsafe"] = int(fs.sum())
            trig["nav_states_seen"] = sorted(set(int(x) for x in ns))
        if d.name == "failsafe_flags":
            for f in d.field_data:
                arr = np.asarray(d.data[f.field_name])
                if f.field_name != "timestamp" and arr.dtype.kind in "biu" and arr.max() > 0:
                    trig[f"ff.{f.field_name}"] = int((arr > 0).sum())
    rec["triggers"] = trig
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(
        os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".ulg")
    )
    if args.limit:
        files = files[: args.limit]

    recs = []
    for i, p in enumerate(files, 1):
        r = audit_one(p)
        recs.append(r)
        print(f"[{i}/{len(files)}] {r['file'][:12]} "
              f"{r['bytes']/1e6:6.1f}MB {r.get('parse_s',0):5.2f}s "
              f"{'ERR:'+r['parse_error'][:40] if 'parse_error' in r else str(len(r.get('topics',{})))+' topics'}",
              file=sys.stderr)

    ok = [r for r in recs if "parse_error" not in r]
    print("\n" + "=" * 72)
    print(f"parsed OK: {len(ok)}/{len(recs)}   failures: {len(recs)-len(ok)}")
    for r in recs:
        if "parse_error" in r:
            print(f"   FAIL {r['file']}: {r['parse_error'][:90]}")

    if ok:
        ps = np.array([r["parse_s"] for r in ok])
        mb = np.array([r["bytes"] / 1e6 for r in ok])
        print(f"\nparse time: mean {ps.mean():.2f}s  p50 {np.median(ps):.2f}s  "
              f"p90 {np.percentile(ps,90):.2f}s  max {ps.max():.2f}s")
        print(f"throughput: {(mb/ps).mean():.1f} MB/s/worker")

        print("\n--- MODEL-INPUT TOPIC AVAILABILITY ---")
        for t in WANTED:
            have = [r for r in ok if t in r["topics"]]
            if not have:
                print(f"  {t:32s}   0.0%")
                continue
            hz = [r["topics"][t]["hz"] for r in have if r["topics"][t]["hz"]]
            nf = [r["topics"][t]["nonfinite_frac"] for r in have]
            gf = [r["topics"][t].get("gap_frac") or 0 for r in have]
            print(f"  {t:32s} {100*len(have)/len(ok):5.1f}%  "
                  f"hz p50={np.median(hz):7.1f} p10={np.percentile(hz,10):6.1f} p90={np.percentile(hz,90):7.1f}  "
                  f"nonfinite={max(nf):.5f}  gapfrac_p90={np.percentile(gf,90):.3f}")

        print("\n--- LABEL-ONLY TOPIC AVAILABILITY ---")
        for t in LABEL_ONLY:
            have = sum(1 for r in ok if t in r["topics"])
            print(f"  {t:32s} {100*have/len(ok):5.1f}%")

        # channel intersection (gate measurement 4)
        sets = [set(r["topics"]) for r in ok]
        inter = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        print(f"\n--- CHANNEL SET DIVERSITY (gate 4) ---")
        print(f"  topics present in ALL logs : {len(inter)}")
        print(f"  topics present in ANY log  : {len(union)}")
        print(f"  distinct topic-set signatures: {len(set(frozenset(s) for s in sets))}/{len(sets)}")
        wi = sorted(inter & set(WANTED))
        print(f"  wanted topics in intersection: {len(wi)}/{len(WANTED)} -> {wi}")

        print("\n--- QUALITY DEFECTS ---")
        nd = sum(1 for r in ok if r["dropouts"] > 0)
        print(f"  logs with dropouts: {nd}/{len(ok)}  "
              f"total dropout ms p90={np.percentile([r['dropout_ms'] for r in ok],90):.0f}")
        nm = sum(1 for r in ok if any(t.get("nonmono", 0) > 0 for t in r["topics"].values()))
        print(f"  logs with non-monotonic timestamps: {nm}/{len(ok)}")
        nn = sum(1 for r in ok if any(t.get("nonfinite_frac", 0) > 0 for t in r["topics"].values()))
        print(f"  logs with non-finite values: {nn}/{len(ok)}")
        arm = [r["armed_s"] for r in ok if r["armed_s"]]
        print(f"  armed_s recoverable in {len(arm)}/{len(ok)} logs; "
              f"p50={np.median(arm) if arm else 0:.0f}s")

        print("\n--- TRIGGER FIRE RATE (gate 2) ---")
        tc = Counter()
        for r in ok:
            for k in r["triggers"]:
                if k != "nav_states_seen":
                    tc[k] += 1
        for k, c in tc.most_common(20):
            print(f"  {k:44s} {c:4d}/{len(ok)}  {100*c/len(ok):5.1f}%")
        anyt = sum(1 for r in ok if any(k != "nav_states_seen" for k in r["triggers"]))
        print(f"  ANY trigger: {anyt}/{len(ok)} = {100*anyt/len(ok):.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(recs, f)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
