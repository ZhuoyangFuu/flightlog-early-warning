#!/usr/bin/env python
"""EWS Tier-0 sweep: freeze the Q1 curve (PROJECT_PLAN §1, RESUME §8).

One configuration is not a fair baseline, so this sweeps the two knobs the plan
names — window length and channel aggregation — and gives classical EWS its best
shot before Q1 is concluded:

    window_s in {2.5, 5, 10, 20}  x  topk in {1, 3, 5}   (12 configs)

Each config is scored by ews_baseline.py on the curated shards, then evaluated
twice by ews_eval.py: on all v1 positives, and on trigger-set-v2 positives only
(meta/trigger_v2_exclude.csv dropped — demoted-trigger logs, t_e unreliable).

RESUMABLE. A config whose traces parquet already exists is skipped, so the sweep
can be killed and relaunched at any time and loses at most one config (~20 min).
The (5, 3) config reuses the existing ews_traces_curated.parquet.

FREEZE RULE (decided before looking at results): among v2 curves across all
configs, take the operating point with the highest detection rate subject to
FAR <= 2.0/h; ties broken by longer median lead. That config's v2 curve is
copied to meta/ews_curve_FROZEN.csv and the choice recorded in
meta/ews_frozen_config.json. Detection at the plan's reference FAR ~1.6/h is
reported for continuity with the earlier single-config number.

    python ews_sweep.py                # run (use setsid nohup, see RESUME §2)
    python ews_sweep.py --status       # one-command progress, runs nothing
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
META = os.path.join(ROOT, "meta")
EXCLUDE = os.path.join(META, "trigger_v2_exclude.csv")

WINDOWS = [2.5, 5.0, 10.0, 20.0]
TOPKS = [1, 3, 5]
FAR_CAP = 2.0          # /h, freeze-rule constraint
REF_FAR = 1.6          # /h, continuity with the pre-sweep headline number


def tag_of(w, k):
    return f"w{w:g}k{k}"


def traces_path(w, k):
    if (w, k) == (5.0, 3):                      # the pre-sweep run, reused
        return os.path.join(META, "ews_traces_curated.parquet")
    return os.path.join(META, f"ews_traces_{tag_of(w, k)}.parquet")


def curve_path(w, k, v2):
    return os.path.join(META, f"ews_curve_{tag_of(w, k)}{'_v2' if v2 else ''}.csv")


def configs():
    return [(w, k) for w in WINDOWS for k in TOPKS]


def status():
    print(f"EWS sweep status  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    done = 0
    for w, k in configs():
        t = os.path.exists(traces_path(w, k))
        c = os.path.exists(curve_path(w, k, v2=True))
        done += t and c
        print(f"  w={w:>4g}s topk={k}   traces={'DONE' if t else '....'}   "
              f"curves={'DONE' if c else '....'}")
    print(f"  {done}/{len(configs())} configs complete")
    fro = os.path.join(META, "ews_frozen_config.json")
    if os.path.exists(fro):
        with open(fro) as f:
            j = json.load(f)
        print(f"  FROZEN: {j['tag']}  threshold {j['threshold']:g}  "
              f"FAR {j['far_per_hour']:.2f}/h  detection {j['detection_rate']:.3f}  "
              f"median lead {j['median_lead_s']:.1f}s")
    else:
        print("  not frozen yet")


def run(cmd, log):
    log.write("+ " + " ".join(cmd) + "\n")
    log.flush()
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.status:
        status()
        return

    if not os.path.exists(EXCLUDE):
        sys.exit(f"missing {EXCLUDE} — build it from quality_curated.parquet first")

    t0 = time.time()
    log = sys.stdout
    for i, (w, k) in enumerate(configs(), 1):
        tp = traces_path(w, k)
        print(f"\n===== [{i}/{len(configs())}] window {w:g}s topk {k}  "
              f"(elapsed {(time.time()-t0)/60:.0f} min) =====", flush=True)
        if os.path.exists(tp):
            print(f"traces exist, skipping score: {os.path.basename(tp)}", flush=True)
        else:
            run([PY, os.path.join(ROOT, "ews_baseline.py"),
                 "--shards", os.path.join(ROOT, "proc", "curated-*.parquet"),
                 "--tag", tag_of(w, k), "--workers", str(args.workers),
                 "--window-s", str(w), "--topk", str(k)], log)
        for v2 in (False, True):
            cmd = [PY, os.path.join(ROOT, "ews_eval.py"),
                   "--traces", tp, "--out", curve_path(w, k, v2)]
            if v2:
                cmd += ["--exclude-csv", EXCLUDE]
            run(cmd, log)

    # ---- summary over all configs (v2 = the official numbers) -------------
    rows = []
    for w, k in configs():
        for v2 in (False, True):
            c = pd.read_csv(curve_path(w, k, v2)).dropna(subset=["far_per_hour"])
            c = c.sort_values("far_per_hour")
            lo = c.iloc[0]
            capped = c[c.far_per_hour <= FAR_CAP]
            best = capped.loc[capped.detection_rate.idxmax()] if len(capped) else None
            ref = c.iloc[(c.far_per_hour - REF_FAR).abs().argmin()]
            rows.append({
                "window_s": w, "topk": k, "labelset": "v2" if v2 else "v1",
                "min_far": lo.far_per_hour, "det_at_min_far": lo.detection_rate,
                "det_at_far_cap": best.detection_rate if best is not None else np.nan,
                "far_at_cap_pt": best.far_per_hour if best is not None else np.nan,
                "lead_at_cap_pt": best.median_lead_s if best is not None else np.nan,
                "thresh_at_cap_pt": best.threshold if best is not None else np.nan,
                f"det_near_{REF_FAR}h": ref.detection_rate,
            })
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(META, "ews_sweep_summary.csv"), index=False)
    print("\n===== SWEEP SUMMARY (see meta/ews_sweep_summary.csv) =====")
    print(summ.to_string(index=False), flush=True)

    # ---- freeze ------------------------------------------------------------
    v2s = summ[(summ.labelset == "v2") & summ.det_at_far_cap.notna()]
    if v2s.empty:
        print("\nNO config reaches FAR <= 2/h — nothing frozen; Q1 statement "
              "becomes 'classical EWS cannot operate below 2 false alarms/hour'.")
        return
    best = v2s.sort_values(["det_at_far_cap", "lead_at_cap_pt"],
                           ascending=False).iloc[0]
    w, k = float(best.window_s), int(best.topk)
    shutil.copy(curve_path(w, k, v2=True), os.path.join(META, "ews_curve_FROZEN.csv"))
    frozen = {
        "tag": tag_of(w, k), "window_s": w, "topk": k, "labelset": "v2",
        "rule": f"max detection s.t. FAR<={FAR_CAP}/h over v2 curves, "
                f"tie-break longer median lead; decided pre-hoc in ews_sweep.py",
        "threshold": float(best.thresh_at_cap_pt),
        "far_per_hour": float(best.far_at_cap_pt),
        "detection_rate": float(best.det_at_far_cap),
        "median_lead_s": float(best.lead_at_cap_pt),
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_curve": os.path.basename(curve_path(w, k, v2=True)),
    }
    with open(os.path.join(META, "ews_frozen_config.json"), "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"\nFROZEN Q1 Tier-0 curve: {frozen['tag']}  "
          f"threshold {frozen['threshold']:g}  FAR {frozen['far_per_hour']:.2f}/h  "
          f"detection {frozen['detection_rate']:.3f}  "
          f"median lead {frozen['median_lead_s']:.1f}s")
    print("wrote meta/ews_curve_FROZEN.csv and meta/ews_frozen_config.json")
    print(f"total elapsed {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
