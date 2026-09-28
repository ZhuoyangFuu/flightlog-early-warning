#!/usr/bin/env python
"""Extract the per-log schema-presence mask and fold it into the split tables.

WHY THIS EXISTS. `failsafe_flags` fields vary by firmware: of 54 observed fields only
~30 appear in every log. A flag that "did not fire" may simply be absent from that
firmware's schema, so trigger rate confounds with version -- and Split C holds out
firmware versions. Without conditioning on schema presence, Split C measures FIELD
AVAILABILITY rather than generalization. That is the single live data risk in the
benchmark (PROJECT_PLAN §8, data-gotcha 5).

The mask does not need a re-parse of the 6.2 TB of raw logs (which are deleted for the
bulk tier anyway). The tensorizer already wrote it, buried inside the `label` JSON blob
of the quality table:

    {"t_e": ..., "n_primary": 0, "primary_fields": [], "secondary_fields": [],
     "armed_s": 0.0, "n_ff_fields": 38, "n_fds_fields": 7, "unknown_ff": [], "notes": []}

`n_ff_fields` / `n_fds_fields` are the schema-presence signal and they really do vary
(observed 0, 37, 38, 39, 41, 42, 45...). **`n_ff_fields == 0` matters most: such a log
cannot produce a failsafe-based trigger at all**, so it is structurally incapable of
being positive and must not be scored as a negative that the model "missed".

Also unpacks `primary_fields`, which trigger set v2 needs (it is a join, not a re-parse)
and which lives nowhere else.

Outputs (all in meta/):
  schema_mask.parquet        one row per log: the unpacked label fields
  schema_mask_report.json    Split C confound quantified, per arm
  splits_eval.parquet        rewritten with the mask columns joined on
  splits_pretrain.parquet    same

CPU only, no GPU. Reads the quality table plus TWO STRING COLUMNS of the curated shards
-- parquet is columnar, so the 14.7 GB of tensors on those files is never touched.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "meta")
PROC = os.path.join(HERE, "proc")
MASK = os.path.join(META, "schema_mask.parquet")
REPORT = os.path.join(META, "schema_mask_report.json")

FIELDS = ["t_e", "n_primary", "n_secondary", "armed_s", "n_ff_fields", "n_fds_fields"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unpack(labels):
    """Parse the label JSON column into a frame. Rows that are null or unparseable get
    NaN rather than being dropped -- a log with no label is a real state (parse failure)
    and silently losing it would understate the corpus."""
    out = {f: [] for f in FIELDS}
    out["primary_fields"] = []
    out["unknown_ff"] = []
    bad = 0
    for s in labels:
        try:
            d = json.loads(s) if isinstance(s, str) else {}
        except (json.JSONDecodeError, TypeError):
            d = {}
            bad += 1
        for f in FIELDS:
            v = d.get(f)
            out[f].append(np.nan if v is None else v)
        out["primary_fields"].append(",".join(d.get("primary_fields") or []))
        out["unknown_ff"].append(",".join(d.get("unknown_ff") or []))
    return pd.DataFrame(out), bad


def load_bulk():
    log("reading bulk quality table (label column)...")
    frames = []
    for f in sorted(glob.glob(os.path.join(META, "qparts_bulk", "*.parquet"))):
        df = pd.read_parquet(f)
        if "label" not in df.columns:
            continue                      # the two 5-column parts predate the label field
        frames.append(df[["log_id", "status", "label"]])
    q = pd.concat(frames, ignore_index=True)
    q = q[q.status == "ok"].drop_duplicates("log_id")
    log(f"  {len(q)} bulk ok logs")
    return q


def load_curated():
    """The curated tier's quality table is a single file, meta/quality_curated.parquet --
    NOT a qparts_* directory like the bulk tier, which is why it is easy to miss. The
    curated shards themselves carry no `label` column (their schema is log_id/n_bins/
    n_ch/t0_unix/x/mask/t_e_bin), so the shards are not a fallback."""
    p = os.path.join(META, "quality_curated.parquet")
    if not os.path.exists(p):
        log(f"  WARN {p} missing -- curated tier will have no mask")
        return pd.DataFrame(columns=["log_id", "label"])
    c = pd.read_parquet(p, columns=["log_id", "status", "label"])
    c = c[c.status == "ok"].drop_duplicates("log_id")
    log(f"  {len(c)} curated ok logs (of {len(pd.read_parquet(p, columns=['log_id']))})")
    return c[["log_id", "label"]]


def arm_stats(df, col, arm):
    sub = df[df[col] == arm]
    if not len(sub):
        return {}
    ff = sub.n_ff_fields
    return {
        "n": int(len(sub)),
        "n_ff_fields_mean": round(float(ff.mean()), 2) if ff.notna().any() else None,
        "n_ff_fields_median": float(ff.median()) if ff.notna().any() else None,
        "pct_no_failsafe_schema": round(100 * float((ff == 0).mean()), 2),
        "pct_positive": round(100 * float(sub.is_positive.mean()), 2),
    }


def main():
    t0 = time.time()
    bulk = load_bulk()
    cur = load_curated()

    parts = []
    for name, frame in (("bulk", bulk), ("curated", cur)):
        if not len(frame):
            continue
        log(f"unpacking {name} label JSON ({len(frame)} rows)...")
        un, bad = unpack(frame.label.tolist())
        un.insert(0, "log_id", frame.log_id.values)
        un["tier"] = name
        if bad:
            log(f"  {bad} unparseable label blobs in {name}")
        parts.append(un)

    mask = pd.concat(parts, ignore_index=True).drop_duplicates("log_id")
    mask["has_failsafe_schema"] = mask.n_ff_fields.fillna(0) > 0
    mask.to_parquet(MASK, index=False)
    log(f"wrote {MASK}  ({len(mask)} logs)")

    report = {
        "n_logs": int(len(mask)),
        "n_ff_fields_distribution": {str(k): int(v) for k, v in
                                     mask.n_ff_fields.value_counts()
                                     .head(15).sort_index().items()},
        "pct_with_no_failsafe_schema": round(
            100 * float((mask.n_ff_fields.fillna(0) == 0).mean()), 2),
    }

    # Join onto both split tables and quantify the Split C confound on each.
    join_cols = ["log_id", "n_ff_fields", "n_fds_fields", "has_failsafe_schema",
                 "primary_fields", "unknown_ff", "armed_s"]
    for tag, path in (("eval", os.path.join(META, "splits_eval.parquet")),
                      ("pretrain", os.path.join(META, "splits_pretrain.parquet"))):
        if not os.path.exists(path):
            log(f"  SKIP {tag}: {path} missing -- run build_splits.py first")
            continue
        sp = pd.read_parquet(path)
        sp = sp.drop(columns=[c for c in join_cols if c != "log_id" and c in sp.columns])
        sp = sp.merge(mask[join_cols], on="log_id", how="left")
        # Split C conditioned on schema presence. A log with no failsafe_flags fields
        # cannot produce a failsafe-based trigger, so scoring it as a negative the model
        # "missed" measures firmware field availability, not generalization. Kept as a
        # SEPARATE column: split_C stays as built so nothing already computed moves.
        if "split_C" in sp.columns:
            sp["split_C_cond"] = np.where(~sp.has_failsafe_schema.fillna(False),
                                          "excluded", sp.split_C)
        sp.to_parquet(path, index=False)
        cov = round(100 * float(sp.n_ff_fields.notna().mean()), 2)
        log(f"joined mask onto {tag} ({len(sp)} rows, {cov}% matched)")
        report[f"split_C_confound_{tag}"] = {
            "mask_coverage_pct": cov,
            "as_built": {"train": arm_stats(sp, "split_C", "train"),
                         "test": arm_stats(sp, "split_C", "test")},
            "conditioned": {"train": arm_stats(sp, "split_C_cond", "train"),
                            "test": arm_stats(sp, "split_C_cond", "test")},
        }

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {REPORT}")
    print(json.dumps(report, indent=2), flush=True)

    for tag in ("eval", "pretrain"):
        c = report.get(f"split_C_confound_{tag}")
        if not c:
            continue
        for kind in ("as_built", "conditioned"):
            a = c.get(kind, {})
            if not a.get("train") or not a.get("test"):
                continue
            gap = abs(a["train"]["pct_no_failsafe_schema"]
                      - a["test"]["pct_no_failsafe_schema"])
            ratio = (a["test"]["pct_positive"] / a["train"]["pct_positive"]
                     if a["train"]["pct_positive"] else float("nan"))
            verdict = "CONFOUNDED" if gap > 5 else "arms comparable"
            log(f"{tag}/{kind}: no-schema gap {gap:.2f} pts, "
                f"test/train positive ratio {ratio:.2f}x -> {verdict}")

    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
