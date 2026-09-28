#!/usr/bin/env python
"""Item 1: the detection-lead frontier (PROJECT_PLAN §7.5, §11.0.1).

The claim the project is trying to establish is **dominance over Tier-0 on both
axes** -- higher detection *and* longer lead, at a FAR no worse than Tier-0's.
Tier-0 is one point; a learned model sweeps a *curve* of (detection, median lead)
as its threshold moves. Dominance means the curve passes above and to the right
of that point.

**The Tier-0 point is PER SPLIT** (`ews_split.py`, RESULTS.md §2.1) -- split A
det 0.141 / 217.3 s, `B_fixed` det 0.179 / 53.1 s at an INADMISSIBLE FAR 2.251/h.
The corpus-wide 0.186 / 66.0 s covers 1,777 positives against split A's 354 and
must never be compared with a split-restricted model.

Everything here is read off the cached `*_curve_*.csv` files -- the ones already
written by `tier1_baseline.py`, `probe.py` (via `eval_common.score_report`) and
`finetune.py`. **No model is refitted and no number is recomputed**, so this
cannot drift from `RESULTS.md`: it is a different *view* of the same curves.

Three quantities, defined once here so they are quotable:

  lead@det_ews   median lead when the model is thresholded down to Tier-0's own
                 detection rate. THE apples-to-apples speed comparison.
  dom_*          the best median lead available anywhere in the dominance region
                 (FAR <= Tier-0's FAR **and** detection >= Tier-0's), with the
                 detection and FAR at which it occurs. Absent => no dominance
                 point exists at all. `dominates` is the flag that matters:
                 that lead ALSO clears Tier-0's. Tier-0 trivially dominates
                 itself, which is the sanity check that the definition is right.
  det@REF_FAR    the existing headline, recomputed here only as a cross-check
                 against `RESULTS.md` -- if these disagree, something is wrong.
                 (Verified 2026-08-21: GBT B_fixed 0.364, probe 0.334 / C 0.484,
                 LR B_fixed 0.202 -- all reproduce §3 and §4.1 exactly.)

Caveat that must travel with every lead number (§7.5): median lead is
conditioned on *detected* events, so Tier-0's 66.0 s is measured on the 18.6% of
events it happens to catch. Comparing lead at different detection rates compares
different event subsets. `lead@det.186` controls for the rate but not for *which*
events; the matched-event version is item 4, not this file.

Usage
-----
    python frontier.py                 # table -> meta/frontier.csv, figure -> meta/fig/
    python frontier.py --no-fig        # table only
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier1_baseline import REF_FAR, FAR_CAP  # noqa: F401  (FAR_CAP documents the cap)

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "meta")
FIGDIR = os.path.join(META, "fig")

# Tier-0's frozen operating point (RESULTS.md §2, meta/ews_frozen_config.json).
# Read from the artifact rather than hard-coded so a re-freeze propagates.
EWS_CURVE = os.path.join(META, "ews_curve_FROZEN.csv")

# The frontier's FAR cap is TIER-0'S OWN FAR, not REF_FAR, and this is deliberate.
# Tier-0 was frozen under the rule "max detection subject to FAR <= FAR_CAP = 2.0"
# and landed at 1.617/h -- which is ABOVE REF_FAR = 1.6/h, so **the Tier-0
# operating point has zero admissible rows under the 1.6/h cap the learned tiers
# are scored at** (measured: 0 of its curve rows clear 1.6). Capping the frontier
# at 1.6 would therefore compare the learned tiers against a Tier-0 point that the
# same constraint forbids -- quietly generous to Tier-0. The claim being tested is
# "at a false-alarm rate no worse than Tier-0's", so Tier-0's FAR is the cap and
# every tier is held to the identical number. `det@REF_FAR` is still reported as
# the cross-check against RESULTS.md, which is unaffected by any of this.

# Median lead over a handful of events is noise, and the high-threshold tail of
# every curve is exactly that (n_detected falls to 6 on some arms). Lead-based
# quantities are computed only where at least this many events were detected.
# It removes only the degenerate tail: at Tier-0's detection rate the arms carry
# 66-253 detected events, far above the floor.
MIN_DETECTED = 20

# (curve file stem, label, split, tier) -- the tier drives colour, the split drives panel.
SERIES = [
    ("ews_curve_FROZEN",            "Tier-0 EWS",      "both",    "ews"),
    ("tier1_curve_lr_A_seed0",      "Tier-1 LR",       "A",       "lr"),
    ("tier1_curve_gbt_A_seed0",     "Tier-1 GBT",      "A",       "gbt"),
    ("tier2_curve_lr_A_seed0_it5000", "Tier-2 probe",  "A",       "probe"),
    ("tier1_curve_lr_B_fixed",      "Tier-1 LR",       "B_fixed", "lr"),
    ("tier1_curve_gbt_B_fixed",     "Tier-1 GBT",      "B_fixed", "gbt"),
    ("tier2_curve_lr_B_fixed_it5000", "Tier-2 probe",  "B_fixed", "probe"),
    ("ft_curve_B_fixed_pretrained", "Fine-tuned (pt)", "B_fixed", "ft"),
    ("ft_curve_B_fixed_scratch",    "Fine-tuned (scratch)", "B_fixed", "fts"),
    ("tier2_curve_lr_C_it5000",     "Tier-2 probe",    "C",       "probe"),
]

# Item-2/2c horizon arms (horizon.py). Same frozen encoder and frozen evaluation
# band; only the TRAINING label band differs, so they belong on the same frontier.
# DISCOVERED from meta/ rather than listed: the grid grew from 14 arms to 22 in one
# afternoon, and a hand-maintained list silently omits whatever was added last --
# which on 2026-08-21 would have hidden the split-A result.
def _discover_horizon():
    import re as _re
    pat = _re.compile(r"horizon_curve_(widen|shift)(\d+)(?:w(\d+))?_"
                      r"(A|B|B_fixed|C)(?:_seed(\d+))?\.csv$")
    found = []
    for f in sorted(glob.glob(os.path.join(META, "horizon_curve_*.csv"))):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        mode, h, w, sp, sd = m.groups()
        if mode == "widen" and h == "10":
            continue          # the reproduction control == the plain probe
        lbl = f"probe {mode} {h}s" + (f" w{w}" if w else "")
        if sd and sd != "0":
            lbl += f" s{sd}"
        found.append((os.path.basename(f)[:-4], lbl, sp,
                      "hwiden" if mode == "widen" else "hshift"))
    return found


SERIES.extend(_discover_horizon())


def ews_reference():
    """Tier-0's frozen operating point, taken from its own curve under its own rule.

    The freeze rule (tier1_baseline.op_point branch (a), mirroring Tier-0): max
    detection subject to FAR <= FAR_CAP. Reading it back rather than hard-coding
    0.186/66.0 means a re-freeze of Tier-0 propagates here automatically.

    THIS IS THE CORPUS-WIDE POINT (1,777 positives) and is NOT comparable to a
    split-restricted model. It is kept only as the frozen Q1 headline; every
    comparison uses `ews_reference_for(split)` below.
    """
    c = pd.read_csv(EWS_CURVE).dropna(subset=["far_per_hour"])
    capped = c[c.far_per_hour <= FAR_CAP]
    if not len(capped):
        raise SystemExit(f"{EWS_CURVE}: no row clears FAR <= {FAR_CAP}")
    b = capped.loc[capped.detection_rate.idxmax()]
    return {"det": float(b.detection_rate), "lead": float(b.median_lead_s),
            "far": float(b.far_per_hour), "scope": "corpus-wide"}


def ews_reference_for(split):
    """Tier-0 restricted to `split`'s TEST ARM -- the only like-for-like reference.

    Written by `ews_split.py` (see RESULTS.md §2.1). Falls back to the corpus-wide
    point with a loud warning rather than silently comparing populations, because
    that silent mismatch is exactly the bug this function exists to prevent.

    `admissible` is False when Tier-0 has no operating point clearing FAR_CAP on
    this split -- true of `B_fixed`, where its best is 2.251/h. There is then no
    Tier-0 point to dominate, and the honest claim becomes "the learned model
    operates where Tier-0 cannot"; the reference is still returned so the
    comparison can be reported, flagged.
    """
    # The EXTENDED-grid re-score supersedes the original wherever it exists: it is
    # the same indicator (determinism verified 2026-08-21, 211,200 crossing
    # comparisons, zero mismatches) on a 30-point grid reaching 200 instead of a
    # 16-point grid truncated at 80. The truncation is what put Tier-0's `B_fixed`
    # point at an artificially high FAR, so the original must never be preferred.
    p = os.path.join(META, "ews_bysplit_ext.json")
    if not os.path.exists(p):
        p = os.path.join(META, "ews_bysplit.json")
    if not os.path.exists(p):
        print(f"  WARNING: {p} missing -- falling back to the CORPUS-WIDE Tier-0 "
              f"point, which is NOT comparable to a split-restricted model.\n"
              f"  Run:  python ews_split.py", flush=True)
        return ews_reference()
    with open(p) as f:
        d = json.load(f)["by_split"]
    if split not in d:
        print(f"  WARNING: no per-split Tier-0 for {split}; using corpus-wide.",
              flush=True)
        return ews_reference()
    pt = d[split]
    far = pt.get("far_per_hour")
    admissible = far is not None and np.isfinite(far)
    if admissible:
        return {"det": float(pt["detection_rate"]), "lead": float(pt["median_lead_s"]),
                "far": float(far), "scope": f"split {split}", "admissible": True}
    # No admissible point: fall back to its best available row so the comparison
    # can still be made, and mark it.
    c = pd.read_csv(os.path.join(META, f"ews_curve_split_{split}.csv"))
    c = c.dropna(subset=["far_per_hour"]).sort_values("far_per_hour")
    b = c.iloc[0]
    return {"det": float(b.detection_rate), "lead": float(b.median_lead_s),
            "far": float(b.far_per_hour), "scope": f"split {split}",
            "admissible": False}


def load_curve(stem):
    p = os.path.join(META, f"{stem}.csv")
    if not os.path.exists(p):
        return None
    c = pd.read_csv(p).dropna(subset=["far_per_hour"])
    return c.sort_values("far_per_hour").reset_index(drop=True) if len(c) else None


def lead_reliable(c):
    """Rows whose median lead rests on enough detected events to mean anything."""
    return c[c.median_lead_s.notna() & (c.n_detected >= MIN_DETECTED)]


def frontier_row(c, ref):
    """The three quantities above, for one curve, against Tier-0 reference `ref`."""
    out = {}
    # The cap must satisfy BOTH constraints: no worse than Tier-0's FAR, and inside
    # the project's own admissible region. On `B_fixed` Tier-0's best point is
    # 2.251/h, so taking its FAR alone would license the learned models to operate
    # at a FAR the protocol forbids everywhere else -- a win bought by moving the
    # goalposts. min() keeps the comparison honest in both directions.
    cap = min(ref["far"], FAR_CAP)
    out["far_cap_used"] = float(cap)
    c_rel = lead_reliable(c)
    f = c.far_per_hour.to_numpy(float)
    d = c.detection_rate.to_numpy(float)
    l = c.median_lead_s.to_numpy(float)
    fin = np.isfinite(l)

    # --- det / lead at the reference FAR (cross-check against RESULTS.md)
    if f.min() <= REF_FAR <= f.max():
        out[f"det@{REF_FAR}h"] = float(np.interp(REF_FAR, f, d))
        if fin.sum() >= 2 and f[fin].min() <= REF_FAR <= f[fin].max():
            out[f"lead@{REF_FAR}h"] = float(np.interp(REF_FAR, f[fin], l[fin]))

    # --- lead at Tier-0's detection rate. Interpolate along the DETECTION axis,
    # which requires sorting by it; the curve is monotone in threshold, not in
    # detection, so re-sort rather than assuming.
    o = c_rel.sort_values("detection_rate")
    dd, ll, ff = (o.detection_rate.to_numpy(float), o.median_lead_s.to_numpy(float),
                  o.far_per_hour.to_numpy(float))
    if len(dd) >= 2 and dd.min() <= ref["det"] <= dd.max():
        out["lead@det_ews"] = float(np.interp(ref["det"], dd, ll))
        out["far@det_ews"] = float(np.interp(ref["det"], dd, ff))

    # --- dominance region: FAR <= Tier-0's FAR AND detection >= Tier-0's.
    # Report the longest lead available in it; absence of any row is the finding.
    dom = c_rel[(c_rel.far_per_hour <= cap) & (c_rel.detection_rate >= ref["det"])]
    out["n_dom_rows"] = int(len(dom))
    if len(dom):
        b = dom.loc[dom.median_lead_s.idxmax()]
        out.update(dom_lead=float(b.median_lead_s), dom_det=float(b.detection_rate),
                   dom_far=float(b.far_per_hour), dom_threshold=float(b.threshold))
        # Dominance is lead >= Tier-0's AND detection >= Tier-0's, inside the FAR cap.
        out["dominates"] = bool(b.median_lead_s >= ref["lead"])
    else:
        out["dominates"] = False
    return out


def build_table(ref):
    rows = []
    for stem, label, split, tier in SERIES:
        c = load_curve(stem)
        if c is None:
            print(f"  (missing) {stem}", flush=True)
            continue
        r = ref if split == "both" else ews_reference_for(split)
        rows.append({"model": label, "split": split, "tier": tier, "curve": stem,
                     "ews_det": r["det"], "ews_lead": r["lead"], "ews_far": r["far"],
                     "ews_admissible": r.get("admissible", True),
                     **frontier_row(c, r)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure ----
# Nature spec: 183 mm double column, <=170 mm tall, Liberation Sans (Arial
# metrics, TrueType so it embeds under pdf.fonttype 42), all text 5-7 pt,
# hairline 0.5 pt axes, no top/right spines, colourblind-safe with no red/green
# pair. Point every mathtext slot at the same face or subscripts silently fall
# back to DejaVu.
MM = 1 / 25.4
# Baselines are GREYS, learned arms carry hue -- otherwise Tier-1 GBT's blue
# collides with the shift ramp and the eye reads it as another horizon.
PALETTE = {"ews": "#000000", "lr": "#B0B0B0", "gbt": "#4D4D4D",
           "probe": "#E69F00", "ft": "#009E73", "fts": "#CC79A7",
           "hwiden": "#D55E00", "hshift": "#56B4E9"}

# Training horizon is ordered, so it is encoded as a sequential ramp: light =
# short band, dark = long. Colourblind-safe (single hue, luminance-ordered) and
# it survives greyscale printing, which a six-dash-pattern scheme does not.
SHIFT_RAMP = {"30": "#C6DBEF", "60": "#9ECAE1", "120": "#4292C6",
              "150": "#2171B5", "180": "#08519C", "240": "#08306B"}


def _style(base_pt=7.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam = "Liberation Sans"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam, "DejaVu Sans"],
        "font.size": base_pt, "axes.labelsize": base_pt, "axes.titlesize": base_pt,
        "xtick.labelsize": base_pt - 1, "ytick.labelsize": base_pt - 1,
        "legend.fontsize": base_pt - 1,
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": fam,
        "mathtext.bf": fam, "mathtext.sf": fam, "mathtext.tt": fam,
        "mathtext.cal": fam,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.0, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
    })
    return plt


def make_figure(ref, table, out_stem, base_pt=7.0, width_mm=183.0, height_mm=95.0,
                margins=(0.055, 0.985, 0.215, 0.86)):
    """Three panels: split A, split B_fixed, split C. Lead (y) vs detection (x).

    Only the FAR <= REF_FAR segment of each curve is drawn -- outside it the
    operating point is inadmissible, and plotting it invites reading a win off a
    region the protocol forbids.
    """
    plt = _style(base_pt)
    panels = [("A", "in-distribution"), ("B_fixed", "cross-airframe"),
              ("C", "schema-conditioned")]
    fig, axes = plt.subplots(1, 3, figsize=(width_mm * MM, height_mm * MM))
    l, r, b, t = margins
    fig.subplots_adjust(left=l, right=r, bottom=b, top=t, wspace=0.22)
    YMAX = 300.0

    for ax, (split, subtitle) in zip(axes, panels):
        # PER-SPLIT Tier-0 reference: the corpus-wide point is a different flight
        # population and comparing against it is the bug ews_split.py exists to fix.
        ref = ews_reference_for(split)
        cap = min(ref["far"], FAR_CAP)   # see frontier_row: both constraints
        # The dominance region is the upper-right QUADRANT -- both more accurate
        # and earlier than Tier-0. Shading the whole det >= 0.186 column would
        # imply detection alone is the target, which is the error §7.5 exists to
        # correct.
        ax.add_patch(plt.Rectangle((ref["det"], ref["lead"]), 1.0 - ref["det"],
                                   YMAX - ref["lead"], facecolor="#EDEDED",
                                   edgecolor="none", zorder=0))
        ax.axhline(ref["lead"], color="#BBBBBB", lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.axvline(ref["det"], color="#BBBBBB", lw=0.5, ls=(0, (3, 2)), zorder=1)

        for stem, label, sp, tier in SERIES:
            if sp not in (split, "both"):
                continue
            # The grid reached 22 arms; plotting all of them is illegible. The
            # figure carries the baselines plus the SHIFT family, which is where
            # every dominance result lives. Replicate seeds and the widen family
            # are in meta/frontier.csv and RESULTS.md, not here.
            if tier == "hwiden" or label.endswith(("s1", "s2")) or " w" in label:
                continue
            # Tier-0 has exactly ONE row inside its own FAR -- it is a point, not
            # a curve, and drawing it as a line would claim a legend entry for an
            # invisible mark. It appears as the star, and in the table.
            if tier == "ews":
                continue
            c = load_curve(stem)
            if c is None:
                continue
            seg = lead_reliable(c)
            seg = seg[seg.far_per_hour <= cap].sort_values("detection_rate")
            if not len(seg):
                continue
            # Within a horizon family the three arms share a colour, so the
            # horizon is carried by line style -- otherwise 30/60/120 s are
            # indistinguishable and the panel is unreadable in print.
            # Horizon is an ORDERED variable, so it gets a sequential colour ramp
            # (light = short, dark = long) rather than one flat colour plus six
            # dash patterns, which was unreadable at print size.
            ls, col = "-", PALETTE[tier]
            if tier == "fts":
                ls = "--"
            elif tier == "hshift":
                col = SHIFT_RAMP.get(label.split()[-1].rstrip("s"), PALETTE[tier])
            ax.plot(seg.detection_rate, seg.median_lead_s, color=col,
                    lw=1.0, zorder=3, label=label, ls=ls)

        # Tier-0's frozen operating point -- the thing to dominate
        adm = ref.get("admissible", True)
        ax.plot([ref["det"]], [ref["lead"]], marker="*" if adm else "X",
                ms=7 if adm else 5,
                color="#000000", mec="none", zorder=5,
                label=("Tier-0 (this split)" if adm else "Tier-0 (FAR inadmissible)"))
        ax.set_title(f"{split}  ({subtitle})\n"
                     f"Tier-0: det {ref['det']:.3f}, {ref['lead']:.0f} s, "
                     f"FAR {ref['far']:.2f}/h" + ("" if adm else "  \u2014 >cap"),
                     pad=3, fontsize=base_pt - 1)

        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, YMAX)
        ax.set_xlabel("detection rate")
        if split == "A":
            ax.set_ylabel("median lead time (s)")

    # panel letters, bold lower-case, alphabetical
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes, fontsize=base_pt + 1,
                fontweight="bold", va="top", ha="left")

    pairs = []
    for _ax in axes:
        pairs += list(zip(*_ax.get_legend_handles_labels()))
    seen, hh, ll_ = set(), [], []
    for a, bl in pairs:
        if bl not in seen:
            seen.add(bl); hh.append(a); ll_.append(bl)
    fig.legend(hh, ll_, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.005),
               handlelength=1.6, columnspacing=1.2, borderaxespad=0.0)

    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png", dpi=450)
    plt.close(fig)
    return out_stem


CAPTION = """\
Detection-lead frontier at a false-alarm rate no worse than Tier-0's. Median lead time
against detection rate as the alarm threshold sweeps, for (a) split A (in-distribution),
(b) split B_fixed (cross-airframe, the headline split) and (c) split C
(schema-conditioned). The star marks Tier-0 classical EWS at its frozen operating point
(detection {det:.3f}, median lead {lead:.1f} s, FAR {ews_far:.2f}/h). Only the segment of
each curve with FAR <= {ews_far:.2f}/h is drawn -- Tier-0's own rate, used as the shared
cap so that every tier is held to an identical constraint that Tier-0 itself satisfies.
Dashed guides and the shaded quadrant mark the dominance region, where a model is both
more accurate and earlier than Tier-0; no model enters it. Rows resting on fewer than
{min_det} detected events are omitted, since a median over a handful of events is noise.
Median lead is conditioned on detected events, so points at different detection rates
summarise different event subsets. Curves are read from the cached evaluation artifacts
and are the same fits reported in RESULTS.md; no model was refitted for this figure.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--out", default=os.path.join(FIGDIR, "frontier"))
    args = ap.parse_args()

    ref = ews_reference()
    print(f"Tier-0 reference: det {ref['det']:.3f}  lead {ref['lead']:.1f}s  "
          f"FAR {ref['far']:.3f}/h", flush=True)
    print(f"frontier FAR cap = Tier-0's own {ref['far']:.3f}/h "
          f"(REF_FAR={REF_FAR} would exclude Tier-0 itself); "
          f"lead needs n_detected >= {MIN_DETECTED}\n", flush=True)

    t = build_table(ref)
    cols = ["model", "split", f"det@{REF_FAR}h", f"lead@{REF_FAR}h",
            "lead@det_ews", "far@det_ews", "dom_lead", "dom_det", "dom_far",
            "n_dom_rows", "dominates"]
    cols = [c for c in cols if c in t.columns]
    pd.set_option("display.width", 200)
    print(t[cols].round(3).to_string(index=False))

    out_csv = os.path.join(META, "frontier.csv")
    t.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    dom = t[t.dominates] if "dominates" in t.columns else t.iloc[:0]
    dom = dom[dom.split != "both"]     # Tier-0 dominating itself is the sanity check
    print("\nDOMINANCE is judged PER SPLIT, against that split's own Tier-0 point")
    print("(RESULTS.md §2.1); the corpus-wide 0.186/66.0s is a different population.")
    for _sp in ("A", "B_fixed", "C"):
        r = ews_reference_for(_sp)
        adm = "" if r.get("admissible", True) else "  [FAR INADMISSIBLE - see §2.1]"
        d2 = dom[dom.split == _sp]
        print(f"  {_sp:8s} bar: det >= {r['det']:.3f} AND lead >= {r['lead']:.1f}s "
              f"at FAR <= {min(r['far'], FAR_CAP):.3f}/h{adm}")
        print(f"           -> {', '.join(d2.model) if len(d2) else 'NONE'}")

    if not args.no_fig:
        stem = make_figure(ref, t, args.out)
        cap = CAPTION.format(det=ref["det"], lead=ref["lead"],
                             ews_far=ref["far"], min_det=MIN_DETECTED)
        with open(stem + "_caption.txt", "w") as f:
            f.write(cap)
        print(f"wrote {stem}.pdf / .png / _caption.txt")


if __name__ == "__main__":
    main()
