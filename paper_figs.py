#!/usr/bin/env python
"""The four paper figures, built from `meta/paper/*.csv` (see `paper_tables.py`).

Every number is read from the aggregated tables, so a figure cannot drift from the
result JSONs. Nothing is refitted here.

  fig1_frontier.pdf   Detection-lead plane, one panel per split, with each split's
                      Tier-0 best-over-grid point (per-axis maxima over the 176-config channel × rule cross, 2026-09-11) and the region that dominates it.
                      Answers: does anything beat classical EWS on both axes, where.
  fig2_band.pdf       The same arms as a function of the TRAINING LABEL BAND.
                      Answers: why. Lead is bought by the band, for every feature set.
  fig3_representation What pretraining buys: detection at matched head, the
                      detection/lead trade, and the seed stability that decides
                      which single-seed numbers are quotable.
  fig4_rules.pdf      Robustness to the sequential alarm rule: the fraction of the
                      20 (band, rule) configurations that dominate, per family.

Style: Nature artwork spec -- 183 mm double column, 5-7 pt sans, hairline rules,
TrueType embedded, no gridlines. The only straight reference lines in the figures
are the Tier-0 bars, which are results, not decoration.

Usage: python paper_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402

PAPER = Path("/mnt/wsssd/FlightLog/meta/paper")
FIG = Path("/mnt/wsssd/FlightLog/meta/fig")

MM = 1 / 25.4
INK, INK2 = "#000000", "#4d4d4d"

# One hue per FEATURE SET (the comparison the paper is about), one marker per HEAD.
EMB, STATS, T0 = "#0d3b70", "#e8710a", "#4d4d4d"
COL = {"emb": EMB, "stats": STATS}
MRK = {"lr": "o", "gbt": "s"}
FEAT_LABEL = {"emb": "pretrained embeddings", "stats": "hand-built statistics"}
HEAD_LABEL = {"lr": "linear head", "gbt": "boosted-tree head"}
SPLIT_LABEL = {"A": "A — in-distribution",
               "B_fixed": "B — cross-airframe",
               "C": "C — schema-conditioned"}
BAND_ORDER = ["1-10 s", "111-120 s", "141-150 s"]


def rc(scale: float = 1.0) -> dict:
    s = lambda p: p * scale                                        # noqa: E731
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
                            "DejaVu Sans"],
        "font.size": s(7), "axes.labelsize": s(7), "axes.titlesize": s(7),
        "xtick.labelsize": s(6), "ytick.labelsize": s(6), "legend.fontsize": s(6),
        "axes.linewidth": 0.5 * scale,
        "xtick.major.width": 0.5 * scale, "ytick.major.width": 0.5 * scale,
        "xtick.major.size": 2 * scale, "ytick.major.size": 2 * scale,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 0.9 * scale,
        "legend.frameon": False, "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }


def panel_label(ax, letter, scale=1.0, dx=-0.16, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8 * scale,
            fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None)
        print(f"  wrote {p}")
    plt.close(fig)


def load():
    s = pd.read_csv(PAPER / "summary.csv")
    t0 = pd.read_csv(PAPER / "tier0.csv")
    arms = pd.read_csv(PAPER / "arms.csv")
    rules = pd.read_csv(PAPER / "rules.csv")
    return s[s["n"] >= 3].copy(), t0, arms, rules


# --------------------------------------------------------------------------- fig 1
def fig1(summ, t0):
    """Detection-lead plane. The dominance region is shaded because it IS the
    claim: an arm counts only if it lands above and to the right of Tier-0's best."""
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM))
        for ax, sp in zip(axes, ["A", "B_fixed", "C"]):
            bar = t0[t0["split"] == sp].iloc[0]
            d = summ[summ["split"] == sp]
            xmax = max(0.62, (d["det"] + d["det_sd"]).max() * 1.10)
            # the Tier-0 bar must always be ON the axes -- on split A it sits at
            # 372.9 s, far above every learned arm, and that IS the split-A result
            ymax = max((d["lead"] + d["lead_sd"]).max(), bar["best_lead"]) * 1.16

            # the region that dominates Tier-0 on BOTH axes
            ax.add_patch(Rectangle((bar["best_det"], bar["best_lead"]),
                                   xmax - bar["best_det"], ymax - bar["best_lead"],
                                   facecolor="#eef3f8", edgecolor="none", zorder=0))
            ax.plot([bar["best_det"]], [bar["best_lead"]], marker="*", ms=7,
                    color=T0, mec="white", mew=0.4, zorder=5)
            ax.annotate("Tier-0 best", (bar["best_det"], bar["best_lead"]),
                        textcoords="offset points", xytext=(-4, 6),
                        fontsize=5.5, color=INK2, ha="right")

            for _, r in d.iterrows():
                ax.errorbar(r["det"], r["lead"], xerr=r["det_sd"], yerr=r["lead_sd"],
                            fmt=MRK[r["head"]], ms=3.4, mew=0, color=COL[r["features"]],
                            ecolor=COL[r["features"]], elinewidth=0.6, capsize=1.2,
                            alpha=0.95, zorder=4)
            ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)

            # Band labels, decluttered. The 120 s and 150 s embedding+gbt arms land
            # ~0.5 s apart on B_fixed, so a naive label writes one on top of the
            # other; near-coincident points get ONE merged label instead.
            pts = [(r["det"] / xmax, r["lead"] / ymax, r) for _, r in d.iterrows()
                   if r["band_label"] != "1-10 s"]
            used: list[tuple[float, float]] = []
            merged: dict[int, list] = {}
            for i, (nx, ny, r) in enumerate(pts):
                hit = next((j for j, (mx, my, _) in enumerate(pts[:i])
                            if abs(mx - nx) < 0.02 and abs(my - ny) < 0.02
                            and pts[j][2]["features"] == r["features"]), None)
                merged.setdefault(hit if hit is not None else i, []).append(r)
            for i, group in merged.items():
                nx, ny, r0 = pts[i]
                txt = ", ".join(sorted(g["band_label"].split("-")[1].replace(" s", "")
                                       for g in group))
                dy = 5 if r0["head"] == "gbt" else -8
                while any(abs(ux - nx) < 0.05 and abs(uy - (ny + dy / 300)) < 0.045
                          for ux, uy in used):
                    dy += 9 if dy > 0 else -9
                used.append((nx, ny + dy / 300))
                ax.annotate(txt, (r0["det"], r0["lead"]), textcoords="offset points",
                            xytext=(5, dy), fontsize=5, color=COL[r0["features"]])
            ax.set_xlabel("Detection rate at FAR 1.6 h$^{-1}$")
            ax.set_title(SPLIT_LABEL[sp], fontsize=6.5, pad=3)
        axes[0].set_ylabel("Median lead time (s)")
        for ax, L in zip(axes, "abc"):
            panel_label(ax, L, dx=-0.14)

        handles = ([Line2D([], [], marker="*", ls="", ms=6, color=T0,
                           label="Tier-0 (classical EWS), best over the channel × rule grid")] +
                   [Line2D([], [], marker=MRK[h], ls="", ms=3.4, color=COL[f],
                           label=f"{FEAT_LABEL[f]}, {HEAD_LABEL[h]}")
                    for f in ("stats", "emb") for h in ("lr", "gbt")])
        axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                       ncol=3, fontsize=5.5)
        fig.subplots_adjust(left=0.055, right=0.995, top=0.92, bottom=0.30, wspace=0.22)
        save(fig, "fig1_frontier")


# --------------------------------------------------------------------------- fig 2
def fig2(summ, t0):
    """The band effect. Two rows (the two splits where dominance holds), two
    columns (the two axes). The Tier-0 bar is drawn as a solid rule because it is
    the threshold the claim is made against."""
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 96 * MM), sharex=True)
        arms4 = [("stats", "lr"), ("stats", "gbt"), ("emb", "lr"), ("emb", "gbt")]
        x = np.arange(len(BAND_ORDER))
        for i, sp in enumerate(["B_fixed", "C"]):
            bar = t0[t0["split"] == sp].iloc[0]
            for j, (metric, sdcol, ylab) in enumerate(
                    [("det", "det_sd", "Detection rate at FAR 1.6 h$^{-1}$"),
                     ("lead", "lead_sd", "Median lead time (s)")]):
                ax = axes[i, j]
                barval = bar["best_det"] if metric == "det" else bar["best_lead"]
                ax.axhline(barval, color=T0, lw=0.8, zorder=1)
                ax.annotate("Tier-0, best over channel × rule grid", (0.015, barval),
                            xycoords=("axes fraction", "data"), fontsize=5.5,
                            color=T0, va="bottom", ha="left",
                            xytext=(0, 2), textcoords="offset points")
                for f, h in arms4:
                    d = summ[(summ["split"] == sp) & (summ["features"] == f)
                             & (summ["head"] == h)].set_index("band_label")
                    d = d.reindex(BAND_ORDER)
                    ax.errorbar(x, d[metric], yerr=d[sdcol], marker=MRK[h], ms=3.4,
                                mew=0, lw=0.9, color=COL[f], ecolor=COL[f],
                                elinewidth=0.6, capsize=1.2, zorder=3)
                ax.set_ylabel(ylab)
                ax.set_ylim(bottom=0)
                ax.set_xlim(-0.28, len(x) - 1 + 0.28)
                if i == 1:
                    ax.set_xticks(x)
                    ax.set_xticklabels(["1–10 s\n(original)", "111–120 s", "141–150 s"])
                    ax.set_xlabel("Training label band, before the event")
            axes[i, 0].text(-0.245, 0.5, "Split " + SPLIT_LABEL[sp],
                            transform=axes[i, 0].transAxes, rotation=90,
                            va="center", ha="center", fontsize=6.5, color=INK)
        for ax, L in zip(axes.ravel(), "abcd"):
            panel_label(ax, L, dx=-0.12)
        handles = [Line2D([], [], marker=MRK[h], ls="-", ms=3.4, color=COL[f],
                          label=f"{FEAT_LABEL[f]}, {HEAD_LABEL[h]}")
                   for f, h in arms4]
        axes[1, 0].legend(handles=handles, loc="upper center",
                          bbox_to_anchor=(1.1, -0.30), ncol=4, fontsize=5.5)
        fig.subplots_adjust(left=0.125, right=0.99, top=0.97, bottom=0.20,
                            wspace=0.19, hspace=0.10)
        save(fig, "fig2_band")


# --------------------------------------------------------------------------- fig 3
def fig3(summ, arms):
    """What the representation buys, and what it does not."""
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM),
                                 gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.2]))
        ax_a, ax_b, ax_c = axes

        # (a) linear head, embeddings vs statistics -- the surviving claim
        cells = [("B_fixed", "1-10 s"), ("B_fixed", "111-120 s"),
                 ("C", "1-10 s"), ("C", "111-120 s")]
        w, xs = 0.36, np.arange(len(cells))
        for k, f in enumerate(("stats", "emb")):
            vals, errs = [], []
            for sp, bl in cells:
                r = summ[(summ["split"] == sp) & (summ["band_label"] == bl)
                         & (summ["features"] == f) & (summ["head"] == "lr")]
                vals.append(r["det"].iloc[0] if len(r) else np.nan)
                errs.append(r["det_sd"].iloc[0] if len(r) else np.nan)
            ax_a.bar(xs + (k - 0.5) * w, vals, w, yerr=errs, color=COL[f],
                     linewidth=0, error_kw=dict(elinewidth=0.6, capsize=1.4,
                                                ecolor=INK2), label=FEAT_LABEL[f])
        for i, (sp, bl) in enumerate(cells):
            a = summ[(summ["split"] == sp) & (summ["band_label"] == bl)
                     & (summ["head"] == "lr")].set_index("features")["det"]
            if {"emb", "stats"} <= set(a.index):
                ax_a.text(i, max(a) + 0.022, f"+{(a['emb']/a['stats']-1)*100:.0f}%",
                          ha="center", fontsize=5.5, color=EMB, fontweight="bold")
        ax_a.set_xticks(xs)
        ax_a.set_xticklabels([f"{'B' if sp=='B_fixed' else sp}\n{bl}" for sp, bl in cells])
        ax_a.set_ylabel("Detection rate at FAR 1.6 h$^{-1}$")
        ax_a.set_title("Linear head, both feature sets", fontsize=6.5, pad=3)
        ax_a.legend(loc="upper right", fontsize=5.5)
        ax_a.set_ylim(0, 0.60)

        # (b) matched head -- the trade, as a percentage change of emb over stats
        pairs = [("B_fixed", "111-120 s", "gbt"), ("B_fixed", "141-150 s", "gbt"),
                 ("B_fixed", "111-120 s", "lr"), ("B_fixed", "141-150 s", "lr")]
        dd, dl, labs = [], [], []
        for sp, bl, h in pairs:
            g = summ[(summ["split"] == sp) & (summ["band_label"] == bl)
                     & (summ["head"] == h)].set_index("features")
            dd.append((g.loc["emb", "det"] / g.loc["stats", "det"] - 1) * 100)
            dl.append((g.loc["emb", "lead"] / g.loc["stats", "lead"] - 1) * 100)
            labs.append(f"{bl}\n{'tree head' if h == 'gbt' else 'linear head'}")
        xs = np.arange(len(pairs))
        ax_b.bar(xs - 0.19, dd, 0.38, color="#7fa8cd", linewidth=0, label="detection")
        ax_b.bar(xs + 0.19, dl, 0.38, color=EMB, linewidth=0, label="lead time")
        ax_b.axhline(0, color=INK2, lw=0.6)
        ax_b.set_xticks(xs)
        ax_b.set_xticklabels(labs, fontsize=5.5)
        ax_b.set_ylabel("Embeddings vs statistics (%)")
        ax_b.set_title("Matched head, cross-airframe", fontsize=6.5, pad=3)
        lo, hi = min(dd + dl), max(dd + dl)
        ax_b.set_ylim(lo - 0.18 * (hi - lo), hi + 0.30 * (hi - lo))
        ax_b.legend(loc="upper right", fontsize=5.5, ncol=2)

        # (c) seed stability at the original band -- which single-seed numbers count
        cells3 = [("stats", "lr"), ("emb", "lr"), ("stats", "gbt"), ("emb", "gbt")]
        xs = np.arange(len(cells3))
        for i, (f, h) in enumerate(cells3):
            pts = arms[(arms["split"] == "B_fixed") & (arms["band"] == "widen10")
                       & (arms["features"] == f) & (arms["head"] == h)]["det"].values
            r = summ[(summ["split"] == "B_fixed") & (summ["band_label"] == "1-10 s")
                     & (summ["features"] == f) & (summ["head"] == h)].iloc[0]
            ax_c.errorbar(i, r["det"], yerr=r["det_sd"], fmt=MRK[h], ms=4.5, mew=0,
                          color=COL[f], ecolor=COL[f], elinewidth=0.8, capsize=2,
                          zorder=3)
            ax_c.scatter(np.full(len(pts), i) + 0.19, pts, s=5, facecolor="white",
                         edgecolor=COL[f], linewidth=0.5, zorder=4)
            # the sd is the point of this panel, so print it beside each arm
            ax_c.annotate(f"±{r['det_sd']:.3f}", (i, r["det"]),
                          textcoords="offset points", xytext=(-8, 0),
                          ha="right", va="center", fontsize=5.5, color=INK)
        ax_c.set_xticks(xs)
        ax_c.set_xticklabels([f"{'stats' if f=='stats' else 'emb'}\n{h}"
                              for f, h in cells3])
        ax_c.set_xlim(-0.72, len(cells3) - 0.35)
        ax_c.set_ylabel("Detection rate at FAR 1.6 h$^{-1}$")
        ax_c.set_title("Seed spread, original band, split B", fontsize=6.5, pad=3)

        for ax, L in zip(axes, "abc"):
            panel_label(ax, L, dx=-0.14)
        fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.19, wspace=0.30)
        save(fig, "fig3_representation")


# --------------------------------------------------------------------------- fig 4
def fig4(rules):
    """Robustness to the alarm rule. Reported as a COUNT over the whole grid
    rather than a selected best, because the rule was never chosen on held-out
    data (see the caveat in the results document)."""
    with plt.rc_context(rc()):
        fig, ax = plt.subplots(figsize=(120 * MM, 56 * MM))
        fams = [("stats", "lr"), ("stats", "gbt"), ("emb", "lr"), ("emb", "gbt")]
        splits = ["B_fixed", "C"]
        w = 0.36
        xs = np.arange(len(fams))
        for k, sp in enumerate(splits):
            vals = []
            for f, h in fams:
                r = rules[(rules["split"] == sp) & (rules["features"] == f)
                          & (rules["head"] == h)]
                vals.append(r["frac"].iloc[0] if len(r) else np.nan)
            bars = ax.bar(xs + (k - 0.5) * w, vals, w, linewidth=0,
                          color="#9dbcd8" if sp == "B_fixed" else EMB,
                          label=f"{SPLIT_LABEL[sp].split(' — ')[0]} "
                                f"({SPLIT_LABEL[sp].split(' — ')[1]})")
            for b, f_, (ff, hh) in zip(bars, vals, fams):
                r = rules[(rules["split"] == sp) & (rules["features"] == ff)
                          & (rules["head"] == hh)]
                if len(r):
                    ax.text(b.get_x() + b.get_width() / 2, f_ + 0.02,
                            f"{int(r['n_dominating'].iloc[0])}/{int(r['n_configs'].iloc[0])}",
                            ha="center", fontsize=5.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{FEAT_LABEL[f].split()[-1]}\n{HEAD_LABEL[h].split()[0]}"
                            for f, h in fams])
        ax.set_ylabel("Fraction of rule settings\nthat dominate Tier-0")
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(loc="upper left", fontsize=5.5)
        fig.subplots_adjust(left=0.16, right=0.99, top=0.96, bottom=0.20)
        save(fig, "fig4_rules")


def main():
    summ, t0, arms, rules = load()
    print(f"loaded {len(summ)} n>=3 configs, {len(arms)} arm runs")
    fig1(summ, t0)
    fig2(summ, t0)
    fig3(summ, arms)
    fig4(rules)


if __name__ == "__main__":
    main()
