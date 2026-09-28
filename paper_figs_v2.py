#!/usr/bin/env python
"""Paper figures, version 2 (2026-09-14) — every learned arm against the FULL Tier-0 cloud.

Built from `meta/paper/*.csv` (paper_tables.py) and `meta/sequential.csv`; nothing is
refitted or rescored. Writes meta/fig/v2/.

  fig1_frontier    detection–lead plane per split. Grey dots = every one of the 165 Tier-0
                   configurations of the channel × rule cross (15 channel sets × 11 rules);
                   dashed rules = its per-axis maxima; shaded = the region that beats both.
                   Learned arms at the 111–120 s band with seed spread.
  fig2_band        the band effect as a dumbbell: each arm's lead and detection at the
                   original 1–10 s band vs the 111–120 s band, on B_fixed and C.
  fig3_linear      representation at matched head: detection by family (statistics,
                   MOMENT TSFM, ours, ours + flight-relative) at the linear head and at the
                   boosted-tree head, per split and band.
  fig4_rules       fraction of the (band, rule) configurations that dominate the cross bar,
                   per family and head (20 per family; 10 for the flight-relative arms, which
                   were run at the 111–120 s band only).
  fig5_ablation    items 6 + 8 (2026-09-18): read-out view (mean / most-diverging channel /
                   per-role) × encoder (original / channel-mask continued / random-mask
                   continued) at the 111–120 s band, both heads, against the cross bar.

Usage: python paper_figs_v2.py
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

from paper_figs import rc, panel_label, INK, INK2, MM              # noqa: E402

PAPER = Path("/mnt/wsssd/FlightLog/meta/paper")
META = Path("/mnt/wsssd/FlightLog/meta")
FIG = Path("/mnt/wsssd/FlightLog/meta/fig/v2")

COL = {"stats": "#e8710a", "tsfm": "#8c6bb1", "emb": "#0d3b70", "relcat": "#4585c7",
       "rel": "#9ecae1"}
T0 = "#7f7f7f"
MRK = {"lr": "o", "gbt": "s"}
FAM = {"stats": "hand-built statistics", "tsfm": "MOMENT (general TSFM)",
       "emb": "ours (flight-pretrained)", "relcat": "ours + flight-relative",
       "rel": "flight-relative only"}
HEAD = {"lr": "linear head", "gbt": "boosted-tree head"}
SPLIT = {"A": "A — in-distribution", "B_fixed": "B — cross-airframe",
         "C": "C — schema-conditioned"}
FAM_ORDER = ["stats", "tsfm", "emb", "relcat"]


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=600 if ext == "png" else None)
        print(f"  wrote {FIG / f'{stem}.{ext}'}")
    plt.close(fig)


def load():
    s = pd.read_csv(PAPER / "summary.csv")
    s = s[s["n"] >= 3].copy()
    t0 = pd.read_csv(PAPER / "tier0.csv").set_index("split")
    sq = META / "sequential.csv"
    if sq.exists():
        seq = pd.read_csv(sq)
        cloud = seq[seq["arm"].str.startswith("ews_scores")
                    & (seq["arm"] != "ews_scores_smoke_dump")].copy()   # 15 sets × 11 rules = 165
    else:                                   # table being rebuilt: draw fig 1 without the cloud
        cloud = pd.DataFrame(columns=["arm", "eval_split", "det@1.6h", "lead@1.6h"])
    rules = pd.read_csv(PAPER / "rules.csv")
    return s, t0, cloud, rules


def arm(s, f, h, band, sp):
    d = s[(s["features"] == f) & (s["head"] == h) & (s["band"] == band) & (s["split"] == sp)]
    return d.iloc[0] if len(d) else None


# --------------------------------------------------------------------------- fig 1
def fig1(s, t0, cloud):
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 66 * MM))
        for ax, sp in zip(axes, ["A", "B_fixed", "C"]):
            c = cloud[cloud["eval_split"] == sp]
            bd, bl = t0.loc[sp, "best_det"], t0.loc[sp, "best_lead"]
            if sp == "A":
                arms = [("emb", "lr", b) for b in ("shift120", "shift240", "shift300", "shift360")]
            else:
                arms = [(f, h, "shift120") for f in FAM_ORDER for h in ("lr", "gbt")]
            rows = [(f, h, b, arm(s, f, h, b, sp)) for f, h, b in arms]
            rows = [r for r in rows if r[3] is not None and np.isfinite(r[3]["lead"])]
            xmax = max([r[3]["det"] + r[3]["det_sd"] for r in rows] + [bd]) * 1.18
            ymax = max([r[3]["lead"] + r[3]["lead_sd"] for r in rows] + [bl] + [c["lead@1.6h"].max()]) * 1.12
            ax.add_patch(Rectangle((bd, bl), xmax - bd, ymax - bl, facecolor="#eef3f8",
                                   edgecolor="none", zorder=0))
            ax.scatter(c["det@1.6h"], c["lead@1.6h"], s=4, color=T0, alpha=0.55,
                       linewidths=0, zorder=2)
            ax.axvline(bd, color=T0, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.axhline(bl, color=T0, lw=0.6, ls=(0, (3, 2)), zorder=1)
            for f, h, b, r in rows:
                ax.errorbar(r["det"], r["lead"], xerr=r["det_sd"], yerr=r["lead_sd"],
                            fmt=MRK[h], ms=3.8, mew=0, color=COL[f], ecolor=COL[f],
                            elinewidth=0.6, capsize=1.2, zorder=4)
                if sp == "A":
                    ax.annotate(b.replace("shift", "") + " s", (r["det"], r["lead"]),
                                textcoords="offset points", xytext=(4, 3), fontsize=5,
                                color=COL[f])
            ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
            ax.set_xlabel("detection rate at FAR 1.6 h$^{-1}$")
            ax.set_title(SPLIT[sp], fontsize=6.5, pad=3)
            ax.text(0.98, 0.97, f"n = {len(c)} Tier-0 configurations", transform=ax.transAxes,
                    ha="right", va="top", fontsize=5.2, color=INK2)
        axes[0].set_ylabel("median lead time (s)")
        for ax, L in zip(axes, "abc"):
            panel_label(ax, L, dx=-0.16)
        handles = [Line2D([], [], marker=".", ls="", ms=5, color=T0,
                          label="classical EWS, one of the 165 channel × rule configurations"),
                   Line2D([], [], ls=(0, (3, 2)), color=T0, lw=0.8,
                          label="its per-axis maximum (the bar)")]
        handles += [Line2D([], [], marker=MRK[h], ls="", ms=3.6, color=COL[f],
                           label=f"{FAM[f]}, {HEAD[h]}")
                    for f in FAM_ORDER for h in ("lr", "gbt")]
        axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.26),
                       ncol=3, fontsize=5.4, columnspacing=1.2)
        fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.36, wspace=0.24)
        save(fig, "fig1_frontier")


# --------------------------------------------------------------------------- fig 2
def fig2(s, t0):
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 92 * MM))
        for i, sp in enumerate(["B_fixed", "C"]):
            fams = [f for f in FAM_ORDER if arm(s, f, "lr", "widen10", sp) is not None]
            arms = [(f, h) for f in fams for h in ("lr", "gbt")
                    if arm(s, f, h, "widen10", sp) is not None and arm(s, f, h, "shift120", sp) is not None]
            ys = np.arange(len(arms))[::-1]
            for j, (metric, sd, xlab) in enumerate((("lead", "lead_sd", "median lead time (s)"),
                                                    ("det", "det_sd", "detection rate at FAR 1.6 h$^{-1}$"))):
                ax = axes[i, j]
                bar = t0.loc[sp, "best_lead" if metric == "lead" else "best_det"]
                ax.axvline(bar, color=T0, lw=0.7, ls=(0, (3, 2)), zorder=1)
                for y, (f, h) in zip(ys, arms):
                    a, b = arm(s, f, h, "widen10", sp), arm(s, f, h, "shift120", sp)
                    if not (np.isfinite(a[metric]) and np.isfinite(b[metric])):
                        continue
                    ax.plot([a[metric], b[metric]], [y, y], color=COL[f], lw=1.0, zorder=2)
                    ax.scatter([a[metric]], [y], s=16, facecolors="white", edgecolors=COL[f],
                               linewidths=0.8, marker=MRK[h], zorder=3)
                    ax.errorbar(b[metric], y, xerr=b[sd], fmt=MRK[h], ms=4, mew=0, color=COL[f],
                                ecolor=COL[f], elinewidth=0.6, capsize=1.2, zorder=4)
                    if metric == "lead":
                        ax.annotate(f"×{b[metric] / a[metric]:.1f}", (b[metric], y),
                                    textcoords="offset points", xytext=(6, -2), fontsize=5.2,
                                    color=COL[f], va="center")
                ax.set_yticks(ys)
                ax.set_yticklabels([f"{FAM[f]}, {h}" for f, h in arms] if j == 0 else [])
                ax.set_xlabel(xlab)
                ax.margins(y=0.12)
                if metric == "lead":
                    ax.set_xscale("log")
                    ax.set_xlim(2.5, 400)
                else:
                    ax.set_xlim(0, 0.65)
                ax.text(0.99, 0.02, "Tier-0 bar" if j == 0 else "Tier-0 bar", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=5, color=T0)
                ax.set_title(SPLIT[sp] if j == 0 else "", fontsize=6.5, pad=3, loc="left")
        for ax, L in zip(axes.ravel(), "abcd"):
            panel_label(ax, L, dx=-0.62 if ax in axes[:, 0] else -0.10)
        handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=4, label="trained on the 1–10 s band"),
                   Line2D([], [], marker="o", ls="", color=INK2, ms=4, label="trained on the 111–120 s band (± sd, n = 3)")]
        axes[1, 0].legend(handles=handles, loc="upper center", bbox_to_anchor=(1.15, -0.32), ncol=2, fontsize=5.6)
        fig.subplots_adjust(left=0.30, right=0.99, top=0.95, bottom=0.17, wspace=0.10, hspace=0.42)
        save(fig, "fig2_band")


# --------------------------------------------------------------------------- fig 3
def fig3(s):
    cells = [("B_fixed", "widen10"), ("B_fixed", "shift120"), ("C", "widen10"), ("C", "shift120"),
             ("C", "shift150")]
    lab = {"widen10": "1–10 s", "shift120": "111–120 s", "shift150": "141–150 s"}
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM))
        for ax, h in zip(axes, ("lr", "gbt")):
            x0 = 0
            ticks, tlabs = [], []
            for sp, band in cells:
                fams = [f for f in FAM_ORDER if arm(s, f, h, band, sp) is not None]
                w = 0.8 / max(len(FAM_ORDER), 1)
                for k, f in enumerate(FAM_ORDER):
                    r = arm(s, f, h, band, sp)
                    if r is None:
                        continue
                    xx = x0 + k * w
                    ax.bar(xx, r["det"], width=w * 0.92, color=COL[f], zorder=3, linewidth=0)
                    ax.errorbar(xx, r["det"], yerr=r["det_sd"], fmt="none", ecolor=INK2,
                                elinewidth=0.6, capsize=1.2, zorder=4)
                st, em = arm(s, "stats", h, band, sp), arm(s, "emb", h, band, sp)
                if st is not None and em is not None:
                    g = 100 * (em["det"] / st["det"] - 1)
                    ax.text(x0 + 2 * w, em["det"] + em["det_sd"] + 0.012, f"{g:+.0f}%", ha="center",
                            va="bottom", fontsize=5.4, color=COL["emb"], fontweight="bold")
                ticks.append(x0 + 1.5 * w); tlabs.append(f"{sp.replace('_fixed', '')}\n{lab[band]}")
                x0 += 1.2
            ax.set_xticks(ticks); ax.set_xticklabels(tlabs)
            ax.set_ylabel("detection rate at FAR 1.6 h$^{-1}$")
            ax.set_title(HEAD[h], fontsize=6.5, pad=3)
            ax.set_ylim(0, 0.68)
            ax.grid(axis="y", color="#eeeeee", lw=0.4, zorder=0)
        for ax, L in zip(axes, "ab"):
            panel_label(ax, L, dx=-0.16)
        handles = [Line2D([], [], marker="s", ls="", color=COL[f], ms=5, label=FAM[f]) for f in FAM_ORDER]
        axes[0].legend(handles=handles, loc="upper left", fontsize=5.6, ncol=2)
        fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.20, wspace=0.22)
        save(fig, "fig3_linear")


# --------------------------------------------------------------------------- fig 4
def fig4(rules):
    r = rules[rules.split.isin(["B_fixed", "C"]) & rules.features.isin(FAM)].copy()
    r["fam"] = pd.Categorical(r.features, [f for f in FAM_ORDER if f in set(r.features)])
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 52 * MM), sharey=False)
        for ax, sp in zip(axes, ["B_fixed", "C"]):
            d = r[r.split == sp].sort_values(["fam", "head"]).reset_index(drop=True)
            ys = np.arange(len(d))[::-1]
            ax.barh(ys, d.frac, color=[COL[f] for f in d.features], height=0.7, zorder=3, linewidth=0)
            for y, (_, row) in zip(ys, d.iterrows()):
                ax.text(row.frac + 0.01, y, f"{int(row.n_dominating)}/{int(row.n_configs)}", va="center",
                        fontsize=5.4, color=INK2)
            ax.set_yticks(ys)
            ax.set_yticklabels([f"{FAM[f]}, {h}" for f, h in zip(d["features"], d["head"])])
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("fraction of rule configurations beating the Tier-0 bar")
            ax.set_title(SPLIT[sp], fontsize=6.5, pad=3)
            ax.grid(axis="x", color="#eeeeee", lw=0.4, zorder=0)
        for ax, L in zip(axes, "ab"):
            panel_label(ax, L, dx=-0.80)
        fig.subplots_adjust(left=0.22, right=0.985, top=0.90, bottom=0.26, wspace=0.95)
        save(fig, "fig4_rules")


# --------------------------------------------------------------------------- fig 5
ENC = {"v": ("original encoder", "#0d3b70"), "ch": ("+ channel-mask pretraining", "#1b9e77"),
       "rd": ("+ random-mask pretraining (control)", "#e8710a")}
VIEW = {"mean": ("mean + last (baseline read-out)", "o"), "dev": ("most-diverging channel + last + ‖dev‖", "^"),
        "role": ("per-role (8 × 512) + last", "D")}


def fig5(s, t0):
    """Item 6 / item 8 ablation: read-out view × continued-pretraining pretext, at the
    111–120 s band, in the detection–lead plane against the Tier-0 cross bar.
    Marker = view, filled = linear head, hollow = boosted-tree head; colour = encoder."""
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 66 * MM))
        for ax, sp in zip(axes, ["B_fixed", "C"]):
            bd, bl = t0.loc[sp, "best_det"], t0.loc[sp, "best_lead"]
            pts = []
            for pre, (elab, col) in ENC.items():
                for view, (vlab, mk) in VIEW.items():
                    tag = f"{pre}{view}"
                    for h in ("lr", "gbt"):
                        r = arm(s, tag, h, "shift120", sp)
                        if r is None or not np.isfinite(r["lead"]):
                            continue
                        pts.append((r, col, mk, h))
            xmax = max([r["det"] + r["det_sd"] for r, *_ in pts] + [bd]) * 1.15
            ymax = max([r["lead"] + r["lead_sd"] for r, *_ in pts] + [bl]) * 1.10
            ax.add_patch(Rectangle((bd, bl), xmax - bd, ymax - bl, facecolor="#eef3f8",
                                   edgecolor="none", zorder=0))
            ax.axvline(bd, color=T0, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.axhline(bl, color=T0, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ref = arm(s, "relcat", "gbt", "shift120", sp)
            if ref is not None and np.isfinite(ref["lead"]):
                ax.scatter([ref["det"]], [ref["lead"]], marker="*", s=28, color=COL["relcat"],
                           linewidths=0, zorder=3)
            for r, col, mk, h in pts:
                ax.errorbar(r["det"], r["lead"], xerr=r["det_sd"], yerr=r["lead_sd"], fmt=mk,
                            ms=4 if mk != "^" else 4.6, color=col, ecolor=col, elinewidth=0.6,
                            capsize=1.2, zorder=4, mfc=col if h == "lr" else "white",
                            mec=col, mew=0.8)
            ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
            ax.set_xlabel("detection rate at FAR 1.6 h$^{-1}$")
            ax.set_title(SPLIT[sp], fontsize=6.5, pad=3)
            ax.text(0.99, 0.02, "Tier-0 cross bar", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5, color=T0)
        axes[0].set_ylabel("median lead time (s)")
        for ax, L in zip(axes, "ab"):
            panel_label(ax, L, dx=-0.14)
        handles = [Line2D([], [], marker="s", ls="", ms=4, color=c, label=l) for l, c in ENC.values()]
        handles += [Line2D([], [], marker=m, ls="", ms=4, color=INK2, label=l) for l, m in VIEW.values()]
        handles += [Line2D([], [], marker="o", ls="", ms=4, color=INK2, label="linear head"),
                    Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=INK2, label="boosted-tree head"),
                    Line2D([], [], marker="*", ls="", ms=6, color=COL["relcat"],
                           label="ours + flight-relative, boosted-tree (§5.3 reference)")]
        axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(1.1, -0.24), ncol=3,
                       fontsize=5.2, columnspacing=1.0)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.36, wspace=0.18)
        save(fig, "fig5_ablation")


def main():
    s, t0, cloud, rules = load()
    fig1(s, t0, cloud); fig2(s, t0); fig3(s); fig4(rules); fig5(s, t0)


if __name__ == "__main__":
    main()
