#!/usr/bin/env python
"""HotMobile figures (v2, simple). Inputs: HotMobile/curves/*.csv (alarm_curves.py),
meta/paper/summary.csv, and two score dumps for the example flight.
Writes HotMobile/paper/figs/{warning,horizon,flight}.pdf (+png previews in figsrc/preview/)."""
import glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/wsssd/FlightLog")
from paper_figs import rc as _rc, INK, INK2, MM
def rc(sc):  # HotMobile: black axes, ticks and tick labels on every figure
    d = _rc(sc); d.update({"xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK}); return d
ROOT = "/mnt/wsssd/FlightLog"; C = f"{ROOT}/HotMobile/curves"
FIG = f"{ROOT}/HotMobile/paper/figs"; PRE = f"{ROOT}/HotMobile/figsrc/preview"
os.makedirs(FIG, exist_ok=True); os.makedirs(PRE, exist_ok=True)
SC = 1.2
OURS, REL, STATS, TSFM, GREY = "#1f5fbf", "#6c757d", "#e8710a", "#8c6bb1", "#9a9a9a"

def save(fig, stem):
    fig.savefig(f"{FIG}/{stem}.pdf"); fig.savefig(f"{PRE}/{stem}.png", dpi=300); plt.close(fig); print("wrote", stem)

def curve(pattern):
    fs = sorted(f for f in glob.glob(f"{C}/{pattern}") if f.endswith(".csv"))
    if not fs: return None
    F = np.mean([pd.read_csv(f).F.to_numpy() for f in fs], axis=0)
    return pd.read_csv(fs[0]).tau.to_numpy(), F

def envelope(split):
    fs = sorted(f for f in glob.glob(f"{C}/ews_scores_*__{split}.csv") if f.endswith(".csv"))
    if not fs: return None
    F = np.max([pd.read_csv(f).F.to_numpy() for f in fs], axis=0)
    return pd.read_csv(fs[0]).tau.to_numpy(), F, len(fs)

# ------------------------------------------------------------ Fig: warning curves
with plt.rc_context(rc(SC)):
    fig, axes = plt.subplots(2, 1, figsize=(84 * MM, 90 * MM), sharex=True)
    for ax, sp, title in zip(axes, ["B_fixed", "C"], ["Cross-airframe", "Cross-firmware"]):
        env = envelope(sp)
        if env:
            tau, F, n = env
            ax.fill_between(tau, 0, F, color=GREY, alpha=0.28, lw=0)
        arms = [(f"horizon_scores_shift120_{sp}*[!t]__raw*", OURS, "-", "Flight-pretrained"),
                (f"horizon_scores_shift120_{sp}*_relcat__raw*", OURS, (0, (2.5, 1.5)), "+ flight-relative"),
                (f"h1_scores_lr_shift120_{sp}*__raw*", STATS, "-", "Window statistics")]
        if sp == "C":
            arms.append((f"horizon_scores_shift120_C*_tsfm__raw*", TSFM, "-", "MOMENT"))
        for pat, col, ls, lab in arms:
            pat = pat.replace("*[!t]", "") if "relcat" in pat else pat
            fs = [f for f in glob.glob(f"{C}/{pat}") if f.endswith(".csv") and ("relcat" in f) == ("relcat" in pat) and ("tsfm" in f) == ("tsfm" in pat)]
            if not fs: continue
            F = np.mean([pd.read_csv(f).F.to_numpy() for f in fs], axis=0); tau = pd.read_csv(fs[0]).tau.to_numpy()
            ax.plot(tau, F, color=col, ls=ls, lw=1.3, label=lab)
        ax.set_xlim(0, 300); ax.set_ylim(0, 0.32); ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_title(title, fontsize=7 * SC, loc="left", pad=3, color=INK)
        ax.set_ylabel("Warned ≥ τ s ahead")
    axes[1].set_xlabel("τ (s)")
    from matplotlib.patches import Patch
    h, l = axes[1].get_legend_handles_labels(); h.append(Patch(facecolor=GREY, alpha=0.28, lw=0)); l.append("Classical")
    axes[1].legend(h, l, fontsize=6 * SC, loc="upper center", bbox_to_anchor=(0.4, -0.32), ncol=2, frameon=False, handlelength=1.8, columnspacing=1.5)
    for ax, L in zip(axes, "ab"):
        ax.text(-0.2, 1.04, L, transform=ax.transAxes, fontsize=9 * SC, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.25, hspace=0.4)
    save(fig, "warning")

# ------------------------------------------------------------ Fig: horizon knob
s = pd.read_csv(f"{ROOT}/meta/paper/summary.csv"); s = s[(s.n >= 3) & (s.features == "emb") & (s["head"] == "lr")]
bands = [("widen10", 5.5), ("shift120", 115.5), ("shift150", 145.5)]
with plt.rc_context(rc(SC)):
    fig, axes = plt.subplots(1, 2, figsize=(84 * MM, 46 * MM))
    for sp, col, mk, lab in [("B_fixed", OURS, "o", "Cross-airframe"), ("C", REL, "s", "Cross-firmware")]:
        x, lead, lsd, det, dsd = [], [], [], [], []
        for b, mid in bands:
            mid = {5.5: 0, 115.5: 1, 145.5: 2}[mid]
            r = s[(s.split == sp) & (s.band == b)]
            if len(r): x.append(mid); lead.append(r.lead.iloc[0]); lsd.append(r.lead_sd.iloc[0]); det.append(r.det.iloc[0]); dsd.append(r.det_sd.iloc[0])
        axes[0].errorbar(x, lead, yerr=lsd, color=col, marker=mk, ms=3.5, lw=1.2, capsize=1.5, label=lab)
        axes[1].errorbar(x, det, yerr=dsd, color=col, marker=mk, ms=3.5, lw=1.2, capsize=1.5, label=lab)
    for ax in axes:
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["1–10", "111–120", "141–150"])
        ax.set_xlabel(""); ax.set_xlim(-0.3, 2.3)
    axes[0].set_ylabel("Median lead (s)"); axes[0].set_ylim(0, 120)
    axes[1].set_ylabel("Detection"); axes[1].set_ylim(0, 0.55)
    fig.text(0.55, 0.2, "Training band (s before failsafe)", ha="center", fontsize=7 * SC)
    axes[0].legend(fontsize=6 * SC, loc="upper center", bbox_to_anchor=(1.12, -0.5), ncol=2, frameon=False, columnspacing=1.5)
    for ax, L in zip(axes, "ab"):
        ax.text(-0.3, 1.04, L, transform=ax.transAxes, fontsize=9 * SC, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.4, wspace=0.5)
    save(fig, "horizon")

# ------------------------------------------------------------ Fig: one flight
LID = "062c856c-ff13-442d-8773-9a8e861ac839"
th_o = float(open(f"{C}/horizon_scores_shift120_B_fixed__raw__None__own.csv.th").read())
th_s = float(open(f"{C}/h1_scores_lr_shift120_B_fixed__raw__None__own.csv.th").read())
sp = pd.read_parquet(f"{ROOT}/meta/splits_eval.parquet").set_index("log_id"); te = sp.t_e_bin[LID]
o = pd.read_parquet(f"{ROOT}/meta/scores/horizon_scores_shift120_B_fixed.parquet"); o = o[o.log_id == LID].sort_values("t_bin")
st = pd.read_parquet(f"{ROOT}/meta/scores/h1_scores_lr_shift120_B_fixed.parquet"); st = st[st.log_id == LID].sort_values("t_bin")
print("flight", LID, sp.mav_type[LID], "t_e", te / 20, "s; n_bins", sp.n_bins[LID])
with plt.rc_context(rc(SC)):
    fig, ax = plt.subplots(figsize=(84 * MM, 42 * MM))
    for df, th, col, lab in [(st, th_s, STATS, "Window statistics"), (o, th_o, OURS, "Flight-pretrained")]:
        t = (df.t_bin.to_numpy() - te) / 20; r = df.score.to_numpy() / th
        ax.plot(t, r, color=col, lw=1.1, label=lab)
        hit = np.flatnonzero((r >= 1) & (t < 0))
        if len(hit):
            ax.plot(t[hit[0]], r[hit[0]], "o", color=col, ms=4.5, mec="white", mew=0.6)
            ax.annotate(f"{-t[hit[0]]:.0f} s early", (t[hit[0]], r[hit[0]]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=6.5 * SC, color=col)
    ax.axhline(1, color=INK2, lw=0.7, ls=(0, (3, 2))); ax.text(-150, 1.35, "Threshold", fontsize=6 * SC, color=INK2, ha="center")
    ax.axvline(0, color="#c0392b", lw=0.8); ax.text(-3, 14, "Failsafe", fontsize=6 * SC, color="#c0392b", va="center", ha="right")
    ax.set_yscale("log"); ax.set_xlim(-240, 8); ax.set_ylim(0.005, 30)
    ax.set_xlabel("Time to failsafe (s)"); ax.set_ylabel("Score / threshold")
    ax.legend(fontsize=5.8 * SC, loc="upper left", frameon=False, bbox_to_anchor=(-0.01, 1.03))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.24)
    save(fig, "flight")
