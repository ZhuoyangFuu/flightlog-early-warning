#!/usr/bin/env bash
# RESULTS.md (v9) + the v2 figures -> docs/RESULTS.docx (pandoc). Figures are appended as a
# final section so the Markdown stays readable on its own.
ROOT=/mnt/wsssd/FlightLog; cd $ROOT
tmp=$(mktemp --suffix=.md)
cp RESULTS.md "$tmp"
cat >> "$tmp" <<'MD'

---

## Figures (rendered)

![Figure 1 | Detection–lead plane per split. Grey dots: all 165 classical-EWS configurations (15 channel sets × 11 alarm rules); dashed: their per-axis maxima; shaded: the region beating both. Learned arms at the 111–120 s band, n = 3 ± sd.](meta/fig/v2/fig1_frontier.png)

![Figure 2 | The label band sets the lead. Each arm's median lead (log axis, multiplier printed) and detection at the 1–10 s band (hollow) versus the 111–120 s band (filled, ± sd), splits B_fixed and C; dashed: Tier-0 cross bar.](meta/fig/v2/fig2_band.png)

![Figure 3 | Representation at matched head. Detection at FAR 1.6 h⁻¹ by feature family at a linear head (a) and a boosted-tree head (b), per split and band; the printed percentage is ours versus hand-built statistics.](meta/fig/v2/fig3_linear.png)

![Figure 4 | Robustness to the alarm rule: fraction of the (band, rule) configurations whose n = 3 mean beats the Tier-0 cross bar on both axes, per family and head (20 per family; 10 for the flight-relative arms, swept at the 111–120 s band only).](meta/fig/v2/fig4_rules.png)

![Figure 5 | Read-out and pretext ablation (items 6 and 8) at the 111–120 s band. Colour: encoder (original; + 4 epochs of channel-mask continued pretraining; + 4 epochs of random-mask continued pretraining, the extra-steps control). Marker: pooled read-out (mean + last; most-diverging channel + last + per-channel deviation norms; per-role, pending). Filled = linear head, hollow = boosted-tree head; n = 3 ± sd. Dashed: Tier-0 cross bar; shaded: dominance region. Star: the flight-relative boosted-tree arm of §5.3.](meta/fig/v2/fig5_ablation.png)
MD
pandoc "$tmp" -o docs/RESULTS.docx --resource-path=$ROOT && echo "wrote $ROOT/docs/RESULTS.docx"
rm -f "$tmp"
