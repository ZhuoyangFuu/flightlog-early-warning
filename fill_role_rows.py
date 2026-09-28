#!/usr/bin/env python
"""Auto-fill the per-role read-out rows of RESULTS.md §5.6 from meta/paper/{summary,rules}.csv.

Idempotent: the generated block sits between <!-- role-rows --> markers (first run replaces
the '*re-running (OOM-killed 15–16 Sept)*' placeholder row). Also drops in a bullet with
the verdict and updates the v9 status sentences. Run after paper_tables.py.
  python fill_role_rows.py [--dry-run]
"""
import re, sys, datetime as dt
import numpy as np, pandas as pd

ROOT = "/mnt/wsssd/FlightLog"
RES = f"{ROOT}/RESULTS.md"
ENC = {"vrole": "original", "chrole": "A channel-mask", "rdrole": "B random-mask (control)"}
DRY = "--dry-run" in sys.argv

s = pd.read_csv(f"{ROOT}/meta/paper/summary.csv")
s = s[(s.n >= 3) & (s.band == "shift120")]
r = pd.read_csv(f"{ROOT}/meta/paper/rules.csv")
t0 = pd.read_csv(f"{ROOT}/meta/paper/tier0.csv").set_index("split")

def cell(tag, head, sp):
    d = s[(s.features == tag) & (s["head"] == head) & (s.split == sp)]
    if not len(d):
        return None
    d = d.iloc[0]
    rr = r[(r.features == tag) & (r["head"] == head) & (r.split == sp)]
    rules = f"{int(rr.n_dominating.iloc[0])} / {int(rr.n_configs.iloc[0])}" if len(rr) else "—"
    lead = f"{d.lead:.1f} s" if np.isfinite(d.lead) else "n/a"
    txt = f"{d.det:.3f} ± {d.det_sd:.3f} / {lead}"
    if bool(d.dominates):
        txt = f"**{txt}**"
    return txt, rules, d

rows, facts = [], []
for tag, enc in ENC.items():
    for head in ("lr", "gbt"):
        b, c = cell(tag, head, "B_fixed"), cell(tag, head, "C")
        if b is None and c is None:
            continue
        rows.append(f"| {enc} | role | {head} | {b[0] if b else '—'} | {b[1] if b else '—'} | "
                    f"{c[0] if c else '—'} | {c[1] if c else '—'} |")
        for sp, x in (("B_fixed", b), ("C", c)):
            if x is not None:
                facts.append((enc, head, sp, x[2]))
if not rows:
    print("no role arms at n>=3 yet; nothing to fill"); sys.exit(0)

# verdict vs the v8 arm (emb + lr) and the bar, both axes
ref = {sp: s[(s.features == "emb") & (s["head"] == "lr") & (s.split == sp)].iloc[0] for sp in ("B_fixed", "C")}
beats_ref, dominates = [], []
for enc, head, sp, d in facts:
    e = ref[sp]
    if np.isfinite(d.lead) and d.det > e.det and d.lead > e.lead:
        beats_ref.append(f"{enc} / {head} on {sp} ({d.det:.3f} / {d.lead:.0f} s vs {e.det:.3f} / {e.lead:.0f} s)")
    if bool(d.dominates):
        dominates.append(f"{enc} / {head} on {sp} ({d.det:.3f} / {d.lead:.0f} s)")
n_cells = len(facts)
today = dt.date.today().isoformat()
best = max(facts, key=lambda f: f[3].det)
bullet = (f"- **Per-role read-out (auto-filled {today}, {n_cells} cells at n = 3).** "
          + (f"Cells beating the v8 `emb` + lr arm on both axes: {'; '.join(beats_ref)}. "
             if beats_ref else "No per-role cell beats the v8 `emb` + lr arm on both axes. ")
          + (f"Cells dominating the Tier-0 cross bar: {'; '.join(dominates)}. "
             if dominates else "No per-role cell dominates the Tier-0 cross bar. ")
          + f"Highest per-role detection: {best[0]} / {best[1]} on {best[2]} at {best[3].det:.3f} ± "
            f"{best[3].det_sd:.3f} / {best[3].lead:.0f} s. Rule counts are in the table; Fig. 5 "
            f"shows the diamonds. Read the numbers, not this sentence, before quoting.")

txt = open(RES).read()
block = "<!-- role-rows -->\n" + "\n".join(rows) + "\n<!-- /role-rows -->"
if "<!-- role-rows -->" in txt:
    txt = re.sub(r"<!-- role-rows -->.*?<!-- /role-rows -->", block, txt, flags=re.S)
else:
    ph = "| original | role | lr, gbt | *re-running (OOM-killed 15–16 Sept)* | | | |"
    assert ph in txt, "placeholder row not found"
    txt = txt.replace(ph, block)
bb = "<!-- role-bullet -->"
if bb in txt:
    txt = re.sub(r"<!-- role-bullet -->.*?<!-- /role-bullet -->", f"{bb}\n{bullet}\n<!-- /role-bullet -->", txt, flags=re.S)
else:
    anchor = "- **The v8 arms stand.**"
    assert anchor in txt
    txt = txt.replace(anchor, f"{bb}\n{bullet}\n<!-- /role-bullet -->\n{anchor}")
txt = txt.replace("Whether the channel-resolved `role` view does better is the open cell.",
                  "The channel-resolved `role` view is reported in the auto-filled bullet below.")
txt = txt.replace("""per-role read-out (4096-d) was OOM-killed in every one of its 18 runs and is being re-run
now with a memory-safe loader (`bash run_role_probes.sh status`); it is the one open cell.""",
                  f"""per-role read-out (4096-d) was OOM-killed in every one of its 18 runs on 15–16 September;
it was re-run with a memory-safe loader and its rows landed on {today} (auto-filled, §5.6).""")
txt = txt.replace("""**Status.** Every planned experiment is complete except the per-role read-out probes
(§5.6, re-running on CPU, ~1 day). Nothing below depends on them; if they land before the
full-paper deadline they add a row to the §5.6 table and a marker to Fig. 5.""",
                  f"""**Status.** Every planned experiment is complete; the per-role read-out probes landed on
{today} ({n_cells} of 12 cells at n = 3; see §5.6 and the diamonds in Fig. 5).""")
txt = txt.replace("- The 4096-d per-role read-out at the old loader: all 18 probes OOM-killed (the loader\n  peaked at ~4× the 112 GiB matrix); re-running with a preallocating float16 loader that is\n  verified bit-identical on the cached views.",
                  "- The 4096-d per-role read-out at the old loader: all 18 probes OOM-killed (the loader\n  peaked at ~4× the 112 GiB matrix); re-run with a preallocating float16 loader verified\n  bit-identical on the cached views — results in §5.6.")
txt = txt.replace("with the flight-relative boosted-tree arm as the reference star. Per-role markers appear\n  when the re-run lands.",
                  "with the flight-relative boosted-tree arm as the reference star; diamonds = per-role read-out.")
print("\n".join(rows)); print(bullet)
if not DRY:
    open(RES, "w").write(txt); print("RESULTS.md updated")
