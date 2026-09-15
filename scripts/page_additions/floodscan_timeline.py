"""Timeline per two-gauge flood season (both rivers together) with a FloodScan inundation: day 0 is
inundation (flood exposure across the 14 anticipatory-action districts over its own 1-in-5). Marks,
each the earliest on either river: the day a river's second SWALIM gauge went over its own 1-in-3
level; the first day a window's calibration record (reanalysis) met the action rule; the first
forecast issue that met it. Reads floodscan_lead.json (floodscan_season.py); writes
figs/floodscan_timeline.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = Path(__file__).parent
lead = [o for o in json.load(open(S / "floodscan_lead.json")) if o["inundation"]]
RP = json.load(open(S / "floodscan_rank.json"))["deyr"].get("rp", 5)
C_SFED, C_G, C_R, C_F = "#0E8A7B", "#111827", "#7C3AED", "#B34036"
XMAX = 30
MARK = (("gauge", "o", C_G, 8), ("reanalysis", "s", C_R, 7), ("forecast", "D", C_F, 7))

fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(lead) + 2.1))
for i, o in enumerate(lead):
    xs = []
    for key, m, col, ms in MARK:
        ld = o[key + "_lead"]
        if ld is None:
            ax.plot([-XMAX - 1], [i], "x", color=col, ms=7, mew=1.6, zorder=6)     # never: at the left edge
            continue
        x = max(min(ld, XMAX), -XMAX)                                              # lead > 0 = before inundation, drawn to the LEFT
        xs.append(-x)
        ax.plot([-x], [i], m, color=col, ms=ms, zorder=5, mec="white", mew=.6)
        if x != ld:
            ax.annotate(f"{ld:+d} d", (-x, i), xytext=(6, 6), textcoords="offset points", fontsize=8, color=col)
    if xs:
        ax.plot([min(xs + [0]), max(xs + [0])], [i, i], color="#e5e7eb", lw=3, zorder=1, solid_capstyle="round")
ax.axvline(0, color=C_SFED, lw=1.6)
ax.set_yticks(range(len(lead))); ax.set_yticklabels([f"{o['season'].title()} {o['year']}" for o in lead])
ax.set_ylim(len(lead) - 0.5, -1.9)                                                      # inverted, with headroom for the band labels
ax.set_xlim(-XMAX - 3, XMAX + 2)
ticks = [-30, -20, -10, 0, 10, 20, 30]
ax.set_xticks(ticks); ax.set_xticklabels([f"{-t} d before" if t < 0 else "inundation" if t == 0 else f"{t} d after" for t in ticks], fontsize=8.5)
ax.axvspan(-7.5, 0, color="#f0fdf4", zorder=0)
ax.text(-3.75, -1.2, "up to 7 d", color="#4d7c0f", fontsize=8.5, ha="center", va="center")
ax.text(-19, -1.2, "8 d or more before inundation", color="#166534", fontsize=8.5, ha="center", va="center")
ax.set_xlabel(f"days before (left) and after (right) FloodScan flood exposure across the 14 AA districts went over its own 1-in-{RP} level", fontsize=8.8, color="#374151")
ax.grid(axis="x", color="#f1f5f9"); ax.tick_params(length=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=C_G, ms=8, label="SWALIM gauges: second gauge over its own 1-in-3 level"),
                   Line2D([], [], marker="s", ls="none", color=C_R, ms=7, label="reanalysis: window's calibration record meets the action rule"),
                   Line2D([], [], marker="D", ls="none", color=C_F, ms=7, label="forecast: first issue meeting the action rule"),
                   Line2D([], [], marker="x", ls="none", color="#6b7280", mew=1.6, label="never (shown at the left edge)")],
          loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.0), fontsize=8.4)
fig.tight_layout()
out = S / "wt-trigger/pages/trigger-single-model/figs/floodscan_timeline.png"
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.12); plt.close(fig)
print("wrote", out, len(lead), "rows")
