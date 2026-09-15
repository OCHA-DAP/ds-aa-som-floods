"""Timeline per flood season: first gauge over 1-in-3 (day 0), second gauge over 1-in-3 (the
benchmark date), and FloodScan inundation (river-buffer flooded fraction over its own 1-in-3).
Seasons both records call a flood. Reads floodscan_timing.json; writes
pages/trigger-single-model/figs/floodscan_timeline.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = Path(__file__).parent
res = json.load(open(S / "floodscan_timing.json"))
NAME = {"gu_juba": "Juba Gu", "deyr_juba": "Juba Deyr", "gu_shabelle": "Shabelle Gu", "deyr_shabelle": "Shabelle Deyr"}
ORDER = ["gu_juba", "deyr_juba", "gu_shabelle", "deyr_shabelle"]
C_G1, C_G2, C_SFED = "#9ca3af", "#111827", "#0E8A7B"
XMAX = 35

rows = []
for k in ORDER:
    r = res[k]
    for y in r["years_both"]:
        v1, v2 = r["vs_first"][str(y)], r["vs_second"][str(y)]
        rows.append((f"{NAME[k]} {y}", v1 - v2, v1))          # second gauge at (onset-first)-(onset-second); inundation at onset-first

fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(rows) + 1.5))
for i, (label, x2, xs) in enumerate(rows):
    xs_c = max(min(xs, XMAX), -XMAX)
    ax.plot([0, max(x2, xs_c, 0)], [i, i], color="#e5e7eb", lw=3, zorder=1, solid_capstyle="round")
    if xs_c < 0:
        ax.plot([xs_c, 0], [i, i], color="#e5e7eb", lw=3, zorder=1, solid_capstyle="round")
    ax.plot([0], [i], "o", color="white", mec=C_G1, mew=2, ms=8, zorder=3)
    ax.plot([x2], [i], "o", color=C_G2, ms=8, zorder=4)
    ax.plot([xs_c], [i], "s", color=C_SFED, ms=8, zorder=5)
    if xs != xs_c:
        ax.annotate(f"{xs:+d} d", (xs_c, i), xytext=(7, 7), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=C_SFED)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.axvline(0, color="#9ca3af", lw=1, ls=":")
ax.set_xlim(-XMAX - 2, XMAX + 2)
ax.set_xticks(range(-30, 31, 10)); ax.set_xticklabels([f"{t:+d} d" if t else "first gauge\nover 1-in-3" for t in range(-30, 31, 10)], fontsize=8.5)
ax.set_xlabel("days after the river's first gauge went over its own 1-in-3 level", fontsize=9, color="#374151")
ax.grid(axis="x", color="#f1f5f9"); ax.tick_params(length=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="none", color="white", mec=C_G1, mew=2, ms=8, label="first gauge over 1-in-3"),
                   Line2D([], [], marker="o", ls="none", color=C_G2, ms=8, label="second gauge over 1-in-3 (the benchmark date)"),
                   Line2D([], [], marker="s", ls="none", color=C_SFED, ms=8, label="inundation: FloodScan flooded fraction over its own 1-in-3")],
          loc="lower left", frameon=False, ncol=1, bbox_to_anchor=(0, 1.0), fontsize=8.6)
fig.tight_layout()
out = S / "wt-trigger/pages/trigger-single-model/figs/floodscan_timeline.png"
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.12); plt.close(fig)
print("wrote", out, len(rows), "rows")
