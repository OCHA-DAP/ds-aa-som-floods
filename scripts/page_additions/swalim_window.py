"""Lead time of each flag against the flood onset: days between the flag and the
gauges' first two-gauge 1-in-3 crossing (not the peak), for SWALIM's first bulletin,
the trigger's first day and the GloFAS v4 forecast's first issue. The action
window is 1 to 7 days before the onset, readiness 8 to 12.

Gauge dates (onset = second gauge over its own 1-in-3, and the 1-in-5 crossing) come
from somlib.gauge_crossings, the same two-gauge benchmark as every other page section
(levels fitted 2000-2023). SWALIM, trigger and v4 issue dates come from
swalim_timeline.json (swalim_timeline.py: SWALIM bulletin dates hand-read from the
archive, model and forecast dates computed from somlib).

    SOM_DATA_REPO=<checkout with data/processed> python scripts/page_additions/swalim_window.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

S = Path(__file__).parent
sys.path.insert(0, str(S))
import somlib as L  # noqa: E402

OUT = S.parents[1] / "pages" / "trigger-single-model" / "figs" / "k_swalim_window.png"


def D(s):
    return date.fromisoformat(s) if s else None


def gauge_date(season, river, rp):
    """Date the river's second gauge reached its own 1-in-rp level that season (somlib)."""
    name, year = season.split()
    d = L.gauge_crossings(river.lower(), name.lower(), rp, span=range(1999, 2025)).get(int(year))
    return d.date().isoformat() if d is not None else None


# season, river, note: the rows with a gauge event (Deyr 2019 Juba and Gu 2021 have none)
NOTES = [
 ("Deyr 2006", "Juba",     "the second gauge crossed 1-in-3 and 1-in-5 on the same day"),
 ("Deyr 2006", "Shabelle", None),
 ("Deyr 2014", "Juba",     None),
 ("Deyr 2014", "Shabelle", None),
 ("Deyr 2019", "Shabelle", None),
 ("Deyr 2020", "Shabelle", "SWALIM's bulletin was for the September flood"),
 ("Deyr 2023", "Juba",     None),
 ("Deyr 2023", "Shabelle", None),
 ("Gu 2016",   "Shabelle", None),
 ("Gu 2020",   "Juba",     None),
 ("Gu 2020",   "Shabelle", None),
 ("Gu 2023",   "Shabelle", None),
 ("Gu 2024",   "Juba",     None),
 ("Gu 2024",   "Shabelle", None),
]
TL = {(r["season"], r["river"]): r for r in json.loads((S / "swalim_timeline.json").read_text(encoding="utf-8"))}
# Deyr 2020 Shabelle: SWALIM's only bulletin (8 Sep) concerned the September flood, before the
# Deyr window, so it is not counted as a flag for the October rise
NO_SWALIM = {("Deyr 2020", "Shabelle")}
FLAGS = [(season, river, None if (season, river) in NO_SWALIM else TL[(season, river)]["swalim_first"],
          TL[(season, river)]["trigger"], TL[(season, river)]["v4_issue"], note)
         for season, river, note in NOTES]
ROWS = []
for season, river, sw, trig, v4, note in FLAGS:
    onset = gauge_date(season, river, 3) or gauge_date(season, river, 5)
    g5 = gauge_date(season, river, 5)
    ROWS.append((season, river, onset, g5, sw, trig, v4, note))
    print(f"{season:10s}{river:9s} onset {onset}  1-in-5 {g5}")
C_SW, C_TR, C_V4 = "#d97706", "#1d4ed8", "#0f766e"


def lead(onset, flag):
    """days before the onset; positive = before"""
    return (onset - flag).days


def band(n):
    if n is None:
        return None
    if 1 <= n <= 7:
        return "action window"
    if 8 <= n <= 12:
        return "readiness window"
    if n > 12:
        return "earlier than readiness"
    return "on or after onset"


def phrase(name, n, never):
    if n is None:
        return f"{name} {never}"
    if n == 0:
        return f"{name} on the day"
    if n > 0:
        return f"{name} {n} d before"
    return f"{name} {-n} d after"


fig, ax = plt.subplots(figsize=(12.5, 7.6))
ys = list(range(len(ROWS)))[::-1]
ax.axvspan(-7.5, -0.5, color="#dcfce7", alpha=.9, zorder=0)
ax.axvspan(-12.5, -7.5, color="#f0fdf4", alpha=.9, zorder=0)
ax.axvline(0, color="#374151", lw=1.2, zorder=1)
ax.text(-7.2, len(ROWS) - .35, "action window\n1 to 7 d before", ha="left", va="bottom", fontsize=8.6, color="#166534")
ax.text(-7.9, len(ROWS) - .35, "readiness\n8 to 12 d", ha="right", va="bottom", fontsize=8.6, color="#4d7c0f")
ax.text(0.3, len(ROWS) - .35, "gauges cross\n1-in-3", ha="left", va="bottom", fontsize=8.6, color="#374151")
out = []
XMIN, XMAX = -23, 23
for y, r in zip(ys, ROWS):
    season, river, onset, g5, sw, trig, v4, note = r
    onset, g5, sw, v4 = D(onset), D(g5), D(sw), D(v4)
    trig_na = trig == "n/a"
    trig = None if trig_na else D(trig)
    deyr = season.startswith("Deyr")
    ls, lt = (lead(onset, sw) if sw else None), (lead(onset, trig) if trig else None)
    lv = (lead(onset, v4) if (v4 and deyr) else None)    # v4 shown for Deyr only: it is not the Gu model
    if g5 and g5 != onset:
        xg = -lead(onset, g5)
        if xg > XMAX - .4:
            ax.annotate(f"1-in-5 {-lead(onset, g5)} d after", (XMAX - .4, y), xytext=(-2, 7), textcoords="offset points",
                        fontsize=7.3, color="#111827", ha="right")
            xg = XMAX - .4
        ax.plot([xg], [y], marker="|", color="#111827", ms=13, mew=2.2, zorder=2)
    for val, c, m, dy, ms in ((ls, C_SW, "o", .0, 8.5), (lt, C_TR, "D", .0, 7), (lv, C_V4, "^", .0, 8.5)):
        if val is not None:
            x = max(min(-val, XMAX - .4), XMIN + .4)
            ax.plot([x], [y + dy], marker=m, color=c, ms=ms, zorder=5, clip_on=False,
                    mec="white", mew=.8)
            if -val < XMIN or -val > XMAX:
                ax.annotate(f"{abs(val)} d {'before' if val > 0 else 'after'}", (x, y), xytext=(0, -13),
                            textcoords="offset points", fontsize=7.3, color=c, ha="center")
    parts = [phrase("SWALIM", ls, "no bulletin"),
             ((f"{'Google' if season.startswith('Gu') else 'GloFAS v5'} record ends 2023") if trig_na
              else phrase("Google" if season.startswith("Gu") else "GloFAS v5", lt, "never crossed")),
             *([phrase("v4 issue", lv, "never crossed")] if deyr else [])]
    ax.text(XMAX + .6, y, "  |  ".join(parts), va="center", fontsize=8.6, color="#111827")
    out.append({"season": season, "river": river, "onset": onset.isoformat(), "severe": g5.isoformat() if g5 else None,
                "swalim_lead": ls, "trigger_lead": lt, "v4_lead": lv,
                "swalim_band": band(ls), "trigger_band": None if trig_na else band(lt), "v4_band": band(lv), "note": note})
ax.set_yticks(ys)
ax.set_yticklabels([f"{r[0]}  {r[1]}" for r in ROWS], fontsize=9.5)
ax.set_ylim(-.7, len(ROWS) + .6)
ax.set_xlim(XMIN, XMAX)
ticks = list(range(-21, 22, 7)) + [0]
ax.set_xticks(sorted(set(ticks)))
ax.set_xticklabels([("onset" if t == 0 else f"{-t} d before" if t < 0 else f"{t} d after") for t in sorted(set(ticks))], fontsize=9)
ax.grid(axis="x", color="#e5e7eb", lw=.8)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#d1d5db")
ax.tick_params(length=0)
handles = [
    Line2D([], [], marker="o", color=C_SW, ls="none", ms=8.5, label="SWALIM: first bulletin flagging risk"),
    Line2D([], [], marker="D", color=C_TR, ls="none", ms=7, label="window's model on reanalysis, first day the rule crossed: GloFAS v5 (Deyr), Google (Gu)"),
    Line2D([], [], marker="^", color=C_V4, ls="none", ms=8.5, label="GloFAS v4 forecast, Deyr only (live stand-in for v5): first issue over"),
    Line2D([], [], marker="|", color="#111827", ls="none", ms=13, mew=2.2, label="gauges cross 1-in-5"),
]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.15, .995), ncol=2, frameon=False, fontsize=8.8)
fig.subplots_adjust(left=.15, right=.615, top=.87, bottom=.07)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
json.dump(out, open(S / "swalim_window.json", "w"), indent=1)
for o in out:
    print(f"{o['season']:10s}{o['river']:9s} SWALIM {str(o['swalim_lead']):5s} {str(o['swalim_band']):24s} "
          f"trigger {str(o['trigger_lead']):5s} {str(o['trigger_band']):24s} v4 {str(o['v4_lead']):5s} {o['v4_band']}")
