"""Lead time of each flag against the flood onset: days between the flag and the
gauges' first two-gauge 1-in-3 crossing (not the peak), for SWALIM's first bulletin,
the trigger's first day and the GloFAS v4 forecast's first issue. The action
window is 1 to 7 days before the onset, readiness 8 to 12."""
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

S = Path(__file__).parent
PAGE_DIR = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model"
OUT = PAGE_DIR / "figs/k_swalim_window.png"


def D(s):
    return date.fromisoformat(s) if s else None


# season, river, onset (two-gauge 1-in-3), 1-in-5, SWALIM first bulletin, trigger, v4 issue, note
ROWS = [
 ("Deyr 2006", "Juba",     "2006-10-30", "2006-10-30", "2006-10-31", "2006-10-29", None,         "onset taken as the 1-in-5 date; 1-in-3 date not in the record"),
 ("Deyr 2006", "Shabelle", "2006-11-02", "2006-11-18", "2006-10-31", "2006-11-02", None,         None),
 ("Deyr 2014", "Juba",     "2014-10-22", "2014-10-24", "2014-10-15", None,         "2014-10-23", None),
 ("Deyr 2014", "Shabelle", "2014-10-20", "2014-10-29", "2014-10-15", "2014-10-16", "2014-10-07", None),
 ("Deyr 2019", "Shabelle", "2019-10-14", "2019-10-24", "2019-10-22", "2019-10-11", "2019-10-02", None),
 ("Deyr 2020", "Shabelle", "2020-10-01", "2020-10-06", None,         "2020-10-10", "2020-10-02", "SWALIM's bulletin was for the September flood"),
 ("Deyr 2023", "Juba",     "2023-10-25", "2023-10-25", "2023-10-20", "2023-10-29", "2023-10-21", None),
 ("Deyr 2023", "Shabelle", "2023-11-07", "2023-11-20", "2023-10-21", "2023-11-09", None,         None),
 ("Gu 2016",   "Shabelle", "2016-05-11", "2016-05-18", None,         "2016-05-12", "2016-05-08", None),
 ("Gu 2020",   "Juba",     "2020-04-22", "2020-05-12", "2020-04-27", "2020-04-29", "2020-05-01", None),
 ("Gu 2020",   "Shabelle", "2020-05-06", "2020-05-12", "2020-04-27", "2020-04-30", None,         None),
 ("Gu 2023",   "Shabelle", "2023-04-17", "2023-05-23", "2023-05-08", None,         None,         None),
 ("Gu 2024",   "Juba",     "2024-05-10", None,         "2024-05-09", "n/a",        "2024-05-05", None),
 ("Gu 2024",   "Shabelle", "2024-05-08", "2024-05-20", "2024-04-19", "n/a",        None,         None),
]
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
    ls, lt, lv = (lead(onset, sw) if sw else None), (lead(onset, trig) if trig else None), (lead(onset, v4) if v4 else None)
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
             ("trigger record ends 2023" if trig_na else phrase("trigger", lt, "never crossed")),
             phrase("v4 issue", lv, "never crossed")]
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
    Line2D([], [], marker="D", color=C_TR, ls="none", ms=7, label="trigger: first day (Google for Gu, GloFAS v5 for Deyr)"),
    Line2D([], [], marker="^", color=C_V4, ls="none", ms=8.5, label="GloFAS v4 forecast: first issue with enough points over"),
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
