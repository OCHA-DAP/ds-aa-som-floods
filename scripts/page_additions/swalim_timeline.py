"""Timeline of who flagged first, SWALIM or the trigger, per river-season.
Dates are the dates the information was available: SWALIM bulletin issue dates
(first flag, then the ladder steps), the trigger's first day on its reanalysis,
the GloFAS v4 forecast's first issue with enough points over, and the gauges'
two-gauge 1-in-3 / 1-in-5 crossings as the reference band."""
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
OUT = PAGE_DIR / "figs/k_swalim_timeline.png"


def D(s):
    return date.fromisoformat(s) if s else None


def fmt(d):
    return d.strftime("%d %b").lstrip("0")


# season, river, SWALIM first flag, moderate, high, bank full | trigger | v4 issue | gauge 1-in-3, 1-in-5
ROWS = [
 ("Deyr 2006", "Juba",     "2006-10-31", None,         "2006-10-31", "2006-10-31", "2006-10-29", None,         None,         "2006-10-30"),
 ("Deyr 2006", "Shabelle", "2006-10-31", None,         "2006-10-31", "2006-10-31", "2006-11-02", None,         "2006-11-02", "2006-11-18"),
 ("Deyr 2014", "Juba",     "2014-10-15", "2014-10-15", "2014-10-21", "2014-10-28", None,         "2014-10-23", "2014-10-22", "2014-10-24"),
 ("Deyr 2014", "Shabelle", "2014-10-15", "2014-10-15", "2014-10-20", "2014-10-24", "2014-10-16", "2014-10-07", "2014-10-20", "2014-10-29"),
 ("Deyr 2019", "Juba",     "2019-10-22", None,         "2019-10-22", "2019-10-22", None,         None,         None,         None),
 ("Deyr 2019", "Shabelle", "2019-10-22", None,         "2019-10-22", "2019-10-22", "2019-10-11", "2019-10-02", "2019-10-14", "2019-10-24"),
 ("Deyr 2020", "Shabelle", "2020-09-08", None,         "2020-09-08", "2020-09-08", "2020-10-10", "2020-10-02", "2020-10-01", "2020-10-06"),
 ("Deyr 2023", "Juba",     "2023-10-20", "2023-10-20", "2023-10-23", "2023-11-13", "2023-10-29", "2023-10-21", "2023-10-25", "2023-10-25"),
 ("Deyr 2023", "Shabelle", "2023-10-21", "2023-10-29", "2023-11-02", "2023-11-13", "2023-11-09", None,         "2023-11-07", "2023-11-20"),
 ("Gu 2016",   "Shabelle", None,         None,         None,         None,         "2016-05-12", "2016-05-08", "2016-05-11", "2016-05-18"),
 ("Gu 2020",   "Juba",     "2020-04-27", None,         "2020-04-27", "2020-04-27", "2020-04-29", "2020-05-01", "2020-04-22", "2020-05-12"),
 ("Gu 2020",   "Shabelle", "2020-04-27", "2020-05-04", "2020-05-04", "2020-05-18", "2020-04-30", None,         "2020-05-06", "2020-05-12"),
 ("Gu 2021",   "Juba",     "2021-05-10", "2021-05-10", None,         None,         None,         None,         None,         None),
 ("Gu 2021",   "Shabelle", "2021-05-10", "2021-05-10", "2021-05-19", "2021-05-25", None,         None,         None,         None),
 ("Gu 2023",   "Shabelle", "2023-05-08", None,         "2023-05-08", None,         None,         None,         "2023-04-17", "2023-05-23"),
 ("Gu 2024",   "Juba",     "2024-05-09", "2024-05-09", "2024-05-09", None,         "n/a",        "2024-05-05", "2024-05-10", None),
 ("Gu 2024",   "Shabelle", "2024-04-19", "2024-05-01", "2024-04-19", "2024-05-22", "n/a",        None,         "2024-05-08", "2024-05-20"),
]

C_SW, C_SWD, C_TR, C_V4, C_G = "#d97706", "#92400e", "#1d4ed8", "#0f766e", "#9ca3af"


def doy(d):
    """day offset within a common non-leap year"""
    return date(2001, d.month, d.day).toordinal() - date(2001, 1, 1).toordinal()


def verdict(sw, other, other_name, other_never):
    """Who flagged first: SWALIM's first bulletin against another date."""
    if sw is None and other is None:
        return "neither flagged"
    if sw is None:
        return f"no SWALIM bulletin; {other_name} {fmt(other)}"
    if other is None:
        return f"SWALIM only; {other_never}"
    n = (other - sw).days
    if n == 0:
        return "same day"
    return f"SWALIM first by {n} d" if n > 0 else f"{other_name} first by {-n} d"


def verdicts(r):
    season, river, first, mod, high, bank, trig, v4, g3, g5 = r
    f, v = D(first), D(v4)
    if season == "Deyr 2020":
        # SWALIM's only bulletin concerned the September flood, before the Deyr window
        return ("no bulletin for the October rise", "(September flood bulletin only)")
    if trig == "n/a":
        v1 = "trigger record ends 2023"
    else:
        v1 = verdict(f, D(trig), "trigger", "the trigger never crossed")
    v2 = verdict(f, v, "v4 issue", "no v4 issue crossed")
    return v1, v2


def panel(ax, rows, x0, x1, title):
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        season, river, first, mod, high, bank, trig, v4, g3, g5 = r
        first, mod, high, bank, v4, g3, g5 = map(D, (first, mod, high, bank, v4, g3, g5))
        trig = None if trig in (None, "n/a") else D(trig)
        if g3 or g5:
            a = doy(g3) if g3 else doy(g5)
            b = doy(g5) if g5 else doy(g3)
            ax.plot([a, b], [y, y], color=C_G, lw=9, solid_capstyle="butt", alpha=.45, zorder=1)
            if g3:
                ax.plot([doy(g3)], [y], marker="|", color="#4b5563", ms=14, mew=1.6, zorder=2)
            if g5:
                ax.plot([doy(g5)], [y], marker="|", color="#111827", ms=14, mew=2.4, zorder=2)
        sw_pts = [p for p in (first, mod, high, bank) if p]
        if sw_pts:
            ax.plot([doy(min(sw_pts)), doy(max(sw_pts))], [y + .18, y + .18], color=C_SW, lw=1.2, zorder=3)
            ax.plot(doy(first), y + .18, marker="o", mfc="none", mec=C_SW, ms=12.5, mew=1.4, zorder=6)
            if mod:
                ax.plot(doy(mod), y + .18, marker="o", color=C_SW, ms=5.5, zorder=5, alpha=.65)
            if high:
                ax.plot(doy(high), y + .18, marker="o", color=C_SW, ms=7.5, zorder=5)
            if bank:
                ax.plot(doy(bank), y + .18, marker="s", color=C_SWD, ms=7.5, zorder=5)
        if trig:
            ax.plot(doy(trig), y - .18, marker="D", color=C_TR, ms=7, zorder=5)
        if v4:
            ax.plot(doy(v4), y - .18, marker="^", color=C_V4, ms=8, zorder=5)
        v1, v2 = verdicts(r)
        ax.text(x1 + 1.5, y + .16, v1, va="center", fontsize=9.2, color="#111827")
        ax.text(x1 + 1.5, y - .2, v2, va="center", fontsize=8.6, color="#6b7280")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}  {r[1]}" for r in rows], fontsize=9.5)
    ax.set_ylim(-.7, len(rows) - .3)
    ax.set_xlim(x0, x1)
    months = [(9, "Sep"), (10, "Oct"), (11, "Nov"), (12, "Dec"), (4, "Apr"), (5, "May"), (6, "Jun")]
    ticks = [(doy(date(2001, m, 1)), n) for m, n in months if x0 <= doy(date(2001, m, 1)) <= x1]
    ticks += [(doy(date(2001, m, 15)), "15") for m, _ in months if x0 <= doy(date(2001, m, 15)) <= x1]
    ticks.sort()
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([n for _, n in ticks], fontsize=9)
    ax.grid(axis="x", color="#e5e7eb", lw=.8)
    ax.grid(axis="y", color="#f3f4f6", lw=.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color="#111827", pad=8)


deyr = [r for r in ROWS if r[0].startswith("Deyr")]
gu = [r for r in ROWS if r[0].startswith("Gu")]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 9.6),
                             gridspec_kw={"hspace": .38, "height_ratios": [len(deyr), len(gu)]})
panel(a1, deyr, doy(date(2001, 9, 1)), doy(date(2001, 12, 10)), "Deyr seasons")
panel(a2, gu, doy(date(2001, 4, 10)), doy(date(2001, 6, 5)), "Gu seasons")
fig.subplots_adjust(left=.16, right=.75, top=.87, bottom=.05)
handles = [
    Line2D([], [], marker="o", mfc="none", mec=C_SW, color="none", ms=11, mew=1.4, label="SWALIM: first bulletin flagging risk (ring)"),
    Line2D([], [], marker="o", color=C_SW, ls="none", ms=5.5, alpha=.65, label="SWALIM: moderate risk"),
    Line2D([], [], marker="o", color=C_SW, ls="none", ms=7.5, label="SWALIM: high risk"),
    Line2D([], [], marker="s", color=C_SWD, ls="none", ms=7.5, label="SWALIM: bank full or overflow"),
    Line2D([], [], marker="D", color=C_TR, ls="none", ms=7, label="trigger: first day (Google for Gu, GloFAS v5 for Deyr)"),
    Line2D([], [], marker="^", color=C_V4, ls="none", ms=8, label="GloFAS v4 forecast: first issue with enough points over"),
    Patch(color=C_G, alpha=.45, label="gauges: two-gauge 1-in-3 (thin tick) to 1-in-5 (thick tick)"),
]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.16, .995), ncol=2, frameon=False,
           fontsize=8.8, handletextpad=.6, columnspacing=1.6)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)

out = []
for r in ROWS:
    v1, v2 = verdicts(r)
    out.append({"season": r[0], "river": r[1], "swalim_first": r[2], "trigger": r[6], "v4_issue": r[7],
                "vs_trigger": v1, "vs_v4": v2})
json.dump(out, open(S / "swalim_timeline.json", "w"), indent=1)
for o in out:
    print(f"{o['season']:10s}{o['river']:9s} | {o['vs_trigger']:34s} | {o['vs_v4']}")
