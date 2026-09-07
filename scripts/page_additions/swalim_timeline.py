"""Timeline of who flagged first, SWALIM or the window's model, per river-season.
Dates are the dates the information was available: SWALIM bulletin issue dates
(first flag, then the ladder steps), the first day the window's model crossed on its
reanalysis (GloFAS v5 in Deyr, Google in Gu), the GloFAS v4 forecast's first issue
with enough points over, and the gauges' two-gauge 1-in-3 / 1-in-5 crossings."""
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


# season, river, SWALIM first flag, moderate, high, bank full | model reanalysis crossing | v4 issue | gauge 1-in-3, 1-in-5
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
MODEL = {"Deyr": "GloFAS v5", "Gu": "Google"}          # the window's model, on reanalysis
C_SW, C_SWD, C_G, C_V5, C_V4, C_GAUGE = "#d97706", "#92400e", "#1d4ed8", "#0f766e", "#6b7280", "#9ca3af"
C_MODEL = {"Deyr": C_V5, "Gu": C_G}


def doy(d):
    return date(2001, d.month, d.day).toordinal() - date(2001, 1, 1).toordinal()


def compare(sw, other, other_name):
    """Who was first, SWALIM's first bulletin against another date."""
    if sw is None and other is None:
        return f"no SWALIM bulletin; {other_name} never crossed"
    if sw is None:
        return f"no SWALIM bulletin; {other_name} {fmt(other)}"
    if other is None:
        return f"SWALIM only; {other_name} never crossed"
    n = (other - sw).days
    if n == 0:
        return f"same day as {other_name}"
    return f"SWALIM {n} d before {other_name}" if n > 0 else f"{other_name} {-n} d before SWALIM"


def verdicts(r):
    season, river, first, mod, high, bank, model_d, v4, g3, g5 = r
    m = MODEL[season.split()[0]]
    f, v = D(first), D(v4)
    if season == "Deyr 2020":
        return "no bulletin for the October rise", "(September flood bulletin only)"
    v1 = f"{m} record ends 2023" if model_d == "n/a" else compare(f, D(model_d), m)
    v2 = "no v4 issue crossed" if (f is None and v is None) else compare(f, v, "v4 forecast issue")
    return v1, v2


def panel(ax, rows, x0, x1, title):
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        season, river, first, mod, high, bank, model_d, v4, g3, g5 = r
        first, mod, high, bank, v4, g3, g5 = map(D, (first, mod, high, bank, v4, g3, g5))
        model_d = None if model_d in (None, "n/a") else D(model_d)
        cm = C_MODEL[season.split()[0]]
        if y % 2 == 0:
            ax.axhspan(y - .5, y + .5, color="#f8fafc", zorder=0)
        ax.axhline(y - .5, color="#eef1f4", lw=.8, zorder=0)
        # gauges: band from 1-in-3 to 1-in-5
        if g3 or g5:
            a = doy(g3) if g3 else doy(g5)
            b = doy(g5) if g5 else doy(g3)
            ax.plot([a, b], [y, y], color=C_GAUGE, lw=9, solid_capstyle="butt", alpha=.45, zorder=1)
            if g3:
                ax.plot([doy(g3)], [y], marker="|", color="#4b5563", ms=14, mew=1.6, zorder=2)
            if g5:
                ax.plot([doy(g5)], [y], marker="|", color="#111827", ms=14, mew=2.4, zorder=2)
        # upper track: SWALIM
        sw_pts = [p for p in (first, mod, high, bank) if p]
        if sw_pts:
            ax.plot([doy(min(sw_pts)), doy(max(sw_pts))], [y + .2, y + .2], color=C_SW, lw=1.2, zorder=3)
            ax.plot(doy(first), y + .2, marker="o", mfc="none", mec=C_SW, ms=13, mew=1.4, zorder=6)
            if mod:
                ax.plot(doy(mod), y + .2, marker="o", color=C_SW, ms=6, zorder=5, alpha=.65)
            if high:
                ax.plot(doy(high), y + .2, marker="o", color=C_SW, ms=8, zorder=5)
            if bank:
                ax.plot(doy(bank), y + .2, marker="s", color=C_SWD, ms=8, zorder=5)
        # lower track: the window's model and the v4 forecast issue
        if model_d:
            ax.plot(doy(model_d), y - .2, marker="D", color=cm, ms=7.5, mec="white", mew=.8, zorder=5)
        if v4:
            ax.plot(doy(v4), y - .2, marker="^", color=C_V4, ms=9, mec="white", mew=.8, zorder=5)
        v1, v2 = verdicts(r)
        ax.text(x1 + 1.5, y + .17, v1, va="center", fontsize=9.6, color="#111827")
        ax.text(x1 + 1.5, y - .2, v2, va="center", fontsize=8.8, color="#6b7280")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}  {r[1]}" for r in rows], fontsize=10)
    ax.set_ylim(-.5, len(rows) - .5)
    ax.set_xlim(x0, x1)
    months = [(9, "Sep"), (10, "Oct"), (11, "Nov"), (12, "Dec"), (4, "Apr"), (5, "May"), (6, "Jun")]
    ticks = [(doy(date(2001, m, 1)), f"1 {n}") for m, n in months if x0 <= doy(date(2001, m, 1)) <= x1]
    ticks += [(doy(date(2001, m, 15)), f"15 {n}") for m, n in months if x0 <= doy(date(2001, m, 15)) <= x1]
    ticks.sort()
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([n for _, n in ticks], fontsize=9.5)
    ax.grid(axis="x", color="#e5e7eb", lw=.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color="#111827", pad=8)
    ytop = ys[0]
    ax.text(x0 + .6, ytop + .2, "SWALIM bulletins", va="center", ha="left", fontsize=8, color=C_SW, style="italic")
    ax.text(x0 + .6, ytop - .2, "models", va="center", ha="left", fontsize=8, color="#475569", style="italic")


deyr = [r for r in ROWS if r[0].startswith("Deyr")]
gu = [r for r in ROWS if r[0].startswith("Gu")]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 10.4),
                             gridspec_kw={"hspace": .34, "height_ratios": [len(deyr), len(gu)]})
panel(a1, deyr, doy(date(2001, 9, 1)), doy(date(2001, 12, 10)), "Deyr seasons: the window's model is GloFAS v5")
panel(a2, gu, doy(date(2001, 4, 10)), doy(date(2001, 6, 5)), "Gu seasons: the window's model is Google")
fig.subplots_adjust(left=.155, right=.72, top=.86, bottom=.05)
handles = [
    Line2D([], [], marker="o", mfc="none", mec=C_SW, color="none", ms=11, mew=1.4, label="SWALIM: first bulletin flagging risk (ring)"),
    Line2D([], [], marker="o", color=C_SW, ls="none", ms=6, alpha=.65, label="SWALIM: moderate risk reported"),
    Line2D([], [], marker="o", color=C_SW, ls="none", ms=8, label="SWALIM: high risk reported"),
    Line2D([], [], marker="s", color=C_SWD, ls="none", ms=8, label="SWALIM: bank full or overflow reported"),
    Line2D([], [], marker="D", color=C_V5, ls="none", ms=7.5, label="GloFAS v5 reanalysis: first day the Deyr rule crossed"),
    Line2D([], [], marker="D", color=C_G, ls="none", ms=7.5, label="Google reanalysis: first day the Gu rule crossed"),
    Line2D([], [], marker="^", color=C_V4, ls="none", ms=9, label="GloFAS v4 forecast: first issue with enough points over"),
    Patch(color=C_GAUGE, alpha=.45, label="gauges: two over 1-in-3 (thin tick) to two over 1-in-5 (thick tick)"),
]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.155, .995), ncol=2, frameon=False,
           fontsize=9, handletextpad=.6, columnspacing=1.8)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)

out = []
for r in ROWS:
    v1, v2 = verdicts(r)
    out.append({"season": r[0], "river": r[1], "model": MODEL[r[0].split()[0]], "swalim_first": r[2], "trigger": r[6],
                "v4_issue": r[7], "vs_trigger": v1, "vs_v4": v2})
json.dump(out, open(S / "swalim_timeline.json", "w"), indent=1)
for o in out:
    print(f"{o['season']:10s}{o['river']:9s} | {o['vs_trigger']:40s} | {o['vs_v4']}")
