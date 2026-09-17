"""Timing-of-activations charts, both seasons.

One figure per river-season: model discharge stacked by station (GloFAS version 4 in Deyr,
Google Flood Hub in Gu, matching each window's activation source), SWALIM river levels with
each gauge's own 1-in-3, flood exposure across the 14 anticipatory-action districts, and the
activation marks. Deyr is always presented before Gu.
"""
import sys
import json

sys.path.insert(0, "C:/Users/pauni/Desktop/Work/OCHA/GitHub/ds-aa-som-floods")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.utils import weibull_threshold

D = "C:/Users/pauni/Desktop/Work/OCHA/GitHub/ds-aa-som-floods/data/processed/"
S = "C:/Users/pauni/AppData/Local/Temp/claude/C--Users-pauni-Desktop-Work-OCHA-GitHub/f11e6aed-1a18-4740-beb3-3ee68ff96279/scratchpad/"

ST = {"Juba": ["dollow", "luuq", "bardheere", "bualle"],
      "Shabelle": ["belet_weyne", "bulo_burti", "jowhar"]}
# activation rule per (river, season): source, return period, stations required
RULES = {
    ("Juba", "deyr"): ("glofas_v4", 4, 3), ("Shabelle", "deyr"): ("glofas_v4", 4, 2),
    ("Juba", "gu"): ("google_grrr", 5, 3), ("Shabelle", "gu"): ("google_grrr", 6, 2),
}
MONTHS = {"deyr": [10, 11, 12], "gu": [3, 4, 5]}
WINDOW = {"deyr": ("09-20", "12-05"), "gu": ("02-15", "06-25")}
SOURCE_TITLE = {"glofas_v4": "GloFAS version 4", "google_grrr": "Google Flood Hub"}
SEASON_TITLE = {"deyr": "Deyr", "gu": "Gu"}
NDIST = 3

T = {"dollow": "Dollow", "luuq": "Luuq", "bardheere": "Bardheere", "bualle": "Bualle",
     "belet_weyne": "Belet Weyne", "bulo_burti": "Bulo Burti", "jowhar": "Jowhar"}
C = {"dollow": "#BBD4E4", "luuq": "#8DB8D4", "bardheere": "#5E9BC4", "bualle": "#2E7CA8",
     "belet_weyne": "#BBD4E4", "bulo_burti": "#8DB8D4", "jowhar": "#2E7CA8"}
LN = {"dollow": "#2F6B8F", "luuq": "#B3603A", "bardheere": "#4B7F5C", "bualle": "#8A6BA8",
      "belet_weyne": "#2F6B8F", "bulo_burti": "#B3603A", "jowhar": "#4B7F5C"}
INK, FAINT, RED, SAND, SANDL = "#1F2324", "#98A2A4", "#B34036", "#A8791F", "#D9A441"

_rows = json.load(open(S + "wt-monitoring/src/monitoring/thresholds.json"))["rows"]


def thresholds(source, season, rp):
    return {r["station"]: r["threshold"] for r in _rows
            if r["basis"] == "reanalysis" and r["source"] == source
            and r["season"] == season and r["rp"] == rp}


DAILY = {}
for src in ("glofas_v4", "google_grrr"):
    d = pd.read_parquet(D + f"discharge_daily_{src}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    DAILY[src] = d

FORE = {}
for src, fn in (("glofas_v4", "reforecast_glofas_v4"), ("google_grrr", "reforecast_google_grrr")):
    r = pd.read_parquet(D + fn + ".parquet")
    for c in ("issued_time", "valid_time"):
        r[c] = pd.to_datetime(r[c])
    r = r[r.leadtime_days.between(1, 7)]
    FORE[src] = r.groupby(["station", "issued_time", "valid_time", "leadtime_days"])["discharge"].median().reset_index()

lv = pd.read_parquet(D + "swalim_levels.parquet")
lv["date"] = pd.to_datetime(lv["date"])
lv = lv.dropna(subset=["level_m"])

G3 = {}
for season, months in MONTHS.items():
    h = lv[lv.date.dt.month.isin(months) & (lv.date.dt.year >= 2000)]
    G3[season] = {}
    for st, g in h.groupby("station"):
        yrs = [y for y in sorted(set(g.date.dt.year)) if (g.date.dt.year == y).sum() >= 45]
        if len(yrs) >= 6:
            sub = g[g.date.dt.year.isin(yrs)]
            G3[season][st] = weibull_threshold(sub.groupby(sub.date.dt.year)["level_m"].max().values, 3)

ex = pd.read_parquet(S + "floodscan_exposure_14.parquet")
ex["valid_date"] = pd.to_datetime(ex["valid_date"])
TOTAL = ex.groupby("valid_date")["sum"].sum()
DL = {}
for season, months in MONTHS.items():
    e = ex[ex.valid_date.dt.month.isin(months)]
    DL[season] = {}
    for p, g in e.groupby("pcode"):
        s = g.set_index("valid_date")["sum"]
        DL[season][p] = weibull_threshold(s.groupby(s.index.year).max().values, 5)


def rule_day(df, sts, n, thr, dc, vc="discharge"):
    s = df[df.station.isin(sts)].copy()
    s["over"] = s[vc] >= s.station.map(thr)
    v = s.groupby(dc)["over"].sum()
    h = v[v >= n]
    return h.index.min() if len(h) else None


def flood_date(season, y):
    e = ex[ex.valid_date.dt.month.isin(MONTHS[season]) & (ex.valid_date.dt.year == y)]
    ds = []
    for p, L in DL[season].items():
        s = e[e.pcode == p].set_index("valid_date")["sum"]
        c = s[s >= L]
        if len(c):
            ds.append(c.index.min())
    ds.sort()
    return (ds[NDIST - 1] if len(ds) >= NDIST else None), len(ds)


def benchmark(season, y):
    """Per river, the second gauge to cross its own 1-in-3."""
    out = {}
    h = lv[lv.date.dt.month.isin(MONTHS[season]) & (lv.date.dt.year == y)]
    for river, sts in ST.items():
        cs = []
        for st in sts:
            if st in G3[season]:
                c = h[(h.station == st) & (h.level_m >= G3[season][st])]
                if len(c):
                    cs.append(c.date.min())
        if len(cs) >= 2:
            out[river] = sorted(cs)[1]
    return out


def activations(season, y):
    out = {}
    for river, sts in ST.items():
        src, rp, n = RULES[(river, season)]
        thr = thresholds(src, season, rp)
        d = DAILY[src]
        ra = rule_day(d[(d.date.dt.year == y) & d.date.dt.month.isin(MONTHS[season])], sts, n, thr, "date")
        fo = FORE[src]
        fo = fo[(fo.issued_time.dt.year == y) & fo.valid_time.dt.month.isin(MONTHS[season])]
        fc = None
        for iss, g in fo.groupby("issued_time"):
            if rule_day(g, sts, n, thr, "valid_time") is not None:
                fc = iss
                break
        out[river] = (fc, ra, len(fo) > 0)
    return out


def chart(season, y):
    a, b = WINDOW[season]
    A, B = f"{y}-{a}", f"{y}-{b}"
    acts = activations(season, y)
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.6), dpi=170, sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 3, 1.5]))
    for ax, (river, sts) in zip(axes[:2], ST.items()):
        src, rp, n = RULES[(river, season)]
        d = DAILY[src]
        piv = d[(d.date >= A) & (d.date <= B) & d.station.isin(sts)].pivot_table(
            index="date", columns="station", values="discharge").reindex(columns=sts)
        ax.stackplot(piv.index, [piv[s].values for s in sts], colors=[C[s] for s in sts], edgecolor="none")
        tops = piv.cumsum(axis=1)
        for s in sts:
            ax.text(piv.index[-1] + pd.Timedelta(days=2), tops[s].iloc[-1] - piv[s].iloc[-1] / 2,
                    T[s], color=C[s], fontsize=9.5, va="center", fontweight="bold")
        ax2 = ax.twinx()
        span = (piv.index[-1] - piv.index[0]).days
        w_all = lv[(lv.date >= A) & (lv.date <= B)]
        for i, s in enumerate(sts):
            w = w_all[w_all.station == s].sort_values("date")
            if len(w):
                ax2.plot(w.date, w.level_m, color=LN[s], lw=1.5)
            if s in G3[season]:
                ax2.axhline(G3[season][s], color=LN[s], lw=0.8, ls=(0, (4, 3)), alpha=0.4)
                ax2.text(piv.index[0] + pd.Timedelta(days=int(span * (0.02 + 0.17 * i))), G3[season][s],
                         f"{T[s]} 1-in-3", color=LN[s], fontsize=7.5, alpha=0.85, va="bottom", ha="left",
                         bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.85))
        fc, ra, has_arch = acts[river]
        top = ax.get_ylim()[1]
        marks = {}
        for dte, lab in ((fc, "forecast"), (ra, "reanalysis")):
            if dte is not None:
                marks.setdefault(dte, []).append(lab)
        for dte, labs in marks.items():
            ax.axvline(dte, color=RED, lw=1.3, alpha=0.85)
            ax.annotate(" + ".join(labs) + f" activation\n{dte:%d %b}", xy=(dte, top * 0.99), xytext=(6, 0),
                        textcoords="offset points", color=RED, fontsize=8.5, va="top", ha="left")
        ax.set_ylabel("discharge  m3/s", color=FAINT, fontsize=9.5)
        ax2.set_ylabel("level  m", color=FAINT, fontsize=9.5)
        ax.text(0, 1.07, f"{river}   ·   {SOURCE_TITLE[src]}, {n} of {len(sts)} over 1-in-{rp}",
                transform=ax.transAxes, fontsize=12.5, color=INK)
        ax.grid(axis="y", color="#F2F5F5", lw=1)
        ax.set_axisbelow(True)
        for aa in (ax, ax2):
            for sp in ("top", "right", "left"):
                aa.spines[sp].set_visible(False)
            aa.tick_params(colors=FAINT, labelsize=9, length=0)
        ax.spines["bottom"].set_color("#E4E9E9")
        ax.set_xlim(piv.index[0], piv.index[-1] + pd.Timedelta(days=int(span * 0.09)))
        ax.set_ylim(0, top)
    axe = axes[2]
    e = TOTAL[(TOTAL.index >= A) & (TOTAL.index <= B)]
    axe.fill_between(e.index, e.values, color=SANDL, alpha=0.35, lw=0)
    axe.plot(e.index, e.values, color=SAND, lw=1.2)
    fd, nd = flood_date(season, y)
    first = [d for d, _, _ in acts.values() if d is not None]
    if first:
        d0 = min(first)
        axe.axvline(d0, color=RED, lw=1.3, alpha=0.85)
        axe.annotate(f"activation\n{d0:%d %b}", xy=(d0, axe.get_ylim()[1] * 0.98), xytext=(6, 0),
                     textcoords="offset points", color=RED, fontsize=8.5, va="top", ha="left")
    axe.set_ylabel("people exposed", color=FAINT, fontsize=9.5)
    axe.text(0, 1.13, "Flood exposure, 14 anticipatory-action districts", transform=axe.transAxes,
             fontsize=11, color=INK)
    axe.grid(axis="y", color="#F2F5F5", lw=1)
    axe.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axe.spines[sp].set_visible(False)
    axe.spines["bottom"].set_color("#E4E9E9")
    axe.tick_params(colors=FAINT, labelsize=9, length=0)
    axe.set_ylim(0, None)
    axe.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    axe.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    axe.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.text(0.008, 0.99, f"{SEASON_TITLE[season]} {y}", fontsize=15, color=INK, va="top")
    fig.text(0.008, 0.963,
             "Shaded: model discharge, stacked by station.   Lines: SWALIM river level, dashed = that gauge's 1-in-3.   Red: activation.",
             fontsize=9.5, color=FAINT, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = S + f"{season}_{y}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, acts, (fd, nd), benchmark(season, y)


def seasons_of_interest(season):
    """Years that activated, met the two-gauge benchmark, or flooded three districts."""
    src_years = set()
    for river in ST:
        src, _, _ = RULES[(river, season)]
        d = DAILY[src]
        src_years |= set(d[d.date.dt.month.isin(MONTHS[season])].date.dt.year.unique())
    out = []
    for y in sorted(y for y in src_years if y >= 1999):
        acts = activations(season, y)
        fired = any(fc is not None for fc, _, _ in acts.values())
        bench = bool(benchmark(season, y))
        fd, _ = flood_date(season, y)
        if fired or bench or fd is not None:
            out.append(y)
    return out
