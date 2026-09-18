"""Timing-of-activations charts, both seasons.

One figure per season. Upper panels, one per river: the window's own activation source as a
share of each station's own return-period level (filled), and the SWALIM gauges as a share of
their own 1-in-3 level (lines), so a single 100% line serves both. Lower panel: flood exposure
across the 14 anticipatory-action districts. Activation dates are marked.

Data comes from DATA_REPO/data/processed (see somlib: set SOM_DATA_REPO when running from a
worktree). Restore it with scripts/restore_from_blob.py, which now includes
floodscan_exposure_14.parquet.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import somlib as L  # noqa: E402
from src.constants import TRIGGER_CONFIG, TRIGGER_STATIONS  # noqa: E402

FIGS = Path(__file__).resolve().parents[2] / "pages" / "activation-timing" / "figs"

RIVERS = ["juba", "shabelle"]
MONTHS = {"deyr": [10, 11, 12], "gu": [3, 4, 5]}
WINDOW = {"deyr": ("09-20", "12-31"), "gu": ("02-15", "06-25")}
SEASON_TITLE = {"deyr": "Deyr", "gu": "Gu"}
SOURCE_TITLE = {"glofas_v4": "GloFAS version 4", "google_grrr": "Google Flood Hub"}
GAUGE_RP = 3          # the two-gauge benchmark level
NDIST = 3             # districts that must reach their own 1-in-5 exposure level
DIST_RP = 5

# one colour per station: the filled area (model) and the line (gauge) share it
C = {"dollow": "#2F6B8F", "luuq": "#B3603A", "bardheere": "#4B7F5C", "bualle": "#8A6BA8",
     "belet_weyne": "#2F6B8F", "bulo_burti": "#B3603A", "jowhar": "#4B7F5C"}
LN = C
INK, FAINT, RED, SAND, SANDL, VOTE = "#1F2324", "#98A2A4", "#B34036", "#A8791F", "#D9A441", "#B34036"


# The Deyr legs were calibrated on GloFAS v5 (TRIGGER_CONFIG) but the operational system has
# been v4 since July 2023, and the reforecast archive is v4. This page reports what the live
# system would have done, so the Deyr source is read as v4 and the page says so.
# See pages/glofas-version/ for what the version change does to the thresholds.
OPERATIONAL = {"glofas_v5": "glofas_v4"}


def rule(river, season):
    """Activation source, return period and stations required, as the live system runs them."""
    cfg = TRIGGER_CONFIG[(river, season)]
    return OPERATIONAL.get(cfg["source"], cfg["source"]), cfg["rp"], cfg["n_req"]


def exposure():
    return pd.read_parquet(L.P / "floodscan_exposure_14.parquet").assign(
        valid_date=lambda d: pd.to_datetime(d["valid_date"]))


def district_levels(season):
    e = exposure()
    e = e[e.valid_date.dt.month.isin(MONTHS[season])]
    out = {}
    for p, g in e.groupby("pcode"):
        s = g.set_index("valid_date")["sum"]
        out[p] = float(L.weibull_threshold(s.groupby(s.index.year).max().values, DIST_RP))
    return out


def flood_date(season, y):
    """Day the NDIST-th district reaches its own 1-in-5 seasonal exposure level."""
    e = exposure()
    e = e[e.valid_date.dt.month.isin(MONTHS[season]) & (e.valid_date.dt.year == y)]
    lev = district_levels(season)
    ds = []
    for p, Lv in lev.items():
        s = e[e.pcode == p].set_index("valid_date")["sum"]
        c = s[s >= Lv]
        if len(c):
            ds.append(c.index.min())
    ds.sort()
    return (ds[NDIST - 1] if len(ds) >= NDIST else None), len(ds)


def benchmark(season, y):
    """Per river, the day the second gauge reaches its own 1-in-3 level (somlib's fit)."""
    out = {}
    lv = L.levels()
    for river in RIVERS:
        lev = L.gauge_levels(river, season, GAUGE_RP)
        cs = []
        for st, level in lev.items():
            if np.isnan(level):
                continue
            s = lv[(lv.station == st) & (lv.date.dt.year == y) & lv.date.dt.month.isin(MONTHS[season])]
            c = s[s.level_m >= level]
            if len(c):
                cs.append(c.date.min())
        if len(cs) >= 2:
            out[river] = sorted(cs)[1]
    return out


def _rule_day(df, sts, n, thr, dc, vc="discharge"):
    s = df[df.station.isin(sts)].copy()
    s["over"] = s[vc] >= s.station.map(thr)
    v = s.groupby(dc)["over"].sum()
    h = v[v >= n]
    return h.index.min() if len(h) else None


def activations(season, y):
    """Per river: (first forecast issue meeting the rule, first reanalysis day, archive covers y)."""
    out = {}
    for river in RIVERS:
        src, rp, n = rule(river, season)
        sts = TRIGGER_STATIONS[river]
        thr = L.model_thresholds(src, river, season, rp).to_dict()
        d = L.daily(src)
        d = d[(d.date.dt.year == y) & d.date.dt.month.isin(MONTHS[season])]
        ra = _rule_day(d, sts, n, thr, "date")
        rf = L.reforecast(src)
        rf = rf[(rf.leadtime_days.between(1, 7)) & (rf.issued_time.dt.year == y)
                & rf.valid_time.dt.month.isin(MONTHS[season])]
        med = rf.groupby(["station", "issued_time", "valid_time"])["discharge"].median().reset_index()
        fc = None
        for iss, g in med.groupby("issued_time"):
            if _rule_day(g, sts, n, thr, "valid_time") is not None:
                fc = iss
                break
        out[river] = (fc, ra, len(med) > 0)
    return out


def chart(season, y):
    a, b = WINDOW[season]
    A, B = pd.Timestamp(f"{y}-{a}"), pd.Timestamp(f"{y}-{b}")
    acts = activations(season, y)
    lv = L.levels()
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.6), dpi=170, sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 3, 1.5]))
    for ax, river in zip(axes[:2], RIVERS):
        src, rp, n = rule(river, season)
        sts = TRIGGER_STATIONS[river]
        thr = L.model_thresholds(src, river, season, rp).to_dict()
        d = L.daily(src)
        d = d[(d.date >= A) & (d.date <= B) & d.station.isin(sts)]
        ends = []
        for st in sts:
            s = d[d.station == st].sort_values("date")
            if s.empty or not thr.get(st):
                continue
            pct = 100 * s.discharge / thr[st]
            ax.fill_between(s.date, pct, color=C[st], alpha=0.22, lw=0)
            ax.plot(s.date, pct, color=C[st], lw=1.3)
            ends.append([pct.iloc[-1], st])
        glev = L.gauge_levels(river, season, GAUGE_RP)
        for st in sts:
            level = glev.get(st, np.nan)
            if np.isnan(level):
                continue
            w = lv[(lv.station == st) & (lv.date >= A) & (lv.date <= B)].sort_values("date")
            if len(w):
                ax.plot(w.date, 100 * w.level_m / level, color=C[st], lw=1.4, ls=(0, (5, 2)))
        ax.axhline(100, color=VOTE, ls="--", lw=1.1, zorder=3)
        fc, ra, _ = acts[river]
        top = max(ax.get_ylim()[1], 130)
        marks = {}
        for dte, lab in ((fc, "forecast"), (ra, "reanalysis")):
            if dte is not None:
                marks.setdefault(dte, []).append(lab)
        for dte, labs in marks.items():
            ax.axvline(dte, color=RED, lw=1.3, alpha=0.85)
            ax.annotate(" + ".join(labs) + f" activation\n{dte:%d %b}", xy=(dte, top * 0.99),
                        xytext=(6, 0), textcoords="offset points", color=RED, fontsize=8.5,
                        va="top", ha="left")
        # station names at the right edge, pushed apart so they never overlap
        gap = top * 0.075
        ends.sort()
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < gap:
                ends[i][0] = ends[i - 1][0] + gap
        for yv, st in ends:
            ax.text(B + pd.Timedelta(days=2), min(yv, top * 0.98), L.NAME[st], color=C[st],
                    fontsize=9, va="center", fontweight="bold")
        ax.set_ylabel("% of its own level", color=FAINT, fontsize=9.5)
        ax.text(0, 1.07, f"{river.title()}   ·   {SOURCE_TITLE[src]}, {n} of {len(sts)} over 1-in-{rp}",
                transform=ax.transAxes, fontsize=12.5, color=INK)
        ax.grid(axis="y", color="#F2F5F5", lw=1)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=FAINT, labelsize=9, length=0)
        ax.spines["bottom"].set_color("#E4E9E9")
        ax.set_xlim(A, B + pd.Timedelta(days=int((B - A).days * 0.09)))
        ax.set_ylim(0, top)
    axe = axes[2]
    e = exposure().set_index("valid_date")["sum"].groupby(level=0).sum()
    e = e[(e.index >= A) & (e.index <= B)]
    axe.fill_between(e.index, e.values, color=SANDL, alpha=0.35, lw=0)
    axe.plot(e.index, e.values, color=SAND, lw=1.2)
    first = [f for f, _, _ in acts.values() if f is not None]
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
             "Solid: model discharge as a share of each station's own return-period level.   "
             "Dashed, same colour: that station's SWALIM gauge as a share of its own 1-in-3.   "
             "Red dashes: 100%.",
             fontsize=9.5, color=FAINT, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"{season}_{y}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, acts, flood_date(season, y), benchmark(season, y)


def seasons_of_interest(season):
    """Years that activate, meet the two-gauge benchmark, or reach the three-district level."""
    years = set()
    for river in RIVERS:
        src, _, _ = rule(river, season)
        d = L.daily(src)
        years |= set(d[d.date.dt.month.isin(MONTHS[season])].date.dt.year.unique())
    out = []
    for y in sorted(y for y in years if y >= L.Y0):
        acts = activations(season, y)
        fd, _ = flood_date(season, y)
        if any(f is not None for f, _, _ in acts.values()) or benchmark(season, y) or fd is not None:
            out.append(int(y))
    return out
