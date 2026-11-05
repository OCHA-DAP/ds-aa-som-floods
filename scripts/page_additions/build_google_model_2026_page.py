"""Build pages/google-model-2026/: Google's November 2026 river model read with the trigger's
own rules at the seven points, next to the sources the trigger runs on, on two records side by
side: each model's reanalysis and each model's forecast archive. Scored against the SWALIM gauges.
The page states what each record shows; it draws no conclusion.

Inputs
- the seven-station extracts of the new model on blob, ds-aa-som-floods/google_grrr_2026_11/
  (dev "projects" container), or a local folder with the same file names via SOM_GRRR_2026_DIR;
- the repo's processed data (SOM_DATA_REPO when run from a worktree without data/processed).
Charts follow scripts/summary_figures.py (SVG, apply_chart_style).

Usage (from repo root):
    .venv/Scripts/python.exe scripts/page_additions/build_google_model_2026_page.py
"""
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import somlib as L  # noqa: E402
from summary_figures import heat  # noqa: E402
from src.constants import (BODY, FAINT, GRID, INK, SEASONS, SOURCE_COLORS,  # noqa: E402
                           TRIGGER_CONFIG, TRIGGER_STATIONS)
from src.plots import apply_chart_style, style_ax  # noqa: E402

apply_chart_style()

# ----------------------------------------------------------------------------- settings
OUT = HERE.parents[1] / "pages" / "google-model-2026"
FIGS = OUT / "figures"
BLOB = "ds-aa-som-floods/google_grrr_2026_11"
NEW = "grrr_2026_11"
RIVERS = ["juba", "shabelle"]
STATIONS = TRIGGER_STATIONS["juba"] + TRIGGER_STATIONS["shabelle"]
RIVER_OF = {st: r for r, sts in TRIGGER_STATIONS.items() for st in sts}
NAME = L.NAME
SEASON_TITLE = {"deyr": "Deyr, Oct to Dec", "gu": "Gu, Mar to May"}
SHORT = {NEW: "Google, new", "google_grrr": "Google, current", "glofas_v4": "GloFAS v4", "glofas_v5": "GloFAS v5"}
COL = {**SOURCE_COLORS, NEW: "#0E8A7B"}
MODELS = {"deyr": [NEW, "google_grrr", "glofas_v4", "glofas_v5"], "gu": [NEW, "google_grrr", "glofas_v4"]}
# each model's record: reanalysis years and forecast-archive years
RA_SPAN = {NEW: (1999, 2025), "google_grrr": (1999, 2023), "glofas_v4": (1999, 2023), "glofas_v5": (1999, 2023)}
FC_SPAN = {NEW: (2016, 2025), "google_grrr": (2016, 2023), "glofas_v4": (2003, 2023)}   # v5: no archive
# the current Google archive ends 2023-06-30: it holds Gu 2023 but no Deyr 2023
FC_END = {("google_grrr", "deyr"): 2022}
SCORE_RA = (1999, 2023)          # every model and the gauges overlap here
SCORE_FC = (2016, 2023)          # every forecast archive overlaps here
YEARS = list(range(1999, 2026))
CELL = {"hit_sev": "#123F5F", "hit": "#3E86BE", "false": "#F4A93B", "miss_sev": "#B34036",
        "miss": "#E3B4AF", "quiet": "#EDF1F4", "none": "#FFFFFF"}
WINDOW = {"deyr": ("09-20", "12-31"), "gu": ("02-15", "06-25")}
BASIS_TITLE = {"ra": "reanalysis", "fc": "forecasts"}
ok = lambda x: x is not None and not pd.isna(x)
fmt = lambda d: d.strftime("%d %b") if ok(d) else "never"
yrs = lambda xs: ", ".join(str(y) for y in sorted(xs)) if xs else "none"
rng = lambda a: f"{a[0]} to {a[1]}"


# ---------------------------------------------------------------------- the new model
def _read(name):
    local = os.environ.get("SOM_GRRR_2026_DIR")
    if local:
        return pd.read_parquet(Path(local) / name)
    import ocha_stratus as stratus
    return pd.read_parquet(io.BytesIO(stratus.load_blob_data(f"{BLOB}/{name}", stage="dev")))


def _read_json(name):
    local = os.environ.get("SOM_GRRR_2026_DIR")
    if local:
        return json.loads((Path(local) / name).read_text())
    import ocha_stratus as stratus
    return json.loads(stratus.load_blob_data(f"{BLOB}/{name}", stage="dev"))


_cache = {}


def new_daily():
    if "daily" not in _cache:
        w = _read("reanalysis_7stations.parquet")
        w.index = pd.to_datetime(w.index)
        d = w.stack().rename("discharge").reset_index()
        d.columns = ["date", "station", "discharge"]
        _cache["daily"] = d
    return _cache["daily"]


def new_reforecast():
    """Long format as somlib expects. In the new archive lead 0 is the state on the day before
    the issue (Reanalysis[T] == Reforecast[T+1, lead 0]), so lead k describes issue + k - 1 days.
    leadtime_days is set to k, which makes "day 1 = issue day" as in the GloFAS archive."""
    if "rf" not in _cache:
        r = _read("reforecast_7stations.parquet")
        r["issued_time"] = pd.to_datetime(r["issue_time"])
        lead = pd.to_timedelta(r["lead_time"])
        r["leadtime_days"] = lead.dt.days
        r["valid_time"] = r["issued_time"] + lead - pd.Timedelta(days=1)
        r["member"] = 0
        _cache["rf"] = r.rename(columns={"streamflow": "discharge"})[
            ["station", "issued_time", "valid_time", "leadtime_days", "member", "discharge"]]
    return _cache["rf"]


_daily0, _rf0 = L.daily, L.reforecast
L.daily = lambda m: new_daily() if m == NEW else _daily0(m)
L.reforecast = lambda m: new_reforecast() if m == NEW else _rf0(m)


# ---------------------------------------------------------------------------- record
def cfg(river, season):
    c = TRIGGER_CONFIG[(river, season)]
    return c["rp"], c["n_req"]


def gauge_peaks(river, season):
    lv = L.levels()
    d = lv[lv.station.isin(TRIGGER_STATIONS[river]) & lv.date.dt.month.isin(SEASONS[season])].dropna(subset=["level_m"])
    return {y: g.loc[g.level_m.idxmax(), "date"].normalize() for y, g in d.groupby(d.date.dt.year) if y in YEARS}


def record():
    """Per (river, season): the gauge benchmark and, per model, the first day the rule is met in
    the reanalysis (ra) and the first forecast issue that meets it (fc), by year."""
    R = {}
    for season in ("deyr", "gu"):
        for river in RIVERS:
            fl, sev, assess = L.benchmark(river, season, span=YEARS)
            models = {}
            for m in MODELS[season]:
                ra_years = list(range(RA_SPAN[m][0], RA_SPAN[m][1] + 1))
                ra = L.first_crossing_dates(m, river, season, *cfg(river, season), span=ra_years)
                fc, fc_years = {}, []
                if m in FC_SPAN:
                    fc_years = list(range(FC_SPAN[m][0], FC_END.get((m, season), FC_SPAN[m][1]) + 1))
                    fc = {y: v[0] for y, v in L.first_issue_dates(m, river, season, *cfg(river, season), span=fc_years).items()}
                models[m] = dict(ra=ra, fc=fc, ra_years=set(ra_years), fc_years=set(fc_years))
            R[(river, season)] = dict(flood=fl, severe=sev, assess=assess,
                                      bench=L.gauge_crossings(river, season, 3, span=YEARS),
                                      peak=gauge_peaks(river, season), models=models)
    return R


def activations_of(rec, m, basis):
    return set(rec["models"][m][basis])


def years_of(rec, m, basis):
    return rec["models"][m][f"{basis}_years"]


def score(rec, m, basis, span):
    yrs_ = set(range(span[0], span[1] + 1)) & years_of(rec, m, basis) & rec["assess"]
    f = activations_of(rec, m, basis) & yrs_
    sev, fl = rec["severe"] & yrs_, rec["flood"] & yrs_
    pod = len(f & sev) / len(sev) if sev else np.nan
    far = len(f - fl) / len(f) if f else np.nan
    prec = 1 - far if f else np.nan
    if np.isnan(pod):
        f1 = np.nan
    elif f and (pod + prec) > 0:
        f1 = 2 * pod * prec / (pod + prec)
    else:
        f1 = 0.0
    return dict(model=m, basis=basis, span=rng(span), n_years=len(yrs_), threshold_reached=sorted(f), severe=sorted(sev),
                caught=sorted(f & sev), ordinary=sorted((f & fl) - sev), false=sorted(f - fl),
                missed_severe=sorted(sev - f), POD=pod, FAR=far, F1=f1)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}.svg"


# ------------------------------------------------------- figure: season-by-season grid
def fig_backtest(R, season, basis):
    span = SCORE_RA if basis == "ra" else SCORE_FC
    models = [m for m in MODELS[season] if basis == "ra" or m in FC_SPAN]
    lines = []
    for river in RIVERS:
        rec = R[(river, season)]
        for m in models:
            lines.append((f"{river.title()}: {SHORT[m]}", rec, m))
    fig, ax = plt.subplots(figsize=(12.6, 0.6 * len(lines) + 2.4))
    for i, (name, rec, m) in enumerate(lines[::-1]):
        f, yr = activations_of(rec, m, basis), years_of(rec, m, basis)
        for j, y in enumerate(YEARS):
            if y not in yr or y not in rec["assess"]:
                kind = "none"
            elif y in f:
                kind = "hit_sev" if y in rec["severe"] else "hit" if y in rec["flood"] else "false"
            else:
                kind = "miss_sev" if y in rec["severe"] else "miss" if y in rec["flood"] else "quiet"
            ax.add_patch(Rectangle((j - 0.46, i - 0.36), 0.92, 0.72, facecolor=CELL[kind],
                                   edgecolor="white" if kind != "none" else GRID, linewidth=0.8))
        s = score(rec, m, basis, span)
        ax.text(len(YEARS) + 0.2, i + 0.14, f"{len(s['caught'])} of {len(s['severe'])} severe caught", va="center", fontsize=9.5, color=INK)
        ax.text(len(YEARS) + 0.2, i - 0.2, f"{len(s['false'])} with no gauge flood", va="center", fontsize=9.5,
                color="#B07A1E" if s["false"] else FAINT)
    ax.axhline(len(models) - 0.5, color=INK, linewidth=1.0)
    ax.set_xlim(-0.6, len(YEARS) + 4.8)
    ax.set_ylim(-0.6, len(lines) - 0.4)
    ax.set_yticks(range(len(lines)), [ln[0] for ln in lines[::-1]], fontsize=10.5)
    ax.set_xticks(range(0, len(YEARS), 2), [str(y) for y in YEARS[::2]], fontsize=9.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [Rectangle((0, 0), 1, 1, facecolor=CELL["hit_sev"], label="threshold reached, severe flood"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["hit"], label="threshold reached, ordinary flood"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["false"], label="threshold reached, no flood at the gauges"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss_sev"], label="severe flood missed"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss"], label="ordinary flood, threshold not reached"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["quiet"], edgecolor=GRID, label="quiet season"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID, label="no record")]
    ax.legend(handles=H, ncol=4, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.45, -0.08))
    what = "reanalysis" if basis == "ra" else "forecasts, leads 1 to 7"
    ax.set_title(f"{SEASON_TITLE[season]}: {what}. Tallies on the right count {rng(span)}", fontsize=12.5)
    return save(fig, f"backtest_{season}_{basis}")


# ---------------------------------------------------------- figure: timing vs the gauges
def worst_third(season, river):
    """Seasons in the worst third of the gauge ranking (second-gauge share index): year -> gauge rank."""
    _, Rk = rank_table(season, river)
    g = Rk["gauge"].dropna()
    cut = int(np.ceil(len(g) / 3))
    return {int(y): int(r) for y, r in g.items() if r <= cut}


def _timing_rows(R, season, basis):
    rows = []
    wt = {river: worst_third(season, river) for river in RIVERS}
    years = set().union(*(set(wt[r]) for r in RIVERS))
    for y in sorted(years, reverse=True):
        if basis == "fc" and y < SCORE_FC[0]:
            continue
        for river in RIVERS:
            rec = R[(river, season)]
            if y in wt[river] and y in rec["bench"]:
                rows.append((river, y, rec["bench"][y], rec["peak"].get(y), wt[river][y]))
    return rows


def fig_timing(R, season, basis):
    rows = _timing_rows(R, season, basis)
    models = [m for m in MODELS[season] if basis == "ra" or m in FC_SPAN]
    fig, ax = plt.subplots(figsize=(11.6, 0.52 * len(rows) + 2.8))
    ax.axvspan(-30, 0, color="#F9ECE9", zorder=0)
    ax.axvspan(0, 42, color="#EAF3EE", zorder=0)
    ax.axvspan(-42, -30, color="#F2F4F5", zorder=0)
    ax.text(21, len(rows) - 0.35, "threshold reached before the second gauge crossed", ha="center", va="center", fontsize=9, color="#2F6B4A", fontweight="bold")
    ax.text(-15, len(rows) - 0.35, "threshold reached after", ha="center", va="center", fontsize=9, color="#9E4B3F", fontweight="bold")
    ax.text(-36, len(rows) - 0.35, "threshold\nnot reached", ha="center", va="center", fontsize=8.5, color=BODY, fontweight="bold")
    ax.axvline(0, color=INK, linewidth=1.2, zorder=2)
    ax.axvline(-30, color="white", linewidth=1.5, zorder=1)
    for i, (river, y, b, pk, sev) in enumerate(rows[::-1]):
        ax.axhline(i, color="white", linewidth=1.0, zorder=0.5)
        if ok(pk):
            ax.scatter((b - pk).days, i, marker="|", s=200, color=INK, linewidths=1.6, zorder=3)
        rec = R[(river, season)]
        for k, m in enumerate(models):
            if y not in years_of(rec, m, basis):
                continue
            dy = (k - (len(models) - 1) / 2) * 0.2
            d = rec["models"][m][basis].get(y)
            x = max(min((b - d).days, 41), -29) if ok(d) else -36 + (k - (len(models) - 1) / 2) * 2.2
            ax.scatter(x, i + dy, s=58, color=COL[m], edgecolors="white", linewidths=0.9, zorder=4)
    ax.set_yticks(range(len(rows)), [f"{y} {r.title()}  (gauge rank {s})" for r, y, _, _, s in rows[::-1]], fontsize=10.5)
    ax.set_xlim(-42, 42)
    ax.set_ylim(-0.6, len(rows) + 0.1)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40], ["30", "20", "10", "0", "10", "20", "30", "40"])
    ax.set_xlabel("days before the second gauge crossed its 1-in-3 level", fontsize=10.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [Line2D([], [], marker="o", ls="", color=COL[m], markersize=8, label=SHORT[m]) for m in models]
    H += [Line2D([], [], marker="|", ls="", color=INK, markersize=12, markeredgewidth=1.6, label="river peak")]
    ax.legend(handles=H, ncol=len(H), fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    what = "reanalysis" if basis == "ra" else "forecasts, first issue at leads 1 to 7"
    ax.set_title(f"{SEASON_TITLE[season]}: {what}. Worst third of seasons at the gauges", fontsize=12)
    return save(fig, f"timing_{season}_{basis}")


# --------------------------------------------------- figure: one season at all seven points
def fig_season(R, season, year):
    """One season in the style of the activation-timing page: per river, the new model at each
    station as a filled share of its own trigger level, the station's gauge as a dashed share of
    its own 1-in-3, the days the threshold was reached (new model and the source running the
    window today), and flood exposure in the 14 districts where FloodScan covers the season."""
    import season_charts as sc
    a, b = WINDOW[season]
    A, B = pd.Timestamp(f"{year}-{a}"), pd.Timestamp(f"{year}-{b}")
    lv = L.levels()
    live = {"deyr": "glofas_v4", "gu": "google_grrr"}[season]
    has_exp = year <= 2023
    n_rows = 3 if has_exp else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(11.5, 9.6 if has_exp else 7.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 3, 1.5] if has_exp else [3, 3]))
    for ax, river in zip(axes[:2], RIVERS):
        rec = R[(river, season)]
        rp, n = cfg(river, season)
        sts = TRIGGER_STATIONS[river]
        d = L.daily(NEW)
        d = d[(d.date >= A) & (d.date <= B) & d.station.isin(sts)]
        ends = []
        for st in sts:
            s = d[d.station == st].sort_values("date")
            thr = L.model_threshold(NEW, st, season, rp)
            if s.empty or np.isnan(thr):
                continue
            pct = 100 * s.discharge / thr
            ax.fill_between(s.date, pct, color=sc.C[st], alpha=0.22, lw=0)
            ax.plot(s.date, pct, color=sc.C[st], lw=1.3)
            ends.append([pct.iloc[-1], st])
        glev = L.gauge_levels(river, season, 3)
        for st in sts:
            level = glev.get(st, np.nan)
            if np.isnan(level):
                continue
            w = lv[(lv.station == st) & (lv.date >= A) & (lv.date <= B)].sort_values("date")
            if len(w):
                ax.plot(w.date, 100 * w.level_m / level, color=sc.C[st], lw=1.4, ls=(0, (5, 2)))
        ax.axhline(100, color=sc.VOTE, ls="--", lw=1.1, zorder=3)
        top = max(ax.get_ylim()[1], 130)
        marks = []
        ra = rec["models"][NEW]["ra"].get(year)
        if ok(ra):
            marks.append((ra, COL[NEW], f"new model\n{ra:%d %b}"))
        if year in rec["models"][live]["ra_years"]:
            rl = rec["models"][live]["ra"].get(year)
            if ok(rl):
                marks.append((rl, COL[live], f"{SHORT[live]}\n{rl:%d %b}"))
        bd = rec["bench"].get(year)
        if ok(bd):
            marks.append((bd, INK, f"second gauge\n{bd:%d %b}"))
        for k, (dte, colr, lab) in enumerate(sorted(marks, key=lambda t: t[0])):
            ax.axvline(dte, color=colr, lw=1.3, alpha=0.85)
            ax.annotate(lab, xy=(dte, top * (0.99 - 0.16 * k)), xytext=(5, 0), textcoords="offset points", color=colr,
                        fontsize=8.5, va="top", ha="left")
        gap = top * 0.075
        ends.sort()
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < gap:
                ends[i][0] = ends[i - 1][0] + gap
        for yv, st in ends:
            ax.text(B + pd.Timedelta(days=2), min(yv, top * 0.98), NAME[st], color=sc.C[st], fontsize=9, va="center", fontweight="bold")
        ax.set_ylabel("% of its own level", color=FAINT, fontsize=9.5)
        ax.text(0, 1.07, f"{river.title()}   ·   new Google model, {n} of {len(sts)} over 1-in-{rp}", transform=ax.transAxes, fontsize=12.5, color=INK)
        ax.grid(axis="y", color="#F2F5F5", lw=1)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=FAINT, labelsize=9, length=0)
        ax.spines["bottom"].set_color("#E4E9E9")
        ax.set_xlim(A, B + pd.Timedelta(days=int((B - A).days * 0.09)))
        ax.set_ylim(0, top)
    if has_exp:
        axe = axes[2]
        e = sc.exposure().set_index("valid_date")["sum"].groupby(level=0).sum()
        e = e[(e.index >= A) & (e.index <= B)]
        axe.fill_between(e.index, e.values, color=sc.SANDL, alpha=0.35, lw=0)
        axe.plot(e.index, e.values, color=sc.SAND, lw=1.2)
        fd, _ = sc.flood_date(season, year)
        if fd is not None:
            axe.axvline(fd, color=INK, lw=1.3, alpha=0.85)
            axe.annotate(f"third district\n{fd:%d %b}", xy=(fd, axe.get_ylim()[1] * 0.98), xytext=(5, 0), textcoords="offset points",
                         color=INK, fontsize=8.5, va="top", ha="left")
        axe.set_ylabel("people exposed", color=FAINT, fontsize=9.5)
        axe.text(0, 1.13, "Flood exposure, 14 anticipatory-action districts", transform=axe.transAxes, fontsize=11, color=INK)
        axe.grid(axis="y", color="#F2F5F5", lw=1)
        axe.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            axe.spines[sp].set_visible(False)
        axe.spines["bottom"].set_color("#E4E9E9")
        axe.tick_params(colors=FAINT, labelsize=9, length=0)
        axe.set_ylim(0, None)
        axe.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.text(0.008, 0.99, f"{season.title()} {year}", fontsize=15, color=INK, va="top")
    fig.text(0.008, 0.963, "Solid: new Google model per station, % of its trigger level.  Dashed, same colour: the station's gauge, % of its 1-in-3.  Red dashes: 100%.",
             fontsize=9.5, color=FAINT, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return save(fig, f"season_{season}_{year}")


# -------------------------------------------------------- figure: seasons ranked by peak
RANK_COLS = [("gauge", None), (NEW, "ra"), (NEW, "fc"), ("google_grrr", "ra"), ("google_grrr", "fc"),
             ("glofas_v4", "ra"), ("glofas_v4", "fc"), ("glofas_v5", "ra")]
N_RANK = 5
RANK_RP = 3   # each station's peak is read as a share of its own 1-in-3 level on that source


def _fc_peaks(m, st, season):
    rf = L.reforecast(m)
    rf = rf[(rf.station == st) & rf.leadtime_days.between(1, 7) & rf.valid_time.dt.month.isin(SEASONS[season])]
    med = rf.groupby(["issued_time", "valid_time"]).discharge.median().reset_index()
    years = set(range(FC_SPAN[m][0], FC_END.get((m, season), FC_SPAN[m][1]) + 1))
    mx = med.groupby(med.valid_time.dt.year).discharge.max()
    return mx[mx.index.isin(years)]


def season_index(season, river, source, basis):
    """Per year, the median across the river's stations of the season peak as a share of that
    station's own 1-in-RANK_RP level on the same source. Forecast peaks are read against the
    model's reanalysis levels, as the trigger reads them."""
    cols = {}
    if source == "gauge":
        lv = L.levels()
        lev = L.gauge_levels(river, season, RANK_RP)
        for st in TRIGGER_STATIONS[river]:
            if np.isnan(lev[st]):
                continue
            g = lv[(lv.station == st) & lv.date.dt.month.isin(SEASONS[season])].set_index("date").level_m.dropna()
            gm = g.groupby(g.index.year).max()
            cols[st] = gm[gm.index.isin(YEARS)] / lev[st]
    else:
        for st in TRIGGER_STATIONS[river]:
            lev = L.model_threshold(source, st, season, RANK_RP)
            if basis == "ra":
                x = L.season_series(source, st, season, span=RA_SPAN[source])
                cols[st] = x.groupby(x.index.year).max() / lev
            else:
                cols[st] = _fc_peaks(source, st, season) / lev
    # the second-highest station share: the level the river's second gauge reaches, as in the benchmark
    df = pd.DataFrame(cols)
    return df.apply(lambda r: r.dropna().nlargest(2).min() if r.notna().sum() >= 2 else np.nan, axis=1)


def rank_table(season, river):
    """Season index per source (all stations), and each source's rank of every season."""
    idx = {"gauge": season_index(season, river, "gauge", None)}
    for m, basis in RANK_COLS[1:]:
        if basis == "ra" or m in FC_SPAN:
            idx[(m, basis)] = season_index(season, river, m, basis)
    P = pd.DataFrame(idx)
    ranks = P.rank(ascending=False, method="min")
    return P, ranks


def consistency(season, top_n=5):
    """Per river: the gauges' worst seasons that every source with a record ranks in its top_n,
    and those that fewer than half the sources do."""
    out = {}
    for river in RIVERS:
        P, Rk = rank_table(season, river)
        top = Rk["gauge"].dropna().sort_values().index[:N_RANK]
        srcs = [c for c in Rk.columns if c != "gauge"]
        all_top, few_top = [], []
        for y in top:
            r = Rk.loc[y, srcs].dropna()
            if len(r) == 0:
                continue
            k = int((r <= top_n).sum())
            if k == len(r):
                all_top.append(f"{int(y)} ({len(r)} sources)")
            elif k * 2 < len(r):
                few_top.append(f"{int(y)} ({k} of {len(r)})")
        out[river] = (all_top, few_top)
    return out


def fig_ranks(season):
    """One panel per season: the gauges' N_RANK worst seasons on each river, Juba then Shabelle,
    and the rank every source gives the same season within its own record."""
    rows, mats, labels_y = [], [], []
    keys = None
    for river in RIVERS:
        P, Rk = rank_table(season, river)
        top = Rk["gauge"].dropna().sort_values().index[:N_RANK]
        cols = [c for c in RANK_COLS if (c == ("gauge", None)) or (c in Rk.columns)]
        keys = ["gauge" if c == ("gauge", None) else c for c in cols]
        n = {k: int(Rk[k].notna().sum()) for k in keys}
        for y in top:
            mats.append(Rk.loc[y, keys].values.astype(float))
            labels_y.append(f"{river.title()} {int(y)}   {100 * P.loc[y, 'gauge']:.0f}%")
            rows.append(river)
    mat = np.array(mats)
    fig, ax = plt.subplots(figsize=(11.6, 0.5 * len(rows) + 2.4))
    shown = np.where(np.isnan(mat), np.nan, 1 - (mat - 1) / 24)
    ax.imshow(shown, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor=GRID, linewidth=0.8))
                continue
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=10, color="white" if shown[i, j] > 0.55 else BODY)
    labels = []
    for k in keys:
        if k == "gauge":
            labels.append(f"gauges\n({n[k]} seasons)")
        else:
            labels.append(f"{SHORT[k[0]].replace(', ', ' ')}\n{'reanalysis' if k[1] == 'ra' else 'forecasts'}\n({n[k]} seasons)")
    ax.set_xticks(range(len(keys)), labels, fontsize=8.5)
    ax.set_yticks(range(len(rows)), labels_y, fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    n_juba = rows.count("juba")
    ax.axhline(n_juba - 0.5, color=INK, linewidth=1.2)
    ax.set_title(f"{SEASON_TITLE[season]}: the {N_RANK} worst gauge seasons on each river, worst first, and the rank each source gives them "
                 f"(label: second gauge, % of its 1-in-3; 1 = that source's highest; dark = high)", fontsize=10.5)
    return save(fig, f"ranks_{season}")


# ------------------------------------------------------------ figure: scale of the levels
def fig_levels():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6), sharey=True)
    ys = list(range(len(STATIONS)))[::-1]
    for ax, season in zip(axes, ("deyr", "gu")):
        for y, st in zip(ys, STATIONS):
            rp, _ = cfg(RIVER_OF[st], season)
            o, nw = L.model_threshold("google_grrr", st, season, rp), L.model_threshold(NEW, st, season, rp)
            ax.plot([o, nw], [y, y], color=GRID, lw=2.4, zorder=1)
            ax.scatter(o, y, s=52, color=COL["google_grrr"], edgecolors="white", zorder=3, label="Google, current" if y == ys[0] else None)
            ax.scatter(nw, y, s=52, color=COL[NEW], edgecolors="white", zorder=3, label="Google, new" if y == ys[0] else None)
            ax.text(max(o, nw) * 1.1, y, f"{nw / o:.1f}x", va="center", fontsize=8.5, color=FAINT)
        ax.set_xscale("log")
        ax.set_xlim(80, 3000)
        ax.set_title(SEASON_TITLE[season])
        ax.set_xlabel("trigger level at the point, m³/s (log scale)")
        style_ax(ax, grid="x")
    axes[0].set_yticks(ys, [NAME[s] for s in STATIONS])
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.subplots_adjust(wspace=0.06)
    return save(fig, "levels")


# ------------------------------------------------------------- figure: station by station
def station_scores():
    rows = []
    lv = L.levels()
    span = set(range(SCORE_RA[0], SCORE_RA[1] + 1))
    for season in ("deyr", "gu"):
        for st in STATIONS:
            r = RIVER_OF[st]
            rp, _ = cfg(r, season)
            lev5, lev3 = L.gauge_levels(r, season, 5)[st], L.gauge_levels(r, season, 3)[st]
            s = lv[lv.station == st].set_index("date").level_m.dropna()
            s = s[s.index.month.isin(SEASONS[season]) & s.index.year.isin(span)]
            am = s.groupby(s.index.year).max()
            if np.isnan(lev5) or len(am) < 8:
                continue
            sev, fl, have = set(am[am >= lev5].index), set(am[am >= lev3].index), set(am.index)
            for m in MODELS[season]:
                x = L.season_series(m, st, season)
                mx = x.groupby(x.index.year).max()
                f = set(mx[mx >= L.model_threshold(m, st, season, rp)].index) & have
                pod = len(f & sev) / len(sev) if sev else np.nan
                far = len(f - fl) / len(f) if f else np.nan
                prec = 1 - far if f else np.nan
                rows.append(dict(season=season, station=st, model=m, n_years=len(have), POD=pod, FAR=far,
                                 F1=2 * pod * prec / (pod + prec) if f and (pod + prec) > 0 else 0.0))
    return pd.DataFrame(rows)


def fig_stations(S, season):
    ms = MODELS[season]
    sts = [s for s in STATIONS if s in set(S[S.season == season].station)]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 0.5 * len(sts) + 1.8))
    for ax, met, title in zip(axes, ["POD", "FAR", "F1"], ["Severe seasons caught (share)", "False alarms (share)", "F1"]):
        mat = [[float(S[(S.season == season) & (S.station == st) & (S.model == m)][met].iloc[0]) for m in ms] for st in sts]
        heat(ax, mat, [SHORT[m] for m in ms], [NAME[s] for s in sts] if met == "POD" else [""] * len(sts), reverse=(met == "FAR"))
        ax.set_title(title + "\n(dark = better)", fontsize=10)
    fig.subplots_adjust(wspace=0.08)
    return save(fig, f"stations_{season}")


# ------------------------------------------------------------------------------ html
def shade(v, reverse=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    x = 1 - v if reverse else v
    return f' style="background:rgba(30,121,95,{0.08 + 0.42 * x:.2f})"'


def num(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def score_table(R, season):
    """One table per season: a row per source and record, the two rivers side by side."""
    hdr = ("<table class='data'><thead><tr><th rowspan='2'>Source</th><th rowspan='2'>Record</th>"
           "<th colspan='3'>Juba</th><th colspan='3'>Shabelle</th></tr>"
           "<tr><th>severe caught</th><th>no gauge flood</th><th>F1</th><th>severe caught</th><th>no gauge flood</th><th>F1</th></tr></thead><tbody>")
    out = [hdr]
    for m in MODELS[season]:
        for basis, span in (("ra", SCORE_RA), ("fc", SCORE_FC)):
            if basis == "fc" and m not in FC_SPAN:
                continue
            cells = [f"<td>{SHORT[m]}</td><td class='muted'>{'reanalysis ' + rng(SCORE_RA) if basis == 'ra' else 'forecasts ' + rng(SCORE_FC)}</td>"]
            for river in RIVERS:
                rec = R[(river, season)]
                s = score(rec, m, basis, span)
                yr = years_of(rec, m, basis)
                tag = "" if yr >= set(range(span[0], span[1] + 1)) else f" <span class='muted'>(to {max(yr)})</span>"
                cells.append(f"<td>{len(s['caught'])} of {len(s['severe'])}{tag}</td><td>{len(s['false'])}</td>"
                             f"<td{shade(s['F1'])}>{num(s['F1'])}</td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def v4_full_archive(R, season):
    """GloFAS v4's forecast record over its whole archive, 2003 to 2023, as one sentence per river."""
    parts = []
    for river in RIVERS:
        s = score(R[(river, season)], "glofas_v4", "fc", FC_SPAN["glofas_v4"])
        parts.append(f"{river.title()}: reaches the threshold in {yrs(s['threshold_reached'])}; severe seasons {yrs(s['severe'])}, of which it catches {yrs(s['caught'])}; "
                     f"threshold reached with no gauge flood {yrs(s['false'])}.")
    return " ".join(parts)


def after_2023(R, season, y):
    act_r = [r.title() for r in RIVERS if y in R[(r, season)]["models"][NEW]["ra"]]
    fc_r = [r.title() for r in RIVERS if y in R[(r, season)]["models"][NEW]["fc"]]
    outc = []
    for r in RIVERS:
        rec = R[(r, season)]
        outc.append(f"{r.title()} {'severe' if y in rec['severe'] else 'flood' if y in rec['flood'] else 'nothing' if y in rec['assess'] else 'no gauge record'}")
    say = lambda rs: ("threshold reached on the " + " and ".join(rs)) if rs else "threshold not reached"
    if act_r == fc_r:
        rec_txt = say(act_r) + " on both records"
    else:
        rec_txt = f"reanalysis: {say(act_r)}; forecasts: {say(fc_r)}"
    return f"{rec_txt}; gauges: {', '.join(outc)}"


def build():
    R = record()
    ST = station_scores()
    F = {}
    for s in ("deyr", "gu"):
        for b in ("ra", "fc"):
            F[f"bt_{s}_{b}"] = fig_backtest(R, s, b)
            F[f"tm_{s}_{b}"] = fig_timing(R, s, b)
        F[f"st_{s}"] = fig_stations(ST, s)
    F["d23"] = fig_season(R, "deyr", 2023)
    F["d24"] = fig_season(R, "deyr", 2024)
    F["g24"] = fig_season(R, "gu", 2024)
    F["lv"] = fig_levels()
    F["rk_deyr"], F["rk_gu"] = fig_ranks("deyr"), fig_ranks("gu")
    C = {s: consistency(s) for s in ("deyr", "gu")}
    cons_txt = lambda xs: ", ".join(xs) if xs else "none"
    T = {s: score_table(R, s) for s in ("deyr", "gu")}

    # everything behind the page, for review
    dump, sc = {}, []
    for (river, season), rec in R.items():
        dump[f"{river}_{season}"] = dict(
            flood=sorted(rec["flood"]), severe=sorted(rec["severe"]), assessable=sorted(rec["assess"]),
            second_gauge_1in3={str(y): v.strftime("%Y-%m-%d") for y, v in rec["bench"].items()},
            gauge_peak={str(y): v.strftime("%Y-%m-%d") for y, v in rec["peak"].items()},
            models={m: dict(reanalysis={str(y): v.strftime("%Y-%m-%d") for y, v in mm["ra"].items()},
                            forecast={str(y): v.strftime("%Y-%m-%d") for y, v in mm["fc"].items()},
                            reanalysis_years=rng(RA_SPAN[m]), forecast_years=rng(FC_SPAN[m]) if m in FC_SPAN else None)
                    for m, mm in rec["models"].items()})
        for m in MODELS[season]:
            sc.append(dict(river=river, season=season, **score(rec, m, "ra", SCORE_RA)))
            if m in FC_SPAN:
                sc.append(dict(river=river, season=season, **score(rec, m, "fc", SCORE_FC)))
                if m == "glofas_v4":
                    sc.append(dict(river=river, season=season, **score(rec, m, "fc", FC_SPAN[m])))
    levels = {m: {s: {st: round(L.model_threshold(m, st, s, cfg(RIVER_OF[st], s)[0]), 1) for st in STATIONS} for s in ("deyr", "gu")}
              for m in [NEW, "google_grrr", "glofas_v4", "glofas_v5"]}
    (OUT / "data.json").write_text(json.dumps(dict(
        outlets=_read_json("chosen_outlets.json"), trigger_levels=levels, record=dump, scores=sc,
        station_scores=json.loads(ST.to_json(orient="records"))), indent=1, default=str))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Google's new river model: Somalia Riverine Flood Trigger</title>
<meta name="description" content="Google's November 2026 river model at the seven Somalia trigger points, read with the trigger's rules next to GloFAS v4, GloFAS v5 and the current Google model, on the reanalysis and on the forecast archives, against the SWALIM gauges.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css">
<style>
.hero.hero-sub {{ padding:40px 44px 34px; }}
.hero.hero-sub h1 {{ font-size:28px; }}
.crumb {{ font-size:12px; margin:0 0 14px; }}
.crumb a {{ color:rgba(255,255,255,.85); text-decoration:none; }}
article {{ padding:8px 44px 24px; max-width:960px; }}
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:21px; color:var(--n9); margin:44px 0 10px; line-height:1.25; }}
article h2 .num {{ color:var(--b5); font-size:15px; margin-right:8px; }}
article h3 {{ font-family:'Merriweather',Georgia,serif; font-size:15.5px; color:var(--n9); margin:26px 0 6px; }}
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.65; }}
article ul {{ padding-left:22px; }}
article code {{ background:var(--n05); border:1px solid #e2e7e7; padding:1px 5px; border-radius:3px; font-size:12.5px; color:var(--b7); }}
.muted {{ color:#6b7683; }}
figure.chart {{ margin:18px 0 26px; }}
figure.chart img {{ width:100%; height:auto; display:block; }}
figure.chart figcaption {{ font-size:13px; color:#55606d; line-height:1.55; margin-top:10px; padding-left:2px; }}
.callout {{ margin:20px 0; padding:13px 17px; border-radius:4px; background:var(--b05); border-left:5px solid var(--b5); font-size:13.5px; color:var(--n8); line-height:1.6; }}
.callout p {{ margin:0 0 7px; font-size:13.5px; }}
.callout p:last-child {{ margin:0; }}
.callout b {{ color:var(--n9); }}
.tablewrap {{ overflow-x:auto; }}
table.data {{ border-collapse:collapse; width:100%; margin:16px 0 22px; font-size:13px; }}
table.data caption {{ text-align:left; font-family:'Merriweather',Georgia,serif; font-size:14px; color:var(--n9); padding:0 0 8px; }}
table.data th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:7px 10px 6px; }}
table.data td {{ border-bottom:1px solid #eef1f1; padding:7px 10px; color:var(--n8); vertical-align:top; }}
table.data tr.riv td {{ color:var(--n9); font-weight:500; padding-top:12px; border-bottom:1px solid #d9dfdf; background:#fff; }}
table.data td:first-child, table.data td:nth-child(2) {{ white-space:nowrap; }}
table.data td:nth-child(4), table.data td:nth-child(7) {{ border-right:1px solid #e2e7e7; }}
table.data th[colspan] {{ text-align:center; border-bottom:1px solid #e2e7e7; }}
table.data .y {{ display:block; font-size:11px; color:var(--n7); }}
.provenance {{ margin:34px 44px 40px; padding-top:14px; border-top:1px solid #e2e7e7; font-size:12px; color:var(--n7); line-height:1.7; }}
.provenance li {{ font-size:12px; color:var(--n7); margin:0 0 6px; }}
@media (max-width:640px) {{ article {{ padding:4px 22px 16px; }} .hero.hero-sub {{ padding:32px 22px 28px; }} .provenance {{ margin:28px 22px 32px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <div class="inner">
      <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / Google's new river model</p>
      <h1>Google's new river model</h1>
      <p>Google's November 2026 river model at the seven trigger points, read with the trigger's own rules next to GloFAS v4,
         GloFAS v5 and the current Google model, on each model's reanalysis and on its forecast archive, against the SWALIM
         gauges. The live Flood Hub feed does not run the new model yet.</p>
    </div>
  </header>
  <article>
    <div class="callout">
      <p><b>Threshold reached:</b> three of the four Juba points, or two of the three Shabelle points, over their own level on the
         same day (Juba 1-in-{TRIGGER_CONFIG[('juba','deyr')]['rp']} and Shabelle 1-in-{TRIGGER_CONFIG[('shabelle','deyr')]['rp']} in Deyr, Juba 1-in-{TRIGGER_CONFIG[('juba','gu')]['rp']} and Shabelle 1-in-{TRIGGER_CONFIG[('shabelle','gu')]['rp']} in Gu, each model's
         own levels on its 1999 to 2023 seasonal maxima).</p>
      <p><b>Reanalysis:</b> the model's own record of the past. <b>Forecasts:</b> the first issue reaching the threshold at leads 1 to 7 days,
         what the live trigger reads. Archives: GloFAS v4 2003 to 2023, current Google 2016 to June 2023, new Google 2016 to 2025,
         GloFAS v5 none. Scores: {rng(SCORE_RA)} on the reanalysis, {rng(SCORE_FC)} on the forecasts.</p>
      <p><b>Gauges:</b> a flood is two SWALIM gauges on the river at their own 1-in-3 level, severe at 1-in-5. Gauge data run to July 2026.</p>
    </div>

    <h2><span class="num">1</span>The worst floods first: the benchmark</h2>
    <p>The gauges' five worst seasons, worst first, and the rank each source gives the same seasons in its own record. A season's
       index is the level its second-highest station reached, as a share of that station's own 1-in-3 on that source.</p>
    <figure class="chart"><img src="{F['rk_deyr']}" alt="Deyr, seasons ranked">
      <figcaption>Deyr. In the top five of every source that covers the season: Juba {cons_txt(C['deyr']['juba'][0])}; Shabelle {cons_txt(C['deyr']['shabelle'][0])}.
        In the top five of fewer than half: Juba {cons_txt(C['deyr']['juba'][1])}; Shabelle {cons_txt(C['deyr']['shabelle'][1])}.</figcaption></figure>
    <figure class="chart"><img src="{F['rk_gu']}" alt="Gu, seasons ranked">
      <figcaption>Gu. In the top five of every source that covers the season: Juba {cons_txt(C['gu']['juba'][0])}; Shabelle {cons_txt(C['gu']['shabelle'][0])}.
        In the top five of fewer than half: Juba {cons_txt(C['gu']['juba'][1])}; Shabelle {cons_txt(C['gu']['shabelle'][1])}.
        Belet Weyne's Gu record is capped at 8.30 m in six seasons.</figcaption></figure>

    <h2><span class="num">2</span>Deyr, season by season</h2>
    <figure class="chart"><img src="{F['bt_deyr_ra']}" alt="Deyr, reanalysis, season by season"><figcaption>Reanalysis. One row per model and river, one cell per season.</figcaption></figure>
    <figure class="chart"><img src="{F['bt_deyr_fc']}" alt="Deyr, forecasts, season by season"><figcaption>Forecasts, leads 1 to 7. Blank cells are seasons outside the archive.</figcaption></figure>
    <div class="tablewrap">{T['deyr']}</div>
    <p class="muted" style="font-size:12.5px;margin-top:-10px">Severe caught: of the severe seasons in the scoring span. No gauge flood: seasons with the threshold reached and no flood at the gauges. F1 combines the two; POD, FAR and the years are in <code>data.json</code>.</p>
    <p class="muted" style="font-size:13px">New model, Deyr 2024: {after_2023(R, 'deyr', 2024)}. Deyr 2025: {after_2023(R, 'deyr', 2025)}.</p>

    <h2><span class="num">3</span>Gu, season by season</h2>
    <figure class="chart"><img src="{F['bt_gu_ra']}" alt="Gu, reanalysis, season by season"><figcaption>Reanalysis.</figcaption></figure>
    <figure class="chart"><img src="{F['bt_gu_fc']}" alt="Gu, forecasts, season by season"><figcaption>Forecasts, leads 1 to 7.</figcaption></figure>
    <div class="tablewrap">{T['gu']}</div>
    <p class="muted" style="font-size:13px">New model, Gu 2024: {after_2023(R, 'gu', 2024)}. Gu 2025: {after_2023(R, 'gu', 2025)}.</p>

    <h2><span class="num">4</span>Timing against the gauges</h2>
    <p>The seasons in the worst third of the gauge ranking, one row per river and season. Zero is the day the river's second
       gauge crossed its 1-in-3 level. A dot on the green side means that source reached the threshold that many days before the
       gauges confirmed the flood; on the red side, that many days after; in the grey column, not at all that season. The bar is
       the river's peak.</p>
    <figure class="chart"><img src="{F['tm_deyr_ra']}" alt="Deyr timing, reanalysis"><figcaption>Deyr, reanalysis.</figcaption></figure>
    <figure class="chart"><img src="{F['tm_deyr_fc']}" alt="Deyr timing, forecasts"><figcaption>Deyr, forecasts. GloFAS v5 has no archive; the current Google archive ends June 2023.</figcaption></figure>
    <figure class="chart"><img src="{F['tm_gu_ra']}" alt="Gu timing, reanalysis"><figcaption>Gu, reanalysis.</figcaption></figure>
    <figure class="chart"><img src="{F['tm_gu_fc']}" alt="Gu timing, forecasts"><figcaption>Gu, forecasts.</figcaption></figure>

    <h2><span class="num">5</span>Three seasons at all seven points</h2>
    <p>Same layout as the season charts on the timing page: the new model at each station as a share of its own trigger level, the gauges
       dashed as a share of their 1-in-3, the days the threshold was reached, and flood exposure where FloodScan covers the season.</p>
    <figure class="chart"><img src="{F['d23']}" alt="Deyr 2023, all points"></figure>
    <figure class="chart"><img src="{F['d24']}" alt="Deyr 2024, all points"></figure>
    <figure class="chart"><img src="{F['g24']}" alt="Gu 2024, all points"></figure>

    <h2><span class="num">6</span>Station by station</h2>
    <p>Each point against its own gauge, reanalysis {rng(SCORE_RA)}; severe is the gauge over its own 1-in-5, a false alarm is the model over its level with the gauge under its 1-in-3.</p>
    <figure class="chart"><img src="{F['st_deyr']}" alt="Deyr station scores"><figcaption>Deyr.</figcaption></figure>
    <figure class="chart"><img src="{F['st_gu']}" alt="Gu station scores"><figcaption>Gu.</figcaption></figure>

    <h2><span class="num">7</span>The two Google models are on different scales</h2>
    <figure class="chart"><img src="{F['lv']}" alt="Trigger levels, current against new"><figcaption>The trigger level at each point on the current Google model and on the new one, both fitted the same way.</figcaption></figure>
  </article>

  <div class="provenance">
    <ul>
      <li>Source: Google Runoff Reanalysis and Reforecast, November 2026 release (reanalysis 1981 to 2025, reforecast 2016 to 2025, leads 0 to 10 days),
          one HydroBasins outlet per station, the main-stem outlet within 15 km with the largest 1-in-5 flow; ids in <code>data.json</code>.</li>
      <li>Leads: GloFAS day n is issue + (n - 1); the current Google archive's lead n is issue + n; the new archive is relabelled so day 1 is the
          issue day. "Leads 1 to 7" is the issue day to six days ahead for GloFAS and the new model, one to seven days ahead for the current Google model.</li>
      <li>On the new model, Juba Deyr 2006 and 2017 and Juba Gu 2013 and 2023 have all four points over their levels within the season, by 1 to 4%,
          but never on the same day. Read at leads 0 to 6, the current Google model does not reach the threshold in Gu 2023.</li>
      <li>Shabelle Deyr 2020, Gu 2016 and Gu 2023 count as severe on a gauge reading equal to its level (Jowhar 5.30 m; Belet Weyne at its 8.30 m
          ceiling). The Juba Deyr 2023 peaks are capped at bank full. Dollow's Deyr record has eight seasons.</li>
      <li>Eight seasons on the forecast record; one to five severe seasons per window. GloFAS forecasts are an ensemble median, the Google archives
          a single trace. The current Google archive has not been verified as the live Flood Hub model; the new model's scale is not the live feed's.
          2026 is in no archive.</li>
      <li>Generated by <code>scripts/page_additions/build_google_model_2026_page.py</code>; dates, levels and scores in <code>data.json</code>.</li>
    </ul>
  </div>
</div>
</body>
</html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("written", OUT / "index.html")
    print(pd.DataFrame(sc)[["season", "river", "model", "basis", "span", "caught", "false", "POD", "FAR", "F1"]].to_string(index=False))


if __name__ == "__main__":
    build()
