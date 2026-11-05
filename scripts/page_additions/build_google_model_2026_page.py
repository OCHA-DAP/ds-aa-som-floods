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


def fired(rec, m, basis):
    return set(rec["models"][m][basis])


def years_of(rec, m, basis):
    return rec["models"][m][f"{basis}_years"]


def score(rec, m, basis, span):
    yrs_ = set(range(span[0], span[1] + 1)) & years_of(rec, m, basis) & rec["assess"]
    f = fired(rec, m, basis) & yrs_
    sev, fl = rec["severe"] & yrs_, rec["flood"] & yrs_
    pod = len(f & sev) / len(sev) if sev else np.nan
    far = len(f - fl) / len(f) if f else np.nan
    prec = 1 - far if f else np.nan
    f1 = 2 * pod * prec / (pod + prec) if f and not np.isnan(pod) and (pod + prec) > 0 else (0.0 if f or sev else np.nan)
    return dict(model=m, basis=basis, span=rng(span), n_years=len(yrs_), fires=sorted(f), severe=sorted(sev),
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
        f, yr = fired(rec, m, basis), years_of(rec, m, basis)
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
    H = [Rectangle((0, 0), 1, 1, facecolor=CELL["hit_sev"], label="activated, severe flood"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["hit"], label="activated, ordinary flood"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["false"], label="activated, no flood at the gauges"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss_sev"], label="severe flood missed"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss"], label="ordinary flood, no activation"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["quiet"], edgecolor=GRID, label="quiet season"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID, label="no record")]
    ax.legend(handles=H, ncol=4, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.45, -0.08))
    what = "reanalysis" if basis == "ra" else "forecasts, leads 1 to 7"
    ax.set_title(f"{SEASON_TITLE[season]}: {what}. Tallies on the right count {rng(span)}", fontsize=12.5)
    return save(fig, f"backtest_{season}_{basis}")


# ---------------------------------------------------------- figure: timing vs the gauges
def _timing_rows(R, season, basis):
    rows = []
    for y in sorted(set().union(*(R[(r, season)]["flood"] for r in RIVERS)), reverse=True):
        if basis == "fc" and y < SCORE_FC[0]:
            continue
        for river in RIVERS:
            rec = R[(river, season)]
            if y in rec["flood"] and y in rec["bench"]:
                rows.append((river, y, rec["bench"][y], rec["peak"].get(y), y in rec["severe"]))
    return rows


def fig_timing(R, season, basis):
    rows = _timing_rows(R, season, basis)
    models = [m for m in MODELS[season] if basis == "ra" or m in FC_SPAN]
    fig, ax = plt.subplots(figsize=(11.6, 0.52 * len(rows) + 2.8))
    ax.axvspan(-30, 0, color="#F9ECE9", zorder=0)
    ax.axvspan(0, 42, color="#EAF3EE", zorder=0)
    ax.axvspan(-42, -30, color="#F2F4F5", zorder=0)
    ax.text(21, len(rows) - 0.35, "activated before the second gauge crossed", ha="center", va="center", fontsize=9, color="#2F6B4A", fontweight="bold")
    ax.text(-15, len(rows) - 0.35, "activated after", ha="center", va="center", fontsize=9, color="#9E4B3F", fontweight="bold")
    ax.text(-36, len(rows) - 0.35, "no\nactivation", ha="center", va="center", fontsize=8.5, color=BODY, fontweight="bold")
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
    ax.set_yticks(range(len(rows)), [f"{y} {r.title()}" + ("  (severe)" if s else "") for r, y, _, _, s in rows[::-1]], fontsize=10.5)
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
    what = "reanalysis, activation day" if basis == "ra" else "forecasts, first issue that activates"
    ax.set_title(f"{SEASON_TITLE[season]}: {what}", fontsize=12.5)
    return save(fig, f"timing_{season}_{basis}")


# --------------------------------------------------- figure: one season at all seven points
def fig_season(R, season, year):
    a, b = WINDOW[season]
    A, B = pd.Timestamp(f"{year}-{a}"), pd.Timestamp(f"{year}-{b}")
    lv = L.levels()
    models = [m for m in MODELS[season] if year in R[("juba", season)]["models"][m]["ra_years"]]
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.6), sharex=True, sharey=True)
    seen = {}
    for row, river in zip(axes, RIVERS):
        rec = R[(river, season)]
        rp, _ = cfg(river, season)
        g3 = L.gauge_levels(river, season, 3)
        for ax, st in zip(row, TRIGGER_STATIONS[river]):
            sg = lv[(lv.station == st) & (lv.date >= A) & (lv.date <= B)].set_index("date").level_m.dropna()
            if len(sg) and not np.isnan(g3[st]):
                (h,) = ax.plot(sg.index, 100 * sg / g3[st], color=INK, lw=2.0, zorder=4)
                seen.setdefault("gauge, % of its 1-in-3 level", h)
            for m in models:
                d = L.daily(m)
                d = d[(d.station == st) & (d.date >= A) & (d.date <= B)].set_index("date").discharge
                (h,) = ax.plot(d.index, 100 * d / L.model_threshold(m, st, season, rp), color=COL[m], lw=1.5)
                seen.setdefault(f"{SHORT[m]}, % of its trigger level", h)
                ra = rec["models"][m]["ra"].get(year)
                if ok(ra):
                    ax.axvline(ra, color=COL[m], lw=1.1, ls="--")
            bd = rec["bench"].get(year)
            if ok(bd):
                ax.axvline(bd, color=INK, lw=1.1, ls=":")
            ax.axhline(100, color=BODY, lw=0.8, ls=":")
            ax.set_title(f"{river.title()}: {NAME[st]}", fontsize=10)
            ax.set_xlim(A, B)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.tick_params(axis="x", labelsize=8)
            style_ax(ax, grid="y")
    # the empty eighth panel lists the dates
    ax = axes[1][3]
    ax.axis("off")
    lines = [f"{season.title()} {year}", ""]
    for river in RIVERS:
        rec = R[(river, season)]
        lines.append(f"{river.title()}: reanalysis / forecast")
        for m in models:
            fc = rec["models"][m]["fc"].get(year) if year in rec["models"][m]["fc_years"] else None
            fc_txt = fmt(fc) if year in rec["models"][m]["fc_years"] else "no archive"
            lines.append(f"  {SHORT[m]}: {fmt(rec['models'][m]['ra'].get(year))} / {fc_txt}")
        lines.append(f"  2nd gauge at 1-in-3: {fmt(rec['bench'].get(year))}")
        lines.append("")
    ax.text(-0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=7.6, color=BODY, linespacing=1.3, transform=ax.transAxes)
    seen["dashed: activation day on the reanalysis"] = Line2D([], [], color=BODY, ls="--")
    seen["dotted: second gauge over its 1-in-3"] = Line2D([], [], color=INK, ls=":")
    axes[0][0].set_ylabel("% of own level")
    axes[1][0].set_ylabel("% of own level")
    axes[0][0].set_ylim(0, 420)
    fig.legend(list(seen.values()), list(seen.keys()), ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.03))
    fig.subplots_adjust(wspace=0.08, hspace=0.3, bottom=0.14)
    return save(fig, f"season_{season}_{year}")


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
    for ax, met, title in zip(axes, ["POD", "FAR", "F1"], ["Severe seasons caught (share)", "False alarms (share of activations)", "F1"]):
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


def score_table(R, season, basis):
    span = SCORE_RA if basis == "ra" else SCORE_FC
    models = [m for m in MODELS[season] if basis == "ra" or m in FC_SPAN]
    out = [f"<table class='sc'><thead><tr><th>{BASIS_TITLE[basis].title()}, {rng(span)}</th><th>severe seasons caught</th><th>ordinary floods covered</th>"
           f"<th>activations with no gauge flood</th><th>severe seasons missed</th><th>POD</th><th>FAR</th><th>F1</th></tr></thead><tbody>"]
    for river in RIVERS:
        rec = R[(river, season)]
        sev = rec["severe"] & set(range(span[0], span[1] + 1))
        out.append(f"<tr class='riv'><td colspan='8'>{river.title()}: severe seasons {yrs(sev)}</td></tr>")
        for m in models:
            s = score(rec, m, basis, span)
            yr = years_of(rec, m, basis)
            tag = "" if yr >= set(range(span[0], span[1] + 1)) else f"<span class='y'>archive {min(yr)} to {max(yr)}</span>"
            out.append(f"<tr><td>{SHORT[m]}{tag}</td><td>{len(s['caught'])} of {len(s['severe'])}<span class='y'>{yrs(s['caught'])}</span></td>"
                       f"<td>{len(s['ordinary'])}<span class='y'>{yrs(s['ordinary'])}</span></td>"
                       f"<td>{len(s['false'])}<span class='y'>{yrs(s['false'])}</span></td>"
                       f"<td>{len(s['missed_severe'])}<span class='y'>{yrs(s['missed_severe'])}</span></td>"
                       f"<td{shade(s['POD'])}>{num(s['POD'])}</td><td{shade(s['FAR'], True)}>{num(s['FAR'])}</td><td{shade(s['F1'])}>{num(s['F1'])}</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def v4_full_archive(R, season):
    """GloFAS v4's forecast record over its whole archive, 2003 to 2023, as one sentence per river."""
    parts = []
    for river in RIVERS:
        s = score(R[(river, season)], "glofas_v4", "fc", FC_SPAN["glofas_v4"])
        parts.append(f"{river.title()}: activates in {yrs(s['fires'])}; severe seasons {yrs(s['severe'])}, of which it catches {yrs(s['caught'])}; "
                     f"activations with no gauge flood {yrs(s['false'])}.")
    return " ".join(parts)


def after_2023(R, season, y):
    fired_r = [r.title() for r in RIVERS if y in R[(r, season)]["models"][NEW]["ra"]]
    fc_r = [r.title() for r in RIVERS if y in R[(r, season)]["models"][NEW]["fc"]]
    outc = []
    for r in RIVERS:
        rec = R[(r, season)]
        outc.append(f"{r.title()} {'severe' if y in rec['severe'] else 'flood' if y in rec['flood'] else 'nothing' if y in rec['assess'] else 'no gauge record'}")
    return (f"reanalysis {('activates on the ' + ' and '.join(fired_r)) if fired_r else 'no activation'}; "
            f"forecasts {('activate on the ' + ' and '.join(fc_r)) if fc_r else 'no activation'}; gauges: {', '.join(outc)}")


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
    T = {f"{s}_{b}": score_table(R, s, b) for s in ("deyr", "gu") for b in ("ra", "fc")}

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
article {{ padding:8px 44px 40px; max-width:1000px; }}
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:22px; color:var(--n9); margin:44px 0 8px; padding-top:18px; border-top:1px solid #e9eded; }}
article h3 {{ font-family:'Merriweather',Georgia,serif; font-size:16px; color:var(--n9); margin:28px 0 6px; }}
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.6; }}
.muted {{ color:var(--n7); }}
figure {{ margin:10px 0 4px; }}
figure img {{ width:100%; height:auto; display:block; }}
.key {{ background:var(--n05); border:1px solid #e7ecec; border-radius:5px; padding:14px 16px; margin:18px 0 6px; font-size:13.5px; }}
.key p {{ margin:0 0 6px; font-size:13.5px; }}
.key p:last-child {{ margin:0; }}
.key b {{ color:var(--n9); }}
table.sc {{ border-collapse:collapse; width:100%; font-size:12.5px; margin:10px 0 18px; }}
.sc th, .sc td {{ border-bottom:1px solid #e9eded; padding:5px 7px; text-align:left; vertical-align:top; }}
.sc th {{ background:var(--n05); color:var(--n7); font-weight:500; font-size:11.5px; }}
.sc th:first-child {{ color:var(--n9); font-weight:700; font-size:12.5px; }}
.sc tr.riv td {{ background:#fff; color:var(--n9); font-weight:500; padding-top:10px; border-bottom:1px solid #d9dfdf; }}
.sc td:first-child {{ white-space:nowrap; }}
.sc .y {{ display:block; font-size:11px; color:var(--n7); }}
.read {{ font-size:13px; color:var(--n7); margin:4px 0 0; }}
@media (max-width:640px) {{ article {{ padding:8px 22px 30px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / Google's new river model</p>
    <h1>Google's new river model</h1>
    <p>Google's November 2026 river model, read at the seven trigger points with the trigger's own rules, next to
       the three sources the trigger has used: GloFAS v4 (runs Deyr today), the current Google model (runs Gu today)
       and GloFAS v5 (the Deyr design). Each model is shown on two records, its reanalysis and its forecast archive,
       and scored against the SWALIM gauges. The live Flood Hub feed does not run the new model yet.</p>
  </header>
  <article>
    <div class="key">
      <p><b>The trigger.</b> A window <b>activates</b> when three of the four Juba gauges, or two of the three Shabelle gauges, are
         over their own level on the same day in the model being read. Levels are each model's own return levels (Weibull plotting position on its
         1999 to 2023 seasonal maxima): Juba 1-in-{TRIGGER_CONFIG[('juba','deyr')]['rp']} and Shabelle 1-in-{TRIGGER_CONFIG[('shabelle','deyr')]['rp']} in Deyr,
         Juba 1-in-{TRIGGER_CONFIG[('juba','gu')]['rp']} and Shabelle 1-in-{TRIGGER_CONFIG[('shabelle','gu')]['rp']} in Gu.</p>
      <p><b>Two records per model.</b> <b>Reanalysis</b>: the model's own record of the past, read day by day.
         <b>Forecasts</b>: the first issue that activates at leads 1 to 7 days, which is what the live trigger reads.
         Archives differ: GloFAS v4 2003 to 2023 (ensemble median, about two issues a week), the current Google model 2016 to June 2023
         (one trace, daily; it holds Gu 2023 but not Deyr 2023), the new Google model 2016 to 2025 (one trace, daily), GloFAS v5 none. Scores are given over
         {rng(SCORE_RA)} on the reanalysis and {rng(SCORE_FC)} on the forecasts, the years every record covers.</p>
      <p><b>The gauges.</b> A river has a <b>flood</b> in a season when two of its SWALIM gauges reach their own 1-in-3 level, and a
         <b>severe</b> flood at 1-in-5 (levels fitted on 2000 to 2023). An activation in a season with no gauge flood is counted as a false alarm.
         Gauge data run to July 2026, so the new model's 2024 and 2025 seasons are scored on gauges too.</p>
    </div>

    <h2>Deyr</h2>
    <h3>Season by season</h3>
    <figure><img src="{F['bt_deyr_ra']}" alt="Deyr, reanalysis, season by season"></figure>
    <figure><img src="{F['bt_deyr_fc']}" alt="Deyr, forecasts, season by season"></figure>
    <h3>Scores</h3>
    {T['deyr_ra']}
    {T['deyr_fc']}
    <p class="read">GloFAS v4 over its whole forecast archive, 2003 to 2023. {v4_full_archive(R, 'deyr')}</p>
    <p class="read">The new model after 2023. Deyr 2024: {after_2023(R, 'deyr', 2024)}. Deyr 2025: {after_2023(R, 'deyr', 2025)}.</p>

    <h2>Gu</h2>
    <h3>Season by season</h3>
    <figure><img src="{F['bt_gu_ra']}" alt="Gu, reanalysis, season by season"></figure>
    <figure><img src="{F['bt_gu_fc']}" alt="Gu, forecasts, season by season"></figure>
    <h3>Scores</h3>
    {T['gu_ra']}
    {T['gu_fc']}
    <p class="read">GloFAS v4 over its whole forecast archive, 2003 to 2023. {v4_full_archive(R, 'gu')}</p>
    <p class="read">The new model after 2023. Gu 2024: {after_2023(R, 'gu', 2024)}. Gu 2025: {after_2023(R, 'gu', 2025)}.</p>

    <h2>Timing against the gauges</h2>
    <p>One row per flood season. The vertical line is the day the second gauge crossed its 1-in-3 level. Green side: the
       model activated before that day. Red side: after it. Grey column: no activation that season. The bar is the river's peak.</p>
    <h3>Deyr</h3>
    <figure><img src="{F['tm_deyr_ra']}" alt="Deyr timing, reanalysis"></figure>
    <figure><img src="{F['tm_deyr_fc']}" alt="Deyr timing, forecasts"></figure>
    <h3>Gu</h3>
    <figure><img src="{F['tm_gu_ra']}" alt="Gu timing, reanalysis"></figure>
    <figure><img src="{F['tm_gu_fc']}" alt="Gu timing, forecasts"></figure>

    <h2>Three seasons at all seven points</h2>
    <p>The gauge as a share of its 1-in-3 level and each model's reanalysis at the same point as a share of its own trigger
       level. Above the dotted 100% line is over the level. Lines above 400% are cut off. The last panel lists the dates on
       both records.</p>
    <figure><img src="{F['d23']}" alt="Deyr 2023, all points"></figure>
    <figure><img src="{F['d24']}" alt="Deyr 2024, all points"></figure>
    <figure><img src="{F['g24']}" alt="Gu 2024, all points"></figure>

    <h2>Station by station</h2>
    <p>Each point on its own against its own gauge, reanalysis, {rng(SCORE_RA)}: a severe season is the gauge over its own 1-in-5,
       a false alarm is the model over its level with the gauge under its 1-in-3.</p>
    <figure><img src="{F['st_deyr']}" alt="Deyr station scores"></figure>
    <figure><img src="{F['st_gu']}" alt="Gu station scores"></figure>

    <h2>The two Google models are on different scales</h2>
    <p>The trigger level at each point on the current Google model and on the new one, both fitted the same way.</p>
    <figure><img src="{F['lv']}" alt="Trigger levels, current against new"></figure>

    <h2>Notes</h2>
    <ul>
      <li>Source: Google Runoff Reanalysis and Reforecast, November 2026 release, public bucket, CC-BY-4.0. Reanalysis 1981 to 2025;
          reforecast 2016 to 2025 at leads 0 to 10 days. One HydroBasins outlet per station, chosen as the main-stem outlet within
          15 km with the largest 1-in-5 flow; the outlet ids are in <code>data.json</code> beside this page.</li>
      <li>Lead labelling differs between archives. GloFAS day n is the calendar day issue + (n - 1). The current Google archive's lead n is
          issue + n. The new archive's lead 0 is the day before the issue, and it is relabelled here so that day 1 is the issue day, as
          for GloFAS. "Leads 1 to 7" therefore spans the issue day to six days ahead for GloFAS and the new model, and one to seven
          days ahead for the current Google model, as in the rest of this site.</li>
      <li>The same-day rule is sensitive at the margin on the new model. In Juba Deyr 2006 and 2017 and Juba Gu 2013 and 2023 all
          four Juba outlets go over their levels within the season, by 1 to 4%, but never on the same day, so those seasons count
          as no activation. The lead window also matters: read at leads 0 to 6 (the window used for GloFAS and the new model) the
          current Google model does not activate in Gu 2023; read at leads 2 to 8 the new model's years are unchanged.</li>
      <li>Three severe seasons rest on a gauge reading equal to its level: Shabelle Deyr 2020 (Jowhar 5.30 m against a 5.30 m
          level), Shabelle Gu 2016 and Gu 2023 (Belet Weyne at its 8.30 m ceiling). The Juba Deyr 2023 peaks are capped at bank
          full at all four gauges, and Dollow's Deyr record has eight seasons, the minimum for a fit.</li>
      <li>Sample sizes are small: {len(range(SCORE_FC[0], SCORE_FC[1] + 1))} seasons on the forecast record, and one to five severe seasons per window. GloFAS v4 forecasts are an
          ensemble median; both Google archives are a single trace.</li>
      <li>The current Google model's forecast archive has not been verified to be the same model as the live Flood Hub feed, and the
          new model's scale is not the live feed's scale. 2026 is in no archive.</li>
      <li>Everything on this page is generated by <code>scripts/page_additions/build_google_model_2026_page.py</code>; the dates, levels and
          scores behind every chart are in <code>data.json</code>.</li>
    </ul>
  </article>
</div>
</body>
</html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("written", OUT / "index.html")
    print(pd.DataFrame(sc)[["season", "river", "model", "basis", "span", "caught", "false", "POD", "FAR", "F1"]].to_string(index=False))


if __name__ == "__main__":
    build()
