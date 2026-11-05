"""Build pages/google-model-2026/: Google's November 2026 river model read with the trigger's
own rules at the seven points, against the sources the trigger runs on and the gauges.
Two questions: would it have carried Deyr (and kept Gu), and when does it fire relative to
the rivers rising.

Inputs: the seven-station pulls on blob (ds-aa-som-floods/google_grrr_2026_11/, dev "projects";
or a local folder with the same file names via SOM_GRRR_2026_DIR) and the repo's processed data
(SOM_DATA_REPO when run from a worktree). Charts follow scripts/summary_figures.py.

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
from src.constants import (BODY, FAINT, GRID, INK, REFERENCE_GAUGE, SEASONS,  # noqa: E402
                           SOURCE_COLORS, TRIGGER_CONFIG, TRIGGER_STATIONS)
from src.plots import apply_chart_style, style_ax  # noqa: E402

apply_chart_style()
OUT = HERE.parents[1] / "pages" / "google-model-2026"
FIGS = OUT / "figures"
BLOB = "ds-aa-som-floods/google_grrr_2026_11"
NEW = "grrr_2026_11"
YEARS = list(range(1999, 2026))
SPAN = L.SPAN                      # 1999 to 2023, where every source overlaps the gauges
RIVERS = ["juba", "shabelle"]
STATIONS = TRIGGER_STATIONS["juba"] + TRIGGER_STATIONS["shabelle"]
RIVER_OF = {st: r for r, sts in TRIGGER_STATIONS.items() for st in sts}
NAME = L.NAME
MODELS = {"deyr": [NEW, "google_grrr", "glofas_v4", "glofas_v5"], "gu": [NEW, "google_grrr", "glofas_v4"]}
SHORT = {NEW: "Google, new", "glofas_v4": "GloFAS v4", "glofas_v5": "GloFAS v5", "google_grrr": "Google, current"}
COL = {**SOURCE_COLORS, NEW: "#0E8A7B"}
SEASON_TITLE = {"deyr": "Deyr, Oct to Dec", "gu": "Gu, Mar to May"}
CELL = {"hit_sev": "#123F5F", "hit": "#3E86BE", "false": "#F4A93B", "miss_sev": "#B34036",
        "miss": "#E3B4AF", "quiet": "#EDF1F4", "none": "#FFFFFF"}
WINDOW = {"deyr": ("09-20", "12-31"), "gu": ("02-15", "06-25")}
NL = chr(10)
ok = lambda x: x is not None and not pd.isna(x)
fmt = lambda d: d.strftime("%d %b") if ok(d) else "never"


# the new model, plugged into somlib's loaders ------------------------------------------
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
    """Long format as somlib expects. Google's lead 0 is the state on the day before the issue,
    so valid = issue + lead - 1 day: lead 1 is the issue day, as in the GloFAS archive."""
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


# the record ----------------------------------------------------------------------------
def cfg(river, season):
    c = TRIGGER_CONFIG[(river, season)]
    return c["rp"], c["n_req"]


def gauge_peaks(river, season):
    lv = L.levels()
    d = lv[lv.station.isin(TRIGGER_STATIONS[river]) & lv.date.dt.month.isin(SEASONS[season])].dropna(subset=["level_m"])
    out = {}
    for y, g in d.groupby(d.date.dt.year):
        if y in YEARS:
            out[y] = g.loc[g.level_m.idxmax(), "date"].normalize()
    return out


def record():
    """Per (river, season): gauge years and dates, and per model the first crossing / issue by year."""
    R = {}
    for season in ("deyr", "gu"):
        for river in RIVERS:
            fl, sev, assess = L.benchmark(river, season, span=YEARS)
            models = {}
            for m in MODELS[season]:
                span = YEARS if m == NEW else SPAN
                has_fc = m == NEW or (L.P / f"reforecast_{m}.parquet").exists()
                fc = L.first_issue_dates(m, river, season, *cfg(river, season), span=span) if has_fc else {}
                models[m] = dict(ra=L.first_crossing_dates(m, river, season, *cfg(river, season), span=span),
                                 fc={y: v[0] for y, v in fc.items()}, years=set(span))
            R[(river, season)] = dict(flood=fl, severe=sev, assess=assess,
                                      bench=L.gauge_crossings(river, season, 3, span=YEARS),
                                      peak=gauge_peaks(river, season), models=models)
    return R


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}.svg"


# figure: backtest grid, one row per model and river ------------------------------------
def fig_backtest(R, season):
    lines = []
    for river in RIVERS:
        rec = R[(river, season)]
        for m in MODELS[season]:
            lines.append((f"{river.title()}: {SHORT[m]}", set(rec["models"][m]["ra"]), rec["models"][m]["years"],
                          rec["flood"], rec["severe"], rec["assess"]))
    fig, ax = plt.subplots(figsize=(13.2, 0.55 * len(lines) + 2.4))
    for i, (name, fired, years, fl, sev, assess) in enumerate(lines[::-1]):
        tally = {"hit_sev": 0, "false": 0, "miss_sev": 0}
        for j, y in enumerate(YEARS):
            if y not in years or y not in assess:
                kind = "none"
            elif y in fired:
                kind = "hit_sev" if y in sev else "hit" if y in fl else "false"
            else:
                kind = "miss_sev" if y in sev else "miss" if y in fl else "quiet"
            if y in SPAN and kind in tally:
                tally[kind] += 1
            ax.add_patch(Rectangle((j - 0.46, i - 0.36), 0.92, 0.72, facecolor=CELL[kind],
                                   edgecolor="white" if kind != "none" else GRID, linewidth=0.8))
        n_sev = len(sev & set(SPAN) & years)
        ax.text(len(YEARS) + 0.2, i + 0.12, f"{tally['hit_sev']} of {n_sev} severe caught", va="center", fontsize=8.5, color=INK)
        ax.text(len(YEARS) + 0.2, i - 0.2, f"{tally['false']} false fires", va="center", fontsize=8.5, color="#B07A1E" if tally["false"] else FAINT)
    ax.axhline(len(MODELS[season]) - 0.5, color=INK, linewidth=1.0)
    ax.set_xlim(-0.6, len(YEARS) + 4.6)
    ax.set_ylim(-0.6, len(lines) - 0.4)
    ax.set_yticks(range(len(lines)), [ln[0] for ln in lines[::-1]], fontsize=9.5)
    ax.set_xticks(range(0, len(YEARS), 2), [str(y) for y in YEARS[::2]], fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [Rectangle((0, 0), 1, 1, facecolor=CELL["hit_sev"], label="fired, severe flood followed"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["hit"], label="fired, ordinary flood followed"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["false"], label="fired, nothing at the gauges"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss_sev"], label="severe flood missed"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss"], label="ordinary flood, not fired"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["quiet"], edgecolor=GRID, label="quiet season"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID, label="no record")]
    ax.legend(handles=H, ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.08))
    ax.set_title(f"{SEASON_TITLE[season]}: what each model would have done, season by season (tallies 1999 to 2023)")
    return save(fig, f"backtest_{season}")


# scores per window and model -----------------------------------------------------------
def scores(R):
    rows = []
    for season in ("deyr", "gu"):
        for river in RIVERS:
            rec = R[(river, season)]
            fl, sev, assess = rec["flood"] & set(SPAN), rec["severe"] & set(SPAN), rec["assess"] & set(SPAN)
            for m in MODELS[season]:
                f = set(rec["models"][m]["ra"]) & assess & set(SPAN)
                pod = len(f & sev) / len(sev) if sev else np.nan
                far = len(f - fl) / len(f) if f else np.nan
                prec = 1 - far if f else np.nan
                f1 = 2 * pod * prec / (pod + prec) if f and (pod + prec) > 0 else 0.0
                rows.append(dict(season=season, river=river, model=m, fires=len(f), hits_severe=len(f & sev),
                                 n_severe=len(sev), false=len(f - fl), POD=pod, FAR=far, F1=f1))
    return pd.DataFrame(rows)


def fig_scores(S, season):
    ms = MODELS[season]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 2.5))
    for ax, met, title in zip(axes, ["POD", "FAR", "F1"],
                              ["Severe seasons caught", "False fires", "F1: the two combined"]):
        mat = [[float(S[(S.season == season) & (S.river == r) & (S.model == m)][met].iloc[0]) for m in ms] for r in RIVERS]
        heat(ax, mat, [SHORT[m] for m in ms], [r.title() for r in RIVERS] if met == "POD" else ["", ""], reverse=(met == "FAR"))
        ax.set_title(title + "\n(dark = better)", fontsize=10)
    fig.subplots_adjust(wspace=0.08)
    return save(fig, f"scores_{season}")


# figure: timing against the gauges -----------------------------------------------------
def _timing_rows(R, season, need_fc=False):
    rows = []
    for y in sorted(set().union(*(R[(r, season)]["flood"] for r in RIVERS)), reverse=True):
        if need_fc and y < 2016:
            continue
        for river in RIVERS:
            rec = R[(river, season)]
            if y in rec["flood"] and y in rec["bench"]:
                rows.append((river, y, rec["bench"][y], rec["peak"].get(y), y in rec["severe"]))
    return rows


def _timing_axes(ax, rows, ms, xlab):
    ax.axvspan(-30, 0, color="#F9ECE9", zorder=0)
    ax.axvspan(0, 42, color="#EAF3EE", zorder=0)
    ax.axvspan(-42, -30, color="#F2F4F5", zorder=0)
    ax.text(21, len(rows) - 0.35, "fired BEFORE the gauges confirmed", ha="center", va="center", fontsize=9, color="#2F6B4A", fontweight="bold")
    ax.text(-15, len(rows) - 0.35, "fired AFTER", ha="center", va="center", fontsize=9, color="#9E4B3F", fontweight="bold")
    ax.text(-36, len(rows) - 0.35, "did not\nfire", ha="center", va="center", fontsize=8.5, color=BODY, fontweight="bold")
    ax.axvline(0, color=INK, linewidth=1.2, zorder=2)
    ax.axvline(-30, color="white", linewidth=1.5, zorder=1)
    for i in range(len(rows)):
        ax.axhline(i, color="white", linewidth=1.0, zorder=0.5)
    ax.set_yticks(range(len(rows)), [f"{y} {r.title()}" + ("  (severe)" if s else "") for r, y, _, _, s in rows[::-1]], fontsize=9.5)
    ax.set_xlim(-42, 42)
    ax.set_ylim(-0.6, len(rows) + 0.1)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40], ["30", "20", "10", "0", "10", "20", "30", "40"])
    ax.set_xlabel(xlab)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [Line2D([], [], marker="o", ls="", color=COL[m], markersize=8, label=SHORT[m]) for m in ms]
    H += [Line2D([], [], marker="|", ls="", color=INK, markersize=12, markeredgewidth=1.6, label="river peak")]
    ax.legend(handles=H, ncol=len(H), fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13))


def fig_timing(R, season):
    rows = _timing_rows(R, season)
    ms = MODELS[season]
    fig, ax = plt.subplots(figsize=(11.6, 0.52 * len(rows) + 2.8))
    _timing_axes(ax, rows, ms, "days before the second gauge crossed its 1-in-3 level")
    for i, (river, y, b, pk, sev) in enumerate(rows[::-1]):
        if ok(pk):
            ax.scatter((b - pk).days, i, marker="|", s=200, color=INK, linewidths=1.6, zorder=3)
        rec = R[(river, season)]
        for k, m in enumerate(ms):
            if y not in rec["models"][m]["years"]:
                continue
            dy = (k - (len(ms) - 1) / 2) * 0.2
            ra = rec["models"][m]["ra"].get(y)
            x = (b - ra).days if ok(ra) else -36 + (k - (len(ms) - 1) / 2) * 2.2
            x = max(min(x, 41), -29) if ok(ra) else x
            ax.scatter(x, i + dy, s=58, color=COL[m], edgecolors="white", linewidths=0.9, zorder=4)
    ax.set_title(f"{SEASON_TITLE[season]}: when each model's rule was met in its record, against the gauges")
    return save(fig, f"timing_{season}")


def fig_timing_fc(R, season):
    rows = _timing_rows(R, season, need_fc=True)
    ms = [m for m in MODELS[season] if m != "glofas_v5"]
    fig, ax = plt.subplots(figsize=(11.6, 0.52 * len(rows) + 2.8))
    _timing_axes(ax, rows, ms, "days before the second gauge crossed its 1-in-3 level")
    for i, (river, y, b, pk, sev) in enumerate(rows[::-1]):
        if ok(pk):
            ax.scatter((b - pk).days, i, marker="|", s=200, color=INK, linewidths=1.6, zorder=3)
        rec = R[(river, season)]
        for k, m in enumerate(ms):
            if y not in rec["models"][m]["years"]:
                continue
            dy = (k - (len(ms) - 1) / 2) * 0.2
            fc = rec["models"][m]["fc"].get(y)
            x = (b - fc).days if ok(fc) else -36 + (k - (len(ms) - 1) / 2) * 2.2
            x = max(min(x, 41), -29) if ok(fc) else x
            ax.scatter(x, i + dy, s=58, color=COL[m], edgecolors="white", linewidths=0.9, zorder=4)
    ax.set_title(f"{SEASON_TITLE[season]}: the first forecast issue that met the rule (leads 1 to 7 days), 2016 on")
    return save(fig, f"timing_fc_{season}")


# figure: one season in full, model against gauge, per river ----------------------------
def fig_season(R, season, year, models):
    """All seven points for one season: the gauge as a share of its 1-in-3, each model as a share
    of its own trigger level, the day each model's rule fired and the day the second gauge crossed."""
    a, b = WINDOW[season]
    A, B = pd.Timestamp(f"{year}-{a}"), pd.Timestamp(f"{year}-{b}")
    lv = L.levels()
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.4), sharex=True, sharey=True)
    seen = {}
    grid = {"juba": axes[0], "shabelle": axes[1]}
    for river in RIVERS:
        rec = R[(river, season)]
        rp, _ = cfg(river, season)
        g3 = L.gauge_levels(river, season, 3)
        for ax, st in zip(grid[river], TRIGGER_STATIONS[river]):
            sg = lv[(lv.station == st) & (lv.date >= A) & (lv.date <= B)].set_index("date").level_m.dropna()
            if len(sg) and not np.isnan(g3[st]):
                (h,) = ax.plot(sg.index, 100 * sg / g3[st], color=INK, lw=2.0, zorder=4)
                seen.setdefault("gauge, % of its 1-in-3 level", h)
            for m in models:
                if year not in rec["models"][m]["years"]:
                    continue
                d = L.daily(m)
                d = d[(d.station == st) & (d.date >= A) & (d.date <= B)].set_index("date").discharge
                thr = L.model_threshold(m, st, season, rp)
                (h,) = ax.plot(d.index, 100 * d / thr, color=COL[m], lw=1.5)
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
        if river == "shabelle":
            grid[river][3].axis("off")
            fired = "; ".join(f"{SHORT[m]} {fmt(rec['models'][m]['ra'].get(year))}" for m in models if year in rec["models"][m]["years"])
            txt = f"{season.title()} {year}" + NL + NL + "Shabelle rule met:" + NL + fired.replace("; ", NL) + NL + NL + "second gauge at 1-in-3:" + NL + fmt(rec["bench"].get(year))
            grid[river][3].text(0, 0.95, txt, va="top", fontsize=9, color=BODY, transform=grid[river][3].transAxes)
            rj = R[("juba", season)]
            firedj = NL.join(f"{SHORT[m]} {fmt(rj['models'][m]['ra'].get(year))}" for m in models if year in rj["models"][m]["years"])
            txtj = "Juba rule met:" + NL + firedj + NL + NL + "second gauge at 1-in-3:" + NL + fmt(rj["bench"].get(year))
            grid[river][3].text(0, 0.42, txtj, va="top", fontsize=9, color=BODY, transform=grid[river][3].transAxes)
    seen["dashed: the day the model's rule fired"] = Line2D([], [], color=BODY, ls="--")
    seen["dotted: second gauge over its 1-in-3"] = Line2D([], [], color=INK, ls=":")
    axes[0, 0].set_ylabel("% of own level")
    axes[1, 0].set_ylabel("% of own level")
    axes[0, 0].set_ylim(0, 420)
    fig.legend(list(seen.values()), list(seen.keys()), ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.03))
    fig.subplots_adjust(wspace=0.08, hspace=0.3, bottom=0.14)
    return save(fig, f"season_{season}_{year}")


# figure: scale change ------------------------------------------------------------------
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
            ax.text(max(o, nw) * 1.1, y, f"{nw / o:.1f}×", va="center", fontsize=8.5, color=FAINT)
        ax.set_xscale("log")
        ax.set_xlim(80, 3000)
        ax.set_title(SEASON_TITLE[season])
        ax.set_xlabel("trigger level at the point, m³/s (log scale)")
        style_ax(ax, grid="x")
    axes[0].set_yticks(ys, [NAME[s] for s in STATIONS])
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.subplots_adjust(wspace=0.06)
    return save(fig, "levels")


# per-station skill ---------------------------------------------------------------------
def station_scores():
    rows = []
    lv = L.levels()
    for season in ("deyr", "gu"):
        for st in STATIONS:
            r = RIVER_OF[st]
            rp, _ = cfg(r, season)
            lev5, lev3 = L.gauge_levels(r, season, 5)[st], L.gauge_levels(r, season, 3)[st]
            s = lv[lv.station == st].set_index("date").level_m.dropna()
            s = s[s.index.month.isin(SEASONS[season]) & s.index.year.isin(SPAN)]
            am = s.groupby(s.index.year).max()
            if np.isnan(lev5) or len(am) < 8:
                continue
            sev, fl, yrs = set(am[am >= lev5].index), set(am[am >= lev3].index), set(am.index)
            for m in MODELS[season]:
                x = L.season_series(m, st, season)
                mx = x.groupby(x.index.year).max()
                f = set(mx[mx >= L.model_threshold(m, st, season, rp)].index) & yrs
                pod = len(f & sev) / len(sev) if sev else np.nan
                far = len(f - fl) / len(f) if f else np.nan
                prec = 1 - far if f else np.nan
                rows.append(dict(season=season, station=st, model=m, POD=pod, FAR=far,
                                 F1=2 * pod * prec / (pod + prec) if f and (pod + prec) > 0 else 0.0))
    return pd.DataFrame(rows)


def fig_stations(S, season):
    ms = MODELS[season]
    sts = [s for s in STATIONS if s in set(S[S.season == season].station)]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 0.5 * len(sts) + 1.8))
    for ax, met, title in zip(axes, ["POD", "FAR", "F1"], ["Severe seasons caught", "False alarms", "F1"]):
        mat = [[float(S[(S.season == season) & (S.station == st) & (S.model == m)][met].iloc[0]) for m in ms] for st in sts]
        heat(ax, mat, [SHORT[m] for m in ms], [NAME[s] for s in sts] if met == "POD" else [""] * len(sts), reverse=(met == "FAR"))
        ax.set_title(title + "\n(dark = better)", fontsize=10)
    fig.subplots_adjust(wspace=0.08)
    return save(fig, f"stations_{season}")


# html ----------------------------------------------------------------------------------
def build():
    R = record()
    S = scores(R)
    ST = station_scores()
    F = {}
    for s in ("deyr", "gu"):
        F[f"bt_{s}"] = fig_backtest(R, s)
        F[f"sc_{s}"] = fig_scores(S, s)
        F[f"tm_{s}"] = fig_timing(R, s)
        F[f"tf_{s}"] = fig_timing_fc(R, s)
        F[f"st_{s}"] = fig_stations(ST, s)
    F["d23"] = fig_season(R, "deyr", 2023, [NEW, "google_grrr", "glofas_v4"])
    F["d24"] = fig_season(R, "deyr", 2024, [NEW])
    F["g24"] = fig_season(R, "gu", 2024, [NEW])
    F["lv"] = fig_levels()

    g = lambda s, r, m: S[(S.season == s) & (S.river == r) & (S.model == m)].iloc[0]
    jd, jd4, jd5 = g("deyr", "juba", NEW), g("deyr", "juba", "glofas_v4"), g("deyr", "juba", "glofas_v5")
    sd, sd4 = g("deyr", "shabelle", NEW), g("deyr", "shabelle", "glofas_v4")
    jdc, sdc = g("deyr", "juba", "google_grrr"), g("deyr", "shabelle", "google_grrr")
    jg, jgc = g("gu", "juba", NEW), g("gu", "juba", "google_grrr")
    sg, sgc = g("gu", "shabelle", NEW), g("gu", "shabelle", "google_grrr")

    def lead_med(season, m):
        ds = []
        for river in RIVERS:
            rec = R[(river, season)]
            for y, b in rec["bench"].items():
                ra = rec["models"][m]["ra"].get(y)
                if ok(ra) and y in rec["models"][m]["years"]:
                    ds.append((b - ra).days)
        return (int(np.median(ds)), len(ds)) if ds else (0, 0)

    ld_new, ld_v4 = lead_med("deyr", NEW), lead_med("deyr", "glofas_v4")
    lg_new, lg_c = lead_med("gu", NEW), lead_med("gu", "google_grrr")

    def aft(s, y):
        fired = [r.title() for r in RIVERS if y in R[(r, s)]["models"][NEW]["ra"]]
        outc = ("severe" if any(y in R[(r, s)]["severe"] for r in RIVERS) else
                "ordinary flood" if any(y in R[(r, s)]["flood"] for r in RIVERS) else "nothing")
        return (f"fired on the {' and '.join(fired)}" if fired else "did not fire") + f", gauges: {outc}"

    d = lambda r, s, m, y: fmt(R[(r, s)]["models"][m]["ra"].get(y))
    bd = lambda r, s, y: fmt(R[(r, s)]["bench"].get(y))

    dump = {}
    for (river, season), rec in R.items():
        dump[f"{river}_{season}"] = dict(
            flood=sorted(rec["flood"]), severe=sorted(rec["severe"]),
            second_gauge_1in3={str(y): v.strftime("%Y-%m-%d") for y, v in rec["bench"].items()},
            models={m: dict(reanalysis={str(y): v.strftime("%Y-%m-%d") for y, v in mm["ra"].items()},
                            forecast={str(y): v.strftime("%Y-%m-%d") for y, v in mm["fc"].items()}) for m, mm in rec["models"].items()})
    (OUT / "data.json").write_text(json.dumps(dict(outlets=_read_json("chosen_outlets.json"), record=dump,
                                                   scores=json.loads(S.to_json(orient="records")),
                                                   station_scores=json.loads(ST.to_json(orient="records"))), indent=1))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Google's new river model: Somalia Riverine Flood Trigger</title>
<meta name="description" content="Google's November 2026 river model at the seven Somalia trigger points: would it have carried Deyr, does it keep Gu, and when does it fire relative to the river gauges.">
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
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:21px; color:var(--n9); margin:40px 0 8px; }}
article h3 {{ font-family:'Merriweather',Georgia,serif; font-size:16px; color:var(--n9); margin:26px 0 6px; }}
article p {{ font-size:14.5px; color:var(--n8); line-height:1.6; }}
.muted {{ color:var(--n7); }}
figure {{ margin:10px 0 4px; }}
figure img {{ width:100%; height:auto; display:block; }}
.answer {{ background:var(--b05); border-left:5px solid var(--b5); padding:12px 16px; border-radius:4px; margin:14px 0; font-size:14.5px; line-height:1.6; }}
.answer.flag {{ background:#fff7e8; border-left-color:#e0a04b; }}
.read {{ font-size:13px; color:var(--n7); margin:4px 0 0; }}
@media (max-width:640px) {{ article {{ padding:8px 22px 30px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / Google's new river model</p>
    <h1>Google's new river model</h1>
    <p>Google has a new river model (November 2026). We read it at the seven trigger points with the trigger's
       own rules and asked two things: would it have carried Deyr instead of GloFAS, and when does it fire
       compared with the rivers actually rising. The live Flood Hub feed does not run this model yet.</p>
  </header>
  <article>
    <p class="read" style="margin-top:22px">A season is <b>severe</b> when two gauges on the river reach their own 1-in-5 level, a <b>flood</b> at 1-in-3.
       A model <b>fires</b> when the trigger rule is met in its record: three of four Juba points, or two of three Shabelle
       points, over their own levels on the same day. 1999 to 2023 for every source, to 2025 for the new model.</p>

    <h2>Would Google have carried Deyr?</h2>
    <div class="answer flag"><b>Juba: no.</b> It catches {jd.hits_severe} of the {jd.n_severe} severe seasons; GloFAS v4 catches {jd4.hits_severe}, v5 {jd5.hits_severe}, the Google model running today {jdc.hits_severe}.
       It fires {jd.false} times with nothing at the gauges, v4 {jd4.false} times, today's Google {jdc.false} times.<br>
       <b>Shabelle: yes.</b> It catches {sd.hits_severe} of {sd.n_severe} against v4's {sd4.hits_severe}, with {sd.false} false fire against {sd4.false}, and it sees the 2023 flood that v4 never did.
       Today's Google model matches it on the Shabelle ({sdc.hits_severe} of {sdc.n_severe}, {sdc.false} false fires): it misses 2020 where the new one misses 2006.</div>
    <figure><img src="{F['bt_deyr']}" alt="Deyr backtest grid"></figure>
    <p class="read">One row per model and river, one cell per season: did it fire, and what did the gauges do.</p>
    <figure><img src="{F['sc_deyr']}" alt="Deyr scores"></figure>

    <h2>Does it keep Gu?</h2>
    <div class="answer"><b>Yes.</b> It catches Juba {jg.hits_severe} of {jg.n_severe} and Shabelle {sg.hits_severe} of {sg.n_severe} severe seasons, against {jgc.hits_severe} and {sgc.hits_severe}
       for the Google model running today, with {jg.false + sg.false} false fires against {jgc.false + sgc.false}. It is the only source that fires in Gu 2023.</div>
    <figure><img src="{F['bt_gu']}" alt="Gu backtest grid"></figure>
    <figure><img src="{F['sc_gu']}" alt="Gu scores"></figure>

    <h2>Timing against the gauges</h2>
    <p>Every season in which a river flooded. Zero is the day the second gauge crossed its 1-in-3 level.
       Green side: the model fired before the gauges confirmed. Red side: after. Grey column: it never fired that season.</p>
    <figure><img src="{F['tm_deyr']}" alt="Deyr timing"></figure>
    <p class="read">Deyr median: the new model fires {ld_new[0]:+d} days before the gauges confirm (over {ld_new[1]} river-seasons), GloFAS v4 {ld_v4[0]:+d} (over {ld_v4[1]}).
       GloFAS is earlier on the Juba. On the Shabelle in 2023 the new model is the one that fires at all.</p>
    <figure><img src="{F['tm_gu']}" alt="Gu timing"></figure>
    <p class="read">Gu median: the new model {lg_new[0]:+d} days (over {lg_new[1]}), the Google model running today {lg_c[0]:+d} (over {lg_c[1]}).</p>
    <h3>The same, read from the forecasts</h3>
    <p>What the operator would have seen: the first forecast issue on which the rule was met at leads 1 to 7 days.
       Forecast archives start in 2016 for both Google models; GloFAS v5 has none.</p>
    <figure><img src="{F['tf_deyr']}" alt="Deyr forecast timing"></figure>
    <figure><img src="{F['tf_gu']}" alt="Gu forecast timing"></figure>

    <h3>Three seasons in full</h3>
    <p>All seven points. The gauge as a share of its 1-in-3 level, each model at the same point as a share of its
       own trigger level. Above the dotted 100% line means over the level. Lines above 400% are cut off.</p>
    <figure><img src="{F['d23']}" alt="Deyr 2023"></figure>
    <p class="read">Deyr 2023. Juba: GloFAS v4 fired {d('juba', 'deyr', 'glofas_v4', 2023)}, the new model {d('juba', 'deyr', NEW, 2023)}, the second gauge crossed {bd('juba', 'deyr', 2023)}.
       Shabelle: only the new model fired, {d('shabelle', 'deyr', NEW, 2023)}; the gauges crossed {bd('shabelle', 'deyr', 2023)}.</p>
    <figure><img src="{F['d24']}" alt="Deyr 2024"></figure>
    <p class="read">Deyr 2024: the new model {aft('deyr', 2024)}. The Shabelle gauges crossed {bd('shabelle', 'deyr', 2024)}.</p>
    <figure><img src="{F['g24']}" alt="Gu 2024"></figure>
    <p class="read">Gu 2024: the new model {aft('gu', 2024)}. 2025: Deyr {aft('deyr', 2025)}; Gu {aft('gu', 2025)}.</p>

    <h2>Station by station</h2>
    <p>The same scores for each point on its own, against that gauge's own levels.</p>
    <figure><img src="{F['st_deyr']}" alt="Deyr station scores"></figure>
    <figure><img src="{F['st_gu']}" alt="Gu station scores"></figure>

    <h2>The numbers are on a different scale</h2>
    <p>At the same points the new model's flows are several times the current Google model's. Every Google-based
       level, the Gu triggers and the 1-in-6 line quoted at Luuq included, would have to be refitted.</p>
    <figure><img src="{F['lv']}" alt="Levels, current against new"></figure>

    <p class="muted" style="font-size:12.5px;margin-top:30px">Source: Google Runoff Reanalysis and Reforecast, November 2026 release
       (reanalysis 1981 to 2025, reforecast 2016 to 2025 at leads 0 to 10 days), one HydroBasins outlet per station. Levels are fitted on each
       model's own 1999 to 2023 seasonal maxima. Gauges: SWALIM, to July 2026. Forecast issues use leads 1 to 7 days, lead 1 being the issue day.
       The dates behind the charts are in <code>data.json</code> beside this page; the seven-station extracts are on the team's storage.</p>
  </article>
</div>
</body>
</html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("written", OUT / "index.html")
    print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    build()
