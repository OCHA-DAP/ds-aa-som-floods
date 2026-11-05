"""Build pages/google-model-2026/: Google's November 2026 river model read with the trigger's
own rules at the seven stations, next to the sources the trigger runs on, on two records side by
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
COL = {**SOURCE_COLORS, NEW: "#6A4C93"}
MODELS = {"deyr": [NEW, "google_grrr", "glofas_v4", "glofas_v5"], "gu": [NEW, "google_grrr", "glofas_v4", "glofas_v5"]}
# each model's record: reanalysis years and forecast-archive years
RA_SPAN = {NEW: (1999, 2025), "google_grrr": (1999, 2023), "glofas_v4": (1999, 2023), "glofas_v5": (1999, 2023)}
FC_SPAN = {NEW: (2016, 2025), "google_grrr": (2016, 2023), "glofas_v4": (2003, 2023)}   # v5: no archive
# the current Google archive ends 2023-06-30: it holds Gu 2023 but no Deyr 2023
FC_END = {("google_grrr", "deyr"): 2022}
TRIGGER_RP = 3                   # one return period at every station, on every source, and at the gauges
BENCH_RP = 3                     # the scores: a gauge flood is two gauges over their own 1-in-3
TIMING_RP = 3                    # the timing and the ranking: the second gauge crossing its 1-in-3
MIN_DAYS = 30                    # a gauge season counts only with this many readings, as somlib.assessable_years
SCORE_RA = (1999, 2023)          # every model and the gauges overlap here
SCORE_FC = (2016, 2023)          # every forecast archive overlaps here
YEARS = list(range(1999, 2026))
CELL = {"hit": "#1B5E20", "false": "#B08D57", "miss": "#3D3D3D", "quiet": "#E3E8EC", "none": "#FFFFFF"}
WINDOW = {"deyr": ("10-01", "12-31"), "gu": ("03-01", "05-31")}
# the 14 anticipatory-action districts by river (as scripts/page_additions/floodscan_districts.py)
DISTRICTS = {"juba": ["SO2605", "SO2606", "SO2602", "SO2703", "SO2701", "SO2702"],
             "shabelle": ["SO2001", "SO2002", "SO2003", "SO2101", "SO2103", "SO2302", "SO2305", "SO2301"]}
BASIS_TITLE = {"ra": "reanalysis", "fc": "forecasts"}
GAUGE_C = "#7C8794"   # gauge marks on the timelines; the peak tick is INK on top of them
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
# gauge levels at the gauge's own precision (centimetres): a reading equal to its fitted level is
# a crossing, which float noise in the Weibull interpolation (5.080000000000001) otherwise denies
_gl0 = L.gauge_levels
L.gauge_levels = lambda river, season, rp: _gl0(river, season, rp).round(2)


# ---------------------------------------------------------------------------- record
def cfg(river, season):
    """The vote the trigger uses at each river, read at TRIGGER_RP on every source."""
    return TRIGGER_RP, TRIGGER_CONFIG[(river, season)]["n_req"]


def gauge_peaks(river, season):
    lv = L.levels()
    d = lv[lv.station.isin(TRIGGER_STATIONS[river]) & lv.date.dt.month.isin(SEASONS[season])].dropna(subset=["level_m"])
    return {y: g.loc[g.level_m.idxmax(), "date"].normalize() for y, g in d.groupby(d.date.dt.year) if y in YEARS}


def first_issue_in_season(model, river, season, rp, n_req, span):
    """somlib.first_issue_dates with one more condition: the issue itself is dated within the
    season months, so nothing before 1 October (Deyr) or 1 March (Gu) counts."""
    rf = L.reforecast(model)
    rf = rf[rf.station.isin(TRIGGER_STATIONS[river]) & rf.leadtime_days.between(1, 7)
            & rf.valid_time.dt.month.isin(SEASONS[season]) & rf.issued_time.dt.month.isin(SEASONS[season])]
    med = rf.groupby(["issued_time", "valid_time", "station"]).discharge.median().unstack("station")
    thr = L.model_thresholds(model, river, season, rp)
    cols = [c for c in thr.index if c in med.columns and not np.isnan(thr[c])]
    votes = (med[cols] >= thr[cols]).sum(axis=1)
    hit = votes[votes >= n_req].reset_index()
    hit.columns = ["issued", "valid", "votes"]
    hit["year"] = hit["valid"].dt.year
    out = {}
    for y, g in hit.groupby("year"):
        if y in span:
            g = g.sort_values(["issued", "valid"])
            out[y] = g.iloc[0]["issued"].normalize()
    return out


def record():
    """Per (river, season): the gauge benchmark and, per model, the first day the rule is met in
    the reanalysis (ra) and the first forecast issue that meets it (fc), by year."""
    R = {}
    for season in ("deyr", "gu"):
        for river in RIVERS:
            fl, _, assess = L.benchmark(river, season, span=YEARS)
            sev = set(L.gauge_crossings(river, season, BENCH_RP, span=YEARS))   # the benchmark flood, at BENCH_RP
            models = {}
            for m in MODELS[season]:
                ra_years = list(range(RA_SPAN[m][0], RA_SPAN[m][1] + 1))
                ra = L.first_crossing_dates(m, river, season, *cfg(river, season), span=ra_years)
                fc, fc_years = {}, []
                if m in FC_SPAN:
                    fc_years = list(range(FC_SPAN[m][0], FC_END.get((m, season), FC_SPAN[m][1]) + 1))
                    fc = first_issue_in_season(m, river, season, *cfg(river, season), span=fc_years)
                models[m] = dict(ra=ra, fc=fc, ra_years=set(ra_years), fc_years=set(fc_years))
            R[(river, season)] = dict(flood=fl, severe=sev, assess=assess,
                                      bench=L.gauge_crossings(river, season, TIMING_RP, span=YEARS),
                                      peak=gauge_peaks(river, season), models=models)
    return R


def activations_of(rec, m, basis):
    return set(rec["models"][m][basis])


def years_of(rec, m, basis):
    return rec["models"][m][f"{basis}_years"]


def score(rec, m, basis, span):
    yrs_ = set(range(span[0], span[1] + 1)) & years_of(rec, m, basis) & rec["assess"]
    f = activations_of(rec, m, basis) & yrs_
    sev = rec["severe"] & yrs_
    pod = len(f & sev) / len(sev) if sev else np.nan
    far = len(f - sev) / len(f) if f else np.nan
    prec = 1 - far if f else np.nan
    if np.isnan(pod):
        f1 = np.nan
    elif f and (pod + prec) > 0:
        f1 = 2 * pod * prec / (pod + prec)
    else:
        f1 = 0.0
    return dict(model=m, basis=basis, span=(f"{min(yrs_)} to {max(yrs_)}" if yrs_ else rng(span)), n_years=len(yrs_), threshold_reached=sorted(f), severe=sorted(sev),
                caught=sorted(f & sev), false=sorted(f - sev), missed_severe=sorted(sev - f), POD=pod, FAR=far, F1=f1)


def save(fig, name, tight=True):
    FIGS.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.3) if tight else {}
    fig.savefig(FIGS / f"{name}.svg", format="svg", **kw)
    if os.environ.get("SOM_FIG_PNG_DIR"):  # raster copies for review, outside the repo
        png = Path(os.environ["SOM_FIG_PNG_DIR"])
        png.mkdir(parents=True, exist_ok=True)
        fig.savefig(png / f"{name}.png", format="png", dpi=130, **kw)
    plt.close(fig)
    return f"figures/{name}.svg"


# ------------------------------------------------------- figure: season-by-season grid
def fig_backtest(R, season):
    """One grid per season. Under each source a reanalysis row and, where an archive exists, a
    forecast row. The tallies on the right count the scored seasons: every season with a gauge
    record for the reanalysis, 2016 to 2023 (dotted lines) for the forecasts."""
    models = MODELS[season]
    lines = []            # (label, river, model, basis)
    for river in RIVERS:
        for m in models:
            lines.append((f"{river.title()}: {SHORT[m]}, reanalysis", river, m, "ra"))
            if m in FC_SPAN:
                lines.append((f"{river.title()}: {SHORT[m]}, forecasts", river, m, "fc"))
    fig, ax = plt.subplots(figsize=(14.8, 0.5 * len(lines) + 2.8))
    n_juba = sum(1 for ln in lines if ln[1] == "juba")
    for i, (name, river, m, basis) in enumerate(lines[::-1]):
        rec = R[(river, season)]
        span = SCORE_RA if basis == "ra" else SCORE_FC
        f, yr = activations_of(rec, m, basis), years_of(rec, m, basis)
        for j, y in enumerate(YEARS):
            if y not in yr or y not in rec["assess"]:
                kind = "none"
            elif y in f:
                kind = "hit" if y in rec["severe"] else "false"
            else:
                kind = "miss" if y in rec["severe"] else "quiet"
            ax.add_patch(Rectangle((j - 0.46, i - 0.36), 0.92, 0.72, facecolor=CELL[kind],
                                   edgecolor="white" if kind != "none" else GRID, linewidth=0.8))
        sc_ = score(rec, m, basis, span)
        n_sev, n_f = len(sc_["severe"]), len(sc_["false"])
        sev_txt = f"reached in {len(sc_['caught'])} of {n_sev} gauge {'flood' if n_sev == 1 else 'floods'}" if sc_["severe"] else "no gauge flood in these seasons"
        ax.text(len(YEARS) + 0.2, i + 0.14, sev_txt, va="center", fontsize=9, color=INK)
        ax.text(len(YEARS) + 0.2, i - 0.2, f"reached in {n_f} {'season' if n_f == 1 else 'seasons'} with no gauge flood", va="center", fontsize=9,
                color="#B07A1E" if sc_["false"] else FAINT)
    ax.axhline(len(lines) - n_juba - 0.5, color=INK, linewidth=1.0)
    for edge_year in (SCORE_FC[0] - 0.5, SCORE_FC[1] + 0.5):
        ax.axvline(YEARS.index(int(edge_year + 0.5)) - 0.5, color=INK, linewidth=0.9, ls=":", zorder=4)
    ax.text(-0.5, -0.62, f" forecast rows are tallied between the dotted lines ({rng(SCORE_FC)}; Google, current in Deyr to 2022); reanalysis rows over every season with a gauge record to {SCORE_RA[1]}",
            ha="left", va="top", fontsize=8.5, color=FAINT, bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=6)
    ax.set_xlim(-0.6, len(YEARS) + 9.6)
    ax.set_ylim(-1.1, len(lines) - 0.4)
    ax.set_yticks(range(len(lines)), [ln[0] for ln in lines[::-1]], fontsize=9.5)
    ax.set_xticks(range(0, len(YEARS), 2), [str(y) for y in YEARS[::2]], fontsize=9.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [Rectangle((0, 0), 1, 1, facecolor=CELL["hit"], label="threshold reached, gauge flood"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["false"], label="threshold reached, no gauge flood"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss"], label="gauge flood, threshold not reached"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["quiet"], edgecolor=GRID, label="no gauge flood, threshold not reached"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID, label="no model record or no gauge record")]
    ax.legend(handles=H, ncol=3, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.45, -0.06))
    ax.set_title(f"{SEASON_TITLE[season]}: reanalysis, and forecasts (first issue reaching the threshold, leads 1 to 7 days)", fontsize=12, loc="left")
    fig.subplots_adjust(left=0.17, right=0.995, top=0.95, bottom=0.14)
    return save(fig, f"backtest_{season}", tight=False)


# ---------------------------------------------------------- figure: timing vs the gauges
def _timing_seasons(season):
    """The gauges' N_RANK worst seasons per river (the benchmark tables), with their gauge rank."""
    out = []
    for river in RIVERS:
        _, Rk = rank_table(season, river)
        g = Rk["gauge"].dropna()
        for y in top_seasons(Rk):
            out.append((river, int(y), int(g.loc[y])))
    return out


def fig_timing(R, season):
    """Days between the threshold being reached and the second gauge crossing its 1-in-3, as
    labelled bars: one row per season and source, the reanalysis as a solid bar and the forecasts
    as a lighter bar beneath it. Left of zero: before the gauges; right: after."""
    models = MODELS[season]
    seasons = _timing_seasons(season)
    rows = []            # (river, year, rank, model) in drawing order, top first; model None = header line
    for river, y, rank in seasons:
        rows.append((river, y, rank, None))
        for m in models:
            rows.append((river, y, rank, m))
    H = 0.36             # bar height
    fig, ax = plt.subplots(figsize=(11.6, 0.56 * len(rows) + 2.4))
    ymax = len(rows)
    ax.axvspan(-40, 0, color="#F3F5F7", zorder=0)
    ax.axvspan(0, 40, color="#EBEEF1", zorder=0)
    ax.axvline(0, color=INK, linewidth=1.2, zorder=3)
    ax.text(-20, ymax + 0.35, "threshold reached before the second gauge crossed its 1-in-3", ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(20, ymax + 0.35, "threshold reached after", ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
    n_at_rank = {}
    for river, y, rank in seasons:
        n_at_rank[(river, rank)] = n_at_rank.get((river, rank), 0) + 1
    for i, (river, y, rank, m) in enumerate(rows):
        yc = ymax - 1 - i
        if m is None:
            word = "largest" if rank == 1 else str(rank) + ("nd" if rank == 2 else "rd" if rank == 3 else "th") + " largest"
            tie = ", tied" if n_at_rank[(river, rank)] > 1 else ""
            no_b = "" if ok(R[(river, season)]["bench"].get(y)) else "; second gauge never over its 1-in-3, no timing"
            ax.text(-39, yc, f"{river.title()} {y}   ({word} gauge flood{tie}{no_b})", ha="left", va="center", fontsize=10, color=INK, fontweight="bold", zorder=5)
            if i:
                ax.axhline(yc + 0.5, color="white", linewidth=3, zorder=2)
            continue
        rec = R[(river, season)]
        b = rec["bench"].get(y)
        mm = rec["models"][m]
        for k, (basis, alpha) in enumerate((("ra", 1.0), ("fc", 0.45))):
            yb = yc + (0.2 if k == 0 else -0.2)
            if not ok(b):
                continue
            have = y in mm[f"{basis}_years"]
            d = mm[basis].get(y)
            tag = "reanalysis" if basis == "ra" else "forecasts"
            if not have:
                ax.text(1.2, yb, f"{tag}: no archive" if basis == "fc" else "outside the model record", va="center", fontsize=8.5, color=FAINT)
                continue
            if not ok(d):
                ax.text(1.2, yb, f"{tag}: not reached", va="center", fontsize=8.5, color=FAINT)
                continue
            n = (d - b).days            # negative: before the gauges (left), positive: after (right)
            x = max(min(n, 38), -38)
            ax.barh(yb, x if n else 0.4, height=H, color=COL[m], alpha=alpha, edgecolor="white", linewidth=0.5, zorder=4)
            lab = f"{abs(n)} d {'before' if n < 0 else 'after'}" if n else "same day"
            if abs(n) > 38:
                lab += " (bar cut)"
            if abs(n) > 22:   # long bar: the label sits inside it
                ax.text(x - (1.2 if n >= 0 else -1.2), yb, lab, va="center", ha="right" if n >= 0 else "left", fontsize=8.5,
                        color="white" if alpha == 1.0 else INK, zorder=5)
            else:
                ax.text(x + (1.2 if n >= 0 else -1.2), yb, lab, va="center", ha="left" if n >= 0 else "right", fontsize=8.5, color=BODY, zorder=5)
        ax.text(-38, yc, SHORT[m], ha="left", va="center", fontsize=9, color=BODY, zorder=5)
    ax.set_yticks([])
    ax.set_ylim(-0.7, ymax + 0.8)
    ax.set_xlim(-40, 40)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30], ["30 before", "20", "10", "0", "10", "20", "30 after"])
    ax.set_xlabel("days between the threshold being reached and the second gauge crossing its 1-in-3", fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 1.0, "solid bar: reanalysis.   light bar: first forecast issue reaching the threshold (leads 1 to 7 days).   colour: the source named at the left",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color=BODY)
    ax.set_title(f"{SEASON_TITLE[season]}: days before or after the second gauge crossed its 1-in-3, five largest gauge floods per river", fontsize=11, pad=40, loc="left")
    fig.subplots_adjust(left=0.04, right=0.98)
    return save(fig, f"timing_{season}")


def timing_stats(R, season, basis, m):
    """Days before the second gauge crossed its 1-in-3 (positive = before), over the five largest
    gauge floods per river where the source reached the threshold: (mean, median, n reached, n seasons)."""
    ds, n_all = [], 0
    for river, y, rank in _timing_seasons(season):
        rec = R[(river, season)]
        mm = rec["models"][m]
        if y not in mm[f"{basis}_years"] or y not in rec["bench"]:
            continue
        if y > (SCORE_FC[1] if basis == "fc" else SCORE_RA[1]) or (basis == "fc" and y < SCORE_FC[0]):
            continue
        n_all += 1
        d = mm[basis].get(y)
        if ok(d):
            ds.append((rec["bench"][y] - d).days)
    if not ds:
        return (None, None, 0, n_all)
    return (float(np.mean(ds)), float(np.median(ds)), len(ds), n_all)


def timing_summary_html(R):
    half_up = lambda v: int(np.floor(v + 0.5))
    fmt_d = lambda v: ("same day" if half_up(v) == 0 else f"{abs(half_up(v))} d {'before' if half_up(v) > 0 else 'after'}")
    head = ("<table class='data'><thead><tr><th rowspan='2'>Source</th><th colspan='2'>Deyr</th><th colspan='2'>Gu</th></tr>"
            "<tr><th>reanalysis</th><th>forecasts</th><th>reanalysis</th><th>forecasts</th></tr></thead><tbody>")
    out = [head]
    for m in [NEW, "google_grrr", "glofas_v4", "glofas_v5"]:
        cells = [f"<td>{SHORT[m]}</td>"]
        for season in ("deyr", "gu"):
            for basis in ("ra", "fc"):
                if basis == "fc" and m not in FC_SPAN:
                    cells.append("<td class='muted'>no archive</td>")
                    continue
                mean, med, n, n_all = timing_stats(R, season, basis, m)
                if n == 0:
                    cells.append(f"<td class='muted'>not reached in {n_all} seasons</td>")
                else:
                    cells.append(f"<td>mean {fmt_d(mean)}<span class='y'>median {fmt_d(med)}; reached in {n} of {n_all} seasons</span></td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def fig_timeline(R, season, year):
    """Calendar timeline for one season, Juba in the upper panel and Shabelle in the lower.
    Gauge rows: the first day the gauge crossed its 1-in-3 (circle), its 1-in-5 (diamond) and its
    peak (tick). Source rows: the day the threshold was reached on the reanalysis (filled) and the
    first forecast issue that reached it (hollow)."""
    a, b = WINDOW[season]
    A, B = pd.Timestamp(f"{year}-{a}"), pd.Timestamp(f"{year}-{b}")
    lv = L.levels()
    models = MODELS[season]
    import season_charts as sc
    n_rows = {r: len(TRIGGER_STATIONS[r]) + len(models) for r in RIVERS}
    has_exp = year <= 2023
    ratios = ([n_rows["juba"], 3, n_rows["shabelle"], 3] if has_exp else [n_rows["juba"], n_rows["shabelle"]])
    fig, axes = plt.subplots(len(ratios), 1, figsize=(11.6, 0.72 * sum(ratios) + 2.6), sharex=True,
                             gridspec_kw=dict(height_ratios=ratios))
    row_axes = [axes[0], axes[2]] if has_exp else [axes[0], axes[1]]
    for ax, river in zip(row_axes, RIVERS):
        rec = R[(river, season)]
        sts = TRIGGER_STATIONS[river]
        g3, g5 = L.gauge_levels(river, season, 3), L.gauge_levels(river, season, 5)
        n = n_rows[river]
        ylab = []
        # gauge rows
        for i, st in enumerate(sts):
            yc = n - 1 - i
            ax.axhline(yc, color=GRID, linewidth=0.8, zorder=0)
            s = lv[(lv.station == st) & (lv.date.dt.year == year) & lv.date.dt.month.isin(SEASONS[season])].dropna(subset=["level_m"]).sort_values("date")
            edge = lambda d: "left" if (d - A).days < 12 else "right" if (B - d).days < 12 else "center"
            notes = []
            row_label = f"{NAME[st]} gauge"
            if len(s) < 30:
                notes.append("no gauge record" if s.empty else "gauge record too short")
            else:
                pk = s.loc[s.level_m.idxmax(), "date"]
                ax.scatter(pk, yc, s=560, marker="|", color=INK, linewidths=2.2, zorder=6)
                row_label = f"{NAME[st]} gauge, peak {pk:%d %b}"
                first = {}
                for lab, lev in (("1-in-3", g3[st]), ("1-in-5", g5[st])):
                    c = s[s.level_m >= lev] if not np.isnan(lev) else s.iloc[0:0]
                    first[lab] = c.date.iloc[0] if len(c) else None
                for lev, marker, lab in ((g3[st], "o", "1-in-3"), (g5[st], "D", "1-in-5")):
                    if np.isnan(lev):
                        continue
                    # both levels crossed on the same day: show the 1-in-5 only
                    if lab == "1-in-3" and first["1-in-5"] is not None and first["1-in-3"] == first["1-in-5"]:
                        continue
                    c = s[s.level_m >= lev]
                    if len(c):
                        d = c.date.iloc[0]
                        ax.scatter(d, yc, s=70 if marker == "o" else 60, marker=marker, color=GAUGE_C, edgecolors="white", linewidths=0.8, zorder=5)
                        dx = pd.Timedelta(days=1.5) if d == pk else pd.Timedelta(0)
                        ax.text(d + dx, yc + (0.22 if marker == "o" else -0.22), f"{d:%d %b}", ha="left" if d == pk else edge(d), va="bottom" if marker == "o" else "top", fontsize=8.5, color=BODY)
                if first["1-in-3"] is None:
                    notes.append("not over its 1-in-3")
            if notes:
                ax.text(B - pd.Timedelta(days=1), yc, "; ".join(notes), va="center", ha="right", fontsize=8.5, color=FAINT)
            ylab.append((yc, row_label))
        # source rows, set half a row below the gauge rows
        for j, m in enumerate(models):
            yc = n - 1 - len(sts) - j - 0.5
            ax.axhline(yc, color=GRID, linewidth=0.8, zorder=0)
            mm = rec["models"][m]
            notes = []
            same_day = (year in mm["ra_years"] and year in mm["fc_years"] and ok(mm["ra"].get(year)) and ok(mm["fc"].get(year))
                        and mm["ra"].get(year) == mm["fc"].get(year))
            for basis, filled in (("ra", True), ("fc", False)):
                if year not in mm[f"{basis}_years"]:
                    notes.append("no forecast archive" if basis == "fc" else "outside the model record")
                    continue
                d = mm[basis].get(year)
                if not ok(d):
                    notes.append("not reached" if basis == "ra" else "forecasts: not reached")
                    continue
                if same_day:
                    if filled:
                        ax.scatter(d, yc, s=80, marker="s", color=COL[m], edgecolors="white", linewidths=0.8, zorder=4)
                        ax.scatter(d, yc, s=150, marker="s", facecolors="none", edgecolors=COL[m], linewidths=1.6, zorder=4)
                        ax.text(d, yc + 0.2, d.strftime("%d %b") + " (reanalysis and forecasts)", ha="left" if (d - A).days < 12 else "right" if (B - d).days < 12 else "center", va="bottom", fontsize=8.5, color=COL[m])
                    continue
                if filled:
                    ax.scatter(d, yc, s=80, marker="s", color=COL[m], edgecolors="white", linewidths=0.8, zorder=4)
                    ax.text(d, yc + 0.2, d.strftime("%d %b"), ha="left" if (d - A).days < 12 else "right" if (B - d).days < 12 else "center", va="bottom", fontsize=8.5, color=COL[m])
                else:
                    ax.scatter(d, yc, s=100, marker="s", facecolors="none", edgecolors=COL[m], linewidths=1.6, zorder=4)
                    ax.text(d, yc - 0.2, d.strftime("%d %b"), ha="left" if (d - A).days < 12 else "right" if (B - d).days < 12 else "center", va="top", fontsize=8.5, color=COL[m])
            if notes == ["outside the model record", "no forecast archive"]:
                notes = ["outside the model record"]
            if notes == ["not reached", "forecasts: not reached"]:
                notes = ["not reached on either record"]
            if notes == ["not reached", "no forecast archive"]:
                notes = ["not reached; no forecast archive"]
            if notes:
                ax.text(B - pd.Timedelta(days=1), yc, "; ".join(notes), va="center", ha="right", fontsize=8.5, color=FAINT)
            ylab.append((yc, SHORT[m]))
        ax.axhline(n - len(sts) - 0.75, color=INK, linewidth=0.9, zorder=1)
        ax.set_yticks([y for y, _ in ylab], [l for _, l in ylab], fontsize=10)
        ax.set_ylim(-1.2, n - 0.3)
        ax.set_xlim(A - pd.Timedelta(days=4), B)
        state = ("gauge flood" if year in rec["severe"] else "no gauge flood" if year in rec["assess"] else "no gauge record")
        ax.set_title(f"{river.title()} ({state})", fontsize=11.5, loc="left")
        ax.tick_params(length=0, labelbottom=True, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    if has_exp:
        ex = sc.exposure()
        ex = ex[(ex.valid_date >= A) & (ex.valid_date <= B)]
        fd, n_d = sc.flood_date(season, year)
        colr = "#6B7683"
        for axe, river_d in ((axes[1], "juba"), (axes[3], "shabelle")):
            e = ex[ex.pcode.isin(DISTRICTS[river_d])].groupby("valid_date")["sum"].sum()
            axe.fill_between(e.index, e.values, color=colr, alpha=0.15, lw=0)
            axe.plot(e.index, e.values, color=colr, lw=1.4)
            axe.set_ylim(0, max(float(e.max()), 1.0) * 1.4)    # room above the peak for the note
            if fd is None:
                axe.text(A + pd.Timedelta(days=1), axe.get_ylim()[1] * 0.96, f" fewer than three of the 14 districts over their 1-in-5 exposure ({n_d} district{'s' if n_d != 1 else ''})", va="top", ha="left", fontsize=8, color=INK)
            if fd is not None:
                axe.axvline(fd, color=INK, lw=1.1, ls=":", zorder=1)
                axe.text(fd, axe.get_ylim()[1] * 0.96, f" dotted line: the third of the 14 districts over its 1-in-5 exposure, {fd:%d %b}", va="top", ha="left", fontsize=8, color=INK)
            axe.set_title(f"Flood exposure, {len(DISTRICTS[river_d])} {river_d.title()} districts (FloodScan)", fontsize=10.5, loc="left")
            axe.set_ylabel("people in flooded area", color=BODY, fontsize=9)
            axe.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k" if v else "0"))
            axe.tick_params(length=0, labelsize=8.5)
            for sp in axe.spines.values():
                sp.set_visible(False)
            axe.grid(axis="y", color=GRID, linewidth=0.8)
            axe.set_axisbelow(True)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    Hh = [Line2D([], [], marker="o", ls="", color=GAUGE_C, markersize=7, label="gauge over its 1-in-3"),
          Line2D([], [], marker="D", ls="", color=GAUGE_C, markersize=6, label="gauge over its 1-in-5"),
          Line2D([], [], marker="|", ls="", color=INK, markersize=14, markeredgewidth=2.2, label="gauge peak"),
          Line2D([], [], marker="s", ls="", color=INK, markersize=8, label="threshold reached, reanalysis (source colour)"),
          Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=9, markeredgewidth=1.6, label="first forecast issue reaching the threshold")]
    fig.legend(handles=Hh, ncol=5, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 0.03), frameon=False)
    fig.suptitle(f"{season.title()} {year}", x=0.02, ha="left", fontsize=13)
    fig.subplots_adjust(hspace=0.35, top=0.92, bottom=0.12, right=0.97, left=0.16)
    return save(fig, f"timeline_{season}_{year}")


# -------------------------------------------------------- figure: seasons ranked by peak
RANK_COLS = [("gauge", None), (NEW, "ra"), (NEW, "fc"), ("google_grrr", "ra"), ("google_grrr", "fc"),
             ("glofas_v4", "ra"), ("glofas_v4", "fc"), ("glofas_v5", "ra")]
N_RANK = 5
RANK_RP = TIMING_RP   # each station's peak is read as a share of its own 1-in-3 level on that source


def _fc_peaks(m, st, season):
    rf = L.reforecast(m)
    rf = rf[(rf.station == st) & rf.leadtime_days.between(1, 7) & rf.valid_time.dt.month.isin(SEASONS[season])]
    med = rf.groupby(["issued_time", "valid_time"]).discharge.median().reset_index()
    years = set(range(FC_SPAN[m][0], FC_END.get((m, season), FC_SPAN[m][1]) + 1))
    med = med[med.issued_time.dt.month.isin(SEASONS[season])]
    mx = med.groupby(med.valid_time.dt.year).discharge.max()
    return mx[mx.index.isin(years)]


def season_index(season, river, source, basis):
    """Per year, the second-highest station's season peak as a share of that station's own
    1-in-RANK_RP level on the same source: the level the river's second gauge reached. Forecast peaks are read against the
    model's reanalysis levels, as the trigger reads them."""
    cols = {}
    if source == "gauge":
        lv = L.levels()
        lev = L.gauge_levels(river, season, RANK_RP)
        for st in TRIGGER_STATIONS[river]:
            if np.isnan(lev[st]):
                continue
            g = lv[(lv.station == st) & lv.date.dt.month.isin(SEASONS[season])].set_index("date").level_m.dropna()
            gm = g.groupby(g.index.year).max()[g.groupby(g.index.year).size() >= MIN_DAYS]
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


def top_seasons(Rk):
    """The gauges' seasons ranked within N_RANK, ties included, in a stable order."""
    g = Rk["gauge"].dropna().sort_index().sort_values(kind="stable")
    return [int(y) for y, r in g.items() if r <= N_RANK]


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
        top = top_seasons(Rk)
        srcs = [c for c in Rk.columns if c != "gauge" and c[0] in MODELS[season]]
        all_top, few_top = [], []
        for y in top:
            r = Rk.loc[y, srcs].dropna()
            if len(r) < 3:
                continue
            k = int((r <= top_n).sum())
            if k == len(r):
                all_top.append(f"{int(y)} ({len(r)} records)")
            elif k * 2 < len(r):
                few_top.append(f"{int(y)} ({k} of {len(r)} records)")
        out[river] = (all_top, few_top)
    return out


# --------------------------------------------- figure: correlation with the gauge, by station
CORR_RP = 3   # a 1-in-3 season: the gauge over its own 1-in-3, or the model over its own


def best_lag_rho(model_series, gauge_series, lags=range(-10, 31), min_obs=60):
    """Highest Spearman rank correlation over lags, pairing the model on day t with the gauge on
    day t + lag (by date, so gaps between seasons do not shift the pairing). A positive lag means
    the model rises before the gauge. Returns (rho, lag, n pairs at that lag)."""
    best = (np.nan, None, 0)
    for lag in lags:
        g = gauge_series.copy()
        g.index = g.index - pd.Timedelta(days=lag)
        j = pd.concat([model_series, g], axis=1, join="inner").dropna()
        if len(j) < min_obs:
            continue
        r = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
        if pd.notna(r) and (np.isnan(best[0]) or r > best[0]):
            best = (float(r), int(lag), int(len(j)))
    return best


def correlations():
    """Per season, station and model: rank correlation at the best lag over the 1-in-3 seasons,
    and over every season with a gauge record for reference (rho_all_seasons)."""
    lv = L.levels()
    rows = []
    for season in ("deyr", "gu"):
        for st in STATIONS:
            river = RIVER_OF[st]
            g = lv[(lv.station == st)].set_index("date").level_m.dropna().sort_index()
            g = g[g.index.month.isin(SEASONS[season]) & g.index.year.isin(range(SCORE_RA[0], SCORE_RA[1] + 1))]
            size = g.groupby(g.index.year).size()
            with_record = set(size[size >= MIN_DAYS].index)
            g = g[g.index.year.isin(with_record)]
            g3 = L.gauge_levels(river, season, CORR_RP)[st]
            gmax = g.groupby(g.index.year).max()
            g_ev = set(gmax[gmax >= g3].index) if not np.isnan(g3) else set()
            for m in [NEW, "google_grrr", "glofas_v4", "glofas_v5"]:
                x = L.season_series(m, st, season)
                m3 = L.model_threshold(m, st, season, CORR_RP)
                xmax = x.groupby(x.index.year).max()
                m_ev = set(xmax[xmax >= m3].index) & with_record if not np.isnan(m3) else set()
                ev = g_ev | m_ev
                rho, lag, n = best_lag_rho(x[x.index.year.isin(ev)], g[g.index.year.isin(ev)]) if ev else (np.nan, None, 0)
                rho_all, lag_all, n_all = best_lag_rho(x, g) if len(g) else (np.nan, None, 0)
                rows.append(dict(season=season, station=st, model=m, rho=rho, lag_days=lag, n_days=n, n_seasons=len(ev),
                                 n_gauge_events=len(g_ev), n_model_events=len(m_ev), seasons=sorted(int(y) for y in ev),
                                 rho_all_seasons=rho_all, lag_all_seasons=lag_all, n_seasons_with_record=len(with_record)))
    return pd.DataFrame(rows)


def fig_corr(CR):
    ms = [NEW, "google_grrr", "glofas_v4", "glofas_v5"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 0.56 * len(STATIONS) + 2.3))
    for ax, season in zip(axes, ("deyr", "gu")):
        sub = lambda st, m: CR[(CR.season == season) & (CR.station == st) & (CR.model == m)].iloc[0]
        mat = [[float(sub(st, m).rho) for m in ms] for st in STATIONS]
        heat(ax, mat, [SHORT[m].replace(", ", chr(10)) for m in ms], [NAME[s] for s in STATIONS] if season == "deyr" else [""] * len(STATIONS),
             vmin=0.0, vmax=1.0, cmap="Greens")
        for i, st in enumerate(STATIONS):
            for j, m in enumerate(ms):
                r = sub(st, m)
                if not np.isnan(r.rho):
                    ax.text(j, i + 0.3, f"{int(r.n_seasons)} seasons", ha="center", va="center", fontsize=6.5,
                            color="white" if r.rho > 0.62 else BODY)
        ax.set_title(f"{SEASON_TITLE[season]}\n(dark = high)", fontsize=10)
    fig.suptitle("Rank correlation between each model's daily flow and the gauge level, at the lag with the highest agreement,\nseasons in which the gauge or the model went over its own 1-in-3, with a gauge record, to 2023",
                 x=0.02, ha="left", fontsize=11)
    fig.subplots_adjust(wspace=0.08, top=0.8)
    return save(fig, "correlation")


# ------------------------------------------------------------ figure: scale of the levels
# ------------------------------------------------------------- figure: station by station
def station_scores():
    rows = []
    lv = L.levels()
    span = set(range(SCORE_RA[0], SCORE_RA[1] + 1))
    for season in ("deyr", "gu"):
        for st in STATIONS:
            r = RIVER_OF[st]
            rp, _ = cfg(r, season)
            lev5 = L.gauge_levels(r, season, BENCH_RP)[st]
            s = lv[lv.station == st].set_index("date").level_m.dropna()
            s = s[s.index.month.isin(SEASONS[season]) & s.index.year.isin(span)]
            am = s.groupby(s.index.year).max()[s.groupby(s.index.year).size() >= MIN_DAYS]
            if np.isnan(lev5) or len(am) < 8:
                continue
            sev, have = set(am[am >= lev5].index), set(am.index)
            for m in MODELS[season]:
                x = L.season_series(m, st, season)
                mx = x.groupby(x.index.year).max()
                f = set(mx[mx >= L.model_threshold(m, st, season, rp)].index) & have
                pod = len(f & sev) / len(sev) if sev else np.nan
                far = len(f - sev) / len(f) if f else np.nan
                prec = 1 - far if f else np.nan
                f1 = np.nan if np.isnan(pod) else (2 * pod * prec / (pod + prec) if f and (pod + prec) > 0 else 0.0)
                rows.append(dict(season=season, station=st, model=m, n_years=len(have), n_severe=len(sev), POD=pod, FAR=far, F1=f1))
    return pd.DataFrame(rows)


def fig_stations(S, season):
    ms = MODELS[season]
    sts = [s for s in STATIONS if s in set(S[S.season == season].station)]
    nsev = {st: int(S[(S.season == season) & (S.station == st)].n_severe.iloc[0]) for st in sts}
    ylab = [f"{NAME[s]} ({nsev[s]} flood {'season' if nsev[s] == 1 else 'seasons'})" for s in sts]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 0.5 * len(sts) + 2.3))
    for ax, met, title in zip(axes, ["POD", "FAR", "F1"], ["POD", "FAR", "F1"]):
        mat = [[float(S[(S.season == season) & (S.station == st) & (S.model == m)][met].iloc[0]) for m in ms] for st in sts]
        heat(ax, mat, [SHORT[m].replace(", ", chr(10)) for m in ms], ylab if met == "POD" else [""] * len(sts), cmap="Greys" if met == "FAR" else "Greens", vmin=-0.12, vmax=1.0)
        ax.set_title(title + "\n(dark = high)", fontsize=9.5)
    fig.suptitle(f"{SEASON_TITLE[season]}, reanalysis, seasons with a gauge record to 2023, each station's model against its gauge", x=0.02, ha="left", fontsize=11)
    fig.subplots_adjust(wspace=0.25, top=0.78)
    return save(fig, f"stations_{season}")


# ------------------------------------------------------------------------------ html
def shade(v, reverse=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    x = 1 - v if reverse else v
    return f' style="background:rgba(30,121,95,{0.08 + 0.42 * x:.2f})"'


def num(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def score_both(R, seasons, m, basis, span):
    """One score over both rivers (and over the seasons given): a year is reached when the
    threshold is reached on either river, and a gauge flood when either river has one; the
    universe is the scored years with a gauge record on either river."""
    pairs = [(r, se) for se in seasons for r in RIVERS]
    recs = {k: R[k] for k in pairs}
    yrs = set(range(span[0], span[1] + 1))
    own = {k: yrs & years_of(rec, m, basis) & rec["assess"] for k, rec in recs.items()}
    universe = set.union(*own.values())
    f = set.union(*[activations_of(rec, m, basis) & own[k] for k, rec in recs.items()])
    sev = set.union(*[rec["severe"] & own[k] for k, rec in recs.items()])
    pod = len(f & sev) / len(sev) if sev else np.nan
    far = len(f - sev) / len(f) if f else np.nan
    prec = 1 - far if f else np.nan
    f1 = np.nan if np.isnan(pod) else (2 * pod * prec / (pod + prec) if f and (pod + prec) > 0 else 0.0)
    return dict(POD=pod, FAR=far, F1=f1, n_years=len(universe), reached=sorted(f), floods=sorted(sev), caught=sorted(f & sev), false=sorted(f - sev))


def both_rivers_table(R):
    cols = [("Deyr, both rivers", ("deyr",)), ("Gu, both rivers", ("gu",)), ("Either season, either river", ("deyr", "gu"))]
    head = ("<table class='data'><thead><tr><th rowspan='2'>Source</th><th rowspan='2'>Record</th>"
            + "".join(f"<th colspan='3'>{c}</th>" for c, _ in cols) + "</tr><tr>" + "<th>POD</th><th>FAR</th><th>F1</th>" * len(cols) + "</tr></thead><tbody>")
    out = [head]
    for m in [NEW, "google_grrr", "glofas_v4", "glofas_v5"]:
        for basis, span in (("ra", SCORE_RA), ("fc", SCORE_FC)):
            if basis == "fc" and m not in FC_SPAN:
                continue
            cells = [f"<td>{SHORT[m]}</td><td class='muted'>{'reanalysis' if basis == 'ra' else 'forecasts'}</td>"]
            for _, seasons in cols:
                b = score_both(R, seasons, m, basis, span)
                cells.append(f"<td>{num(b['POD'])}<span class='y'>{len(b['caught'])} of {len(b['floods'])}</span></td>"
                             f"<td>{num(b['FAR'])}<span class='y'>{len(b['false'])} of {len(b['reached'])} {'year' if len(b['reached']) == 1 else 'years'}</span></td>"
                             f"<td>{num(b['F1'])}<span class='y'>{b['n_years']} years</span></td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def score_table(R, season):
    """One table per season: a row per source and record, the two rivers side by side."""
    hdr = ("<table class='data'><thead><tr><th rowspan='2'>Source</th><th rowspan='2'>Record</th>"
           "<th colspan='3'>Juba</th><th colspan='3'>Shabelle</th></tr>"
           "<tr><th>POD</th><th>FAR</th><th>F1</th><th>POD</th><th>FAR</th><th>F1</th></tr></thead><tbody>")
    out = [hdr]
    for m in MODELS[season]:
        for basis, span in (("ra", SCORE_RA), ("fc", SCORE_FC)):
            if basis == "fc" and m not in FC_SPAN:
                continue
            cells = [f"<td>{SHORT[m]}</td><td class='muted'>{'reanalysis' if basis == 'ra' else 'forecasts'}</td>"]
            for river in RIVERS:
                rec = R[(river, season)]
                s = score(rec, m, basis, span)
                cells.append(f"<td>{num(s['POD'])}<span class='y'>{len(s['caught'])} of {len(s['severe'])}</span></td>"
                             f"<td>{num(s['FAR'])}<span class='y'>{len(s['false'])} of {len(s['threshold_reached'])} {'season' if len(s['threshold_reached']) == 1 else 'seasons'}</span></td>"
                             f"<td>{num(s['F1'])}</td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def rank_html(season):
    """The gauges' N_RANK worst seasons per river and each source's own rank of them, as a table.
    Cell: rank on the reanalysis / rank on the forecasts. Shaded where the rank is within the top N_RANK."""
    models = [m for m in [NEW, "google_grrr", "glofas_v4", "glofas_v5"] if m in MODELS[season]]
    head = ("<table class='data rk'><thead><tr><th>Season</th><th>Second-highest gauge, % of its 1-in-3</th>"
            + "".join(f"<th>{SHORT[m]}<span class='y'>reanalysis / forecasts</span></th>" for m in models)
            + "</tr></thead><tbody>")
    out = [head]
    for river in RIVERS:
        P, Rk = rank_table(season, river)
        top = top_seasons(Rk)
        out.append(f"<tr class='riv'><td colspan='{2 + len(models)}'>{river.title()}</td></tr>")
        for y in top:
            cells = [f"<td>{int(y)}</td><td>{np.floor(1000 * P.loc[y, 'gauge']) / 10:.1f}%</td>"]
            for m in models:
                ra = Rk[(m, "ra")].loc[y] if (m, "ra") in Rk.columns else np.nan
                fc = Rk[(m, "fc")].loc[y] if (m, "fc") in Rk.columns else np.nan
                has_fc = (m, "fc") in Rk.columns and y in set(range(FC_SPAN[m][0], FC_END.get((m, season), FC_SPAN[m][1]) + 1))
                if np.isnan(ra):
                    cells.append("<td class='muted'>no record</td>")
                    continue
                fc_txt = (f"{int(fc)}" if not np.isnan(fc) else "no rank") if has_fc else "no archive"
                cls = " class='top'" if (ra <= N_RANK) else ""
                cells.append(f"<td{cls}>{int(ra)} <span class='muted'>/ {fc_txt}</span></td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def span_text(R):
    """The scored reanalysis seasons per river and season, from the gauge record itself."""
    parts = []
    for season in ("deyr", "gu"):
        for river in RIVERS:
            a = sorted(y for y in R[(river, season)]["assess"] if SCORE_RA[0] <= y <= SCORE_RA[1])
            parts.append((f"{river.title()} {season.title()}", f"{a[0]} to {a[-1]}" if a else "none"))
    by_span = {}
    for name, sp in parts:
        by_span.setdefault(sp, []).append(name)
    return "; ".join(f"{sp} for {', '.join(names)}" for sp, names in by_span.items())


def levels_note():
    """Google, new against Google, current at the TRIGGER_RP level, from the levels themselves."""
    ratios = {}
    for season in ("deyr", "gu"):
        for st in STATIONS:
            o, nw = L.model_threshold("google_grrr", st, season, TRIGGER_RP), L.model_threshold(NEW, st, season, TRIGGER_RP)
            if not (np.isnan(o) or np.isnan(nw)):
                ratios[(season, st)] = nw / o
    hi = [k for k, v in ratios.items() if v > 1]
    (season, st), top = max(ratios.items(), key=lambda kv: kv[1])
    where = f"{NAME[st]} in {season.title()}"
    if len(hi) == len(ratios):
        return f"Google, new levels are higher than Google, current's at every station, by the most at {where}."
    if not hi:
        return "Google, new levels are lower than Google, current's at every station."
    return (f"Google, new levels are higher than Google, current's at {len(hi)} of {len(ratios)} stations, by the most at {where}, "
            f"and lower at {', '.join(NAME[k[1]] + ' in ' + k[0].title() for k in ratios if k not in hi)}.")


def clean(x):
    """Plain JSON: numpy scalars to Python, NaN to null, nested."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    return x


def near_miss_note(R):
    """Seasons in which every Juba point of Google, new goes above its level within the season but the
    same-day rule is not met, with the smallest margin by which the lagging point cleared its level."""
    items = []
    for season in ("deyr", "gu"):
        rec = R[("juba", season)]
        rp, n_req = cfg("juba", season)
        d = L.daily(NEW)
        d = d[d.station.isin(TRIGGER_STATIONS["juba"]) & d.date.dt.month.isin(SEASONS[season])]
        thr = {st: L.model_threshold(NEW, st, season, rp) for st in TRIGGER_STATIONS["juba"]}
        for y in range(RA_SPAN[NEW][0], RA_SPAN[NEW][1] + 1):
            if y in rec["models"][NEW]["ra"]:
                continue
            x = d[d.date.dt.year == y]
            if x.empty:
                continue
            ratios = {st: x[x.station == st].discharge.max() / thr[st] for st in thr}
            if all(v >= 1 for v in ratios.values()):
                items.append(f"{season.title()} {y} (smallest margin {100 * (min(ratios.values()) - 1):.1f}%)")
    if not items:
        return "On the Google, new reanalysis, every season in which all four Juba stations go above their levels has them above on the same day."
    return ("On the Google, new reanalysis, all four Juba stations go above their levels within the season but never on the same day in "
            + ", ".join(items) + ".")


# ------------------------------------------------- a 1-in-3 framework on the new model
FRAMEWORK_RP = {("juba", "deyr"): 7, ("shabelle", "deyr"): 6, ("juba", "gu"): 7, ("shabelle", "gu"): 6}
FRAMEWORK_FC_SPAN = (2016, 2025)


def framework(R):
    """The chosen return periods on Google, new: activation years on both records, the 1-in-5
    gauge floods they meet, scores, and timing against the second gauge crossing its 1-in-3."""
    out = {}
    for (river, season), rp in FRAMEWORK_RP.items():
        rec = R[(river, season)]
        n_req = TRIGGER_CONFIG[(river, season)]["n_req"]
        ra = {int(y): d for y, d in L.first_crossing_dates(NEW, river, season, rp, n_req, span=YEARS).items()}
        fc = {int(y): d for y, d in first_issue_in_season(NEW, river, season, rp, n_req, span=range(FRAMEWORK_FC_SPAN[0], FRAMEWORK_FC_SPAN[1] + 1)).items()}
        sev = set(L.gauge_crossings(river, season, 5, span=YEARS))
        thr = L.model_thresholds(NEW, river, season, rp)
        out[(river, season)] = dict(rp=rp, ra=ra, fc=fc, sev=sev, thr=thr, assess=rec["assess"], bench=rec["bench"], flood3=rec["severe"])
    return out


def fw_score(F, keys, basis, span):
    yrs = set(range(span[0], span[1] + 1))
    universe, f, sev = set(), set(), set()
    for k in keys:
        x = F[k]
        own = yrs & x["assess"] & (set(YEARS) if basis == "ra" else set(range(FRAMEWORK_FC_SPAN[0], FRAMEWORK_FC_SPAN[1] + 1)))
        universe |= own
        f |= set(x[basis]) & own
        sev |= x["sev"] & own
    pod = len(f & sev) / len(sev) if sev else np.nan
    far = len(f - sev) / len(f) if f else np.nan
    prec = 1 - far if f else np.nan
    f1 = np.nan if np.isnan(pod) else (2 * pod * prec / (pod + prec) if f and (pod + prec) > 0 else 0.0)
    return dict(POD=pod, FAR=far, F1=f1, caught=sorted(f & sev), floods=sorted(sev), false=sorted(f - sev), reached=sorted(f), n=len(universe))


def framework_html(F):
    """One table: level, activation years, floods not reached, activations with no flood, timing. One
    score table for the combined scopes. Two notes: the forecast replay and the station levels."""
    name = lambda k: f"{k[0].title()} {k[1].title()}"
    half_up = lambda v: int(np.floor(v + 0.5))
    word = lambda v: "same day" if half_up(v) == 0 else f"{abs(half_up(v))} d {'before' if half_up(v) > 0 else 'after'}"
    a = ["<table class='data'><thead><tr><th>River, season</th><th>Level</th><th>Activations<span class='y'>reanalysis 1999 to 2023</span></th>"
         "<th>1-in-5 gauge floods not reached</th><th>Reached, no 1-in-5 flood</th><th>Timing, mean<span class='y'>against the second gauge's 1-in-3; reanalysis / forecasts</span></th></tr></thead><tbody>"]
    fc_note = []
    for k, x in F.items():
        scored = set(range(SCORE_RA[0], SCORE_RA[1] + 1)) & x["assess"]
        sev = x["sev"] & scored
        ra23 = set(x["ra"]) & scored
        d_ra = [-(x["ra"][y] - x["bench"][y]).days for y in ra23 if ok(x["bench"].get(y))]
        d_fc = [-(x["fc"][y] - x["bench"][y]).days for y in x["fc"] if y <= SCORE_RA[1] and ok(x["bench"].get(y))]
        tm = (word(float(np.mean(d_ra))) if d_ra else "none") + " / " + (word(float(np.mean(d_fc))) if d_fc else "none")
        a.append(f"<tr><td>{name(k)}</td><td>1-in-{x['rp']}</td><td>{yrs(x['ra'])}</td><td>{yrs(sev - ra23)}</td><td>{yrs(ra23 - sev)}</td><td>{tm}</td></tr>")
        ra16 = {y for y in x["ra"] if y >= FRAMEWORK_FC_SPAN[0]}
        extra, missing = set(x["fc"]) - ra16, ra16 - set(x["fc"])
        if extra or missing:
            fc_note.append(f"{name(k)}: " + "; ".join(filter(None, [f"forecasts add {yrs(extra)}" if extra else "", f"forecasts do not reach {yrs(missing)}" if missing else ""])))
    a.append("</tbody></table>")
    b_ = ["<table class='data'><thead><tr><th rowspan='2'>Scope</th><th colspan='3'>Reanalysis, seasons with a gauge record to 2023</th><th colspan='3'>Forecasts 2016 to 2023</th></tr>"
          "<tr><th>POD</th><th>FAR</th><th>F1</th><th>POD</th><th>FAR</th><th>F1</th></tr></thead><tbody>"]
    scopes = [("Deyr, both rivers", [k for k in F if k[1] == "deyr"]), ("Gu, both rivers", [k for k in F if k[1] == "gu"]), ("Either season, either river", list(F))]
    for lab, keys in scopes:
        cells = [f"<td>{lab}</td>"]
        for basis, span in (("ra", SCORE_RA), ("fc", SCORE_FC)):
            sc_ = fw_score(F, keys, basis, span)
            cells.append(f"<td>{num(sc_['POD'])}<span class='y'>{len(sc_['caught'])} of {len(sc_['floods'])}</span></td>"
                         f"<td>{num(sc_['FAR'])}<span class='y'>{len(sc_['false'])} of {len(sc_['reached'])}</span></td><td>{num(sc_['F1'])}</td>")
        b_.append("<tr>" + "".join(cells) + "</tr>")
    b_.append("</tbody></table>")
    fc_txt = ("Forecasts 2016 to 2025 reproduce every reanalysis activation from 2016" + (" except: " + "; ".join(fc_note) if fc_note else "") + ".")
    lv_txt = "Levels, m³/s: " + "; ".join(f"{name(k)} " + ", ".join(f"{NAME[st]} {v:.0f}" for st, v in x["thr"].items()) for k, x in F.items()) + "."
    return "\n".join(a), "\n".join(b_), fc_txt, lv_txt


def compare_table(R):
    """Google, new against each other source on the seasons both cover: gauge floods with the
    threshold reached, and seasons with the threshold reached and no gauge flood."""
    others = [("google_grrr", "Google, current"), ("glofas_v4", "GloFAS v4"), ("glofas_v5", "GloFAS v5")]
    head = ("<table class='data'><thead><tr><th rowspan='2'>Season</th><th rowspan='2'>River</th><th rowspan='2'>Record</th>"
            + "".join(f"<th colspan='2'>{name}</th>" for _, name in others) + "</tr><tr>"
            + "".join("<th>gauge floods reached<span class='y'>new / other</span></th><th>reached, no gauge flood<span class='y'>new / other</span></th>" for _ in others)
            + "</tr></thead><tbody>")
    out = [head]
    for season in ("deyr", "gu"):
        for river in RIVERS:
            rec = R[(river, season)]
            for basis, span, lab in (("ra", SCORE_RA, "reanalysis"), ("fc", SCORE_FC, "forecasts")):
                cells = [f"<td>{season.title()}</td><td>{river.title()}</td><td class='muted'>{lab}</td>"]
                for other, _ in others:
                    if basis == "fc" and other not in FC_SPAN:
                        cells.append("<td class='muted' colspan='2'>no archive</td>")
                        continue
                    common = years_of(rec, NEW, basis) & years_of(rec, other, basis) & set(range(span[0], span[1] + 1)) & rec["assess"]
                    if not common:
                        cells.append("<td class='muted' colspan='2'>no common season</td>")
                        continue
                    cspan = (min(common), max(common))
                    own = years_of(rec, NEW, basis) & set(range(span[0], span[1] + 1)) & rec["assess"]
                    a_, b_ = score(rec, NEW, basis, cspan), score(rec, other, basis, cspan)
                    note = f"<span class='y'>{rng(cspan)}</span>" if cspan != (min(own), max(own)) else ""
                    for key, tot in (("caught", len(a_["severe"])), ("false", None)):
                        x, y = len(a_[key]), len(b_[key])
                        cls = " class='top'" if x != y else ""
                        if tot == 0:
                            cells.append(f"<td class='muted'>no gauge flood{note}</td>")
                            continue
                        txt = f"{x} / {y}" + (f" of {tot}" if tot is not None else "")
                        cells.append(f"<td{cls}>{txt}{note}</td>")
                out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def findings(R, C):
    """Key takeaways: short statements without counts, generated from the same comparisons as the tables."""
    g = lambda s, r, m, b, span: score(R[(r, s)], m, b, span)
    top = lambda xs: [x.split(" (")[0] for x in xs]
    out = []
    agree = []
    for season in ("deyr", "gu"):
        for river in RIVERS:
            ys = top(C[season][river][0])
            if ys:
                agree.append(f"{season.title()} {' and '.join(ys)} on the {river.title()}")
    out.append("<strong>Largest floods.</strong> In the top five of every record that covers them: " + "; ".join(agree) + ".")
    miss = []
    for season in ("deyr", "gu"):
        for river in RIVERS:
            sc_ = g(season, river, NEW, "ra", SCORE_RA)
            if sc_["missed_severe"]:
                miss.append(f"{season.title()} {yrs(sc_['missed_severe'])} on the {river.title()}")
    out.append("<strong>Gauge floods with the threshold not reached by Google, new on its reanalysis:</strong> " + ("; ".join(miss) or "none") + ".")

    def compare(other, name):
        more_sev, fewer_sev, more_nf, fewer_nf = [], [], [], []
        for season in ("deyr", "gu"):
            for river in RIVERS:
                for basis, span, lab in (("ra", SCORE_RA, "reanalysis"), ("fc", SCORE_FC, "forecasts")):
                    if basis == "fc" and other not in FC_SPAN:
                        continue
                    ya, yb = years_of(R[(river, season)], NEW, basis), years_of(R[(river, season)], other, basis)
                    common = ya & yb & set(range(span[0], span[1] + 1))
                    if not common:
                        continue
                    cspan = (min(common), max(common))
                    a_, b_ = g(season, river, NEW, basis, cspan), g(season, river, other, basis, cspan)
                    tag = f"{river.title()} {season.title()} {lab}" + (f" ({rng(cspan)})" if cspan != span else "")
                    dc, df_ = len(a_["caught"]) - len(b_["caught"]), len(a_["false"]) - len(b_["false"])
                    (more_sev if dc > 0 else fewer_sev).append(tag) if dc else None
                    (more_nf if df_ > 0 else fewer_nf).append(tag) if df_ else None
        parts = []
        if more_sev:
            parts.append("threshold reached in more gauge floods in " + ", ".join(more_sev))
        if fewer_sev:
            parts.append("threshold reached in fewer gauge floods in " + ", ".join(fewer_sev))
        if more_nf:
            parts.append("more seasons with the threshold reached and no gauge flood in " + ", ".join(more_nf))
        if fewer_nf:
            parts.append("fewer seasons with the threshold reached and no gauge flood in " + ", ".join(fewer_nf))
        if not parts:
            return f"<strong>Google, new against {name}:</strong> the same counts in every river, season and record."
        return f"<strong>Google, new against {name}:</strong> " + "; ".join(parts) + ". In every other river, season and record the counts are the same."

    out.append(compare("glofas_v4", "GloFAS v4"))

    def when(season, m):
        mean, med, n, n_all = timing_stats(R, season, "fc", m)
        if n == 0:
            return "not reached"
        if int(np.floor(mean + 0.5)) == 0:
            return "on the day the second gauge crossed"
        return ("before" if mean > 0 else "after") + " the second gauge crossed"
    def earlier(season):
        """Mean difference over the seasons both sources reached on their forecasts."""
        diffs = []
        for river, y, rank in _timing_seasons(season):
            if y < SCORE_FC[0] or y > SCORE_FC[1]:
                continue
            rec = R[(river, season)]
            a_, b_ = rec["models"][NEW]["fc"].get(y), rec["models"]["glofas_v4"]["fc"].get(y)
            if ok(a_) and ok(b_) and y in rec["bench"]:
                diffs.append((b_ - a_).days)
        if not diffs:
            return ""
        d = float(np.mean(diffs))
        return ", Google, new earlier on average in the seasons both reached" if d > 0 else ", GloFAS v4 earlier on average in the seasons both reached" if d < 0 else ", the same on average in the seasons both reached"
    def pair(season):
        a_, b_ = when(season, NEW), when(season, "glofas_v4")
        if a_ == b_:
            return f"Google, new and GloFAS v4 {a_}{earlier(season)}"
        return f"Google, new {a_}; GloFAS v4 {b_}"
    out.append("<strong>Timing on the forecasts</strong> (mean days before the second gauge crossed its 1-in-3, over the largest gauge floods from 2016 in which the threshold was reached). "
               f"Deyr: {pair('deyr')}; Google, current {when('deyr', 'google_grrr')} (its Deyr archive ends in 2022). "
               f"Gu: {pair('gu')}; Google, current {when('gu', 'google_grrr')}.")
    return "\n".join(f"<li>{x}</li>" for x in out)


def after_2023(R, season, y):
    act_r = [r.title() for r in RIVERS if y in R[(r, season)]["models"][NEW]["ra"]]
    fc_r = [r.title() for r in RIVERS if y in R[(r, season)]["models"][NEW]["fc"]]
    outc = []
    for r in RIVERS:
        rec = R[(r, season)]
        outc.append(f"{r.title()} {'gauge flood' if y in rec['severe'] else 'no gauge flood' if y in rec['assess'] else 'no gauge record'}")
    say = lambda rs: ("threshold reached on the " + " and ".join(rs)) if rs else "threshold not reached"
    if act_r == fc_r:
        rec_txt = say(act_r) + (" on both records" if act_r else " on either record")
    else:
        rec_txt = f"reanalysis: {say(act_r)}; forecasts: {say(fc_r)}"
    return f"{rec_txt} (gauges: {', '.join(outc)})"


def build():
    R = record()
    ST = station_scores()
    F = {}
    for s in ("deyr", "gu"):
        F[f"bt_{s}"] = fig_backtest(R, s)
        F[f"tm_{s}"] = fig_timing(R, s)
        F[f"st_{s}"] = fig_stations(ST, s)
    CR = correlations()
    F["corr"] = fig_corr(CR)
    for k in (("deyr", 2023), ("deyr", 2019), ("deyr", 2020)):
        F[f"tl_{k[0]}_{k[1]}"] = fig_timeline(R, *k)
    RK = {s: rank_html(s) for s in ("deyr", "gu")}
    C = {s: consistency(s) for s in ("deyr", "gu")}
    cons_txt = lambda xs: ", ".join(xs) if xs else "none"
    FIND = findings(R, C)
    T = {s: score_table(R, s) for s in ("deyr", "gu")}
    TB = both_rivers_table(R)
    import datetime as _dt
    TODAY = _dt.date.today().isoformat()
    TS = timing_summary_html(R)
    NEAR_MISS = near_miss_note(R)
    SPAN_TXT = span_text(R)
    CMP = compare_table(R)
    FW = framework(R)
    FW_A, FW_B, FW_FC, FW_LV = framework_html(FW)
    fw_deyr = {y for k, x in FW.items() if k[1] == "deyr" for y in x["ra"] if y <= SCORE_RA[1]}
    fw_gu = {y for k, x in FW.items() if k[1] == "gu" for y in x["ra"] if y <= SCORE_RA[1]}
    FW_SEASONS = (f"Deyr {yrs(fw_deyr)}; Gu {yrs(fw_gu)}: {len(fw_deyr) + len(fw_gu)} season activations in {len(YEARS) - 2} years, one per "
                  f"{(len(YEARS) - 2) / (len(fw_deyr) + len(fw_gu)):.1f} years, with both seasons in {yrs(fw_deyr & fw_gu)}.")

    # everything behind the page, for review
    dump, sc = {}, []
    for (river, season), rec in R.items():
        dump[f"{river}_{season}"] = dict(
            flood_1in3=sorted(rec["flood"]), severe=sorted(rec["severe"]), assessable=sorted(rec["assess"]),
            second_gauge_1in3={str(y): v.strftime("%Y-%m-%d") for y, v in rec["bench"].items()},
            gauge_peak={str(y): v.strftime("%Y-%m-%d") for y, v in rec["peak"].items()},
            models={m: dict(reanalysis={str(y): v.strftime("%Y-%m-%d") for y, v in mm["ra"].items()},
                            forecast={str(y): v.strftime("%Y-%m-%d") for y, v in mm["fc"].items()},
                            reanalysis_years=rng(RA_SPAN[m]), forecast_years=(rng((min(mm["fc_years"]), max(mm["fc_years"]))) if mm["fc_years"] else None))
                    for m, mm in rec["models"].items()})
        for m in MODELS[season]:
            sc.append(dict(river=river, season=season, **score(rec, m, "ra", SCORE_RA)))
            if m in FC_SPAN:
                sc.append(dict(river=river, season=season, **score(rec, m, "fc", SCORE_FC)))
                if m == "glofas_v4":
                    sc.append(dict(river=river, season=season, **score(rec, m, "fc", FC_SPAN[m])))
    levels = {m: {s: {st: round(L.model_threshold(m, st, s, TRIGGER_RP), 1) for st in STATIONS} for s in ("deyr", "gu")}
              for m in [NEW, "google_grrr", "glofas_v4", "glofas_v5"]}
    ranks = {}
    for season in ("deyr", "gu"):
        for river in RIVERS:
            P, Rk = rank_table(season, river)
            ranks[f"{river}_{season}"] = dict(
                index={str(int(y)): {("gauge" if k == "gauge" else f"{k[0]}_{k[1]}"): (None if pd.isna(v) else round(float(v), 4)) for k, v in P.loc[y].items()} for y in P.index},
                rank={str(int(y)): {("gauge" if k == "gauge" else f"{k[0]}_{k[1]}"): (None if pd.isna(v) else int(v)) for k, v in Rk.loc[y].items()} for y in Rk.index})
    (OUT / "data.json").write_text(json.dumps(clean(dict(
        outlets=_read_json("chosen_outlets.json"), trigger_levels=levels, record=dump, scores=sc, benchmark_ranks=ranks,
        framework={f"{k[0]}_{k[1]}": dict(rp=x["rp"], levels={st: float(v) for st, v in x["thr"].items()},
                                          reanalysis={str(y): d.strftime("%Y-%m-%d") for y, d in x["ra"].items()},
                                          forecast={str(y): d.strftime("%Y-%m-%d") for y, d in x["fc"].items()},
                                          gauge_floods_1in5=sorted(int(y) for y in x["sev"])) for k, x in FW.items()},
        gauge_levels={f"{r}_{se}": {f"1in{rp}": {st: (None if np.isnan(v) else float(v)) for st, v in L.gauge_levels(r, se, rp).items()} for rp in (3, 5)}
                      for r in RIVERS for se in ("deyr", "gu")},
        timing_stats=[dict(season=se, basis=b, model=m, **dict(zip(("mean_days_before", "median_days_before", "n_reached", "n_seasons"), timing_stats(R, se, b, m))))
                      for se in ("deyr", "gu") for b in ("ra", "fc") for m in MODELS[se] if b == "ra" or m in FC_SPAN],
        station_scores=json.loads(ST.to_json(orient="records")),
        correlations=json.loads(CR.to_json(orient="records")))), indent=1))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Google's new river model: Somalia Riverine Flood Trigger</title>
<meta name="description" content="Google's November 2026 river model at the seven Somalia trigger stations, scored with the trigger's rules and compared with GloFAS v4, GloFAS v5 and the current Google model, on the reanalysis and on the forecast archives, against the SWALIM gauges.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css">
<style>
.hero.hero-sub {{ padding:40px 44px 34px; }}
.hero.hero-sub h1 {{ font-size:28px; }}
.crumb {{ font-size:12px; margin:0 0 14px; }}
.crumb a {{ color:rgba(255,255,255,.85); text-decoration:none; }}
article {{ padding:8px 44px 24px; max-width:none; }}
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:21px; color:var(--n9); margin:40px 0 10px; line-height:1.25; }}
article h2 .num {{ color:var(--b5); font-size:15px; margin-right:8px; }}
article h3 {{ font-family:'Merriweather',Georgia,serif; font-size:15.5px; color:var(--n9); margin:26px 0 6px; }}
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.65; }}
article ul {{ padding-left:22px; }}
article strong {{ color:var(--n9); }}
article code {{ background:var(--n05); border:1px solid #e2e7e7; padding:1px 5px; border-radius:3px; font-size:12.5px; color:var(--b7); }}
.muted {{ color:#6b7683; }}
p.note {{ font-size:13px; color:var(--n7); margin:6px 0 0; }}
figure.chart {{ margin:24px 0 30px; }}
figure.chart img {{ width:100%; height:auto; display:block; }}
figure.chart figcaption {{ font-size:13px; color:#55606d; line-height:1.55; margin-top:10px; padding-left:2px; }}
.callout {{ margin:20px 0; padding:13px 17px; border-radius:4px; background:var(--b05); border-left:5px solid var(--b5); font-size:13.5px; color:var(--n8); line-height:1.6; }}
.callout p {{ margin:0 0 7px; font-size:13.5px; }}
.callout p:last-child {{ margin:0; }}
.callout strong {{ color:var(--n9); }}
.takeaways {{ margin:18px 0 6px; padding:0; list-style:none; }}
.takeaways li {{ border-left:3px solid var(--b1); padding:0 0 0 14px; margin:0 0 16px; }}
.tablewrap {{ overflow:visible; }}
table.data {{ border-collapse:collapse; width:100%; margin:16px 0 22px; font-size:14px; table-layout:auto; }}
table.data th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:6px 6px 5px; }}
table.data th[colspan] {{ text-align:center; border-bottom:1px solid #e2e7e7; }}
table.data td {{ border-bottom:1px solid #eef1f1; padding:7px 5px; color:var(--n8); vertical-align:top; }}
table.data tr.riv td {{ color:var(--n9); font-weight:500; padding-top:12px; border-bottom:1px solid #d9dfdf; background:#fff; }}
table.data td:first-child {{ white-space:nowrap; }}
table.data td:nth-child(5) {{ border-right:1px solid #e2e7e7; }}
table.data .y {{ display:block; font-size:11.5px; color:var(--n7); font-weight:400; text-transform:none; letter-spacing:0; }}
table.data td.top {{ background:rgba(30,121,95,.12); }}
.notes li {{ font-size:13px; color:var(--n7); margin:0 0 8px; }}
@media (max-width:640px) {{ article {{ padding:4px 22px 16px; }} .hero.hero-sub {{ padding:32px 22px 28px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <div class="inner">
      <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / Google's new river model</p>
      <h1>Google's new river model</h1>
      <p>Google's November 2026 river model at the seven trigger stations, scored with the trigger's rules and compared with GloFAS v4,
         GloFAS v5 and Google, current, on each model's reanalysis and forecast archive, against the SWALIM gauges.
         The live Flood Hub feed does not run the new model. Generated {TODAY} from the source data.</p>
    </div>
  </header>
  <article>
    <div class="callout">
      <p><strong>Sources:</strong> Google, new (the November 2026 model); Google, current (Google's published 2024 model, not verified as the live
         Flood Hub model); GloFAS v4 (the source the Deyr trigger runs on today); GloFAS v5 (the model the Deyr thresholds were designed on). <strong>Records:</strong> reanalysis, the model's own
         record of the past; forecasts, the first issue dated in the season that reaches the threshold at any lead from 1 to 7 days. The live trigger reads forecasts this way.
         Archives: GloFAS v4 2003 to 2023, Google, current 2016 to June 2023, Google, new 2016 to 2025, GloFAS v5 none.</p>
      <p><strong>Threshold rule:</strong> three of the four Juba stations, or two of the three Shabelle stations, over their own 1-in-{TRIGGER_RP} level on the
         same day. Each model's levels are fitted on its own 1999 to 2023 seasonal maxima, with the same return period at every station and in both seasons. Scores: seasons with a
         gauge record, {SPAN_TXT}, on the reanalysis; {rng(SCORE_FC)} on the forecasts (Google, current in Deyr: 2016 to 2022).</p>
      <p><strong>Gauge benchmark:</strong> a gauge flood is two SWALIM gauges on the same river over their own 1-in-{BENCH_RP} level, fitted on 2000 to 2023.
         A threshold reached in a season with no gauge flood is a false alarm. The ranking of seasons and the timing use the day the second
         gauge crossed. Gauge data run to July 2026.</p>
    </div>

    <h2><span class="num">1</span>A 1-in-3 framework on Google, new</h2>
    <p>Return periods from a search over 1-in-3 to 1-in-10 per river and season on the reanalysis, for about eight season activations in 25 years
       (one per season in which either river reaches its threshold), Juba stricter than Shabelle, scored against the two-gauge 1-in-5 floods.</p>
    <div class="tablewrap">{FW_A}</div>
    <p class="note">{FW_SEASONS} {FW_FC}</p>
    <h3>POD, FAR and F1 against the 1-in-5 gauge floods</h3>
    <div class="tablewrap">{FW_B}</div>

    <h2><span class="num">2</span>Key takeaways</h2>
    <ul class="takeaways">
{FIND}
    </ul>

    <h2><span class="num">3</span>Threshold outcome and scores by season</h2>
    <p>Each model's threshold outcome in every season against the gauge benchmark: under each source a reanalysis row and, where an archive exists,
       a forecast row. White cells are seasons outside the model record or without a gauge record. Reanalysis rows are tallied over every season
       with a gauge record, forecast rows between the dotted lines.</p>
    <h3>Deyr</h3>
    <figure class="chart"><img src="{F['bt_deyr']}" alt="Deyr, outcome by season, reanalysis and forecasts"></figure>
    <div class="tablewrap">{T['deyr']}</div>
    <p class="note">Gauge floods counted are those in the seasons scored: {SPAN_TXT}.</p>
    <h3>Gu</h3>
    <figure class="chart"><img src="{F['bt_gu']}" alt="Gu, outcome by season, reanalysis and forecasts"></figure>
    <div class="tablewrap">{T['gu']}</div>
    <p class="note">The Google, current archive labels the issue day lead 0; the page reads its leads 1 to 7, the seven days after the issue day, as the live
       trigger does. Read at leads 0 to 6 instead, Google, current does not reach the threshold in Gu 2023 on the Shabelle.</p>
    <h3>Both rivers together</h3>
    <p>A year counts as reached when the threshold is reached on either river, and as a gauge flood when either river has one; the last column
       joins the two seasons: the framework as a whole, by year.</p>
    <div class="tablewrap">{TB}</div>
    <h3>Google, new against each other source</h3>
    <p>Counts over the seasons both sources cover: gauge floods with the threshold reached, and seasons with the threshold reached and no gauge flood,
       Google, new first. Shaded where the two differ.</p>
    <div class="tablewrap">{CMP}</div>

    <h2><span class="num">4</span>POD, FAR and F1 by station</h2>
    <p>Each station's model against its gauge, reanalysis, seasons with a gauge record to 2023. Here a flood season is that one gauge over its own 1-in-{BENCH_RP},
       not the two-gauge benchmark used elsewhere.</p>
    <figure class="chart"><img src="{F['st_deyr']}" alt="Deyr, POD, FAR and F1 per station"></figure>
    <figure class="chart"><img src="{F['st_gu']}" alt="Gu, POD, FAR and F1 per station"></figure>

    <h2><span class="num">5</span>Correlation with the gauges by station</h2>
    <p>For each station, the model's daily flow is set against the gauge's daily level in 1-in-3 seasons: seasons in which the gauge went over its
       own 1-in-3 level or the model went over its own, among seasons with a gauge record to 2023. Spearman rank correlation orders the days by
       model flow and, separately, by gauge level, and measures how far the two orders agree: 1 means the model's highest days are the gauge's
       highest days, 0 means no relation. It uses the order of the days, not their values, so the scale of the model does not matter. The lag
       is the shift in days between the two series that gives the highest agreement, tried from 10 days behind to 30 days ahead; a positive lag
       means the model rises before the gauge. The figure shows the value at that lag. It measures day-to-day tracking in flood seasons,
       not whether a threshold is reached.</p>
    <figure class="chart"><img src="{F['corr']}" alt="Correlation with the gauges by station"><figcaption>The small figure under each value is the number of seasons.</figcaption></figure>

    <h2><span class="num">6</span>Benchmark: the largest floods at the gauges</h2>
    <p>The five largest gauge floods per river, largest first, and each source's own rank of them (1 = highest; reanalysis / forecasts).
       Index: the second-highest gauge's peak as a percentage of its own 1-in-{TIMING_RP}; for a model, the same at its stations. Tied seasons share a rank.
       Shaded cells: top five on the reanalysis. No record: season outside the reanalysis. No archive: no forecast archive. The notes under each table
       list only seasons covered by at least three records.</p>
    <h3>Deyr</h3>
    <div class="tablewrap">{RK['deyr']}</div>
    <p class="note">In the top five of every record that covers the season: Juba {cons_txt(C['deyr']['juba'][0])}; Shabelle {cons_txt(C['deyr']['shabelle'][0])}.
       In the top five of fewer than half: Juba {cons_txt(C['deyr']['juba'][1])}; Shabelle {cons_txt(C['deyr']['shabelle'][1])}.</p>
    <h3>Gu</h3>
    <div class="tablewrap">{RK['gu']}</div>
    <p class="note">In the top five of every record that covers the season: Juba {cons_txt(C['gu']['juba'][0])}; Shabelle {cons_txt(C['gu']['shabelle'][0])}.
       In the top five of fewer than half: Juba {cons_txt(C['gu']['juba'][1])}; Shabelle {cons_txt(C['gu']['shabelle'][1])}. The Belet Weyne gauge reads no higher than
       8.30 m, which six Gu seasons reach, so those seasons tie at that gauge.</p>

    <h2><span class="num">7</span>Timing against the second gauge crossing its 1-in-3</h2>
    <p>The five largest gauge floods per river. Zero is the day the second gauge crossed its 1-in-3; bars to the left reached the
       threshold before that day, bars to the right after it. Solid: reanalysis. Light: first forecast issue reaching the threshold (leads 1 to 7 days).</p>
    <figure class="chart"><img src="{F['tm_deyr']}" alt="Deyr timing"></figure>
    <figure class="chart"><img src="{F['tm_gu']}" alt="Gu timing"></figure>
    <h3>Average timing</h3>
    <p>Mean and median days between the threshold being reached and the second gauge crossing its 1-in-3, over the seasons above that fall in the
       scored years (reanalysis to 2023, forecasts 2016 to 2023), counting only seasons in which the source reached the threshold.</p>
    <div class="tablewrap">{TS}</div>

    <h2><span class="num">8</span>Event chronologies: Deyr 2023, Deyr 2019, Deyr 2020</h2>
    <p>Gauge rows: circle, first day over the 1-in-3; diamond, first day over the 1-in-5 (only the diamond is shown when both fall on the same day);
       tick, the peak, dated in the row label. Source rows: threshold reached on the reanalysis (filled square), first forecast issue reaching the
       threshold (hollow square), one mark when both fall on the same day. FloodScan people in flooded area, by river, where it covers the season;
       the dotted line is the day the third of the 14 framework districts went over its own 1-in-5 exposure (people in flooded area).</p>
    <figure class="chart"><img src="{F['tl_deyr_2023']}" alt="Deyr 2023 day by day"><figcaption>Deyr 2023.</figcaption></figure>
    <figure class="chart"><img src="{F['tl_deyr_2019']}" alt="Deyr 2019 day by day"><figcaption>Deyr 2019.</figcaption></figure>
    <figure class="chart"><img src="{F['tl_deyr_2020']}" alt="Deyr 2020 day by day"><figcaption>Deyr 2020. A 1 October date means the level was already exceeded when the season opened.</figcaption></figure>

    <h2><span class="num">9</span>Notes</h2>
    <p class="note">Source: Google Runoff Reanalysis and Reforecast, November 2026 release (reanalysis 1981 to 2025, reforecast 2016 to 2025, leads 0 to 10 days where lead 0 is the day before the issue, so its leads 1 to 7 are read as the issue day and the six days after it; the archive has leads to 10
       days), one HydroBasins outlet per station, the main-stem outlet within 15 km with the largest 1-in-5 flow. Leads 1 to 7 cover the issue day and
       the six days after it for GloFAS and Google, new, and days 1 to 7 after the issue day for Google, current, following each archive's own labelling;
       only issues dated within October to December or March to May are read, and only days in those months count, so a threshold date of 1 October or
       1 March means the level was already exceeded when the season opened. {NEAR_MISS} A gauge reading equal to its level counts as a crossing; the Juba
       Deyr 2023 peaks are capped at bank full; Dollow's Deyr record covers eight seasons only. Forecast windows cover 7 or 8 seasons (Google, current in Deyr: 7);
       GloFAS forecasts are the middle value of the ensemble members, the Google archives a single trace. The Google, current archive has not been verified
       as the live Flood Hub model; Google, new flows are on a different scale from the live feed's; 2026 is in no archive. The dates, levels, ranks and
       scores behind every chart and table are published beside this page.</p>
  </article>  </article>
</div>
</body>
</html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("written", OUT / "index.html")
    print(pd.DataFrame(sc)[["season", "river", "model", "basis", "span", "caught", "false", "POD", "FAR", "F1"]].to_string(index=False))


if __name__ == "__main__":
    build()
