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
COL = {**SOURCE_COLORS, NEW: "#6A4C93"}
MODELS = {"deyr": [NEW, "google_grrr", "glofas_v4", "glofas_v5"], "gu": [NEW, "google_grrr", "glofas_v4", "glofas_v5"]}
# each model's record: reanalysis years and forecast-archive years
RA_SPAN = {NEW: (1999, 2025), "google_grrr": (1999, 2023), "glofas_v4": (1999, 2023), "glofas_v5": (1999, 2023)}
FC_SPAN = {NEW: (2016, 2025), "google_grrr": (2016, 2023), "glofas_v4": (2003, 2023)}   # v5: no archive
# the current Google archive ends 2023-06-30: it holds Gu 2023 but no Deyr 2023
FC_END = {("google_grrr", "deyr"): 2022}
SCORE_RA = (1999, 2023)          # every model and the gauges overlap here
SCORE_FC = (2016, 2023)          # every forecast archive overlaps here
YEARS = list(range(1999, 2026))
CELL = {"hit_sev": "#123F5F", "hit": "#7FA7C9", "false": "#9A8C7A", "miss_sev": "#C2185B",
        "miss": "#E9B9CB", "quiet": "#EDF1F4", "none": "#FFFFFF"}
WINDOW = {"deyr": ("10-01", "12-31"), "gu": ("03-01", "05-31")}
# the 14 anticipatory-action districts by river (as scripts/page_additions/floodscan_districts.py)
DISTRICTS = {"juba": ["SO2605", "SO2606", "SO2602", "SO2703", "SO2701", "SO2702"],
             "shabelle": ["SO2001", "SO2002", "SO2003", "SO2101", "SO2103", "SO2302", "SO2305", "SO2301"]}
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
            fl, sev, assess = L.benchmark(river, season, span=YEARS)
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


def save(fig, name, tight=True):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.svg", format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    return f"figures/{name}.svg"


# ------------------------------------------------------- figure: season-by-season grid
def fig_backtest(R, season, basis):
    span = SCORE_RA if basis == "ra" else SCORE_FC
    models = MODELS[season]
    lines = []
    for river in RIVERS:
        rec = R[(river, season)]
        for m in models:
            lines.append((f"{river.title()}: {SHORT[m]}", rec, m))
    fig, ax = plt.subplots(figsize=(13.4, 0.6 * len(lines) + 2.6))
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
        if basis == "fc" and m not in FC_SPAN:
            ax.text(len(YEARS) / 2, i, "no forecast archive", ha="center", va="center", fontsize=9.5, color=FAINT)
            continue
        s = score(rec, m, basis, span)
        sev_txt = f"{len(s['caught'])} of {len(s['severe'])} severe seasons" if s["severe"] else "no severe season in archive"
        ax.text(len(YEARS) + 0.2, i + 0.14, sev_txt, va="center", fontsize=9.5, color=INK)
        ax.text(len(YEARS) + 0.2, i - 0.2, f"{len(s['false'])} with no gauge flood", va="center", fontsize=9.5,
                color="#B07A1E" if s["false"] else FAINT)
    ax.axhline(len(models) - 0.5, color=INK, linewidth=1.0)
    for edge_year in (span[0] - 0.5, span[1] + 0.5):
        ax.axvline(YEARS.index(int(edge_year + 0.5)) - 0.5, color=INK, linewidth=0.9, ls=":", zorder=4)
    ax.text(YEARS.index(span[0]) - 0.5, -0.62, f" dotted lines: the tallies on the right count {rng(span)}", ha="left", va="top", fontsize=8.5, color=FAINT)
    ax.set_xlim(-0.6, len(YEARS) + 5.6)
    ax.set_ylim(-1.1, len(lines) - 0.4)
    ax.set_yticks(range(len(lines)), [ln[0] for ln in lines[::-1]], fontsize=10.5)
    ax.set_xticks(range(0, len(YEARS), 2), [str(y) for y in YEARS[::2]], fontsize=9.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [Rectangle((0, 0), 1, 1, facecolor=CELL["hit_sev"], label="threshold reached, severe flood"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["hit"], label="threshold reached, ordinary flood"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["false"], label="threshold reached, no flood at the gauges"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss_sev"], label="severe flood, threshold not reached"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["miss"], label="ordinary flood, threshold not reached"),
         Rectangle((0, 0), 1, 1, facecolor=CELL["quiet"], edgecolor=GRID, label="no gauge flood, threshold not reached"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID, label="no record")]
    ax.legend(handles=H, ncol=4, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.45, -0.08))
    what = "reanalysis" if basis == "ra" else "forecasts, first issue reaching the threshold (leads 1 to 7 days)"
    ax.set_title(f"{SEASON_TITLE[season]}, {what}", fontsize=12, loc="left")
    fig.subplots_adjust(left=0.16, right=0.985, top=0.92, bottom=0.26 if len(lines) <= 6 else 0.2)
    return save(fig, f"backtest_{season}_{basis}", tight=False)


# ---------------------------------------------------------- figure: timing vs the gauges
def _timing_seasons(season):
    """The gauges' N_RANK worst seasons per river (the benchmark tables), with their gauge rank."""
    out = []
    for river in RIVERS:
        _, Rk = rank_table(season, river)
        g = Rk["gauge"].dropna().sort_values()
        for y, r in g.head(N_RANK).items():
            out.append((river, int(y), int(r)))
    return out


def fig_timing(R, season):
    """Days between the threshold being reached and the second gauge crossing its 1-in-3, as
    labelled bars: one row per season and source, the reanalysis as a solid bar and the forecasts
    as a lighter bar beneath it. Right of zero: before the gauges; left: after."""
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
    ylabels = []
    for i, (river, y, rank, m) in enumerate(rows):
        yc = ymax - 1 - i
        if m is None:
            ax.text(-39, yc, f"{river.title()} {y}   ({'largest' if rank == 1 else str(rank) + ('nd' if rank == 2 else 'rd' if rank == 3 else 'th') + ' largest'} gauge flood)", ha="left", va="center", fontsize=10, color=INK, fontweight="bold", zorder=5)
            if i:
                ax.axhline(yc + 0.5, color="white", linewidth=3, zorder=2)
            continue
        rec = R[(river, season)]
        b = rec["bench"].get(y)
        mm = rec["models"][m]
        for k, (basis, alpha) in enumerate((("ra", 1.0), ("fc", 0.45))):
            yb = yc + (0.2 if k == 0 else -0.2)
            have = y in mm[f"{basis}_years"]
            d = mm[basis].get(y)
            tag = "reanalysis" if basis == "ra" else "forecasts"
            if not have:
                ax.text(1.2, yb, f"{tag}: no archive" if basis == "fc" else "outside the model record", va="center", fontsize=8.5, color=FAINT)
                continue
            if not ok(d) or not ok(b):
                ax.text(1.2, yb, f"{tag}: not reached" if ok(b) else "no gauge flood", va="center", fontsize=8.5, color=FAINT)
                continue
            n = (d - b).days            # negative: before the gauges (left), positive: after (right)
            x = max(min(n, 38), -38)
            ax.barh(yb, x if n else 0.4, height=H, color=COL[m], alpha=alpha, edgecolor="white", linewidth=0.5, zorder=4)
            lab = f"{abs(n)} d {'before' if n < 0 else 'after'}" if n else "same day"
            if abs(n) > 38:
                lab += " (bar cut)"
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
    Hh = [Rectangle((0, 0), 1, 1, facecolor=BODY, alpha=1.0, label="solid: reanalysis"),
          Rectangle((0, 0), 1, 1, facecolor=BODY, alpha=0.45, label="light: first forecast issue reaching the threshold (leads 1 to 7 days)")]
    ax.legend(handles=Hh, ncol=2, fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title(f"{SEASON_TITLE[season]}: days between the threshold being reached and the second gauge crossing its 1-in-3, five largest gauge floods per river", fontsize=11, pad=40)
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
        n_all += 1
        d = mm[basis].get(y)
        if ok(d):
            ds.append((rec["bench"][y] - d).days)
    if not ds:
        return (None, None, 0, n_all)
    return (float(np.mean(ds)), float(np.median(ds)), len(ds), n_all)


def timing_summary_html(R):
    fmt_d = lambda v: ("same day" if round(v) == 0 else f"{abs(v):.0f} d {'before' if v > 0 else 'after'}")
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
    fig, axes = plt.subplots(len(ratios), 1, figsize=(11.6, 0.5 * sum(ratios) + 2.6), sharex=True,
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
            if len(s) < 30:
                notes.append("no gauge record" if s.empty else "gauge record too short")
            else:
                pk = s.loc[s.level_m.idxmax(), "date"]
                ax.scatter(pk, yc, s=170, marker="|", color=INK, linewidths=1.8, zorder=4)
                near_edge = (B - pk).days < 14
                ax.text(pk + pd.Timedelta(days=-2 if near_edge else 2), yc, f"peak {pk:%d %b}", ha="right" if near_edge else "left", va="center", fontsize=8.5, color=INK)
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
                        ax.scatter(d, yc, s=70 if marker == "o" else 60, marker=marker, color=BODY, edgecolors="white", linewidths=0.8, zorder=5)
                        ax.text(d, yc + (0.22 if marker == "o" else -0.22), f"{d:%d %b}", ha=edge(d), va="bottom" if marker == "o" else "top", fontsize=8.5, color=BODY)
            if notes:
                ax.text(B - pd.Timedelta(days=1), yc, "; ".join(notes), va="center", ha="right", fontsize=7.5, color=FAINT, linespacing=1.1)
            ylab.append((yc, f"{NAME[st]} gauge"))
        # source rows
        for j, m in enumerate(models):
            yc = n - 1 - len(sts) - j
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
                        ax.scatter(d, yc, s=90, color=COL[m], edgecolors="white", linewidths=0.8, zorder=4)
                        ax.scatter(d, yc, s=150, facecolors="none", edgecolors=COL[m], linewidths=1.6, zorder=4)
                        ax.text(d, yc + 0.22, d.strftime("%d %b") + " (both)", ha="center", va="bottom", fontsize=8.5, color=COL[m])
                    continue
                if filled:
                    ax.scatter(d, yc, s=90, color=COL[m], edgecolors="white", linewidths=0.8, zorder=4)
                    ax.text(d, yc + 0.22, d.strftime("%d %b"), ha="left" if (d - A).days < 12 else "right" if (B - d).days < 12 else "center", va="bottom", fontsize=8.5, color=COL[m])
                else:
                    ax.scatter(d, yc, s=110, facecolors="none", edgecolors=COL[m], linewidths=1.6, zorder=4)
                    ax.text(d, yc - 0.22, d.strftime("%d %b"), ha="left" if (d - A).days < 12 else "right" if (B - d).days < 12 else "center", va="top", fontsize=8.5, color=COL[m])
            if notes == ["outside the model record", "no forecast archive"]:
                notes = ["outside the model record"]
            if notes == ["not reached", "forecasts: not reached"]:
                notes = ["not reached"]
            if notes:
                ax.text(B - pd.Timedelta(days=1), yc, "; ".join(notes), va="center", ha="right", fontsize=8.5, color=FAINT)
            ylab.append((yc, SHORT[m]))
        ax.axhline(n - len(sts) - 0.5, color=INK, linewidth=0.9, zorder=1)
        ax.set_yticks([y for y, _ in ylab], [l for _, l in ylab], fontsize=10)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(A - pd.Timedelta(days=4), B)
        state = ("severe flood season at the gauges" if year in rec["severe"] else "ordinary flood season at the gauges" if year in rec["flood"]
                 else "no flood at the gauges" if year in rec["assess"] else "no gauge record")
        ax.set_title(f"{river.title()} ({state})", fontsize=11.5, loc="left")
        ax.tick_params(length=0, labelbottom=True, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    if has_exp:
        from src.constants import C_JUBA, C_SHAB
        ex = sc.exposure()
        ex = ex[(ex.valid_date >= A) & (ex.valid_date <= B)]
        fd, _ = sc.flood_date(season, year)
        for axe, river_d, colr in ((axes[1], "juba", C_JUBA), (axes[3], "shabelle", C_SHAB)):
            e = ex[ex.pcode.isin(DISTRICTS[river_d])].groupby("valid_date")["sum"].sum()
            axe.fill_between(e.index, e.values, color=colr, alpha=0.15, lw=0)
            axe.plot(e.index, e.values, color=colr, lw=1.4)
            if fd is not None:
                axe.axvline(fd, color=INK, lw=1.1, ls=":", zorder=1)
                if river_d == "juba":
                    axe.text(fd, axe.get_ylim()[1] * 0.95, f" third of the 14 districts over its 1-in-5 {fd:%d %b}", va="top", ha="left", fontsize=8.5, color=INK)
            axe.set_title(f"Flood exposure, {len(DISTRICTS[river_d])} {river_d.title()} districts (FloodScan)", fontsize=10.5, loc="left")
            axe.set_ylabel("people in flooded area", color=BODY, fontsize=9)
            axe.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
            axe.set_ylim(0, None)
            axe.tick_params(length=0, labelsize=8)
            for sp in axe.spines.values():
                sp.set_visible(False)
            axe.grid(axis="y", color=GRID, linewidth=0.8)
            axe.set_axisbelow(True)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    Hh = [Line2D([], [], marker="o", ls="", color=BODY, markersize=7, label="gauge over its 1-in-3"),
          Line2D([], [], marker="D", ls="", color=BODY, markersize=6, label="gauge over its 1-in-5"),
          Line2D([], [], marker="|", ls="", color=INK, markersize=12, markeredgewidth=1.8, label="gauge peak"),
          Line2D([], [], marker="o", ls="", color=INK, markersize=8, label="threshold reached, reanalysis"),
          Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=9, markeredgewidth=1.6, label="first forecast issue reaching the threshold")]
    fig.legend(handles=Hh, ncol=5, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 0.03), frameon=False)
    fig.suptitle(f"{season.title()} {year}", x=0.02, ha="left", fontsize=13)
    fig.subplots_adjust(hspace=0.35, top=0.92, bottom=0.12, right=0.97, left=0.16)
    return save(fig, f"timeline_{season}_{year}")


def sc_colour(st):
    import season_charts as sc
    return sc.C[st]


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
        outcome = f"threshold reached {ra:%d %b}" if ok(ra) else "threshold not reached"
        ax.text(0, 1.07, f"{river.title()}   ·   rule {n} of {len(sts)} over 1-in-{rp}   ·   new model: {outcome}", transform=ax.transAxes, fontsize=12, color=INK)
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
        from floodscan_districts import DISTRICTS
        from src.constants import C_JUBA, C_SHAB
        ex = sc.exposure()
        ex = ex[(ex.valid_date >= A) & (ex.valid_date <= B)]
        for river_d, colr in (("juba", C_JUBA), ("shabelle", C_SHAB)):
            e = ex[ex.pcode.isin(DISTRICTS[river_d])].groupby("valid_date")["sum"].sum()
            axe.fill_between(e.index, e.values, color=colr, alpha=0.12, lw=0)
            axe.plot(e.index, e.values, color=colr, lw=1.4, label=f"{river_d.title()} districts ({len(DISTRICTS[river_d])})")
        axe.legend(fontsize=8.5, loc="upper left", frameon=False)
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
        srcs = [c for c in Rk.columns if c != "gauge" and c[0] in MODELS[season]]
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
    """One figure: Deyr and Gu side by side. Rows: the gauges' N_RANK worst seasons on each river,
    Juba then Shabelle. Columns: the gauges and the four models. Each model cell carries the season's
    rank on the model's reanalysis (large, coloured) and on its forecast archive (small, below)."""
    models = [NEW, "google_grrr", "glofas_v4", "glofas_v5"]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * 2 * N_RANK + 1.9))
    for _ in (season,):
        ms = [m for m in models if m in MODELS[season]]
        rows, ra_mat, fc_mat, ylab = [], [], [], []
        n_ra, n_fc = {}, {}
        for river in RIVERS:
            P, Rk = rank_table(season, river)
            top = Rk["gauge"].dropna().sort_values().index[:N_RANK]
            n_ra["gauge"] = int(Rk["gauge"].notna().sum())
            for m in ms:
                n_ra[m] = int(Rk[(m, "ra")].notna().sum())
                n_fc[m] = int(Rk[(m, "fc")].notna().sum()) if (m, "fc") in Rk.columns else 0
            for y in top:
                ra_mat.append([Rk.loc[y, "gauge"]] + [Rk[(m, "ra")].loc[y] for m in ms])
                fc_mat.append([np.nan] + [Rk[(m, "fc")].loc[y] if (m, "fc") in Rk.columns else np.nan for m in ms])
                ylab.append(f"{river.title()} {int(y)}   {100 * P.loc[y, 'gauge']:.0f}%")
                rows.append(river)
        ra_mat, fc_mat = np.array(ra_mat, dtype=float), np.array(fc_mat, dtype=float)
        shown = np.where(np.isnan(ra_mat), np.nan, 1 - (ra_mat - 1) / 24)
        ax.imshow(shown, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(ra_mat.shape[0]):
            for j in range(ra_mat.shape[1]):
                v = ra_mat[i, j]
                if np.isnan(v):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor=GRID, linewidth=0.8))
                    continue
                dark = shown[i, j] > 0.55
                f = fc_mat[i, j]
                if j == 0:
                    ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=9.5, color="white" if dark else BODY)
                else:
                    ax.text(j, i - 0.13, f"{int(v)}", ha="center", va="center", fontsize=9.5, color="white" if dark else BODY)
                    ax.text(j, i + 0.27, "forecasts " + (f"{int(f)}" if not np.isnan(f) else "none") if n_fc.get(ms[j - 1], 0) else "no archive",
                            ha="center", va="center", fontsize=6.5, color="white" if dark else FAINT)
        labels = [f"gauges\n{n_ra['gauge']}"] + [
            f"{SHORT[m].replace(', ', chr(10))}\n{n_ra[m]}" + (f" / {n_fc[m]}" if n_fc[m] else "") for m in ms]
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks(range(len(rows)), ylab, fontsize=9)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        ax.axhline(rows.count("juba") - 0.5, color=INK, linewidth=1.2)
    ax.set_title(f"{SEASON_TITLE[season]}", fontsize=11.5)
    fig.subplots_adjust(top=0.9, bottom=0.12)
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
        ax.set_xlim(0, 1250)
        ax.set_title(SEASON_TITLE[season])
        ax.set_xlabel("trigger level at the point, m³/s (ratio: new over current)")
        style_ax(ax, grid="x")
    axes[0].set_yticks(ys, [NAME[s] for s in STATIONS])
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    for ax in axes:
        ax.axhline(len(STATIONS) - len(TRIGGER_STATIONS["juba"]) - 0.5, color=INK, linewidth=0.9)
        ax.text(ax.get_xlim()[1] * 0.98, len(STATIONS) - 0.6, "Juba", ha="right", va="top", fontsize=9, color=FAINT)
        ax.text(ax.get_xlim()[1] * 0.98, len(STATIONS) - len(TRIGGER_STATIONS["juba"]) - 0.7, "Shabelle", ha="right", va="top", fontsize=9, color=FAINT)
    fig.subplots_adjust(wspace=0.06, top=0.84)
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
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 0.5 * len(sts) + 2.3))
    for ax, met, title in zip(axes, ["POD", "FAR", "F1"], ["POD: share of severe seasons\nwith the threshold reached", "FAR: share of threshold seasons\nwith no gauge flood", "F1"]):
        mat = [[float(S[(S.season == season) & (S.station == st) & (S.model == m)][met].iloc[0]) for m in ms] for st in sts]
        heat(ax, mat, [SHORT[m].replace(", ", chr(10)) for m in ms], [NAME[s] for s in sts] if met == "POD" else [""] * len(sts),
             cmap="Oranges" if met == "FAR" else "Blues")
        ax.set_title(title + "\n(dark = high)", fontsize=9.5)
    fig.suptitle(f"{SEASON_TITLE[season]}, reanalysis {rng(SCORE_RA)}, each point against its own gauge", x=0.02, ha="left", fontsize=11)
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


def score_table(R, season):
    """One table per season: a row per source and record, the two rivers side by side."""
    hdr = ("<table class='data'><thead><tr><th rowspan='2'>Source</th><th rowspan='2'>Record</th>"
           "<th colspan='5'>Juba</th><th colspan='5'>Shabelle</th></tr>"
           "<tr><th>severe seasons, threshold reached</th><th>threshold reached, no gauge flood</th><th>POD</th><th>FAR</th><th>F1</th>"
           "<th>severe seasons, threshold reached</th><th>threshold reached, no gauge flood</th><th>POD</th><th>FAR</th><th>F1</th></tr></thead><tbody>")
    out = [hdr]
    for m in MODELS[season]:
        for basis, span in (("ra", SCORE_RA), ("fc", SCORE_FC)):
            if basis == "fc" and m not in FC_SPAN:
                continue
            yr0 = years_of(R[("juba", season)], m, basis) & set(range(span[0], span[1] + 1))
            cells = [f"<td>{SHORT[m]}</td><td class='muted'>{'reanalysis' if basis == 'ra' else 'forecasts'} {min(yr0)} to {max(yr0)}</td>"]
            for river in RIVERS:
                rec = R[(river, season)]
                s = score(rec, m, basis, span)
                cells.append(f"<td>{len(s['caught'])} of {len(s['severe'])}</td><td>{len(s['false'])}</td>"
                             f"<td{shade(s['POD'])}>{num(s['POD'])}</td><td{shade(s['FAR'], True)}>{num(s['FAR'])}</td><td{shade(s['F1'])}>{num(s['F1'])}</td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def rank_html(season):
    """The gauges' N_RANK worst seasons per river and each source's own rank of them, as a table.
    Cell: rank on the reanalysis / rank on the forecasts. Shaded where the rank is within the top N_RANK."""
    models = [m for m in [NEW, "google_grrr", "glofas_v4", "glofas_v5"] if m in MODELS[season]]
    head = ("<table class='data rk'><thead><tr><th>Season</th><th>second gauge, % of its 1-in-3</th>"
            + "".join(f"<th>{SHORT[m]}<span class='y'>reanalysis / forecasts</span></th>" for m in models)
            + "</tr></thead><tbody>")
    out = [head]
    for river in RIVERS:
        P, Rk = rank_table(season, river)
        top = Rk["gauge"].dropna().sort_values().index[:N_RANK]
        out.append(f"<tr class='riv'><td colspan='{2 + len(models)}'>{river.title()}</td></tr>")
        for y in top:
            cells = [f"<td>{int(y)}</td><td>{100 * P.loc[y, 'gauge']:.0f}%</td>"]
            for m in models:
                ra = Rk[(m, "ra")].loc[y] if (m, "ra") in Rk.columns else np.nan
                fc = Rk[(m, "fc")].loc[y] if (m, "fc") in Rk.columns else np.nan
                has_fc = (m, "fc") in Rk.columns and y in set(range(FC_SPAN[m][0], FC_END.get((m, season), FC_SPAN[m][1]) + 1))
                if np.isnan(ra):
                    cells.append("<td class='muted'>no record</td>")
                    continue
                fc_txt = (f"{int(fc)}" if not np.isnan(fc) else "not reached") if has_fc else "no archive"
                cls = " class='top'" if (ra <= N_RANK) else ""
                cells.append(f"<td{cls}>{int(ra)} <span class='muted'>/ {fc_txt}</span></td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def findings(R, C):
    """Summary findings, generated from the numbers on the page. Short, facts only."""
    g = lambda s, r, m, b, span: score(R[(r, s)], m, b, span)
    top = lambda xs: ", ".join(x.split(" (")[0] for x in xs if int(x.split("(")[1].split()[0]) >= 3) or "none"
    out = [f"<b>In every source's top five:</b> Deyr Juba {top(C['deyr']['juba'][0])}, Shabelle {top(C['deyr']['shabelle'][0])}; "
           f"Gu Juba {top(C['gu']['juba'][0])}, Shabelle {top(C['gu']['shabelle'][0])}."]
    for season in ("deyr", "gu"):
        for basis, span, lab in (("ra", SCORE_RA, "reanalysis"), ("fc", SCORE_FC, "forecasts")):
            ms = [m for m in MODELS[season] if basis == "ra" or m in FC_SPAN]
            riv = []
            for river in RIVERS:
                n_sev = len(R[(river, season)]["severe"] & set(range(span[0], span[1] + 1)))
                caught = ", ".join(f"{SHORT[m].split(', ')[-1] if m != NEW else 'new'} {len(g(season, river, m, basis, span)['caught'])}" for m in ms)
                nf = ", ".join(f"{SHORT[m].split(', ')[-1] if m != NEW else 'new'} {len(g(season, river, m, basis, span)['false'])}" for m in ms)
                riv.append(f"{river.title()} ({n_sev} severe): threshold reached in severe seasons: {caught}; threshold reached with no gauge flood: {nf}")
            out.append(f"<b>{season.title()}, {lab}:</b> " + ". ".join(riv) + ".")
    # the new model against each other source, severe seasons with the threshold reached and seasons with no gauge flood
    for other, name in (("google_grrr", "the current Google model"), ("glofas_v4", "GloFAS v4"), ("glofas_v5", "GloFAS v5")):
        parts = []
        for season in ("deyr", "gu"):
            for basis, span, lab in (("ra", SCORE_RA, "reanalysis"), ("fc", SCORE_FC, "forecasts")):
                if basis == "fc" and other not in FC_SPAN:
                    continue
                for river in RIVERS:
                    a_, b_ = g(season, river, NEW, basis, span), g(season, river, other, basis, span)
                    dc, df_ = len(a_["caught"]) - len(b_["caught"]), len(a_["false"]) - len(b_["false"])
                    if dc == 0 and df_ == 0:
                        continue
                    bits = []
                    if dc:
                        bits.append(f"{abs(dc)} {'more' if dc > 0 else 'fewer'} severe season{'s' if abs(dc) > 1 else ''} reached")
                    if df_:
                        bits.append(f"{abs(df_)} {'more' if df_ > 0 else 'fewer'} season{'s' if abs(df_) > 1 else ''} with no gauge flood")
                    parts.append(f"{season.title()} {river.title()}, {lab}: " + " and ".join(bits))
        out.append(f"<b>New model against {name}:</b> " + ("; ".join(parts) if parts else "no difference") + ". Windows not listed are equal.")
    d = lambda r, s, m, b, y: R[(r, s)]["models"][m][b].get(y)
    bj, bs = R[("juba", "deyr")]["bench"].get(2023), R[("shabelle", "deyr")]["bench"].get(2023)
    lead = lambda x, b: (f"{(b - x).days} d before" if (b - x).days > 0 else f"{(x - b).days} d after") if ok(x) else "not reached"
    out.append(f"<b>Deyr 2023, forecasts against the gauges:</b> Juba, GloFAS v4 {lead(d('juba','deyr','glofas_v4','fc',2023), bj)}, new model {lead(d('juba','deyr',NEW,'fc',2023), bj)}; "
               f"Shabelle, GloFAS v4 {lead(d('shabelle','deyr','glofas_v4','fc',2023), bs)}, new model {lead(d('shabelle','deyr',NEW,'fc',2023), bs)}. The current Google archive ends June 2023.")
    tl = []
    for season in ("deyr", "gu"):
        bits = []
        for m in [NEW, "google_grrr", "glofas_v4"]:
            mean, med, n, n_all = timing_stats(R, season, "fc", m)
            if n:
                bits.append(f"{SHORT[m]} {'+' if mean >= 0 else ''}{mean:.0f} d ({n} of {n_all})")
            else:
                bits.append(f"{SHORT[m]} not reached")
        tl.append(f"{season.title()}: " + ", ".join(bits))
    out.append("<b>Average timing on the forecasts</b> (days before the second gauge crossed, five largest gauge floods per river, seasons reached): " + "; ".join(tl) + ".")
    out.append(f"<b>New model after 2023:</b> Deyr 2024 {after_2023(R, 'deyr', 2024)}. Gu 2024 {after_2023(R, 'gu', 2024)}. 2025: not reached in either season.")
    ratios = {}
    for season in ("deyr", "gu"):
        for river in RIVERS:
            rp, _ = cfg(river, season)
            rs = [L.model_threshold(NEW, st, season, rp) / L.model_threshold("google_grrr", st, season, rp) for st in TRIGGER_STATIONS[river]]
            ratios[(season, river)] = (min(rs), max(rs))
    out.append(f"<b>Scale:</b> new model levels are {ratios[('deyr','juba')][0]:.1f} to {ratios[('deyr','juba')][1]:.1f} times the current Google model's on the Juba in Deyr, "
               f"{ratios[('deyr','shabelle')][0]:.1f} to {ratios[('deyr','shabelle')][1]:.1f} on the Shabelle; in Gu {ratios[('gu','juba')][0]:.1f} to {ratios[('gu','juba')][1]:.1f} and {ratios[('gu','shabelle')][0]:.1f} to {ratios[('gu','shabelle')][1]:.1f}.")
    return "\n".join(f"<li>{x}</li>" for x in out)


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
    return f"{rec_txt} (gauges: {', '.join(outc)})"


def build():
    R = record()
    ST = station_scores()
    F = {}
    for s in ("deyr", "gu"):
        for b in ("ra", "fc"):
            F[f"bt_{s}_{b}"] = fig_backtest(R, s, b)
        F[f"tm_{s}"] = fig_timing(R, s)
        F[f"st_{s}"] = fig_stations(ST, s)
    F["lv"] = fig_levels()
    for k in (("deyr", 2023), ("deyr", 2019), ("gu", 2024)):
        F[f"tl_{k[0]}_{k[1]}"] = fig_timeline(R, *k)
    RK = {s: rank_html(s) for s in ("deyr", "gu")}
    C = {s: consistency(s) for s in ("deyr", "gu")}
    cons_txt = lambda xs: ", ".join(xs) if xs else "none"
    FIND = findings(R, C)
    T = {s: score_table(R, s) for s in ("deyr", "gu")}
    TS = timing_summary_html(R)

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
.takeaways {{ margin:18px 0 6px; padding:0; list-style:none; }}
.takeaways li {{ border-left:3px solid var(--b1); padding:0 0 0 14px; margin:0 0 14px; font-size:14px; }}
.takeaways b {{ color:var(--n9); }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media (max-width:760px) {{ .pair {{ grid-template-columns:1fr; }} }}
table.data {{ border-collapse:collapse; width:100%; margin:16px 0 22px; font-size:13px; }}
table.data caption {{ text-align:left; font-family:'Merriweather',Georgia,serif; font-size:14px; color:var(--n9); padding:0 0 8px; }}
table.data th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:7px 10px 6px; }}
table.data td {{ border-bottom:1px solid #eef1f1; padding:7px 10px; color:var(--n8); vertical-align:top; }}
table.data tr.riv td {{ color:var(--n9); font-weight:500; padding-top:12px; border-bottom:1px solid #d9dfdf; background:#fff; }}
table.data td:first-child, table.data td:nth-child(2) {{ white-space:nowrap; }}
table.data td:nth-child(7) {{ border-right:1px solid #e2e7e7; }}
table.data th[colspan] {{ text-align:center; border-bottom:1px solid #e2e7e7; }}
table.data .y {{ display:block; font-size:11px; color:var(--n7); font-weight:400; text-transform:none; letter-spacing:0; }}
table.data td.top {{ background:var(--b05); }}
table.tim th[colspan] {{ text-align:left; }}
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
      <p>Google's November 2026 river model at the seven trigger points, read with the trigger's own rules next to GloFAS v4,
         GloFAS v5 and the current Google model, on each model's reanalysis and on its forecast archive, against the SWALIM
         gauges. The live Flood Hub feed does not run the new model.</p>
    </div>
  </header>
  <article>
    <div class="callout">
      <p><b>Threshold rule:</b> three of the four Juba points, or two of the three Shabelle points, over their own level on the
         same day (Juba 1-in-{TRIGGER_CONFIG[('juba','deyr')]['rp']} and Shabelle 1-in-{TRIGGER_CONFIG[('shabelle','deyr')]['rp']} in Deyr, Juba 1-in-{TRIGGER_CONFIG[('juba','gu')]['rp']} and Shabelle 1-in-{TRIGGER_CONFIG[('shabelle','gu')]['rp']} in Gu, each model's
         own levels on its 1999 to 2023 seasonal maxima).</p>
      <p><b>Records:</b> <b>reanalysis</b>, the model's own record of the past; <b>forecasts</b>, the first forecast issue dated within the season that reaches the
         threshold at leads 1 to 7 days, which is what the live trigger reads. Archives: GloFAS v4 2003 to 2023, current Google 2016 to June 2023, new Google 2016 to 2025,
         GloFAS v5 none. Scores: {rng(SCORE_RA)} on the reanalysis, {rng(SCORE_FC)} on the forecasts.</p>
      <p><b>Gauge benchmark:</b> a flood is two SWALIM gauges on the river at their own 1-in-3 level, severe at 1-in-5. Gauge data run to July 2026.
         <b>Scores:</b> POD, the share of severe seasons with the threshold reached; FAR, the share of threshold seasons with no gauge flood; F1, the harmonic mean of POD and 1 minus FAR.</p>
    </div>

    <h2><span class="num">1</span>Summary of results</h2>
    <ul class="takeaways">
{FIND}
    </ul>

    <h2><span class="num">2</span>Benchmark: the largest floods at the gauges</h2>
    <p>The gauges' five worst seasons, worst first, and each source's own rank of them (1 = highest; reanalysis / forecasts).
       Index: the second-highest station's peak as a share of its own 1-in-3 on that source.</p>
    <h3>Deyr</h3>
    <div class="tablewrap">{RK['deyr']}</div>
    <p class="muted" style="font-size:12.5px">In the top five of every source that covers the season: Juba {cons_txt(C['deyr']['juba'][0])}; Shabelle {cons_txt(C['deyr']['shabelle'][0])}.
       In the top five of fewer than half: Juba {cons_txt(C['deyr']['juba'][1])}; Shabelle {cons_txt(C['deyr']['shabelle'][1])}.</p>
    <h3>Gu</h3>
    <div class="tablewrap">{RK['gu']}</div>
    <p class="muted" style="font-size:12.5px">In the top five of every source that covers the season: Juba {cons_txt(C['gu']['juba'][0])}; Shabelle {cons_txt(C['gu']['shabelle'][0])}.
       In the top five of fewer than half: Juba {cons_txt(C['gu']['juba'][1])}; Shabelle {cons_txt(C['gu']['shabelle'][1])}. Belet Weyne's Gu record is capped at 8.30 m in six seasons.
       Shaded cells: ranked in that source's top five on its reanalysis.</p>

    <h2><span class="num">3</span>Deyr: threshold outcome by season</h2>
    <figure class="chart"><img src="{F['bt_deyr_ra']}" alt="Deyr, reanalysis, season by season"><figcaption>Reanalysis. One row per model and river, one cell per season; blank cells are seasons outside the record or without a gauge record.</figcaption></figure>
    <figure class="chart"><img src="{F['bt_deyr_fc']}" alt="Deyr, forecasts, season by season"><figcaption>Forecasts, leads 1 to 7. Blank cells are seasons outside the archive or without a gauge record; the tallies count 2016 to 2023.</figcaption></figure>
    <div class="tablewrap">{T['deyr']}</div>
    <p class="muted" style="font-size:12.5px;margin-top:-10px">POD: share of severe seasons with the threshold reached. FAR: share of threshold seasons with no gauge flood.
       F1: the harmonic mean of POD and 1 minus FAR. Severe seasons counted are those in the record scored (gauges report from 2001 on the Juba and 2002 on the Shabelle). Years are in <code>data.json</code>.</p>
    <p class="muted" style="font-size:13px">New model, Deyr 2024: {after_2023(R, 'deyr', 2024)}. Deyr 2025: {after_2023(R, 'deyr', 2025)}.</p>

    <h2><span class="num">4</span>Gu: threshold outcome by season</h2>
    <figure class="chart"><img src="{F['bt_gu_ra']}" alt="Gu, reanalysis, season by season"><figcaption>Reanalysis.</figcaption></figure>
    <figure class="chart"><img src="{F['bt_gu_fc']}" alt="Gu, forecasts, season by season"><figcaption>Forecasts, leads 1 to 7.</figcaption></figure>
    <div class="tablewrap">{T['gu']}</div>
    <p class="muted" style="font-size:13px">New model, Gu 2024: {after_2023(R, 'gu', 2024)}. Gu 2025: {after_2023(R, 'gu', 2025)}.</p>

    <h2><span class="num">5</span>Timing relative to gauge confirmation</h2>
    <p>The gauges' five worst seasons per river. Zero is the day the second gauge crossed its 1-in-3; bars to the left reached the
       threshold before that, bars to the right after. Solid: reanalysis. Light: forecasts.</p>
    <figure class="chart"><img src="{F['tm_deyr']}" alt="Deyr timing"></figure>
    <figure class="chart"><img src="{F['tm_gu']}" alt="Gu timing"></figure>
    <h3>Average timing</h3>
    <p>Days between the threshold being reached and the second gauge crossing its 1-in-3, over the same five seasons per river,
       counting only seasons in which the source reached the threshold. Positive: before the gauges.</p>
    <div class="tablewrap">{TS}</div>

    <h2><span class="num">6</span>Event chronologies: Deyr 2023, Deyr 2019, Gu 2024</h2>
    <p>Gauge rows: circle, first day over the 1-in-3; diamond, first day over the 1-in-5; tick, peak. Source rows: threshold reached on
       the reanalysis (filled), first forecast issue reaching the threshold (hollow). FloodScan exposure where it covers the season.</p>
    <figure class="chart"><img src="{F['tl_deyr_2023']}" alt="Deyr 2023 day by day"></figure>
    <figure class="chart"><img src="{F['tl_deyr_2019']}" alt="Deyr 2019 day by day"></figure>
    <figure class="chart"><img src="{F['tl_gu_2024']}" alt="Gu 2024 day by day"></figure>

    <h2><span class="num">7</span>Skill by station</h2>
    <p>Each point against its own gauge, reanalysis {rng(SCORE_RA)}.</p>
    <figure class="chart"><img src="{F['st_deyr']}" alt="Deyr station scores"><figcaption>Deyr.</figcaption></figure>
    <figure class="chart"><img src="{F['st_gu']}" alt="Gu station scores"><figcaption>Gu.</figcaption></figure>

    <h2><span class="num">8</span>Trigger levels of the two Google models</h2>
    <figure class="chart"><img src="{F['lv']}" alt="Trigger levels, current against new"><figcaption>The trigger level at each point on the current Google model and on the new one, both fitted the same way.</figcaption></figure>
    <h2><span class="num">9</span>Data and method notes</h2>
    <ul class="notes">
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
  </article>
</div>
</body>
</html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("written", OUT / "index.html")
    print(pd.DataFrame(sc)[["season", "river", "model", "basis", "span", "caught", "false", "POD", "FAR", "F1"]].to_string(index=False))


if __name__ == "__main__":
    build()
