"""Build pages/google-model-review/: a review of Google's November 2026 river model at the seven
Somalia stations against the SWALIM gauges, beside Google's 2024 model, GloFAS v4 and GloFAS v5.
Every score is given by season, river and station; the largest gauge floods are drawn day by day.
The page states what the records show and draws no conclusion.

Inputs (as build_google_model_2026_page.py): SOM_DATA_REPO (processed data), SOM_GRRR_2026_DIR
(the new model's seven-station extracts) and SOM_OBS_Q_DIR (SWALIM discharge series, four stations).

Usage (from repo root):
    .venv/Scripts/python.exe scripts/page_additions/build_google_model_review_page.py
"""
import datetime as dt
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_google_model_2026_page as G  # noqa: E402
import google_review_metrics as R  # noqa: E402
from src.constants import BODY, FAINT, GRID, INK  # noqa: E402

L, NEW = G.L, G.NEW
OUT = HERE.parents[1] / "pages" / "google-model-review"
FIGS = OUT / "figures"
MODELS = R.MODELS
SHORT, COL, NAME = G.SHORT, G.COL, G.NAME
STATIONS, RIVER_OF = G.STATIONS, G.RIVER_OF
N_JUBA = len(G.TRIGGER_STATIONS["juba"])
SEASON_TITLE = G.SEASON_TITLE
PNG_DIR = os.environ.get("SOM_FIG_PNG_DIR")


def save(fig, name, tight=True):
    FIGS.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.25) if tight else {}
    fig.savefig(FIGS / f"{name}.svg", format="svg", **kw)
    if PNG_DIR:
        Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PNG_DIR) / f"{name}.png", format="png", dpi=120, **kw)
    plt.close(fig)
    return f"figures/{name}.svg"


# ----------------------------------------------------------------------------- grids
def _ink(face):
    """Dark or white text, by the brightness of the cell."""
    r, g, b = face[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.52 else BODY


def _colour(v, kind):
    """Cell colour. good: a share from 0 to 1 (green, dark = high). corr: the same over 0.4 to 1.
    auc: over 0.5 to 1. bad: a share from 0 to 1 (grey, dark = high). eff: green above 0, brown below.
    days: signed days, teal = model ahead, brown = behind. ratio: teal above 1, brown below."""
    if v is None or np.isnan(v):
        return (1, 1, 1, 1)
    clip = lambda x: min(max(x, 0.0), 1.0)
    if kind == "good":
        return plt.cm.Greens(0.06 + 0.82 * clip(v))
    if kind == "corr":
        return plt.cm.Greens(0.06 + 0.82 * clip((v - 0.4) / 0.6))
    if kind == "auc":
        return plt.cm.Greens(0.06 + 0.82 * clip((v - 0.5) / 0.5))
    if kind == "bad":
        return plt.cm.Greys(0.05 + 0.72 * clip(v))
    if kind == "eff":
        return plt.cm.Greens(0.10 + 0.78 * clip(v)) if v >= 0 else plt.cm.BrBG(0.42 - 0.34 * clip(-v))
    if kind == "days":
        x = min(max(v / 15, -1), 1)
        return plt.cm.BrBG(0.5 + 0.40 * x)
    if kind == "ratio":
        x = min(max(np.log2(v), -1), 1) if v > 0 else -1
        return plt.cm.BrBG(0.5 + 0.40 * x)
    return (1, 1, 1, 1)


def signed(v):
    """Whole days with a sign, and a plain 0."""
    r = int(np.floor(abs(v) + 0.5)) * (1 if v > 0 else -1)
    return "0" if r == 0 else f"{r:+d}"


def grid_fig(df, col, name, title, kind, fmt="{:.2f}", sub=None, note=None, rows=None, panels=None):
    """One figure: a panel per season (or per period), a row per station, a column per source."""
    panels = panels or [("deyr", SEASON_TITLE["deyr"]), ("gu", SEASON_TITLE["gu"])]
    rows = rows or STATIONS
    n_j = sum(RIVER_OF[st] == "juba" for st in rows)
    pcol = "season" if "season" in df.columns else "period"
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels) + 1.2, 0.5 * len(rows) + 2.0))
    axes = np.atleast_1d(axes)
    for ax, (pk, ptitle) in zip(axes, panels):
        d = df[df[pcol] == pk]
        for i, st in enumerate(rows):
            for j, m in enumerate(MODELS):
                r = d[(d.station == st) & (d.model == m)]
                v = float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan
                face = _colour(v, kind)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", linewidth=1.6))
                if np.isnan(v):
                    ax.text(j, i, "no record" if not len(r) else "n/a", ha="center", va="center", fontsize=8, color=FAINT)
                    continue
                ink = _ink(face)
                s_ = sub(r.iloc[0]) if sub else None
                ax.text(j, i - (0.13 if s_ else 0), fmt(v) if callable(fmt) else fmt.format(v), ha="center", va="center", fontsize=10, color=ink)
                if s_:
                    ax.text(j, i + 0.26, s_, ha="center", va="center", fontsize=7.5, color=ink)
        ax.set_xlim(-0.5, len(MODELS) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(range(len(MODELS)), [SHORT[m].replace(", ", "\n") for m in MODELS], fontsize=9.5, color=BODY)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)), [NAME[st] for st in rows] if ax is axes[0] else [""] * len(rows), fontsize=10, color=BODY)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if 0 < n_j < len(rows):
            ax.axhline(n_j - 0.5, color=INK, linewidth=1.0)
            if ax is axes[0]:
                ax.text(-1.72, (n_j - 1) / 2, "JUBA", ha="center", va="center", rotation=90, fontsize=8.5, color=FAINT, clip_on=False)
                ax.text(-1.72, n_j + (len(rows) - n_j - 1) / 2, "SHABELLE", ha="center", va="center", rotation=90, fontsize=8.5, color=FAINT, clip_on=False)
        ax.set_title(ptitle, fontsize=11, loc="left", pad=34)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, y=1.02)
    if note:
        fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8.5, color=FAINT)
    fig.subplots_adjust(wspace=0.06, left=0.16 if len(panels) == 2 else 0.12, right=0.99, top=0.78, bottom=0.03)
    return save(fig, name)


# ---------------------------------------------------------------------- worst-year series
def fig_ts(season, year, variant):
    """One season at the seven stations: the gauge and the models as a share of their own 1-in-3
    level. variant 'new': gauge, the new model's reanalysis and its forecast issued 7 days ahead.
    variant 'all': gauge and the four reanalyses."""
    a, b = G.WINDOW[season]
    A, B = pd.Timestamp(f"{year}-{a}"), pd.Timestamp(f"{year}-{b}")
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.6), sharey=True)
    order = [(0, k, st) for k, st in enumerate(G.TRIGGER_STATIONS["juba"])] + [(1, k, st) for k, st in enumerate(G.TRIGGER_STATIONS["shabelle"])]
    top = 130.0
    series = {}
    for r_, c_, st in order:
        g = R.gauge_daily(st, season)
        g = g[g.index.year == year]
        gl = R.gauge_level(st, season, 3)
        lines = []
        if len(g) and not np.isnan(gl):
            lines.append(("gauge", 100 * g / gl))
        ms = [NEW] if variant == "new" else MODELS
        for m in ms:
            if year > G.RA_SPAN[m][1]:
                continue
            x = R.model_daily(m, st, season, y1=G.RA_SPAN[m][1])
            x = x[x.index.year == year]
            if len(x):
                lines.append((m, 100 * x / R.model_level(m, st, season, 3)))
        if variant == "new" and year >= G.FC_SPAN[NEW][0]:
            f = R.lead_series(NEW, st, season, 7)
            f = f[f.index.year == year]
            if len(f):
                lines.append(("fc7", 100 * f / R.model_level(NEW, st, season, 3)))
        series[st] = lines
        for _, s in lines:
            top = max(top, float(s.max()) * 1.06)
    for r_, c_, st in order:
        ax = axes[r_, c_]
        ax.axhline(100, color=INK, lw=1.0, ls=":", zorder=1)
        have_gauge = False
        for key, s in series[st]:
            if key == "gauge":
                ax.plot(s.index, s.values, color=INK, lw=2.0, zorder=4)
                have_gauge = True
            elif key == "fc7":
                ax.plot(s.index, s.values, color=COL[NEW], lw=1.3, ls=(0, (4, 2)), zorder=3)
            else:
                ax.plot(s.index, s.values, color=COL[key], lw=2.0 if key == NEW else 1.2, zorder=3 if key == NEW else 2,
                        alpha=1.0 if key == NEW else 0.9)
        ax.set_title(NAME[st], fontsize=11, loc="left")
        if not have_gauge:
            ax.set_title("no gauge record this season", fontsize=9, loc="right", color=FAINT, fontweight="normal")
        ax.set_xlim(A, B)
        ax.set_ylim(0, top)
        ax.set_xticks([pd.Timestamp(year, mth, 1) for mth in G.SEASONS[season]])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        for lab_ in ax.get_xticklabels():
            lab_.set_horizontalalignment("left")
        ax.tick_params(length=0, labelsize=9, colors=BODY)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        if c_ == 0:
            ax.set_ylabel(f"{'Juba' if r_ == 0 else 'Shabelle'}\n% of own 1-in-3 level", fontsize=9.5, color=BODY)
    axes[1, 3].axis("off")
    H = [Line2D([], [], color=INK, lw=2.0, label="SWALIM gauge level"),
         Line2D([], [], color=COL[NEW], lw=2.0, label="Google, new: reanalysis")]
    if variant == "new":
        if year >= G.FC_SPAN[NEW][0]:
            H.append(Line2D([], [], color=COL[NEW], lw=1.3, ls=(0, (4, 2)), label="Google, new: forecast, 7 days ahead"))
    else:
        H += [Line2D([], [], color=COL[m], lw=1.2, label=f"{SHORT[m]}: reanalysis") for m in MODELS[1:] if year <= G.RA_SPAN[m][1]]
    H.append(Line2D([], [], color=INK, lw=1.0, ls=":", label="100%: each line's own 1-in-3 level"))
    axes[1, 3].legend(handles=H, loc="center left", bbox_to_anchor=(-0.05, 0.5), fontsize=10, frameon=False)
    fig.suptitle(f"{season.title()} {year}", x=0.01, ha="left", fontsize=13)
    fig.subplots_adjust(wspace=0.08, hspace=0.32, left=0.07, right=0.99, top=0.9, bottom=0.07)
    return save(fig, f"ts_{season}_{year}_{variant}", tight=False)


# --------------------------------------------------------------------------- by lead
def lead_table(BL):
    g = BL.groupby(["model", "season", "river", "lead"]).agg(
        peak=("peak_ratio_median", "median"), rho=("rho_vs_gauge", "mean"), rho_ra=("rho_vs_reanalysis", "mean"),
        a=("hits", "sum"), b=("false_alarms", "sum"), c=("misses", "sum")).reset_index()
    g["POD"] = g.a / (g.a + g.c)
    g["FAR"] = g.b / (g.a + g.b)
    return g


def fig_lead(g, col, name, ylabel, ylim, ref=None):
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9), sharey=True)
    k = 0
    for season in ("deyr", "gu"):
        for river in G.RIVERS:
            ax = axes[k]
            k += 1
            for m in (NEW, "google_grrr", "glofas_v4"):
                d = g[(g.model == m) & (g.season == season) & (g.river == river)].sort_values("lead")
                ax.plot(d.lead, d[col], color=COL[m], lw=2.2 if m == NEW else 1.5, marker="o", ms=4 if m == NEW else 3, label=SHORT[m])
            if ref is not None:
                ax.axhline(ref, color=INK, lw=0.9, ls=":")
            ax.set_title(f"{season.title()}, {river.title()}", fontsize=11, loc="left")
            ax.set_xticks(range(1, 11))
            ax.set_xlim(0.6, 10.4)
            ax.set_ylim(ylim[0] - 0.04 * (ylim[1] - ylim[0]), ylim[1])
            ax.set_xlabel("days ahead", fontsize=9.5, color=BODY)
            ax.tick_params(length=0, labelsize=9, colors=BODY)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel(ylabel, fontsize=9.5, color=BODY)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=3, fontsize=9.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.07, left=0.06, right=0.99, top=0.88, bottom=0.27)
    return save(fig, name, tight=False)


# ------------------------------------------------------------------------------ html
def num(v, f="{:.2f}"):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def yrs(xs):
    return ", ".join(str(int(y)) for y in xs) if len(xs) else "none"


def river_table(Rv, rp, season):
    head = ("<table class='data'><thead><tr><th>River</th><th>Source</th><th>Record</th><th>Seasons</th><th>Gauge floods</th>"
            "<th>Reached in a flood</th><th>Reached, no flood</th><th>Flood not reached</th>"
            "<th>POD</th><th>FAR</th><th>POFD</th><th>CSI</th><th>F1</th><th>Frequency bias</th><th>HSS</th>"
            "<th>Days against the gauges<span class='y'>median, floods reached</span></th></tr></thead><tbody>")
    out = [head]
    d = Rv[(Rv.rp == rp) & (Rv.season == season)]
    for river in G.RIVERS:
        first = True
        for m in MODELS:
            for rec in [x for x in d[d.model == m].record.unique()]:
                r = d[(d.river == river) & (d.model == m) & (d.record == rec)]
                if not len(r):
                    continue
                r = r.iloc[0]
                lead = r.lead_days_median
                lead_txt = "n/a" if np.isnan(lead) else ("same day" if round(lead) == 0 else f"{abs(lead):.0f} {'before' if lead > 0 else 'after'}")
                cls = " class='sep'" if first else ""
                out.append(f"<tr{cls}><td>{river.title() if first else ''}</td><td>{SHORT[m]}</td><td class='muted'>{rec}</td>"
                           f"<td>{int(r.n)}</td><td>{int(r.n_floods)}</td>"
                           f"<td>{int(r.hits)}<span class='y'>{yrs(r.hit_years)}</span></td>"
                           f"<td>{int(r.false_alarms)}<span class='y'>{yrs(r.false_years)}</span></td>"
                           f"<td>{int(r.misses)}<span class='y'>{yrs(r.missed_years)}</span></td>"
                           f"<td>{num(r.POD)}</td><td>{num(r.FAR)}</td><td>{num(r.POFD)}</td><td>{num(r.CSI)}</td><td>{num(r.F1)}</td>"
                           f"<td>{num(r.freq_bias)}</td><td>{num(r.HSS)}</td><td>{lead_txt}</td></tr>")
                first = False
    out.append("</tbody></table>")
    return "\n".join(out)


LONG = {"POD": "POD, probability of detection", "FAR": "FAR, false alarm ratio", "POFD": "POFD, probability of false detection",
        "CSI": "CSI, critical success index", "F1": "F1", "freq_bias": "Frequency bias", "accuracy": "Accuracy",
        "HSS": "HSS, Heidke skill score", "PSS": "PSS, Peirce skill score", "KGE": "KGE, Kling-Gupta efficiency", "NSE": "NSE, Nash-Sutcliffe efficiency"}
METRICS = [  # key, label, colour kind, format, one-sentence meaning
    ("POD", "POD", "good", "{:.2f}", "POD, probability of detection: of the flood seasons at the gauge, the share in which the model reached its own threshold. 1 means the threshold was reached in every flood season; 0 means in none."),
    ("FAR", "FAR", "bad", "{:.2f}", "FAR, false alarm ratio: of the seasons in which the model reached its threshold, the share with no flood at the gauge. 0 means no false alarm."),
    ("POFD", "POFD", "bad", "{:.2f}", "POFD, probability of false detection: of the seasons with no flood at the gauge, the share in which the model still reached its threshold. 0 means the threshold was never reached without a flood."),
    ("CSI", "CSI", "good", "{:.2f}", "CSI, critical success index: the floods reached, divided by the floods reached plus the floods not reached plus the false alarms. It falls with either kind of error; 1 means neither occurred."),
    ("F1", "F1", "good", "{:.2f}", "F1: twice the floods reached, divided by twice the floods reached plus the floods not reached plus the false alarms. It is high only when POD is high and FAR is low; 1 means every flood was reached with no false alarm."),
    ("freq_bias", "Frequency bias", "ratio", "{:.2f}", "Frequency bias: seasons in which the model reached its threshold divided by flood seasons at the gauge. Above 1 the model reaches it more often than the gauge floods; below 1, less often."),
    ("accuracy", "Accuracy", "good", "{:.2f}", "Accuracy: the share of all seasons in which model and gauge agree, flood or no flood."),
    ("HSS", "HSS", "eff", "{:.2f}", "HSS, Heidke skill score: accuracy with the agreement expected by chance taken out. 0 means no more agreement than chance would give; 1 means model and gauge agree in every season."),
    ("PSS", "PSS", "eff", "{:.2f}", "PSS, Peirce skill score: POD minus POFD. 0 means the model does not separate flood seasons from the rest; 1 means it separates them fully."),
]
Q_METRICS = [
    ("ratio_median", "Ratio of medians", "ratio", "{:.2f}", "Ratio of medians: the model's typical flow divided by SWALIM's typical flow on the same days. 1 means equal; 2 means the model runs at twice the measured flow."),
    ("ratio_peak_median", "Ratio of season peaks", "ratio", "{:.2f}", "Ratio of season peaks: the model's highest flow of each season divided by SWALIM's, middle value over the seasons. SWALIM's series stops rising at bank-full, so a ratio above 1 can come from that ceiling."),
    ("pearson_r", "Correlation", "good", "{:.2f}", "Correlation of daily values, from 0 (no relation) to 1 (the two rise and fall together in proportion)."),
    ("variability_ratio", "Variability ratio", "ratio", "{:.2f}", "Variability ratio: the size of the model's swings divided by the size of SWALIM's. 1 means the same spread."),
    ("KGE", "KGE", "eff", "{:.2f}", "KGE, Kling-Gupta efficiency: one score from three parts, the correlation, the variability ratio and the ratio of averages. 1 is perfect; below about -0.4 the model is no closer than the long-term average would be."),
    ("NSE", "NSE", "eff", "{:.2f}", "NSE, Nash-Sutcliffe efficiency: 1 is perfect, 0 is no closer than the average of the measurements, and negative is further away than that average."),
]


def pick(group, dim, options, cls="pick"):
    """A row of buttons. options: [(key, label)]; the first is on."""
    return (f"<div class='{cls}' data-group='{group}' data-dim='{dim}'>"
            + "".join(f"<button type='button' data-key='{k}'" + (" class='on'" if i == 0 else "") + f">{lab}</button>" for i, (k, lab) in enumerate(options))
            + "</div>")


def sw(group, attrs, inner, tag="div"):
    a = " ".join(f"data-{k}='{v}'" for k, v in attrs.items())
    return f"<{tag} class='sw' data-group='{group}' {a}>{inner}</{tag}>"


def fig_html(src, alt, cap=""):
    return f"<figure class='chart'><img src='{src}' alt='{alt}' loading='lazy'>{'<figcaption>' + cap + '</figcaption>' if cap else ''}</figure>"


def findings(M, LT):
    T, E3, Rv, Q, BL = M["tracking"], M["station_3"], M["river_3"], M["discharge"], M["by_lead"]
    out = []
    # 1 daily agreement
    piv = T.pivot_table(index=["season", "station"], columns="model", values="rho_same_day")
    top = int((piv.idxmax(axis=1) == NEW).sum())
    out.append(f"<strong>Day-to-day agreement with the gauges.</strong> Rank correlation runs from 0 (no relation) to 1 (the model's highest days are the "
               f"gauge's highest days). The new model has the highest value of the four sources at {top} of the {len(piv)} combinations of station and season, "
               f"between {piv[NEW].min():.2f} and {piv[NEW].max():.2f}.")
    # 2 position of the curve against the gauge
    lag = T[T.model == NEW].set_index(["season", "river", "station"]).best_lag_days

    def lag_txt(season, river):
        v = lag.loc[(season, river)]
        lo, hi = int(v.min()), int(v.max())
        days = lambda a, b: (f"{a} day" + ("" if a == 1 else "s")) if a == b else f"{a} to {b} days"
        where = f"{river.title()} in {season.title()}"
        if hi < 0:
            return f"{where}: {days(-hi, -lo)} behind the gauge"
        if lo > 0:
            return f"{where}: {days(lo, hi)} ahead of the gauge"
        return f"{where}: between {-lo} days behind and {hi} days ahead"
    out.append("<strong>Where the new model's daily curve sits against the gauge's</strong> (the shift at which the two agree most). "
               + "; ".join(lag_txt(s, r) for s in ("deyr", "gu") for r in G.RIVERS) + ".")
    # 3 station events, the seven stations added up
    parts = []
    for season in ("deyr", "gu"):
        d = E3[(E3.model == NEW) & (E3.season == season)]
        parts.append(f"{season.title()}: the gauges had {int(d.n_floods.sum())} flood seasons and the model reached its threshold in {int(d.hits.sum())} of them; "
                     f"it reached its threshold {int(d.n_reached.sum())} times, {int(d.false_alarms.sum())} of them with no flood at the gauge")
    jo, others = E3[E3.station == "jowhar"], E3[E3.station != "jowhar"]
    out.append("<strong>Flood seasons at single stations (1-in-3), new model, the seven stations added up.</strong> " + ". ".join(parts)
               + f". At Jowhar every source reaches its threshold in fewer than half of the flood seasons, in both seasons (POD {jo.POD.min():.2f} to {jo.POD.max():.2f}; "
               f"at the other six stations {others.POD.min():.2f} to {others.POD.max():.2f}).")
    # 4 river rule
    def rr(season, rec):
        d = Rv[(Rv.model == NEW) & (Rv.season == season) & (Rv.record == rec) & (Rv.rp == 3)]
        return int(d.hits.sum()), int(d.n_floods.sum()), int(d.false_alarms.sum())
    fa = lambda n: f"{n} false alarm" + ("" if n == 1 else "s")
    parts = []
    for season in ("deyr", "gu"):
        a, b = rr(season, "reanalysis"), rr(season, "forecasts")
        parts.append(f"{season.title()}: on the reanalysis the threshold was reached in {a[0]} of {a[1]} river floods, with {fa(a[2])}; on the forecasts in {b[0]} of {b[1]}, with {fa(b[2])}")
    out.append("<strong>The trigger rule by river (1-in-3), new model, the two rivers added up.</strong> " + ". ".join(parts) + ".")
    # 5 forecasts against the reanalysis
    d = LT[LT.model == NEW]
    st = BL[BL.model == NEW]
    hits = st.groupby("lead").hits.sum()
    fal = st.groupby("lead").false_alarms.sum()
    out.append(f"<strong>Forecasts.</strong> From 1 to 10 days ahead, the new model's forecast peak is {d.peak.min():.2f} to {d.peak.max():.2f} times its reanalysis peak "
               f"(middle value over the stations of a river; single stations {st.peak_ratio_median.min():.2f} to {st.peak_ratio_median.max():.2f}). "
               f"Added over all stations, the floods reached stay at {int(hits.min())} to {int(hits.max())} and the false alarms at {int(fal.min())} to {int(fal.max())} whatever the lead. "
               "The forecast archive is close to a copy of the reanalysis.")
    # 6 size of the flows
    q = Q[(Q.period == "year")].pivot_table(index="station", columns="model", values="ratio_median")
    three = [s for s in ("luuq", "belet_weyne", "bulo_burti") if s in q.index]
    names = ", ".join(NAME[s] for s in three[:-1]) + " and " + NAME[three[-1]]
    rg = lambda m: f"{q.loc[three, m].min():.1f} to {q.loc[three, m].max():.1f}"
    out.append(f"<strong>Size of the flows against SWALIM's discharge series, over the whole year.</strong> At {names} the new model's typical flow is "
               f"{rg(NEW)} times the measured one (GloFAS v5 {rg('glofas_v5')}, GloFAS v4 {rg('glofas_v4')}, Google, current {rg('google_grrr')}). "
               f"At Bardheere it is {q.loc['bardheere', NEW]:.1f} times.")
    return "\n".join(f"<li>{x}</li>" for x in out)


def build():
    FIGS.mkdir(parents=True, exist_ok=True)
    M = R.compute()
    T, Q, BL = M["tracking"], M["discharge"], M["by_lead"]
    LT = lead_table(BL)
    Rv = pd.concat([M["river_3"], M["river_5"]], ignore_index=True)
    F = {}

    # worst years
    ts_opts = []
    ts_figs = []
    for season in ("deyr", "gu"):
        for y in M["worst"][season]:
            key = f"{season}_{y}"
            ts_opts.append((key, f"{season.title()} {y}"))
            for variant in ("new", "all"):
                src = fig_ts(season, y, variant)
                ts_figs.append(sw("ts", {"year": key, "lines": variant}, fig_html(src, f"{season.title()} {y}, seven stations, {variant}")))

    # tracking grids
    seasons_note = "Seasons with a gauge record, 1999 to 2023. Dark = high; the shading runs from 0.4 to 1."
    ahead_note = "Plus: the model ahead of the gauge. Minus: behind. Teal = ahead, brown = behind."
    F["rho0"] = grid_fig(T, "rho_same_day", "rho_same_day", "Rank correlation of daily values, model and gauge on the same day", "corr", note=seasons_note)
    F["rhob"] = grid_fig(T, "rho_best_lag", "rho_best_lag", "Rank correlation of daily values with the model's curve slid to where the value is highest (10 days behind to 30 ahead)", "corr",
                         sub=lambda r: f"{int(r.best_lag_days):+d} d", note=seasons_note + " Small figure: the shift in days; plus means the model's curve runs ahead of the gauge's.")
    F["rhof"] = grid_fig(T, "rho_flood_seasons", "rho_flood_seasons", "Rank correlation of daily values, slid as above, on flood seasons only (gauge or model over its own 1-in-3), 1999 to 2023", "corr",
                         sub=lambda r: f"{int(r.n_flood_seasons)} seasons", note="Dark = high; the shading runs from 0.4 to 1. Small figure: seasons used, which differ by source because each source has its own flood seasons.")
    F["pk"] = grid_fig(T.assign(v=-T.peak_days_late_median), "v", "peak_days", "Days from the model's peak to the gauge's peak, flood seasons at the gauge (middle value)", "days", fmt=signed,
                       sub=lambda r: f"{int(r.n_peaks)} seasons", note=ahead_note + " Small figure: seasons used.")
    F["cx"] = grid_fig(T, "crossing_days_early_median", "crossing_days", "Days from the model reaching its 1-in-3 to the gauge reaching its own (middle value)", "days", fmt=signed,
                       sub=lambda r: f"{int(r.n_crossings)} seasons", note=ahead_note + " Small figure: seasons in which both reach their level.")

    # ordering
    E3, E5 = M["station_3"], M["station_5"]
    F["rmax"] = grid_fig(E3, "rho_season_peaks", "rho_season_peaks", "Rank correlation of season peaks, model against gauge", "good", note="Seasons with a gauge record, 1999 to 2023. Dark = high.")
    auc_note = "0.5 is a coin toss. Dark = high; the shading runs from 0.5 to 1."
    F["auc3"] = grid_fig(E3, "AUC", "auc_3", "AUC: chance that a flood season has a higher model peak than a season without a flood. Flood = gauge over its 1-in-3", "auc", note=auc_note)
    F["auc5"] = grid_fig(E5, "AUC", "auc_5", "AUC: chance that a flood season has a higher model peak than a season without a flood. Flood = gauge over its 1-in-5", "auc", note=auc_note)

    # station events
    ev_blocks = []
    for rp, E in ((3, E3), (5, E5)):
        for key, lab, kind, fmt, _ in METRICS:
            sub = None
            if key == "POD":
                sub = lambda r: f"{int(r.hits)} of {int(r.n_floods)}"
            elif key == "FAR":
                sub = lambda r: f"{int(r.false_alarms)} of {int(r.n_reached)}"
            elif key == "POFD":
                sub = lambda r: f"{int(r.false_alarms)} of {int(r.false_alarms + r.correct_negatives)}"
            elif key in ("CSI", "F1"):
                sub = lambda r: f"{int(r.hits)} / {int(r.misses)} / {int(r.false_alarms)}"
            elif key == "accuracy":
                sub = lambda r: f"{int(r.hits + r.correct_negatives)} of {int(r.n)}"
            note = {"good": "Dark = high.", "bad": "Dark = high.", "eff": "Green: above 0. Brown: below 0.", "ratio": "Teal: above 1. Brown: below 1."}[kind]
            note += {"POD": " Small figures: flood seasons with the threshold reached, of flood seasons at the gauge.",
                     "FAR": " Small figures: seasons with no flood, of seasons with the threshold reached.",
                     "POFD": " Small figures: seasons with the threshold reached, of seasons with no flood.",
                     "CSI": " Small figures: floods reached / floods not reached / false alarms.",
                     "F1": " Small figures: floods reached / floods not reached / false alarms.",
                     "accuracy": " Small figures: seasons in which model and gauge agree, of all seasons."}.get(key, "")
            src = grid_fig(E, key, f"st_{key}_{rp}", f"{LONG[key]}. Flood season = the station's gauge over its 1-in-{rp}", kind, fmt=fmt, sub=sub, note=note)
            ev_blocks.append(sw("ev", {"metric": key, "level": rp}, fig_html(src, f"{lab} by station, 1-in-{rp}")))
    ev_text = "".join(sw("ev", {"metric": key}, text, tag="p") for key, _, _, _, text in METRICS)

    # river rule
    Rrec = G.record()
    G.FIGS = FIGS
    for s in ("deyr", "gu"):
        F[f"bt_{s}"] = G.fig_backtest(Rrec, s)
    rv_blocks = "".join(sw("rv", {"level": rp}, f"<h3>{s.title()}</h3><div class='tablewrap'>{river_table(Rv, rp, s)}</div>")
                        for rp in (3, 5) for s in ("deyr", "gu"))

    # by lead
    F["lead_peak"] = fig_lead(LT, "peak", "lead_peak", "forecast peak / reanalysis peak", (0.8, 1.25), ref=1.0)
    F["lead_rho"] = fig_lead(LT, "rho", "lead_rho", "rank correlation with the gauge", (0, 1.0))
    F["lead_pod"] = fig_lead(LT, "POD", "lead_pod", "share of flood seasons with the threshold reached (POD)", (0, 1.0))
    F["lead_far"] = fig_lead(LT, "FAR", "lead_far", "share of threshold seasons with no flood (FAR)", (0, 1.0))
    d7 = BL[(BL.model == NEW) & (BL.lead == 7)]

    # discharge
    q_rows = [s for s in ("luuq", "bardheere", "belet_weyne", "bulo_burti") if s in set(Q.station)]
    q_panels = [("deyr", SEASON_TITLE["deyr"]), ("gu", SEASON_TITLE["gu"]), ("year", "Whole year")]
    q_blocks = []
    for key, lab, kind, fmt, _ in Q_METRICS:
        note = {"good": "Dark = high.", "eff": "Green: above 0. Brown: below 0; -1 and lower take the darkest brown.", "ratio": "Teal: above 1. Brown: below 1; the shading is full at 2 and at 0.5."}[kind]
        src = grid_fig(Q, key, f"q_{key}", f"{LONG.get(key, lab)}: model flow against SWALIM's discharge series, 1999 to 2023", kind, fmt=fmt, note=note, rows=q_rows, panels=q_panels)
        q_blocks.append(sw("q", {"metric": key}, fig_html(src, f"{lab} against SWALIM discharge")))
    q_text = "".join(sw("q", {"metric": key}, text, tag="p") for key, _, _, _, text in Q_METRICS)
    ceil = []
    for st in q_rows:
        o = R.obs_discharge(st)
        n_top = int((o >= o.max() * 0.999).sum())
        if n_top >= 10:
            ceil.append(f"{NAME[st]} on {n_top} days")
    CEIL_TXT = ("The series repeats its highest value at " + " and ".join(ceil) + ", which is what a ceiling in the measurement looks like, so the largest floods may be higher than the series shows."
                if ceil else "")

    # data file
    PUB = {"hits": "floods_reached", "misses": "floods_not_reached", "correct_negatives": "neither", "hit_years": "years_flood_reached",
           "false_years": "years_false_alarm", "missed_years": "years_flood_not_reached"}

    def recs(df):
        return json.loads(df.rename(columns=PUB).to_json(orient="records"))
    (OUT / "data.json").write_text(json.dumps(G.clean(dict(
        sources={m: SHORT[m] for m in MODELS},
        worst_years=M["worst"],
        daily_agreement=recs(T), station_floods_1in3=recs(E3), station_floods_1in5=recs(E5),
        river_rule=recs(Rv), forecasts_by_lead=recs(BL), discharge=recs(Q))), indent=1))
    for nm, df in (("daily_agreement", T), ("station_floods_1in3", E3), ("station_floods_1in5", E5), ("river_rule", Rv), ("forecasts_by_lead", BL), ("discharge", Q)):
        df.assign(model=df.model.map(SHORT)).rename(columns=PUB).to_csv(OUT / f"{nm}.csv", index=False)

    today = dt.date.today().isoformat()
    FIND = findings(M, LT)
    n_lo, n_hi, f_lo, f_hi = int(E3.n.min()), int(E3.n.max()), int(E3.n_floods.min()), int(E3.n_floods.max())
    short = E3[E3.n < 15].groupby("station").n.max()
    SAMPLE_TXT = (f"The counts are small: a station has {n_lo} to {n_hi} seasons with a gauge record"
                  + (" (" + ", ".join(f"{NAME[s]} {int(v)}" for s, v in short.items()) + " at most)" if len(short) else "")
                  + f", of which {f_lo} to {f_hi} are flood seasons at the 1-in-3 level, so a single season moves POD by {1 / f_hi:.2f} to {1 / f_lo:.2f}.")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review of Google's new river model: Somalia Riverine Flood Trigger</title>
<meta name="description" content="A review of Google's November 2026 river model at the seven Somalia stations against the SWALIM gauges, beside Google's 2024 model, GloFAS v4 and GloFAS v5: the largest floods day by day and every score by season, river and station.">
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
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.65; max-width:1100px; }}
article ul {{ padding-left:22px; }}
article strong {{ color:var(--n9); }}
.muted {{ color:#6b7683; }}
p.note {{ font-size:13px; color:var(--n7); margin:6px 0 0; }}
figure.chart {{ margin:18px 0 26px; }}
figure.chart img {{ width:100%; height:auto; display:block; }}
figure.chart figcaption {{ font-size:13px; color:#55606d; line-height:1.55; margin-top:8px; }}
.callout {{ margin:20px 0; padding:13px 17px; border-radius:4px; background:var(--b05); border-left:5px solid var(--b5); font-size:13.5px; color:var(--n8); line-height:1.6; }}
.callout p {{ margin:0 0 7px; font-size:13.5px; max-width:none; }}
.callout p:last-child {{ margin:0; }}
.takeaways {{ margin:18px 0 6px; padding:0; list-style:none; }}
.takeaways li {{ border-left:3px solid var(--b1); padding:0 0 0 14px; margin:0 0 14px; }}
.pick {{ display:flex; flex-wrap:wrap; gap:6px; margin:12px 0 4px; }}
.pick button {{ font:500 13px Roboto,Arial,sans-serif; color:var(--n8); background:#fff; border:1px solid #cfd6d6; border-radius:16px; padding:5px 13px; cursor:pointer; }}
.pick button.on {{ background:var(--b5); border-color:var(--b5); color:#fff; }}
.pick button:focus-visible {{ outline:2px solid var(--b7); outline-offset:2px; }}
.picklabel {{ font-size:12px; color:var(--n7); text-transform:uppercase; letter-spacing:.05em; margin:14px 0 0; }}
.tablewrap {{ overflow-x:auto; }}
table.data {{ border-collapse:collapse; width:100%; margin:12px 0 20px; font-size:14px; }}
table.data th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:6px 6px 5px; vertical-align:bottom; }}
table.data td {{ border-bottom:1px solid #eef1f1; padding:7px 6px; color:var(--n8); vertical-align:top; }}
table.data tr.sep td {{ border-top:2px solid #d9dfdf; }}
table.data .y {{ display:block; font-size:11.5px; color:var(--n7); font-weight:400; text-transform:none; letter-spacing:0; }}
.files a {{ margin-right:14px; }}
@media (max-width:640px) {{ article {{ padding:4px 18px 16px; }} .hero.hero-sub {{ padding:32px 18px 28px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <div class="inner">
      <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / Review of Google's new river model</p>
      <h1>Review of Google's new river model</h1>
      <p>Google's November 2026 river model at the seven stations of the trigger, checked against the SWALIM gauges and set beside
         Google's 2024 model, GloFAS v4 and GloFAS v5. Generated {today}.</p>
    </div>
  </header>
  <article>
    <div class="callout">
      <p><strong>Sources.</strong> Google, new is the November 2026 model under review. Google, current is the model Google published in 2024. GloFAS v4
         and GloFAS v5 are the two versions of the European model. Each has a reanalysis, the model's own record of past flow, scored on 1999 to 2023.
         Three have a forecast archive: Google, new 2016 to 2025, Google, current 2016 to June 2023, GloFAS v4 2003 to 2023. Forecasts are compared on
         2016 to 2023.</p>
      <p><strong>Benchmark.</strong> The SWALIM river gauges at the same seven stations. A gauge measures water level and a model gives flow, so each is read
         against its own levels. The 1-in-3 level of a gauge or a model is the height its season peak reaches in about one season in three; the 1-in-5
         level, one in five. A flood season at a station is a season whose peak reaches the 1-in-3 level (or the 1-in-5 where stated). Gauge levels are
         set from 2000 to 2023, model levels from 1999 to 2023. The seasons are Deyr, October to December, and Gu, March to May.</p>
    </div>

    <h2><span class="num">1</span>What the records show</h2>
    <ul class="takeaways">
{FIND}
    </ul>

    <h2><span class="num">2</span>The largest floods, day by day</h2>
    <p>The gauges' five largest seasons on each river, one panel per station. Every line is drawn as a share of its own 1-in-3 level, so 100% is the
       flood level for the gauge and for the model alike. The gauge is a water level and the model a flow, so the heights of the curves differ; what can be
       compared is when each rises and whether each reaches its 100% line. Where a gauge line runs flat at its top, the record holds one value for
       those days. A line that stops before the end of the season is the end of that record. The vertical scale changes from one season to the next.</p>
    <p class="picklabel">Season</p>
    {pick("ts", "year", ts_opts)}
    <p class="picklabel">Lines</p>
    {pick("ts", "lines", [("new", "Gauge and the new model"), ("all", "Gauge and all four sources")])}
    {"".join(ts_figs)}

    <h2><span class="num">3</span>Day-to-day agreement with the gauges</h2>
    <p>Rank correlation puts the days in order by model flow and, separately, by gauge level, and measures how far the two orders agree: 1 means the
       model's highest days are the gauge's highest days, 0 means no relation. It uses the order of the days, not their values, so the scale of a model
       does not matter.</p>
    {fig_html(F["rho0"], "Rank correlation, same day")}
    <p>The same measure with the model's curve slid along the calendar, from 10 days behind the gauge to 30 days ahead of it, keeping the position with
       the highest value. The small figure is that position in days: plus means the model's curve runs ahead of the gauge's, minus means it runs behind.</p>
    {fig_html(F["rhob"], "Rank correlation at the shift with the highest value")}
    {fig_html(F["rhof"], "Rank correlation on flood seasons")}

    <h2><span class="num">4</span>How each model orders the seasons</h2>
    <p>Two measures of whether a model's large seasons are the gauge's large seasons. The rank correlation of season peaks compares the order of the
       seasons by their highest model flow with their order by highest gauge level. AUC is the chance that a flood season at the gauge carries a higher
       model peak than a season without a flood: 0.5 is a coin toss, 1 means every flood season has a higher model peak than every season without a flood.
       Neither depends on where the model's threshold is set.</p>
    {fig_html(F["rmax"], "Rank correlation of season peaks")}
    <p class="picklabel">Flood level at the gauge</p>
    {pick("auc", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {sw("auc", {"level": 3}, fig_html(F["auc3"], "AUC, 1-in-3"))}
    {sw("auc", {"level": 5}, fig_html(F["auc5"], "AUC, 1-in-5"))}

    <h2><span class="num">5</span>Flood seasons at each station</h2>
    <p>Each season at a station falls into one of four boxes: a flood at the gauge with the model's threshold reached; a flood with the threshold not
       reached; the threshold reached with no flood, which is a false alarm; or neither. Every score below is a different way of reading those four
       counts. {SAMPLE_TXT}</p>
    <p class="picklabel">Score</p>
    {pick("ev", "metric", [(k, lab) for k, lab, _, _, _ in METRICS])}
    <p class="picklabel">Flood level</p>
    {pick("ev", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {ev_text}
    {"".join(ev_blocks)}

    <h2><span class="num">6</span>The trigger rule by river</h2>
    <p>The trigger does not read one station. It needs three of the four Juba stations, or two of the three Shabelle stations, over their own level on the
       same day. The benchmark is a river flood: two gauges on the river over their own level at any time in the season. The grids show every season at
       the 1-in-3 level, with a reanalysis row and a forecast row for each source; the tables give the counts and scores at 1-in-3 and 1-in-5. For the
       forecast rows the rule is checked on each day's forecast for the following 7 days, and the first forecast of the season that meets it is taken.</p>
    {fig_html(F["bt_deyr"], "Deyr, threshold outcome by season")}
    {fig_html(F["bt_gu"], "Gu, threshold outcome by season")}
    <p class="picklabel">Level</p>
    {pick("rv", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {rv_blocks}
    <p class="note">Scores as in section 5. Days against the gauges: from the day the rule is met (reanalysis) or the day of the first forecast
       meeting it (forecasts) to the day the second gauge crossed its level. The rows marked "to 2025" score the new model's forecasts over its whole
       archive, 2016 to 2025; no other source has forecasts after 2023. On the Juba, 2024 and 2025 rest on the Dollow and Luuq gauges only.</p>

    <h2><span class="num">7</span>Timing at each station</h2>
    <p>Two measures in days, over the flood seasons at each gauge: when the model peaks against when the gauge peaks, and when the model first reaches its
       1-in-3 level against when the gauge first reaches its own. Plus means the model is ahead of the gauge, as in section 3. Where a gauge holds one
       value at its top for several days, its peak is dated to the first of those days.</p>
    {fig_html(F["pk"], "Days between model peak and gauge peak")}
    {fig_html(F["cx"], "Days between model and gauge reaching the 1-in-3")}

    <h2><span class="num">8</span>The forecasts, by days ahead</h2>
    <p>Each forecast archive is read one lead time at a time: the forecast for a day made 1 day ahead, 2 days ahead, and so on (the new model to 10 days,
       the others to 7). The first chart divides the forecast's season peak by the reanalysis peak on the same days: 1 means the forecast carries the same
       peak as the reanalysis. Seasons 2016 to 2023; GloFAS v4 is the middle of its ensemble members.</p>
    <p class="picklabel">Measure</p>
    {pick("ld", "metric", [("peak", "Forecast peak / reanalysis peak"), ("rho", "Rank correlation with the gauge"), ("pod", "POD"), ("far", "FAR")])}
    {sw("ld", {"metric": "peak"}, fig_html(F["lead_peak"], "Forecast peak over reanalysis peak by days ahead"))}
    {sw("ld", {"metric": "rho"}, fig_html(F["lead_rho"], "Rank correlation with the gauge by days ahead"))}
    {sw("ld", {"metric": "pod"}, fig_html(F["lead_pod"], "POD by days ahead"))}
    {sw("ld", {"metric": "far"}, fig_html(F["lead_far"], "FAR by days ahead"))}
    <p class="note">POD and FAR here are at single stations at the 1-in-3 level, with the counts of the stations of a river added up.
       At 7 days ahead the new model's daily forecast and its reanalysis have a rank correlation of {d7.rho_vs_reanalysis.min():.3f} to {d7.rho_vs_reanalysis.max():.3f} across the seven stations and two seasons.</p>

    <h2><span class="num">9</span>Size of the flows against SWALIM's discharge series</h2>
    <p>SWALIM publishes a discharge series, in cubic metres per second, at four of the seven stations. These are the only measurements of flow we have
       for the stations, so they are the only check on the size of a model's flow. {CEIL_TXT}</p>
    <p class="picklabel">Measure</p>
    {pick("q", "metric", [(k, lab) for k, lab, _, _, _ in Q_METRICS])}
    {q_text}
    {"".join(q_blocks)}
    <p class="note">The trigger reads each model against the model's own levels, so a model that runs high or low by a steady factor is not penalised in
       sections 3 to 8. This section is the only one about absolute size.</p>

    <h2><span class="num">10</span>What this review does not establish</h2>
    <ul>
      <li>On 5 November 2026 the live Flood Hub feed was not serving the new model. Everything here is from Google's published reanalysis and forecast archive.</li>
      <li>The new model's forecast archive is close to a copy of its reanalysis (section 8). Which rainfall the archive was run with has not been checked,
          so it is not known whether a live forecast would stay this close when the rain forecast is off.</li>
      <li>The forecast comparison rests on eight seasons, 2016 to 2023, with two to four river floods per river and season.</li>
      <li>The Bardheere gauge has no readings after November 2023 and the Bualle gauge none after March 2024, so the lower Juba cannot be checked from then on.</li>
      <li>Each model's flow is read at one place on its river network per station: the cell or reach on the main river nearest the station.</li>
    </ul>

    <h2><span class="num">11</span>Method and files</h2>
    <p class="note">Levels are read from each record's own season peaks (Weibull plotting position), gauges 2000 to 2023, models 1999 to 2023; a value
       equal to its level counts as reaching it. A gauge season counts with at least 30 readings. Station floods compare the season's highest values; the
       river rule counts the stations over their level on the same day, in the season months only. Days ahead: for Google, new and GloFAS, 1 is the day
       the forecast is issued; for Google, current, 1 is the day after. Sources: Google Runoff Reanalysis and Reforecast, November 2026 release; SWALIM gauge
       levels and discharge series; GloFAS v4 and v5 from the Copernicus Emergency Management Service.</p>
    <p class="note files">Every score, by season, river and station:
      <a href="daily_agreement.csv">daily agreement</a><a href="station_floods_1in3.csv">station floods, 1-in-3</a><a href="station_floods_1in5.csv">station floods, 1-in-5</a>
      <a href="river_rule.csv">river rule</a><a href="forecasts_by_lead.csv">forecasts by lead</a><a href="discharge.csv">discharge</a><a href="data.json">all, as JSON</a></p>
  </article>
</div>
<script>
(function () {{
  var state = {{}};
  function apply(group) {{
    document.querySelectorAll('.sw[data-group="' + group + '"]').forEach(function (el) {{
      var show = true;
      for (var dim in state[group]) {{
        var v = el.getAttribute('data-' + dim);
        if (v !== null && v !== state[group][dim]) show = false;
      }}
      el.style.display = show ? '' : 'none';
    }});
  }}
  document.querySelectorAll('.pick').forEach(function (p) {{
    var group = p.getAttribute('data-group'), dim = p.getAttribute('data-dim');
    state[group] = state[group] || {{}};
    state[group][dim] = p.querySelector('button.on').getAttribute('data-key');
    p.addEventListener('click', function (e) {{
      var b = e.target.closest('button');
      if (!b) return;
      p.querySelectorAll('button').forEach(function (x) {{ x.classList.remove('on'); }});
      b.classList.add('on');
      state[group][dim] = b.getAttribute('data-key');
      apply(group);
    }});
  }});
  Object.keys(state).forEach(apply);
}})();
</script>
</body>
</html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("written", OUT / "index.html", len(list(FIGS.glob("*.svg"))), "figures")


if __name__ == "__main__":
    build()
