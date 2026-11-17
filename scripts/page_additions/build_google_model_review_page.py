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
def _lum(rgb):
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


_BODY_RGB = matplotlib.colors.to_rgb(BODY)


def _ink(face):
    """White or body-colour text, whichever stands out more on the cell."""
    lf = _lum(face)
    c_white = 1.05 / (lf + 0.05)
    c_body = (lf + 0.05) / (_lum(_BODY_RGB) + 0.05)
    return "white" if c_white > c_body else BODY


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


def grid_fig(df, col, name, title, kind, fmt="{:.2f}", sub=None, note=None, rows=None, panels=None, row_labels=None):
    """One figure: a panel per season (or per period), a row per station, a column per source."""
    panels = panels or [("deyr", SEASON_TITLE["deyr"]), ("gu", SEASON_TITLE["gu"])]
    rows = rows or STATIONS
    n_j = sum(RIVER_OF[st] == "juba" for st in rows)
    pcol = "season" if "season" in df.columns else "period"
    fig_w = 5.6 * len(panels) + 1.2
    k = fig_w / 12.4                      # the page scales every figure to one width: keep the text one size
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_w, (0.5 * len(rows) + 2.0) * k))
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
                    ax.text(j, i, "no record" if not len(r) else "n/a", ha="center", va="center", fontsize=8 * k, color=FAINT)
                    continue
                ink = _ink(face)
                s_ = sub(r.iloc[0]) if sub else None
                ax.text(j, i - (0.13 if s_ else 0), fmt(v) if callable(fmt) else fmt.format(v), ha="center", va="center", fontsize=10 * k, color=ink)
                if s_:
                    ax.text(j, i + 0.26, s_, ha="center", va="center", fontsize=7.5 * k, color=ink)
        ax.set_xlim(-0.5, len(MODELS) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(range(len(MODELS)), [SHORT[m].replace(", ", "\n") for m in MODELS], fontsize=9.5 * k, color=BODY)
        ax.xaxis.tick_top()
        labs = [(row_labels or {}).get(st, NAME[st]) for st in rows]
        ax.set_yticks(range(len(rows)), labs if ax is axes[0] else [""] * len(rows), fontsize=10 * k, color=BODY)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if 0 < n_j < len(rows):
            ax.axhline(n_j - 0.5, color=INK, linewidth=1.0)
            if ax is axes[0]:
                ax.text(-1.72, (n_j - 1) / 2, "JUBA", ha="center", va="center", rotation=90, fontsize=8.5 * k, color=FAINT, clip_on=False)
                ax.text(-1.72, n_j + (len(rows) - n_j - 1) / 2, "SHABELLE", ha="center", va="center", rotation=90, fontsize=8.5 * k, color=FAINT, clip_on=False)
        ax.set_title(ptitle, fontsize=11 * k, loc="left", pad=34 * k)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5 * k, y=1.02)
    if note:
        fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=9 * k, color=BODY)
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
def fig_outcomes(Rrec, season):
    """Every season at the 1-in-3 level: under each source a reanalysis row and, where an archive
    exists, a forecast row. The counts at the right cover the scored seasons."""
    cell = {"hit": "#1B5E20", "false": "#B08D57", "miss": "#3D3D3D", "quiet": "#D5DCE3", "none": "#FFFFFF"}
    years = G.YEARS
    lines = []
    for river in G.RIVERS:
        for m in MODELS:
            lines.append((f"{river.title()}: {SHORT[m]}, reanalysis", river, m, "ra"))
            if m in G.FC_SPAN:
                lines.append((f"{river.title()}: {SHORT[m]}, forecasts", river, m, "fc"))
    fig, ax = plt.subplots(figsize=(14.8, 0.5 * len(lines) + 2.9))
    n_juba = sum(1 for ln in lines if ln[1] == "juba")
    for i, (name, river, m, basis) in enumerate(lines[::-1]):
        rec = Rrec[(river, season)]
        span = G.SCORE_RA if basis == "ra" else G.SCORE_FC
        f, yr = G.activations_of(rec, m, basis), G.years_of(rec, m, basis)
        for j, y in enumerate(years):
            if y not in yr or y not in rec["assess"]:
                kind = "none"
            elif y in f:
                kind = "hit" if y in rec["severe"] else "false"
            else:
                kind = "miss" if y in rec["severe"] else "quiet"
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.36), 0.92, 0.72, facecolor=cell[kind], edgecolor="white" if kind != "none" else "#C9D1D8", linewidth=0.8))
        sc_ = G.score(rec, m, basis, span)
        n_f, n_a = len(sc_["severe"]), len(sc_["false"])
        t1 = f"reached in {len(sc_['caught'])} of {n_f} river flood{'s' if n_f != 1 else ''}" if n_f else "no river flood in these seasons"
        ax.text(len(years) + 0.2, i + 0.15, t1, va="center", fontsize=9, color=INK)
        ax.text(len(years) + 0.2, i - 0.2, f"{n_a} false alarm{'s' if n_a != 1 else ''}", va="center", fontsize=9, color="#8A6A2B" if n_a else FAINT)
    ax.axhline(len(lines) - n_juba - 0.5, xmin=0.0, xmax=(len(years) + 0.6) / (len(years) + 9.6 + 0.6), color=INK, linewidth=1.0)
    for edge in (G.SCORE_FC[0], G.SCORE_FC[1] + 1):
        ax.axvline(years.index(edge) - 0.5, ymin=0.06, color=INK, linewidth=0.9, ls=":", zorder=4)
    ax.set_xlim(-0.6, len(years) + 9.6)
    ax.set_ylim(-1.0, len(lines) - 0.4)
    ax.set_yticks(range(len(lines)), [ln[0] for ln in lines[::-1]], fontsize=9.5)
    ax.set_xticks(range(0, len(years), 2), [str(y) for y in years[::2]], fontsize=9.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [plt.Rectangle((0, 0), 1, 1, facecolor=cell["hit"], label="threshold reached, river flood at the gauges"),
         plt.Rectangle((0, 0), 1, 1, facecolor=cell["false"], label="threshold reached, no river flood (false alarm)"),
         plt.Rectangle((0, 0), 1, 1, facecolor=cell["miss"], label="river flood, threshold not reached"),
         plt.Rectangle((0, 0), 1, 1, facecolor=cell["quiet"], label="neither"),
         plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#C9D1D8", label="no model record or no gauge record")]
    ax.legend(handles=H, ncol=3, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.42, -0.05), frameon=False)
    note = f"Counts at the right: reanalysis rows over every season with a gauge record to {G.SCORE_RA[1]}; forecast rows between the dotted lines, {G.SCORE_FC[0]} to {G.SCORE_FC[1]}"
    note += " (Google, current to 2022)." if season == "deyr" else "."
    fig.text(0.17, 0.02, note + chr(10) + "A river flood is two gauges on the river over their own 1-in-3 level.", fontsize=9, color=BODY, ha="left", va="bottom", linespacing=1.5)
    ax.set_title(f"{SEASON_TITLE[season]}: the trigger rule at the 1-in-3 level, season by season", fontsize=12, loc="left")
    fig.subplots_adjust(left=0.17, right=0.995, top=0.95, bottom=0.17)
    return save(fig, f"outcomes_{season}", tight=False)


def lead_table(BL):
    g = BL.groupby(["model", "season", "river", "lead"]).agg(
        peak=("peak_ratio_median", "median"), rise=("rise_share_median", "median"), rho=("rho_vs_gauge", "mean"), rho_ra=("rho_vs_reanalysis", "mean"),
        a=("hits", "sum"), b=("false_alarms", "sum"), c=("misses", "sum")).reset_index()
    g["POD"] = g.a / (g.a + g.c)
    g["FAR"] = g.b / (g.a + g.b)
    return g


def fig_lead(g, col, name, ylabel, ylim, ref=None):
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.3), sharey=True)
    k = 0
    for season in ("deyr", "gu"):
        for river in G.RIVERS:
            ax = axes[k]
            k += 1
            for m in (NEW, "google_grrr", "glofas_v4"):
                d = g[(g.model == m) & (g.season == season) & (g.river == river)].sort_values("lead")
                style = {NEW: dict(lw=3.0, marker="o", ms=5, zorder=2),
                         "google_grrr": dict(lw=1.6, ls=(0, (4, 2)), marker="s", ms=4.5, mfc="white", zorder=4),
                         "glofas_v4": dict(lw=1.6, marker="^", ms=4.5, zorder=3)}[m]
                ax.plot(d.lead, d[col], color=COL[m], label=SHORT[m], **style)
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
    fig.suptitle(ylabel, x=0.01, ha="left", fontsize=11.5)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=3, fontsize=9.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.07, left=0.04, right=0.99, top=0.80, bottom=0.27)
    return save(fig, name, tight=False)


# ------------------------------------------------------------------------------ html
def num(v, f="{:.2f}"):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def yrs(xs):
    return ", ".join(str(int(y)) for y in xs) if len(xs) else "none"


def rng2(lo, hi):
    return "" if (lo is None or np.isnan(lo) or np.isnan(hi)) else f"{lo:.2f} to {hi:.2f}"


def notice(v):
    """Days between the threshold and the second gauge, in words."""
    if v is None or np.isnan(v):
        return "n/a"
    r = int(np.floor(abs(v) + 0.5))
    return "same day" if r == 0 else f"{r} {'before' if v > 0 else 'after'}"


REC_LABEL = {"reanalysis": "reanalysis, 1999 to 2023", "reanalysis, forecast seasons": "reanalysis, forecast seasons only",
             "forecasts": "forecasts, 2016 to 2023"}


def river_table(Rv, rp, season):
    head = ("<table class='data'><thead><tr><th>River</th><th>Source</th><th>Record</th><th>Seasons</th><th>River floods</th>"
            "<th>Threshold reached</th><th>Reached in a flood</th><th>Reached, no flood<span class='y'>false alarms</span></th><th>Flood not reached</th>"
            "<th>POD</th><th>FAR</th><th>POFD</th><th>CSI</th><th>F1</th><th>Frequency bias</th><th>HSS</th>"
            "<th>Reached against the second gauge<span class='y'>days, middle value over floods reached</span></th></tr></thead><tbody>")
    out = [head]
    d = Rv[(Rv.rp == rp) & (Rv.season == season) & ~Rv.record.str.contains("held out")]
    for river in G.RIVERS:
        first = True
        for m in MODELS:
            for rec in list(d[d.model == m].record.unique()):
                r = d[(d.river == river) & (d.model == m) & (d.record == rec)]
                if not len(r):
                    continue
                r = r.iloc[0]
                cls = " class='sep'" if first else ""
                pod_r, far_r = rng2(r.get("POD_lo"), r.get("POD_hi")), rng2(r.get("FAR_lo"), r.get("FAR_hi"))
                out.append(f"<tr{cls}><td>{river.title() if first else ''}</td><td>{SHORT[m]}</td><td class='muted'>{REC_LABEL.get(rec, rec.replace('forecasts to', 'forecasts, 2016 to'))}</td>"
                           f"<td>{int(r.n)}</td><td>{int(r.n_floods)}</td><td>{int(r.n_reached)}</td>"
                           f"<td>{int(r.hits)}<span class='y'>{yrs(r.hit_years)}</span></td>"
                           f"<td>{int(r.false_alarms)}<span class='y'>{yrs(r.false_years)}</span></td>"
                           f"<td>{int(r.misses)}<span class='y'>{yrs(r.missed_years)}</span></td>"
                           f"<td>{num(r.POD)}<span class='y'>{pod_r}</span></td><td>{num(r.FAR)}<span class='y'>{far_r}</span></td>"
                           f"<td>{num(r.POFD)}</td><td>{num(r.CSI)}</td><td>{num(r.F1)}</td>"
                           f"<td>{num(r.freq_bias)}</td><td>{num(r.HSS)}</td><td>{notice(r.lead_days_median)}</td></tr>")
                first = False
    out.append("</tbody></table>")
    return "\n".join(out)


def held_out_table(Rv, rp):
    """The reanalysis result with every level fitted on all seasons, beside the result with the
    level for each season fitted without that season."""
    head = ("<table class='data'><thead><tr><th rowspan='2'>Season</th><th rowspan='2'>River</th><th rowspan='2'>Source</th>"
            "<th colspan='3'>Levels fitted on all seasons</th><th colspan='3'>Levels fitted without the season scored</th></tr>"
            "<tr><th>Floods reached</th><th>False alarms</th><th>F1</th><th>Floods reached</th><th>False alarms</th><th>F1</th></tr></thead><tbody>")
    out = [head]
    d = Rv[Rv.rp == rp]
    for season in ("deyr", "gu"):
        for river in G.RIVERS:
            first = True
            for m in MODELS:
                a = d[(d.season == season) & (d.river == river) & (d.model == m) & (d.record == "reanalysis")].iloc[0]
                b = d[(d.season == season) & (d.river == river) & (d.model == m) & (d.record == "reanalysis, levels held out")].iloc[0]
                cls = " class='sep'" if first else ""
                out.append(f"<tr{cls}><td>{season.title() if first else ''}</td><td>{river.title() if first else ''}</td><td>{SHORT[m]}</td>"
                           f"<td>{int(a.hits)} of {int(a.n_floods)}</td><td>{int(a.false_alarms)}</td><td>{num(a.F1)}</td>"
                           f"<td>{int(b.hits)} of {int(b.n_floods)}</td><td>{int(b.false_alarms)}</td><td>{num(b.F1)}</td></tr>")
                first = False
    out.append("</tbody></table>")
    return "\n".join(out)


def gauge_table(GN):
    head = ("<table class='data'><thead><tr><th rowspan='2'>River</th><th rowspan='2'>Gauge</th><th colspan='4'>Deyr</th><th colspan='4'>Gu</th></tr><tr>"
            + "<th>Seasons on record</th><th>At or over the 1-in-3 level</th><th>Seasons at their top reading for 5 days or more</th><th>Most common season peak</th>" * 2
            + "</tr></thead><tbody>")
    out = [head]
    for river in G.RIVERS:
        first = True
        for st in G.TRIGGER_STATIONS[river]:
            cells = []
            for season in ("deyr", "gu"):
                r = GN[(GN.season == season) & (GN.station == st)].iloc[0]
                cells.append(f"<td>{int(r.n_seasons)}</td><td>{int(r.n_at_or_over)}</td><td>{int(r.n_flat_top)}</td>"
                             f"<td>{r.common_peak:.2f} m<span class='y'>in {int(r.n_common_peak)} season{'s' if r.n_common_peak != 1 else ''}</span></td>")
            cls = " class='sep'" if first else ""
            out.append(f"<tr{cls}><td>{river.title() if first else ''}</td><td>{NAME[st]}</td>{''.join(cells)}</tr>")
            first = False
    out.append("</tbody></table>")
    return "\n".join(out)


LONG = {"POD": "POD, probability of detection", "FAR": "FAR, false alarm ratio", "POFD": "POFD, probability of false detection",
        "CSI": "CSI, critical success index", "F1": "F1", "freq_bias": "Frequency bias", "accuracy": "Accuracy",
        "HSS": "HSS, Heidke skill score", "PSS": "PSS, Peirce skill score", "KGE": "KGE, Kling-Gupta efficiency", "NSE": "NSE, Nash-Sutcliffe efficiency"}
METRICS = [  # key, label, colour kind, format, plain meaning
    ("POD", "POD", "good", "{:.2f}", "POD, probability of detection: of the flood seasons at the gauge, the share in which the model reached its own threshold. 1 means the threshold was reached in every flood season; 0 means in none."),
    ("FAR", "FAR", "bad", "{:.2f}", "FAR, false alarm ratio: of the seasons in which the model reached its threshold, the share with no flood at the gauge. 0 means no false alarm."),
    ("POFD", "POFD", "bad", "{:.2f}", "POFD, probability of false detection: of the seasons with no flood at the gauge, the share in which the model still reached its threshold. 0 means the threshold was never reached without a flood."),
    ("CSI", "CSI", "good", "{:.2f}", "CSI, critical success index: the floods reached, divided by the floods reached plus the floods not reached plus the false alarms. It falls with either kind of error; 1 means neither occurred."),
    ("F1", "F1", "good", "{:.2f}", "F1: twice the floods reached, divided by twice the floods reached plus the floods not reached plus the false alarms. It is high only when POD is high and FAR is low; 1 means every flood was reached with no false alarm."),
    ("freq_bias", "Frequency bias", "ratio", "{:.2f}", "Frequency bias: seasons in which the model reached its threshold divided by flood seasons at the gauge. Above 1 the model reaches its threshold more often than the gauge floods; below 1, less often."),
    ("accuracy", "Accuracy", "good", "{:.2f}", "Accuracy: the share of all seasons in which model and gauge agree, flood or no flood. Most seasons have no flood, so accuracy stays high even for a model that reaches its threshold in no flood season."),
    ("HSS", "HSS", "eff", "{:.2f}", "HSS, Heidke skill score: accuracy with the agreement that chance alone would give taken out. 1 means model and gauge agree in every season; 0 means no more agreement than chance; below 0, less than chance."),
    ("PSS", "PSS", "eff", "{:.2f}", "PSS, Peirce skill score: POD minus POFD, from -1 to 1. 1 means the model separates flood seasons from the rest fully; 0 means it does not separate them."),
]
Q_METRICS = [
    ("ratio_median", "Ratio of medians", "ratio", "{:.2f}", "Ratio of medians: the model's flow on a typical day divided by SWALIM's on a typical day. 1 means equal; 2 means the model runs at twice SWALIM's flow. Typical days are low-flow days, so this says little about floods."),
    ("ratio_peak_median", "Ratio of season peaks", "ratio", "{:.2f}", "Ratio of season peaks: the model's highest flow of each season divided by SWALIM's, middle value over the seasons. Where SWALIM's series stops at a top value, a ratio above 1 can come from that ceiling."),
    ("pearson_r", "Correlation", "good", "{:.2f}", "Correlation of daily values, from -1 to 1. 1 means the two rise and fall together in proportion; 0 means no straight-line relation."),
    ("variability_ratio", "Variability ratio", "ratio", "{:.2f}", "Variability ratio: the size of the model's swings divided by the size of SWALIM's. 1 means the same spread."),
    ("KGE", "KGE", "eff", "{:.2f}", "KGE, Kling-Gupta efficiency: one score built from the correlation, the variability ratio and the ratio of averages. 1 means the model matches SWALIM's series in all three; the lower the score, the further the model is from it."),
    ("NSE", "NSE", "eff", "{:.2f}", "NSE, Nash-Sutcliffe efficiency: 1 means the model matches SWALIM's series; 0 means it is no closer to it than a flat line at SWALIM's average; below 0, further away than that line."),
]
D_METRICS = [  # daily agreement: key, label, plain meaning
    ("same", "Same day", "Model and gauge compared on the same day. The small figures are the range the value takes when the seasons are redrawn at random, which shows how much it depends on which seasons happen to be on record."),
    ("cycle", "Seasonal cycle removed", "Every river rises and falls at about the same time each season, which alone can give any model a high value. Here the usual level for each calendar day is taken out of both series first, so what is left is whether the model is high in the weeks when the gauge is higher than usual."),
    ("shift", "Slid to the closest match", "The model's curve is slid along the calendar, from 10 days behind the gauge to 30 days ahead of it, and the highest value is kept. The small figure is that position in days: plus means the model's curve runs ahead of the gauge's, minus means it runs behind."),
    ("flood", "Flood seasons only", "Slid as before, using only seasons in which the gauge or the model went over its own 1-in-3 level. The small figure is the number of seasons used."),
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
    return f"<figure class='chart'><a href='{src}'><img src='{src}' alt='{alt}' loading='lazy'></a>{'<figcaption>' + cap + '</figcaption>' if cap else ''}</figure>"


def flood_view_note(T):
    """Where the flood-season correlation here differs from the google-model-2026 page, and any value
    whose shift sits at the end of the range tried."""
    out = ""
    edge = T[T.shift_flood_seasons.isin([min(R.LAGS), max(R.LAGS)]) & T.rho_flood_seasons.notna()]
    if len(edge):
        names = [f"{SHORT[r.model]} at {NAME[r.station]} in {r.season.title()} ({int(r.shift_flood_seasons):+d} days)" for r in edge.itertuples()]
        out += " The highest value sits at the end of the range tried for " + "; ".join(names) + "."
    sib = OUT.parent / "google-model-2026" / "data.json"
    if sib.exists():
        c = pd.DataFrame(json.loads(sib.read_text(encoding="utf-8"))["correlations"])
        j = c.merge(T, on=["season", "station", "model"])
        if len(j) == len(T):
            d = (j.rho_flood_seasons - j.rho)
            big = j[d.abs() > 0.05]
            out += (f" The <a href='../google-model-2026/'>google-model-2026</a> page shows this score with the gauge's season days only, so at long shifts"
                    f" the last days of the season drop out; here every season day of the model is used at every shift."
                    f" The two agree within 0.05 at {int((d.abs() <= 0.05).sum())} of the {len(j)} values.")
            if len(big):
                which = "; ".join(f"{SHORT[r.model]} at {NAME[r.station]} in {r.season.title()}: {r.rho:.2f} there, {r.rho_flood_seasons:.2f} here, at a shift of {int(r.shift_flood_seasons)} days"
                                  for r in big.sort_values("rho_flood_seasons").itertuples())
                out += f" The others: {which}."
    return out


def rule_summary(M):
    """Section 1 table: the trigger rule on the new model's reanalysis at both levels."""
    head = ("<table class='data'><thead><tr><th rowspan='2'>Season</th><th rowspan='2'>River</th><th colspan='3'>1-in-3 level</th><th colspan='3'>1-in-5 level</th></tr>"
            "<tr>" + "<th>River floods with the threshold reached</th><th>Floods not reached</th><th>False alarms</th>" * 2 + "</tr></thead><tbody>")
    out = [head]
    for season in ("deyr", "gu"):
        for k, river in enumerate(G.RIVERS):
            cells = []
            for Rv in (M["river_3"], M["river_5"]):
                r = Rv[(Rv.model == NEW) & (Rv.river == river) & (Rv.season == season) & (Rv.record == "reanalysis")].iloc[0]
                cells.append(f"<td>{int(r.hits)} of {int(r.n_floods)}</td><td>{yrs(r.missed_years)}</td><td>{int(r.false_alarms)}<span class='y'>{yrs(r.false_years) if r.false_years else ''}</span></td>")
            cls = " class='sep'" if k == 0 else ""
            out.append(f"<tr{cls}><td>{season.title() if k == 0 else ''}</td><td>{river.title()}</td>{''.join(cells)}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def after_line(Rrec, season, y):
    """One line for a season after 2023, new model, 1-in-3 level."""
    def where(basis):
        rs = [r.title() for r in G.RIVERS if y in Rrec[(r, season)]["models"][NEW][basis]]
        return "both rivers" if len(rs) == 2 else (f"the {rs[0]}" if rs else "neither river")
    ra, fc = where("ra"), where("fc")
    reached = f"threshold reached on {ra}, on the reanalysis and on the forecasts" if ra == fc else f"threshold reached on {ra} on the reanalysis and on {fc} on the forecasts"
    if ra == fc == "neither river":
        reached = "threshold not reached"
    g = []
    for r in G.RIVERS:
        rec = Rrec[(r, season)]
        g.append(f"{r.title()}: " + ("river flood" if y in rec["severe"] else "no river flood" if y in rec["assess"] else "no gauge record"))
    return f"{season.title()} {y}: {reached}. " + "; ".join(g) + "."


def findings(M, LT, Rrec):
    T, E3, Q, BL = M["tracking"], M["station_3"], M["discharge"], M["by_lead"]
    R3 = M["river_3"]
    out = []

    # forecasts against the reanalysis on the same seasons
    same = tot = 0
    for season in ("deyr", "gu"):
        for river in G.RIVERS:
            ra = M["river_dates_3"][(season, river, NEW, "reanalysis", 3)]
            fc = M["river_dates_3"][(season, river, NEW, "forecasts", 3)]
            for y in range(R.FC0, R.FC1 + 1):
                tot += 1
                same += (y in ra) == (y in fc)
    d7 = LT[(LT.model == NEW) & (LT.lead == 7)]
    v7 = LT[(LT.model == "glofas_v4") & (LT.lead == 7)]
    out.append(f"<strong>Forecasts.</strong> Over 2016 to 2023 the new model's forecasts match its reanalysis in {same} of the {tot} river seasons. "
               f"Seven days ahead, on the days of the largest rises in its own reanalysis, the forecast already showed {100 * d7.rise.min():.0f} to {100 * d7.rise.max():.0f}% of the rise, "
               f"by river and season (GloFAS v4: {100 * v7.rise.min():.0f} to {100 * v7.rise.max():.0f}%; below 0 the forecast showed a fall).")

    def nt(river, season, rec):
        r = R3[(R3.model == NEW) & (R3.river == river) & (R3.season == season) & (R3.record == rec)].iloc[0]
        return notice(r.lead_days_median)
    out.append("<strong>Days between the threshold being reached and the second gauge crossing its 1-in-3</strong> (middle value over the floods reached, 2016 to 2023). "
               + " ".join(f"{r.title()} {s.title()}: forecasts {nt(r, s, 'forecasts')}, reanalysis {nt(r, s, 'reanalysis, forecast seasons')}." for s in ("deyr", "gu") for r in G.RIVERS))

    out.append("<strong>After 2023, new model only, 1-in-3 level.</strong> " + " ".join(after_line(Rrec, s, y) for y in (2024, 2025) for s in ("gu", "deyr")))

    parts = []
    for season in ("deyr", "gu"):
        d = E3[(E3.model == NEW) & (E3.season == season)]
        parts.append(f"{season.title()}: threshold reached in {int(d.hits.sum())} of {int(d.n_floods.sum())} flood seasons, with {int(d.false_alarms.sum())} false alarms. "
                     f"With each level fitted without the season scored: {int(d.hits_held_out.sum())} of {int(d.n_floods_held_out.sum())}, with {int(d.false_alarms_held_out.sum())}.")
    out.append("<strong>Single stations, 1-in-3 level, new model, the seven stations added up.</strong> " + " ".join(parts))

    jo = E3[E3.station == "jowhar"]
    gj = M["gauges"][M["gauges"].station == "jowhar"].set_index("season")
    out.append(f"<strong>Jowhar.</strong> Every source reaches its threshold in fewer than half of the flood seasons there, in Deyr and in Gu (POD {jo.POD.min():.2f} to {jo.POD.max():.2f}). "
               f"The Jowhar gauge sat at one top reading for five days or more in {int(gj.loc['deyr', 'n_flat_top'])} of its {int(gj.loc['deyr', 'n_seasons'])} Deyr seasons.")

    piv = T.pivot_table(index=["season", "station"], columns="model", values="rho_same_day")
    pv2 = T.pivot_table(index=["season", "station"], columns="model", values="rho_cycle_removed")
    clear = {m: int((T[T.model == m].new_minus_this_lo > 0).sum()) for m in MODELS[1:]}
    out.append(f"<strong>Day-to-day agreement with the gauges</strong> (rank correlation: 0 is no relation, 1 is the same order of days). The new model has the highest value of the four sources at "
               f"{int((piv.idxmax(axis=1) == NEW).sum())} of the {len(piv)} combinations of station and season ({piv[NEW].min():.2f} to {piv[NEW].max():.2f}), and at "
               f"{int((pv2.idxmax(axis=1) == NEW).sum())} with the seasonal cycle removed ({pv2[NEW].min():.2f} to {pv2[NEW].max():.2f}). "
               f"When the seasons are redrawn, its gap to another source stays above zero (5th to 95th percentile) at {clear['google_grrr']} of {len(piv)} against Google, current, "
               f"{clear['glofas_v4']} against GloFAS v4 and {clear['glofas_v5']} against GloFAS v5.")

    a3 = E3.pivot_table(index=["season", "station"], columns="model", values="AUC").round(2)
    rp_ = E3.pivot_table(index=["season", "station"], columns="model", values="rho_season_peaks").round(2)
    top = lambda p_: int(p_.eq(p_.max(axis=1), axis=0)[NEW].sum())
    last = [f"{NAME[st]} in {se.title()} ({a3.loc[(se, st), NEW]:.2f}; the others {a3.loc[(se, st)].drop(NEW).min():.2f} to {a3.loc[(se, st)].drop(NEW).max():.2f})"
            for (se, st) in a3.index if a3.loc[(se, st), NEW] < a3.loc[(se, st)].drop(NEW).min()]
    out.append(f"<strong>Which seasons are the large ones, 1-in-3 level.</strong> AUC is the chance that a flood season has a higher model peak than a season without a flood. "
               f"The new model has the highest, or equal highest, AUC at {top(a3)} of {len(a3)} combinations of station and season, and the highest rank correlation of season peaks at {top(rp_)}."
               + (" It has the lowest AUC of the four at " + " and at ".join(last) + "." if last else ""))

    qy = Q[Q.period == "year"]
    q = qy.pivot_table(index="station", columns="model", values="ratio_median")
    qp = qy.pivot_table(index="station", columns="model", values="ratio_peak_median")
    three = [s_ for s_ in ("luuq", "belet_weyne", "bulo_burti") if s_ in q.index]
    names = ", ".join(NAME[s_] for s_ in three[:-1]) + " and " + NAME[three[-1]]
    rg = lambda m: f"{q.loc[three, m].min():.1f} to {q.loc[three, m].max():.1f}"
    out.append(f"<strong>Size of the flows against SWALIM's discharge series, whole year.</strong> At {names}, on a typical day the new model runs at {rg(NEW)} times SWALIM's flow "
               f"(GloFAS v5 {rg('glofas_v5')}, GloFAS v4 {rg('glofas_v4')}, Google, current {rg('google_grrr')}). At the yearly peaks it runs at "
               f"{qp.loc[three, NEW].min():.1f} to {qp.loc[three, NEW].max():.1f} times.")
    return "\n".join(f"<li>{x}</li>" for x in out)


def build():
    FIGS.mkdir(parents=True, exist_ok=True)
    for old in FIGS.glob("*.svg"):
        old.unlink()
    M = R.compute()
    T, Q, BL, GN, QN = M["tracking"], M["discharge"], M["by_lead"], M["gauges"], M["discharge_notes"]
    LT = lead_table(BL)
    Rv = pd.concat([M["river_3"], M["river_5"]], ignore_index=True)
    E3, E5 = M["station_3"], M["station_5"]
    Rrec = G.record()
    F = {}

    # ---- the largest floods, day by day
    ts_opts, ts_figs = [], []
    for season in ("deyr", "gu"):
        for y in M["worst"][season]:
            key = f"{season}_{y}"
            ts_opts.append((key, f"{season.title()} {y}"))
            for variant in ("new", "all"):
                src = fig_ts(season, y, variant)
                ts_figs.append(sw("ts", {"year": key, "lines": variant}, fig_html(src, f"{season.title()} {y}, seven stations, {variant}")))

    # ---- the trigger rule by river
    for s in ("deyr", "gu"):
        F[f"bt_{s}"] = fig_outcomes(Rrec, s)
    rv_blocks = "".join(sw("rv", {"level": rp}, f"<h3>{s.title()}, 1-in-{rp} level</h3><div class='tablewrap'>{river_table(Rv, rp, s)}</div>")
                        for rp in (3, 5) for s in ("deyr", "gu"))
    ho_blocks = "".join(sw("rv", {"level": rp}, f"<div class='tablewrap'>{held_out_table(Rv, rp)}</div>") for rp in (3, 5))

    # ---- forecasts by days ahead
    F["lead_rise"] = fig_lead(LT.assign(rise_pct=100 * LT.rise), "rise_pct", "lead_rise", "Share of the coming rise already in the forecast, % (middle value over the stations of the river)", (-20, 140), ref=100.0)
    F["lead_peak"] = fig_lead(LT, "peak", "lead_peak", "Forecast peak divided by reanalysis peak (middle value over the stations of the river)", (0.8, 1.25), ref=1.0)
    F["lead_rho"] = fig_lead(LT, "rho", "lead_rho", "Rank correlation of the forecast with the gauge (average over the stations of the river)", (0, 1.0))
    F["lead_pod"] = fig_lead(LT, "POD", "lead_pod", "POD: share of flood seasons with the threshold reached, single stations at the 1-in-3 level, added up by river", (0, 1.0))
    F["lead_far"] = fig_lead(LT, "FAR", "lead_far", "FAR: share of threshold seasons with no flood, single stations at the 1-in-3 level, added up by river", (0, 1.0))
    d7 = BL[(BL.model == NEW) & (BL.lead == 7)]

    # ---- timing at stations
    ahead_note = "Plus: the model ahead of the gauge. Minus: behind. Teal = ahead, brown = behind; the shading is full at 15 days."
    F["pk"] = grid_fig(T, "peak_days_ahead_median", "peak_days", "Days from the model's peak to the gauge's peak, flood seasons at the gauge (middle value)", "days", fmt=signed,
                       sub=lambda r: f"{int(r.n_peaks)} seasons", note=ahead_note + " Small figure: seasons used.")
    F["cx"] = grid_fig(T, "crossing_days_ahead_median", "crossing_days", "Days from the model reaching its 1-in-3 to the gauge reaching its own (middle value)", "days", fmt=signed,
                       sub=lambda r: f"{int(r.n_crossings)} seasons", note=ahead_note + " Small figure: seasons in which both reach their level.")

    # ---- flood seasons at each station
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
            note = {"F1": "1 = every flood reached with no false alarm. ", "HSS": "1 = model and gauge agree in every season; 0 = no more than chance. ",
                    "PSS": "1 = flood seasons fully separated from the rest; 0 = not separated. ",
                    "freq_bias": "1 = the model reaches its threshold as often as the gauge floods. "}.get(key, "") + note
            note += {"POD": " Small figures: flood seasons with the threshold reached, of flood seasons at the gauge.",
                     "FAR": " Small figures: seasons with no flood, of seasons with the threshold reached.",
                     "POFD": " Small figures: seasons with the threshold reached, of seasons with no flood.",
                     "CSI": " Small figures: floods reached / floods not reached / false alarms.",
                     "F1": " Small figures: floods reached / floods not reached / false alarms.",
                     "accuracy": " Small figures: seasons in which model and gauge agree, of all seasons."}.get(key, "")
            src = grid_fig(E, key, f"st_{key}_{rp}", f"{LONG[key]}. Flood season = the station's gauge over its 1-in-{rp}", kind, fmt=fmt, sub=sub, note=note)
            ev_blocks.append(sw("ev", {"metric": key, "level": rp}, fig_html(src, f"{lab} by station, 1-in-{rp}")))
    ev_text = "".join(sw("ev", {"metric": key}, text, tag="p") for key, _, _, _, text in METRICS)
    n_lo, n_hi, f_lo, f_hi = int(E3.n.min()), int(E3.n.max()), int(E3.n_floods.min()), int(E3.n_floods.max())
    short = E3[E3.n < 15].groupby("station").n.max()
    SAMPLE_TXT = (f"The counts are small: a station has {n_lo} to {n_hi} seasons with a gauge record"

                  + f", of which {f_lo} to {f_hi} are flood seasons at the 1-in-3 level, so a single season moves POD by {1 / f_hi:.2f} to {1 / f_lo:.2f}.")

    # ---- day-to-day agreement
    span_note = "Seasons with a gauge record, 1999 to 2023. Dark = high; the shading runs from 0.4 to 1."
    dg = {
        "same": grid_fig(T, "rho_same_day", "rho_same_day", "Rank correlation of daily values, model and gauge on the same day", "corr",
                         sub=lambda r: f"{r.rho_lo:.2f} to {r.rho_hi:.2f}", note=span_note + " Small figures: the range when the seasons are redrawn (5th to 95th percentile)."),
        "cycle": grid_fig(T, "rho_cycle_removed", "rho_cycle_removed", "Rank correlation of daily values with the usual level for each calendar day taken out", "corr", note=span_note),
        "shift": grid_fig(T, "rho_best_shift", "rho_best_shift", "Rank correlation of daily values with the model's curve slid to where the value is highest (10 days behind to 30 ahead)", "corr",
                          sub=lambda r: f"{int(r.best_shift_days):+d} d", note=span_note + " Small figure: the shift in days; plus means the model's curve runs ahead of the gauge's."),
        "flood": grid_fig(T, "rho_flood_seasons", "rho_flood_seasons", "Rank correlation of daily values with the model's curve slid to where the value is highest, on flood seasons only (gauge or model over its own 1-in-3)", "corr",
                          sub=lambda r: f"{int(r.n_flood_seasons)} seasons", note="Dark = high; the shading runs from 0.4 to 1. Small figure: seasons used, which differ by source because each source has its own flood seasons."),
    }
    dg_extra = {"flood": flood_view_note(T)}
    dg_blocks = "".join(sw("dg", {"view": k}, f"<p>{text}{dg_extra.get(k, '')}</p>" + fig_html(dg[k], lab)) for k, lab, text in D_METRICS)

    # ---- ordering of seasons
    F["rmax"] = grid_fig(E3, "rho_season_peaks", "rho_season_peaks", "Rank correlation of season peaks, model against gauge", "good", sub=lambda r: f"{int(r.n)} seasons",
                         note="Seasons with a gauge record, 1999 to 2023. Dark = high; the shading runs from 0 to 1. Small figure: seasons used.")
    auc_note = "0.5 is a coin toss. Dark = high; the shading runs from 0.5 to 1."
    F["auc3"] = grid_fig(E3, "AUC", "auc_3", "AUC: chance that a flood season has a higher model peak than a season without a flood. Flood = gauge over its 1-in-3", "auc", note=auc_note)
    F["auc5"] = grid_fig(E5, "AUC", "auc_5", "AUC: chance that a flood season has a higher model peak than a season without a flood. Flood = gauge over its 1-in-5", "auc", note=auc_note)

    # ---- size of the flows
    q_rows = [s for s in ("luuq", "bardheere", "belet_weyne", "bulo_burti") if s in set(Q.station)]
    q_panels = [("deyr", SEASON_TITLE["deyr"]), ("gu", SEASON_TITLE["gu"]), ("year", "Whole year")]
    q_blocks = []
    for key, lab, kind, fmt, _ in Q_METRICS:
        note = {"good": "Dark = high.", "eff": "Green: above 0. Brown: below 0; -1 and lower take the darkest brown.", "ratio": "Teal: above 1. Brown: below 1; the shading is full at 2 and at 0.5."}[kind]
        note = {"KGE": "1 = the model matches SWALIM's series. ", "NSE": "1 = the model matches SWALIM's series; 0 = no closer than SWALIM's average. ",
                "pearson_r": "Plain correlation of the daily values, not the rank correlation of section 7. The shading runs from 0.4 to 1. "}.get(key, "") + note
        if "bardheere" in q_rows:
            note += " * SWALIM's Bardheere series does not agree with its Luuq series (see the text)."
        src = grid_fig(Q, key, f"q_{key}", f"{LONG.get(key, lab)}: model flow against SWALIM's discharge series, 1999 to 2023", "corr" if key == "pearson_r" else kind, fmt=fmt,
                       note=note, rows=q_rows, panels=q_panels, row_labels={"bardheere": "Bardheere *"})
        q_blocks.append(sw("q", {"metric": key}, fig_html(src, f"{lab} against SWALIM discharge")))
    q_text = "".join(sw("q", {"metric": key}, text, tag="p") for key, _, _, _, text in Q_METRICS)
    ceil = [f"{NAME[st]} on {QN[st]['days_at_top']} days" for st in q_rows if QN[st]["days_at_top"] >= 10]
    CEIL_TXT = ("The series repeats its highest value at " + " and at ".join(ceil) + ", so floods above that value are not told apart. " if ceil else "")
    bl = QN.get("bardheere_over_luuq")
    BARD_TXT = (f"At Bardheere, SWALIM's typical flow is {bl['swalim']:.1f} times its typical flow at Luuq, the next station upstream, while every model has about the same flow at the two "
                f"({min(bl[m] for m in MODELS):.1f} to {max(bl[m] for m in MODELS):.1f} times). The Bardheere series is therefore not used here as a measure of size." if bl else "")

    # ---- gauges
    flat = GN[GN.n_flat_top >= 3].sort_values("n_flat_top", ascending=False)
    FLAT_TXT = "; ".join(f"{NAME[r.station]} in {r.season.title()} ({int(r.n_flat_top)} of {int(r.n_seasons)} seasons)" for r in flat.itertuples())

    # ---- published files, with plain column names
    PUB = {"hits": "floods_reached", "misses": "floods_not_reached", "correct_negatives": "neither", "hit_years": "years_flood_reached",
           "false_years": "years_false_alarm", "missed_years": "years_flood_not_reached", "hits_held_out": "floods_reached_held_out",
           "misses_held_out": "floods_not_reached_held_out", "lead": "days_ahead", "lead_days_median": "days_before_second_gauge_median"}

    def recs(df):
        return json.loads(df.rename(columns=PUB).to_json(orient="records"))
    tables = (("daily_agreement", T), ("station_floods_1in3", E3), ("station_floods_1in5", E5), ("river_rule", Rv), ("forecasts_by_days_ahead", BL),
              ("discharge", Q), ("gauges", GN))
    (OUT / "data.json").write_text(json.dumps(G.clean(dict(
        sources={m: SHORT[m] for m in MODELS}, largest_seasons=M["worst"], discharge_series_notes=QN,
        **{nm: recs(df) for nm, df in tables})), indent=1))
    for old in OUT.glob("*.csv"):
        old.unlink()
    for nm, df in tables:
        d = df.assign(model=df.model.map(SHORT)) if "model" in df.columns else df
        d.rename(columns=PUB).to_csv(OUT / f"{nm}.csv", index=False)

    today = dt.date.today().isoformat()
    FIND = findings(M, LT, Rrec)
    RULE = rule_summary(M)
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
.pick {{ display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 4px; }}
.pick button {{ font:500 13px Roboto,Arial,sans-serif; color:var(--n8); background:#fff; border:1px solid #cfd6d6; border-radius:16px; padding:5px 13px; cursor:pointer; }}
.pick button.on {{ background:var(--b5); border-color:var(--b5); color:#fff; }}
.pick button:focus-visible {{ outline:2px solid var(--b7); outline-offset:2px; }}
.picklabel {{ font-size:12px; color:var(--n7); text-transform:uppercase; letter-spacing:.05em; margin:14px 0 0; }}
.tablewrap {{ overflow-x:auto; }}
table.data {{ border-collapse:collapse; width:100%; margin:12px 0 20px; font-size:14px; }}
table.data th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:6px 6px 5px; vertical-align:bottom; }}
table.data th[colspan] {{ text-align:center; border-bottom:1px solid #e2e7e7; }}
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
         and GloFAS v5 are two versions of the Copernicus Global Flood Awareness System. Each has a reanalysis: the model run over past years with the
         weather on record, 1999 to 2023 here.
         Three have an archive of the forecasts they would have issued: Google, new 2016 to 2025, Google, current 2016 to June 2023, GloFAS v4 2003 to 2023.
         Forecasts are compared on 2016 to 2023.</p>
      <p><strong>Benchmark.</strong> The SWALIM river gauges at the same seven stations. A gauge measures water height and a model gives flow, so each is read
         against its own levels. The 1-in-3 level of a gauge or a model is the value its season peak reaches in about one season in three; the 1-in-5
         level, one in five. A flood season at a station is a season whose peak reaches the 1-in-3 level (or the 1-in-5 where stated). Gauge levels are
         set from 2000 to 2023, model levels from 1999 to 2023. The seasons are Deyr, October to December, and Gu, March to May.</p>
    </div>

    <h2><span class="num">1</span>What the records show</h2>
    <p>The trigger rule on the new model's reanalysis, 1999 to 2023: how many of the river floods at the gauges had the threshold reached, which floods did not,
       and the seasons in which the threshold was reached with no river flood. Section 3 has the same for every source.</p>
    <div class="tablewrap">{RULE}</div>
    <ul class="takeaways">
{FIND}
    </ul>

    <h2><span class="num">2</span>The largest floods, day by day</h2>
    <p>The seasons that rank among the gauges' five largest on either river, one panel per station. Every line is drawn as a share of its own 1-in-3 level, so 100% is the
       flood level for the gauge and for the model alike. The gauge is a water level and the model a flow, so the heights of the curves differ; what can be
       compared is when each rises and whether each reaches its 100% line. Where a gauge line runs flat at its top, the gauge gave the same reading on
       those days (section 10). A line that stops before the end of the season is the end of that record. The vertical scale changes from one season to
       the next.</p>
    <p class="picklabel">Season</p>
    {pick("ts", "year", ts_opts)}
    <p class="picklabel">Lines</p>
    {pick("ts", "lines", [("new", "Gauge and the new model"), ("all", "Gauge and all four sources")])}
    {"".join(ts_figs)}

    <h2><span class="num">3</span>The trigger rule by river</h2>
    <p>The trigger needs three of the four Juba stations, or two of the three Shabelle stations, over their own level on the same day. The benchmark is a river flood: two gauges on the river over their own level at any time in the season. The grids show every season at
       the 1-in-3 level, with a reanalysis row and a forecast row for each source. For the forecast rows the rule is checked on each day's forecast, 1 to 7 days
       ahead, and the first forecast of the season that meets it is taken.</p>
    {fig_html(F["bt_deyr"], "Deyr, the trigger rule season by season")}
    {fig_html(F["bt_gu"], "Gu, the trigger rule season by season")}
    <p>The tables give the counts, the years and the scores (each score is explained in section 6). The forecast rows cover 2016 to 2023 only, so beside
       each one is the reanalysis on those same seasons: the two can be compared, and neither can be compared with the 1999 to 2023 row above them.
       Under POD and FAR is the range each takes when the seasons are redrawn at random, which shows how much a handful of seasons can move it.</p>
    <p class="picklabel">Level</p>
    {pick("rv", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {rv_blocks}
    <p class="note">Reached against the second gauge: from the day the rule is met (reanalysis) or the day of the first forecast meeting it (forecasts) to
       the day the second gauge crossed its level. The rows marked 2016 to 2025 score the new model's forecasts over its whole archive; no other source
       has forecasts after 2023. On the Juba, 2024 and 2025 rest on the Dollow and Luuq gauges only.</p>
    <h3>With each level fitted without the season being scored</h3>
    <p>A level fitted on the same seasons it is then scored on has already seen the answers. Here the level for each season, for the model and for
       the gauges, is fitted on all the other seasons, and the season is then scored. The gauge levels are refitted too, so the number of river floods
       can change. Reanalysis, 1999 to 2023.</p>
    {ho_blocks}

    <h2><span class="num">4</span>The forecasts, by days ahead</h2>
    <p>Each forecast archive is read one step at a time. Day 1 is the day the forecast is issued, day 2 the next day, and so on.
       The new model goes to 10 days, the others to 7. Seasons 2016 to 2023. GloFAS v4 runs several versions of each forecast; the middle one is used.</p>
    <p class="picklabel">Measure</p>
    {pick("ld", "metric", [("rise", "Share of the coming rise"), ("peak", "Forecast peak / reanalysis peak"), ("rho", "Rank correlation with the gauge"), ("pod", "POD"), ("far", "FAR")])}
    {sw("ld", {"metric": "rise"}, "<p>Take a forecast made some days ahead, and the days on which the river then rose most over those days (the top tenth of rises in the model's own reanalysis). The measure is the rise the forecast showed, divided by the rise that came. 100% means the forecast showed all of it; 0% means it still showed the river where it was; below 0%, it showed a fall; above 100%, more than came.</p>" + fig_html(F["lead_rise"], "Share of the coming rise already in the forecast, by days ahead"))}
    {sw("ld", {"metric": "peak"}, "<p>The forecast's highest value of the season divided by the reanalysis's highest value on the same days. At 1 the forecast carries the same peak as the reanalysis, so a level fitted on the reanalysis is reached as easily on the forecast; below 1, less easily; above 1, more easily.</p>" + fig_html(F["lead_peak"], "Forecast peak over reanalysis peak by days ahead"))}
    {sw("ld", {"metric": "rho"}, "<p>Rank correlation of the forecast with the gauge on the day forecast (section 7 explains the measure), averaged over the stations of the river.</p>" + fig_html(F["lead_rho"], "Rank correlation with the gauge by days ahead"))}
    {sw("ld", {"metric": "pod"}, "<p>POD at single stations at the 1-in-3 level, with the counts of the stations of a river added up.</p>" + fig_html(F["lead_pod"], "POD by days ahead"))}
    {sw("ld", {"metric": "far"}, "<p>FAR at single stations at the 1-in-3 level, with the counts of the stations of a river added up.</p>" + fig_html(F["lead_far"], "FAR by days ahead"))}
    <p class="note">At 7 days ahead the new model's daily forecast and its reanalysis have a rank correlation of {d7.rho_vs_reanalysis.min():.3f} to {d7.rho_vs_reanalysis.max():.3f}
       across the seven stations and two seasons. Google's description of its forecasting model (Cohen and others, 2026, preprint) says the days ahead are
       run with archived weather forecasts from ECMWF and GraphCast. We have not confirmed that this holds for the November 2026 release.</p>

    <h2><span class="num">5</span>Timing at each station</h2>
    <p>Two measures in days, over the flood seasons at each gauge: when the model peaks against when the gauge peaks, and when the model first reaches its
       1-in-3 level against when the gauge first reaches its own. Plus means the model is ahead of the gauge. Where a gauge holds one value at its top
       for several days, its peak is dated to the first of those days, so peak dates at those gauges are loose. These two measures and the shift in
       section 7 look at different seasons and different parts of the curve, so they can differ in sign at one station.</p>
    {fig_html(F["pk"], "Days between model peak and gauge peak")}
    {fig_html(F["cx"], "Days between model and gauge reaching the 1-in-3")}

    <h2><span class="num">6</span>Flood seasons at each station</h2>
    <p>Each season at a station falls into one of four boxes: a flood at the gauge with the model's threshold reached; a flood with the threshold not
       reached; the threshold reached with no flood, which is a false alarm; or neither. Every score below is a different way of reading those four
       counts. {SAMPLE_TXT}</p>
    <p class="picklabel">Score</p>
    {pick("ev", "metric", [(k, lab) for k, lab, _, _, _ in METRICS])}
    <p class="picklabel">Flood level</p>
    {pick("ev", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {ev_text}
    {"".join(ev_blocks)}

    <h2><span class="num">7</span>Day-to-day agreement with the gauges</h2>
    <p>Rank correlation puts the days in order by model flow and, separately, by gauge height, and measures how far the two orders agree: 1 means the
       days fall in exactly the same order, 0 means no relation. It uses the order of the days, not their values, so the scale of a model does not
       matter.</p>
    <p class="picklabel">View</p>
    {pick("dg", "view", [(k, lab) for k, lab, _ in D_METRICS])}
    {dg_blocks}

    <h2><span class="num">8</span>How each model orders the seasons</h2>
    <p>Two measures of whether a model's large seasons are the gauge's large seasons. The rank correlation of season peaks compares the order of the
       seasons by their highest model flow with their order by highest gauge level. AUC is the chance that a flood season picked at random carries a higher
       model peak than a season without a flood picked at random: 0.5 is a coin toss, 1 means every flood season has a higher model peak than every season without a flood.
       Neither depends on where the model's threshold is set.</p>
    {fig_html(F["rmax"], "Rank correlation of season peaks")}
    <p class="picklabel">Flood level at the gauge</p>
    {pick("auc", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {sw("auc", {"level": 3}, fig_html(F["auc3"], "AUC, 1-in-3"))}
    {sw("auc", {"level": 5}, fig_html(F["auc5"], "AUC, 1-in-5"))}

    <h2><span class="num">9</span>Size of the flows against SWALIM's discharge series</h2>
    <p>SWALIM publishes a discharge series, in cubic metres per second, at four of the seven stations. It is worked out from the gauge height: the two
       move in exactly the same order from day to day. {CEIL_TXT}{BARD_TXT}</p>
    <p class="picklabel">Measure</p>
    {pick("q", "metric", [(k, lab) for k, lab, _, _, _ in Q_METRICS])}
    {q_text}
    {"".join(q_blocks)}
    <p class="note">The trigger reads each model against the model's own levels, so a constant scaling of a model's flow does not change the scores in
       sections 3 to 8.</p>

    <h2><span class="num">10</span>The gauges used as the benchmark</h2>
    <p>A level meant to be reached in one season in three would be reached in about a third of the seasons on record. The table shows how often each
       gauge reaches it, and how often a gauge sits at one reading for five days or more at the height of a season: {FLAT_TXT}. At those gauges the size
       and the date of a peak are not known beyond that reading.</p>
    <div class="tablewrap">{gauge_table(GN)}</div>

    <h2><span class="num">11</span>What this review does not establish</h2>
    <ul>
      <li>On 5 November 2026 Google's Flood Hub, its public forecast service, was not yet serving the new model. Everything here is from Google's published
          reanalysis and forecast archive.</li>
      <li>The new model's forecast archive matches its reanalysis closely, and shows almost the whole of a coming rise a week ahead (section 4).
          Whether a live forecast will match as closely is not known from the archive.</li>
      <li>The forecast comparison rests on eight seasons, 2016 to 2023, with two to four river floods per river and season.</li>
      <li>The river benchmark (two gauges, at any time in the season) and the trigger rule (three of four or two of three stations, on the same day) are
          not the same test, and the benchmark depends on which gauges were reporting.</li>
      <li>The Bardheere gauge has no readings after November 2023 and the Bualle gauge none after March 2024, so the lower Juba cannot be checked from then on.</li>
      <li>Each model's flow is read at one place on its river network per station: the cell or reach on the main river nearest the station.</li>
      <li>Stations on one river rise together, so the seven stations are not seven separate tests, and totals added over stations count the same floods more than once.</li>
    </ul>

    <h2><span class="num">12</span>Method and files</h2>
    <p class="note">Levels are read from each record's own season peaks: the peaks are ranked, and the 1-in-3 level is the value at the rank that one
       season in three reaches (Weibull plotting position). Gauges 2000 to 2023, models 1999 to 2023. A value equal to its level counts as reaching it. A gauge season counts with at least 30 readings; a reading of zero is treated as a gap. Station floods
       compare the season's highest values; the river rule counts the stations over their level on the same day, in the season months only. Ranges come
       from redrawing the seasons with replacement ({R.N_BOOT} times for correlations, 1,000 for counts) and are the 5th to 95th percentile. Middle values of
       days that fall on a half are rounded away from zero. Days ahead: 1 is the day the forecast is issued. Google, current's archive stamps its forecast
       days one day later than its reanalysis days. In section 4 it is lined up on its reanalysis. In the river rule its days 1 to 7 are the seven days
       after the issue day, as the live trigger reads them.
       Sources: Google Runoff Reanalysis and Reforecast, November 2026 release; SWALIM gauge levels and discharge series; GloFAS v4 and v5 from the
       Copernicus Emergency Management Service.</p>
    <p class="note files">Every score, by season, river and station:
      <a href="river_rule.csv">river rule</a><a href="station_floods_1in3.csv">station floods, 1-in-3</a><a href="station_floods_1in5.csv">station floods, 1-in-5</a>
      <a href="daily_agreement.csv">daily agreement and timing</a><a href="forecasts_by_days_ahead.csv">forecasts by days ahead</a><a href="discharge.csv">discharge</a>
      <a href="gauges.csv">gauges</a><a href="data.json">all, as JSON</a></p>
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
