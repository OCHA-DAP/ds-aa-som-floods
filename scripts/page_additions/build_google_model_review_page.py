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
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
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


DARK_INK = "#141414"
_BODY_RGB = matplotlib.colors.to_rgb(DARK_INK)


def _ink(face):
    """White or body-colour text, whichever stands out more on the cell."""
    lf = _lum(face)
    c_white = 1.05 / (lf + 0.05)
    c_body = (lf + 0.05) / (_lum(_BODY_RGB) + 0.05)
    return "white" if c_white > c_body else DARK_INK


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


def grid_fig(df, col, name, title, kind, fmt="{:.2f}", sub=None, note=None, rows=None, panels=None, row_labels=None, models=None):
    """One figure: a panel per season (or per period), a row per station, a column per source."""
    panels = panels or [("deyr", SEASON_TITLE["deyr"]), ("gu", SEASON_TITLE["gu"])]
    rows = rows or STATIONS
    MODELS = models or globals()["MODELS"]
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
                    ax.text(j, i + 0.26, s_, ha="center", va="center", fontsize=8.8 * k, color=ink)
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
        fig.text(0.01, -0.02, textwrap.fill(note, 165), ha="left", va="top", fontsize=9 * k, color=BODY, linespacing=1.45)
    fig.subplots_adjust(wspace=0.06, left=0.16 if len(panels) == 2 else 0.12, right=0.99, top=0.78, bottom=0.03)
    return save(fig, name)


# ---------------------------------------------------------------------- worst-year series
FC_COL = "#2A1740"   # the new model's forecast: a dark shade of its colour


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
                ax.plot(s.index, s.values, color=FC_COL, lw=1.6, ls=(0, (3, 2.2)), zorder=3.5)
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
            H.append(Line2D([], [], color=FC_COL, lw=1.6, ls=(0, (3, 2.2)), label="Google, new: forecast day 7\n(issued six days earlier; where it cannot\nbe seen it lies on the solid line)"))
    else:
        H += [Line2D([], [], color=COL[m], lw=1.2, label=f"{SHORT[m]}: reanalysis") for m in MODELS[1:] if year <= G.RA_SPAN[m][1]]
        gone = [SHORT[m] for m in MODELS[1:] if year > G.RA_SPAN[m][1]]
        if gone:
            H.append(Line2D([], [], color="none", label=textwrap.fill("No record in " + str(year) + ": " + "; ".join(gone), 44)))
    H.append(Line2D([], [], color=INK, lw=1.0, ls=":", label="100%: each line's own 1-in-3 level"))
    axes[1, 3].legend(handles=H, loc="center left", bbox_to_anchor=(-0.05, 0.5), fontsize=10, frameon=False)
    fig.suptitle(f"{season.title()} {year}", x=0.01, ha="left", fontsize=13)
    fig.subplots_adjust(wspace=0.08, hspace=0.32, left=0.07, right=0.99, top=0.9, bottom=0.07)
    return save(fig, f"ts_{season}_{year}_{variant}", tight=False)


# --------------------------------------------------------------------------- by lead
CELL = {"hit": "#1B5E20", "false": "#B08D57", "miss": "#3D3D3D", "quiet": "#D5DCE3", "none": "#FFFFFF"}
REC_OF = {"ra": "reanalysis", "fc": "forecasts"}
PANELS4 = [(s, r) for s in ("deyr", "gu") for r in G.RIVERS]
LABEL_BOX = dict(facecolor="white", edgecolor="none", pad=1.2)


def rule_cells(M, rp, season, river, m, basis):
    """Year -> outcome of the trigger rule against the river flood, for one record of one source."""
    D = M[f"river_dates_{rp}"]
    assess = set(D[(season, river, "assess", "assess", rp)])
    floods = set(D[(season, river, "gauge", "gauge", rp)])
    reached = set(D[(season, river, m, REC_OF[basis], rp)])
    have = set(range(R.Y0, G.RA_SPAN[m][1] + 1)) if basis == "ra" else R.fc_years(m, season)
    out = {}
    for y in G.YEARS:
        if y not in have or y not in assess:
            out[y] = "none"
        elif y in reached:
            out[y] = "hit" if y in floods else "false"
        else:
            out[y] = "miss" if y in floods else "quiet"
    return out


def fig_rule_grid(M, Rv, rp, lines, name, title, thin=(), thick=()):
    """One row per record, one cell per season. lines: [(label, season, river, model, basis)], top
    to bottom. The counts at the right are the scored ones: reanalysis to 2023, forecasts 2016 to 2023."""
    years = G.YEARS
    n = len(lines)
    fig_h = 0.5 * n + 2.6
    fig, ax = plt.subplots(figsize=(14.8, fig_h))
    for i, (label, season, river, m, basis) in enumerate(lines):
        yy = n - 1 - i
        cells = rule_cells(M, rp, season, river, m, basis)
        for j, y in enumerate(years):
            kind = cells[y]
            ax.add_patch(plt.Rectangle((j - 0.46, yy - 0.36), 0.92, 0.72, facecolor=CELL[kind], edgecolor="white" if kind != "none" else "#C9D1D8", linewidth=0.8))
        r = Rv[(Rv.rp == rp) & (Rv.season == season) & (Rv.river == river) & (Rv.model == m) & (Rv.record == REC_OF[basis])].iloc[0]
        lo, hi = (R.Y0, R.Y1) if basis == "ra" else (R.FC0, R.FC1)
        inside = [cells[y] for y in years if lo <= y <= hi]
        assert (inside.count("hit"), inside.count("false"), inside.count("miss")) == (int(r.hits), int(r.false_alarms), int(r.misses)), (name, label)
        n_f, n_a = int(r.n_floods), int(r.false_alarms)
        t1 = f"reached in {int(r.hits)} of {n_f} river flood{'s' if n_f != 1 else ''}" if n_f else "no river flood in these seasons"
        ax.text(len(years) + 0.2, yy + 0.16, t1, va="center", fontsize=10, color=INK)
        ax.text(len(years) + 0.2, yy - 0.21, f"{n_a} false alarm{'s' if n_a != 1 else ''}", va="center", fontsize=10, color="#8A6A2B" if n_a else FAINT)
    for i in thin:
        ax.plot([-0.6, len(years) - 0.4], [n - 1 - i - 0.5] * 2, color="#B8C1C9", linewidth=0.8)
    for i in thick:
        ax.plot([-0.6, len(years) - 0.4], [n - 1 - i - 0.5] * 2, color=INK, linewidth=1.0)
    for edge in (R.FC0, R.FC1 + 1):
        ax.axvline(years.index(edge) - 0.5, color=INK, linewidth=0.9, ls=":", zorder=4)
    ax.set_xlim(-0.6, len(years) + 9.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n), [ln[0] for ln in lines[::-1]], fontsize=10.5)
    ax.set_xticks(range(0, len(years), 2), [str(y) for y in years[::2]], fontsize=10.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [plt.Rectangle((0, 0), 1, 1, facecolor=CELL["hit"], label="threshold reached, river flood at the gauges"),
         plt.Rectangle((0, 0), 1, 1, facecolor=CELL["false"], label="threshold reached, no river flood (false alarm)"),
         plt.Rectangle((0, 0), 1, 1, facecolor=CELL["miss"], label="river flood, threshold not reached"),
         plt.Rectangle((0, 0), 1, 1, facecolor=CELL["quiet"], label="neither"),
         plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#C9D1D8", label="no model record or no gauge record")]
    part1 = (f"Counts at the right. Reanalysis rows: every season with a gauge record to {R.Y1}. Forecast rows: the seasons between the dotted lines, {R.FC0} to {R.FC1}"
             + (" (Google, current in Deyr: to 2022)." if any(ln[3] == "google_grrr" and ln[1] == "deyr" for ln in lines) else "."))
    part2 = f"A river flood is two gauges on the river at or over their own 1-in-{rp} level. On the Juba, 2024 and 2025 rest on the Dollow and Luuq gauges only."
    note = textwrap.fill(part1, 150) + chr(10) + textwrap.fill(part2, 150)
    nl = note.count(chr(10)) + 1
    fig.text(0.215, 0.08 / fig_h, note, fontsize=10, color=BODY, ha="left", va="bottom", linespacing=1.5)
    fig.legend(handles=H, ncol=3, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, (0.75 + 0.22 * nl) / fig_h), frameon=False)
    ax.set_title(title, fontsize=12.5, loc="left")
    fig.subplots_adjust(left=0.215, right=0.995, top=1 - 0.42 / fig_h, bottom=(1.1 + 0.22 * nl) / fig_h)
    return save(fig, name, tight=False)


def fig_new_model(M, Rv, rp):
    """Section 1: the new model only, reanalysis and forecasts, both seasons and both rivers."""
    lines = []
    for season, river in PANELS4:
        lines.append((f"{season.title()}, {river.title()}: reanalysis", season, river, NEW, "ra"))
        lines.append((f"{season.title()}, {river.title()}: forecasts", season, river, NEW, "fc"))
    return fig_rule_grid(M, Rv, rp, lines, f"new_model_{rp}", f"Google, new: the trigger rule at the 1-in-{rp} level, season by season", thin=(1, 5), thick=(3,))


def fig_outcomes(M, Rv, rp, season):
    """Section 3: every source, a reanalysis row and, where an archive exists, a forecast row."""
    lines = []
    for river in G.RIVERS:
        for m in MODELS:
            lines.append((f"{river.title()}: {SHORT[m]}, reanalysis", season, river, m, "ra"))
            if m in G.FC_SPAN:
                lines.append((f"{river.title()}: {SHORT[m]}, forecasts", season, river, m, "fc"))
    n_juba = sum(1 for ln in lines if ln[2] == "juba")
    return fig_rule_grid(M, Rv, rp, lines, f"outcomes_{season}_{rp}", f"{SEASON_TITLE[season]}: the trigger rule at the 1-in-{rp} level, season by season", thick=(n_juba - 1,))


# ------------------------------------------------------------------- bars, one panel per river and season
def hbar_fig(rows, cell, name, title, xlim, xticks, note, ref=None, groups=(), xlabel=None, xticklabels=None, legend=None):
    """rows: [(key, label)], top to bottom. cell(season, river, key) -> dict with v (bar length),
    text (label at the end of the bar), color, hatch (True for a forecast row), lo and hi (a range
    drawn as a black line), or segs [(length, colour)] for a bar made of counts."""
    n = len(rows)
    fig_h = 0.43 * n + 2.0
    fig, axes = plt.subplots(1, 4, figsize=(15.2, fig_h), sharey=True)
    span = xlim[1] - xlim[0]
    for ax, (season, river) in zip(axes, PANELS4):
        for i, (key, _) in enumerate(rows):
            c = cell(season, river, key)
            if c.get("segs") is not None:
                x = 0.0
                for length, colour in c["segs"]:
                    if length > 0:
                        ax.barh(i, length, left=x, height=0.62, color=colour, edgecolor="white", linewidth=1.0)
                        ax.text(x + length / 2, i, f"{int(length)}", ha="center", va="center", fontsize=10.5, color=_ink(matplotlib.colors.to_rgb(colour)))
                    x += length
                ax.text(x + 0.02 * span, i, c["text"], va="center", ha="left", fontsize=10.5, color=INK)
                continue
            v = c.get("v")
            if v is None or np.isnan(v):
                ax.text(0.02 * span, i, c.get("text", "n/a"), va="center", ha="left", fontsize=10, color=BODY)
                continue
            if c.get("hatch"):
                ax.barh(i, v, height=0.62, facecolor="white", edgecolor=c["color"], hatch="////", linewidth=1.2)
            else:
                ax.barh(i, v, height=0.62, color=c["color"], edgecolor=c["color"], linewidth=1.2)
            end = v
            if c.get("lo") is not None and not np.isnan(c["lo"]):
                ax.plot([c["lo"], c["hi"]], [i, i], color=INK, lw=1.4, zorder=5)
                for e in (c["lo"], c["hi"]):
                    ax.plot([e, e], [i - 0.16, i + 0.16], color=INK, lw=1.4, zorder=5)
                end = max(v, c["hi"])
            tx, ha = (v - 0.02 * span, "right") if v < 0 else (end + 0.02 * span, "left")
            ax.text(tx, i - (0.17 if c.get("sub") else 0), c["text"], va="center", ha=ha, fontsize=10.5, color=INK, bbox=LABEL_BOX, zorder=6)
            if c.get("sub"):
                ax.text(tx, i + 0.21, c["sub"], va="center", ha=ha, fontsize=9.5, color=BODY, zorder=6)
        for g_ in groups:
            ax.axhline(g_ - 0.5, color="#B8C1C9", linewidth=0.8)
        if ref is not None:
            ax.axvline(ref, color=INK, lw=0.9, ls=":", zorder=1)
        if xlim[0] < 0:
            ax.axvline(0, color=INK, lw=0.9, zorder=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(n - 0.45, -0.55)
        ax.set_xticks(xticks, xticklabels or [f"{t:g}" for t in xticks])
        ax.set_title(f"{season.title()}, {river.title()}", fontsize=11.5, loc="left")
        ax.tick_params(length=0, labelsize=10, colors=BODY)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, color=BODY)
    axes[0].set_yticks(range(n), [lab for _, lab in rows], fontsize=10.5, color=BODY)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, y=1 - 0.12 / fig_h)
    wrapped = textwrap.fill(note, 185)
    foot = 0.12 + 0.21 * (wrapped.count(chr(10)) + 1)
    fig.text(0.01, 0.08 / fig_h, wrapped, fontsize=10, color=BODY, ha="left", va="bottom", linespacing=1.45)
    if legend:
        fig.legend(handles=legend, ncol=len(legend), fontsize=10.5, frameon=False, loc="lower left", bbox_to_anchor=(0.245, foot / fig_h))
        foot += 0.36
    foot += 0.56 if xlabel else 0.36
    fig.subplots_adjust(wspace=0.06, left=0.255, right=0.995, top=1 - 0.80 / fig_h, bottom=foot / fig_h)
    return save(fig, name, tight=False)


def rv_rows():
    """The records of the river rule, in page order: (model, record), label."""
    rows = []
    for m in MODELS:
        rows.append(((m, "reanalysis"), f"{SHORT[m]}: reanalysis, {R.Y0} to {R.Y1}"))
        if m in G.FC_SPAN:
            star = " *" if any(k[0] == m for k in G.FC_END) else ""
            rows.append(((m, "reanalysis, forecast seasons"), f"{SHORT[m]}: reanalysis, {R.FC0} to {R.FC1}{star}"))
            rows.append(((m, "forecasts"), f"{SHORT[m]}: forecasts, {R.FC0} to {R.FC1}{star}"))
            if G.FC_SPAN[m][1] > R.FC1:
                rows.append(((m, f"forecasts to {G.FC_SPAN[m][1]}"), f"{SHORT[m]}: forecasts, {R.FC0} to {G.FC_SPAN[m][1]}"))
    return rows


RV_METRICS = [  # key, button, title, meaning
    ("POD", "POD", "POD: share of river floods with the threshold reached",
     "POD: the share of river floods with the threshold reached. In brackets: river floods with the threshold reached, of all river floods. The black line is the range POD takes when it is recomputed on random picks of the seasons (section 13)."),
    ("FAR", "FAR", "FAR: share of seasons with the threshold reached that had no river flood",
     "FAR: the share of seasons with the threshold reached that had no river flood. In brackets: those seasons, of all seasons with the threshold reached. The black line is the range FAR takes when it is recomputed on random picks of the seasons (section 13)."),
    ("POFD", "POFD", "POFD: share of seasons with no river flood in which the threshold was reached",
     "POFD: the share of seasons with no river flood in which the threshold was reached. In brackets: those seasons, of all seasons with no river flood."),
    ("CSI", "CSI", "CSI: river floods with the threshold reached, divided by all river floods plus the false alarms",
     "CSI: river floods with the threshold reached, divided by all river floods plus the false alarms. 1 means the threshold was reached in every river flood with no false alarm."),
    ("F1", "F1", "F1: twice the river floods with the threshold reached, divided by that number plus all river floods plus the false alarms",
     "F1: twice the river floods with the threshold reached, divided by that number plus all river floods plus the false alarms. 1 means the threshold was reached in every river flood with no false alarm."),
    ("freq_bias", "Frequency bias", "Frequency bias: seasons with the threshold reached divided by river floods",
     "Frequency bias: seasons with the threshold reached divided by river floods. At 1, the dotted line, there are as many of one as of the other; above 1 the threshold is reached in more seasons than there are river floods."),
    ("HSS", "HSS", "HSS: agreement between the rule and the river floods beyond what chance would give",
     "HSS: agreement between the rule and the river floods beyond what chance would give. 1 means they agree in every season; 0 means no more than chance."),
    ("accuracy", "Accuracy", "Accuracy: share of seasons in which the rule and the river floods agree",
     "Accuracy: the share of all seasons in which the rule and the river floods agree, river flood or none. In brackets: those seasons, of all seasons. A rule that never reached its threshold would score the share of seasons with no river flood."),
    ("PSS", "PSS", "PSS: POD minus POFD",
     "PSS: POD minus POFD, from -1 to 1. 1 means the threshold was reached in every river flood and in no other season; 0 means the rule does not separate river floods from other seasons."),
    ("days", "Days before the second gauge", "Days from the threshold being reached to the second gauge reaching its level (middle value)",
     "Days from the threshold being reached to the day the second gauge reached its level, middle value over the river floods with the threshold reached. Reanalysis rows count from the day the rule is met; forecast rows from the day of the first forecast that meets it. Bars to the right: the threshold came before the second gauge. Bars to the left: after. n/a: no river flood with the threshold reached."),
]


def fig_rv_score(Rv, rp, key, title):
    d = Rv[(Rv.rp == rp) & ~Rv.record.str.contains("held out")]
    rows = rv_rows()
    col = "lead_days_median" if key == "days" else key

    def cell(season, river, k):
        m, rec = k
        r = d[(d.season == season) & (d.river == river) & (d.model == m) & (d.record == rec)]
        if not len(r):
            return dict(v=None, text="no forecasts after 2023")
        r = r.iloc[0]
        v = r[col]
        out = dict(v=v, color=COL[m], hatch=rec.startswith("forecasts"))
        if pd.isna(v):
            out["text"] = {"POD": "n/a: no river flood", "FAR": "n/a: threshold not reached", "days": "n/a"}.get(key, "n/a")
            return out
        if key == "POD":
            out.update(text=f"{v:.2f} ({int(r.hits)} of {int(r.n_floods)})", lo=r.POD_lo, hi=r.POD_hi)
        elif key == "FAR":
            out.update(text=f"{v:.2f} ({int(r.false_alarms)} of {int(r.n_reached)})", lo=r.FAR_lo, hi=r.FAR_hi)
        elif key == "POFD":
            out["text"] = f"{v:.2f} ({int(r.false_alarms)} of {int(r.n - r.n_floods)})"
        elif key == "accuracy":
            out["text"] = f"{v:.2f} ({int(r.hits + r.correct_negatives)} of {int(r.n)})"
        elif key == "days":
            out.update(text=notice(v), sub=f"{int(r.hits)} flood{'s' if r.hits != 1 else ''}")
        else:
            out["text"] = f"{v:.2f}"
        return out

    vals = d[col].dropna()
    groups = [i for i in range(1, len(rows)) if rows[i][0][0] != rows[i - 1][0][0]]
    note = (f"Solid bars: reanalysis. Hatched bars: forecasts. Rows for {R.FC0} to {R.FC1} hold the seasons with a gauge record in those years"
            + (" (* Google, current in Deyr: to 2022)." if G.FC_END else "."))
    if key in ("POD", "FAR"):
        note += " Black line: the range the score takes when the seasons are redrawn at random (middle 90%); a single tick means every redraw gave the same value."
    if key == "freq_bias":
        note += " Dotted line: 1, as many seasons with the threshold reached as river floods."
    if key == "days":
        note += (" Reanalysis rows count from the day the rule is met; forecast rows from the day of issue of the first forecast that meets it."
                 " Under each value: the river floods it is the middle value of.")
        big = float(vals.abs().max())
        top = max(5, int(np.floor(big / 5) * 5))
        return hbar_fig(rows, cell, f"rv_{key}_{rp}", f"{title}. 1-in-{rp} level", (-2.6 * big, 2.6 * big), [-top, 0, top], note, groups=groups,
                        xticklabels=[f"{top}", "0", f"{top}"], xlabel="days after  |  days before")
    if key == "freq_bias":
        top = max(1.5, float(np.ceil(vals.max() * 2) / 2))
        return hbar_fig(rows, cell, f"rv_{key}_{rp}", f"{title}. 1-in-{rp} level", (0, 1.5 * top), list(np.arange(0, top + 0.01, 0.5)), note, ref=1.0, groups=groups)
    lo = min(0.0, float(np.floor(vals.min() * 4) / 4))
    wide = 2.0 if key in ("POD", "FAR", "POFD", "accuracy") else 1.45
    return hbar_fig(rows, cell, f"rv_{key}_{rp}", f"{title}. 1-in-{rp} level", (lo * 1.9 if lo < 0 else 0, wide), ([lo] if lo < 0 else []) + [0, 0.5, 1], note, groups=groups)


def fig_held_out(Rv, rp):
    """Counts with each level fitted on all seasons, beside the counts with the level for each
    season fitted without that season."""
    d = Rv[Rv.rp == rp]
    rows = []
    for m in MODELS:
        rows.append(((m, "reanalysis"), f"{SHORT[m]}: fitted on all seasons"))
        rows.append(((m, "reanalysis, levels held out"), f"{SHORT[m]}: without the season scored"))

    def cell(season, river, k):
        m, rec = k
        r = d[(d.season == season) & (d.river == river) & (d.model == m) & (d.record == rec)].iloc[0]
        return dict(segs=[(int(r.hits), CELL["hit"]), (int(r.misses), CELL["miss"]), (int(r.false_alarms), CELL["false"])], text=f"F1 {num(r.F1)}")

    top = int((d.hits + d.misses + d.false_alarms).max())
    H = [plt.Rectangle((0, 0), 1, 1, facecolor=CELL["hit"], label="river floods with the threshold reached"),
         plt.Rectangle((0, 0), 1, 1, facecolor=CELL["miss"], label="river floods, threshold not reached"),
         plt.Rectangle((0, 0), 1, 1, facecolor=CELL["false"], label="threshold reached, no river flood (false alarm)")]
    return hbar_fig(rows, cell, f"held_out_{rp}", f"Seasons counted with each level fitted on all seasons, and with the level for each season fitted without that season. 1-in-{rp} level",
                    (0, top * 1.38), list(range(0, top + 1, 4)), f"Reanalysis, seasons with a gauge record, {R.Y0} to {R.Y1}. The gauge levels are refitted too, so the number of river floods can change.",
                    groups=range(2, len(rows), 2), xlabel="seasons", legend=H)


def fig_gauges(GN):
    """Per gauge and season: seasons on record, seasons at or over the 1-in-3 level, and seasons
    in which the gauge sat at one top reading for five days or more."""
    sts = [st for r in G.RIVERS for st in G.TRIGGER_STATIONS[r]]
    n_j = len(G.TRIGGER_STATIONS["juba"])
    xmax = int(GN.n_seasons.max())
    c_over, c_flat, c_all = "#2F6DA8", "#8A5A2B", "#E1E6EA"
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.1), sharey=True)
    for ax, season in zip(axes, ("deyr", "gu")):
        d = GN[GN.season == season].set_index("station")
        for i, st in enumerate(sts):
            r = d.loc[st]
            n = int(r.n_seasons)
            for off, val, colour in ((-0.19, int(r.n_at_or_over), c_over), (0.19, int(r.n_flat_top), c_flat)):
                ax.barh(i + off, n, height=0.33, color=c_all)
                ax.barh(i + off, val, height=0.33, color=colour)
                ax.text(n + 0.4, i + off, f"{val} of {n}", va="center", ha="left", fontsize=10.5, color=INK)
            ax.plot([n / 3, n / 3], [i - 0.19 - 0.21, i - 0.19 + 0.21], color=INK, lw=1.8, zorder=5)
            peak = f"{r.common_peak:.2f} m in {int(r.n_common_peak)} seasons" if r.n_common_peak >= 2 else "none repeated"
            ax.text(xmax + 6.5, i, peak, va="center", ha="left", fontsize=10.5, color=INK if r.n_common_peak >= 2 else BODY)
        ax.text(xmax + 6.5, -0.78, "Most common season peak", va="center", ha="left", fontsize=10, color=BODY)
        ax.axhline(n_j - 0.5, color=INK, linewidth=1.0)
        ax.set_xlim(0, xmax + 22)
        ax.set_ylim(len(sts) - 0.45, -1.05)
        ax.set_xticks(range(0, xmax + 1, 5))
        ax.set_xlabel("seasons", fontsize=10, color=BODY, loc="left")
        ax.set_title(SEASON_TITLE[season], fontsize=11.5, loc="left")
        ax.tick_params(length=0, labelsize=10, colors=BODY)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(sts)), [NAME[st] for st in sts], fontsize=11, color=BODY)
    axes[0].text(-0.20, (n_j - 1) / 2, "JUBA", transform=axes[0].get_yaxis_transform(), ha="center", va="center", rotation=90, fontsize=8.5, color=FAINT, clip_on=False)
    axes[0].text(-0.20, n_j + (len(sts) - n_j - 1) / 2, "SHABELLE", transform=axes[0].get_yaxis_transform(), ha="center", va="center", rotation=90, fontsize=8.5, color=FAINT, clip_on=False)
    H = [plt.Rectangle((0, 0), 1, 1, facecolor=c_over, label="seasons with the gauge at or over its 1-in-3 level"),
         plt.Rectangle((0, 0), 1, 1, facecolor=c_flat, label="seasons at one top reading for 5 days or more"),
         plt.Rectangle((0, 0), 1, 1, facecolor=c_all, label="seasons on record"),
         Line2D([0], [0], color=INK, lw=1.8, label="one third of the seasons on record")]
    fig.legend(handles=H, ncol=4, fontsize=10.5, frameon=False, loc="lower left", bbox_to_anchor=(0.095, 0.03))
    fig.suptitle(f"The SWALIM gauges, seasons to {R.Y1}: how often each is at or over its 1-in-3 level, and how often it sits at one top reading", x=0.01, ha="left", fontsize=12, y=0.985)
    fig.subplots_adjust(wspace=0.04, left=0.105, right=0.995, top=0.90, bottom=0.17)
    return save(fig, "gauges", tight=False)


FR_CELL = {"both": "#33415C", "fc": "#D98A2B", "ra": "#7FA6C9", "neither": "#D5DCE3", "none": "#FFFFFF"}


def fig_rule_fc_vs_ra(M, rp):
    """The trigger rule on the forecasts beside the rule on the same model's reanalysis, one square
    per season, for every season in which the model has both."""
    D = M[f"river_dates_{rp}"]
    fcm = [m for m in MODELS if m in G.FC_SPAN]
    years = list(range(min(G.FC_SPAN[m][0] for m in fcm), G.YEARS[-1] + 1))
    lines = [(season, river, m) for season, river in PANELS4 for m in fcm]
    n = len(lines)
    fig_h = 0.5 * n + 2.9
    fig, ax = plt.subplots(figsize=(14.8, fig_h))
    for i, (season, river, m) in enumerate(lines):
        yy = n - 1 - i
        ra, fc = D[(season, river, m, "reanalysis", rp)], D[(season, river, m, "forecasts", rp)]
        have = R.fc_years(m, season) & set(range(R.Y0, G.RA_SPAN[m][1] + 1))
        agree, gaps = 0, []
        for j, y in enumerate(years):
            a, b = y in ra, y in fc
            kind = "none" if y not in have else "both" if a and b else "ra" if a else "fc" if b else "neither"
            if y in have:
                agree += a == b
                if a and b:
                    gaps.append((ra[y] - fc[y]).days)
            ax.add_patch(plt.Rectangle((j - 0.46, yy - 0.36), 0.92, 0.72, facecolor=FR_CELL[kind], edgecolor="white" if kind != "none" else "#C9D1D8", linewidth=0.8))
        ax.text(len(years) + 0.2, yy + 0.16, f"agree in {agree} of {len(have)} seasons", va="center", fontsize=10, color=INK)
        if gaps:
            nt = notice(float(np.median(gaps)))
            t2 = "first forecast issued on the reanalysis day" if nt == "same day" else f"first forecast issued {nt.split()[0]} day{'' if nt.split()[0] == '1' else 's'} {nt.split()[1]} the reanalysis day"
            ax.text(len(years) + 0.2, yy - 0.21, t2, va="center", fontsize=10, color=BODY)
    for i in range(len(fcm) - 1, n - 1, len(fcm)):
        thick = (i + 1) == n // 2
        ax.plot([-0.6, len(years) - 0.4], [n - 1 - i - 0.5] * 2, color=INK if thick else "#B8C1C9", linewidth=1.0 if thick else 0.8)
    ax.set_xlim(-0.6, len(years) + 13.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n), [f"{s.title()}, {r.title()}: {SHORT[m]}" for s, r, m in lines[::-1]], fontsize=10.5)
    ax.set_xticks(range(0, len(years), 2), [str(y) for y in years[::2]], fontsize=10.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    H = [plt.Rectangle((0, 0), 1, 1, facecolor=FR_CELL["both"], label="threshold reached on both"),
         plt.Rectangle((0, 0), 1, 1, facecolor=FR_CELL["fc"], label="on the forecasts only"),
         plt.Rectangle((0, 0), 1, 1, facecolor=FR_CELL["ra"], label="on the reanalysis only"),
         plt.Rectangle((0, 0), 1, 1, facecolor=FR_CELL["neither"], label="on neither"),
         plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#C9D1D8", label="no forecast archive")]
    spans = "; ".join(f"{SHORT[m]} {G.FC_SPAN[m][0]} to {G.FC_SPAN[m][1]}" + (" (Deyr to 2022)" if ("google_grrr", "deyr") in G.FC_END and m == "google_grrr" else "") for m in fcm)
    note = (textwrap.fill(f"Each square is a season in which the model has a reanalysis and a forecast archive: {spans}. Gauges are not used here.", 150) + chr(10)
            + textwrap.fill("Forecast rows of the rule: the first forecast of the season that meets it, forecast days 1 to 7. Reanalysis day: the first day the rule is met on the reanalysis. "
                            "Days: middle value over the seasons with the threshold reached on both.", 150))
    nl = note.count(chr(10)) + 1
    fig.text(0.215, 0.08 / fig_h, note, fontsize=10, color=BODY, ha="left", va="bottom", linespacing=1.5)
    fig.legend(handles=H, ncol=5, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, (0.5 + 0.22 * nl) / fig_h), frameon=False)
    ax.set_title(f"The trigger rule at the 1-in-{rp} level: the forecasts beside the same model's reanalysis, season by season", fontsize=12.5, loc="left")
    fig.subplots_adjust(left=0.215, right=0.995, top=1 - 0.42 / fig_h, bottom=(0.85 + 0.22 * nl) / fig_h)
    return save(fig, f"rule_fc_vs_ra_{rp}", tight=False)


def fig_fc_vs_ra(B7, col, name, title, note, fmt="{:.2f}"):
    """Per station: a score of the reanalysis and of the forecast for forecast day 7, both against
    the gauge on the same seasons. One panel per season and source."""
    fcm = [m for m in MODELS if m in G.FC_SPAN]
    n_j = sum(RIVER_OF[st] == "juba" for st in STATIONS)
    vals = pd.concat([B7[col], B7["ra_" + col]]).dropna()
    lo = min(0.0, float(np.floor(vals.min() * 4) / 4))
    hi = max(1.0, float(np.ceil(vals.max() * 4) / 4))
    span = hi - lo
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6), sharey=True, sharex=True)
    for r_, season in enumerate(("deyr", "gu")):
        for c_, m in enumerate(fcm):
            ax = axes[r_, c_]
            d = B7[(B7.season == season) & (B7.model == m)].set_index("station")
            for i, st in enumerate(STATIONS):
                if st not in d.index:
                    ax.text(lo + 0.02 * span, i, "no gauge record in these seasons", va="center", ha="left", fontsize=10, color=BODY)
                    continue
                a, b = d.loc[st, "ra_" + col], d.loc[st, col]
                if pd.notna(a) and pd.notna(b):
                    ax.plot([a, b], [i, i], color="#8E99A6", lw=2.2, zorder=2, solid_capstyle="butt")
                if pd.notna(a):
                    ax.plot(a, i, "o", ms=9.5, color=COL[m], zorder=3)
                if pd.notna(b):
                    ax.plot(b, i, "D", ms=12, mfc="none", mec=DARK_INK, mew=1.7, zorder=4)
                ax.text(hi + 0.08 * span, i, f"{num(a, fmt)}  |  {num(b, fmt)}", va="center", ha="left", fontsize=10.5, color=INK)
            ax.text(hi + 0.08 * span, -0.95, "reanalysis  |  forecast", va="center", ha="left", fontsize=9.5, color=BODY)
            ax.axhline(n_j - 0.5, color=INK, linewidth=1.0)
            if lo < 0:
                ax.axvline(0, color=INK, lw=0.9, zorder=1)
            ax.set_xlim(lo - 0.06 * span, hi + 0.74 * span)
            ax.set_ylim(len(STATIONS) - 0.5, -1.45)
            ax.set_xticks([x for x in np.arange(lo, hi + 0.01, 0.25 if span <= 1.5 else 0.5)])
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.set_title(f"{season.title()}: {SHORT[m]}", fontsize=11.5, loc="left")
            ax.tick_params(length=0, labelsize=10, colors=BODY)
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[r_, 0].set_yticks(range(len(STATIONS)), [NAME[st] for st in STATIONS], fontsize=10.5, color=BODY)
    H = [Line2D([0], [0], marker="o", ls="", ms=9.5, color=BODY, label="reanalysis, on the same seasons and days (in the source's colour)"),
         Line2D([0], [0], marker="D", ls="", ms=12, mfc="none", mec=DARK_INK, mew=1.7, label="forecast for forecast day 7")]
    fig.legend(handles=H, ncol=2, fontsize=10.5, frameon=False, loc="lower left", bbox_to_anchor=(0.07, 0.065))
    fig.text(0.01, 0.012, textwrap.fill(note, 185), fontsize=10, color=BODY, ha="left", va="bottom", linespacing=1.45)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, y=0.985)
    fig.subplots_adjust(wspace=0.05, hspace=0.24, left=0.08, right=0.995, top=0.915, bottom=0.155)
    return save(fig, name, tight=False)


def add_scores(g, a="hits", b="false_alarms", c="misses", d="correct_negatives"):
    """Every score of google_review_metrics.contingency, from columns of counts."""
    A, B, C, D = (g[x].astype(float) for x in (a, b, c, d))
    n = A + B + C + D
    div = lambda x, y: (x / y).where(y > 0)
    g["POD"], g["FAR"], g["POFD"] = div(A, A + C), div(B, A + B), div(B, B + D)
    g["CSI"], g["F1"], g["freq_bias"] = div(A, A + B + C), div(2 * A, 2 * A + B + C), div(A + B, A + C)
    g["accuracy"] = div(A + D, n)
    exp = div((A + B) * (A + C) + (C + D) * (B + D), n)
    g["HSS"] = div((A + D) - exp, n - exp)
    g["PSS"] = g.POD - g.POFD
    return g


def lead_table(BL):
    g = BL.groupby(["model", "season", "river", "lead"]).agg(
        peak=("peak_ratio_median", "median"), rise=("rise_share_median", "median"), rho=("rho_vs_gauge", "mean"), rho_ra=("rho_vs_reanalysis", "mean"),
        a=("hits", "sum"), b=("false_alarms", "sum"), c=("misses", "sum"), d=("correct_negatives", "sum"),
        fr_ratio=("fr_ratio_median", "median"), fr_var=("fr_variability_ratio", "median"), fr_KGE=("fr_KGE", "median"), fr_NSE=("fr_NSE", "median"),
        fb=("fr_both", "sum"), ff=("fr_forecast_only", "sum"), fa=("fr_reanalysis_only", "sum"), fn=("fr_neither", "sum")).reset_index()
    tot = g.fb + g.ff + g.fa + g.fn
    g["fr_agree"] = ((g.fb + g.fn) / tot).where(tot > 0)
    return add_scores(g, "a", "b", "c", "d")


def fig_lead(g, col, name, ylabel, ylim, ref=None, ref_label=""):
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
            ax.set_title(f"{season.title()}, {river.title()}", fontsize=11.5, loc="left")
            ax.set_xticks(range(1, 11))
            ax.set_xlim(0.6, 10.4)
            ax.set_ylim(ylim[0] - 0.04 * (ylim[1] - ylim[0]), ylim[1] + 0.035 * (ylim[1] - ylim[0]))
            ax.set_xlabel("forecast day (1 = day of issue)", fontsize=10.5, color=BODY)
            ax.tick_params(length=0, labelsize=10, colors=BODY)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
    fig.suptitle(ylabel, x=0.01, ha="left", fontsize=12)
    h_, l_ = axes[0].get_legend_handles_labels()
    note = textwrap.fill("Forecasts of the seasons 2016 to 2023 (Google, current: Deyr to 2022). Google, current and GloFAS v4 end at forecast day 7. "
                         + (ref_label + " " if ref_label else "") + "Where a marker cannot be seen it lies under another.", 195)
    nl = note.count(chr(10)) + 1
    fig.text(0.04, 0.02, note, fontsize=10, color=BODY, ha="left", va="bottom", linespacing=1.45)
    fig.legend(h_, l_, ncol=3, fontsize=10.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.03 + 0.047 * nl))
    fig.subplots_adjust(wspace=0.07, left=0.04, right=0.99, top=0.80, bottom=0.245 + 0.047 * nl)
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


LONG = {"POD": "POD, probability of detection", "FAR": "FAR, false alarm ratio", "POFD": "POFD, probability of false detection",
        "CSI": "CSI, critical success index", "F1": "F1", "freq_bias": "Frequency bias", "accuracy": "Accuracy",
        "HSS": "HSS, Heidke skill score", "PSS": "PSS, Peirce skill score", "KGE": "KGE, Kling-Gupta efficiency", "NSE": "NSE, Nash-Sutcliffe efficiency"}
METRICS = [  # key, label, colour kind, format, plain meaning
    ("POD", "POD", "good", "{:.2f}", "POD, probability of detection: of the flood seasons at the gauge, the share in which the model reached its own threshold. 1 means the threshold was reached in every flood season; 0 means in none."),
    ("FAR", "FAR", "bad", "{:.2f}", "FAR, false alarm ratio: of the seasons in which the model reached its threshold, the share with no flood at the gauge. 0 means no false alarm."),
    ("POFD", "POFD", "bad", "{:.2f}", "POFD, probability of false detection: of the seasons with no flood at the gauge, the share in which the model still reached its threshold. 0 means the threshold was never reached without a flood."),
    ("CSI", "CSI", "good", "{:.2f}", "CSI, critical success index: the flood seasons with the threshold reached, divided by all flood seasons plus the false alarms. It falls with either kind of error; 1 means neither occurred."),
    ("F1", "F1", "good", "{:.2f}", "F1: twice the flood seasons with the threshold reached, divided by that number plus all flood seasons plus the false alarms. It is high only when POD is high and FAR is low; 1 means the threshold was reached in every flood season with no false alarm."),
    ("freq_bias", "Frequency bias", "ratio", "{:.2f}", "Frequency bias: seasons in which the model reached its threshold divided by flood seasons at the gauge. Above 1 the model reaches its threshold more often than the gauge floods; below 1, less often."),
    ("accuracy", "Accuracy", "good", "{:.2f}", "Accuracy: the share of all seasons in which model and gauge agree, flood or no flood. A model that never reached its threshold would score the share of seasons with no flood: ACC_RANGE."),
    ("HSS", "HSS", "eff", "{:.2f}", "HSS, Heidke skill score: accuracy with the agreement that chance alone would give taken out. 1 means model and gauge agree in every season; 0 means no more agreement than chance; below 0, less than chance."),
    ("PSS", "PSS", "eff", "{:.2f}", "PSS, Peirce skill score: POD minus POFD, from -1 to 1. 1 means the model separates flood seasons from the rest fully; 0 means it does not separate them."),
]
Q_METRICS = [
    ("ratio_median", "Flow on a typical day", "ratio", "{:.2f}", "Flow on a typical day (the middle day when all days are ranked by flow): the model's divided by SWALIM's. 1 means equal; 2 means the model runs at twice SWALIM's flow. Typical days are low-flow days; the next view compares the season peaks."),
    ("ratio_peak_median", "Ratio of season peaks", "ratio", "{:.2f}", "Ratio of season peaks: the model's highest flow of each season divided by SWALIM's, middle value over the seasons. Where SWALIM's series stops at a top value, its season peak is that value."),
    ("pearson_r", "Correlation", "good", "{:.2f}", "Correlation of daily values, from -1 to 1. 1 means the two rise and fall together in proportion; 0 means no straight-line relation."),
    ("variability_ratio", "Variability ratio", "ratio", "{:.2f}", "Variability ratio: the size of the model's swings divided by the size of SWALIM's. 1 means the same spread."),
    ("KGE", "KGE", "eff", "{:.2f}", "KGE, Kling-Gupta efficiency: one score built from the correlation, the variability ratio and the ratio of averages. 1 means the model matches SWALIM's series in all three; the lower the score, the further the model is from it."),
    ("NSE", "NSE", "eff", "{:.2f}", "NSE, Nash-Sutcliffe efficiency: 1 means the model matches SWALIM's series; 0 means it is no closer to it than a flat line at SWALIM's average; below 0, further away than that line."),
]
D_METRICS = [  # daily agreement: key, label, plain meaning
    ("same", "Same day", "Model and gauge compared on the same day. The small figures are the range the value takes when the seasons are redrawn at random, which shows how much it depends on which seasons happen to be on record."),
    ("cycle", "Seasonal cycle removed", "The rivers rise and fall at about the same time each season, and that shared timing counts towards the same-day value. Here the usual value for each calendar day is taken out of both series first, so what is left is whether the model is higher than usual on the days when the gauge is higher than usual."),
    ("shift", "Shifted to the closest match", "The model's curve is set against the gauge's curve of 10 days earlier up to 30 days later, and the highest value is kept. The small figure is that shift in days: plus means the model's curve runs earlier than the gauge's, minus later."),
    ("flood", "Seasons at the 1-in-3 level", "The same shifted comparison, using only seasons in which the gauge or the model was at or over its own 1-in-3 level. The small figure is the number of seasons used."),
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
    return out


def sibling_note(T):
    """Where the correlation on seasons at the 1-in-3 level differs from the page 'Google's new river model'."""
    out = ""
    sib = OUT.parent / "google-model-2026" / "data.json"
    if sib.exists():
        c = pd.DataFrame(json.loads(sib.read_text(encoding="utf-8"))["correlations"])
        j = c.merge(T, on=["season", "station", "model"])
        if len(j) == len(T):
            d = (j.rho_flood_seasons - j.rho)
            big = j[d.abs() > 0.05]
            out += (f" The page <a href='../google-model-2026/'>Google's new river model</a> shows the rank correlation on seasons at the 1-in-3 level with the gauge's"
                    f" season days only, so at long shifts the last days of the season drop out; here every season day of the model is used at every shift."
                    f" The two agree within 0.05 at {int((d.abs() <= 0.05).sum())} of the {len(j)} values.")
            if len(big):
                which = "; ".join(f"{SHORT[r.model]} at {NAME[r.station]} in {r.season.title()}: {r.rho:.2f} there, {r.rho_flood_seasons:.2f} here, at a shift of {int(r.shift_flood_seasons)} days"
                                  for r in big.sort_values("rho_flood_seasons").itertuples())
                out += f" The others: {which}."
    return out


def after_line(Rrec, season, y):
    """One line for a season after 2023, new model, 1-in-3 level."""
    def where(basis):
        rs = [r.title() for r in G.RIVERS if y in Rrec[(r, season)]["models"][NEW][basis]]
        return "both rivers" if len(rs) == 2 else (f"the {rs[0]}" if rs else "neither river")
    ra, fc = where("ra"), where("fc")
    say = lambda x: "threshold not reached" if x == "neither river" else f"threshold reached on {x}"
    reached = f"threshold reached on {ra}, on the reanalysis and on the forecasts" if ra == fc else f"reanalysis, {say(ra)}; forecasts, {say(fc)}"
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
    diff = []
    for season in ("deyr", "gu"):
        for river in G.RIVERS:
            ra = M["river_dates_3"][(season, river, NEW, "reanalysis", 3)]
            fc = M["river_dates_3"][(season, river, NEW, "forecasts", 3)]
            for y in range(R.FC0, R.FC1 + 1):
                tot += 1
                same += (y in ra) == (y in fc)
                if (y in ra) != (y in fc):
                    diff.append(f"the {river.title()} in {season.title()} {y}, reached on the {'forecasts' if y in fc else 'reanalysis'} only")
    def agree_of(m):
        s_ = t_ = 0
        for season in ("deyr", "gu"):
            for river in G.RIVERS:
                ra_, fc_ = M["river_dates_3"][(season, river, m, "reanalysis", 3)], M["river_dates_3"][(season, river, m, "forecasts", 3)]
                for y in sorted(R.fc_years(m, season) & set(range(R.FC0, R.FC1 + 1))):
                    t_ += 1
                    s_ += (y in ra_) == (y in fc_)
        return f"{s_} of {t_}"
    d7 = LT[(LT.model == NEW) & (LT.lead == 7)]
    v7 = LT[(LT.model == "glofas_v4") & (LT.lead == 7)]
    s7 = BL[(BL.model == NEW) & (BL.lead == 7)]
    w7 = BL[(BL.model == "glofas_v4") & (BL.lead == 7)]
    c7, k7 = LT[(LT.model == "google_grrr") & (LT.lead == 7)], BL[(BL.model == "google_grrr") & (BL.lead == 7)]
    out.append(f"<strong>Forecasts.</strong> At the 1-in-3 level, 2016 to 2023, the new model's forecasts and its reanalysis agree on threshold reached or not reached "
               f"in {same} of the {tot} combinations of river, season and year" + (f" (the exception: {'; '.join(diff)})" if diff else "") + f". Google, current: {agree_of('google_grrr')}. GloFAS v4: {agree_of('glofas_v4')} (section 5). "
               f"On forecast day 7, six days after the day of issue, on the days of the largest rises in its own reanalysis, the forecast already showed {100 * d7.rise.min():.0f} to {100 * d7.rise.max():.0f}% of the rise "
               f"(middle value over the stations of each river and season; single stations {100 * s7.rise_share_median.min():.0f} to {100 * s7.rise_share_median.max():.0f}%). "
               f"Google, current: {100 * c7.rise.min():.0f} to {100 * c7.rise.max():.0f}% (single stations {100 * k7.rise_share_median.min():.0f} to {100 * k7.rise_share_median.max():.0f}%). "
               f"GloFAS v4: {100 * v7.rise.min():.0f} to {100 * v7.rise.max():.0f}% (single stations {100 * w7.rise_share_median.min():.0f} to {100 * w7.rise_share_median.max():.0f}%); below 0 the forecast showed a fall. "
               f"Whether the new model's archived forecasts were computed from the weather forecasts available at the time is not confirmed (section 5).")

    def nt(river, season, rec):
        r = R3[(R3.model == NEW) & (R3.river == river) & (R3.season == season) & (R3.record == rec)].iloc[0]
        return notice(r.lead_days_median)
    out.append("<strong>Days between the threshold being reached and the second gauge reaching its 1-in-3 level</strong> (middle value over the river floods with the threshold reached, 2016 to 2023). "
               + " ".join(f"{r.title()} {s.title()}: forecasts {nt(r, s, 'forecasts')}, reanalysis {nt(r, s, 'reanalysis, forecast seasons')}." for s in ("deyr", "gu") for r in G.RIVERS))

    out.append("<strong>After 2023, new model only, 1-in-3 level.</strong> " + " ".join(after_line(Rrec, s, y) for y in (2024, 2025) for s in ("gu", "deyr"))
               + " On the Juba these two years rest on the Dollow and Luuq gauges only.")

    parts = []
    for season in ("deyr", "gu"):
        d = E3[(E3.model == NEW) & (E3.season == season)]
        parts.append(f"{season.title()}: threshold reached in {int(d.hits.sum())} of {int(d.n_floods.sum())} flood seasons, with {int(d.false_alarms.sum())} false alarms. "
                     f"With each level fitted without the season scored: {int(d.hits_held_out.sum())} of {int(d.n_floods_held_out.sum())}, with {int(d.false_alarms_held_out.sum())}.")
    out.append("<strong>Single stations, 1-in-3 level, new model, the seven stations added up.</strong> " + " ".join(parts)
               + " The gauge levels are refitted too, so the number of flood seasons changes.")

    jo = E3[E3.station == "jowhar"]
    gj = M["gauges"][M["gauges"].station == "jowhar"].set_index("season")
    out.append(f"<strong>Jowhar.</strong> At the 1-in-3 level every source reaches its threshold in fewer than half of the flood seasons there, in Deyr and in Gu (POD {jo.POD.min():.2f} to {jo.POD.max():.2f}). "
               f"The Jowhar gauge sat at one top reading for five days or more in {int(gj.loc['deyr', 'n_flat_top'])} of its {int(gj.loc['deyr', 'n_seasons'])} Deyr seasons.")

    piv = T.pivot_table(index=["season", "station"], columns="model", values="rho_same_day")
    pv2 = T.pivot_table(index=["season", "station"], columns="model", values="rho_cycle_removed")
    clear = {m: int((T[T.model == m].new_minus_this_lo > 0).sum()) for m in MODELS[1:]}
    out.append(f"<strong>Day-to-day agreement with the gauges</strong> (rank correlation: 0 is no relation, 1 is the same order of days). The new model has the highest value of the four sources at "
               f"{int((piv.idxmax(axis=1) == NEW).sum())} of the {len(piv)} combinations of station and season ({piv[NEW].min():.2f} to {piv[NEW].max():.2f}), and at "
               f"{int((pv2.idxmax(axis=1) == NEW).sum())} with the seasonal cycle removed ({pv2[NEW].min():.2f} to {pv2[NEW].max():.2f}). "
               f"When the value is recomputed on random picks of the seasons (section 13), the new model's value stays above the other source's in at least 95% of the picks at "
               f"{clear['google_grrr']} of {len(piv)} against Google, current, {clear['glofas_v4']} against GloFAS v4 and {clear['glofas_v5']} against GloFAS v5.")

    a3 = E3.pivot_table(index=["season", "station"], columns="model", values="AUC")
    rp_ = E3.pivot_table(index=["season", "station"], columns="model", values="rho_season_peaks")
    top = lambda p_: int((p_[NEW] >= p_.max(axis=1) - 1e-9).sum())

    def near(p_):
        """Where the new model is not the highest but shows the same two decimals as the highest."""
        w = [f"{NAME[st]} in {se.title()}" for (se, st) in p_.index
             if p_.loc[(se, st), NEW] < p_.loc[(se, st)].max() - 1e-9 and round(p_.loc[(se, st), NEW], 2) == round(p_.loc[(se, st)].max(), 2)]
        return f" (counted before rounding: at {' and at '.join(w)} it shows the same two decimals as the highest and is not counted)" if w else ""
    last = [f"{NAME[st]} in {se.title()} ({a3.loc[(se, st), NEW]:.2f}; the others {a3.loc[(se, st)].drop(NEW).min():.2f} to {a3.loc[(se, st)].drop(NEW).max():.2f})"
            for (se, st) in a3.index if a3.loc[(se, st), NEW] < a3.loc[(se, st)].drop(NEW).min() - 1e-9]
    out.append(f"<strong>Which seasons are the large ones, 1-in-3 level.</strong> AUC is the chance that a flood season has a higher model peak than a season without a flood. "
               f"The new model has the highest, or equal highest, AUC at {top(a3)} of {len(a3)} combinations of station and season{near(a3)}, and the highest, or equal highest, rank correlation of season peaks at {top(rp_)}{near(rp_)}."
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
                ts_figs.append(sw("ts", {"year": key, "lines": variant}, fig_html(src, f"{season.title()} {y}: gauge and {'Google, new' if variant == 'new' else 'the four sources'} at the seven stations, as % of each line's own 1-in-3 level")))

    # ---- the trigger rule by river
    s1_figs = "".join(fig_html(fig_new_model(M, Rv, rp), f"Google, new: the trigger rule at the 1-in-{rp} level, season by season, reanalysis and forecasts") for rp in (3, 5))
    grid_blocks = "".join(sw("rv", {"level": rp}, fig_html(fig_outcomes(M, Rv, rp, s), f"{s.title()}, the trigger rule season by season, 1-in-{rp} level"))
                          for rp in (3, 5) for s in ("deyr", "gu"))
    rv_blocks = "".join(sw("rv", {"metric": key, "level": rp}, fig_html(fig_rv_score(Rv, rp, key, title), f"{lab} by river, season, source and record, 1-in-{rp} level"))
                        for rp in (3, 5) for key, lab, title, _ in RV_METRICS)
    rv_text = "".join(sw("rv", {"metric": key}, text, tag="p") for key, _, _, text in RV_METRICS)
    ho_blocks = "".join(sw("rv", {"level": rp}, fig_html(fig_held_out(Rv, rp), f"Counts with levels fitted on all seasons and with levels held out, 1-in-{rp} level")) for rp in (3, 5))

    # ---- forecasts by days ahead
    F["lead_rise"] = fig_lead(LT.assign(rise_pct=100 * LT.rise), "rise_pct", "lead_rise", "Share of the coming rise already in the forecast, % (middle value over the stations of the river)", (-20, 140), ref=100.0, ref_label="Dotted line: 100%, the whole rise.")
    F["lead_peak"] = fig_lead(LT, "peak", "lead_peak", "Forecast peak divided by reanalysis peak (middle value over the stations of the river)", (0.8, 1.25), ref=1.0, ref_label="Dotted line: 1, the same peak as the reanalysis.")
    F["lead_rho"] = fig_lead(LT, "rho", "lead_rho", "Rank correlation of the forecast with the gauge (average over the stations of the river)", (0, 1.0))
    F["lead_pod"] = fig_lead(LT, "POD", "lead_pod", "POD: share of flood seasons with the threshold reached, single stations at the 1-in-3 level, added up by river", (0, 1.0))
    F["lead_far"] = fig_lead(LT, "FAR", "lead_far", "FAR: share of threshold seasons with no flood, single stations at the 1-in-3 level, added up by river", (0, 1.0))
    LD_MORE = [k for k in ("POFD", "CSI", "F1", "freq_bias", "accuracy", "HSS", "PSS")]
    for key in LD_MORE:
        v_ = LT[key].dropna()
        if key == "freq_bias":
            lim, ref, rl = (0, max(2.0, float(np.ceil(v_.max() * 2) / 2))), 1.0, "Dotted line: 1, the threshold reached as often as the gauge floods."
        elif key in ("HSS", "PSS"):
            lim, ref, rl = (min(-0.25, float(np.floor(v_.min() * 4) / 4)), 1.0), 0.0, "Dotted line: 0."
        else:
            lim, ref, rl = (0, 1.0), None, ""
        F[f"lead_{key}"] = fig_lead(LT, key, f"lead_{key}", f"{LONG[key]}: single stations at the 1-in-3 level, added up by river", lim, ref=ref, ref_label=rl)
    d7 = BL[(BL.model == NEW) & (BL.lead == 7)]
    B7 = BL[BL.lead == 7].copy()
    B7 = add_scores(B7)
    fc_models = [m for m in MODELS if m in G.FC_SPAN]
    st7_note = "Forecast day 7, six days after the day of issue. Seasons 2016 to 2023 (Google, current: Deyr to 2022). GloFAS v5 has no forecast archive. "
    st7 = lambda col, nm, title, kind, note, **kw: grid_fig(B7, col, nm, title, kind, note=st7_note + note, models=fc_models, **kw)
    F["st7_rise"] = st7("rise_share_median", "st7_rise", "Share of the coming rise already in the forecast on forecast day 7, by station", "good", "Dark = high; the shading is full at 100%. Small figures: the days of large rises the value rests on.", fmt=lambda v: f"{100 * v:.0f}%", sub=lambda r: f"{int(r.n_rise_days)} days")
    F["st7_peak"] = st7("peak_ratio_median", "st7_peak", "Forecast peak divided by reanalysis peak on forecast day 7, by station (middle value over the seasons)", "ratio", "Teal: above 1. Brown: below 1; the shading is full at 2 and at 0.5.")
    F["st7_rho"] = st7("rho_vs_gauge", "st7_rho", "Rank correlation of the forecast for forecast day 7 with the gauge, by station", "corr", "Dark = high; the shading runs from 0.4 to 1.")
    F["st7_pod"] = st7("POD", "st7_pod", "POD on forecast day 7, by station. Flood season = the station's gauge at or over its 1-in-3 level", "good",
                       "Dark = high. Small figures: flood seasons with the threshold reached, of flood seasons at the gauge.", sub=lambda r: f"{int(r.hits)} of {int(r.hits + r.misses)}")
    F["st7_far"] = st7("FAR", "st7_far", "FAR on forecast day 7, by station. Flood season = the station's gauge at or over its 1-in-3 level", "bad",
                       "Dark = high. Small figures: seasons with no flood, of seasons with the threshold reached.", sub=lambda r: f"{int(r.false_alarms)} of {int(r.hits + r.false_alarms)}")
    kinds = {k: (kind, fmt) for k, _, kind, fmt, _ in METRICS}
    ld_more = ""
    for key in LD_MORE:
        kind, fmt = kinds[key]
        nt = {"good": "Dark = high.", "bad": "Dark = high.", "eff": "Green: above 0. Brown: below 0.", "ratio": "Teal: above 1. Brown: below 1; the shading is full at 2 and at 0.5."}[kind]
        F[f"st7_{key}"] = st7(key, f"st7_{key}", f"{LONG[key]} on forecast day 7, by station. Flood season = the station's gauge at or over its 1-in-3 level", kind,
                             nt + " Small figures: flood seasons with the threshold reached / without / false alarms / neither.", fmt=fmt,
                             sub=lambda r: f"{int(r.hits)} / {int(r.misses)} / {int(r.false_alarms)} / {int(r.correct_negatives)}")
        ld_more += sw("ld", {"metric": key}, f"<p>{LONG[key]}, at single stations at the 1-in-3 level, on the forecast for each forecast day, with the counts of the stations of a river added up. Section 7 explains the score.</p>"
                      + fig_html(F[f"lead_{key}"], f"{LONG[key]} by forecast day") + fig_html(F[f"st7_{key}"], f"{LONG[key]} on forecast day 7, by station"))
    # ---- forecasts against their own reanalysis
    RA7 = add_scores(B7[["ra_hits", "ra_false_alarms", "ra_misses", "ra_correct_negatives"]].copy(), "ra_hits", "ra_false_alarms", "ra_misses", "ra_correct_negatives")
    for k in ("POD", "FAR", "POFD", "CSI", "F1", "HSS"):
        B7["ra_" + k] = RA7[k]
    B7["rho_g"], B7["ra_rho_g"] = B7.rho_vs_gauge, B7.rho_ra_vs_gauge
    tot7 = B7.fr_both + B7.fr_forecast_only + B7.fr_reanalysis_only + B7.fr_neither
    B7["fr_agree"] = ((B7.fr_both + B7.fr_neither) / tot7).where(tot7 > 0)
    wide = lambda v, lo0, hi0: (min(lo0, float(np.floor(v.min() * 10) / 10)), max(hi0, float(np.ceil(v.max() * 10) / 10)))
    mid = " (middle value over the stations of the river)"
    FR = [  # key, button, river column, station column, title, limits, reference line, grid colours, grid format, explanation
        ("rho", "Rank correlation", "rho_ra", "rho_vs_reanalysis", "Rank correlation of the forecast with the same model's reanalysis", (0.9, 1.0), None, "corr", "{:.3f}",
         "Rank correlation of the forecast for each forecast day with the model's reanalysis on the day the forecast is for. 1 means the days fall in the same order."),
        ("ratio", "Flow on a typical day", "fr_ratio", "fr_ratio_median", "Forecast flow on a typical day divided by reanalysis flow", (0.8, 1.2), 1.0, "ratio", "{:.2f}",
         "The forecast's flow on a typical day (the middle day when all days are ranked by flow) divided by the reanalysis's on the same days. 1 means equal."),
        ("var", "Variability ratio", "fr_var", "fr_variability_ratio", "Size of the forecast's swings divided by the size of the reanalysis's", (0.8, 1.2), 1.0, "ratio", "{:.2f}",
         "Variability ratio: the size of the forecast's swings divided by the size of the reanalysis's on the same days. 1 means the same spread."),
        ("KGE", "KGE", "fr_KGE", "fr_KGE", "KGE of the forecast against the same model's reanalysis", (0.0, 1.0), None, "eff", "{:.2f}",
         "KGE, Kling-Gupta efficiency of the forecast against the reanalysis: one score built from the correlation, the variability ratio and the ratio of averages. 1 means the forecast matches the reanalysis in all three."),
        ("NSE", "NSE", "fr_NSE", "fr_NSE", "NSE of the forecast against the same model's reanalysis", (0.0, 1.0), None, "eff", "{:.2f}",
         "NSE, Nash-Sutcliffe efficiency of the forecast against the reanalysis: 1 means the forecast matches the reanalysis; 0 means it is no closer to it than a flat line at the reanalysis's average."),
        ("agree", "Threshold agreement", "fr_agree", "fr_agree", "Share of seasons in which forecast and reanalysis agree on whether the station's 1-in-3 threshold is reached", (0.0, 1.0), None, "good", "{:.2f}",
         "For each forecast day, the seasons in which the forecast and the reanalysis agree on the station's 1-in-3 threshold, reached or not reached, as a share of the seasons; the counts of the stations of a river are added up."),
    ]
    fr_blocks = ""
    for key, lab, lcol, scol, title, lim0, ref, kind, fmt, text in FR:
        lim = wide(LT[lcol].dropna(), *lim0)
        a_ = fig_lead(LT, lcol, f"fr_lead_{key}", title + ("" if key == "agree" else " (average over the stations of the river)" if key == "rho" else mid), lim,
                      ref=ref, ref_label="Dotted line: 1, the forecast equal to the reanalysis." if ref else "")
        nt = {"corr": "Dark = high; the shading runs from 0.4 to 1.", "ratio": "Teal: above 1. Brown: below 1; the shading is full at 2 and at 0.5.",
              "eff": "Green: above 0. Brown: below 0.", "good": "Dark = high."}[kind]
        kw = dict(sub=lambda r: f"{int(r.fr_both)} / {int(r.fr_forecast_only)} / {int(r.fr_reanalysis_only)} / {int(r.fr_neither)}") if key == "agree" else {}
        if key == "agree":
            nt += " Small figures: seasons with the threshold reached on both / on the forecast only / on the reanalysis only / on neither."
        b_ = st7(scol, f"fr_st7_{key}", title + " on forecast day 7, by station", kind, nt, fmt=fmt, **kw)
        fr_blocks += sw("fr", {"metric": key}, f"<p>{text}</p>" + fig_html(a_, f"{lab}: forecast against reanalysis by forecast day") + fig_html(b_, f"{lab}: forecast against reanalysis on forecast day 7, by station"))
    fr_opts = [(k, lab) for k, lab, *_ in FR[:2]] + [("peak", "Forecast peak / reanalysis peak"), ("rise", "Share of the coming rise")] + [(k, lab) for k, lab, *_ in FR[2:]]
    FG = [("rho_g", "Rank correlation with the gauge", "Rank correlation with the gauge"), ("POD", "POD", LONG["POD"]), ("FAR", "FAR", LONG["FAR"]), ("POFD", "POFD", LONG["POFD"]),
          ("CSI", "CSI", LONG["CSI"]), ("F1", "F1", LONG["F1"]), ("HSS", "HSS", LONG["HSS"])]
    fg_note = "Seasons 2016 to 2023 (Google, current: Deyr to 2022). The reanalysis is read on the same seasons and days as the forecast. Where the dot cannot be seen it lies inside the diamond: the two values are equal."
    fg_blocks = "".join(sw("fg", {"metric": k}, fig_html(fig_fc_vs_ra(B7, k, f"fg_{k}", f"{long_}: the reanalysis and the forecast for forecast day 7, each against the gauge, by station"
                                                                      + ("" if k == "rho_g" else ". 1-in-3 level"), fg_note), f"{lab}: reanalysis and forecast day 7 against the gauge, by station"))
                        for k, lab, long_ in FG)
    fx_blocks = "".join(sw("fx", {"level": rp}, fig_html(fig_rule_fc_vs_ra(M, rp), f"The trigger rule on the forecasts beside the reanalysis, season by season, 1-in-{rp} level")) for rp in (3, 5))
    N7_TXT = f"{int(B7.n_seasons.min())} to {int(B7.n_seasons.max())}" if B7.n_seasons.min() != B7.n_seasons.max() else f"{int(B7.n_seasons.min())}"
    ld_opts = [("rho", "Rank correlation with the gauge"), ("pod", "POD"), ("far", "FAR")] + [
        (k, dict((a, b) for a, b, _, _, _ in METRICS)[k]) for k in LD_MORE]

    # ---- timing at stations
    ahead_note = "Reanalysis, seasons with a gauge record, 1999 to 2023. Plus: the model earlier than the gauge. Minus: later. Teal = earlier, brown = later; the shading is full at 15 days."
    F["pk"] = grid_fig(T, "peak_days_ahead_median", "peak_days", "Days from the model's peak to the gauge's peak, flood seasons at the gauge (middle value)", "days", fmt=signed,
                       sub=lambda r: f"{int(r.n_peaks)} seasons", note=ahead_note + " Small figure: seasons used.")
    F["cx"] = grid_fig(T, "crossing_days_ahead_median", "crossing_days", "Days from the model reaching its 1-in-3 level to the gauge reaching its own (middle value)", "days", fmt=signed,
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
            note = {"good": "Dark = high.", "bad": "Dark = high.", "eff": "Green: above 0. Brown: below 0.", "ratio": "Teal: above 1. Brown: below 1; the shading is full at 2 and at 0.5."}[kind]
            note = "Reanalysis, seasons with a gauge record, 1999 to 2023. " + note
            note = {"F1": "1 = threshold reached in every flood season, no false alarm. ", "HSS": "1 = model and gauge agree in every season; 0 = no more than chance. ",
                    "PSS": "1 = flood seasons fully separated from the rest; 0 = not separated. ",
                    "freq_bias": "1 = the model reaches its threshold as often as the gauge floods. "}.get(key, "") + note
            note += {"POD": " Small figures: flood seasons with the threshold reached, of flood seasons at the gauge.",
                     "FAR": " Small figures: seasons with no flood, of seasons with the threshold reached.",
                     "POFD": " Small figures: seasons with the threshold reached, of seasons with no flood.",
                     "CSI": " Small figures: flood seasons with the threshold reached / without / false alarms.",
                     "F1": " Small figures: flood seasons with the threshold reached / without / false alarms.",
                     "accuracy": " Small figures: seasons in which model and gauge agree, of all seasons."}.get(key, "")
            src = grid_fig(E, key, f"st_{key}_{rp}", f"{LONG[key]}. Flood season = the station's gauge at or over its 1-in-{rp} level", kind, fmt=fmt, sub=sub, note=note)
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
        "shift": grid_fig(T, "rho_best_shift", "rho_best_shift", "Rank correlation of daily values with the model's curve shifted to where the value is highest (gauge 10 days earlier to 30 days later)", "corr",
                          sub=lambda r: f"{int(r.best_shift_days):+d} d", note=span_note + " Small figure: the shift in days; plus means the model's curve runs earlier than the gauge's."),
        "flood": grid_fig(T, "rho_flood_seasons", "rho_flood_seasons", "Rank correlation of daily values at the closest shift, seasons with the gauge or the model at or over its own 1-in-3 level", "corr",
                          sub=lambda r: f"{int(r.n_flood_seasons)} seasons", note="Seasons with a gauge record, 1999 to 2023. Dark = high; the shading runs from 0.4 to 1. Small figure: seasons used, which differ by source because each source reaches its level in its own seasons."),
    }
    dg_extra = {"flood": flood_view_note(T)}
    dg_blocks = "".join(sw("dg", {"view": k}, f"<p>{text}{dg_extra.get(k, '')}</p>" + fig_html(dg[k], lab)) for k, lab, text in D_METRICS)

    # ---- ordering of seasons
    F["rmax"] = grid_fig(E3, "rho_season_peaks", "rho_season_peaks", "Rank correlation of season peaks, model against gauge", "good", sub=lambda r: f"{int(r.n)} seasons",
                         note="Seasons with a gauge record, 1999 to 2023. Dark = high; the shading runs from 0 to 1. Small figure: seasons used.")
    auc_note = "Seasons with a gauge record, 1999 to 2023. 0.5 is a coin toss. Dark = high; the shading runs from 0.5 to 1."
    F["auc3"] = grid_fig(E3, "AUC", "auc_3", "AUC: chance that a flood season has a higher model peak than a season without a flood. Flood = gauge at or over its 1-in-3 level", "auc", note=auc_note)
    F["auc5"] = grid_fig(E5, "AUC", "auc_5", "AUC: chance that a flood season has a higher model peak than a season without a flood. Flood = gauge at or over its 1-in-5 level", "auc", note=auc_note)

    # ---- size of the flows
    q_rows = [s for s in ("luuq", "bardheere", "belet_weyne", "bulo_burti") if s in set(Q.station)]
    q_panels = [("deyr", SEASON_TITLE["deyr"]), ("gu", SEASON_TITLE["gu"]), ("year", "Whole year")]
    q_blocks = []
    q_y0, q_y1 = pd.to_datetime(Q.first_day).min().year, pd.to_datetime(Q.last_day).max().year
    for key, lab, kind, fmt, _ in Q_METRICS:
        note = {"good": "Dark = high.", "eff": "Green: above 0. Brown: below 0; -1 and lower take the darkest brown.", "ratio": "Teal: above 1. Brown: below 1; the shading is full at 2 and at 0.5."}[kind]
        note = {"KGE": "1 = the model matches SWALIM's series. ", "NSE": "1 = the model matches SWALIM's series; 0 = no closer than SWALIM's average. ",
                "pearson_r": "Plain correlation of the daily values, not the rank correlation of section 8. The shading runs from 0.4 to 1. "}.get(key, "") + note
        if "bardheere" in q_rows:
            note += " * SWALIM's Bardheere series does not agree with its Luuq series (see the text)."
        src = grid_fig(Q, key, f"q_{key}", f"{LONG.get(key, lab)}: model flow against SWALIM's discharge series, on the days of that series, {q_y0} to {q_y1}", "corr" if key == "pearson_r" else kind, fmt=fmt,
                       note=note, rows=q_rows, panels=q_panels, row_labels={"bardheere": "Bardheere *"})
        q_blocks.append(sw("q", {"metric": key}, fig_html(src, f"{lab} against SWALIM discharge")))
    q_text = "".join(sw("q", {"metric": key}, text, tag="p") for key, _, _, _, text in Q_METRICS)
    ceil = [f"{NAME[st]} on {QN[st]['days_at_top']} days" for st in q_rows if QN[st]["days_at_top"] >= 10]
    CEIL_TXT = ("The series repeats its highest value at " + " and at ".join(ceil) + ", so floods above that value are not told apart. " if ceil else "")
    bl = QN.get("bardheere_over_luuq")
    BARD_TXT = (f"At Bardheere, SWALIM's typical flow is {bl['swalim']:.1f} times its typical flow at Luuq, the next station upstream, while every model has about the same flow at the two "
                f"({min(bl[m] for m in MODELS):.1f} to {max(bl[m] for m in MODELS):.1f} times). The Bardheere column is marked with an asterisk and is left out of the ranges quoted on this page." if bl else "")

    # ---- gauges
    flat = GN[GN.n_flat_top >= 3].sort_values("n_flat_top", ascending=False)
    same = GN[(GN.n_common_peak >= 2) & ((GN.level_1in3 - GN.common_peak).abs() < 0.005)]
    SAME_TXT = (" The column at the right gives the season peak that occurs most often. At " + "; ".join(
        f"{NAME[r.station]} in {r.season.title()} ({r.common_peak:.2f} m in {int(r.n_common_peak)} seasons)" for r in same.itertuples())
        + " the 1-in-3 level is that repeated peak, and a value equal to its level counts as reaching it.") if len(same) else " The column at the right gives the season peak that occurs most often."
    FLAT_TXT = "; ".join(f"{NAME[r.station]} in {r.season.title()} ({int(r.n_flat_top)} of {int(r.n_seasons)} seasons)" for r in flat.itertuples())

    # ---- published files, with plain column names
    PUB = {"hits": "floods_reached", "misses": "floods_not_reached", "correct_negatives": "neither", "hit_years": "years_flood_reached",
           "false_years": "years_false_alarm", "missed_years": "years_flood_not_reached", "hits_held_out": "floods_reached_held_out",
           "misses_held_out": "floods_not_reached_held_out", "lead": "forecast_day", "lead_days_median": "days_before_second_gauge_median",
           "ra_hits": "reanalysis_floods_reached", "ra_false_alarms": "reanalysis_false_alarms", "ra_misses": "reanalysis_floods_not_reached",
           "ra_correct_negatives": "reanalysis_neither", "rho_ra_vs_gauge": "rho_reanalysis_vs_gauge",
           "fr_ratio_median": "forecast_over_reanalysis_typical_day", "fr_pearson_r": "forecast_reanalysis_pearson_r",
           "fr_variability_ratio": "forecast_reanalysis_variability_ratio", "fr_KGE": "forecast_reanalysis_KGE", "fr_NSE": "forecast_reanalysis_NSE",
           "fr_n_days": "forecast_reanalysis_days", "fr_both": "threshold_forecast_and_reanalysis", "fr_forecast_only": "threshold_forecast_only",
           "fr_reanalysis_only": "threshold_reanalysis_only", "fr_neither": "threshold_neither"}

    def recs(df):
        return json.loads(df.rename(columns=PUB).to_json(orient="records"))
    tables = (("daily_agreement", T), ("station_floods_1in3", E3), ("station_floods_1in5", E5), ("river_rule", Rv), ("forecasts_by_forecast_day", BL),
              ("discharge", Q), ("gauges", GN))
    js = json.dumps(G.clean(dict(
        sources={m: SHORT[m] for m in MODELS}, largest_seasons=M["worst"], discharge_series_notes=QN,
        **{nm: recs(df) for nm, df in tables})), indent=1)
    for m in MODELS:
        js = js.replace(f'"{m}"', json.dumps(SHORT[m]))
    (OUT / "data.json").write_text(js)
    for old in OUT.glob("*.csv"):
        old.unlink()
    for nm, df in tables:
        d = df.assign(model=df.model.map(SHORT)) if "model" in df.columns else df
        d.rename(columns=PUB).to_csv(OUT / f"{nm}.csv", index=False)

    today = dt.date.today().isoformat()
    FIND = findings(M, LT, Rrec)
    SIB_TXT = sibling_note(T)
    ran = Rv[(Rv.rp == 3) & (Rv.record == "reanalysis")]
    NSEAS_TXT = f"{int(ran.n.min())} to {int(ran.n.max())}" if ran.n.min() != ran.n.max() else f"{int(ran.n.min())}"
    acc = lambda E: f"{((E.n - E.n_floods) / E.n).min():.2f} to {((E.n - E.n_floods) / E.n).max():.2f}"
    ev_text = ev_text.replace("ACC_RANGE", f"{acc(E3)} at the 1-in-3 level and {acc(E5)} at the 1-in-5")
    fcn = Rv[(Rv.rp == 3) & (Rv.record == "forecasts")]
    short = fcn[fcn.n < R.FC1 - R.FC0 + 1]
    SHORT_TXT = (" (" + "; ".join(f"{int(r.n)} in {r.season.title()} for {SHORT[r.model]}" for r in short.drop_duplicates(["season", "model"]).itertuples()) + ")") if len(short) else ""
    NFL_TXT = f"{int(fcn.n_floods.min())} to {int(fcn.n_floods.max())}"
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
         weather on record, 1999 to 2023 here (the new model's also covers 2024 and 2025, used only where stated).
         Three have an archive of the forecasts they would have issued: Google, new 2016 to 2025, Google, current 2016 to June 2023, GloFAS v4 2003 to 2023.
         Forecasts are compared on 2016 to 2023.</p>
      <p><strong>Benchmark.</strong> The river gauges of SWALIM (Somalia Water and Land Information Management) at the same seven stations. A gauge
         measures water height and a model gives flow, so each is read against its own levels. The 1-in-3 level of a gauge or a model is the value its
         season peak reaches in about one season in three; the 1-in-5 level, one in five. A flood season at a station is a season in which the gauge is
         at or over its 1-in-3 level (or the 1-in-5 where stated). A river flood is a season in which two gauges on the river are at or over their own level, at any time in the
         season. Gauge levels are set on 2000 to 2023, model levels on 1999 to 2023; seasons are scored from 1999 where the gauges have a record. The seasons are Deyr, October to December, and Gu, March to May.</p>
      <p><strong>Threshold reached.</strong> At a station: the model's flow is at or over the model's own level. On a river, which is the trigger rule:
         three of the four Juba stations, or two of the three Shabelle stations, are at or over their level on the same day. Forecasts are read against
         the level fitted on the same model's reanalysis.</p>
    </div>

    <h2><span class="num">1</span>What the records show</h2>
    <p>The trigger rule on the new model, one square per season: its reanalysis, and from 2016 its forecasts (the first forecast of the season that meets the
       rule, forecast days 1 to 7). First at the 1-in-3 level, then at the 1-in-5. Section 3 has the same for every source.</p>
    {s1_figs}
    <ul class="takeaways">
{FIND}
    </ul>

    <h2><span class="num">2</span>The largest floods, day by day</h2>
    <p>The seasons that rank among the gauges' five largest on either river, one panel per station. Every line is drawn as a share of its own 1-in-3 level, so 100% is the
       flood level for the gauge and for the model alike. The heights of the curves differ; what can be
       compared is when each rises and whether each reaches its 100% line. From 2016 the first view also draws the new model's forecast for forecast
       day 7. The other three sources end in 2023, so the 2024 seasons show the new model only. Where a gauge line runs flat at its top, the gauge gave the same reading on
       those days (section 11). A line that stops before the end of the season is the end of that record. The vertical scale changes from one season to
       the next.</p>
    <p class="picklabel">Season</p>
    {pick("ts", "year", ts_opts)}
    <p class="picklabel">Lines</p>
    {pick("ts", "lines", [("new", "Gauge and the new model"), ("all", "Gauge and all four sources")])}
    {"".join(ts_figs)}

    <h2><span class="num">3</span>The trigger rule by river</h2>
    <p>The trigger rule against the river floods, as defined at the top of the page. The first charts show every season, with a reanalysis
       row for each source and a forecast row for the three that have a forecast archive. For the forecast rows the rule is checked on each day's
       forecast, forecast days 1 to 7, and the first forecast of the season that meets it is taken.</p>
    <p class="picklabel">Level</p>
    {pick("rv", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {grid_blocks}
    <h3>The scores</h3>
    <p>Each score reads the counts above in a different way, for every source and record. The rows marked 1999 to 2023 hold the seasons with a gauge
       record, {NSEAS_TXT} per river and season. Beside each forecast row for 2016 to 2023 is the reanalysis on those same years. The rows marked 2016 to
       2025 score the new model's forecasts over its whole archive; no other source has forecasts after 2023. On the Juba, 2024 and 2025 rest on the
       Dollow and Luuq gauges only.</p>
    <p class="picklabel">Score</p>
    {pick("rv", "metric", [(k, lab) for k, lab, _, _ in RV_METRICS])}
    {rv_text}
    {rv_blocks}
    <h3>With each level fitted without the season being scored</h3>
    <p>Above, each level is fitted on the same seasons it is then scored on. Here the level for each season, for the model and for
       the gauges, is fitted on all the other seasons, and the season is then scored.</p>
    {ho_blocks}

    <h2><span class="num">4</span>The forecasts against the gauges, by forecast day</h2>
    <p>Each forecast archive is read one step at a time. Forecast day 1 is the day the forecast is issued; forecast day 7 is six days later.
       The new model goes to forecast day 10, the others to 7. Seasons 2016 to 2023. GloFAS v4 runs several versions of each forecast; the middle one is used.
       Under each chart by river is the same measure at each station on forecast day 7. The scores are at the 1-in-3 level. Section 5 sets the
       forecasts against the reanalysis.</p>
    <p class="picklabel">Measure</p>
    {pick("ld", "metric", ld_opts)}
    {sw("ld", {"metric": "rho"}, "<p>Rank correlation of the forecast with the gauge on the day the forecast is for, averaged over the stations of the river. 1 means the days fall in the same order by forecast flow and by gauge height; 0 means no relation (section 8).</p>" + fig_html(F["lead_rho"], "Rank correlation with the gauge by forecast day") + fig_html(F["st7_rho"], "Rank correlation with the gauge on forecast day 7, by station"))}
    {sw("ld", {"metric": "pod"}, "<p>POD, the share of flood seasons with the threshold reached, at single stations at the 1-in-3 level, with the counts of the stations of a river added up.</p>" + fig_html(F["lead_pod"], "POD by forecast day") + fig_html(F["st7_pod"], "POD on forecast day 7, by station"))}
    {sw("ld", {"metric": "far"}, "<p>FAR, the share of seasons with the threshold reached that had no flood at the gauge, at single stations at the 1-in-3 level, with the counts of the stations of a river added up.</p>" + fig_html(F["lead_far"], "FAR by forecast day") + fig_html(F["st7_far"], "FAR on forecast day 7, by station"))}
    {ld_more}

    <h2><span class="num">5</span>The forecasts against the reanalysis</h2>
    <p>The levels are fitted on each model's reanalysis and the trigger runs on its forecasts. This section sets each forecast archive against its own
       model's reanalysis: day by day, then both against the gauges on the same seasons, then on the trigger rule. No gauge enters the first and the last.</p>
    <h3>Day by day</h3>
    <p>Seasons 2016 to 2023. Under each chart by river is the same measure at each station on forecast day 7.</p>
    <p class="picklabel">Measure</p>
    {pick("fr", "metric", fr_opts)}
    {fr_blocks}
    {sw("fr", {"metric": "peak"}, "<p>The forecast's highest value of the season divided by the reanalysis's highest value on the same days. At 1 the forecast carries the same peak as the reanalysis, so a level fitted on the reanalysis is reached as easily on the forecast; below 1, less easily; above 1, more easily.</p>" + fig_html(F["lead_peak"], "Forecast peak over reanalysis peak by forecast day") + fig_html(F["st7_peak"], "Forecast peak over reanalysis peak on forecast day 7, by station"))}
    {sw("fr", {"metric": "rise"}, "<p>For each forecast day, take the days with the largest rises in the model's own reanalysis over that many days (the top tenth). The measure is the rise the forecast showed over the same days, divided by the rise in the reanalysis. 100% means the forecast showed all of it; 0% means it still showed the river where it was; below 0%, it showed a fall; above 100%, more than the reanalysis. The measure sets a forecast against its own model's reanalysis, not against the gauge.</p>" + fig_html(F["lead_rise"], "Share of the coming rise already in the forecast, by forecast day") + fig_html(F["st7_rise"], "Share of the coming rise on forecast day 7, by station"))}
    <p class="note">On forecast day 7 the new model's daily forecast and its reanalysis have a rank correlation of {d7.rho_vs_reanalysis.min():.3f} to {d7.rho_vs_reanalysis.max():.3f}
       across the seven stations and two seasons. Google's description of its forecasting model (Cohen and others, 2026, a paper not yet peer reviewed) says the forecast
       days are computed from archived weather forecasts, from the European centre ECMWF and Google's GraphCast weather model. We have not confirmed that this holds for the November 2026 release.</p>
    <h3>Against the gauges: forecast day 7 beside the reanalysis</h3>
    <p>The scores of section 4 for the forecast for forecast day 7, beside the same scores for the reanalysis on the same seasons and days. A station
       has {N7_TXT} seasons here, so one season moves a score a long way.</p>
    <p class="picklabel">Score</p>
    {pick("fg", "metric", [(k, lab) for k, lab, _ in FG])}
    {fg_blocks}
    <h3>The trigger rule: the forecasts beside the reanalysis</h3>
    <p>The trigger rule on each model's forecasts beside the rule on its reanalysis, for every season in which the model has both.</p>
    <p class="picklabel">Level</p>
    {pick("fx", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {fx_blocks}

    <h2><span class="num">6</span>Timing at each station</h2>
    <p>Two measures in days, on the reanalysis, 1999 to 2023, over the flood seasons at each gauge: when the model peaks against when the gauge peaks, and
       when the model first reaches its 1-in-3 level against when the gauge first reaches its own. Plus means the model is earlier than the gauge. Where
       a gauge holds one value at its top for several days, its peak is dated to the first of those days; the true peak may fall on any of them. These two measures and the shift in
       section 8 use different seasons and different parts of the curve, and differ in sign at some stations.</p>
    {fig_html(F["pk"], "Days between model peak and gauge peak")}
    {fig_html(F["cx"], "Days between model and gauge reaching the 1-in-3")}

    <h2><span class="num">7</span>The threshold at each station against its gauge</h2>
    <p>Each season at a station falls into one of four boxes: a flood at the gauge with the model's threshold reached; a flood with the threshold not
       reached; the threshold reached with no flood, which is a false alarm; or neither. Every score below is a different way of reading those four
       counts. {SAMPLE_TXT}</p>
    <p class="picklabel">Score</p>
    {pick("ev", "metric", [(k, lab) for k, lab, _, _, _ in METRICS])}
    <p class="picklabel">Level</p>
    {pick("ev", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {ev_text}
    {"".join(ev_blocks)}

    <h2><span class="num">8</span>Day-to-day agreement with the gauges</h2>
    <p>Rank correlation puts the days in order by model flow and, separately, by gauge height, and measures how far the two orders agree: 1 means the
       days fall in exactly the same order, 0 means no relation. It uses the order of the days, not their values, so the scale of a model does not
       matter.</p>
    <p class="picklabel">View</p>
    {pick("dg", "view", [(k, lab) for k, lab, _ in D_METRICS])}
    {dg_blocks}

    <h2><span class="num">9</span>How each model orders the seasons</h2>
    <p>Two measures of whether a model's large seasons are the gauge's large seasons. The rank correlation of season peaks compares the order of the
       seasons by their highest model flow with their order by highest gauge level. AUC is the chance that a flood season picked at random carries a higher
       model peak than a season without a flood picked at random: 0.5 is a coin toss, 1 means every flood season has a higher model peak than every season without a flood.
       Neither depends on where the model's threshold is set.</p>
    {fig_html(F["rmax"], "Rank correlation of season peaks")}
    <p class="picklabel">Level</p>
    {pick("auc", "level", [("3", "1-in-3"), ("5", "1-in-5")])}
    {sw("auc", {"level": 3}, fig_html(F["auc3"], "AUC, 1-in-3"))}
    {sw("auc", {"level": 5}, fig_html(F["auc5"], "AUC, 1-in-5"))}

    <h2><span class="num">10</span>Size of the flows against SWALIM's discharge series</h2>
    <p>SWALIM publishes a discharge series, in cubic metres per second, at four of the seven stations. It is worked out from the gauge height: the two
       move in exactly the same order from day to day. {CEIL_TXT}{BARD_TXT}</p>
    <p class="picklabel">Measure</p>
    {pick("q", "metric", [(k, lab) for k, lab, _, _, _ in Q_METRICS])}
    {q_text}
    {"".join(q_blocks)}
    <p class="note">The trigger reads each model against the model's own levels, so multiplying a model's flow by a fixed factor does not change the scores in
       sections 3 to 9.</p>

    <h2><span class="num">11</span>The gauges used as the benchmark</h2>
    <p>A level meant to be reached in one season in three would be reached in about a third of the seasons on record. The chart shows how often each
       gauge reaches it, and how often a gauge sits at one reading for five days or more at the height of a season: {FLAT_TXT}. In those seasons the record
       gives one reading for the peak days.{SAME_TXT}</p>
    {fig_html(fig_gauges(GN), "The SWALIM gauges: seasons on record, seasons at or over the 1-in-3 level, seasons at one top reading for five days or more")}

    <h2><span class="num">12</span>Limits of this review</h2>
    <ul>
      <li>On 5 November 2026 Google's Flood Hub, its public forecast service, was not yet serving the new model. Everything here on the new model is from Google's published
          reanalysis and forecast archive.</li>
      <li>On forecast day 7 the new model's forecasts have a rank correlation of {d7.rho_vs_reanalysis.min():.3f} to {d7.rho_vs_reanalysis.max():.3f} with its own
          reanalysis and show {100 * d7.rise_share_median.min():.0f} to {100 * d7.rise_share_median.max():.0f}% of the largest rises in that reanalysis, station by station (section 5). Whether a live forecast will do
          the same is not known from the archive.</li>
      <li>GloFAS v5 is scored on its reanalysis only: no archive of its past forecasts is published. On 17 November 2026 the <a href="https://ewds.climate.copernicus.eu/datasets/cems-glofas-reforecast">Copernicus Early Warning Data
          Store</a> listed such archives (reforecasts) for GloFAS versions 2.2, 3.1 and 4.0, and none for version 5. The <a href="https://confluence.ecmwf.int/x/38QHIQ">Copernicus release page for GloFAS v5</a>
          gives 12 November 2026 for the pre-operational release, "Q4 2026 (TBD)" for the operational release and "tbc" for the reforecasts.</li>
      <li>The forecast comparison rests on eight years, 2016 to 2023{SHORT_TXT}, with {NFL_TXT} river floods at the 1-in-3 level on each river in each season.</li>
      <li>The river benchmark (two gauges, at any time in the season) and the trigger rule (three of four or two of three stations, on the same day) are
          not the same test, and the benchmark depends on which gauges were reporting.</li>
      <li>The Bardheere gauge has no readings after November 2023 and the Bualle gauge none after March 2024, so the lower Juba cannot be checked from then on.</li>
      <li>Each model's flow is read at one place on its river network per station: the grid cell or river segment on the main river nearest the station.</li>
      <li>Stations on one river rise together, so the seven stations are not seven separate tests, and totals added over stations count the same floods more than once.</li>
    </ul>

    <h2><span class="num">13</span>Method and files</h2>
    <p class="note">Levels are read from each record's own season peaks: the peaks are ranked, and the 1-in-3 level is the value at the rank that one
       season in three reaches (Weibull plotting position). A value equal to its level counts as reaching it. A gauge season counts with at least 30 readings; a reading of zero is treated as a gap. Station floods
       compare the season's highest values; the river rule counts the stations over their level on the same day, in the season months only. Ranges: each
       score is recomputed 1,000 times ({R.N_BOOT} for correlations) on a random pick of the same number of seasons, in which a season can be picked more
       than once; the range holds the middle 90% of the results. Middle values of
       days that fall on a half are rounded away from zero. The archive of Google, current stamps its forecast
       days one day later than its reanalysis days. In sections 4 and 5 it is lined up on its reanalysis. In the river rule its forecast days 1 to 7 are the
       seven days after the day of issue, as the live trigger reads them.{SIB_TXT}
       Sources: Google Runoff Reanalysis and Reforecast, 2024 release and November 2026 release; SWALIM gauge levels and discharge series; GloFAS v4 and v5 from the
       Copernicus Emergency Management Service.</p>
    <p class="note files">Every score, by season, river and station:
      <a href="river_rule.csv">river rule</a><a href="station_floods_1in3.csv">station floods, 1-in-3</a><a href="station_floods_1in5.csv">station floods, 1-in-5</a>
      <a href="daily_agreement.csv">daily agreement and timing</a><a href="forecasts_by_forecast_day.csv">forecasts by forecast day</a><a href="discharge.csv">discharge</a>
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
