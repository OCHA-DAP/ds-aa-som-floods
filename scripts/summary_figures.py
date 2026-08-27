"""Charts for the summary page: the deck's figures, redrawn from primary data.

Called by scripts/build_summary_page.py, which loads the tables and hands
this module a small context object. Every figure is written as SVG into
pages/summary/figures/ and embedded in the page, so the page carries the
same visuals as the review deck without anybody re-running a notebook.

Chart order follows the deck:
  1 thresholds      RP3 level against the official Moderate-to-High band
  2 crossings       distinct crossings of the RP3 level, per season-year
  3 backtest        RP3 crossings against official Moderate crossings
  4 skill           POD and FAR per model and gauge, RP3 baseline
  5 correlation     best-lag rank correlation per model, gauge and season
  6 grid            POD / FAR / F1 over the station RP and votes grid
  7 activation      when each window activates, and the envelope it implies
  8 map             the gauges the trigger runs on (optional: needs codab)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.constants import (  # noqa: E402
    BODY,
    C_HIGH,
    C_MAIN,
    C_BAND,
    C_BAND_EDGE,
    C_DEYR,
    C_GU,
    C_JUBA,
    C_REF,
    C_SHAB,
    FAINT,
    GRID,
    INK,
    WINDOW_MODEL,
    SEASONS,
    SOURCE_COLORS,
    STATIONS,
    TRIGGER_CONFIG,
    TRIGGER_STATIONS,
)
from src.plots import apply_chart_style, style_ax  # noqa: E402
from src.utils import episodes, weibull_level, weibull_threshold  # noqa: E402

NICE = {
    "google_grrr": "Google",
    "glofas_v5": "GloFAS v5",
    "glofas_v4": "GloFAS v4",
    "geoglows": "GEOGloWS",
}
RIVER_COLOR = {"juba": C_JUBA, "shabelle": C_SHAB}
apply_chart_style()


def label(station):
    st = STATIONS.get(station)
    tag = "J" if (st and st.river == "juba") else "S"
    return f"{st.name if st else station} ({tag})"


def rows():
    """(river, station) in the order the charts list them: Juba then Shabelle."""
    return [(r, s) for r, ss in TRIGGER_STATIONS.items() for s in ss]


def save(fig, figdir, name):
    path = figdir / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    # SUMMARY_FIG_PNG=<dir> also drops a raster copy there, for eyeballing the
    # charts without a browser. Not part of the published page.
    png_dir = os.environ.get("SUMMARY_FIG_PNG")
    if png_dir:
        d = Path(png_dir)
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}.svg"


def heat(ax, mat, xlabels, ylabels, fmt="{:.2f}", cmap="Blues", vmin=0, vmax=1,
         reverse=False, box=None):
    """A small annotated heatmap, deck style: dark = better."""
    m = np.asarray(mat, dtype=float)
    shown = 1 - m if reverse else m
    ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isnan(m[i, j]):
                continue
            dark = (shown[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.55
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if dark else BODY)
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=9, color=FAINT)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9, color=BODY)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, m.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    if box is not None:
        i, j = box
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=INK, linewidth=2, zorder=5))


# ------------------------------------------------------------------ 1. levels
def fig_thresholds(ctx, figdir):
    """RP3 level per season against the official Moderate-to-High band."""
    lv, th = ctx["lv"], ctx["th"]
    rs = rows()[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 0.62 * len(rs) + 1.5))
    for y, (river, st) in enumerate(rs):
        s = lv[lv.station == st].set_index("date")["level_m"].sort_index()
        if st in th.index:
            mod, high = th.loc[st, "moderate_flood_risk"], th.loc[st, "high_flood_risk"]
            bf = th.loc[st, "bank_full"]
            if pd.notna(mod) and pd.notna(high):
                ax.add_patch(Rectangle((mod, y - 0.3), max(high - mod, 0.02), 0.6,
                                       facecolor=C_BAND, edgecolor=C_BAND_EDGE,
                                       linewidth=0.8, zorder=1))
            if pd.notna(bf):
                ax.plot([bf, bf], [y - 0.34, y + 0.34], color=INK, linewidth=2.2,
                        zorder=3)
        for season, months in SEASONS.items():
            ss = s[s.index.month.isin(months)]
            ss = ss[ss.index.year >= 2000]
            am = ss.groupby(ss.index.year).max().dropna()
            rp3 = weibull_level(am.values, 3)
            if not np.isnan(rp3):
                ax.scatter(rp3, y, s=52, zorder=4, color=C_GU if season == "gu" else C_DEYR,
                           edgecolors="white", linewidths=0.8)
    ax.set_yticks(range(len(rs)), [label(s) for _, s in rs])
    ax.set_ylim(-0.7, len(rs) - 0.3)
    ax.set_xlabel("water level (m)")
    ax.set_title("1-in-3-year (RP3) flood level against the official Moderate to High "
                 "band,\nper gauge and season")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=C_GU, label="RP3: Gu"),
        plt.Line2D([], [], marker="o", linestyle="", color=C_DEYR, label="RP3: Deyr"),
        Rectangle((0, 0), 1, 1, facecolor=C_BAND, edgecolor=C_BAND_EDGE,
                  label="Moderate to High band"),
        plt.Line2D([], [], color=INK, linewidth=2.2, label="Bank full"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    style_ax(ax, grid="x")
    return save(fig, figdir, "thresholds")


# --------------------------------------------------------------- 2. crossings
def _season_rows(ctx):
    """(river, station, season, RP3 level, per-year season series) for each row."""
    lv = ctx["lv"]
    out = []
    for river, st in rows():
        s = lv[lv.station == st].set_index("date")["level_m"].sort_index()
        for season, months in SEASONS.items():
            ss = s[s.index.month.isin(months)]
            modern = ss[ss.index.year >= 2000]
            am = modern.groupby(modern.index.year).max().dropna()
            out.append((river, st, season, weibull_level(am.values, 3), ss))
    return out


def fig_crossings(ctx, figdir):
    """Distinct crossings of the RP3 level per season-year."""
    sr = _season_rows(ctx)
    years = list(range(2000, ctx["Y1"] + 1))
    mat = np.full((len(sr), len(years)), np.nan)
    for i, (_, _, _, rp3, ss) in enumerate(sr):
        if np.isnan(rp3):
            continue
        for j, yr in enumerate(years):
            sy = ss[ss.index.year == yr]
            if not len(sy):
                continue
            mat[i, j] = len(episodes(sy >= rp3))
    fig, ax = plt.subplots(figsize=(12.4, 0.42 * len(sr) + 1.8))
    ax.imshow(np.nan_to_num(mat, nan=-1), cmap="Blues", vmin=-1,
              vmax=max(2, np.nanmax(mat)), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v) or v == 0:
                continue
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7.5,
                    color="white" if v >= 2 else BODY)
    labels = [f"{label(st)} {season.upper()}" for _, st, season, _, _ in sr]
    ax.set_yticks(range(len(sr)), labels, fontsize=8)
    ax.set_xticks(range(0, len(years), 2), [str(y) for y in years[::2]], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sr), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    n_juba = 2 * len(TRIGGER_STATIONS["juba"])
    ax.axhline(n_juba - 0.5, color=INK, linewidth=1.1)
    ax.set_title("Distinct crossings of the RP3 level per season-year "
                 "(number = crossings)\nJuba above the line | pale = no crossing, "
                 "white = no record")
    return save(fig, figdir, "crossings")


# ---------------------------------------------------------------- 3. backtest
def fig_backtest(ctx, figdir):
    """RP3 crossings against official Moderate crossings, year by year."""
    th = ctx["th"]
    sr = _season_rows(ctx)
    years = list(range(2000, ctx["Y1"] + 1))
    CODES = {"both": 3, "rp3": 2, "mod": 1, "neither": 0, "none": np.nan}
    colors = {0: "#EDF1F4", 1: "#F4A93B", 2: "#0E8A7B", 3: "#1F5C8B"}
    mat = np.full((len(sr), len(years)), np.nan)
    for i, (_, st, _, rp3, ss) in enumerate(sr):
        mod = th.loc[st, "moderate_flood_risk"] if st in th.index else np.nan
        for j, yr in enumerate(years):
            sy = ss[ss.index.year == yr]
            if not len(sy):
                continue
            hit_rp3 = (not np.isnan(rp3)) and bool((sy >= rp3).any())
            hit_mod = pd.notna(mod) and bool((sy >= mod).any())
            mat[i, j] = CODES["both"] if hit_rp3 and hit_mod else (
                CODES["rp3"] if hit_rp3 else CODES["mod"] if hit_mod else CODES["neither"])
    rgb = np.ones(mat.shape + (3,))
    for code, hexcol in colors.items():
        c = tuple(int(hexcol[k:k + 2], 16) / 255 for k in (1, 3, 5))
        rgb[mat == code] = c
    fig, ax = plt.subplots(figsize=(12.4, 0.42 * len(sr) + 1.8))
    ax.imshow(rgb, aspect="auto")
    labels = [f"{label(st)} {season.upper()}" for _, st, season, _, _ in sr]
    ax.set_yticks(range(len(sr)), labels, fontsize=8)
    ax.set_xticks(range(0, len(years), 2), [str(y) for y in years[::2]], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sr), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.axhline(2 * len(TRIGGER_STATIONS["juba"]) - 0.5, color=INK, linewidth=1.1)
    handles = [Rectangle((0, 0), 1, 1, facecolor=colors[3], label="crossed both levels"),
               Rectangle((0, 0), 1, 1, facecolor=colors[2], label="RP3 only"),
               Rectangle((0, 0), 1, 1, facecolor=colors[1], label="Moderate only"),
               Rectangle((0, 0), 1, 1, facecolor=colors[0], label="neither"),
               Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID,
                         label="no record")]
    ax.legend(handles=handles, ncol=5, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.09))
    ax.set_title("Backtest: RP3 crossings (the baseline we monitor) against official "
                 "Moderate crossings\nJuba above the line")
    return save(fig, figdir, "backtest")


# ------------------------------------------------------------------- 4. skill
def fig_skill(ctx, figdir):
    """POD and FAR per model and gauge, against the RP3 baseline."""
    scores = pd.DataFrame(ctx["station_scores"])
    if scores.empty:
        return None
    sources = [s for s in ["glofas_v5", "glofas_v4", "google_grrr", "geoglows"]
               if s in set(scores.source)]
    rs = rows()
    ylabels, pod, far = [], [], []
    for river, st in rs:
        sub = scores[(scores.station == st)]
        if sub.empty:
            continue
        n = int(sub.n_events.iloc[0])
        ylabels.append(f"{label(st)}  n={n}")
        pod.append([float(sub[sub.source == s].POD.iloc[0]) if len(sub[sub.source == s])
                    else np.nan for s in sources])
        far.append([float(sub[sub.source == s].FAR.iloc[0]) if len(sub[sub.source == s])
                    else np.nan for s in sources])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 0.52 * len(ylabels) + 2.0))
    heat(axes[0], pod, [NICE[s] for s in sources], ylabels)
    axes[0].set_title("POD: share of RP3 floods the model caught\n(dark = better)")
    heat(axes[1], far, [NICE[s] for s in sources], [""] * len(ylabels), reverse=True)
    axes[1].set_title("FAR: share of the model's alarms that were false\n(dark = better)")
    fig.subplots_adjust(wspace=0.06)
    return save(fig, figdir, "skill")


# ------------------------------------------------------------- 5. correlation
def fig_correlation(ctx, figdir):
    """Best-lag rank correlation against the reference gauge, by season."""
    rho = ctx.get("rho") or {}
    if not rho:
        return None
    sources = [s for s in ["glofas_v5", "glofas_v4", "google_grrr", "geoglows"]
               if any(k[2] == s for k in rho)]
    rs = rows()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 0.52 * len(rs) + 2.0))
    for ax, season in zip(axes, ["gu", "deyr"]):
        mat = [[rho.get((river, st, s, season), np.nan) for s in sources]
               for river, st in rs]
        heat(ax, mat, [NICE[s] for s in sources],
             [label(st) for _, st in rs] if season == "gu" else [""] * len(rs),
             vmin=0.3, vmax=0.95)
        ax.set_title(f"{season.title()}: best-lag rho against the gauge\n(dark = better)")
    fig.subplots_adjust(wspace=0.06)
    return save(fig, figdir, "correlation")


# -------------------------------------------------------------- 6. grid search
def grid_scores(ctx, river, season, model, rps=(3, 4, 5), n_max=None):
    """POD / FAR / F1 over the (station RP, votes required) grid for one window."""
    months = SEASONS[season]
    stns = TRIGGER_STATIONS[river]
    n_max = n_max or len(stns)
    ev = ctx["flood_years"](river, season, 3)
    out = {}
    for rp in rps:
        cols = []
        for st in stns:
            s = ctx["model_season"](model, st, months)
            if len(s) < 100:
                continue
            am = s.groupby(s.index.year).max().dropna()
            t = weibull_threshold(am.values, rp)
            if not np.isnan(t):
                cols.append((s >= t).rename(st))
        if not cols:
            continue
        mat = pd.concat(cols, axis=1).fillna(False)
        mx = mat.sum(axis=1).groupby(mat.index.year).max()
        for n in range(1, n_max + 1):
            act = set(mx[mx >= n].index) & ctx["span"]
            tp, fp, fn = len(act & ev), len(act - ev), len(ev - act)
            pod = tp / (tp + fn) if tp + fn else np.nan
            far = fp / (tp + fp) if tp + fp else np.nan
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
            out[(rp, n)] = (pod, far, f1)
    return out


def fig_grid(ctx, figdir):
    """The grid the configuration was chosen from, one row per window."""
    windows = list(TRIGGER_CONFIG.items())
    fig, axes = plt.subplots(len(windows), 3,
                             figsize=(11.0, 2.1 * len(windows) + 1.0))
    rps = (3, 4, 5)
    for r, ((river, season), spec) in enumerate(windows):
        model = WINDOW_MODEL[(river, season)]
        g = grid_scores(ctx, river, season, model, rps=rps)
        if not g:
            continue
        ns = sorted({n for _, n in g})
        for c, (metric, idx, rev) in enumerate(
            [("POD", 0, False), ("FAR", 1, True), ("F1", 2, False)]
        ):
            ax = axes[r, c]
            mat = [[g.get((rp, n), (np.nan,) * 3)[idx] for rp in rps] for n in ns]
            box = None
            if spec["rp"] in rps and spec["n_req"] in ns:
                box = (ns.index(spec["n_req"]), rps.index(spec["rp"]))
            heat(ax, mat, [str(rp) for rp in rps],
                 [str(n) for n in ns] if c == 0 else [""] * len(ns),
                 reverse=rev, box=box)
            title = f"{river.title()} {season.title()} | {metric}" if c == 0 else metric
            ax.set_title(title, fontsize=10)
            if r == len(windows) - 1:
                ax.set_xlabel("station RP")
            if c == 0:
                ax.set_ylabel("gauges required")
    fig.suptitle("Grid search against the RP3 gauge benchmark (dark = better | "
                 "outlined cell = adopted)",
                 x=0.09, ha="left", fontweight="bold", fontsize=11.5, color=INK)
    fig.subplots_adjust(hspace=0.55, wspace=0.06, top=0.93)
    return save(fig, figdir, "grid")


# --------------------------------------------------------------- 7. activation
def fig_activation(ctx, figdir):
    """When each window activates, against how bad the year actually was.

    At a 1-in-3 envelope rate the trigger is not trying to catch every
    1-in-3 flood, so the chart separates the severe years (the ones it is
    judged on) from the ordinary floods it deliberately leaves uncovered.
    """
    per_window, years = ctx["per_window"], list(range(ctx["Y0"], ctx["Y1"] + 1))
    keys = list(per_window)
    union = {y for a in per_window.values() for y in a}
    any_flood, any_severe = set(), set()
    for river, season in keys:
        any_flood |= ctx["flood_years"](river, season, 3)
        any_severe |= ctx["severe_window_years"](river, season)
    lines = [(f"{r.title()} {s.title()}", set(per_window[(r, s)]),
              ctx["flood_years"](r, s, 3), ctx["severe_window_years"](r, s))
             for r, s in keys]
    lines.append(("Envelope: any window", union, any_flood, any_severe))
    colors = {"hit_sev": "#123F5F", "hit": "#3E86BE", "false": "#F4A93B",
              "miss_sev": "#B34036", "miss": "#E3B4AF", "quiet": "#EDF1F4"}
    fig, ax = plt.subplots(figsize=(12.4, 0.62 * len(lines) + 2.4))
    for i, (name, activated, ev, sev) in enumerate(lines[::-1]):
        for j, y in enumerate(years):
            if y in activated:
                kind = "hit_sev" if y in sev else "hit" if y in ev else "false"
            else:
                kind = "miss_sev" if y in sev else "miss" if y in ev else "quiet"
            ax.add_patch(Rectangle((j - 0.46, i - 0.36), 0.92, 0.72,
                                   facecolor=colors[kind],
                                   edgecolor="white", linewidth=0.8))
    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.set_ylim(-0.6, len(lines) - 0.4)
    ax.set_yticks(range(len(lines)), [ln[0] for ln in lines[::-1]], fontsize=9.5)
    ax.set_xticks(range(0, len(years), 2), [str(y) for y in years[::2]], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0.5, color=INK, linewidth=1.0)
    handles = [Rectangle((0, 0), 1, 1, facecolor=colors["hit_sev"],
                         label="activated, severe flood"),
               Rectangle((0, 0), 1, 1, facecolor=colors["hit"],
                         label="activated, ordinary flood"),
               Rectangle((0, 0), 1, 1, facecolor=colors["false"],
                         label="activated, no flood at the gauge"),
               Rectangle((0, 0), 1, 1, facecolor=colors["miss_sev"],
                         label="severe flood missed"),
               Rectangle((0, 0), 1, 1, facecolor=colors["miss"],
                         label="ordinary flood, not covered"),
               Rectangle((0, 0), 1, 1, facecolor=colors["quiet"], edgecolor=GRID,
                         label="quiet year")]
    ax.legend(handles=handles, ncol=3, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.07))
    ax.set_title(f"Backtest of the adopted configuration, {ctx['Y0']}-{ctx['Y1']}\n"
                 "bottom row: the funding envelope, which releases when any window activates")
    return save(fig, figdir, "activation")


# --------------------------------------------------------------------- 8. map
def fig_map(ctx, figdir):
    """The gauges the trigger runs on. Needs codab; skipped if unavailable."""
    import geopandas as gpd  # noqa: F401
    import ocha_stratus as stratus

    adm = stratus.codab.load_codab_from_blob("som", admin_level=1)
    fig, ax = plt.subplots(figsize=(7.4, 8.4))
    adm.plot(ax=ax, color="#F5F7F9", edgecolor="white", linewidth=0.8)
    for river, stns in TRIGGER_STATIONS.items():
        pts = [(STATIONS[s].lon, STATIONS[s].lat, STATIONS[s].name) for s in stns
               if s in STATIONS]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=RIVER_COLOR[river],
                linewidth=1.0, alpha=0.45, zorder=2)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70,
                   color=RIVER_COLOR[river], edgecolors="white", linewidths=0.9,
                   zorder=4, label=f"{river.title()} ({len(pts)} gauges)")
        for lon, lat, nm in pts:
            ax.annotate(nm, (lon, lat), textcoords="offset points", xytext=(8, 2),
                        fontsize=8.5, color=INK)
    # no reference-gauge marker: a flood year is decided by two or more of the
    # river's gauges, so no single gauge carries the benchmark
    ax.set_xlim(40.5, 48.6)
    ax.set_ylim(-2.0, 6.6)
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("The gauges the trigger runs on\nfour on the Juba, three on the "
                 "Shabelle", fontsize=11.5, loc="left")
    return save(fig, figdir, "map")


# ------------------------------------------------------- 9. forecast correlation
def fig_forecast(ctx, figdir):
    """Best-lag correlation of the FORECASTS against the gauge, leads 1-7.

    The reanalysis says which model describes the river best; this says which
    model describes it best when it is actually forecasting. Dot is the mean
    across leads 1 to 7, the line the spread between the best and worst lead.
    """
    lv = ctx["lv"]
    fc = ctx["reforecast"]()
    if fc is None or fc.empty:
        return None
    rs = rows()
    sources = [s for s in ["google_grrr", "glofas_v4"] if s in set(fc.source_key)]
    if not sources:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 0.52 * len(rs) + 2.2), sharey=True)
    for ax, (season, months) in zip(axes, SEASONS.items()):
        for src in sources:
            ys, means, lo, hi = [], [], [], []
            for y, (river, st) in enumerate(rs[::-1]):
                obs = lv[lv.station == st].set_index("date")["level_m"].sort_index()
                obs = obs[obs.index.month.isin(months)]
                sub = fc[(fc.source_key == src) & (fc.station == st)]
                if sub.empty or obs.empty:
                    continue
                per_lead = []
                for lead in range(1, 8):
                    d = sub[sub.leadtime_days == lead]
                    if d.empty:
                        continue
                    # 50% of the ensemble, as the trigger rule uses
                    s = d.groupby("valid_time")["discharge"].median().sort_index()
                    s = s[s.index.month.isin(months)]
                    best = 0.0
                    for lag in range(-10, 11):
                        j = pd.concat([s, obs.shift(-lag)], axis=1,
                                      join="inner").dropna()
                        if len(j) < 60:
                            continue
                        r = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
                        if pd.notna(r) and abs(r) > abs(best):
                            best = float(r)
                    if best:
                        per_lead.append(best)
                if not per_lead:
                    continue
                ys.append(y)
                means.append(np.mean(per_lead))
                lo.append(min(per_lead))
                hi.append(max(per_lead))
            color = SOURCE_COLORS.get(src, C_MAIN)
            for y, a, b in zip(ys, lo, hi):
                ax.plot([a, b], [y, y], color=color, linewidth=2.4, alpha=0.35,
                        solid_capstyle="round", zorder=2)
            ax.scatter(means, ys, s=52, color=color, edgecolors="white",
                       linewidths=0.8, zorder=3, label=NICE.get(src, src))
        ax.axvline(0.5, color=C_REF, linestyle=":", linewidth=1.0)
        ax.set_title(season.title())
        ax.set_xlabel("best-lag rank correlation against the gauge")
        ax.set_xlim(0.3, 0.95)
        style_ax(ax, grid="x")
    axes[0].set_yticks(range(len(rs)), [label(st) for _, st in rs[::-1]])
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Forecast correlation against the gauge, leads 1 to 7 "
                 "(dot = mean across leads | line = best to worst lead)",
                 x=0.075, ha="left", fontweight="bold", fontsize=11.5, color=INK)
    fig.subplots_adjust(top=0.88, wspace=0.06)
    return save(fig, figdir, "forecast")


# ------------------------------------------------------------- 10. exposure map
def fig_exposure(ctx, figdir):
    """Population exposed to flooding by district, against the gauge locations."""
    import geopandas as gpd
    import ocha_stratus as stratus
    from matplotlib.ticker import FuncFormatter

    exp = ctx["exposure"]()
    if exp is None or exp.empty:
        return None
    adm2 = stratus.codab.load_codab_from_blob("som", admin_level=2)
    peak = (
        exp.groupby([exp.valid_date.dt.year, "pcode"])["sum"].max()
        .groupby("pcode").median().rename("peak_exposed")
    )
    riverine = adm2.merge(peak.reset_index(), left_on="ADM2_PCODE", right_on="pcode")
    if riverine.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    adm2.plot(ax=ax, color="#F5F7F9", edgecolor="white", linewidth=0.5)
    riverine.plot(ax=ax, column="peak_exposed", cmap="Blues", edgecolor="white",
                  linewidth=0.6, legend=True,
                  legend_kwds={"label": "median yearly peak population exposed",
                               "shrink": 0.55,
                               "format": FuncFormatter(lambda v, _: f"{v:,.0f}")})
    for river, stns in TRIGGER_STATIONS.items():
        pts = [(STATIONS[s].lon, STATIONS[s].lat) for s in stns if s in STATIONS]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=46,
                   color=RIVER_COLOR[river], edgecolors="white", linewidths=0.8,
                   zorder=5, label=f"{river.title()} gauges")
    import matplotlib.patheffects as pe

    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    for k, (_, r) in enumerate(riverine.nlargest(6, "peak_exposed").iterrows()):
        pt = r.geometry.representative_point()
        # stagger the offsets: the most exposed districts sit close together
        dx, dy = (0.55, 0.12) if k % 2 else (-0.55, -0.12)
        ax.annotate(f"{r.ADM2_EN}\n{r.peak_exposed:,.0f}", (pt.x, pt.y),
                    xytext=(pt.x + dx, pt.y + dy), ha="center", fontsize=7.5,
                    color=INK, path_effects=halo,
                    arrowprops=dict(arrowstyle="-", color=C_REF, linewidth=0.6,
                                    shrinkA=0, shrinkB=2))
    ax.set_xlim(40.5, 48.6)
    ax.set_ylim(-2.0, 6.6)
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Who sits behind the gauges\nmedian yearly peak population in "
                 "flooded areas per district (FloodScan, 1998 onwards)",
                 fontsize=11.5, loc="left")
    return save(fig, figdir, "exposure")


# ------------------------------------------------------------- 11. the frontier
def fig_frontier(ctx, figdir):
    """What the envelope can buy: activation rate against severe-year detection.

    Each point is the best configuration available at that activation rate.
    Moving left costs money (more frequent releases); moving right costs
    coverage (more severe years missed).
    """
    fr = ctx.get("frontier") or []
    if not fr:
        return None
    fr = sorted(fr, key=lambda r: r["env_rp"])
    x = [r["env_rp"] for r in fr]
    y = [r["severe_caught"] for r in fr]
    n_sev = ctx["n_severe"]
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(x, y, color=C_MAIN, linewidth=2, zorder=2)
    ax.scatter(x, y, s=44, color=C_MAIN, edgecolors="white", linewidths=0.8, zorder=3)
    target = ctx["target_rp"]
    ax.axvline(target, color=C_REF, linestyle=":", linewidth=1.2, zorder=1)
    ax.annotate(f"target: 1-in-{target}", (target, n_sev + 0.45), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=FAINT, va="top")
    adopted = ctx.get("adopted_point")
    if adopted:
        ax.scatter([adopted["env_rp"]], [adopted["severe_caught"]], s=190,
                   facecolor="none", edgecolors=C_HIGH, linewidths=2.2, zorder=4)
        # label to the left: the adopted point sits near the right-hand edge
        ax.annotate(
            f"adopted: 1-in-{adopted['env_rp']},\n"
            f"{adopted['severe_caught']} of {n_sev} severe years",
            (adopted["env_rp"], adopted["severe_caught"]), xytext=(-14, -30),
            textcoords="offset points", fontsize=9.5, color=C_HIGH,
            fontweight="bold", ha="right")
    ax.set_ylim(0, n_sev + 0.9)
    ax.set_xlim(min(x) - 0.15, max(max(x), target) + 0.3)
    ax.set_xlabel("envelope activation rate (one release every N years)")
    ax.set_ylabel(f"severe years caught (of {n_sev})")
    ax.set_title("What the envelope can buy\nbest configuration available at each "
                 "activation rate")
    style_ax(ax, grid="y")
    return save(fig, figdir, "frontier")


# ---------------------------------------------------------------------- driver
BUILDERS = [
    ("thresholds", fig_thresholds,
     "Where the 1-in-3-year level sits against the official marks. The official "
     "Moderate band means very different things from gauge to gauge, which is why "
     "the trigger monitors the computed RP3 instead."),
    ("map", fig_map,
     "Seven gauges still reporting: four on the Juba, three on the Shabelle."),
    ("crossings", fig_crossings,
     "How often each gauge crosses its RP3 level. Most gauges reach it in roughly "
     "one season-year in three, by construction."),
    ("backtest", fig_backtest,
     "The two baselines side by side. Amber cells are years the official Moderate "
     "mark activated and the RP3 did not, which is where the two definitions disagree."),
    ("exposure", fig_exposure,
     "The gauge record matters because of who sits behind it: the districts along "
     "both rivers with the largest populations in flooded areas."),
    ("skill", fig_skill,
     "Each model's 1-in-3-year signal against the floods the gauges actually "
     "recorded, within a 7-day window."),
    ("correlation", fig_correlation,
     "Daily tracking rather than event detection: the best-lag rank correlation "
     "between each model and the river's reference gauge."),
    ("forecast", fig_forecast,
     "The same test run on the forecasts rather than the hindsight simulations. "
     "Accuracy holds across leads 1 to 7, which is what makes a one-week action "
     "window defensible."),
    ("grid", fig_grid,
     "Every combination of station return period and number of gauges that must "
     "agree, scored against the gauge benchmark. The outlined cell is what we adopted."),
    ("frontier", fig_frontier,
     "The trade-off behind the activation rate. Releasing more often catches more "
     "of the severe years; the adopted point is the rarest release that still "
     "catches eight of the ten."),
    ("activation", fig_activation,
     "Year by year, what the adopted configuration would have done, and what the "
     "envelope does when any of the four windows activates."),
]


def build(ctx, figdir):
    """Draw every figure; returns {name: {"src", "caption"}} for the page."""
    figdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn, caption in BUILDERS:
        try:
            src = fn(ctx, figdir)
        except Exception as exc:  # a missing optional input must not kill the page
            print(f"  ! figure {name} skipped: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if src:
            out[name] = {"src": src, "caption": caption}
            print(f"  drew {name}")
    return out
