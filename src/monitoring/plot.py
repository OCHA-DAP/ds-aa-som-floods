"""Daily monitoring chart: every trigger point as a share of its own level.

Two rows (Juba, Shabelle) by two columns (Google Flood Hub, GloFAS ensemble
median). Each line is one point, expressed as a percentage of that point's
threshold for the season being watched, so points with very different
discharges share one axis and 100% is the level that counts as a vote. The x axis is
lead time (days ahead of the issue): the activation band (1 to 7 d) sits left of the
readiness band (8 to 12 d), with calendar dates along the top. Palette and
type follow src.constants so the chart matches the analysis pages.
"""

import io
from datetime import timedelta

import matplotlib
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import ocha_stratus as stratus
from src.constants import (BODY, C_GLOFAS4, C_GOOGLE, FAINT, GRID, INK, SEASONS,
                           TRIGGER_STATIONS)
from src.monitoring import config as cfg
from src.monitoring import thresholds as thr

matplotlib.use("Agg")

STATION_COLORS = {
    "dollow": "#065A82", "luuq": "#1C7293", "bardheere": "#2A78D6", "bualle": "#8E5FA8",
    "belet_weyne": "#065A82", "bulo_burti": "#0E8A7B", "jowhar": "#EB6834",
}
STATUS_COLORS = {"ACTIVATION TRIGGER REACHED": "#B34036", "READINESS TRIGGER REACHED": "#D48F2A",
                 "TRIGGER NOT REACHED": "#2F9E6F", "NO WINDOW OPEN": FAINT}
PRODUCT_COLORS = {"google": C_GOOGLE, "glofas": C_GLOFAS4}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": GRID, "axes.labelcolor": BODY, "xtick.color": FAINT,
    "ytick.color": FAINT, "text.color": INK, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "axes.titlesize": 11, "figure.facecolor": "white",
})


def season_shown(monitoring_date):
    """The season whose monitoring window is open now (calendar months in
    cfg.MONITORING_OPEN_MONTHS), else the one that opens next."""
    for season, months in cfg.MONITORING_OPEN_MONTHS.items():
        if monitoring_date.month in months:
            return season, True
    return ("gu" if monitoring_date.month == 1 else "deyr"), False


def _day_month(d):
    """15 Sep, without the platform-specific %-d / %#d flags."""
    return f"{d.day} {d:%b}"


def chart_blob_name(monitoring_date):
    return f"{cfg.PROJECT_PREFIX}/monitoring/{monitoring_date}.png"


def _panel(ax, df, product, river, season, levels, result_window, role):
    stations = TRIGGER_STATIONS[river]
    sub = df[(df.source == product) & df.station.isin(stations)]
    if sub.empty:
        ax.text(0.5, 0.5, f"no {cfg.SOURCE_TITLE[product]} data retrieved", ha="center",
                va="center", transform=ax.transAxes, color=FAINT, fontsize=10)
    issue = pd.Timestamp(sub.issued_time.iloc[0]).tz_localize(None).normalize() if len(sub) else None
    # x axis is lead time: days ahead of the forecast issue, so the activation band (near)
    # sits left of the readiness band (far). A forecast flood enters on the right and moves
    # left as the days pass: readiness first, then activation.
    a0, a1 = cfg.ACTION_LEADS
    # the two bands meet halfway between the last activation day and the first readiness day
    ax.axvspan(a0, cfg.READINESS_LEADS[0] - 0.5 if product == "glofas" else a1, color="#E6EEF7", zorder=0)
    box = dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85)
    ax.text(a0 + 0.1, 0.97, f"activation (days {a0} to {a1})", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=BODY, va="top", fontweight="bold", bbox=box, zorder=5)
    if product == "glofas":
        r0, r1 = cfg.READINESS_LEADS
        ax.axvspan(r0 - 0.5, r1, color="#FBF3E4", zorder=0)
        ax.text(r0 - 0.4, 0.97, f"readiness (days {r0} to {r1})", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=BODY, va="top", fontweight="bold", bbox=box, zorder=5)
        ax.annotate("", xy=(a0 + 0.2, 0.885), xytext=(r1 - 0.1, 0.885), xycoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1))
        ax.text((a0 + r1) / 2, 0.90, "readiness first, then activation", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=FAINT, ha="center", va="bottom", style="italic", bbox=box, zorder=5)
        xmax = r1
    else:
        xmax = a1
    for st in stations:
        s = sub[sub.station == st].sort_values("leadtime_days")
        lvl = levels.get(st)
        if s.empty or not lvl:
            continue
        pct = 100 * s["value"] / lvl
        exceeds = (pct >= 100).any()
        ax.plot(s["leadtime_days"], pct, color=STATION_COLORS[st], lw=2.2 if exceeds else 1.6,
                marker="o", ms=3, label=f"{cfg.STATION_TITLE[st]} ({lvl:,.0f} m³/s)", zorder=3)
    live = [st for st in stations if not sub[sub.station == st].empty]
    missing = [cfg.STATION_TITLE[st] for st in stations if st not in live]
    ax.axhline(100, color="#B34036", ls="--", lw=1.2, zorder=2)
    ax.text(xmax + 0.45, 98.5, "trigger level", color="#B34036", fontsize=7.5, va="top", ha="right",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85), zorder=5)
    ax.set_ylim(bottom=0)
    top = max((float(np.nanmax(np.asarray(line.get_ydata(), dtype=float))) for line in ax.get_lines() if len(line.get_ydata())), default=0)
    ax.set_ylim(top=max(120, top * 1.25))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_xticks(range(1, xmax + 1))
    ax.set_xticklabels([str(d) for d in range(1, xmax + 1)])
    issued_label = f"days ahead of the {_day_month(issue)} forecast" if issue is not None else "days ahead of the forecast"
    ax.set_xlabel(issued_label, color=BODY, fontsize=9)
    if issue is not None:
        # calendar dates along the top for readers who want them; GloFAS day n is issue+(n-1), Google lead n is issue+n
        off = -1 if product == "glofas" else 0
        top = ax.secondary_xaxis("top")
        top.set_xticks(range(1, xmax + 1, 2))
        top.set_xticklabels([_day_month(issue + pd.Timedelta(days=d + off)) for d in range(1, xmax + 1, 2)], fontsize=7.5, color=FAINT)
        top.tick_params(length=0)
        top.spines["top"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    def _count(leg):
        return (f"{leg['max_votes']} of {leg['n_of']} over"
                + (f" on {_day_month(pd.Timestamp(leg['max_votes_date']))}" if leg["max_votes_date"] else "")
                + f", {leg['n_req']} needed")
    title = f"{cfg.RIVER_TITLE[river]}"
    if result_window is not None:
        if product == "glofas":   # this panel carries both legs
            title += f" · activation: {_count(result_window['action'])} · readiness: {_count(result_window['readiness'])}"
        else:
            title += f" · {role}: {_count(result_window['action'])}"
    ax.set_title(title, color=PRODUCT_COLORS[product], pad=8)
    if missing:
        ax.text(0.995, 0.03, "not in live feed: " + ", ".join(missing), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#B34036")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False, ncol=4)


def monitoring_chart(df, result, levels_df=None):
    """One panel per source the open season actually uses. Deyr runs on GloFAS
    alone (action 1-7 d and readiness 8-12 d), so the two rivers sit side by
    side; Gu adds the Google action panel, so rivers become rows."""
    levels_df = thr.load() if levels_df is None else levels_df
    monitoring_date = pd.Timestamp(result["monitoring_date"]).date()
    season, is_open = season_shown(monitoring_date)
    rivers = ("juba", "shabelle")
    two_sources = any(cfg.ACTION_RULES[(rv, season)]["source"] == "google" for rv in rivers)
    if two_sources:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), dpi=150)
        ax_of, left = (lambda i, j: axes[i, j]), [axes[0, 0], axes[1, 0]]
        head, rect = (0.975, 0.945), (0, 0.03, 1, 0.93)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150)
        ax_of, left = (lambda i, j: axes[i]), [axes[0]]
        head, rect = (0.965, 0.915), (0, 0.04, 1, 0.87)
    for i, river in enumerate(rivers):
        w = (river, season)
        key = cfg.WINDOW_KEY[w]
        stations = TRIGGER_STATIONS[river]
        a = cfg.ACTION_RULES[w]
        r = cfg.READINESS_RULES[w]
        col = 0
        if a["source"] == "google":
            g_levels = thr.lookup(levels_df, "google_grrr", season, a["rp"], stations)
            _panel(ax_of(i, 0), df, "google", river, season, g_levels, result["windows"][key], "activation")
            col = 1
        if a["source"] == "glofas":
            gl_levels = thr.lookup(levels_df, cfg.GLOFAS_OPERATIONAL, season, a["rp"], stations)
            role = "activation"
        else:
            gl_levels = thr.lookup(levels_df, cfg.GLOFAS_OPERATIONAL, season, r["rp"], stations)
            role = "readiness"
        _panel(ax_of(i, col), df, "glofas", river, season, gl_levels, result["windows"][key], role)
    # one y scale for every panel, with headroom for the band labels above the highest line
    top = max(ax.get_ylim()[1] for ax in fig.axes)
    for ax in fig.axes:
        ax.set_ylim(0, top)
    for ax in left:
        ax.set_ylabel("% of the station's trigger level")
    status = result["status"]
    gi = pd.Timestamp(result["glofas_issue"]) if result.get("glofas_issue") else None
    issued = f"GloFAS forecast of {_day_month(gi)} {gi:%Y}" if gi is not None else "no GloFAS forecast"
    if two_sources and result.get("google_issue"):
        go = pd.Timestamp(result["google_issue"][:10])
        issued += f", Google forecast of {_day_month(go)} {go:%Y}"
    fig.text(0.02, head[0], f"Somalia riverine flood trigger · {cfg.SEASON_TITLE[season].split(' (')[0]} season"
                            f"{'' if is_open else ' (not open)'} · {issued}",
             fontsize=11, color=BODY)
    label = fig.text(0.02, head[1], "Status: ", fontsize=11, color=BODY)
    fig.canvas.draw()
    x1 = fig.transFigure.inverted().transform((label.get_window_extent().x1, 0))[0]
    fig.text(x1, head[1], status, fontsize=11, fontweight="bold", color=STATUS_COLORS.get(status, INK))
    v = cfg.GLOFAS_OPERATIONAL.replace("glofas_", "GloFAS ")
    sources = (f"{v} ensemble median; trigger levels from the {v} reanalysis"
               + ("; Google Flood Hub, levels from its retrospective" if two_sources else ""))
    runs = f"Run {result.get('glofas_version') or 'n/a'}" + (f"; Google issue {result.get('google_issue') or 'n/a'}" if two_sources else "")
    fig.text(0.02, 0.012, f"{sources}. {runs}.", fontsize=7.5, color=FAINT, wrap=True)
    fig.tight_layout(rect=rect, h_pad=3.2)
    return fig


def save_chart(fig, monitoring_date):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    data = buf.getvalue()
    stratus.get_container_client("projects", cfg.BLOB_STAGE, write=True).upload_blob(
        name=chart_blob_name(monitoring_date), data=data, overwrite=True)
    return data


def load_chart(monitoring_date):
    from azure.core.exceptions import ResourceNotFoundError

    try:
        return (stratus.get_container_client("projects", cfg.BLOB_STAGE)
                .get_blob_client(chart_blob_name(monitoring_date)).download_blob().readall())
    except ResourceNotFoundError:
        return None
