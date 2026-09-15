"""Daily monitoring chart: every trigger point as a share of its own level.

Two rows (Juba, Shabelle) by two columns (Google Flood Hub, GloFAS ensemble
median). Each line is one point, expressed as a percentage of that point's
threshold for the season being watched, so points with very different
discharges share one axis and 100% is the level that counts as a vote. The
action lead band is shaded darker, the readiness band lighter. Palette and
type follow src.constants so the chart matches the analysis pages.
"""

import io
from datetime import timedelta

import matplotlib
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
STATUS_COLORS = {"ACTION TRIGGER REACHED": "#B34036", "READINESS TRIGGER REACHED": "#D48F2A",
                 "NOT ACTIVATED": "#2F9E6F", "NO WINDOW OPEN": FAINT}
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
    if issue is not None:
        if product == "glofas":
            a0, a1 = issue + pd.Timedelta(days=cfg.ACTION_LEADS[0] - 1), issue + pd.Timedelta(days=cfg.ACTION_LEADS[1] - 1)
            r0, r1 = issue + pd.Timedelta(days=cfg.READINESS_LEADS[0] - 1), issue + pd.Timedelta(days=cfg.READINESS_LEADS[1] - 1)
            ax.axvspan(r0 - pd.Timedelta(hours=12), r1 + pd.Timedelta(hours=12), color="#F1F4F7", zorder=0)
            ax.text(r0, 0.97, "readiness 8–12 d", transform=ax.get_xaxis_transform(), fontsize=8,
                    color=FAINT, va="top")
        else:
            a0, a1 = issue + pd.Timedelta(days=cfg.ACTION_LEADS[0]), issue + pd.Timedelta(days=cfg.ACTION_LEADS[1])
        ax.axvspan(a0 - pd.Timedelta(hours=12), a1 + pd.Timedelta(hours=12), color="#E6EEF7", zorder=0)
        ax.text(a0, 0.97, "action 1–7 d", transform=ax.get_xaxis_transform(), fontsize=8,
                color=FAINT, va="top")
    for st in stations:
        s = sub[sub.station == st].sort_values("valid_date")
        lvl = levels.get(st)
        if s.empty or not lvl:
            continue
        pct = 100 * s["value"] / lvl
        exceeds = (pct >= 100).any()
        ax.plot(s["valid_date"], pct, color=STATION_COLORS[st], lw=2.2 if exceeds else 1.6,
                marker="o", ms=3, label=f"{cfg.STATION_TITLE[st]} ({lvl:,.0f} m³/s)", zorder=3)
    live = [st for st in stations if not sub[sub.station == st].empty]
    missing = [cfg.STATION_TITLE[st] for st in stations if st not in live]
    ax.axhline(100, color="#B34036", ls="--", lw=1.2, zorder=2)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=max(ax.get_ylim()[1], 120))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: _day_month(mdates.num2date(x))))
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    votes = ""
    if result_window is not None:
        leg = result_window["action"] if role == "action" else result_window["readiness"]
        votes = (f"  ·  max {leg['max_votes']} of {leg['n_of']} points over"
                 + (f" on {_day_month(pd.Timestamp(leg['max_votes_date']))}" if leg["max_votes_date"] else "")
                 + f" (rule: {leg['n_req']} of {leg['n_of']})")
    title = f"{cfg.RIVER_TITLE[river]} · {cfg.SOURCE_TITLE[product]} — {role}{votes}"
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
            _panel(ax_of(i, 0), df, "google", river, season, g_levels, result["windows"][key], "action")
            col = 1
        if a["source"] == "glofas":
            gl_levels = thr.lookup(levels_df, cfg.GLOFAS_OPERATIONAL, season, a["rp"], stations)
            role = "action"
        else:
            gl_levels = thr.lookup(levels_df, cfg.GLOFAS_OPERATIONAL, season, r["rp"], stations, basis="readiness_band")
            role = "readiness"
        _panel(ax_of(i, col), df, "glofas", river, season, gl_levels, result["windows"][key], role)
    for ax in left:
        ax.set_ylabel("% of the point's threshold")
    status = result["status"]
    fig.text(0.02, head[0], f"Somalia riverine flood trigger · forecasts retrieved {_day_month(monitoring_date)} {monitoring_date:%Y} · "
                            f"{cfg.SEASON_TITLE[season]}{'' if is_open else ' (out of season)'}",
             fontsize=11, color=BODY)
    label = fig.text(0.02, head[1], "Status: ", fontsize=11, color=BODY)
    fig.canvas.draw()
    x1 = fig.transFigure.inverted().transform((label.get_window_extent().x1, 0))[0]
    fig.text(x1, head[1], status, fontsize=11, fontweight="bold", color=STATUS_COLORS.get(status, INK))
    sources = (f"GloFAS: ensemble median, levels from the {cfg.GLOFAS_OPERATIONAL.replace('_', ' ')} climatology"
               + ("; Google: Flood Hub deterministic, levels from its retrospective" if two_sources else ""))
    runs = f"GloFAS run: {result.get('glofas_version') or 'n/a'}" + (f"; Google issue: {result.get('google_issue') or 'n/a'}" if two_sources else "")
    fig.text(0.02, 0.012, f"Lines: forecast at each trigger point as % of its own threshold ({sources}). "
                          f"Dashed line = vote level. {runs}.",
             fontsize=7.5, color=FAINT, wrap=True)
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
