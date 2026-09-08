"""SWALIM against the models, station by station. One row per station and season:
the first bulletin on which SWALIM reported that gauge at each of its own official
levels (moderate, high, bank full), the gauge's own return-period crossings, and the
first day each model crossed that station's own threshold.
Writes figs/k_swalim_station_<river>.png and swalim_station_timeline.json."""
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import somlib as L
from src.constants import TRIGGER_CONFIG

S = Path(__file__).parent
PAGE_DIR = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model"
FIGS = PAGE_DIR / "figs"
REC = pd.read_csv(S / "swalim_station_records.csv", parse_dates=["date"])
REC["season"] = REC.date.dt.month.map(lambda m: "gu" if m in (3, 4, 5, 6) else ("deyr" if m in (9, 10, 11, 12) else None))
REC["year"] = REC.date.dt.year
TH = L.swalim_thresholds()
C_SW, C_SWD, C_G, C_V5, C_V4, C_GAUGE = "#d97706", "#92400e", "#1d4ed8", "#0f766e", "#6b7280", "#9ca3af"
AXIS = {"deyr": (date(2001, 9, 1), date(2001, 12, 15)), "gu": (date(2001, 3, 15), date(2001, 6, 10))}
MONTHS = {"deyr": [(9, "Sep"), (10, "Oct"), (11, "Nov"), (12, "Dec")], "gu": [(4, "Apr"), (5, "May"), (6, "Jun")]}
LEVEL_NAME = {1: "moderate", 2: "high", 3: "bank full"}


def doy(d):
    d = pd.Timestamp(d)
    return date(2001, d.month, d.day).toordinal() - date(2001, 1, 1).toordinal()


def fmt(d):
    return pd.Timestamp(d).strftime("%d %b").lstrip("0") if d is not None else None


def swalim_steps(st, season, year):
    """{level class: (first bulletin date, reading, from a published reading?)}"""
    g = REC[(REC.station == st) & (REC.season == season) & (REC.year == year)].sort_values("date")
    out = {}
    for lvl in (1, 2, 3):
        hit = g[g.level_class >= lvl]
        if len(hit):
            r = hit.iloc[0]
            out[lvl] = (r["date"], (None if pd.isna(r["reading_m"]) else float(r["reading_m"])), bool(r["from_reading"]))
    return out


def gauge_cross(st, river, season, year, rp):
    lv = L.levels()
    s = lv[(lv.station == st) & (lv.date.dt.year == year) & (lv.date.dt.month.isin(L.SEASONS[season]))]
    s = s.set_index("date")["level_m"].dropna()
    lev = L.gauge_levels(river, season, rp)[st]
    if pd.isna(lev) or not len(s):
        return None
    o = s[s >= lev]
    return o.index[0] if len(o) else None


def model_cross(model, st, season, year, rp):
    s = L.season_series(model, st, season)
    s = s[s.index.year == year]
    thr = L.model_threshold(model, st, season, rp)
    if pd.isna(thr) or not len(s):
        return None
    o = s[s >= thr]
    return o.index[0] if len(o) else None


def v4_issue_cross(st, season, year, rp):
    rf = L.reforecast("glofas_v4")
    rf = rf[(rf.station == st) & rf.leadtime_days.between(1, 7)
            & (rf.valid_time.dt.year == year) & rf.valid_time.dt.month.isin(L.SEASONS[season])]
    if not len(rf):
        return None
    med = rf.groupby(["issued_time", "valid_time"]).discharge.median()
    thr = L.model_threshold("glofas_v4", st, season, rp)
    if pd.isna(thr):
        return None
    hit = med[med >= thr]
    return hit.index.get_level_values("issued_time").min() if len(hit) else None


def lag_phrase(first, g3):
    if g3 is None:
        return "gauge never over 1-in-3"
    n = (pd.Timestamp(g3) - pd.Timestamp(first)).days
    if n == 0:
        return f"gauge 1-in-3 {fmt(g3)}, same day"
    word = "later" if n > 0 else "earlier"
    return f"gauge 1-in-3 {fmt(g3)}, {abs(n)} d {word}"


records = []
for river in ("juba", "shabelle"):
    stations = L.TRIGGER_STATIONS[river]
    have = REC[REC.station.isin(stations) & REC.season.notna()][["season", "year"]].drop_duplicates()
    keys = sorted(have.itertuples(index=False), key=lambda k: ({"deyr": 0, "gu": 1}[k.season], k.year))
    rows = []
    for season, year in [(k.season, k.year) for k in keys]:
        rp = TRIGGER_CONFIG[(river, season)]["rp"]
        for st in stations:
            rows.append({
                "river": river, "season": season, "year": year, "station": st, "rp": rp,
                "swalim": swalim_steps(st, season, year),
                "gauge3": gauge_cross(st, river, season, year, 3),
                "gauge5": gauge_cross(st, river, season, year, 5),
                "google": model_cross("google_grrr", st, season, year, rp),
                "glofas_v5": model_cross("glofas_v5", st, season, year, rp),
                "v4_issue": v4_issue_cross(st, season, year, rp),
            })
    records += rows

    # ---- figure: one panel per season ------------------------------------------
    by_season = {sn: [r for r in rows if r["season"] == sn] for sn in ("deyr", "gu")}
    by_season = {k: v for k, v in by_season.items() if v}
    heights = [len(v) for v in by_season.values()]
    fig, axes = plt.subplots(len(by_season), 1, figsize=(14.2, .40 * sum(heights) + 2.6),
                             gridspec_kw={"hspace": .16, "height_ratios": heights})
    axes = [axes] if len(by_season) == 1 else list(axes)
    for ax, (season, srows) in zip(axes, by_season.items()):
        x0, x1 = doy(AXIS[season][0]), doy(AXIS[season][1])
        n = len(srows)
        ys = list(range(n))[::-1]
        prev_year = None
        for y, r in zip(ys, srows):
            if prev_year is not None and r["year"] != prev_year:
                ax.axhline(y + .5, color="#cbd5e1", lw=1.1, zorder=1)
            prev_year = r["year"]
            if y % 2 == 0:
                ax.axhspan(y - .5, y + .5, color="#f8fafc", zorder=0)
            no_record = r["year"] > 2023
            g3, g5 = r["gauge3"], r["gauge5"]
            if g3 is not None or g5 is not None:
                a = doy(g3) if g3 is not None else doy(g5)
                b = doy(g5) if g5 is not None else doy(g3)
                ax.plot([a, b], [y, y], color=C_GAUGE, lw=8, solid_capstyle="butt", alpha=.45, zorder=2)
                if g3 is not None:
                    ax.plot([doy(g3)], [y], marker="|", color="#4b5563", ms=12, mew=1.5, zorder=3)
                if g5 is not None:
                    ax.plot([doy(g5)], [y], marker="|", color="#111827", ms=12, mew=2.2, zorder=3)
            sw = r["swalim"]
            if sw:
                xs = [doy(v[0]) for v in sw.values()]
                ax.plot([min(xs), max(xs)], [y + .2, y + .2], color=C_SW, lw=1.1, zorder=3)
                for lvl, (d, val, from_read) in sw.items():
                    mk, ms, col = {1: ("o", 5.5, C_SW), 2: ("o", 7.5, C_SW), 3: ("s", 7.5, C_SWD)}[lvl]
                    ax.plot(doy(d), y + .2, marker=mk, ms=ms, mfc=(col if from_read else "white"),
                            mec=col, mew=1.3, alpha=(.7 if lvl == 1 else 1), zorder=5)
            for k2, c, mk, ms in (("google", C_G, "D", 6.5), ("glofas_v5", C_V5, "D", 6.5), ("v4_issue", C_V4, "^", 8)):
                d = r[k2]
                if d is not None:
                    ax.plot([doy(d)], [y - .2], marker=mk, color=c, ms=ms, mec="white", mew=.7, zorder=5)
            first = min(sw.values(), key=lambda v: v[0])[0] if sw else None
            if first is not None:
                seen, parts = set(), []
                for lvl, v in sorted(sw.items()):
                    if v[0] in seen:          # one bulletin reported several steps at once
                        parts[-1] = f"{LEVEL_NAME[lvl]} {fmt(v[0])}"
                    else:
                        parts.append(f"{LEVEL_NAME[lvl]} {fmt(v[0])}")
                        seen.add(v[0])
                t1 = f"SWALIM {', '.join(parts)}; {lag_phrase(first, g3)}"
            else:
                t1 = "no station bulletin" + ("" if g3 is None else f"; gauge 1-in-3 {fmt(g3)}")
            miss = "no record" if no_record else "never"
            t2 = (f"Google {fmt(r['google']) or miss} | GloFAS v5 {fmt(r['glofas_v5']) or miss} "
                  f"| v4 issue {fmt(r['v4_issue']) or ('no archive' if no_record else 'never')}")
            ax.text(x1 + 1.2, y + .18, t1, va="center", fontsize=8.4, color="#111827")
            ax.text(x1 + 1.2, y - .2, t2, va="center", fontsize=7.8, color="#6b7280")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r['year']}  {L.NAME[r['station']]}" for r in srows], fontsize=8.8)
        ax.set_ylim(-.5, n - .5)
        ax.set_xlim(x0, x1)
        ticks = [(doy(date(2001, m, dd)), f"{dd} {nm}") for m, nm in MONTHS[season] for dd in (1, 15)
                 if x0 <= doy(date(2001, m, dd)) <= x1]
        ax.set_xticks([t for t, _ in ticks])
        ax.set_xticklabels([nm for _, nm in ticks], fontsize=8.5)
        ax.grid(axis="x", color="#e5e7eb", lw=.8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#d1d5db")
        ax.tick_params(length=0)
        ax.set_title(f"{season.title()} seasons: model thresholds are each station's own "
                     f"1-in-{TRIGGER_CONFIG[(river, season)]['rp']}",
                     loc="left", fontsize=10, fontweight="bold", color="#111827", pad=6)
    handles = [
        Line2D([], [], marker="o", color=C_SW, ls="none", ms=5.5, alpha=.7, label="SWALIM: gauge at its moderate level"),
        Line2D([], [], marker="o", color=C_SW, ls="none", ms=7.5, label="SWALIM: at its high level"),
        Line2D([], [], marker="s", color=C_SWD, ls="none", ms=7.5, label="SWALIM: at bank full or overflowing"),
        Line2D([], [], marker="o", mfc="white", mec=C_SW, color="none", ms=7.5, mew=1.3, label="hollow: risk stated, no reading published"),
        Line2D([], [], marker="D", color=C_G, ls="none", ms=6.5, label="Google reanalysis over this station's threshold"),
        Line2D([], [], marker="D", color=C_V5, ls="none", ms=6.5, label="GloFAS v5 reanalysis over it"),
        Line2D([], [], marker="^", color=C_V4, ls="none", ms=8, label="GloFAS v4 forecast: first issue over it"),
        Patch(color=C_GAUGE, alpha=.45, label="this gauge: its own 1-in-3 (thin tick) to 1-in-5 (thick tick)"),
    ]
    fig.suptitle(f"{river.title()}: SWALIM's reported levels against the models, station by station",
                 x=.105, ha="left", fontsize=11.5, fontweight="bold", color="#111827",
                 y=1 - .28 / fig.get_figheight())
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.105, 1 - .55 / fig.get_figheight()),
               ncol=2, frameon=False, fontsize=8.4, columnspacing=1.6, handletextpad=.6)
    fig.subplots_adjust(left=.105, right=.575, top=1 - (1.95 / fig.get_figheight()), bottom=.05)
    out = FIGS / f"k_swalim_station_{river}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out.name, f"({len(rows)} rows)")

payload = []
for r in records:
    d = dict(r)
    d["swalim"] = {str(k): [str(pd.Timestamp(v[0]).date()), v[1], v[2]] for k, v in r["swalim"].items()}
    for k in ("gauge3", "gauge5", "google", "glofas_v5", "v4_issue"):
        d[k] = None if r[k] is None else str(pd.Timestamp(r[k]).date())
    payload.append(d)
json.dump(payload, open(S / "swalim_station_timeline.json", "w"), indent=1)
print("rows:", len(records))
