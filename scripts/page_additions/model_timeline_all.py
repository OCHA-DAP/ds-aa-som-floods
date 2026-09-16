"""Every year, every window: when each model would have flagged with the window's
point count over 1-in-3 station levels (fitted on each model's own reanalysis), on
absolute dates within the season. Diamonds: first day the reanalysis crossed (Google,
GloFAS v5). Triangles: first forecast issue that crossed at leads 1-7 (Google 2016-2023,
GloFAS v4 2003-2023). Gauges: two-gauge 1-in-3 (thin tick) to 1-in-5 (thick tick).
Writes figs/l_years_<window>.png and model_timeline_all.json."""
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
RP = 3
C_G, C_V5, C_V4, C_GAUGE = "#1d4ed8", "#0f766e", "#6b7280", "#9ca3af"
AXIS = {"deyr": (date(2001, 9, 25), date(2001, 12, 31)), "gu": (date(2001, 3, 1), date(2001, 6, 10))}
MONTHS = {"deyr": [(10, "Oct"), (11, "Nov"), (12, "Dec")], "gu": [(3, "Mar"), (4, "Apr"), (5, "May"), (6, "Jun")]}


def doy(d):
    return date(2001, d.month, d.day).toordinal() - date(2001, 1, 1).toordinal()


def fmt(d):
    return pd.Timestamp(d).strftime("%d %b").lstrip("0")


def lead_txt(name, onset, d):
    if d is None:
        return f"{name} never"
    if onset is None:
        return f"{name} {fmt(d)}"
    n = (onset - d).days
    return f"{name} on the day" if n == 0 else (f"{name} {n} d before" if n > 0 else f"{name} {-n} d after")


rows_all = []
for river, season in L.WINDOWS:
    cfg = TRIGGER_CONFIG[(river, season)]
    n_req = cfg["n_req"]
    flood, severe, ok = L.benchmark(river, season)
    onset3 = L.gauge_crossings(river, season, 3)
    onset5 = L.gauge_crossings(river, season, 5)
    rean = {m: L.first_crossing_dates(m, river, season, RP, n_req) for m in ("google_grrr", "glofas_v5")}
    iss = {m: L.first_issue_dates(m, river, season, RP, n_req) for m in ("glofas_v4", "google_grrr")}
    years = list(range(L.Y0, L.Y1 + 1))
    rows = []
    for y in years:
        o = onset3.get(y)
        r = {"window": L.wname(river, season), "river": river, "season": season, "year": y,
             "gauge_flood": y in flood, "severe": y in severe, "gauges_reporting": y in ok,
             "onset3": o, "onset5": onset5.get(y),
             "google_rean": rean["google_grrr"].get(y), "glofas_v5_rean": rean["glofas_v5"].get(y),
             "google_issue": iss["google_grrr"][y][0] if y in iss["google_grrr"] else None,
             "glofas_v4_issue": iss["glofas_v4"][y][0] if y in iss["glofas_v4"] else None,
             "v4_archive": y >= 2003, "google_archive": 2016 <= y <= 2023}
        rows.append(r)
    rows_all += rows

    # ---- figure for this window -------------------------------------------------
    x0, x1 = doy(AXIS[season][0]), doy(AXIS[season][1])
    fig, ax = plt.subplots(figsize=(12.5, 10.2))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        if y % 2 == 0:
            ax.axhspan(y - .5, y + .5, color="#f8fafc", zorder=0)
        if not r["gauges_reporting"]:
            ax.text((x0 + x1) / 2, y, "gauges not reporting", ha="center", va="center", fontsize=8, color="#9ca3af", style="italic")
        o3, o5 = r["onset3"], r["onset5"]
        if o3 or o5:
            a = doy(o3) if o3 else doy(o5)
            b = doy(o5) if o5 else doy(o3)
            ax.plot([a, b], [y, y], color=C_GAUGE, lw=9, solid_capstyle="butt", alpha=.45, zorder=1)
            if o3:
                ax.plot([doy(o3)], [y], marker="|", color="#4b5563", ms=14, mew=1.6, zorder=2)
            if o5:
                ax.plot([doy(o5)], [y], marker="|", color="#111827", ms=14, mew=2.4, zorder=2)
        for key, c, mk, dy, ms in (("google_rean", C_G, "D", .2, 7.5), ("glofas_v5_rean", C_V5, "D", -.2, 7.5),
                                   ("google_issue", C_G, "^", .2, 9), ("glofas_v4_issue", C_V4, "^", -.2, 9)):
            d = r[key]
            if d is not None and x0 <= doy(d) <= x1:
                ax.plot([doy(d)], [y + dy], marker=mk, color=c, ms=ms, mec="white", mew=.8, zorder=5)
        # verdicts
        any_issue = r["google_issue"] is not None or r["glofas_v4_issue"] is not None
        if r["gauge_flood"]:
            t1 = f"{lead_txt('Google', o3, r['google_rean'])}  |  {lead_txt('GloFAS v5', o3, r['glofas_v5_rean'])}"
            gi = lead_txt("Google", o3, r["google_issue"]) if r["google_archive"] else "Google: no archive"
            vi = lead_txt("GloFAS v4", o3, r["glofas_v4_issue"]) if r["v4_archive"] else "GloFAS v4: no archive"
            t2 = f"forecast issues: {gi}  |  {vi}"
        else:
            flags = [f"{n} {fmt(r[k])}" for n, k in (("Google", "google_rean"), ("GloFAS v5", "glofas_v5_rean")) if r[k] is not None]
            base = "no gauge flood" if r["gauges_reporting"] else "gauges not reporting"
            t1 = base + ("; flagged by " + ", ".join(flags) if flags else ("" if any_issue else ", no flags"))
            iss = [f"{n} {fmt(r[k])}" for n, k in (("Google", "google_issue"), ("GloFAS v4", "glofas_v4_issue")) if r[k] is not None]
            t2 = ("forecast issues: " + ", ".join(iss)) if iss else ""
        if t2:
            ax.text(x1 + 1.5, y + .17, t1, va="center", fontsize=9.2, color="#111827" if r["gauge_flood"] else "#6b7280")
            ax.text(x1 + 1.5, y - .2, t2, va="center", fontsize=8.4, color="#6b7280")
        else:
            ax.text(x1 + 1.5, y, t1, va="center", fontsize=9.2, color="#111827" if r["gauge_flood"] else "#9ca3af")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['year']}{' *' if r['severe'] else ''}" for r in rows], fontsize=9.5)
    ax.set_ylim(-.5, len(rows) - .5)
    ax.set_xlim(x0, x1)
    ticks = []
    for m, n in MONTHS[season]:
        for dd, lab in ((1, f"1 {n}"), (15, f"15 {n}")):
            v = doy(date(2001, m, dd))
            if x0 <= v <= x1:
                ticks.append((v, lab))
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([n for _, n in ticks], fontsize=9)
    ax.grid(axis="x", color="#e5e7eb", lw=.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(length=0)
    ax.set_title(f"{L.wname(river, season)}: {n_req} of {len(L.TRIGGER_STATIONS[river])} points over their 1-in-3 level "
                 f"(adopted rule uses 1-in-{cfg['rp']}); * severe year", loc="left", fontsize=11, fontweight="bold", color="#111827", pad=8)
    handles = [
        Line2D([], [], marker="D", color=C_G, ls="none", ms=7.5, label="Google reanalysis: first day crossed"),
        Line2D([], [], marker="D", color=C_V5, ls="none", ms=7.5, label="GloFAS v5 reanalysis: first day crossed"),
        Line2D([], [], marker="^", color=C_G, ls="none", ms=9, label="Google forecast: first issue crossed (2016 to 2023)"),
        Line2D([], [], marker="^", color=C_V4, ls="none", ms=9, label="GloFAS v4 forecast: first issue crossed (2003 to 2023)"),
        Patch(color=C_GAUGE, alpha=.45, label="gauges: two over 1-in-3 (thin tick) to two over 1-in-5 (thick tick)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.08, .995), ncol=3, frameon=False, fontsize=8.8, columnspacing=1.6)
    fig.subplots_adjust(left=.08, right=.6, top=.885, bottom=.06)
    out = FIGS / f"l_years_{season}_{river}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out.name)

json.dump([{k: (v.date().isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()} for r in rows_all],
          open(S / "model_timeline_all.json", "w"), indent=1)

# ---- summary numbers for the prose -------------------------------------------------
def lead(o, d):
    return None if (o is None or d is None) else (o - d).days


for river, season in L.WINDOWS:
    rs = [r for r in rows_all if r["river"] == river and r["season"] == season]
    fl = [r for r in rs if r["gauge_flood"]]
    nf = [r for r in rs if not r["gauge_flood"] and r["gauges_reporting"]]
    for name, key in (("Google", "google_rean"), ("GloFAS v5", "glofas_v5_rean")):
        leads = [lead(r["onset3"], r[key]) for r in fl]
        caught = [x for x in leads if x is not None]
        early = [x for x in caught if x >= 1]
        fa = sum(r[key] is not None for r in nf)
        print(f"{L.wname(river, season):14s} {name:10s} floods {len(fl)} | crossed {len(caught)} | before onset {len(early)} "
              f"| leads {sorted(caught)} | flagged in {fa} of {len(nf)} non-flood years")
    for name, key, arch in (("Google issue", "google_issue", "google_archive"), ("v4 issue", "glofas_v4_issue", "v4_archive")):
        fl_a = [r for r in fl if r[arch]]
        leads = [lead(r["onset3"], r[key]) for r in fl_a]
        caught = [x for x in leads if x is not None]
        nf_a = [r for r in nf if r[arch]]
        fa = sum(r[key] is not None for r in nf_a)
        print(f"{'':14s} {name:12s} floods with archive {len(fl_a)} | crossed {len(caught)} | before onset {len([x for x in caught if x >= 1])} "
              f"| leads {sorted(caught)} | flagged in {fa} of {len(nf_a)} non-flood years")
