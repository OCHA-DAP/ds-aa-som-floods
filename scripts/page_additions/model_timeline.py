"""Year by year: how far ahead of the flood Google and GloFAS would have raised a
flag. For every flood season (two gauges over 1-in-3, 1999-2023) and window rule:
the first day each model's reanalysis crossed (flow date), the first forecast
issue that crossed (GloFAS v4 reforecast 2003-2023, Google reforecast 2016-2023),
all relative to the gauges' onset. Writes figs/l_model_timeline.png and
model_timeline.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import somlib as L
from src.constants import TRIGGER_CONFIG

S = Path(__file__).parent
PAGE_DIR = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model"
FIGS = PAGE_DIR / "figs"
C_G, C_V5, C_V4 = "#1d4ed8", "#0f766e", "#6b7280"
XMIN, XMAX = -28, 21


def lead(onset, d):
    return None if d is None else int((onset - d).days)   # positive = before onset


def phrase(name, n, never="never"):
    if n is None:
        return f"{name} {never}"
    if n == 0:
        return f"{name} on the day"
    return f"{name} {n} d before" if n > 0 else f"{name} {-n} d after"


rows = []
for river, season in L.WINDOWS:
    cfg = TRIGGER_CONFIG[(river, season)]
    flood, severe, ok = L.benchmark(river, season)
    onset3 = L.gauge_crossings(river, season, 3)
    onset5 = L.gauge_crossings(river, season, 5)
    rean = {m: L.first_crossing_dates(m, river, season, cfg["rp"], cfg["n_req"]) for m in ("google_grrr", "glofas_v5")}
    iss = {"glofas_v4": L.first_issue_dates("glofas_v4", river, season, cfg["rp"], cfg["n_req"]),
           "google_grrr": L.first_issue_dates("google_grrr", river, season, cfg["rp"], cfg["n_req"])}
    for y in sorted(flood):
        o = onset3[y]
        r = {"window": L.wname(river, season), "river": river, "season": season, "year": y,
             "severe": y in severe, "onset3": o.date().isoformat(),
             "onset5_lead": lead(o, onset5.get(y)),
             "google_rean": lead(o, rean["google_grrr"].get(y)), "glofas_v5_rean": lead(o, rean["glofas_v5"].get(y)),
             "glofas_v4_issue": lead(o, iss["glofas_v4"][y][0]) if y in iss["glofas_v4"] else None,
             "google_issue": lead(o, iss["google_grrr"][y][0]) if y in iss["google_grrr"] else None,
             "v4_archive": y >= 2003, "google_archive": 2016 <= y <= 2023}
        rows.append(r)
        print(f"{r['window']:14s} {y} sev={int(r['severe'])} | Google rean {r['google_rean']} v5 rean {r['glofas_v5_rean']} "
              f"| v4 issue {r['glofas_v4_issue']} Google issue {r['google_issue']} | 1-in-5 {r['onset5_lead']}")
json.dump(rows, open(S / "model_timeline.json", "w"), indent=1)

# ---- figure --------------------------------------------------------------------
deyr = [r for r in rows if r["season"] == "deyr"]
gu = [r for r in rows if r["season"] == "gu"]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.2, 11.2), gridspec_kw={"hspace": .3, "height_ratios": [len(deyr), len(gu)]})


def panel(ax, rs, title):
    ys = list(range(len(rs)))[::-1]
    ax.axvspan(-7.5, -0.5, color="#dcfce7", alpha=.9, zorder=0)
    ax.axvspan(-12.5, -7.5, color="#f0fdf4", alpha=.9, zorder=0)
    ax.axvline(0, color="#374151", lw=1.2, zorder=1)
    for y, r in zip(ys, rs):
        if r["onset5_lead"] is not None:
            xg = -r["onset5_lead"]
            if xg > XMAX - .4:
                ax.annotate(f"1-in-5 {-r['onset5_lead']} d after", (XMAX - .4, y), xytext=(-3, 7), textcoords="offset points",
                            fontsize=7, color="#111827", ha="right")
                xg = XMAX - .4
            ax.plot([xg], [y], marker="|", color="#111827", ms=13, mew=2.2, zorder=2)
        for key, c, mk, dy, ms in (("google_rean", C_G, "D", .17, 7), ("glofas_v5_rean", C_V5, "D", -.17, 7),
                                   ("google_issue", C_G, "^", .17, 8.5), ("glofas_v4_issue", C_V4, "^", -.17, 8.5)):
            v = r[key]
            if v is None:
                continue
            x = max(min(-v, XMAX - .4), XMIN + .4)
            ax.plot([x], [y + dy], marker=mk, color=c, ms=ms, mec="white", mew=.8, zorder=5, clip_on=False)
            if -v < XMIN or -v > XMAX:
                ax.annotate(f"{abs(v)} d {'before' if v > 0 else 'after'}", (x, y + dy), xytext=(0, -12 if dy < 0 else 6),
                            textcoords="offset points", fontsize=7, color=c, ha="center")
        t1 = f"{phrase('Google', r['google_rean'])}  |  {phrase('GloFAS v5', r['glofas_v5_rean'])}"
        t2 = (f"issues: {phrase('Google', r['google_issue'], 'none' if r['google_archive'] else 'no archive')}  |  "
              f"{phrase('GloFAS v4', r['glofas_v4_issue'], 'none' if r['v4_archive'] else 'no archive')}")
        ax.text(XMAX + .6, y + .17, t1, va="center", fontsize=8.6, color="#111827")
        ax.text(XMAX + .6, y - .2, t2, va="center", fontsize=8, color="#6b7280")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['window']} {r['year']}" + (" *" if r["severe"] else "") for r in rs], fontsize=9)
    ax.set_ylim(-.7, len(rs) - .3)
    ax.set_xlim(XMIN, XMAX)
    ticks = list(range(-28, 22, 7))
    ax.set_xticks(ticks)
    ax.set_xticklabels([("onset" if t == 0 else f"{-t} d before" if t < 0 else f"{t} d after") for t in ticks], fontsize=8.5)
    ax.grid(axis="x", color="#e5e7eb", lw=.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#111827", pad=6)


panel(a1, deyr, "Deyr flood seasons (* severe: two gauges over 1-in-5)")
panel(a2, gu, "Gu flood seasons")
handles = [
    Line2D([], [], marker="D", color=C_G, ls="none", ms=7, label="Google reanalysis: first day the window rule crossed"),
    Line2D([], [], marker="D", color=C_V5, ls="none", ms=7, label="GloFAS v5 reanalysis: first day the window rule crossed"),
    Line2D([], [], marker="^", color=C_G, ls="none", ms=8.5, label="Google forecast: first issue that crossed (2016 to 2023)"),
    Line2D([], [], marker="^", color=C_V4, ls="none", ms=8.5, label="GloFAS v4 forecast: first issue that crossed (2003 to 2023)"),
    Line2D([], [], marker="|", color="#111827", ls="none", ms=13, mew=2.2, label="gauges: two over 1-in-5"),
]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.14, .995), ncol=2, frameon=False, fontsize=8.6)
fig.subplots_adjust(left=.14, right=.6, top=.9, bottom=.045)
fig.savefig(FIGS / "l_model_timeline.png", dpi=150)
print("wrote", FIGS / "l_model_timeline.png", len(rows), "rows")
