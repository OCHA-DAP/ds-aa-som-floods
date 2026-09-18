"""Build pages/glofas-version/: what the GloFAS version switch does to the trigger.

The Deyr windows were designed on the GloFAS v5 reanalysis on the assumption
that v5 was the live system. On 2026-09-14 the EWDS catalogue and ECMWF's
release wiki showed the operational forecast is still v4 (v4.5, April 2026;
v5 pre-operational). The monitoring pipeline therefore runs on v4-fitted
thresholds. This page shows both threshold sets, how far apart they sit,
what the Deyr windows and the envelope look like on each version's own
record, and what would happen if one version's levels were applied to the
other's flows. Everything is computed here from data/processed/.

    .venv/bin/python scripts/build_glofas_version_page.py
"""

import html
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.constants import (BODY, C_GLOFAS4, C_GLOFAS5, C_GOOGLE, FAINT, GRID, INK,  # noqa: E402
                           SEASONS, SEVERE_RP, TRIGGER_CONFIG, TRIGGER_STATIONS,
                           TRIGGER_YEARS)
from src.monitoring import config as cfg  # noqa: E402
from src.monitoring import thresholds as thr  # noqa: E402
from src.utils import weibull_level  # noqa: E402

OUT = REPO / "pages" / "glofas-version"
FIGS = OUT / "figs"
P = REPO / "data" / "processed"
Y0, Y1 = TRIGGER_YEARS
SPAN = set(range(Y0, Y1 + 1))
ST = cfg.STATION_TITLE
VERSIONS = {"glofas_v4": "GloFAS v4", "glofas_v5": "GloFAS v5"}
COL = {"glofas_v4": C_GLOFAS4, "glofas_v5": C_GLOFAS5, "google_grrr": C_GOOGLE}

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"],
                     "axes.edgecolor": GRID, "axes.labelcolor": BODY, "xtick.color": FAINT,
                     "ytick.color": FAINT, "text.color": INK, "axes.titlelocation": "left",
                     "axes.titleweight": "bold", "axes.titlesize": 11})


def daily(source):
    d = pd.read_parquet(P / f"discharge_daily_{source}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d


def matrix(d, river, season):
    d = d[d.station.isin(TRIGGER_STATIONS[river]) & d.date.dt.month.isin(SEASONS[season])
          & d.date.dt.year.between(Y0, Y1)]
    return d.pivot_table(index="date", columns="station", values="discharge")


def activation_years(m, levels, n_req):
    cols = [c for c in levels if c in m.columns]
    votes = (m[cols] >= pd.Series(levels)[cols]).sum(axis=1)
    mx = votes.groupby(votes.index.year).max()
    return sorted(set(mx[mx >= n_req].index) & SPAN)


def rp_text(n, years=len(SPAN)):
    """25/8 = 3.125 is printed 1-in-3.2, as on the trigger pages (half up)."""
    from decimal import ROUND_HALF_UP, Decimal
    return f"1-in-{Decimal(years / n).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}" if n else "never"


def gauge_benchmark():
    """Two-gauge flood (RP3) and severe (RP5) years per window, levels fitted 2000-2023."""
    lv = pd.read_parquet(P / "swalim_levels.parquet")
    lv["date"] = pd.to_datetime(lv["date"])
    out = {}
    for w in cfg.WINDOWS:
        river, season = w
        res = {}
        for rp, name in [(3, "flood"), (SEVERE_RP, "severe")]:
            counts = {}
            for st in TRIGGER_STATIONS[river]:
                s = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
                s = s[s.index.month.isin(SEASONS[season]) & (s.index.year >= 2000) & (s.index.year <= Y1)]
                am = s.groupby(s.index.year).max().dropna()
                if not len(am):
                    continue
                level = weibull_level(am.values, rp)
                if np.isnan(level):
                    continue
                for y in set(am[am >= level].index) & SPAN:
                    counts[y] = counts.get(y, 0) + 1
            res[name] = sorted(y for y, n in counts.items() if n >= 2)
        out[w] = res
    return out


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    lv_df = thr.load()
    dd = {v: daily(v) for v in VERSIONS}
    bench = gauge_benchmark()
    flood_all = sorted(set().union(*[set(b["flood"]) for b in bench.values()]))
    severe_all = sorted(set().union(*[set(b["severe"]) for b in bench.values()]))

    # ---------------------------------------------------------------- thresholds
    deyr_rows = []
    for river in ("juba", "shabelle"):
        rp = TRIGGER_CONFIG[(river, "deyr")]["rp"]
        rrp = cfg.READINESS_RULES[(river, "deyr")]["rp"]
        for st in TRIGGER_STATIONS[river]:
            t4 = thr.lookup(lv_df, "glofas_v4", "deyr", rp, [st])[st]
            t5 = thr.lookup(lv_df, "glofas_v5", "deyr", rp, [st])[st]
            tr = thr.lookup(lv_df, "glofas_v4", "deyr", rrp, [st], basis="readiness_band")[st]
            m4 = dd["glofas_v4"][(dd["glofas_v4"].station == st) & dd["glofas_v4"].date.dt.month.isin(SEASONS["deyr"])].discharge
            m5 = dd["glofas_v5"][(dd["glofas_v5"].station == st) & dd["glofas_v5"].date.dt.month.isin(SEASONS["deyr"])].discharge
            deyr_rows.append({"river": river, "station": st, "rp": rp, "v4": t4, "v5": t5, "ratio": t4 / t5,
                              "mean4": m4.mean(), "mean5": m5.mean(), "readiness_rp": rrp, "readiness_band": tr})
    tdf = pd.DataFrame(deyr_rows)

    # Gu readiness levels (GloFAS is readiness-only in Gu) for completeness
    gu_rows = []
    for river in ("juba", "shabelle"):
        rrp = cfg.READINESS_RULES[(river, "gu")]["rp"]
        for st in TRIGGER_STATIONS[river]:
            gu_rows.append({"river": river, "station": st, "rp": rrp,
                            "band": thr.lookup(lv_df, "glofas_v4", "gu", rrp, [st], basis="readiness_band")[st],
                            "v4": thr.lookup(lv_df, "glofas_v4", "gu", rrp, [st])[st],
                            "v5": thr.lookup(lv_df, "glofas_v5", "gu", rrp, [st])[st]})
    gdf = pd.DataFrame(gu_rows)

    # ---------------------------------------------------------------- activations
    acts = {}   # (window, fit_version, flow_version) -> years
    for river in ("juba", "shabelle"):
        w = (river, "deyr")
        rp, n_req = TRIGGER_CONFIG[w]["rp"], TRIGGER_CONFIG[w]["n_req"]
        for fit in VERSIONS:
            levels = thr.lookup(lv_df, fit, "deyr", rp, TRIGGER_STATIONS[river])
            for flow in VERSIONS:
                acts[(w, fit, flow)] = activation_years(matrix(dd[flow], river, "deyr"), levels, n_req)
    # Gu windows on Google, unchanged
    gg = daily("google_grrr")
    gu_acts = {}
    for river in ("juba", "shabelle"):
        w = (river, "gu")
        levels = thr.lookup(lv_df, "google_grrr", "gu", TRIGGER_CONFIG[w]["rp"], TRIGGER_STATIONS[river])
        gu_acts[w] = activation_years(matrix(gg, river, "gu"), levels, TRIGGER_CONFIG[w]["n_req"])
    gu_union = sorted(set(gu_acts[("juba", "gu")]) | set(gu_acts[("shabelle", "gu")]))

    def envelope(version):
        deyr = set(acts[(("juba", "deyr"), version, version)]) | set(acts[(("shabelle", "deyr"), version, version)])
        yrs = sorted(deyr | set(gu_union))
        return {"years": yrs, "n": len(yrs), "rp": len(SPAN) / len(yrs) if yrs else np.inf,
                "severe_caught": sorted(set(yrs) & set(severe_all)),
                "severe_missed": sorted(set(severe_all) - set(yrs)),
                "no_flood": sorted(set(yrs) - set(flood_all))}
    env = {v: envelope(v) for v in VERSIONS}

    # RP sweep on v4 for the Deyr windows: does another level restore the design?
    sweep = []
    for river in ("juba", "shabelle"):
        w = (river, "deyr"); n_req = TRIGGER_CONFIG[w]["n_req"]
        b = bench[w]
        for rp in (3, 4, 5, 6):
            for nr in sorted({n_req, max(2, n_req - 1), min(len(TRIGGER_STATIONS[river]), n_req + 1)}):
                levels = thr.lookup(lv_df, "glofas_v4", "deyr", rp, TRIGGER_STATIONS[river])
                yrs = activation_years(matrix(dd["glofas_v4"], river, "deyr"), levels, nr)
                sweep.append({"window": cfg.WINDOW_TITLE[w], "rp": rp, "n_req": nr, "n_of": len(TRIGGER_STATIONS[river]),
                              "years": yrs, "n": len(yrs), "severe": len(set(yrs) & set(b["severe"])),
                              "n_severe": len(b["severe"]), "false": len(set(yrs) - set(b["flood"])),
                              "adopted": rp == TRIGGER_CONFIG[w]["rp"] and nr == n_req})

    never = {}
    for river in ("juba", "shabelle"):
        w = (river, "deyr")
        caught = set()
        for s in sweep:
            if s["window"] == cfg.WINDOW_TITLE[w]:
                caught |= set(s["years"])
        never[w] = sorted(set(bench[w]["severe"]) - caught)

    # ---------------------------------------------------------------- figures
    # (a) seasonal maxima v4 vs v5 per Deyr point
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    for ax, river in zip(axes, ("juba", "shabelle")):
        for st in TRIGGER_STATIONS[river]:
            a4 = matrix(dd["glofas_v4"], river, "deyr")[st]; a5 = matrix(dd["glofas_v5"], river, "deyr")[st]
            am4 = a4.groupby(a4.index.year).max(); am5 = a5.groupby(a5.index.year).max()
            ax.scatter(am5, am4.reindex(am5.index), s=22, alpha=.8, label=ST[st])
        lim = [50, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color=FAINT, lw=.8, ls="--"); ax.plot(lim, [2 * x for x in lim], color=GRID, lw=.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("GloFAS v5 Deyr seasonal maximum (m³/s)"); ax.set_ylabel("GloFAS v4 Deyr seasonal maximum (m³/s)")
        ax.set_title(f"{cfg.RIVER_TITLE[river]}: the same seasons on the two versions")
        ax.grid(color=GRID, lw=.6); ax.legend(fontsize=8, frameon=False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.005, "Dashed: equal. Thin grey: v4 = 2 x v5. Every Deyr season 1999-2023, one dot per point.", fontsize=8, color=FAINT)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIGS / "seasonal_maxima.png", bbox_inches="tight"); plt.close(fig)

    # (b) Deyr 2023 at the reference gauges, both versions with their own RP4 level
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for ax, (river, st) in zip(axes, [("juba", "luuq"), ("shabelle", "belet_weyne")]):
        rp = TRIGGER_CONFIG[(river, "deyr")]["rp"]
        for v in VERSIONS:
            s = dd[v][(dd[v].station == st) & (dd[v].date >= "2023-09-15") & (dd[v].date <= "2023-12-31")].set_index("date").discharge
            lvl = thr.lookup(lv_df, v, "deyr", rp, [st])[st]
            ax.plot(s.index, s, color=COL[v], lw=1.6, label=f"{VERSIONS[v]} reanalysis")
            ax.axhline(lvl, color=COL[v], ls="--", lw=1, label=f"{VERSIONS[v]} 1-in-{rp} level ({lvl:,.0f})")
        ax.set_title(f"Deyr 2023 at {ST[st]} ({cfg.RIVER_TITLE[river]})"); ax.set_ylabel("m³/s")
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b %Y"))
        ax.grid(axis="y", color=GRID, lw=.6); ax.legend(fontsize=8, frameon=False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "deyr2023.png", bbox_inches="tight"); plt.close(fig)

    # (c) activation strip
    years = list(range(Y0, Y1 + 1))
    rows_strip = [("Gu Juba · Google (unchanged)", gu_acts[("juba", "gu")], C_GOOGLE),
                  ("Gu Shabelle · Google (unchanged)", gu_acts[("shabelle", "gu")], C_GOOGLE),
                  ("Deyr Juba · v5 levels on v5 flows (as designed)", acts[(("juba", "deyr"), "glofas_v5", "glofas_v5")], C_GLOFAS5),
                  ("Deyr Juba · v4 levels on v4 flows (live)", acts[(("juba", "deyr"), "glofas_v4", "glofas_v4")], C_GLOFAS4),
                  ("Deyr Shabelle · v5 levels on v5 flows (as designed)", acts[(("shabelle", "deyr"), "glofas_v5", "glofas_v5")], C_GLOFAS5),
                  ("Deyr Shabelle · v4 levels on v4 flows (live)", acts[(("shabelle", "deyr"), "glofas_v4", "glofas_v4")], C_GLOFAS4),
                  ("Envelope, design (v5 Deyr)", env["glofas_v5"]["years"], INK),
                  ("Envelope, live (v4 Deyr)", env["glofas_v4"]["years"], INK)]
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
    for i, (label, yrs, c) in enumerate(rows_strip):
        y = len(rows_strip) - 1 - i
        ax.scatter([yy for yy in years if yy in yrs], [y] * len([yy for yy in years if yy in yrs]), color=c, s=70, marker="s", zorder=3)
        ax.text(Y0 - 0.7, y, label, ha="right", va="center", fontsize=8.5, color=INK)
    for yy in severe_all:
        ax.axvspan(yy - 0.5, yy + 0.5, color="#F4E3D6", zorder=0)
    for yy in set(flood_all) - set(severe_all):
        ax.axvspan(yy - 0.5, yy + 0.5, color="#F5F1EA", zorder=0)
    ax.set_yticks([]); ax.set_xlim(Y0 - 0.6, Y1 + 0.6); ax.set_xticks(years); ax.tick_params(axis="x", labelsize=8, rotation=90)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_title("Activation years: shaded columns are gauge flood seasons (darker: severe, two gauges over 1-in-5)")
    fig.tight_layout(); fig.savefig(FIGS / "activations.png", bbox_inches="tight"); plt.close(fig)

    # ---------------------------------------------------------------- page
    def e(x): return html.escape(str(x))
    def yrs(v): return ", ".join(str(y) for y in v) if v else "none"
    ratio_med = tdf.ratio.median()

    thr_rows = "".join(
        f"<tr><td>{cfg.RIVER_TITLE[r.river]}</td><td>{ST[r.station]}</td><td>1-in-{r.rp}</td>"
        f"<td class=n>{r.v5:,.0f}</td><td class=n><em>{r.v4:,.0f}</em></td><td class=n>{r.ratio:.2f}x</td>"
        f"<td class=n>{r.mean5:,.0f}</td><td class=n>{r.mean4:,.0f}</td>"
        f"<td class=n>1-in-{r.readiness_rp}: {r.readiness_band:,.0f}</td></tr>"
        for r in tdf.itertuples())
    gu_thr_rows = "".join(
        f"<tr><td>{cfg.RIVER_TITLE[r.river]}</td><td>{ST[r.station]}</td><td>1-in-{r.rp}</td>"
        f"<td class=n><em>{r.band:,.0f}</em></td><td class=n>{r.v4:,.0f}</td><td class=n>{r.v5:,.0f}</td></tr>"
        for r in gdf.itertuples())

    def act_cell(w, fit, flow):
        a = acts[(w, fit, flow)]
        b = bench[w]
        return (f"<td><b>{len(a)}</b> ({rp_text(len(a))})<br><span class=muted>{yrs(a)}</span><br>"
                f"<span class=muted>severe {len(set(a) & set(b['severe']))}/{len(b['severe'])}, "
                f"outside flood years {len(set(a) - set(b['flood']))}</span></td>")
    act_rows = "".join(
        f"<tr><td><b>{cfg.WINDOW_TITLE[w]}</b><br><span class=muted>rule {TRIGGER_CONFIG[w]['n_req']} of "
        f"{len(TRIGGER_STATIONS[w[0]])} over 1-in-{TRIGGER_CONFIG[w]['rp']}</span></td>"
        + act_cell(w, "glofas_v5", "glofas_v5") + act_cell(w, "glofas_v4", "glofas_v4")
        + act_cell(w, "glofas_v5", "glofas_v4") + act_cell(w, "glofas_v4", "glofas_v5") + "</tr>"
        for w in [("juba", "deyr"), ("shabelle", "deyr")])
    env_rows = "".join(
        f"<tr><td>{'Design: v5 levels on v5 flows in Deyr' if v == 'glofas_v5' else 'Live: v4 levels on v4 flows in Deyr'}</td>"
        f"<td class=n><b>{env[v]['n']}</b></td><td class=n>{rp_text(env[v]['n'])}</td>"
        f"<td>{yrs(env[v]['years'])}</td><td class=n>{len(env[v]['severe_caught'])} of {len(severe_all)}</td>"
        f"<td>{yrs(env[v]['severe_missed'])}</td><td>{yrs(env[v]['no_flood'])}</td></tr>" for v in VERSIONS)
    sweep_rows = "".join(
        f"<tr{' class=pick' if s['adopted'] else ''}><td>{s['window']}</td><td>1-in-{s['rp']}</td><td>{s['n_req']} of {s['n_of']}</td>"
        f"<td class=n>{s['n']}</td><td class=n>{rp_text(s['n'])}</td><td class=n>{s['severe']}/{s['n_severe']}</td>"
        f"<td class=n>{s['false']}</td><td class=muted>{yrs(s['years'])}</td></tr>" for s in sweep)
    bench_rows = "".join(
        f"<tr><td>{cfg.WINDOW_TITLE[w]}</td><td>{yrs(bench[w]['flood'])}</td><td>{yrs(bench[w]['severe'])}</td></tr>"
        for w in cfg.WINDOWS)

    v4d = {w: acts[(w, "glofas_v4", "glofas_v4")] for w in [("juba", "deyr"), ("shabelle", "deyr")]}
    v5d = {w: acts[(w, "glofas_v5", "glofas_v5")] for w in [("juba", "deyr"), ("shabelle", "deyr")]}
    mis_v5_on_v4 = {w: acts[(w, "glofas_v5", "glofas_v4")] for w in v4d}

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GloFAS version switch &mdash; Somalia Riverine Flood Trigger</title>
<meta name="description" content="The Deyr thresholds were fitted on GloFAS v5, but the operational forecast is still v4. Thresholds on each version's own climatology, how far apart they sit, and what the Deyr windows and the envelope look like on the v4 record.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css">
<style>
.hero.hero-sub {{ padding:40px 44px 34px; }} .hero.hero-sub h1 {{ font-size:28px; }}
.crumb {{ font-size:12px; margin:0 0 14px; }} .crumb a {{ color:rgba(255,255,255,.85); text-decoration:none; }}
article {{ padding:8px 44px 24px; max-width:900px; }}
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:21px; color:var(--n9); margin:38px 0 10px; line-height:1.25; }}
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.65; }} article a {{ color:var(--b6); }} article strong {{ color:var(--n9); }}
figure {{ margin:20px 0; }} figure img {{ width:100%; height:auto; display:block; border:1px solid #e2e7e7; border-radius:4px; }}
figcaption {{ font-size:12px; color:var(--n7); margin-top:7px; line-height:1.5; }}
table.data {{ border-collapse:collapse; width:100%; margin:16px 0; font-size:13px; }}
table.data th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:7px 10px 6px; vertical-align:bottom; }}
table.data td {{ border-bottom:1px solid #eef1f1; padding:7px 10px; color:var(--n8); vertical-align:top; }}
table.data td.n {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }} table.data td em {{ font-style:normal; color:var(--b6); font-weight:600; }}
table.data tr.pick td {{ background:var(--b05); }} .muted {{ color:#6b7683; font-size:12px; }} .tablewrap {{ overflow-x:auto; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); gap:14px; margin:26px 0 6px; }}
.stat {{ background:#fff; border:1px solid #e2e7e7; border-top:3px solid var(--b5); border-radius:5px; padding:14px 16px 12px; }}
.stat.flag {{ border-top-color:#e0a04b; }} .stat .v {{ font-family:'Merriweather',Georgia,serif; font-size:24px; font-weight:700; color:var(--b6); line-height:1.1; }}
.stat.flag .v {{ color:#b0722b; }} .stat .l {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--n7); margin-top:6px; line-height:1.45; }}
.callout {{ margin:20px 0; padding:13px 17px; border-radius:4px; background:var(--b05); border-left:5px solid var(--b5); font-size:13.5px; color:var(--n8); line-height:1.6; }}
.callout.warn {{ background:#fdf3e7; border-left-color:#e0a04b; }}
.provenance {{ margin:34px 44px 40px; padding-top:14px; border-top:1px solid #e2e7e7; font-size:12px; color:var(--n7); line-height:1.7; }} .provenance a {{ color:var(--b6); }}
@media (max-width:640px) {{ article {{ padding:4px 22px 16px; }} .hero.hero-sub {{ padding:32px 22px 28px; }} .provenance {{ margin:28px 22px 32px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <div class="inner">
      <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / operational note</p>
      <h1>The GloFAS version switch</h1>
      <p>The Deyr windows were designed on the GloFAS version 5 reanalysis, on the understanding that
         version 5 was the live forecast. It is not yet: the operational system is version 4 (4.5 since
         16 April 2026) and version 5 is pre-operational. A threshold belongs to the climatology of the
         model that produces the forecast it is applied to, so the live monitoring runs on levels fitted
         to the version 4 record. This note gives both sets of levels, how far apart they sit, and what
         the trigger looks like on each version's own record. September 2026.</p>
    </div>
  </header>

  <article>
    <div class="stats">
      <div class="stat flag"><div class="v">v4.5</div><div class="l">operational GloFAS, EWDS, 14 Sep 2026</div></div>
      <div class="stat"><div class="v">{ratio_med:.1f}x</div><div class="l">median v4 / v5 ratio of the Deyr 1-in-4 levels</div></div>
      <div class="stat"><div class="v">{env['glofas_v5']['n']} &rarr; {env['glofas_v4']['n']}</div><div class="l">envelope activations 1999&ndash;2023, design &rarr; live</div></div>
      <div class="stat"><div class="v">{len(env['glofas_v5']['severe_caught'])} &rarr; {len(env['glofas_v4']['severe_caught'])} of {len(severe_all)}</div><div class="l">severe years caught, design &rarr; live</div></div>
    </div>

    <h2>What is live</h2>
    <p>Checked on 14 September 2026. The EWDS forecast dataset offers one <code>operational</code> system version plus legacy
       versions 2.1 and 3.1; there is no legacy version 4 stream, which there would be if version 4 had been retired. The
       EWDS reanalysis dataset labels version 4.0 <em>Operational</em> and version 5.0 <em>Pre-operational</em>. ECMWF's
       <a href="https://confluence.ecmwf.int/display/CEMS/Latest+operational+GloFAS+release">latest-release page</a> names
       v4.5 (16 April 2026), and CEMS has announced that v5.0 will become operational after pre-operational testing with
       v4 run in parallel for a period. The operational GRIB carries no version string, only process identifiers
       (generatingProcessIdentifier 5, backgroundProcess 21), which the pipeline records with every row.</p>
    <div class="callout">The monitoring pipeline therefore uses the <strong>{VERSIONS[cfg.GLOFAS_OPERATIONAL]}</strong> levels below.
      It checks the EWDS catalogue every day and fails, rather than emailing, the day a version 4 entry appears under
      Legacy Versions or the GRIB process identifiers change. Switching is one constant
      (<code>GLOFAS_OPERATIONAL</code> in <code>src/monitoring/config.py</code>); both level sets are frozen in
      <code>src/monitoring/thresholds.json</code>.</div>

    <h2>The two climatologies</h2>
    <p>Version 4 runs {ratio_med:.1f} times higher than version 5 on the Deyr seasonal maxima at these points, and nearly
       flat along each river where version 5 decreases downstream. The versions share the river network here, so
       every comparison is at the same cell. A level fitted on one version is meaningless on the other.</p>
    <figure><img src="figs/seasonal_maxima.png" alt="v4 vs v5 seasonal maxima"><figcaption>Deyr seasonal maximum at each monitored point, 1999&ndash;2023, version 4 against version 5. Dashed: equal; thin grey: version 4 twice version 5.</figcaption></figure>
    <figure><img src="figs/deyr2023.png" alt="Deyr 2023 on both versions"><figcaption>Deyr 2023, the largest flood in the record, at the two reference gauges on each version's reanalysis with each version's own 1-in-4 level. At Luuq both versions cross their own level; at Belet Weyne version 5 crosses and version 4 stays below its level through the largest flood on record, which is why no version 4 rule catches Deyr 2023 on the Shabelle.</figcaption></figure>

    <h2>Deyr action levels, both versions</h2>
    <p>Levels at each point's Deyr return period, fitted on the Deyr seasonal maxima 1999&ndash;2023 of each version's own reanalysis
       (Weibull plotting position, as in the trigger analysis). The live column is highlighted. The last column is the
       readiness level, fitted on the version 4 reforecast at leads 8&ndash;12 days, which the trigger page already used and
       which is consistent with a version 4 live forecast.</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>River</th><th>Point</th><th>RP</th><th>v5 level (design)</th><th>v4 level (live)</th><th>v4 / v5</th><th>v5 Deyr mean</th><th>v4 Deyr mean</th><th>Readiness level (v4 band, 8&ndash;12 d)</th></tr></thead>
    <tbody>{thr_rows}</tbody></table></div>
    <p class="muted">Gu readiness levels (GloFAS carries readiness only in Gu; Google carries the action leg): {', '.join(f"{ST[r.station]} {r.band:,.0f}" for r in gdf.itertuples())} m³/s at 1-in-5, version 4 readiness band. On the reanalyses the same points sit at v4 {', '.join(f"{r.v4:,.0f}" for r in gdf.itertuples())} and v5 {', '.join(f"{r.v5:,.0f}" for r in gdf.itertuples())}.</p>

    <h2>What the Deyr windows do on each version</h2>
    <p>Each window's adopted rule replayed on each version's own reanalysis with levels fitted on that version (the two
       consistent columns), and with the levels crossed over (the two inconsistent columns, which is what applying the
       design levels to the live forecast would have meant). Judged against the two-gauge benchmark: a flood season has two
       of the river's gauges over their own 1-in-3 level, a severe one two over 1-in-5, levels fitted 2000&ndash;2023.</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Window</th><th>v5 levels on v5 flows<br><span class=muted>design</span></th><th>v4 levels on v4 flows<br><span class=muted>live</span></th><th>v5 levels on v4 flows<br><span class=muted>the mismatch</span></th><th>v4 levels on v5 flows</th></tr></thead>
    <tbody>{act_rows}</tbody></table></div>
    <div class="callout warn">Applying the design (v5) levels to the live v4 flows would have activated Deyr Juba in
      {len(mis_v5_on_v4[('juba', 'deyr')])} and Deyr Shabelle in {len(mis_v5_on_v4[('shabelle', 'deyr')])} of 25 years:
      the levels sit far below version 4's ordinary Deyr flows. This is the failure the version guard exists to prevent.</div>

    <h2>The envelope</h2>
    <p>The Gu windows run on Google Flood Hub and are unchanged. The union of all four windows, which is what the
       allocation is sized on, on the design and on the live Deyr levels:</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Deyr levels</th><th>Activations</th><th>Return period</th><th>Years</th><th>Severe caught</th><th>Severe missed</th><th>Outside flood years</th></tr></thead>
    <tbody>{env_rows}</tbody></table></div>
    <figure><img src="figs/activations.png" alt="activation strip"><figcaption>Activation years by window and version. Shaded columns are gauge flood seasons on either river (darker: severe). Gu rows are identical in both views.</figcaption></figure>

    <h2>Does another level restore the design on version 4?</h2>
    <p>For the two Deyr windows on version 4, the activation record at every return period the record supports and at
       the adopted vote count and its neighbours. The adopted rule shape is highlighted. Read this before deciding
       whether the Deyr rules need re-tuning for the live version: the analysis page's calibration searched the envelope
       jointly, and any change here should go back through that search.</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Window</th><th>Level</th><th>Votes</th><th>Activations</th><th>RP</th><th>Severe</th><th>Outside flood years</th><th>Years</th></tr></thead>
    <tbody>{sweep_rows}</tbody></table></div>
    <div class="callout warn"><strong>No version 4 rule reproduces the design.</strong> On the version 4 record the adopted
      Deyr rules activate {len(v4d[('juba', 'deyr')])} and {len(v4d[('shabelle', 'deyr')])} times (design: {len(v5d[('juba', 'deyr')])} and
      {len(v5d[('shabelle', 'deyr')])}) and the envelope moves from {rp_text(env['glofas_v5']['n'])} to {rp_text(env['glofas_v4']['n'])}.
      Severe Deyr seasons that version 4 never registers at any level or vote count tested: Juba {yrs(never[('juba', 'deyr')])};
      Shabelle {yrs(never[('shabelle', 'deyr')])}. Raising the level trims the activations outside flood years but does not
      recover those seasons, so the Deyr windows cannot simply be re-levelled on version 4: the working group has to
      decide between accepting weaker Deyr coverage on the live version, or waiting for version 5 to go operational and
      running Deyr on it as designed.</div>
    <p class="muted">Benchmark by window (two gauges, levels fitted 2000&ndash;2023):</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Window</th><th>Flood years (1-in-3)</th><th>Severe years (1-in-5)</th></tr></thead><tbody>{bench_rows}</tbody></table></div>

    <h2>Also live, also different from the design</h2>
    <ul>
      <li><strong>Dollow's Google point is not the design's.</strong> The design used HYBAS gauge
          <code>hybas_1121038740</code>, which the live Flood Hub API does not serve (404) and which turns out to be
          the Dawa branch (31 m³/s mean). Since 15 September 2026 Dollow reads <code>hybas_1121039440</code>, the Juba
          main stem 8 km east-south-east (142 m³/s mean), which the API serves. Against the SWALIM Dollow gauge it
          tracks as well as the design point did in Gu (Spearman 0.86), and the Gu Juba rule gives the same
          activation years with either point. The Gu levels for Dollow on this page and in the pipeline are fitted
          on the main-stem gauge; the trigger pages still show the design point's Dollow numbers until rebuilt.</li>
      <li><strong>Google's live horizon.</strong> The API returns daily values from two days before the issue to five
          days after it, so the action leg reads Google at leads 1&ndash;5, not 1&ndash;7 as in the archive.</li>
      <li><strong>Readiness levels</strong> were fitted on the version 4 reforecast at leads 8&ndash;12 (the only archive
          at those leads), so they were already consistent with a version 4 live forecast; nothing changes there.</li>
      <li><strong>When version 5 goes live</strong>, flip the constant, regenerate this page and the design comparison
          holds in reverse: the v5 columns above become the live ones. The v5 reanalysis extends to mid-2026 on EWDS, so
          the levels can also be refitted on a longer record at that point.</li>
    </ul>
  </article>

  <p class="provenance">Generated by <code>scripts/build_glofas_version_page.py</code> from <code>data/processed/discharge_daily_glofas_v4|v5.parquet</code>,
    <code>discharge_daily_google_grrr.parquet</code>, <code>swalim_levels.parquet</code> and <code>src/monitoring/thresholds.json</code>
    in <a href="https://github.com/OCHA-DAP/ds-aa-som-floods">ds-aa-som-floods</a>. Sources: <a href="https://ewds.climate.copernicus.eu/datasets/cems-glofas-forecast">EWDS cems-glofas-forecast</a>,
    <a href="https://confluence.ecmwf.int/display/CEMS/Latest+operational+GloFAS+release">ECMWF latest operational GloFAS release</a>,
    <a href="https://global-flood.emergency.copernicus.eu/news/252-The%20new%20Copernicus%20GloFAS%20v5.0%20hydrological%20reanalysis%20has%20been%20released/">CEMS GloFAS v5.0 reanalysis release</a>.
    Design pages: <a href="../trigger-single-model/summary.html">trigger summary</a>, <a href="../trigger-single-model/">analysis</a>. Live: <a href="../monitoring/">monitoring</a>.</p>
</div>
</body>
</html>
"""
    (OUT / "index.html").write_text(page)
    summary = {"generated": pd.Timestamp.today().strftime("%Y-%m-%d"), "operational": cfg.GLOFAS_OPERATIONAL,
               "ratio_median_deyr_rp": float(ratio_med),
               "deyr_levels": tdf.round(1).to_dict(orient="records"),
               "activations": {f"{cfg.WINDOW_KEY[w]}|fit={f}|flow={fl}": y for (w, f, fl), y in acts.items()},
               "gu_activations": {cfg.WINDOW_KEY[w]: y for w, y in gu_acts.items()},
               "envelope": {v: {k: (float(x) if k == "rp" else x) for k, x in e_.items()} for v, e_ in env.items()},
               "benchmark": {cfg.WINDOW_KEY[w]: b for w, b in bench.items()}, "sweep_v4": sweep}
    (OUT / "data.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"wrote {OUT / 'index.html'}")
    print("envelope:", {v: (e_['n'], round(e_['rp'], 1), e_['years']) for v, e_ in env.items()})
    print("deyr acts v4:", {cfg.WINDOW_KEY[w]: y for w, y in v4d.items()})
    print("deyr acts v5:", {cfg.WINDOW_KEY[w]: y for w, y in v5d.items()})
    print("mismatch v5 levels on v4 flows:", {cfg.WINDOW_KEY[w]: len(y) for w, y in mis_v5_on_v4.items()})
    print("gu:", gu_acts, "severe:", severe_all, "flood:", flood_all)


if __name__ == "__main__":
    main()
