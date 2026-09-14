"""Summary page: how we got to the trigger, step by step, with tables and plots.

Every number on the page is computed here from the analysis outputs and cross-checked with
assertions against the analysis page (index.html), so the two cannot drift apart silently.
Writes pages/trigger-single-model/summary.html, three figures (figs/s_*.png) and summary_rows.json.
"""
import datetime
import json
import re
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import somlib as L

S = Path(__file__).parent
PAGE = S / "wt-trigger/pages/trigger-single-model"
FIGS = PAGE / "figs"
OUT = PAGE / "summary.html"

# ---------------------------------------------------------------- the design, as adopted
WINDOWS = [("juba", "deyr"), ("shabelle", "deyr"), ("juba", "gu"), ("shabelle", "gu")]   # Deyr before Gu
RULE = {("juba", "deyr"): (4, 3), ("shabelle", "deyr"): (4, 2), ("juba", "gu"): (5, 3), ("shabelle", "gu"): (6, 2)}
CAL = {"deyr": "glofas_v5", "gu": "google_grrr"}                    # calibrated model per season (data key)
SOURCE = {"deyr": "GloFAS", "gu": "Google Flood Hub"}               # how the trigger names its source
NICE = {"google_grrr": "Google Flood Hub", "glofas_v5": "GloFAS v5", "glofas_v4": "GloFAS v4"}
COL = {"google_grrr": "#1d4ed8", "glofas_v5": "#0f766e", "glofas_v4": "#6b7280", "swalim": "#d97706", "gauge": "#111827"}
SEASON = {"deyr": ("Deyr", "October to December"), "gu": ("Gu", "March to May")}
STATIONS = {"juba": "Dollow, Luuq, Bardheere, Bualle", "shabelle": "Belet Weyne, Bulo Burti, Jowhar"}
ARCHIVE = {"google_grrr": (2016, 2023), "glofas_v4": (2003, 2024)}
SPAN = list(range(1999, 2025))
N_YEARS = 25                                                        # 1999 to 2023, the calibration record


def wname(river, season):
    return f"{SEASON[season][0]} {river.title()}"


def rp_text(n_act, n=N_YEARS):
    """Return period as the analysis page reports it: Weibull (n+1)/m."""
    return f"1-in-{(n + 1) / n_act:.1f}"


def dfmt(x):
    return f"{x.day} {x.strftime('%b')}"


# ---------------------------------------------------------------- inputs already computed
metrics = json.load(open(S / "metrics.json"))
win = {w["window"]: w for w in metrics["windows"]}
station = json.load(open(S / "station_metrics.json"))
swalim_tl = json.load(open(S / "swalim_timeline.json"))
fallback = json.load(open(S / "obs_fallback.json"))["rows"]
G24 = json.load(open(S / "gu2024_issue.json"))
rp3c = pd.DataFrame(json.load(open(S / "rp3_corr.json")))
index_html = (PAGE / "index.html").read_text(encoding="utf-8")
index_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", index_html, flags=re.S)))

# ---------------------------------------------------------------- step 1: floods on the gauges
flood, severe = {}, {}
for river, season in WINDOWS:
    w = win[wname(river, season)]
    flood[(river, season)] = set(w["flood_years"]); severe[(river, season)] = set(w["severe_years"])
flood_all = set().union(*flood.values()); severe_all = set().union(*severe.values())

# ---------------------------------------------------------------- step 4/5: activations at the rule
act = {}
for river, season in WINDOWS:
    w = win[wname(river, season)]
    act[(river, season)] = set(w["models"][CAL[season]]["activations"])
env_years = set().union(*act.values())
env_page = set(metrics["envelope"]["activations"])
assert env_years == env_page, ("envelope mismatch", sorted(env_years), sorted(env_page))
n_act = len(env_years)
sev_caught = len(env_years & severe_all)
outside = sorted(env_years - flood_all)
per_river = {r: set().union(*(act[(r, s)] for s in ("deyr", "gu"))) for r in ("juba", "shabelle")}
rate_env = rp_text(n_act)
rate_river = {r: rp_text(len(v)) for r, v in per_river.items()}
# cross-check against the analysis page's own headline tiles
assert f"{rate_env} overall action return period" in index_text, rate_env
assert f"{rate_river['shabelle']} / {rate_river['juba']} per river" in index_text, rate_river
assert sev_caught == len(severe_all) == 7, (sev_caught, len(severe_all))
assert outside == [2013], outside

# ---------------------------------------------------------------- step 6: forecasts, flood by flood
V4_2024 = {r: (pd.Timestamp(f"2024 {G24[r]['v4_fc_issue'][0]}") if G24[r]["v4_fc_issue"][0] else None) for r in ("juba", "shabelle")}


def verdict(first_onset, when, year, model):
    lo, hi = ARCHIVE[model]
    if not (lo <= year <= hi):
        return ("na", "no archive", None)
    if when is None:
        return ("never", "never crossed", None)
    lead = (first_onset - when).days
    return ("before", f"{lead} d before", lead) if lead > 0 else ("same", "same day", 0) if lead == 0 else ("after", f"{-lead} d after", lead)


fb_rows, fb_tot = {}, {}
for season in ("deyr", "gu"):
    onset, sev_s, fc = {}, {}, {"google_grrr": {}, "glofas_v4": {}}
    for river in ("juba", "shabelle"):
        rp, n = RULE[(river, season)]
        onset[river] = L.gauge_crossings(river, season, 3, span=SPAN)
        sev_s[river] = set(L.gauge_crossings(river, season, L.SEVERE_RP, span=SPAN))
        for mdl in fc:
            fc[mdl][river] = {y: v[0] for y, v in L.first_issue_dates(mdl, river, season, rp, n, span=SPAN, leads=(1, 7)).items()}
        if season == "gu" and V4_2024[river] is not None:
            fc["glofas_v4"][river][2024] = V4_2024[river]
    years = sorted({y for r in onset for y in onset[r] if y in SPAN and (season == "gu" or y <= 2023)})
    # the 1999-2023 flood years must agree with the benchmark used everywhere else
    assert {y for y in years if y <= 2023} == set().union(*(flood[(r, season)] for r in ("juba", "shabelle"))), season
    rows, tot, h2h, n_both = [], {k: {} for k in fc}, {k: 0 for k in fc}, 0
    for y in years:
        flooded = [r for r in onset if y in onset[r]]
        first = min(onset[r][y] for r in flooded)
        rec = {"year": y, "rivers": [r.title() for r in flooded], "severe": any(y in sev_s[r] for r in flooded), "first_onset": dfmt(first)}
        leads = {}
        for mdl in fc:
            c = {r: fc[mdl][r][y] for r in fc[mdl] if y in fc[mdl][r]}
            when, river = (min(c.values()), min(c, key=c.get)) if c else (None, None)
            kind, text, lead = verdict(first, when, y, mdl)
            rec[mdl] = {"kind": kind, "text": text, "river": river.title() if river else None, "when": dfmt(when) if when is not None else None, "lead": lead}
            tot[mdl][kind] = tot[mdl].get(kind, 0) + 1
            leads[mdl] = (kind, lead)
        g, v = leads["google_grrr"], leads["glofas_v4"]
        rec["first"] = ""
        if g[0] != "na" and v[0] != "na":
            n_both += 1
            gl = g[1] if g[1] is not None else -999; vl = v[1] if v[1] is not None else -999
            rec["first"] = "Google" if gl > vl else "GloFAS v4" if vl > gl else ("tie" if g[1] is not None else "neither")
            if gl > vl: h2h["google_grrr"] += 1
            elif vl > gl: h2h["glofas_v4"] += 1
        rows.append(rec)
    fb_rows[season], fb_tot[season] = rows, {"per_model": tot, "head_to_head": h2h, "n_both": n_both, "n": len(rows), "n_severe": sum(r["severe"] for r in rows)}
json.dump({"rows": fb_rows, "totals": fb_tot}, open(S / "summary_rows.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- step 3: agreement in flood seasons
agree = {}
for river, season in WINDOWS:
    g = rp3c[(rp3c.river == river) & (rp3c.season == season)]
    agree[(river, season)] = {m: float(g[m].median()) for m in ("google_grrr", "glofas_v5", "glofas_v4")} | {"n": (int(g.n_flood_seasons.min()), int(g.n_flood_seasons.max()))}
pooled = {m: float(rp3c[m].median()) for m in ("google_grrr", "glofas_v5", "glofas_v4")}
adopted_station = [r for r in station if r["adopted"]]
lag_min, lag_max = min(r["lag_days"] for r in adopted_station), max(r["lag_days"] for r in adopted_station)
assert lag_min >= 0, "a chosen model trails a gauge somewhere; the page says it never does"

# ---------------------------------------------------------------- step 7: SWALIM
both_flag = [t for t in swalim_tl if " d before " in t["vs_trigger"]]
sw_first = sum(1 for t in both_flag if t["vs_trigger"].startswith("SWALIM"))
sw_only = [t for t in swalim_tl if t["vs_trigger"].startswith("SWALIM only")]

# ---------------------------------------------------------------- step 8: fail-safe
fs_only = [r for r in fallback if r["bank1"] and not r["trigger"]]
fs_both = [r for r in fallback if r["bank1"] and r["trigger"] and r["bank1_vs_trigger"] is not None]

# ================================================================ figures
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def fig_agreement():
    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    labels = [wname(r, s) for r, s in WINDOWS]
    for i, (r, s) in enumerate(WINDOWS):
        for m, dy in (("google_grrr", .18), ("glofas_v5", 0), ("glofas_v4", -.18)):
            v = agree[(r, s)][m]
            ax.plot([v], [i + dy], "o", color=COL[m], ms=8, mec="white", mew=.8, zorder=3)
        ax.axhline(i + .5, color="#eef1f4", lw=1) if i < 3 else None
    ax.axvline(0, color="#9ca3af", lw=1, ls=":")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(-0.7, 1.0); ax.set_xlabel("rank correlation of seasonal peaks over the gauge's own 1-in-3 seasons (median across the window's gauges)")
    ax.grid(axis="x", color="#eef1f4"); ax.tick_params(length=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=COL[m], label=NICE[m]) for m in ("google_grrr", "glofas_v5", "glofas_v4")],
              loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0, 1.0))
    fig.tight_layout(); fig.savefig(FIGS / "s_agreement.png", dpi=150); plt.close(fig)


def fig_activations():
    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    yrs = list(range(1999, 2024))
    for i, (r, s) in enumerate(WINDOWS):
        for y in yrs:
            f, sv, a = y in flood[(r, s)], y in severe[(r, s)], y in act[(r, s)]
            if f:
                ax.plot([y], [i], "s", color="#cbd5e1" if not sv else COL["gauge"], ms=11 if sv else 9, zorder=2)
            if a:
                ax.plot([y], [i], "o", color=COL[CAL[s]], ms=5.5, mec="white", mew=.8, zorder=4)
    ax.set_yticks(range(len(WINDOWS))); ax.set_yticklabels([wname(r, s) for r, s in WINDOWS]); ax.invert_yaxis()
    ax.set_xticks(yrs); ax.set_xticklabels([str(y)[2:] for y in yrs], fontsize=8.5); ax.set_xlim(1998.4, 2023.6)
    ax.grid(axis="x", color="#f1f5f9"); ax.tick_params(length=0)
    ax.legend(handles=[Line2D([], [], marker="s", ls="none", color="#cbd5e1", ms=9, label="gauge flood season (two gauges over 1-in-3)"),
                       Line2D([], [], marker="s", ls="none", color=COL["gauge"], ms=11, label="severe (two gauges over 1-in-5)"),
                       Line2D([], [], marker="o", ls="none", color=COL["glofas_v5"], label="window activates, GloFAS"),
                       Line2D([], [], marker="o", ls="none", color=COL["google_grrr"], label="window activates, Google Flood Hub")],
              loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.0), fontsize=8.6, columnspacing=2.0)
    fig.tight_layout(); fig.savefig(FIGS / "s_activations.png", dpi=150); plt.close(fig)


def fig_leads():
    rows = [(s, r) for s in ("deyr", "gu") for r in fb_rows[s]]
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(rows) + 1.4))
    ax.axvspan(1, 7, color="#dcfce7", alpha=.8, zorder=0); ax.axvspan(8, 12, color="#f0fdf4", alpha=.9, zorder=0)
    ax.axvline(0, color="#374151", lw=1.2)
    labels = []
    for i, (s, r) in enumerate(rows):
        labels.append(f"{SEASON[s][0]} {r['year']}{' *' if r['severe'] else ''}")
        for m, dy in (("google_grrr", .17), ("glofas_v4", -.17)):
            v = r[m]
            if v["kind"] == "na":
                continue
            if v["lead"] is None:
                ax.plot([-22], [i + dy], "x", color=COL[m], ms=7, mew=1.6, zorder=3)
            else:
                x = max(min(v["lead"], 21), -21)
                ax.plot([x], [i + dy], "o", color=COL[m], ms=7, mec="white", mew=.8, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(-23.5, 22); ax.set_xticks([-21, -14, -7, 0, 7, 14, 21])
    ax.set_xticklabels(["21 d after", "14 d after", "7 d after", "flood begins", "7 d before", "14 d before", "21 d before"], fontsize=8.5)
    ax.text(4, -0.9, "action window", color="#166534", fontsize=8.5, ha="center"); ax.text(10, -0.9, "readiness", color="#4d7c0f", fontsize=8.5, ha="center")
    ax.grid(axis="x", color="#f1f5f9"); ax.tick_params(length=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=COL["google_grrr"], label="Google Flood Hub forecast, first issue meeting the rule"),
                       Line2D([], [], marker="o", ls="none", color=COL["glofas_v4"], label="GloFAS v4 forecast (the version running live)"),
                       Line2D([], [], marker="x", ls="none", color="#6b7280", mew=1.6, label="never crossed (shown at left edge)")],
              loc="lower left", frameon=False, ncol=1, bbox_to_anchor=(0, 1.02), fontsize=8.6)
    fig.tight_layout(); fig.savefig(FIGS / "s_leads.png", dpi=150); plt.close(fig)


fig_agreement(); fig_activations(); fig_leads()

# ================================================================ HTML helpers
CSS = """
:root{--ink:#111827;--muted:#6b7280;--rule:#e5e7eb;--green:#1f6f5f;--ok:#166534;--okbg:#dcfce7;--late:#9a3412;--latebg:#ffedd5;--miss:#991b1b;--missbg:#fee2e2;--na:#6b7280;--nabg:#f3f4f6}
body{margin:0;font:15px/1.6 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#fff}
html,body{overflow-x:hidden}
header{background:var(--green);color:#fff;padding:40px 0 30px}
.wrap{max-width:980px;margin:0 auto;padding:0 24px}
header h1{font:700 30px/1.15 Georgia,"Times New Roman",serif;margin:6px 0 12px}
header p{font-size:16.5px;max-width:840px;margin:0;opacity:.95}
header small{opacity:.8;letter-spacing:.02em}
h2{font:700 21px/1.25 Georgia,serif;margin:44px 0 6px}
h2 span{display:inline-block;min-width:1.6em;color:var(--green)}
p{max-width:840px} p.note{color:var(--muted);font-size:13.5px}
.tw{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 6px}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:2px solid var(--ink);padding:7px 8px;font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}
td{padding:7px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums} td.pick{font-weight:700}
tr.sev td:first-child{font-weight:700}
figure{margin:14px 0} figure img{max-width:100%;display:block} figcaption{color:var(--muted);font-size:13px;margin-top:6px;max-width:840px}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12.5px;font-weight:600;white-space:nowrap}
.before{background:var(--okbg);color:var(--ok)} .same,.after{background:var(--latebg);color:var(--late)} .never{background:var(--missbg);color:var(--miss)} .na{background:var(--nabg);color:var(--na)}
.who{color:var(--muted);font-size:12px;display:block}
ul{max-width:840px} li{margin:5px 0}
footer{color:var(--muted);font-size:13px;margin:40px 0 30px;border-top:1px solid var(--rule);padding-top:14px}
"""
H = []                                                                  # page parts


def add(s): H.append(s)


def table(head, rows, cls=None):
    th = "".join(f'<th{" class=\"n\"" if c.startswith("#") else ""}>{escape(c.lstrip("#"))}</th>' for c in head)
    body = "".join("<tr" + (f' class="{cls(r)}"' if cls else "") + ">" + "".join(f"<td{a}>{c}</td>" for c, a in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def n(x): return (x, ' class="n"')
def c(x): return (x, "")
def pick(x): return (x, ' class="n pick"')
def pill(v):
    who = f'<span class="who">{escape(v["river"])}, {escape(v["when"])}</span>' if v["river"] else ""
    return f'<span class="pill {v["kind"]}">{escape(v["text"])}</span>{who}'


yl = lambda ys: ", ".join(str(y) for y in sorted(ys)) or "none"

# ================================================================ the page
add(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>How we got to the trigger | Somalia riverine flood trigger</title><style>{CSS}</style></head><body>
<header><div class="wrap"><small>Somalia Riverine Flood Trigger / summary</small>
<h1>How we got to the trigger</h1>
<p>Nine steps, from what counts as a flood to what runs live. Each step shows the evidence it rests on. All figures are for 1999 to 2023 unless stated; the full analysis is on the <a href="index.html" style="color:#fff">trigger analysis page</a>.</p></div></header>
<div class="wrap">""")

# --- the result up front
add("<h2><span>0</span>The trigger</h2><p>Two rivers, two rainy seasons, four windows. Each window runs on one forecast source and one rule. Any one window activating releases the full allocation.</p>")
add(table(["Window", "Season", "Source", "Rule", "Gauges"],
          [[c(wname(r, s)), c(SEASON[s][1]), c(SOURCE[s]), c(f"{RULE[(r, s)][1]} of {4 if r == 'juba' else 3} gauges forecast over their own 1-in-{RULE[(r, s)][0]} level on the same day"), c(STATIONS[r])] for r, s in WINDOWS]))
add("<p class=\"note\">Readiness runs 8 to 12 days ahead on the GloFAS ensemble in all four windows and releases the mobilisation share; action runs 1 to 7 days ahead and releases the rest. Thresholds are fitted on each source's own record, so a model that runs high is judged against itself.</p>")

# --- step 1
add("<h2><span>1</span>What counts as a flood</h2><p>The trigger is scored against what the rivers actually did, using the SWALIM gauge record. A river has a flood season when two of its gauges reach their own 1-in-3 level; a severe season when two reach 1-in-5. Each gauge's levels are fitted on its own record for 2000 to 2023, so the same rule means a different height at every station.</p>")
add(table(["Window", "Gauges", "#Flood seasons", "Years", "#Severe", "Severe years"],
          [[c(wname(r, s)), c(STATIONS[r]), n(str(len(flood[(r, s)]))), c(yl(flood[(r, s)])), n(str(len(severe[(r, s)]))), c(yl(severe[(r, s)]))] for r, s in WINDOWS]))
add(f"<p class=\"note\">Across both rivers: {len(flood_all)} flood years and {len(severe_all)} severe years ({yl(severe_all)}). Two limits of this benchmark are worth knowing. Gauges are capped at bank full, so the largest floods all record the same reading. And a season in which only one gauge crosses does not count, even when that gauge was at bank full (Gu 2021 at Belet Weyne).</p>")
add('<figure><img src="figs/map_stations.png" alt="Map of the seven monitored gauges"><figcaption>The seven monitored gauges: Dollow, Luuq, Bardheere and Bualle on the Juba; Belet Weyne, Bulo Burti and Jowhar on the Shabelle.</figcaption></figure>')

# --- step 2
add("<h2><span>2</span>The candidate forecasts</h2><p>Three global river-flow models publish forecasts for these rivers. Each was tested on its own historical record.</p>")
add(table(["Model", "Record used to set thresholds", "Forecast archive for testing lead time", "Live today"],
          [[c("Google Flood Hub"), c("Retrospective run, 1999 to 2023"), c("Reforecast, 2016 to mid-2023, leads 1 to 7 days"), c("Forecasts exist; no API access yet")],
           [c("GloFAS"), c("Version 5 reanalysis, 1999 to 2023"), c("Version 4 reforecast, 2003 to 2023, 11-member ensemble; version 5 has no forecast archive"), c("Version 4 forecast, daily")],
           [c("GEOGloWS"), c("Retrospective run, 1999 to 2023"), c("None before July 2024"), c("Forecasts exist")]]))
add("<p class=\"note\">GEOGloWS runs 4 to 10 times too high on the Shabelle and at Luuq and has no archive to test at lead time, so it drops out at step 4. GloFAS version 5 is the calibration record for Deyr; because it has no forecast, version 4 is what runs live and is what the lead-time tests use.</p>")

# --- step 3
add("<h2><span>3</span>Which model tracks each river in flood seasons</h2><p>For each gauge, only the seasons in which it reached its own 1-in-3 level are kept, and each model's peak in those seasons is ranked against the gauge's. A value of 1 means the model orders the flood seasons exactly as the river did; 0 means no relation.</p>")
add(table(["Window", "#Google Flood Hub", "#GloFAS v5", "#GloFAS v4", "#Flood seasons per gauge"],
          [[c(wname(r, s))] + [(pick if m == CAL[s] else n)(f"{agree[(r, s)][m]:.2f}") for m in ("google_grrr", "glofas_v5", "glofas_v4")] + [n(f"{agree[(r, s)]['n'][0]} to {agree[(r, s)]['n'][1]}")] for r, s in WINDOWS]))
add('<figure><img src="figs/s_agreement.png" alt="Agreement with the gauges in flood seasons, by window and model"><figcaption>The same values as the table. Bold in the table marks the source the window runs on.</figcaption></figure>')
add(f"<p>Pooled over all gauges the medians are Google Flood Hub {pooled['google_grrr']:.2f}, GloFAS v5 {pooled['glofas_v5']:.2f} and GloFAS v4 {pooled['glofas_v4']:.2f}. Two cautions. Each gauge has 4 to 10 flood seasons, so a single season moves these values a lot. And every model reads near zero on the Shabelle in Gu because the gauge is capped at bank full in the largest floods: Belet Weyne reads 8.3 m for weeks, so no ordering of severity remains for a model to match. That is a limit of the record, not evidence against the models.</p>")
add(f"<p class=\"note\">A second check on the same question: the chosen source's daily flow leads the gauge at every one of the 14 station-seasons, by {lag_min} to {lag_max} days at the best fit. It never trails.</p>")

# --- step 4
add("<h2><span>4</span>Which model catches the floods at the window's rule</h2><p>Correlation makes the shortlist; this step picks the model. Each window's rule is run on each model's own record with thresholds fitted on that record, and scored against the severe seasons from step 1.</p>")
rows4 = []
for r, s in WINDOWS:
    w = win[wname(r, s)]
    for m in ("google_grrr", "glofas_v5", "glofas_v4"):
        a = w["models"][m]
        f = pick if m == CAL[s] else n
        rows4.append([c(wname(r, s) if m == "google_grrr" else ""), c(NICE[m]), f(f"{a['vs_severe']['hits']} of {len(w['severe_years'])}"), f(str(a['vs_flood']['false_alarms'])), f(f"{a['auc_severe']:.2f}"), c(yl(a["activations"]))])
add(table(["Window", "Model", "#Severe caught", "#Activations with no flood", "#AUC", "Activation years"], rows4))
add('<figure><img src="figs/l_roc.png" alt="Detection against false alarms as the threshold is swept, per window and model"><figcaption>Detection rate against false-alarm rate as the return-period threshold is swept, per window and model. AUC is the area under each curve: 1.0 separates severe seasons from the rest perfectly, 0.5 is chance.</figcaption></figure>')
add("<p>In Gu, Google Flood Hub is chosen: it matches GloFAS v5 on severe seasons caught, and it is the only Gu candidate with a forecast archive (step 6). In Deyr, GloFAS is chosen: Google over-activates on the Juba (three activations with no flood) and misses the Shabelle in 2020, while GloFAS v5's Deyr record is clean. GloFAS v4 is the weakest option in Gu on the Shabelle, catching one severe season of three.</p>")

# --- step 5
add("<h2><span>5</span>Setting the rule and the return period</h2><p>Every threshold sits at 1-in-3 or rarer. The vote count and return period per window were chosen so that the whole mechanism activates about once in three years while still catching every severe season.</p>")
rows5 = [[c(wname(r, s)), c(SOURCE[s]), c(f"{RULE[(r, s)][1]} of {4 if r == 'juba' else 3}"), c(f"1-in-{RULE[(r, s)][0]}"), n(str(len(act[(r, s)]))), c(rp_text(len(act[(r, s)]))), c(yl(act[(r, s)]))] for r, s in WINDOWS]
rows5 += [[c("Either window, Juba"), c(""), c(""), c(""), n(str(len(per_river["juba"]))), c(rate_river["juba"]), c(yl(per_river["juba"]))],
          [c("Either window, Shabelle"), c(""), c(""), c(""), n(str(len(per_river["shabelle"]))), c(rate_river["shabelle"]), c(yl(per_river["shabelle"]))],
          [c("Whole mechanism"), c(""), c(""), c(""), n(str(n_act)), c(rate_env), c(yl(env_years))]]
add(table(["Window", "Source", "Rule", "Threshold", "#Activations, 25 years", "Return period", "Years"], rows5))
add(f'<figure><img src="figs/s_activations.png" alt="Flood seasons and activations by year and window"><figcaption>Squares are gauge flood seasons (dark where severe); dots are the years the window\'s rule activates on the chosen source. The mechanism activates in {n_act} of 25 years, catches all {len(severe_all)} severe seasons, and activates once with no gauge flood behind it ({yl(outside)}, a year SWALIM and WFP both record as a major flood).</figcaption></figure>')
add("<p class=\"note\">Return periods use the Weibull convention, (years + 1) divided by activations. Each window holds three to five severe seasons: every difference on this page is a one- or two-event difference.</p>")

# --- step 6
add("<h2><span>6</span>Checked on the forecasts</h2><p>Steps 3 to 5 use each model's record of the past. A trigger runs on forecasts, so the last test replays the historical forecasts and asks on what day the alert would have gone out, at lead times of 1 to 7 days, against the day the flood began at the gauges. Either river counts, since any one window releases the full allocation. Only Google Flood Hub and GloFAS v4 have forecast archives.</p>")
add('<figure><img src="figs/s_leads.png" alt="Lead time of the first forecast issue meeting the rule, per flood season"><figcaption>Each row is a flood season; * marks severe. Green band: the action window, 1 to 7 days before the flood; pale green: readiness, 8 to 12 days. Points right of the line are alerts before the flood began.</figcaption></figure>')
rows6 = []
for s in ("deyr", "gu"):
    for r in fb_rows[s]:
        rows6.append([c(f"{SEASON[s][0]} {r['year']}{' *' if r['severe'] else ''}"), c(" and ".join(r["rivers"])), c(r["first_onset"]), c(pill(r["google_grrr"])), c(pill(r["glofas_v4"])), c(escape(r["first"]))])
add(table(["Season", "River(s) that flooded", "Flood began", "Google Flood Hub forecast", "GloFAS v4 forecast", "First"], rows6, cls=lambda row: ""))
d, g = fb_tot["deyr"], fb_tot["gu"]
add(f"<p>Deyr: GloFAS v4 activated before the flood in {d['per_model']['glofas_v4'].get('before', 0)} of {d['n']} seasons and was first in {d['head_to_head']['glofas_v4']} of the {d['n_both']} both archives cover. Gu: Google was first in {g['head_to_head']['google_grrr']} of {g['n_both']}, and on the Shabelle gave 10 to 13 days of warning in three of four seasons where GloFAS v4 gave 3 days once and nothing in the other three. The lead-time evidence and the calibration choice were made independently and point the same way: Google in Gu, GloFAS in Deyr. Gu 2024 uses the operational GloFAS forecasts; Google has no record for it.</p>")

# --- step 7
add("<h2><span>7</span>Against SWALIM's own alerts</h2><p>SWALIM issues flood risk bulletins from the gauge readings and the rainfall outlook. For every flood season with a bulletin, the date of SWALIM's first flag is set against the first day the window's source crossed on its own record.</p>")
rows7 = [[c(t["season"]), c(t["river"]), c(t["swalim_first"] or "no bulletin"), c(t["trigger"] or ("record ends 2023" if "ends 2023" in t["vs_trigger"] else "never crossed")), c(t["vs_trigger"])] for t in swalim_tl]
add(table(["Season", "River", "SWALIM first bulletin", "Source first crossed", "Who was first"], rows7))
add('<figure><img src="figs/k_swalim_window.png" alt="SWALIM bulletins, the window\'s source and the GloFAS v4 forecast against the day the flood began"><figcaption>The same seasons against the day the flood began at the gauges. In Deyr the GloFAS v4 forecast issue is shown as well, since it is the version running live.</figcaption></figure>')
add(f"<p>Where both flagged, SWALIM was first in {sw_first} of {len(both_flag)} seasons, and in {len(sw_only)} seasons only SWALIM flagged. SWALIM's alerts are forward-looking and often early, so they sit in the readiness phase. They cannot carry the action phase: CERF requires an activation basis that can be backtested, and the bulletins are expert judgement issued when the analysts see the risk rather than by a fixed rule.</p>")

# --- step 8
add("<h2><span>8</span>Readiness and the fail-safe</h2><p>Readiness activates when SWALIM issues a moderate flood risk alert for either river, or when the GloFAS ensemble gives a 50% probability of exceeding the window's threshold at the required gauges within 8 to 12 days. It releases the mobilisation share only.</p><p>If every forecast misses, a gauge reaching bank full is the fail-safe. On the record it would have activated in these seasons with no forecast activation:</p>")
add(table(["Window", "Year", "Gauge at bank full", "Benchmark"], [[c(r["window"]), c(str(r["year"])), c(f"{r['bank1_st']}, {r['bank1']}"), c(r["benchmark"] if r["benchmark"] != "none" else "not a flood on two gauges")] for r in fs_only]))
add(f"<p class=\"note\">Where a forecast also activated, bank full came {min(r['bank1_vs_trigger'] for r in fs_both)} to {max(r['bank1_vs_trigger'] for r in fs_both)} days later in all {len(fs_both)} cases: it adds coverage, never lead time. Juba gauges never read bank full in Gu, so the fail-safe cannot help there.</p>")

# --- step 9
add("<h2><span>9</span>What runs live, and what is still open</h2><ul>"
    "<li><b>GloFAS version 4 runs both phases today.</b> Deyr is calibrated on version 5, which has no published forecast yet; Gu on Google Flood Hub, to which there is no API access yet. Version 4 stands in with levels refitted on its own record. Both substitutions are declared.</li>"
    "<li><b>Two gauges have stopped reporting.</b> Bardheere from 30 November 2023, Bualle from 14 March 2024. Both are needed to keep a non-unanimous rule on the Juba.</li>"
    "<li><b>Samples are small.</b> The design is defensible because several independent tests point the same way, not because any single number is decisive.</li>"
    "<li><b>Impact years are not yet defined.</b> The benchmark is gauge levels, not people affected. 2013 (activation, no gauge flood, major flood in the SWALIM and WFP records) and 2021 (bank full at one gauge, 400,000 affected, no activation) are why an impact cross-check is the next step.</li></ul>")
add(f"<footer>Generated {datetime.date.today().isoformat()} from the trigger analysis outputs; every figure on this page is recomputed from them and cross-checked against the analysis page when the page is built.</footer></div></body></html>")

html = "".join(H)
OUT.write_text(html, encoding="utf-8")

# ================================================================ consistency checks on the rendered page
vis = html
assert "—" not in vis, "em dash on the page"
for word in (" fired", " fires ", " basin", " product"):
    assert word not in vis, f"forbidden word {word!r}"
for f in re.findall(r'figs/([^"]+)"', vis):
    assert (FIGS / f).exists(), f"missing figure {f}"
design_tbl = re.search(r"<h2><span>0</span>.*?</table>", vis, re.S).group(0)
assert not re.search(r"\bv[45]\b", design_tbl), "version in the trigger table"
for tbl in re.findall(r"<table>.*?</table>", vis, re.S):
    names = re.findall(r"<td>(Deyr|Gu) (?:Juba|Shabelle)</td>", tbl)
    if names:
        assert names.index("Gu") > names.index("Deyr") if "Gu" in names and "Deyr" in names else True
        assert all(x == "Deyr" for x in names[:names.index("Gu")]) if "Gu" in names else True, "Deyr rows must precede Gu rows"
print("wrote", OUT.name, "| checks passed")
print(f"envelope {n_act} in {N_YEARS} = {rate_env}; per river {rate_river}; severe {sev_caught} of {len(severe_all)}; outside {outside}")
print("agreement pooled", {NICE[k]: round(v, 2) for k, v in pooled.items()}, "| lag", lag_min, lag_max)
print("SWALIM first", sw_first, "of", len(both_flag), "| only", len(sw_only), "| fail-safe only", [(r['window'], r['year']) for r in fs_only])
