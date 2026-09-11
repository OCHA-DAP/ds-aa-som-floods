"""Summary page: the case for the trigger design in digestible numbers, then flood by flood
which forecast would have caught each event and when.
Writes wt-trigger/pages/trigger-single-model/summary.html and summary_rows.json."""
import datetime
import json
import re
from html import escape
from pathlib import Path

import pandas as pd

import somlib as L

S = Path(__file__).parent
PAGE = S / "wt-trigger/pages/trigger-single-model"
OUT = PAGE / "summary.html"
RULE = {("juba", "deyr"): (4, 3), ("shabelle", "deyr"): (4, 2),
        ("juba", "gu"): (5, 3), ("shabelle", "gu"): (6, 2)}
MODEL = {"deyr": "GloFAS v5", "gu": "Google Flood Hub"}          # for data lookups
SOURCE = {"deyr": "GloFAS", "gu": "Google Flood Hub"}           # how the trigger names its source
ARCHIVE = {"google_grrr": (2016, 2023), "glofas_v4": (2003, 2024)}
SPAN = list(range(1999, 2025))
LABEL = {"deyr": ("Deyr", "October to December"), "gu": ("Gu", "March to May")}
STATIONS = {"juba": "Dollow, Luuq, Bardheere, Bualle", "shabelle": "Belet Weyne, Bulo Burti, Jowhar"}

# ---------------------------------------------------------------- inputs already computed
metrics = json.load(open(S / "metrics.json"))
station = json.load(open(S / "station_metrics.json"))
swalim_tl = json.load(open(S / "swalim_timeline.json"))
fallback = json.load(open(S / "obs_fallback.json"))["rows"]
G24 = json.load(open(S / "gu2024_issue.json"))
rp3c = pd.DataFrame(json.load(open(S / "rp3_corr.json")))      # rp3_corr.py: rho over 1-in-3+ seasons only
V4_2024 = {r: (pd.Timestamp(f"2024 {G24[r]['v4_fc_issue'][0]}") if G24[r]["v4_fc_issue"][0] else None)
           for r in ("juba", "shabelle")}

# peak-correlation table from the analysis page (How the model was chosen)
html_page = (PAGE / "index.html").read_text(encoding="utf-8")
m = re.search(r"<table[^>]*>(?:(?!</table>).)*peak Spearman(?:(?!</table>).)*</table>", html_page, re.S)
peak = {}
for r in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(0), re.S):
    cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
    if cells and cells[0] in ("Juba Gu", "Juba Deyr", "Shabelle Gu", "Shabelle Deyr"):
        peak[cells[0]] = {"GEOGloWS": cells[1], "GloFAS v5": cells[2], "Google Flood Hub": cells[3], "GloFAS v4": cells[4]}


def dfmt(x):
    return f"{x.day} {x.strftime('%b')}"


def verdict(first_onset, when, year, model):
    lo, hi = ARCHIVE[model]
    if not (lo <= year <= hi):
        return ("na", "no archive", None)
    if when is None:
        return ("never", "never crossed", None)
    lead = (first_onset - when).days
    if lead > 0:
        return ("before", f"{lead} d before", lead)
    if lead == 0:
        return ("same", "same day", 0)
    return ("after", f"{-lead} d after", lead)


# ---------------------------------------------------------------- flood by flood, either river
rows_all, totals = {}, {}
for season in ("deyr", "gu"):
    onset, severe, fc = {}, {}, {"google_grrr": {}, "glofas_v4": {}}
    for river in ("juba", "shabelle"):
        rp, n = RULE[(river, season)]
        onset[river] = L.gauge_crossings(river, season, 3, span=SPAN)
        severe[river] = set(L.gauge_crossings(river, season, L.SEVERE_RP, span=SPAN))
        for mdl in fc:
            fc[mdl][river] = {y: v[0] for y, v in L.first_issue_dates(mdl, river, season, rp, n, span=SPAN, leads=(1, 7)).items()}
        if season == "gu" and V4_2024[river] is not None:
            fc["glofas_v4"][river][2024] = V4_2024[river]
    years = sorted({y for r in onset for y in onset[r] if y in SPAN and (season == "gu" or y <= 2023)})
    rows, tot, both, n_both = [], {k: {} for k in fc}, {k: 0 for k in fc}, 0
    for y in years:
        flooded = [r for r in onset if y in onset[r]]
        first = min(onset[r][y] for r in flooded)
        rec = {"year": y, "rivers": [r.title() for r in flooded], "severe": any(y in severe[r] for r in flooded),
               "first_onset": dfmt(first)}
        leads = {}
        for mdl in fc:
            c = {r: fc[mdl][r][y] for r in fc[mdl] if y in fc[mdl][r]}
            when, river = (min(c.values()), min(c, key=c.get)) if c else (None, None)
            kind, text, lead = verdict(first, when, y, mdl)
            rec[mdl] = {"kind": kind, "text": text, "river": river.title() if river else None,
                        "when": dfmt(when) if when is not None else None}
            tot[mdl][kind] = tot[mdl].get(kind, 0) + 1
            leads[mdl] = (kind, lead)
        g, v = leads["google_grrr"], leads["glofas_v4"]
        rec["first"] = ""
        if g[0] != "na" and v[0] != "na":
            n_both += 1
            gl = g[1] if g[1] is not None else -999
            vl = v[1] if v[1] is not None else -999
            if gl > vl:
                rec["first"] = "Google"; both["google_grrr"] += 1
            elif vl > gl:
                rec["first"] = "GloFAS v4"; both["glofas_v4"] += 1
            else:
                rec["first"] = "tie" if g[1] is not None else "neither"
        rows.append(rec)
    rows_all[season] = rows
    totals[season] = {"per_model": tot, "head_to_head": both, "n_both": n_both,
                      "n": len(rows), "n_severe": sum(r["severe"] for r in rows)}
json.dump({"rows": rows_all, "totals": totals}, open(S / "summary_rows.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- digestible values
env = metrics["envelope"]
n_years = env["vs_severe"]["n"]
n_act = len(env["activations"])
sev_caught = env["vs_severe"]["hits"]; sev_all = sev_caught + env["vs_severe"]["misses"]
outside = sorted(set(env["activations"]) - set(env["flood_years"]))
win = {w["window"]: w for w in metrics["windows"]}

adopted_station = [r for r in station if r["adopted"]]
rho_min = min(r["rho"] for r in adopted_station); rho_max = max(r["rho"] for r in adopted_station)
lag_min = min(r["lag_days"] for r in adopted_station); lag_max = max(r["lag_days"] for r in adopted_station)
rp3_hits = sum(r["rp3"]["hits"] for r in adopted_station); rp3_ev = sum(r["rp3"]["gauge_events"] for r in adopted_station)
rp5_hits = sum(r["rp5"]["hits"] for r in adopted_station); rp5_ev = sum(r["rp5"]["gauge_events"] for r in adopted_station)

both_flag = [t for t in swalim_tl if " d before " in t["vs_trigger"]]
sw_first = sum(1 for t in both_flag if t["vs_trigger"].startswith("SWALIM"))
model_first = len(both_flag) - sw_first
sw_only = [t for t in swalim_tl if t["vs_trigger"].startswith("SWALIM only")]

fs_only = [r for r in fallback if r["bank1"] and not r["trigger"]]
fs_redundant = [r for r in fallback if r["bank1"] and r["trigger"]]
fs_lag = [r["bank1_vs_trigger"] for r in fs_redundant if r["bank1_vs_trigger"] is not None]

# ---------------------------------------------------------------- HTML
CSS = """
:root{--ink:#111827;--muted:#6b7280;--rule:#e5e7eb;--ok:#166534;--okbg:#dcfce7;--late:#9a3412;--latebg:#ffedd5;
--miss:#991b1b;--missbg:#fee2e2;--na:#6b7280;--nabg:#f3f4f6;--green:#1f6f5f;--g:#1d4ed8;--v5:#0f766e;--v4:#374151}
body{margin:0;font:15px/1.6 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#fff}
header{background:var(--green);color:#fff;padding:40px 0 30px}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
header h1{font:700 30px/1.15 Georgia,"Times New Roman",serif;margin:6px 0 12px}
header p{font-size:16.5px;max-width:840px;margin:0;opacity:.95}
header small{opacity:.8;letter-spacing:.02em}
h2{font:700 22px/1.2 Georgia,serif;margin:44px 0 6px}
h3{font-size:15px;font-weight:600;margin:24px 0 6px}
p{max-width:840px}
p.note{color:var(--muted);font-size:13.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:16px 0 8px}
.tiles>div{border:1px solid var(--rule);border-radius:8px;padding:14px 16px;min-width:0}
.tiles .big{font:700 28px/1.1 Georgia,serif;margin:0 0 6px}
.tiles .lab{font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tiles .sub{font-size:13px;color:var(--muted);margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 6px}
.tw{overflow-x:auto;max-width:100%}
html,body{overflow-x:hidden}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:2px solid var(--ink);padding:8px;font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}
td{padding:8px;border-bottom:1px solid var(--rule);vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums} th.n{text-align:right}
td.pick{font-weight:700}
tr.sev td:first-child{font-weight:700}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12.5px;font-weight:600;white-space:nowrap}
.before{background:var(--okbg);color:var(--ok)} .same,.after{background:var(--latebg);color:var(--late)}
.never{background:var(--missbg);color:var(--miss)} .na{background:var(--nabg);color:var(--na)}
.who{color:var(--muted);font-size:12px;display:block}
.head{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:14px 0 18px}
.head>div{border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
.head b{display:block;font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px}
.head .big{font:700 24px/1.1 Georgia,serif;margin:2px 0 6px}
.head .g .big{color:var(--g)} .head .v4 .big{color:var(--v4)}
.head .sub{color:var(--muted);font-size:13.5px}
ul{max-width:840px} li{margin:4px 0}
footer{color:var(--muted);font-size:13px;margin:40px 0 30px;border-top:1px solid var(--rule);padding-top:14px}
"""


def tile(big, lab, sub=""):
    return f'<div><div class="lab">{escape(lab)}</div><div class="big">{escape(big)}</div><div class="sub">{escape(sub)}</div></div>'


def pill(v):
    who = f'<span class="who">{escape(v["river"])}, {escape(v["when"])}</span>' if v["river"] else ""
    return f'<span class="pill {v["kind"]}">{escape(v["text"])}</span>{who}'


# --- section 1: the design
design_rows = "".join(
    f"<tr><td>{LABEL[s][0]} · {r.title()}</td><td>{LABEL[s][1]}</td><td>{SOURCE[s]}</td>"
    f"<td>{RULE[(r, s)][1]} of {4 if r == 'juba' else 3} points over their own 1-in-{RULE[(r, s)][0]} level, same day</td>"
    f"<td>{STATIONS[r]}</td></tr>"
    for s in ("deyr", "gu") for r in ("juba", "shabelle"))

# --- section 2: tracks the rivers
peak_rows = ""
KEY = {"Google Flood Hub": "google_grrr", "GloFAS v5": "glofas_v5", "GloFAS v4": "glofas_v4"}
for river, season, wname in (("juba", "deyr", "Juba Deyr"), ("shabelle", "deyr", "Shabelle Deyr"),
                             ("juba", "gu", "Juba Gu"), ("shabelle", "gu", "Shabelle Gu")):
    g = rp3c[(rp3c.river == river) & (rp3c.season == season)]
    adopted = MODEL[season]
    cells = "".join(f'<td class="n{" pick" if k == adopted else ""}">{g[KEY[k]].median():.2f}</td>'
                    for k in ("Google Flood Hub", "GloFAS v5", "GloFAS v4"))
    nmin, nmax = int(g.n_flood_seasons.min()), int(g.n_flood_seasons.max())
    peak_rows += f"<tr><td>{wname}</td>{cells}<td class=\"n\">{nmin} to {nmax}</td><td>{adopted}</td></tr>"
pooled = {k: rp3c[v].median() for k, v in KEY.items()}

# --- section 3: catches the floods
win_rows = ""
for wname in ("Deyr Juba", "Deyr Shabelle", "Gu Juba", "Gu Shabelle"):
    w = win[wname]; a = w["models"][w["adopted_model"]]
    win_rows += (f"<tr><td>{wname}</td><td class=\"n\">{len(w['severe_years'])}</td><td class=\"n\">{a['vs_severe']['hits']} of {len(w['severe_years'])}</td>"
                 f"<td class=\"n\">{a['vs_flood']['false_alarms']}</td><td class=\"n\">{a['auc_severe']:.2f}</td>"
                 f"<td>{', '.join(str(y) for y in a['activations'])}</td></tr>")

# --- section 6: flood by flood tables
def section(season):
    name, months = LABEL[season]
    rows, t = rows_all[season], totals[season]
    pm = t["per_model"]
    def counts(mdl):
        c = pm[mdl]; covered = t["n"] - c.get("na", 0)
        return c.get("before", 0), covered, c.get("after", 0) + c.get("same", 0), c.get("never", 0)
    gb, gc, ga, gn = counts("google_grrr"); vb, vc, va, vn = counts("glofas_v4")
    head = (f'<div class="head"><div class="g"><b>Google Flood Hub forecast</b><div class="big">{gb} of {gc} before the flood</div>'
            f'<div class="sub">{ga} after onset, {gn} never crossed; archive covers {gc} of {t["n"]} seasons (2016 to 2023)</div></div>'
            f'<div class="v4"><b>GloFAS v4 forecast (runs live)</b><div class="big">{vb} of {vc} before the flood</div>'
            f'<div class="sub">{va} after onset, {vn} never crossed; archive covers {vc} of {t["n"]} (2003 onward)</div></div></div>')
    hh = t["head_to_head"]
    h2h = (f" In the {t['n_both']} seasons both archives cover, Google was first in {hh['google_grrr']} and GloFAS v4 in {hh['glofas_v4']}."
           if t["n_both"] else "")
    body = "".join(
        f'<tr class="{"sev" if r["severe"] else ""}"><td>{r["year"]}{" *" if r["severe"] else ""}</td>'
        f'<td>{" and ".join(r["rivers"])}</td><td>{r["first_onset"]}</td>'
        f'<td>{pill(r["google_grrr"])}</td><td>{pill(r["glofas_v4"])}</td><td>{escape(r["first"])}</td></tr>' for r in rows)
    return (f"<h3>{name}, {months}</h3>"
            f'<p>{t["n"]} flood seasons on the gauges, {t["n_severe"]} severe (marked *).{h2h}</p>{head}'
            f"<div class='tw'><table><thead><tr><th>Year</th><th>River(s) that flooded</th><th>Flood began</th>"
            f"<th>Google forecast</th><th>GloFAS v4 forecast</th><th>First</th></tr></thead><tbody>{body}</tbody></table></div>")


# per-window row: chosen model, severe caught, agreement in flood seasons, activation years
by_window = ""
for river, season, wname, mname in (("juba", "deyr", "Deyr Juba", "Juba Deyr"), ("shabelle", "deyr", "Deyr Shabelle", "Shabelle Deyr"),
                                    ("juba", "gu", "Gu Juba", "Juba Gu"), ("shabelle", "gu", "Gu Shabelle", "Shabelle Gu")):
    w = win[wname]; a = w["models"][w["adopted_model"]]
    g = rp3c[(rp3c.river == river) & (rp3c.season == season)]
    rho = g[KEY[MODEL[season]]].median()
    by_window += (f"<tr><td>{wname}</td><td>{SOURCE[season]}</td><td class=\"n\">{a['vs_severe']['hits']} of {len(w['severe_years'])}</td>"
                  f"<td class=\"n\">{a['vs_flood']['false_alarms']}</td><td class=\"n\">{rho:.2f}</td>"
                  f"<td>{', '.join(str(y) for y in a['activations'])}</td></tr>")

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The case for the trigger | Somalia riverine flood trigger</title><style>{CSS}
details{{margin:18px 0}} summary{{cursor:pointer;font:700 17px/1.3 Georgia,serif;padding:6px 0}}
</style></head><body>
<header><div class="wrap"><small>Somalia Riverine Flood Trigger / summary</small>
<h1>The case for the trigger, in numbers</h1>
<p>Four river-season windows, each on the one forecast model that tracks that river in that season best. A window activates when enough of the seven monitored gauges are forecast above their own return-period level on the same day. Scored against the SWALIM gauge record, 1999 to 2023. Full analysis: <a href="index.html" style="color:#fff">trigger analysis page</a>.</p></div></header>
<div class="wrap">

<h2>The design</h2>
<div class='tw'><table><thead><tr><th>Window</th><th>Season</th><th>Model</th><th>Rule</th><th>Gauges</th></tr></thead><tbody>{design_rows}</tbody></table></div>
<p class="note">Thresholds are fitted on each model's own record, so a model that runs high is judged against itself. Readiness, 8 to 12 days ahead, runs on the GloFAS ensemble in all windows and releases the mobilisation share only.</p>

<h2>Does it work?</h2>
<div class="tiles">
{tile(f"{sev_caught} of {sev_all}", "severe flood seasons caught", "every season with two gauges over their 1-in-5 level activated a window")}
{tile(f"{n_act} in {n_years}", "years with an activation", "about one year in three; 1-in-3.2 on the Shabelle, 1-in-4.3 on the Juba")}
{tile(f"{len(outside)}", "activation with no gauge flood", f"{', '.join(map(str, outside))}, a year SWALIM and WFP both record as a major flood")}
{tile(f"{pooled['Google Flood Hub']:.2f} / {pooled['GloFAS v5']:.2f} / {pooled['GloFAS v4']:.2f}", "agreement with the gauges in flood seasons", "rank correlation over 1-in-3 seasons: Google / GloFAS v5 / GloFAS v4, the model running live")}
{tile(f"{lag_min:+d} to {lag_max:+d} days", "the chosen model leads the gauge", "best-fit lag at all 14 station-seasons; it rises before the river, never after")}
{tile("10 to 13 days", "warning in Gu on the Shabelle, from Google", "3 of 4 seasons since 2016; GloFAS v4 gave 3 days once and nothing in the other three")}
{tile("4 to 19 days", "warning in Deyr, from GloFAS", "4 of 7 flood seasons, on one river or the other; missed 2006, a day late in 2020 (v4, the version running live)")}
{tile(f"{sw_first} of {len(both_flag)}", "seasons SWALIM's bulletin came first", "so SWALIM alerts sit in readiness; action needs a forecast that can be backtested")}
</div>

<h2>By window</h2>
<div class='tw'><table><thead><tr><th>Window</th><th>Model</th><th class="n">Severe caught</th><th class="n">Activations with no flood</th><th class="n">Agreement in flood seasons</th><th>Activation years</th></tr></thead><tbody>{by_window}</tbody></table></div>
<p class="note">Agreement is the rank correlation between the chosen model's peak and the gauge's over the gauge's own 1-in-3 seasons, median across the window's gauges. Gu Shabelle reads near zero for every model because the gauge is capped at bank full in the largest floods, so no ordering of severity remains; a limit of the record, not of the models. Each window holds three to five severe seasons: every difference here is a one- or two-event difference.</p>

<details><summary>Flood by flood: which forecast would have caught it, and when</summary>
<p class="note">Every flood season on the gauges, 1999 to 2024. Either river counts, since any one window releases the full allocation. Dates are forecast issue dates, the day the alert would have gone out. Google's archive covers 2016 to mid-2023; GloFAS v4's 2003 to 2023 plus the operational Gu 2024 forecasts.</p>
{section("deyr")}
{section("gu")}
</details>

<details><summary>What runs live, and what is still open</summary>
<ul>
<li><b>GloFAS v4 runs both phases today.</b> Deyr is calibrated on GloFAS v5, which has no published forecast yet; Gu on Google Flood Hub, to which there is no API access yet. v4 stands in with levels refitted on its own record.</li>
<li><b>Bank full at a gauge is the fail-safe.</b> It would have activated in {len(fs_only)} seasons no forecast caught, including Gu 2023 on the Shabelle; where a forecast also activated, bank full came {min(fs_lag)} to {max(fs_lag)} days later. Coverage, not lead time.</li>
<li><b>Two gauges have stopped reporting.</b> Bardheere from 30 November 2023, Bualle from 14 March 2024.</li>
<li><b>Impact years are not yet defined.</b> The benchmark is gauge levels, not people affected. 2013 (activation, no gauge flood) and 2021 (bank full at one gauge, 400,000 affected, no activation) are why an impact cross-check is the next step.</li>
</ul>
</details>

<footer>Generated {datetime.date.today().isoformat()} from the trigger analysis outputs.</footer>
</div></body></html>"""
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT.name, "| em dashes:", html.count("—"))
print("envelope", n_act, "in", n_years, "| severe", sev_caught, "of", sev_all, "| outside", outside)
print("station rho", rho_min, rho_max, "| lag", lag_min, lag_max, "| RP3", rp3_hits, rp3_ev, "| RP5", rp5_hits, rp5_ev)
print("swalim first", sw_first, "of", len(both_flag), "| swalim only", len(sw_only), "| fail-safe only", len(fs_only), "| fs lag", min(fs_lag), max(fs_lag))
