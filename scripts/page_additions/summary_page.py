"""Summary page: for each flood event, when would the Google Flood Hub forecast have caught it,
and when would the GloFAS v4 forecast, either river counting. Forecast side only.
Writes wt-trigger/pages/trigger-single-model/summary.html and summary_rows.json."""
import datetime
import json
from html import escape
from pathlib import Path

import pandas as pd

import somlib as L

S = Path(__file__).parent
OUT = S / "wt-trigger/pages/trigger-single-model/summary.html"
RULE = {("juba", "deyr"): (4, 3), ("shabelle", "deyr"): (4, 2),   # adopted return period, points required
        ("juba", "gu"): (5, 3), ("shabelle", "gu"): (6, 2)}
ARCHIVE = {"google_grrr": (2016, 2023), "glofas_v4": (2003, 2024)}
SPAN = list(range(1999, 2025))
LABEL = {"deyr": ("Deyr", "October to December"), "gu": ("Gu", "March to May")}

# Gu 2024: the reforecast archives end in 2023; the v4 operational forecast for Gu 2024 was
# captured separately (gu2024_issue.py). Google has no record for 2024.
G24 = json.load(open(S / "gu2024_issue.json"))
V4_2024 = {r: (pd.Timestamp(f"2024 {G24[r]['v4_fc_issue'][0]}") if G24[r]["v4_fc_issue"][0] else None)
           for r in ("juba", "shabelle")}


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


rows_all, totals = {}, {}
for season in ("deyr", "gu"):
    onset, severe, fc = {}, {}, {"google_grrr": {}, "glofas_v4": {}}
    for river in ("juba", "shabelle"):
        rp, n = RULE[(river, season)]
        onset[river] = L.gauge_crossings(river, season, 3, span=SPAN)
        severe[river] = set(L.gauge_crossings(river, season, L.SEVERE_RP, span=SPAN))
        for m in fc:
            fc[m][river] = {y: v[0] for y, v in L.first_issue_dates(m, river, season, rp, n, span=SPAN, leads=(1, 7)).items()}
        if season == "gu" and V4_2024[river] is not None:
            fc["glofas_v4"][river][2024] = V4_2024[river]
    years = sorted({y for r in onset for y in onset[r] if y in SPAN and (season == "gu" or y <= 2023)})
    rows, tot = [], {m: {} for m in fc}
    both = {m: 0 for m in fc}; n_both = 0
    for y in years:
        flooded = [r for r in onset if y in onset[r]]
        first = min(onset[r][y] for r in flooded)
        sev = any(y in severe[r] for r in flooded)
        rec = {"year": y, "rivers": [r.title() for r in flooded], "severe": sev, "first_onset": dfmt(first)}
        leads = {}
        for m in fc:
            c = {r: fc[m][r][y] for r in fc[m] if y in fc[m][r]}
            when, river = (min(c.values()), min(c, key=c.get)) if c else (None, None)
            kind, text, lead = verdict(first, when, y, m)
            rec[m] = {"kind": kind, "text": text, "river": river.title() if river else None,
                      "when": dfmt(when) if when is not None else None}
            tot[m][kind] = tot[m].get(kind, 0) + 1
            leads[m] = (kind, lead)
        # who was first, only where both archives cover the year
        g, v = leads["google_grrr"], leads["glofas_v4"]
        if g[0] != "na" and v[0] != "na":
            n_both += 1
            gl = g[1] if g[1] is not None else -999; vl = v[1] if v[1] is not None else -999
            if gl > vl: rec["first"] = "Google"; both["google_grrr"] += 1
            elif vl > gl: rec["first"] = "GloFAS v4"; both["glofas_v4"] += 1
            else: rec["first"] = "tie" if g[1] is not None else "neither"
        else:
            rec["first"] = ""
        rows.append(rec)
    rows_all[season] = rows
    totals[season] = {"per_model": tot, "head_to_head": both, "n_both": n_both,
                      "n": len(rows), "n_severe": sum(r["severe"] for r in rows)}

json.dump({"rows": rows_all, "totals": totals}, open(S / "summary_rows.json", "w"), indent=1, default=str)

CSS = """
:root{--ink:#111827;--muted:#6b7280;--rule:#e5e7eb;--ok:#166534;--okbg:#dcfce7;--late:#9a3412;--latebg:#ffedd5;
--miss:#991b1b;--missbg:#fee2e2;--na:#6b7280;--nabg:#f3f4f6;--green:#1f6f5f;--g:#1d4ed8;--v4:#374151}
body{margin:0;font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#fff}
header{background:var(--green);color:#fff;padding:40px 0 30px}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
header h1{font:700 30px/1.15 Georgia,"Times New Roman",serif;margin:6px 0 12px}
header p{font-size:16.5px;max-width:820px;margin:0;opacity:.95}
header small{opacity:.8;letter-spacing:.02em}
h2{font:700 22px/1.2 Georgia,serif;margin:40px 0 6px}
p.lead{max-width:820px;margin:0 0 14px}
p.note{color:var(--muted);font-size:13.5px;max-width:820px}
.head{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0 18px}
.head>div{border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
.head b{display:block;font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px}
.head .big{font:700 26px/1.1 Georgia,serif;margin:2px 0 6px}
.head .g .big{color:var(--g)} .head .v4 .big{color:var(--v4)}
.head .sub{color:var(--muted);font-size:13.5px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:2px solid var(--ink);padding:8px;font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}
td{padding:8px;border-bottom:1px solid var(--rule);vertical-align:top}
tr.sev td:first-child{font-weight:700}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12.5px;font-weight:600;white-space:nowrap}
.before{background:var(--okbg);color:var(--ok)} .same,.after{background:var(--latebg);color:var(--late)}
.never{background:var(--missbg);color:var(--miss)} .na{background:var(--nabg);color:var(--na)}
.who{color:var(--muted);font-size:12px;display:block}
td.first{font-weight:600}
footer{color:var(--muted);font-size:13px;margin:40px 0 30px;border-top:1px solid var(--rule);padding-top:14px}
"""


def pill(v):
    who = f'<span class="who">{escape(v["river"])}, {escape(v["when"])}</span>' if v["river"] else ""
    return f'<span class="pill {v["kind"]}">{escape(v["text"])}</span>{who}'


def section(season):
    name, months = LABEL[season]
    rows, t = rows_all[season], totals[season]
    pm = t["per_model"]
    def counts(m):
        c = pm[m]; covered = t["n"] - c.get("na", 0)
        return c.get("before", 0), covered, c.get("after", 0) + c.get("same", 0), c.get("never", 0)
    gb, gc, ga, gn = counts("google_grrr"); vb, vc, va, vn = counts("glofas_v4")
    head = (f'<div class="head"><div class="g"><b>Google Flood Hub forecast</b><div class="big">{gb} of {gc} caught before the flood</div>'
            f'<div class="sub">{ga} after onset, {gn} never crossed; archive covers {gc} of the {t["n"]} flood seasons (2016 to 2023)</div></div>'
            f'<div class="v4"><b>GloFAS v4 forecast (runs live)</b><div class="big">{vb} of {vc} caught before the flood</div>'
            f'<div class="sub">{va} after onset, {vn} never crossed; archive covers {vc} of the {t["n"]} (2003 onward)</div></div></div>')
    hh = t["head_to_head"]
    h2h = (f"In the {t['n_both']} seasons both archives cover, Google was first in {hh['google_grrr']} and GloFAS v4 in {hh['glofas_v4']}."
           if t["n_both"] else "")
    body = []
    for r in rows:
        body.append(f'<tr class="{"sev" if r["severe"] else ""}"><td>{r["year"]}{" *" if r["severe"] else ""}</td>'
                    f'<td>{" and ".join(r["rivers"])}</td><td>{r["first_onset"]}</td>'
                    f'<td>{pill(r["google_grrr"])}</td><td>{pill(r["glofas_v4"])}</td><td class="first">{escape(r["first"])}</td></tr>')
    return (f"<h2>{name}, {months}</h2>"
            f'<p class="lead">{t["n"]} flood seasons on the gauges, {t["n_severe"]} of them severe (marked *). {h2h}</p>'
            f"{head}"
            f"<table><thead><tr><th>Year</th><th>River(s) that flooded</th><th>Flood began</th>"
            f"<th>Google forecast</th><th>GloFAS v4 forecast</th><th>First</th></tr></thead><tbody>{''.join(body)}</tbody></table>")


html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Google against GloFAS v4, flood by flood | Somalia riverine flood trigger</title><style>{CSS}</style></head><body>
<header><div class="wrap"><small>Somalia Riverine Flood Trigger / summary</small>
<h1>Which forecast would have caught each flood, and when</h1>
<p>Every flood season the gauges recorded, 1999 to 2024. For each, the first day the Google Flood Hub forecast and the GloFAS v4 forecast would have met the window's rule on either river, compared with the day the flood began. Any one window activating releases the full allocation, so activation on either river counts.</p></div></header>
<div class="wrap">
<p class="note">A flood season means two of a river's gauges over their own 1-in-3 level; severe means two over 1-in-5. "Flood began" is the day the second gauge crossed on whichever river flooded first. Forecast dates are issue dates at lead times of 1 to 7 days: the day the alert would have gone out. Each model is run at the window's adopted rule (3 of 4 points on the Juba, 2 of 3 on the Shabelle; 1-in-4 in Deyr, 1-in-5 Juba and 1-in-6 Shabelle in Gu) against thresholds fitted on its own record. The Google reforecast covers 2016 to mid-2023; GloFAS v4 covers 2003 to 2023, plus the operational Gu 2024 forecasts.</p>
{section("deyr")}
{section("gu")}
<footer>Full analysis, skill scores, station detail and the SWALIM comparison: <a href="index.html">trigger analysis page</a>. Generated {datetime.date.today().isoformat()}.</footer>
</div></body></html>"""
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT.name)
for s in ("deyr", "gu"):
    print(s, json.dumps(totals[s]))
