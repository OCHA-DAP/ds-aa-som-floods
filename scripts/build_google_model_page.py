"""Build pages/google-model-update/: the new Google model and the Gu windows.

Replays the adopted Gu rules on Google's November 2026 model with the method
held fixed and compares them with the model the windows were designed on
(scripts/google_model_update.py does the numbers). Needs data/processed/ with
the *_google_grrr_2026_11.parquet files (run `google_model_update.py fetch`
once).

    .venv/bin/python scripts/build_google_model_page.py
"""

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "page_additions"))

import google_model_update as G  # noqa: E402
import somlib as L  # noqa: E402
from sm_chart import SM_CSS, SM_JS, sm_panel  # noqa: E402
from src.constants import TRIGGER_CONFIG, TRIGGER_STATIONS  # noqa: E402

OUT = REPO / "pages" / "google-model-update"
OLD, NEW = G.OLD, G.NEW
C_OLD, C_NEW = "#2a78d6", "#eb6834"  # validated pair (dataviz validate_palette.js, light)
YEARS = list(range(L.Y0, 2026))
ST = {"dollow": "Dollow", "luuq": "Luuq", "bardheere": "Bardheere", "bualle": "Bualle",
      "belet_weyne": "Belet Weyne", "bulo_burti": "Bulo Burti", "jowhar": "Jowhar"}
API_CHECKED = "4 November 2026"


def e(x):
    return html.escape(str(x))


def yrs(v):
    v = sorted(v)
    return ", ".join(str(y) for y in v) if v else "none"


def rp_text(n, years=len(L.SPAN)):
    return f"1-in-{(years + 1) / n:.1f}" if n else "never"


def d(x):
    return f"{x.day} {x.strftime('%b')}" if x is not None and not pd.isna(x) else "&ndash;"


def diff(a, b):
    """New minus old, in days; positive = the new model is later."""
    if a is None or b is None:
        return "&ndash;"
    k = (pd.Timestamp(b) - pd.Timestamp(a)).days
    return "same day" if k == 0 else (f"{k} day" + ("" if abs(k) == 1 else "s") + " later" if k > 0
                                      else f"{-k} day" + ("" if abs(k) == 1 else "s") + " earlier")


def grid(rows, title):
    """Year grid: one row per series, a cell per year. rows: (label, {year: (text, cls, tip)})."""
    head = "".join(f"<th>{str(y)[2:]}</th>" for y in YEARS)
    body = ""
    for label, cells in rows:
        tds = ""
        for y in YEARS:
            txt, cls, tip = cells.get(y, ("", "nd", "no data"))
            tds += f'<td class="{cls}" title="{e(tip)}">{txt}</td>'
        body += f"<tr><th class=rl>{label}</th>{tds}</tr>"
    return (f'<div class="tablewrap"><table class="yg" aria-label="{e(title)}"><thead><tr><th class=rl></th>{head}'
            f"</tr></thead><tbody>{body}</tbody></table></div>")


def main():
    res = G.run()
    OUT.mkdir(parents=True, exist_ok=True)
    bench_all = {w: res[w]["bench_ext"] for w in G.GU}
    deyr_years = res["deyr"]

    # ------------------------------------------------------------ envelope
    union_old = set().union(*(res[w][OLD]["years"] for w in G.GU), *deyr_years.values())
    union_new = set().union(*(res[w][NEW]["years"] for w in G.GU), *deyr_years.values())
    severe_all = set().union(*(res[w]["bench"][1] for w in G.GU), *(b[1] for b in res["deyr_bench"].values()))
    flood_all = set().union(*(res[w]["bench"][0] for w in G.GU), *(b[0] for b in res["deyr_bench"].values()))
    env = {}
    for k, u in (("old", union_old), ("new", union_new)):
        u = set(u) & set(L.SPAN)
        env[k] = {"years": sorted(u), "n": len(u), "severe_caught": sorted(u & severe_all),
                  "severe_missed": sorted(severe_all - u), "no_flood": sorted(u - flood_all)}

    # ------------------------------------------------------------ per window
    win = {}
    for w in G.GU:
        c, r = TRIGGER_CONFIG[w], res[w]
        flood, severe, assess = r["bench_ext"]
        gx = L.gauge_crossings(*w, 3, span=YEARS)
        rows = []
        cells = {}
        for y in YEARS:
            if y not in assess:
                cells[y] = ("", "nd", f"{y}: fewer than two gauges reporting")
            elif y in severe:
                cells[y] = ("S", "sev", f"{y}: severe (two gauges over their 1-in-5 level)")
            elif y in flood:
                cells[y] = ("F", "fl", f"{y}: flood season (two gauges over their 1-in-3 level)")
            else:
                cells[y] = ("", "no", f"{y}: no flood season on the gauges")
        rows.append(("River gauges", cells))
        n_pts = len(TRIGGER_STATIONS[w[0]])
        for model, lab in ((OLD, "Old model"), (NEW, "New model")):
            v = r[model]["votes"]
            mx = v.groupby(v.index.year).max()
            cells = {}
            for y in YEARS:
                if y not in mx.index:
                    continue
                k = int(mx[y])
                act = k >= c["n_req"]
                cells[y] = (str(k), "act" if act else ("z" if k == 0 else "part"),
                            f"{y}: up to {k} of {n_pts} points at or over their level on one day"
                            + (" (rule met)" if act else ""))
            rows.append((f"{lab}: points over", cells))
        for model, lab in ((OLD, "Old model"), (NEW, "New model")):
            fi = r[model]["first_issue"]
            rf = L.reforecast(model)
            have = set(rf.loc[rf.valid_time.dt.month == 5, "valid_time"].dt.year)  # archive covers May
            cells = {}
            for y in YEARS:
                if y not in have:
                    continue
                cells[y] = (("&#9679;", "act", f"{y}: a forecast issued {d(fi[y][0])} met the rule")
                            if y in fi else ("", "z", f"{y}: no forecast at leads 1-7 met the rule"))
            rows.append((f"{lab}: forecast", cells))

        # timing table
        t_years = sorted(set(r[OLD]["first_cross"]) | set(r[NEW]["first_cross"])
                         | set(r[OLD]["first_issue"]) | set(r[NEW]["first_issue"]))
        trows = ""
        for y in t_years:
            oc, nc = r[OLD]["first_cross"].get(y), r[NEW]["first_cross"].get(y)
            oi, ni = r[OLD]["first_issue"].get(y, (None,))[0], r[NEW]["first_issue"].get(y, (None,))[0]
            gb = ("severe" if y in severe else "flood" if y in flood else
                  "no flood" if y in assess else "not assessable")
            trows += (f"<tr><td><b>{y}</b></td><td class=n>{d(oc)}</td><td class=n>{d(nc)}</td><td>{diff(oc, nc)}</td>"
                      f"<td class=n>{d(oi)}</td><td class=n>{d(ni)}</td><td>{diff(oi, ni)}</td>"
                      f"<td class=n>{d(gx.get(y))}</td><td>{gb}</td></tr>")

        # small multiples: the n_req-th highest point as a share of its own level
        panels = []
        for y in t_years:
            oc, nc = r[OLD]["first_cross"].get(y), r[NEW]["first_cross"].get(y)
            note = f"rule met: old {d(oc)} &middot; new {d(nc)}".replace("&ndash;", "no")
            panels.append(sm_panel(str(y), note, f"{y}-03-01", f"{y}-05-31",
                                   {"old": r[OLD]["margin"], "new": r[NEW]["margin"]},
                                   marks={"old": oc, "new": nc}))

        lv = pd.DataFrame({"old": r[OLD]["levels"], "new": r[NEW]["levels"],
                           "mo": r[OLD]["means"], "mn": r[NEW]["means"],
                           "to": pd.Series(r[OLD]["tracking"]), "tn": pd.Series(r[NEW]["tracking"])})
        lrows = "".join(
            f"<tr><td>{ST[s]}</td><td class=n>{x.old:,.0f}</td><td class=n><em>{x.new:,.0f}</em></td>"
            f"<td class=n>{x.new / x.old:.2f}x</td><td class=n>{x.mo:,.0f}</td><td class=n>{x.mn:,.0f}</td>"
            f"<td class=n>{x.to:.2f}</td><td class=n>{x.tn:.2f}</td></tr>" for s, x in lv.iterrows())

        sev, fl = r["bench"][1], r["bench"][0]
        perf = {}
        for model in (OLD, NEW):
            a = set(r[model]["years"])
            perf[model] = {"n": len(a), "rp": rp_text(len(a)), "severe": f"{len(a & sev)} of {len(sev)}",
                           "missed": yrs(sev - a), "outside": yrs(a - fl), "rho": np.median(list(r[model]["tracking"].values()))}
        sweep = "".join(
            f"<tr{' class=pick' if rp == c['rp'] else ''}><td>1-in-{rp}</td>"
            f"<td>{yrs(r[OLD]['sweep'][rp])}</td><td>{yrs(r[NEW]['sweep'][rp])}</td></tr>" for rp in (3, 4, 5, 6))
        win[w] = {"grid": grid(rows, f"{w[0]} Gu activations by year"), "timing": trows, "panels": panels,
                  "levels": lrows, "perf": perf, "sweep": sweep, "n_pts": n_pts, "c": c,
                  "lv": lv, "t_years": t_years}

    # ------------------------------------------------------------ headline numbers
    jw, sw = ("juba", "gu"), ("shabelle", "gu")
    shift_days = []
    for w in G.GU:
        r = res[w]
        for y in set(r[OLD]["first_cross"]) & set(r[NEW]["first_cross"]):
            shift_days.append((r[NEW]["first_cross"][y] - r[OLD]["first_cross"][y]).days)
        for y in set(r[OLD]["first_issue"]) & set(r[NEW]["first_issue"]):
            shift_days.append((r[NEW]["first_issue"][y][0] - r[OLD]["first_issue"][y][0]).days)
    smin, smax = min(shift_days), max(shift_days)
    # the wording below states these; stop if a rebuild changes them
    assert smin > 0, "the page says the new model is later in every shared year"
    assert 2013 in res[jw][OLD]["years"] and 2013 not in res[jw][NEW]["years"]
    assert 2013 in res[sw][OLD]["years"] and 2013 not in res[sw][NEW]["years"]
    assert all(2013 not in res[w]["bench"][0] for w in G.GU)
    assert 2023 in res[sw][NEW]["years"] and 2023 in res[sw]["bench"][1]
    assert 2024 in res[jw][NEW]["years_ext"] and 2024 in bench_all[jw][0]
    assert 2024 not in res[sw][NEW]["years_ext"] and 2024 in bench_all[sw][1]
    assert not any(2025 in res[w][NEW]["years_ext"] for w in G.GU)
    ratio_j = (win[jw]["lv"].new / win[jw]["lv"].old)
    ratio_s = (win[sw]["lv"].new / win[sw]["lv"].old)

    def window_section(w, num):
        x, r, c = win[w], res[w], TRIGGER_CONFIG[w]
        name = f"Gu {w[0].title()}"
        po, pn = x["perf"][OLD], x["perf"][NEW]
        spec = {"yMax": 2.0, "yTicks": [0, 0.5, 1, 1.5, 2], "aria": "share of the rule's level, old and new model",
                "series": [{"key": "old", "label": "Old model", "color": C_OLD},
                           {"key": "new", "label": "New model", "color": C_NEW}],
                "panels": x["panels"]}
        return f"""
    <h2><span class="num">{num}</span>{name}: {c['n_req']} of {x['n_pts']} points over their 1-in-{c['rp']} level</h2>
    <p>Each row of the grid is one series, each column one year. River gauges: <b>F</b> a flood season (two gauges over
       their own 1-in-3 level), <b>S</b> a severe one (two over 1-in-5). Model rows give the most points over their own
       level on any one day of March to May; a filled cell is a year the rule is met (at least {c['n_req']}). Forecast rows mark
       a year in which one forecast, at leads of 1 to 7 days, met the rule. Grey: no data (the old model ends in 2023, its
       forecasts start in 2016). Hover a cell for details.</p>
    {x['grid']}
    <div class="tablewrap"><table class="data cmp"><thead><tr><th></th><th>Old model</th><th>New model</th></tr></thead><tbody>
      <tr><td>Activations 1999&ndash;2023</td><td>{po['n']} ({po['rp']})</td><td><em>{pn['n']} ({pn['rp']})</em></td></tr>
      <tr><td>Severe Gu seasons caught</td><td>{po['severe']}</td><td>{pn['severe']}</td></tr>
      <tr><td>Severe Gu seasons missed</td><td>{po['missed']}</td><td>{pn['missed']}</td></tr>
      <tr><td>Activations outside a Gu flood season</td><td>{po['outside']}</td><td>{pn['outside']}</td></tr>
      <tr><td>Median correlation with the SWALIM gauges, March&ndash;May</td><td>{po['rho']:.2f}</td><td>{pn['rho']:.2f}</td></tr>
    </tbody></table></div>
    <h3>When the rule is met</h3>
    <p>First day the model's own record meets the rule, and first forecast issue date that meets it (leads 1&ndash;7 days, read
       as on the trigger page: valid day = issue day + lead). The last two columns are the reference from the river gauges: the
       day the second gauge crossed its own 1-in-3 level, and how the gauges classed the season.</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Year</th><th>Record: old</th><th>Record: new</th><th>Change</th>
      <th>Forecast: old</th><th>Forecast: new</th><th>Change</th><th>Gauges over 1-in-3</th><th>Gauge season</th></tr></thead>
      <tbody>{x['timing']}</tbody></table></div>
    <div class="sm-legend"><span><span class="sw" style="background:{C_OLD}"></span>Old model</span>
      <span><span class="sw" style="background:{C_NEW}"></span>New model</span>
      <span><span class="sw dash" style="border-color:#5c6b7a"></span>the rule's level (100%)</span></div>
    <div class="sm-grid" id="sm-{w[0]}"></div>
    <p class="figcap">Daily flow at the point ranked {c['n_req']}{'rd' if c['n_req'] == 3 else 'nd'} highest of {x['n_pts']}, as a share of
       that point's own 1-in-{c['rp']} level, March to May, on each model's record. The rule is met on days the line reaches
       100%; dots mark the first such day. Years shown: every year either model or either forecast met the rule.</p>
    <script>drawSM(document.getElementById('sm-{w[0]}'), {json.dumps(spec)});</script>
    <h3>Levels at each point</h3>
    <p>1-in-{c['rp']} level fitted on the March&ndash;May seasonal maxima 1999&ndash;2023 of each model's own record (Weibull
       plotting position, as in the trigger analysis) and mean March&ndash;May flow (m³/s), and the best-lag rank correlation with the
       point's SWALIM gauge, computed as in the trigger page's station table.</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Point</th><th>Old level</th><th>New level</th><th>New / old</th>
      <th>Old mean</th><th>New mean</th><th>Old correlation</th><th>New correlation</th></tr></thead><tbody>{x['levels']}</tbody></table></div>
    <details><summary>Other return periods, same vote count</summary>
    <div class="tablewrap"><table class="data"><thead><tr><th>Level</th><th>Old model, years</th><th>New model, years</th></tr></thead>
      <tbody>{x['sweep']}</tbody></table></div></details>"""

    jx, sx = win[jw], win[sw]
    dj = jx['perf'][NEW]['rho'] - jx['perf'][OLD]['rho']
    track_juba = "about as well as the old model" if abs(dj) < 0.02 else ("better than the old model" if dj > 0 else "less well than the old model")
    assert sx['perf'][NEW]['rho'] - sx['perf'][OLD]['rho'] >= 0.02, "the page says the Shabelle tracks better"
    assert all(len(res[w]['mismatch']) > len(res[w][OLD]['years']) for w in G.GU)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>New Google model and the Gu windows &mdash; Somalia Riverine Flood Trigger</title>
<meta name="description" content="Google released a new version of its flood model in November 2026. The Gu windows replayed on it with the same method: activation years, timing and levels, old against new.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css">
<style>
.hero.hero-sub {{ padding:40px 44px 34px; }} .hero.hero-sub h1 {{ font-size:28px; }}
.crumb {{ font-size:12px; margin:0 0 14px; }} .crumb a {{ color:rgba(255,255,255,.85); text-decoration:none; }}
article {{ padding:8px 44px 24px; max-width:960px; }}
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:21px; color:var(--n9); margin:40px 0 10px; line-height:1.25; }}
article h2 .num {{ color:var(--b5); font-size:15px; margin-right:8px; }}
article h3 {{ font-size:15px; color:var(--n9); margin:26px 0 6px; }}
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.65; }} article a {{ color:var(--b6); }} article strong {{ color:var(--n9); }}
article code {{ background:var(--n05); border:1px solid #e2e7e7; padding:1px 5px; border-radius:3px; font-size:12.5px; }}
table.data {{ border-collapse:collapse; width:100%; margin:14px 0; font-size:13px; }}
table.data th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:7px 10px 6px; vertical-align:bottom; }}
table.data td {{ border-bottom:1px solid #eef1f1; padding:7px 10px; color:var(--n8); vertical-align:top; }}
table.data td.n {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }} table.data td em {{ font-style:normal; color:var(--b6); font-weight:600; }}
table.data.cmp td:first-child {{ width:46%; }} table.data.cmp td {{ width:27%; }}
table.data tr.pick td {{ background:var(--b05); }} .tablewrap {{ overflow-x:auto; }} .muted {{ color:#6b7683; font-size:12px; }}
table.yg {{ border-collapse:separate; border-spacing:2px; margin:12px 0 6px; font-size:11.5px; }}
table.yg th {{ font-weight:500; color:#6b7683; font-size:10.5px; padding:0 0 2px; text-align:center; }}
table.yg th.rl {{ text-align:right; padding-right:8px; white-space:nowrap; color:var(--n8); font-size:12px; font-weight:500; }}
table.yg td {{ width:24px; min-width:24px; height:22px; text-align:center; border-radius:3px; background:#f3f5f7; color:#9aa4ae; font-variant-numeric:tabular-nums; }}
table.yg td.nd {{ background:repeating-linear-gradient(45deg,#f6f7f8 0 3px,#e9ecef 3px 5px); }}
table.yg td.z {{ color:#c4cbd2; }} table.yg td.part {{ color:var(--n8); background:#e6ebf0; }}
table.yg td.act {{ background:#1f5c8b; color:#fff; font-weight:700; }}
table.yg td.fl {{ background:#f6dfb8; color:#7a4d06; font-weight:700; }} table.yg td.sev {{ background:#d9822b; color:#fff; font-weight:700; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:26px 0 6px; }}
.stat {{ background:#fff; border:1px solid #e2e7e7; border-top:3px solid var(--b5); border-radius:5px; padding:14px 16px 12px; }}
.stat.flag {{ border-top-color:#e0a04b; }} .stat .v {{ font-family:'Merriweather',Georgia,serif; font-size:23px; font-weight:700; color:var(--b6); line-height:1.15; }}
.stat.flag .v {{ color:#b0722b; }} .stat .l {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--n7); margin-top:6px; line-height:1.45; }}
.callout {{ margin:20px 0; padding:13px 17px; border-radius:4px; background:var(--b05); border-left:5px solid var(--b5); font-size:13.5px; color:var(--n8); line-height:1.6; }}
.callout.warn {{ background:#fdf3e7; border-left-color:#e0a04b; }}
.figcap {{ font-size:12px !important; color:var(--n7) !important; line-height:1.5 !important; }}
details {{ margin:10px 0 0; font-size:13.5px; }} details summary {{ cursor:pointer; color:var(--b6); }}
.provenance {{ margin:34px 44px 40px; padding-top:14px; border-top:1px solid #e2e7e7; font-size:12px; color:var(--n7); line-height:1.7; }} .provenance a {{ color:var(--b6); }}
@media (max-width:640px) {{ article {{ padding:4px 22px 16px; }} .hero.hero-sub {{ padding:32px 22px 28px; }} .provenance {{ margin:28px 22px 32px; }} }}
{SM_CSS}
</style>
<script>{SM_JS}</script>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <div class="inner">
      <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / operational note</p>
      <h1>The new Google model and the Gu windows</h1>
      <p>Google has released a new version of the flood model behind Flood Hub (November 2026). The two Gu windows run on
         Google. This page replays them on the new model with exactly the method used to design them on the old one: the
         same seven points, the same March&ndash;May season, the same 1999&ndash;2023 calibration years, the same return
         periods and vote counts. Only the per-point levels are refitted, on the new model's own record. The Deyr windows
         and the readiness leg run on GloFAS and do not change. {API_CHECKED}.</p>
    </div>
  </header>

  <article>
    <div class="stats">
      <div class="stat"><div class="v">{jx['perf'][OLD]['n']} &rarr; {jx['perf'][NEW]['n']}</div><div class="l">Gu Juba activations 1999&ndash;2023, old &rarr; new</div></div>
      <div class="stat"><div class="v">{sx['perf'][OLD]['n']} &rarr; {sx['perf'][NEW]['n']}</div><div class="l">Gu Shabelle activations 1999&ndash;2023, old &rarr; new</div></div>
      <div class="stat"><div class="v">{rp_text(env['old']['n'])} &rarr; {rp_text(env['new']['n'])}</div><div class="l">all four windows together, old &rarr; new</div></div>
      <div class="stat flag"><div class="v">{smin}&ndash;{smax} days</div><div class="l">later on the new model, wherever both models meet the rule (record or forecast)</div></div>
    </div>

    <h2><span class="num">1</span>What changes</h2>
    <ul>
      <li><strong>Activation years.</strong> Gu Juba: {yrs(res[jw][OLD]['years'])} on the old model, {yrs(res[jw][NEW]['years'])} on the new.
          Gu Shabelle: {yrs(res[sw][OLD]['years'])} on the old, {yrs(res[sw][NEW]['years'])} on the new. Both windows drop 2013, which the river
          gauges did not record as a Gu flood season on either river; the Shabelle adds 2023, a severe Gu season on the gauges.</li>
      <li><strong>Timing.</strong> Wherever both models meet the rule in the same year, on the record or on the forecasts, the new model
          meets it later: by {smin} to {smax} days. The detail per year is in the timing tables below.</li>
      <li><strong>All four windows together</strong> (Deyr unchanged on GloFAS version 5, as designed): {env['old']['n']} activations
          ({rp_text(env['old']['n'])}) on the old model, {env['new']['n']} ({rp_text(env['new']['n'])}) on the new;
          severe years caught {len(env['old']['severe_caught'])} and {len(env['new']['severe_caught'])} of {len(severe_all)};
          activations outside a flood season: {yrs(env['old']['no_flood'])} and {yrs(env['new']['no_flood'])}.</li>
      <li><strong>Levels.</strong> The new model runs higher at these points, so the levels rise: Juba points {ratio_j.min():.1f}&ndash;{ratio_j.max():.1f}
          times the old levels, Shabelle points {ratio_s.min():.1f}&ndash;{ratio_s.max():.1f} times. Its March&ndash;May flows track the SWALIM
          gauges {track_juba} on the Juba (median correlation {jx['perf'][OLD]['rho']:.2f} &rarr; {jx['perf'][NEW]['rho']:.2f}) and better on the
          Shabelle ({sx['perf'][OLD]['rho']:.2f} &rarr; {sx['perf'][NEW]['rho']:.2f}).</li>
      <li><strong>After 2023</strong>, which only the new model covers: Gu Juba activates in {yrs(set(res[jw][NEW]['years_ext']) - set(L.SPAN))}
          (first day {d(res[jw][NEW]['first_cross'].get(2024))}), a flood season on the Juba gauges; Gu Shabelle does not activate in 2024,
          which its gauges record as severe. Neither window activates in 2025. These years sit outside the calibration record.</li>
    </ul>

    <div class="callout warn"><strong>Before Google switches Flood Hub to the new model.</strong> On {API_CHECKED} the Flood Hub API still
      serves the old model (same gauge identifiers, unchanged warning levels). The monitoring pipeline reads Google with Gu levels fitted
      on the old model, frozen in the pipeline's threshold file. If the API moves to the new model before those levels are refitted,
      the old levels would sit far below the new model's ordinary Gu flows: on the new record they would have activated Gu Juba in
      {len(res[jw]['mismatch'])} and Gu Shabelle in {len(res[sw]['mismatch'])} of the {len(YEARS)} Gu seasons 1999&ndash;2025. The refitted levels
      are the "new level" columns below. Signals that the switch has happened: Flood Hub's warning levels or gauge model identifiers change
      for these gauges, or the gauge identifiers move from <code>hybas_</code> to <code>subhybas_</code> as in the new dataset (not confirmed).</div>

    <h2><span class="num">2</span>The two versions</h2>
    <div class="tablewrap"><table class="data"><thead><tr><th></th><th>Old model (design)</th><th>New model (November 2026)</th></tr></thead><tbody>
      <tr><td>Record (reanalysis)</td><td>1980 &ndash; Dec 2023</td><td>Dec 1980 &ndash; Dec 2025</td></tr>
      <tr><td>Forecast archive (reforecast)</td><td>issued Jan 2016 &ndash; Jun 2023, leads 0&ndash;7 days</td><td>issued Jan 2016 &ndash; Dec 2025, leads 0&ndash;10 days</td></tr>
      <tr><td>Model</td><td>lumped, one model per basin</td><td>semi-distributed, routed along the HydroBASINS river network</td></tr>
      <tr><td>Points</td><td colspan=2>the same seven HydroBASINS basins, with identical outlet coordinates (Dollow on the Juba main stem, as live)</td></tr>
    </tbody></table></div>
    <p>Both versions label days the same way (the record for day T equals the forecast issued on day T+1 at lead 0), so the trigger
       page's reading of forecast dates applies to both unchanged. Leads 8&ndash;10 of the new forecast archive are not used here,
       to keep the action leg as designed.</p>
    {window_section(jw, 3)}
    {window_section(sw, 4)}

    <h2><span class="num">5</span>All four windows together</h2>
    <p>The allocation is sized on the union of the four windows. Deyr runs on GloFAS version 5 as designed and is the same in both
       rows ({yrs(deyr_years[('juba', 'deyr')])} on the Juba, {yrs(deyr_years[('shabelle', 'deyr')])} on the Shabelle). Severe years are
       those in which two gauges on a river crossed their own 1-in-5 level in either season ({yrs(severe_all)}).</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Gu windows on</th><th>Activations</th><th>Return period</th><th>Years</th><th>Severe caught</th><th>Severe missed</th><th>Outside a flood season</th></tr></thead><tbody>
      <tr><td>Old model</td><td class=n>{env['old']['n']}</td><td class=n>{rp_text(env['old']['n'])}</td><td>{yrs(env['old']['years'])}</td><td class=n>{len(env['old']['severe_caught'])} of {len(severe_all)}</td><td>{yrs(env['old']['severe_missed'])}</td><td>{yrs(env['old']['no_flood'])}</td></tr>
      <tr><td>New model</td><td class=n><em>{env['new']['n']}</em></td><td class=n>{rp_text(env['new']['n'])}</td><td>{yrs(env['new']['years'])}</td><td class=n>{len(env['new']['severe_caught'])} of {len(severe_all)}</td><td>{yrs(env['new']['severe_missed'])}</td><td>{yrs(env['new']['no_flood'])}</td></tr>
    </tbody></table></div>
    <p>The windows were calibrated jointly to about 1-in-3 on the old model. On the new model the same settings activate less often
       ({rp_text(env['new']['n'])}). Whether to re-tune for the new model is a design decision for the working group; this page only
       shows the adopted settings carried over.</p>

    <h2><span class="num">6</span>Method</h2>
    <ul>
      <li>Exactly the trigger page's code (<code>scripts/page_additions/somlib.py</code>), run on each model's data in turn:
          per-point levels by Weibull plotting position with log-log interpolation on March&ndash;May maxima 1999&ndash;2023; a day
          counts when at least the required number of points are at or over their own level; forecasts at leads 1&ndash;7 days
          against the same levels.</li>
      <li>The river-gauge benchmark is unchanged (two SWALIM gauges over their own level, levels fitted 2000&ndash;2023). For 2024
          and 2025 the same levels are applied to the gauges still reporting.</li>
      <li>The old model run here reproduces the adopted Gu activation years (2013, 2016, 2018, 2020 on both rivers).</li>
      <li>Data: Google Runoff Reanalysis &amp; Reforecast, public Google Cloud bucket <code>flood-forecasting</code>,
          old version <code>model_id_8583a5c2_v0</code>, new version <code>distributed_model_2026_11</code> (CC-BY-4.0).</li>
    </ul>
  </article>

  <p class="provenance">Generated {pd.Timestamp.today().day} {pd.Timestamp.today().strftime('%B %Y')} by <code>scripts/build_google_model_page.py</code> (numbers in
    <code>scripts/google_model_update.py</code>) in <a href="https://github.com/OCHA-DAP/ds-aa-som-floods">ds-aa-som-floods</a>.
    Design pages: <a href="../trigger-single-model/summary.html">trigger summary</a>, <a href="../trigger-single-model/">analysis</a>.
    Related: <a href="../glofas-version/">GloFAS version switch</a>. Live: <a href="../monitoring/">monitoring</a>.</p>
</div>
</body>
</html>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")

    def ser(r):
        return {k: (sorted(v) if isinstance(v, (set, list)) else v) for k, v in r.items()}
    summary = {
        "generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "api_checked": API_CHECKED,
        "windows": {f"{w[1]}_{w[0]}": {
            "rule": TRIGGER_CONFIG[w],
            "levels": {m: res[w][m]["levels"].round(1).to_dict() for m in (OLD, NEW)},
            "years": {m: res[w][m]["years"] for m in (OLD, NEW)},
            "years_to_2025": {m: res[w][m]["years_ext"] for m in (OLD, NEW)},
            "first_day": {m: {y: str(x.date()) for y, x in res[w][m]["first_cross"].items()} for m in (OLD, NEW)},
            "first_forecast_issue": {m: {y: str(x[0].date()) for y, x in res[w][m]["first_issue"].items()} for m in (OLD, NEW)},
            "tracking": {m: res[w][m]["tracking"] for m in (OLD, NEW)},
            "old_levels_on_new_flows": res[w]["mismatch"],
        } for w in G.GU},
        "envelope": {k: ser(v) for k, v in env.items()},
        "severe_years": sorted(severe_all),
    }
    (OUT / "data.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}")
    print("envelope", {k: (v["n"], v["years"]) for k, v in env.items()}, "shift", smin, smax)


if __name__ == "__main__":
    main()
