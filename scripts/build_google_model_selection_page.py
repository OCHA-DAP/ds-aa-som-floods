"""Build pages/google-model-update/model-selection/: the design's model choice
re-run with the new Google model as a candidate (numbers in
scripts/google_model_selection.py; helpers and styling from
scripts/build_google_model_page.py).

    .venv/bin/python scripts/build_google_model_selection_page.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "page_additions"))

import build_google_model_page as B  # noqa: E402
import google_model_selection as S  # noqa: E402
import somlib as L  # noqa: E402
from src.constants import TRIGGER_CONFIG, TRIGGER_STATIONS  # noqa: E402

OUT = REPO / "pages" / "google-model-update" / "model-selection"
NICE = {"google_grrr": "Google (old)", "google_grrr_2026_11": "Google (new)", "glofas_v4": "GloFAS v4", "glofas_v5": "GloFAS v5"}
WNAME = {w: f"{w[1].title()} {w[0].title()}" for w in S.WINDOWS}
YEARS = list(L.SPAN)


def cfg_cells(combo):
    return "".join(f"<td>{NICE[c['model']]}<br><span class=muted>{c['n_req']} of {len(c['stations'])}, 1-in-{c['rp']}</span></td>"
                   for c in combo)


def season_cells(combo, years):
    """Year -> 'G', 'D' or 'G+D': which seasons release in that year."""
    by = {}
    for w, c in zip(S.WINDOWS, combo):
        for y in c["years"]:
            by.setdefault(y, set()).add(w[1])
    return {y: "".join(s[0].upper() for s in ("gu", "deyr") if s in by[y]) for y in by if y in years}


def main():
    res = S.run()
    OUT.mkdir(parents=True, exist_ok=True)
    old, new = res["old"], res["new"]
    sev, flood = res["severe"], res["any_flood"]
    A_, Bc, C_, D_ = old["capped"], new["capped"], new["years_cap_only"], new["seven"]
    rows_def = [("Adopted design (old Google, GloFAS v4/v5)", A_, "old"),
                ("New Google: design caps (8 years, 8 seasons)", Bc, "new"),
                ("New Google: years cap only (8 years)", C_, "new"),
                ("New Google: 7 years, 8 seasons", D_, "new")]
    for _, x, _ in rows_def:
        x[2]["seasons"] = len(S.released_seasons(x[1]))

    # ------------------------------------------------------------ facts the text states
    trk_new, trk_old = new["tracking"], old["tracking"]
    google_windows = {k: [w for w, c in zip(S.WINDOWS, x[1]) if c["model"] == S.NEW_G] for k, x in (("B", Bc), ("C", C_), ("D", D_))}
    assert google_windows["B"] == [("juba", "gu")]
    assert set(google_windows["C"]) == {("juba", "gu"), ("shabelle", "gu"), ("shabelle", "deyr")}
    assert all(("juba", "deyr") not in v for v in google_windows.values())
    shab_gu_b = dict(zip(S.WINDOWS, Bc[1]))[("shabelle", "gu")]
    assert shab_gu_b["model"] == "glofas_v5" and set(Bc[2]["years"]) - set(D_[2]["years"]) == {2005} and 2005 in flood
    assert len(new["ties"]) == 1
    assert min(trk_new[("shabelle", "gu")], key=trk_new[("shabelle", "gu")].get) == "glofas_v5"
    assert all(trk_new[w][S.NEW_G] > max(v for m, v in trk_new[w].items() if m != S.NEW_G) for w in S.WINDOWS)
    for _, x, _ in rows_def:
        assert x[2]["severe_caught"] == len(sev)

    # ------------------------------------------------------------ tracking table
    models = ["google_grrr", "google_grrr_2026_11", "glofas_v4", "glofas_v5"]
    t_rows = ""
    for w in S.WINDOWS:
        vals = {m: (trk_old[w].get(m) if m != S.NEW_G else trk_new[w].get(m)) for m in models}
        top = max((v, m) for m, v in vals.items() if v is not None)[1]
        t_rows += f"<tr><td>{WNAME[w]}</td>" + "".join(
            f"<td class=n>{'<b>' if m == top else ''}{vals[m]:.2f}{'</b>' if m == top else ''}"
            f"{' <span class=muted>(adopted)</span>' if TRIGGER_CONFIG[w]['source'] == m else ''}</td>" for m in models) + "</tr>"

    # ------------------------------------------------------------ design table
    d_rows = ""
    for lab, x, _ in rows_def:
        combo, r = x[1], x[2]
        d_rows += (f"<tr><td><b>{lab}</b></td>{cfg_cells(combo)}<td class=n>{r['fires']} ({B.rp_text(r['fires'])})</td>"
                   f"<td class=n>{r['seasons']}</td><td class=n>{r['severe_caught']} of {len(sev)}</td>"
                   f"<td>{B.yrs(r['no_flood_years'])}</td></tr>")

    # ------------------------------------------------------------ year grid
    grid_rows = []
    cells = {}
    for y in YEARS:
        cells[y] = (("S", "sev", f"{y}: severe (two gauges over 1-in-5 in a window)") if y in sev else
                    ("F", "fl", f"{y}: flood season (two gauges over 1-in-3)") if y in flood else ("", "no", f"{y}: no flood season"))
    grid_rows.append(("River gauges", cells))
    short = ["Adopted", "Design caps", "Years cap only", "7 years"]
    for (lab, x, _), sh in zip(rows_def, short):
        sc = season_cells(x[1], YEARS)
        grid_rows.append((sh, {y: (sc[y], "act", f"{y}: released in {' and '.join({'G': 'Gu', 'D': 'Deyr'}[ch] for ch in sc[y])}") if y in sc
                               else ("", "z", f"{y}: no activation") for y in YEARS}))
    year_grid = B.grid(grid_rows, "Activation years for the four designs")

    # ------------------------------------------------------------ forecast lead
    def ahead(issue, gauges):
        if issue is None:
            return "&ndash;"
        k = (gauges - issue).days
        return f"{B.d(issue)} <span class=muted>({'same day' if k == 0 else (f'{k} d before' if k > 0 else f'{-k} d after')})</span>"
    l_rows = ""
    for w in S.WINDOWS:
        rows = res["lead"][w]
        c = TRIGGER_CONFIG[w]
        for i, (y, row) in enumerate(rows.items()):
            l_rows += (f"<tr><td>{WNAME[w] if i == 0 else ''}{'<br><span class=muted>' + str(c['n_req']) + ' of ' + str(len(TRIGGER_STATIONS[w[0]])) + ', 1-in-' + str(c['rp']) + '</span>' if i == 0 else ''}</td>"
                       f"<td><b>{y}</b></td><td class=n>{B.d(row['gauges'])}</td>"
                       + "".join(f"<td class=n>{ahead(row[m], row['gauges'])}</td>" for m in (S.OLD_G, S.NEW_G, "glofas_v4")) + "</tr>")
    jd = res["lead"][("juba", "deyr")]
    assert list(jd) == [2017] and jd[2017]["glofas_v4"] is not None and jd[2017]["glofas_v4"] < jd[2017]["gauges"]
    assert jd[2017][S.OLD_G] is None and jd[2017][S.NEW_G] is None
    shab_deyr = res["lead"][("shabelle", "deyr")]
    assert sorted(shab_deyr) == [2019, 2020]
    assert all(r[S.NEW_G] is not None and (r["glofas_v4"] is None or r[S.NEW_G] < r["glofas_v4"]) for r in shab_deyr.values())

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>New Google model: re-running the model choice &mdash; Somalia Riverine Flood Trigger</title>
<meta name="description" content="Variant of the new-Google-model comparison: the design's choice of model per river-season window re-run with the new Google model as a candidate.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../../assets/site.css">
<style>
{B.STYLE}
table.yg td {{ width:26px; min-width:26px; font-size:10.5px; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <div class="inner">
      <p class="crumb"><a href="../../">Somalia Riverine Flood Trigger</a> / <a href="../">new Google model</a> / variant</p>
      <h1>New Google model: re-running the model choice</h1>
      <p>The <a href="../" style="color:#fff">comparison page</a> keeps the adopted design and swaps the new Google model in where
         Google is used (the two Gu windows). Here the design's choice of model for each river-season window is re-run with the new
         Google model as a candidate, alongside GloFAS versions 4 and 5. The monitoring points stay the seven SWALIM gauges: unlike
         Nigeria, Somalia never selected its points by correlation. Everything on 1999&ndash;2023. Analysis of {B.API_CHECKED}.</p>
    </div>
  </header>

  <article>
    <div class="stats">
      <div class="stat"><div class="v">4 of 4</div><div class="l">windows where the new Google model tracks the gauges better than GloFAS</div></div>
      <div class="stat"><div class="v">{len(google_windows['B'])} of 4</div><div class="l">windows it carries under the design's caps</div></div>
      <div class="stat"><div class="v">{len(google_windows['C'])} of 4</div><div class="l">windows it carries without the season cap</div></div>
      <div class="stat"><div class="v">{len(sev)} of {len(sev)}</div><div class="l">severe years caught by every option below</div></div>
    </div>

    <h2><span class="num">1</span>The short answer</h2>
    <ul>
      <li><strong>Tracking.</strong> The new Google model follows the river gauges more closely than GloFAS in all four windows. The
          change is largest in Deyr, where the old Google model was weak ({trk_old[('juba', 'deyr')]['google_grrr']:.2f} and
          {trk_old[('shabelle', 'deyr')]['google_grrr']:.2f}) and the new one is not ({trk_new[('juba', 'deyr')][S.NEW_G]:.2f} and
          {trk_new[('shabelle', 'deyr')][S.NEW_G]:.2f}). That weakness is why Deyr went to GloFAS.</li>
      <li><strong>The design's search, re-run.</strong> With the design's two caps (activate in at most 8 of 25 years, release in at most
          8 river-seasons), the search hands the new Google model only <b>Gu Juba</b>; GloFAS version 5 takes the other three windows.
          With the year cap alone, the new model also takes Gu Shabelle and Deyr Shabelle. Deyr Juba stays on GloFAS version 5 either way.</li>
      <li><strong>Forecast lead in Deyr.</strong> On the Shabelle, the new Google model's forecasts met the rule before GloFAS version 4's in
          both Deyr flood seasons of the shared archive (2019, 2020). On the Juba there is one Deyr flood season to judge by (2017): GloFAS met the
          rule well ahead and neither Google model met it.</li>
    </ul>
    <div class="callout warn"><strong>Where the search and the design's own principle disagree.</strong> Under both caps the search moves Gu
      Shabelle from Google to GloFAS version 5, which tracks that window worst of the three ({trk_new[('shabelle', 'gu')]['glofas_v5']:.2f}
      against {trk_new[('shabelle', 'gu')][S.NEW_G]:.2f} for the new Google model). It does so only because GloFAS activates in 2005, a flood
      season, which lifts the count from 7 to 8 years. The design's stated rule was that where the backtest cannot tell models apart,
      tracking and forecast lead decide. That points to the 7-year option: the adopted design with the new Google model in both Gu
      windows ({B.rp_text(D_[2]['fires'])}, all severe years, no activation outside a flood season). Which to adopt is for the working group.</div>

    <h2><span class="num">2</span>How each model tracks the gauges</h2>
    <p>The search's own measure, used to break ties: mean best-lag rank correlation of the window's points with the river's reference
       gauge (Luuq on the Juba, Belet Weyne on the Shabelle), 1999&ndash;2023. The comparison page reports a different measure (each point
       against its own gauge, as in the trigger page's station table), so the values differ; in Gu Juba the two measures rank the old and
       new Google models the other way round. Bold: best in the window.</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Window</th><th>Google (old)</th><th>Google (new)</th><th>GloFAS v4</th><th>GloFAS v5</th></tr></thead>
      <tbody>{t_rows}</tbody></table></div>

    <h2><span class="num">3</span>The design search, re-run</h2>
    <p>For every window the search tries each model, a station return period from 1-in-3 to 1-in-6 and a vote count from two to all-but-one
       points, with every window activating at least twice. It then takes the combination of the four windows with the most activation
       years under the caps, and among those the one that catches most severe years, has fewest activations outside a flood season, the
       fewest unanimous rules, the best tracking and the lowest return periods. Run on the old models it reproduces the adopted design
       exactly. Under both caps, the new-model pick is the only model assignment with its backtest score.</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Option</th><th>Gu Juba</th><th>Deyr Juba</th><th>Gu Shabelle</th><th>Deyr Shabelle</th>
      <th>Years</th><th>River-seasons</th><th>Severe caught</th><th>Outside a flood season</th></tr></thead><tbody>{d_rows}</tbody></table></div>
    <p>Activation years by option (G: released in Gu, D: in Deyr, GD: both). River gauges: F a flood season, S a severe one.</p>
    {year_grid}

    <h2><span class="num">4</span>Forecast lead</h2>
    <p>The other half of the design's judgement: which forecast reaches the rule first. For each window, the adopted rule (points and
       return period as designed) on each forecast archive against that model's own levels, at leads of 1 to 7 days, in the gauge flood
       seasons all three archives cover (Gu 2016&ndash;2023, Deyr 2016&ndash;2022): the first forecast issue that met the rule, and how long before
       the second gauge crossed its own 1-in-3 level. The GloFAS archive is issued about twice a week ({res['glofas_issues_per_year']:.0f} issues a
       year), Google's daily, which can cost GloFAS up to four days here. The live pipeline reads Google at leads 1&ndash;5 (see the
       <a href="../">comparison page</a>).</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Window</th><th>Year</th><th>Gauges over 1-in-3</th><th>Google (old)</th><th>Google (new)</th><th>GloFAS v4</th></tr></thead>
      <tbody>{l_rows}</tbody></table></div>

    <h2><span class="num">5</span>Method</h2>
    <ul>
      <li>Candidates exactly as in <code>scripts/model_selection.py</code> (models, return periods 1-in-3 to 1-in-6, two to all-but-one votes,
          tracking against the reference gauge, every window at least twice), judged on the two-gauge benchmark of
          <code>scripts/envelope_search.py</code>. GEOGloWS is left out, as in the adopted design.</li>
      <li>The two caps reconstruct the design's two decisions: activate no more often than 1-in-3 (8 of 25 years), and release in no more
          than 8 river-seasons (the change to Deyr Shabelle on 18 September 2026). With both, the search returns the adopted design on the old
          models, which the script asserts. The search is written out in <code>scripts/google_model_selection.py</code>; the pick of
          <code>model_selection.choose</code> is not used because it has no tracking tie-break and depends on candidate order.</li>
      <li>Dollow is the Juba main-stem gauge, as live; the design ran on the Dawa-branch point and the reconstruction still returns the adopted design.</li>
      <li>The Deyr windows were designed on GloFAS version 5 while the live forecast is version 4: see the
          <a href="../../glofas-version/">GloFAS version switch</a>.</li>
    </ul>
  </article>

  <p class="provenance">Generated {pd.Timestamp.today().day} {pd.Timestamp.today().strftime('%B %Y')} by
    <code>scripts/build_google_model_selection_page.py</code> (numbers in <code>scripts/google_model_selection.py</code>) in
    <a href="https://github.com/OCHA-DAP/ds-aa-som-floods">ds-aa-som-floods</a>. Back to the <a href="../">comparison page</a>.</p>
</div>
</body>
</html>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    summary = {"generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
               "tracking": {f"{w[1]}_{w[0]}": {**trk_old[w], S.NEW_G: trk_new[w][S.NEW_G]} for w in S.WINDOWS},
               "options": {lab: {"config": {f"{w[1]}_{w[0]}": v for w, v in S.config_of(x[1]).items()},
                                 "years": x[2]["years"], "river_seasons": x[2]["seasons"],
                                 "severe_caught": x[2]["severe_caught"], "outside_flood": x[2]["no_flood_years"]}
                           for lab, x, _ in rows_def}}
    (OUT / "data.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
