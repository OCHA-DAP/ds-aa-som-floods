"""Assemble the new trigger-page sections from the analysis outputs and fold the
station-level material into one section. Idempotent: each inserted section sits
between HTML comment markers and is replaced on rerun; moved blocks are moved once.
Run toc_inject.py afterwards."""
import json
import re
from pathlib import Path

S = Path(__file__).parent
PAGE_DIR = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model"
PAGE = PAGE_DIR / "index.html"
V = "?v=202609072"
t = PAGE.read_text(encoding="utf-8")

metrics = json.loads((S / "metrics.json").read_text(encoding="utf-8"))
obs = json.loads((S / "obs_fallback.json").read_text(encoding="utf-8"))
tl = json.loads((S / "model_timeline.json").read_text(encoding="utf-8"))
metrics_tables = (S / "metrics_tables.html").read_text(encoding="utf-8")
station_table = (S / "station_metrics_table.html").read_text(encoding="utf-8")
obs_table = (S / "obs_fallback_table.html").read_text(encoding="utf-8")


def replace_section(t, key, html, before_anchor):
    """Put html (wrapped in markers) immediately before before_anchor; replace if present."""
    start, end = f"<!-- begin {key} -->", f"<!-- end {key} -->"
    block = f"{start}\n{html}\n{end}\n"
    if start in t:
        i, j = t.find(start), t.find(end) + len(end) + 1
        return t[:i] + block + t[j:]
    k = t.find(before_anchor)
    assert k > 0, before_anchor
    return t[:k] + block + t[k:]


def cut(t, start_marker, end_marker, include_end=False):
    """Remove the block from start_marker to end_marker (exclusive unless include_end) and return (t, block)."""
    i = t.find(start_marker)
    if i < 0:
        return t, ""
    j = t.find(end_marker, i)
    assert j > i, end_marker
    if include_end:
        j += len(end_marker)
    return t[:i] + t[j:], t[i:j]


def auc_line(key):
    parts = []
    for w in metrics["windows"]:
        a = {m: w["models"][m][key] for m in w["models"]}
        best = max(a, key=a.get)
        nice = {"google_grrr": "Google", "glofas_v5": "GloFAS v5", "glofas_v4": "GloFAS v4"}
        parts.append(f"{w['window']}: " + ", ".join(f"{nice[m]} {a[m]:.2f}" for m in ("google_grrr", "glofas_v5", "glofas_v4"))
                     + f" (best {nice[best]})")
    return "; ".join(parts)


# ------------------------------------------------------------------ 1. skill scores
E = metrics["envelope"]["vs_severe"]
skill = f"""    <h2>Skill scores beyond POD, FAR and F1</h2>
    <p>The rest of the page scores the trigger with POD (probability of detection, the
      same quantity as recall), FAR (false alarm ratio: activations with no flood behind
      them, as a share of all activations) and F1. The tables below give the full
      contingency table and the scores derived from it, for each window's rule on each
      candidate model at the adopted return period and point count, 1999 to 2023. All 25
      years are counted, as elsewhere on the page. The Juba gauges did not report in 1999 to
      2001 and the Shabelle gauges in 1999, so those years count as non-flood years.</p>
    <ul>
      <li><strong>POFD</strong>, the false alarm rate: false alarms divided by non-flood
        years. It does not depend on how many activations there were.</li>
      <li><strong>CSI</strong> (critical success index): hits divided by hits plus misses
        plus false alarms. Correct negatives are left out.</li>
      <li><strong>Bias</strong>: activations divided by flood years. Above 1 the rule
        activates more often than floods occur.</li>
      <li><strong>PSS</strong> (Peirce skill score): POD minus POFD. Zero for a rule that
        activates at random, one for a perfect rule.</li>
      <li><strong>HSS</strong> (Heidke skill score): correct calls beyond chance, using all
        four cells.</li>
      <li><strong>AUC</strong>: the area under the ROC curve. The curve sweeps the station
        return period from 1-in-1.3 to 1-in-12 at the window's point count and plots POD
        against POFD at each setting. 0.5 is chance; 1 means every flood year ranks above
        every non-flood year.</li>
    </ul>
    <figure><img src="figs/l_roc.png{V}" alt="ROC curves per window and model, against severe and flood years">
      <figcaption>Top row against the severe years (two gauges over 1-in-5), bottom row
        against the flood years (two gauges over 1-in-3). Each curve is one model's window
        rule as its station return period is relaxed; the diamond is the adopted model at its
        adopted return period. With three to seven flood years per window the curves are
        coarse, and the AUC ranks the models rather than measuring them precisely.</figcaption></figure>
""" + metrics_tables + f"""
    <p>Against the severe years the envelope has POD {E['POD']:.2f}, FAR {E['FAR']:.2f} and POFD
      {E['POFD']:.2f}: seven hits, no misses, one activation outside the benchmark (2013).
      CSI {E['CSI']:.2f}, bias {E['bias']:.2f}, PSS {E['PSS']:.2f}, HSS {E['HSS']:.2f}. Window
      scores are lower than the envelope's because a window is judged on its own
      river-season, while the envelope is credited with a year whichever window caught it.
      The AUC ranks the models as the selection did. Against severe years GloFAS v5 has the
      largest area on the Shabelle in Deyr (0.99; Google 0.93, v4 0.75). Google has the
      largest on the Juba in Gu (0.95) and ties with v5 on the Shabelle in Gu (0.89). On the
      Juba in Deyr no model separates the years well (0.80 to 0.88, v4 ahead of v5). Against
      the 1-in-3 flood years GloFAS v5 and v4 rank the Gu years better than Google (0.96 to
      0.99 against 0.92 to 0.95); the difference is the single-station false alarms in the
      Google Gu windows.</p>
    <div class="callout warn">
      <strong>Benchmark note.</strong> These tables use the two-gauge benchmark as the repo's
      code now computes it, with gauge levels fitted on 2000 to 2023. On that record there
      are seven severe years (2006, 2014, 2016, 2018, 2019, 2020, 2023); 2008 is a flood year
      but not a severe one. Earlier sections of this page were generated before the fit
      window was corrected and still count 2008 as the eighth severe year and the one miss.
      The activation years are the same in both; only the label on 2008 differs.
    </div>
"""
t = replace_section(t, "skill-scores", skill, "    <h2>Why the models perform the way they do</h2>")

# ------------------------------------------------------------------ 2. station by station
t, gauge_block = cut(t, "    <h3>How each model maps the RP3 events, gauge by gauge</h3>",
                     "    <p>The same comparison at the level the trigger operates:")
t, peaks_block = cut(t, '    <p class="muted carrynote"><em>The seasonal-peak comparison', "</div></details>\n", include_end=True)
t, ownskill_block = cut(t, '    <p class="muted carrynote"><em>The forecast-vs-own-reanalysis comparison', "</div></details>\n", include_end=True)
if "<!-- begin station-by-station -->" in t:
    # a previous run already built the section: recover any block the cuts above did not find
    i, j = t.find("<!-- begin station-by-station -->"), t.find("<!-- end station-by-station -->")
    old = t[i:j]
    if not gauge_block:
        a, b = old.find("    <h3>How each model maps"), old.find("    </div></details>", old.find("    <h3>How each model maps"))
        gauge_block = old[a:b] if a >= 0 else ""
    if not peaks_block:
        m = re.search(r'    <p class="muted carrynote"><em>The seasonal-peak comparison.*?</div></details>\n', old, re.S)
        peaks_block = m.group(0) if m else ""
    if not ownskill_block:
        m = re.search(r'    <p class="muted carrynote"><em>The forecast-vs-own-reanalysis comparison.*?</div></details>\n', old, re.S)
        ownskill_block = m.group(0) if m else ""
assert gauge_block and peaks_block and ownskill_block, "a carried-over block went missing; restore the page from git first"
# the two carried-over studies become sub-sections of this one (h3), so the contents list stays at h2 level
peaks_block = re.sub(r'<h2(?: id="[^"]*")?>Seasonal peaks: model vs gauge</h2>', "<h3>Seasonal peaks: model vs gauge</h3>", peaks_block)
ownskill_block = re.sub(r"<h2(?: id=\"[^\"]*\")?>Forecast skill against each model's own reanalysis</h2>",
                        "<h3>Forecast skill against each model's own reanalysis</h3>", ownskill_block)
gauge_block = gauge_block.replace("<h3>How each model maps the RP3 events, gauge by gauge</h3>",
                                  "<h3>How each model maps the 1-in-3 events, gauge by gauge (event matching within 7 days)</h3>")
station = """    <h2>Station by station</h2>
    <p>This section gathers the station-level results. The table scores Google and GloFAS
      v5 at each of the seven points, per season, on two things: how closely its daily
      series ranks the days like the gauge's level record (Spearman rank correlation at the
      best lag between minus 10 and plus 30 days, positive when the model leads the gauge),
      and whether the model's own 1-in-3 and 1-in-5 crossings in a season match the gauge's
      own crossings, counted by year over the years the gauge reported (at least 30 readings
      in the season, 2000 to 2023). Gauge levels are fitted on 2000 to 2023 and model
      thresholds on 1999 to 2023, as elsewhere on the page. The adopted model for each window
      is in bold. Bardheere's and Bualle's records are short or suspect (see the tail-ratio
      figure above), so their rows carry less weight.</p>
""" + station_table + """
    <p>In Gu, Google has the highest rank correlation with the Juba gauges (0.84 to 0.88 at
      Dollow, Luuq, Bardheere and Bualle) and is level with GloFAS v5 on the Shabelle. In
      Deyr, GloFAS v5 tracks every point better (0.76 to 0.84, against 0.51 to 0.68 for
      Google, whose best lags on the Juba are negative: it trails the gauge). Single-point
      detection is weak for both models. At 1-in-3 a point
      catches one to three of its gauge's events with one to five false alarms, and the
      models disagree on which years those are. This is the basis for requiring a consensus
      of points and for judging the trigger at window level.</p>
    <details class="supp"><summary>Event matching per gauge: hit and false-alarm rates on 1-in-3 crossings</summary>
    <div class="supp-body">
""" + gauge_block + """
    </div></details>
""" + peaks_block + ownskill_block
t = replace_section(t, "station-by-station", station, "    <h2>Thresholds and calibration</h2>")

# ------------------------------------------------------------------ 3. model timeline by year
def lead_txt(n):
    if n is None:
        return "never"
    return "on the day" if n == 0 else (f"{n} d before" if n > 0 else f"{-n} d after")


tl_rows = "".join(
    f"<tr><td>{r['window']}</td><td>{r['year']}{' *' if r['severe'] else ''}</td>"
    f"<td>{lead_txt(r['google_rean'])}</td><td>{lead_txt(r['glofas_v5_rean'])}</td>"
    f"<td>{lead_txt(r['google_issue']) if r['google_archive'] else 'no archive'}</td>"
    f"<td>{lead_txt(r['glofas_v4_issue']) if r['v4_archive'] else 'no archive'}</td>"
    f"<td>{lead_txt(r['onset5_lead']) if r['onset5_lead'] is not None else 'not reached'}</td></tr>"
    for r in tl)
timeline = f"""    <h2>When each model would have flagged, year by year</h2>
    <p>For every flood season on the two-gauge benchmark (1999 to 2023), the chart shows how
      far ahead of the gauges' 1-in-3 crossing each model would have raised a flag under the
      window's rule, whichever model the window adopted. Two kinds of date are shown.
      Diamonds: the first day the model's reanalysis crossed, a flow date. Triangles: the
      first forecast issue that crossed at leads 1 to 7, the date the flag would have been in
      hand. The GloFAS v4 reforecast exists from 2003 and is issued twice a week, so an issue
      can be up to three days later than a daily product; the Google reforecast covers 2016
      to mid-2023. The green band is the action window, 1 to 7 days before onset.</p>
    <figure><img src="figs/l_model_timeline.png{V}" alt="Lead of Google and GloFAS flags before the gauges' 1-in-3 crossing, per flood season">
      <figcaption>One row per flood season; an asterisk marks a severe year. Upper track
        Google (blue), lower track GloFAS (v5 reanalysis and v4 forecast). Text at right gives
        the leads in days. Onset is the first day two gauges of the river had crossed their
        1-in-3 level; the black tick is the 1-in-5 crossing.</figcaption></figure>
    <div class="tablewrap">
    <table class="data" style="font-size:12px"><thead><tr><th>window</th><th>year</th>
      <th>Google reanalysis</th><th>GloFAS v5 reanalysis</th><th>Google forecast issue</th>
      <th>GloFAS v4 forecast issue</th><th>gauges 1-in-5</th></tr></thead>
    <tbody>{tl_rows}</tbody></table></div>
    <p>On the Shabelle in Deyr, Google's reanalysis crosses 6 to 11 days before onset in
      2006, 2014, 2019 and 2023; GloFAS v5 crosses 0 to 4 days before, and 3 days after in
      2023. Google misses Deyr 2020, which v5 catches 9 days late. On the Juba in Deyr
      neither model is early: both cross after onset in 2006 and 2023, and neither crosses in
      2014. In Gu both models cross after onset in every Juba flood (1 to 8 days late in
      2016, 2018 and 2020) and both miss Gu 2023 on both rivers. On the Shabelle in Gu 2020
      Google is 5 days early and v5 2 days. Where forecast archives exist, the first issue
      that crossed is almost always earlier than the reanalysis crossing. The Google forecast
      crossed 10 to 13 days before onset in Gu 2016, 2018 and 2020 on the Shabelle and 12
      days before in Deyr 2019; the v4 forecast 12 to 13 days before in Deyr 2014 and 2019 on
      the Shabelle and 4 days before in Deyr 2023 on the Juba. The forecast runs ahead of,
      and above, its own reanalysis on the rising limb. The operational lead is therefore
      better than the reanalysis backtest suggests on the Shabelle, and worse on the Juba in
      Gu, where the v4 issues are 9 to 17 days late.</p>
"""
t = replace_section(t, "model-timeline", timeline, "    <h2>Open items before a trigger report</h2>")

# ------------------------------------------------------------------ 4. observational fallback
lev = {s["window"]: s["levels"] for s in obs["summary"]}
sm = {s["window"]: s for s in obs["summary"]}
fallback = f"""    <h2>An observational fallback: bank full at the gauges</h2>
    <p>SWALIM publishes an official bank-full level for each gauge, the reading at which the
      river tops its banks (Juba: {lev['Deyr Juba']}; Shabelle: {lev['Deyr Shabelle']}), and
      a high-risk level below it. This section scores an observational trigger set at those
      levels, on its own and as a fallback that releases funds when no forecast window has
      activated. Two limits apply. The gauge record is capped at bank full, so a reading at
      bank full means at least bank full. Bardheere's official bank-full level (10.4 m) is
      above the highest reading in its record (8.0 m), so that gauge can never reach it.</p>
    <figure><img src="figs/l_obs_fallback.png{V}" alt="Years in which the gauges reached the high-risk and bank-full levels, against the benchmark and the forecast trigger">
      <figcaption>Per window and year: the benchmark (two gauges over 1-in-3, over 1-in-5),
        the adopted forecast trigger's activations, and the years in which two gauges read the
        high-risk level, one gauge read bank full, and two did.</figcaption></figure>
""" + obs_table + f"""
    <p>Where a gauge reached bank full in a benchmark flood, it did so between 9 days before
      and 9 days after the 1-in-5 crossing: Deyr Shabelle 2006, 2019 and 2023 at 7 days
      before, 1 day before and 3 days after; Gu Shabelle 2016, 2020 and 2023 at 9 days before
      and twice on the day; Deyr Juba 2023 at 9 days after. In every one of those seasons
      except Gu 2023 on the Shabelle the forecast trigger had already crossed, 2 to 19 days
      earlier. No Juba gauge has read bank full in Gu in the record. Bank full confirms a
      flood; it does not warn of one.</p>
    <p>Behind the adopted trigger the fallback recovers one benchmark season, Gu 2023 on the
      Shabelle (Belet Weyne at bank full on 23 May, the day the gauges crossed 1-in-5). It
      adds three seasons the two-gauge benchmark does not call floods: Deyr 2015 and Gu 2015
      on the Shabelle, and Gu 2021 at Belet Weyne. All three had flooding. Gu 2021 is the
      flood the benchmark misses because the gauge is censored at bank full; SWALIM reported
      flooding upstream of Belet Weyne and EM-DAT records 400,000 people affected. In both
      2015 seasons SWALIM issued high-risk watches or a Flood Alert, with floods reported at
      Jamame and Jilib. The fallback would therefore have released funds in four seasons the
      forecast trigger did not (Gu 2015, Deyr 2015, Gu 2021 and Gu 2023, all on the
      Shabelle), each time at or after the point where the river was already over its banks.
      In 2015 and 2021 no window activated at all, so the envelope's activation rate would
      rise from 8 to 10 years in 25, about 1-in-2.5.</p>
    <p>The high-risk level is too loose to release funds on its own. Two Juba gauges at the
      high-risk level occurs in seven Deyr seasons (2006, 2011, 2014, 2017, 2019, 2022,
      2023), four of them without a benchmark flood. Two Shabelle gauges at the high-risk
      level in Deyr matches the five severe years exactly; in Gu it adds 2005, 2010 and 2018.
      It is the level at which SWALIM's own bulletins escalate, and the SWALIM section above
      shows that step arriving 1 to 7 days before onset in most seasons. On this record the
      fallback would be: bank full at any monitored gauge releases the action funds when no
      forecast window has activated, and two gauges at the high-risk level counts as a
      readiness signal, not a release. SWALIM publishes the gauge readings daily with about
      a day's delay, so the fallback's lead is zero or negative. It covers floods the models
      miss; it does not add lead time.</p>
"""
t = replace_section(t, "obs-fallback", fallback, "    <h2>Open items before a trigger report</h2>")

PAGE.write_text(t, encoding="utf-8")
print("assembled; moved blocks:", bool(gauge_block), bool(peaks_block), bool(ownskill_block))
