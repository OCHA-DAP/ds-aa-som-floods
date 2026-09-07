"""Assemble the new trigger-page sections from the analysis outputs and fold the
station-level material into one section. Idempotent: each inserted section sits
between HTML comment markers and is replaced on rerun; moved blocks are moved once.
Run toc_inject.py afterwards."""
import json
import re
from pathlib import Path

S = Path(__file__).parent
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
    <p>The page scores the trigger with three numbers: POD (probability of detection, the
      same quantity as recall), FAR (the false alarm ratio, the share of activations with
      no flood behind them) and F1. The tables below add the rest of the contingency
      table and the scores built on it, for each window's rule on each candidate model
      at the adopted return period and point count, over 1999 to 2023. All 25 years are
      counted, as on the rest of the page; the Juba gauges did not report in 1999 to 2001
      and the Shabelle's in 1999, so those years can only count as non-flood years.</p>
    <ul>
      <li><strong>POFD</strong>, the false alarm rate: false alarms divided by the non-flood
        years. Unlike FAR it is not inflated by a small number of activations.</li>
      <li><strong>CSI</strong> (critical success index, threat score): hits over hits plus
        misses plus false alarms. Correct negatives do not count, so a rare event scores low
        unless both misses and false alarms are few.</li>
      <li><strong>Bias</strong>: activations over flood years. Above 1 the rule activates
        more often than floods occur.</li>
      <li><strong>PSS</strong> (Peirce skill score): POD minus POFD. Zero for a rule that
        activates at random, one for a perfect one.</li>
      <li><strong>HSS</strong> (Heidke skill score): the share of correct calls beyond what
        chance would give, using all four cells.</li>
      <li><strong>AUC</strong>: the area under the ROC curve in the figure, which sweeps the
        station return period from 1-in-1.3 to 1-in-12 at the window's point count and plots
        POD against POFD at each setting. It scores the model's ranking of years regardless of
        where the threshold is put; 0.5 is chance, 1 is a model that ranks every flood year
        above every non-flood year.</li>
    </ul>
    <figure><img src="figs/l_roc.png{V}" alt="ROC curves per window and model, against severe and flood years">
      <figcaption>Top row against the severe years (two gauges over 1-in-5), bottom row
        against the flood years (two gauges over 1-in-3). Each curve is one model's window
        rule as its station return period is relaxed; the diamond is the adopted model at its
        adopted return period. With three to seven positive years per window the curves are
        coarse, so the AUC is a guide to ranking, not a precise number.</figcaption></figure>
""" + metrics_tables + f"""
    <p>Against the severe years the envelope has POD {E['POD']:.2f}, FAR {E['FAR']:.2f} and POFD
      {E['POFD']:.2f}: seven hits, no misses and one activation outside the benchmark (2013),
      for a CSI of {E['CSI']:.2f}, a bias of {E['bias']:.2f}, a Peirce score of {E['PSS']:.2f}
      and a Heidke score of {E['HSS']:.2f}. Per window the scores are weaker than the
      envelope's, because a window is judged on its own river-season while the envelope is
      credited for a year whichever window caught it. The AUC ranks the models the same way
      the page's selection did: GloFAS v5 has the highest area against severe years on the
      Shabelle in Deyr (0.99, against 0.93 for Google and 0.75 for v4), Google the highest on
      the Juba in Gu (0.95) and a tie with v5 on the Shabelle in Gu (0.89), and on the Juba in
      Deyr no model separates the years well (0.80 to 0.88, with v4 ahead of v5). Against the
      broader 1-in-3 flood years GloFAS v5 and v4 rank the Gu years better than Google (0.96
      to 0.99 against 0.92 to 0.95), which is where the single-station false alarms of the
      Google Gu windows show up.</p>
    <div class="callout warn">
      <strong>Benchmark note.</strong> These tables use the two-gauge benchmark as the repo's
      code now computes it, with gauge levels fitted on 2000 to 2023. On that record there
      are seven severe years (2006, 2014, 2016, 2018, 2019, 2020, 2023) and 2008 is a flood
      year but not a severe one. Earlier sections of this page, generated before that fit
      window was corrected, still count 2008 as the eighth severe year and the one miss. The
      activation years are unchanged; only the label on 2008 differs.
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
    <p>Everything the page knows about the individual gauges is gathered here. The table
      scores each candidate model at each of the seven points, per season: how well its
      daily series ranks the days like the gauge's own level record (Spearman rank
      correlation at the best lag between minus 10 and plus 30 days, positive when the model
      leads the gauge), and whether the model's own 1-in-3 and 1-in-5 crossing in a season
      coincides with the gauge's own crossing, counted by year over the years the gauge
      reported (at least 30 readings in the season, 2000 to 2023). Gauge levels are fitted
      on 2000 to 2023, model thresholds on 1999 to 2023, as elsewhere on the page. The
      adopted model for the window is in bold. Bardheere's and Bualle's records are short
      or suspect (see the tail-ratio figure above), so their rows carry less weight.</p>
""" + station_table + """
    <p>Two patterns carry over from the model choice. In Gu, Google ranks the days most like
      the Juba gauges (0.84 to 0.88 at Dollow, Luuq, Bardheere and Bualle) and is level with
      GloFAS v5 on the Shabelle; in Deyr, GloFAS v5 is the better tracker at every point
      (0.76 to 0.84 against 0.51 to 0.68 for Google, whose best lags on the Juba run
      negative, meaning it trails the gauge). GloFAS v4 in Deyr matches the Shabelle gauges
      only at lags of 11 to 15 days, which is why its Deyr crossings arrive late. Single-point
      detection is weak for every model: at 1-in-3 a point typically catches one to three of
      its gauge's events with one to five false alarms, and the models disagree on which
      years those are. That is the case for a consensus of points rather than any one of
      them, and for judging the trigger at window level rather than station level.</p>
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
    <p>For every flood season on the two-gauge benchmark (1999 to 2023), how far ahead of
      the gauges' 1-in-3 crossing each model would have raised a flag under the window's
      rule, whichever model the window adopted. Two kinds of date are shown. The diamonds
      are the first day the model's reanalysis crossed, a flow date. The triangles are the
      first forecast issue that crossed at leads 1 to 7, the date the flag would have been in
      hand: the GloFAS v4 reforecast is available from 2003 (twice weekly, so an issue can be
      up to three days later than a daily product would be), the Google reforecast from 2016
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
      2006, 2014, 2019 and 2023 while GloFAS v5 crosses 0 to 4 days before (and 3 days after
      in 2023); Google misses Deyr 2020 outright, which v5 catches 9 days late. On the Juba in
      Deyr neither model is early: both cross after onset in 2006 and 2023 and neither crosses
      in 2014. In Gu both models cross after onset in every Juba flood (1 to 8 days late in
      2016, 2018 and 2020) and both miss Gu 2023 on either river; on the Shabelle in Gu 2020
      Google is 5 days early and v5 2 days. The forecast issues are earlier than the reanalysis
      crossings almost everywhere they exist: the Google forecast crossed 10 to 13 days before
      onset in Gu 2016, 2018 and 2020 on the Shabelle and 12 days before in Deyr 2019, and the
      v4 forecast 12 to 13 days before in Deyr 2014 and 2019 on the Shabelle and 4 days before
      in Deyr 2023 on the Juba. That is the forecast running ahead of, and above, its own
      reanalysis on the rising limb, and it is why the operational lead is better than the
      reanalysis backtest suggests on the Shabelle and worse on the Juba in Gu, where the v4
      issues are 9 to 17 days late.</p>
"""
t = replace_section(t, "model-timeline", timeline, "    <h2>Open items before a trigger report</h2>")

# ------------------------------------------------------------------ 4. observational fallback
lev = {s["window"]: s["levels"] for s in obs["summary"]}
sm = {s["window"]: s for s in obs["summary"]}
fallback = f"""    <h2>An observational fallback: bank full at the gauges</h2>
    <p>If every forecast window misses a flood, the gauges themselves are the last line.
      SWALIM publishes an official bank-full level for each gauge, the reading at which the
      river tops its banks (Juba: {lev['Deyr Juba']}; Shabelle: {lev['Deyr Shabelle']}), with
      a high-risk level below it. This section asks what an observational trigger set at
      those levels would have done, alone and as a fallback behind the forecast trigger.
      Two caveats frame it. The gauge record is capped at bank full, so a reading at bank
      full means "at least bank full". And Bardheere's official bank-full level (10.4 m) is
      above the highest reading its record holds (8.0 m), so that gauge can never fire it.</p>
    <figure><img src="figs/l_obs_fallback.png{V}" alt="Years in which the gauges reached the high-risk and bank-full levels, against the benchmark and the forecast trigger">
      <figcaption>Per window and year: the benchmark (two gauges over 1-in-3, over 1-in-5),
        the adopted forecast trigger's activations, and the years in which two gauges read the
        high-risk level, one gauge read bank full, and two did.</figcaption></figure>
""" + obs_table + f"""
    <p><strong>Bank full is a confirmation, not a warning.</strong> Where a gauge reached bank
      full in a benchmark flood, it did so between 9 days before and 9 days after the 1-in-5
      crossing (Deyr Shabelle 2006, 2019 and 2023: 7 days before, 1 day before and 3 days after;
      Gu Shabelle 2016, 2020 and 2023: 9 days before and twice on the day; Deyr Juba 2023: 9
      days after), and in every one of those seasons the forecast trigger had already crossed,
      2 to 19 days earlier, except Gu 2023 on the Shabelle. On the Juba in Gu no gauge has ever
      read bank full in the record.</p>
    <p><strong>What a bank-full fallback adds.</strong> Behind the adopted trigger it recovers
      one benchmark season, Gu 2023 on the Shabelle (Belet Weyne at bank full on the day the
      gauges crossed 1-in-5, 23 May), and adds three seasons the two-gauge benchmark does not
      call floods: Deyr 2015 and Gu 2015 on the Shabelle, and Gu 2021 at Belet Weyne. None of
      the three is a false alarm in the ordinary sense. Gu 2021 is the flood the benchmark
      misses because the gauge is censored at bank full (SWALIM reported flooding upstream of
      Belet Weyne and EM-DAT records 400,000 people affected), and in both 2015 seasons SWALIM
      issued high-risk watches or a Flood Alert with floods reported at Jamame and Jilib. The
      fallback would therefore have released funds in four seasons the forecast trigger did
      not (Gu 2015, Deyr 2015, Gu 2021 and Gu 2023, all on the Shabelle), for floods that
      happened, at or after the moment the river was already over its banks. Two of those
      years, 2015 and 2021, are years in which no window activated at all, so the
      envelope's activation rate would rise from 8 to 10 years in 25, about 1-in-2.5.</p>
    <p><strong>The high-risk level is too loose as a standalone trigger.</strong> Two Juba
      gauges at the high-risk level occurs in seven Deyr seasons (2006, 2011, 2014, 2017,
      2019, 2022, 2023), four of them with no benchmark flood; two Shabelle gauges at the
      high-risk level in Deyr matches the five severe years exactly, and in Gu adds 2005,
      2010 and 2018 to the severe years. It is the level SWALIM's own bulletins escalate at,
      which the SWALIM section above shows arriving 1 to 7 days before onset in most seasons.
      A sensible fallback design is therefore: bank full at any monitored gauge releases the
      action funds if no forecast window has activated; two gauges at the high-risk level is a
      readiness-grade signal, not a release. The operational cost is the gauge reading itself,
      which SWALIM publishes daily with about a day's delay, so the fallback's effective lead
      is zero or negative: it pays for floods the models miss, not for early action.</p>
"""
t = replace_section(t, "obs-fallback", fallback, "    <h2>Open items before a trigger report</h2>")

PAGE.write_text(t, encoding="utf-8")
print("assembled; moved blocks:", bool(gauge_block), bool(peaks_block), bool(ownskill_block))
