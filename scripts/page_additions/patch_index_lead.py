"""(Re)build the 'does the trigger catch the inundation ahead of time?' block in the FloodScan
subsection of the analysis page, by season with both rivers together: the two-SWALIM-gauge flood
seasons only, three records against inundation (SWALIM gauges, reanalysis, forecasts), each the
earliest on either river. Reads floodscan_lead.json (floodscan_season.py). Idempotent."""
import json
import re
from pathlib import Path

S = Path(__file__).parent
page = S / "wt-trigger/pages/trigger-single-model/index.html"
h = page.read_text(encoding="utf-8")
lead = json.load(open(S / "floodscan_lead.json"))
RP = 5

anchor = "    <p>The comparison is thin"
assert h.count(anchor) == 1
h, n_removed = re.subn(r'\s*<p id="lead-to-inundation">.*?(?=    <p>The comparison is thin)', "\n", h, flags=re.S)

GREEN, AMBER, RED = "rgba(14,138,123,.18)", "rgba(244,169,59,.18)", "rgba(179,64,54,.14)"
RECORDS = (("gauge", "SWALIM gauges: second gauge over 1-in-3"), ("reanalysis", "Reanalysis: calibration record meets the action rule"), ("forecast", "Forecasts: first issue meeting the action rule"))
with_in = [o for o in lead if o["inundation"]]
n, n_in = len(lead), len(with_in)
label = lambda o: f"{o['season'].title()} {o['year']}"


def shade(ld):
    return f' style="background:{GREEN if ld >= 8 else AMBER if ld >= 0 else RED}"'


def cells(o, key):
    if o[key + "_date"] is None:
        return "<td>never</td><td>&ndash;</td>"
    ld = o[key + "_lead"]
    return f"<td>{o[key + '_date']}</td>" + (f"<td{shade(ld)}>{ld:+d} d</td>" if ld is not None else "<td>&ndash;</td>")


def count(key, f):
    return sum(1 for o in with_in if o[key + "_lead"] is not None and f(o[key + "_lead"]))


def never(key):
    return sum(1 for o in with_in if o[key + "_lead"] is None)


def yrs(key, f):
    return ", ".join(f"{label(o)} ({o[key + '_lead']:+d} d)" for o in with_in if o[key + "_lead"] is not None and f(o[key + "_lead"]))


summary = "".join(f"<tr><td>{lab}</td><td>{count(k, lambda x: x >= 8)}</td><td>{count(k, lambda x: 0 <= x < 8)}</td><td>{count(k, lambda x: x < 0)}</td><td>{never(k)}</td></tr>" for k, lab in RECORDS)
detail = "".join(f"<tr><td>{label(o)}</td><td>{o['inundation'] or 'below 1-in-' + str(RP)}</td>{cells(o, 'gauge')}{cells(o, 'reanalysis')}{cells(o, 'forecast')}</tr>" for o in lead)
below = ", ".join(label(o) for o in lead if not o["inundation"])

block = f'''    <p id="lead-to-inundation"><strong>Does the trigger catch the inundation ahead of time?</strong> The question is
      asked only for the seasons the two-SWALIM-gauge rule calls a flood on either river, {n} of them
      with a forecast archive (Google Flood Hub 2016&ndash;2023 in Gu, GloFAS v4 2003&ndash;2023 in
      Deyr). In {n_in} of the {n}, exposure across the 14 districts reached its own 1-in-{RP} level, so an
      inundation day exists; in the other {n - n_in} it stayed below ({below}). The inundation day is the
      crossing, not the peak: the peak exposure came 0 to 19 days after it (the same day in Deyr
      2014, 19 days later in Deyr 2023). Three dated records are set against the
      inundation day, each the earliest on either river, as the mechanism counts: the SWALIM
      gauges (the day a river's second gauge went over its own 1-in-3 level, the page's
      benchmark), the reanalysis (the first day a window's calibration record, Google's
      retrospective run in Gu and GloFAS version 5 in Deyr, met the action rule) and the
      forecasts (the first issue on which a window's source, Google in Gu and GloFAS v4 in Deyr,
      met the action rule at leads 1 to 7).</p>
    <div class="tablewrap">
    <table class="data">
    <thead><tr><th>record, either river</th><th>8 days or more before exposure crossed its 1-in-{RP}</th><th>on the day or up to 7 days before</th><th>after the crossing</th><th>never</th></tr></thead>
    <tbody>{summary}</tbody>
    </table>
    </div>
    <figure><img src="figs/floodscan_timeline.png" alt="Per flood season: SWALIM second gauge, reanalysis rule and first forecast issue, in days before or after FloodScan inundation"><figcaption>The {n_in} seasons with an inundation day. Day 0 is the day FloodScan flood exposure across the 14 anticipatory-action districts went over its own 1-in-{RP} level; each mark is a record's earliest date on either river, before the inundation to the left and after it to the right. A cross at the left edge means the record never met its rule that season.</figcaption></figure>
    <p>The forecasts came eight days or more before the inundation in {count("forecast", lambda x: x >= 8)} of the {n_in} seasons
      ({yrs("forecast", lambda x: x >= 8)}), all Deyr on GloFAS v4; on the day or up to seven days before in
      {count("forecast", lambda x: 0 <= x < 8)} ({yrs("forecast", lambda x: 0 <= x < 8)}); and after it in {count("forecast", lambda x: x < 0)} ({yrs("forecast", lambda x: x < 0)}), the season
      every source missed. The SWALIM gauges, the benchmark the rest of this page is scored
      against, sit 3 to 13 days before the inundation and three days after it in Deyr 2017, a
      season the gauges called on the Juba while the exposure came on the Shabelle. The
      reanalysis, the record the windows were calibrated on, never met the rule in Deyr 2017 or
      Gu 2023.</p>
    <div class="tablewrap">
    <table class="data">
    <thead><tr><th>season</th><th>inundation (exposure over 1-in-{RP})</th><th>SWALIM gauges</th><th>lead</th><th>reanalysis rule met</th><th>lead</th><th>forecast issue</th><th>lead</th></tr></thead>
    <tbody>{detail}</tbody>
    </table>
    </div>
'''
h = h.replace(anchor, block + anchor)
page.write_text(h, encoding="utf-8")
print("removed", n_removed, "old block(s); inserted;", n, "rows,", n_in, "with inundation")
