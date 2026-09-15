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
RP = json.load(open(S / "floodscan_rank.json"))["deyr"].get("rp", 5)

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
        return '<td style="white-space:nowrap">never</td><td style="text-align:center">&ndash;</td>'
    ld = o[key + "_lead"]
    lead_td = f'<td style="text-align:center;{"background:" + (GREEN if ld >= 8 else AMBER if ld >= 0 else RED) if ld is not None else ""}">{f"{ld:+d} d" if ld is not None else "&ndash;"}</td>'
    return f'<td style="white-space:nowrap">{o[key + "_date"]}</td>' + lead_td


def count(key, f):
    return sum(1 for o in with_in if o[key + "_lead"] is not None and f(o[key + "_lead"]))


def never(key):
    return sum(1 for o in with_in if o[key + "_lead"] is None)


def yrs(key, f):
    return ", ".join(f"{label(o)} ({o[key + '_lead']:+d} d)" for o in with_in if o[key + "_lead"] is not None and f(o[key + "_lead"]))


C = ' style="text-align:center"'
summary = "".join(f"<tr><td>{lab}</td><td{C}>{count(k, lambda x: x >= 8)}</td><td{C}>{count(k, lambda x: 0 <= x < 8)}</td><td{C}>{count(k, lambda x: x < 0)}</td><td{C}>{never(k)}</td></tr>" for k, lab in RECORDS)
detail = "".join(f"<tr><td style='white-space:nowrap'>{label(o)}</td><td style='white-space:nowrap'>{o['inundation'] or 'below 1-in-' + str(RP)}</td>{cells(o, 'gauge')}{cells(o, 'reanalysis')}{cells(o, 'forecast')}</tr>" for o in lead)
below = ", ".join(label(o) for o in lead if not o["inundation"])

block = f'''    <p id="lead-to-inundation"><strong>Does the trigger catch the inundation ahead of time?</strong> The question is
      asked only for the seasons the two-SWALIM-gauge rule calls a flood on either river, {n} of them
      with a forecast archive (Google Flood Hub 2016&ndash;2023 in Gu, GloFAS v4 2003&ndash;2023 in
      Deyr). In {n_in} of the {n}, exposure across the 14 districts reached its own 1-in-{RP} level, so an
      inundation day exists; in the other {n - n_in} it stayed below ({below}). The inundation day is the
      crossing, not the peak: the peak exposure came 0 to 20 days after it (the same day in Deyr
      2014, 20 days later in Deyr 2023). Three dated records are set against the
      inundation day, each the earliest on either river, as the mechanism counts: the SWALIM
      gauges (the day a river's second gauge went over its own 1-in-3 level, the page's
      benchmark), the reanalysis (the first day a window's calibration record, Google's
      retrospective run in Gu and GloFAS version 5 in Deyr, met the action rule) and the
      forecasts (the first issue on which a window's source, Google in Gu and GloFAS v4 in Deyr,
      met the action rule at leads 1 to 7).</p>
    <div class="tablewrap">
    <table class="data">
    <colgroup><col style="width:40%"><col style="width:18%"><col style="width:18%"><col style="width:13%"><col style="width:11%"></colgroup>
    <thead><tr><th>record, either river</th><th{C}>8 days or more before the 1-in-{RP} crossing</th><th{C}>on the day or up to 7 days before</th><th{C}>after the crossing</th><th{C}>never</th></tr></thead>
    <tbody>{summary}</tbody>
    </table>
    </div>
    <figure><img src="figs/floodscan_timeline.png" alt="Per flood season: SWALIM second gauge, reanalysis rule and first forecast issue, in days before or after FloodScan inundation"><figcaption>The {n_in} seasons with an inundation day. Day 0 is the day FloodScan flood exposure across the 14 anticipatory-action districts went over its own 1-in-{RP} level; each mark is a record's earliest date on either river, before the inundation to the left and after it to the right. A cross at the left edge means the record never met its rule that season.</figcaption></figure>
    <p>The forecasts came eight days or more before the inundation in {count("forecast", lambda x: x >= 8)} of the {n_in} seasons
      ({yrs("forecast", lambda x: x >= 8)}), all Deyr on GloFAS v4; on the day or up to seven days before in
      {count("forecast", lambda x: 0 <= x < 8)} ({yrs("forecast", lambda x: 0 <= x < 8)}); and after it in {count("forecast", lambda x: x < 0)} ({yrs("forecast", lambda x: x < 0)}), the season
      every source missed. The SWALIM gauges, the benchmark the rest of this page is scored
      against, sit 2 to 13 days before the inundation and three days after it in Deyr 2017, a
      season the gauges called on the Juba while the exposure came on the Shabelle. The
      reanalysis, the record the windows were calibrated on, never met the rule in Deyr 2017 or
      Gu 2023 and met it a day after the crossing in Gu 2018.</p>
    <div class="tablewrap">
    <table class="data">
    <thead><tr><th rowspan="2" style="vertical-align:bottom">season</th><th rowspan="2" style="vertical-align:bottom">exposure crossed 1-in-{RP}</th><th colspan="2"{C}>SWALIM gauges, 2nd over 1-in-3</th><th colspan="2"{C}>reanalysis rule met</th><th colspan="2"{C}>first forecast issue</th></tr>
    <tr><th>date</th><th{C}>lead</th><th>date</th><th{C}>lead</th><th>date</th><th{C}>lead</th></tr></thead>
    <tbody>{detail}</tbody>
    </table>
    </div>
'''
h = h.replace(anchor, block + anchor)
page.write_text(h, encoding="utf-8")
print("removed", n_removed, "old block(s); inserted;", n, "rows,", n_in, "with inundation")
