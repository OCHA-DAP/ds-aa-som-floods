"""(Re)build the 'does the trigger catch the inundation ahead of time?' block in the FloodScan
subsection of the analysis page: the two-SWALIM-gauge flood seasons only, three records against
inundation (SWALIM gauges, reanalysis, forecasts), each on either river. Inundation is FloodScan
flood exposure in the river's anticipatory-action districts over its own 1-in-5. Idempotent."""
import json
import re
from pathlib import Path

S = Path(__file__).parent
page = S / "wt-trigger/pages/trigger-single-model/index.html"
h = page.read_text(encoding="utf-8")
lead = json.load(open(S / "floodscan_lead.json"))
RP = json.load(open(S / "floodscan_timing.json"))["gu_juba"].get("sfed_rp", 3)

anchor = "    <p>The comparison is thin."
assert h.count(anchor) == 1
h, n_removed = re.subn(r'\s*<p id="lead-to-inundation">.*?(?=    <p>The comparison is thin\.)', "\n", h, flags=re.S)

NAME = {("juba", "gu"): "Juba Gu", ("juba", "deyr"): "Juba Deyr", ("shabelle", "gu"): "Shabelle Gu", ("shabelle", "deyr"): "Shabelle Deyr"}
GREEN, AMBER, RED = "rgba(14,138,123,.18)", "rgba(244,169,59,.18)", "rgba(179,64,54,.14)"
RECORDS = (("gauge", "SWALIM gauges: second gauge over 1-in-3"), ("reanalysis", "Reanalysis: calibration record meets the action rule"), ("forecast", "Forecasts: first issue meeting the action rule"))
with_in = [o for o in lead if o["inundation"]]
n, n_in = len(lead), len(with_in)


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
    return ", ".join(f"{NAME[(o['river'], o['season'])]} {o['year']}" for o in with_in if o[key + "_lead"] is not None and f(o[key + "_lead"]))


summary = "".join(f"<tr><td>{lab}</td><td>{count(k, lambda x: x >= 8)}</td><td>{count(k, lambda x: 0 <= x < 8)}</td><td>{count(k, lambda x: x < 0)}</td><td>{never(k)}</td></tr>" for k, lab in RECORDS)
detail = "".join(f"<tr><td>{NAME[(o['river'], o['season'])]} {o['year']}</td><td>{o['inundation'] or 'below 1-in-' + str(RP)}</td>{cells(o, 'gauge')}{cells(o, 'reanalysis')}{cells(o, 'forecast')}</tr>" for o in lead)
below = ", ".join(f"{NAME[(o['river'], o['season'])]} {o['year']}" for o in lead if not o["inundation"])

block = f'''    <p id="lead-to-inundation"><strong>Does the trigger catch the inundation ahead of time?</strong> The question is
      asked only for the seasons the two-SWALIM-gauge rule calls a flood, {n} of them with a forecast
      archive (Google Flood Hub 2016&ndash;2023 in Gu, GloFAS v4 2003&ndash;2023 in Deyr). In {n_in} of
      the {n}, flood exposure in the river's districts reached its own 1-in-{RP} level, so an
      inundation day exists; in the other {n - n_in} it stayed below ({below}). Three dated records are set against the
      inundation day, each taken on either river as the mechanism does: the SWALIM gauges (the
      day the river's second gauge went over its own 1-in-3 level, the page's benchmark), the
      reanalysis (the first day the window's calibration record, Google's retrospective run in
      Gu and GloFAS version 5 in Deyr, met the action rule) and the forecasts (the first issue
      on which the window's source, Google in Gu and GloFAS v4 in Deyr, met the action rule at
      leads 1 to 7).</p>
    <div class="tablewrap">
    <table class="data">
    <thead><tr><th>record, either river</th><th>8 days or more before inundation</th><th>on the day or up to 7 days before</th><th>after</th><th>never</th></tr></thead>
    <tbody>{summary}</tbody>
    </table>
    </div>
    <figure><img src="figs/floodscan_timeline.png" alt="Per flood season: SWALIM second gauge, reanalysis rule and first forecast issue, in days before or after FloodScan inundation"><figcaption>The {n_in} seasons with an inundation day. Day 0 is the day FloodScan flood exposure in the river's anticipatory-action districts went over its own 1-in-{RP} level; each mark is a record's date on either river, before the inundation to the left and after it to the right. Marks beyond 30 days are drawn at the edge with their value; a cross at the left edge means the record never met its rule that season.</figcaption></figure>
    <p>The forecasts gave eight days or more in {count("forecast", lambda x: x >= 8)} of the {n_in} seasons ({yrs("forecast", lambda x: x >= 8)}), all
      Deyr on GloFAS v4; on the day or up to seven days before in {count("forecast", lambda x: 0 <= x < 8)} ({yrs("forecast", lambda x: 0 <= x < 8)}),
      all Gu on Google; after the inundation in {count("forecast", lambda x: x < 0)} ({yrs("forecast", lambda x: x < 0)}), the Gu 2023 season
      every source missed; and never in {never("forecast")} ({yrs("forecast", lambda x: False) or "Juba Deyr 2006"}). The SWALIM gauges, the
      benchmark the rest of this page is scored against, sit within a week of the inundation in
      Gu and one to two weeks before it in Deyr. The reanalysis, the record the windows were
      calibrated on, is the weakest of the three: never in Gu 2023 on either river, after the
      inundation in Gu 2016 and 2018 on the Juba.</p>
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
