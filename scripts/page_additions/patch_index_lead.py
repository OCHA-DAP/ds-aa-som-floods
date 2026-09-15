"""(Re)build the 'does the trigger catch the inundation seven days ahead?' block in the FloodScan
subsection of the analysis page, for the two-gauge flood seasons only. Idempotent: removes any
earlier version first. Reads floodscan_lead.json (floodscan_lead.py) and floodscan_timing.json."""
import json
import re
from pathlib import Path

S = Path(__file__).parent
page = S / "wt-trigger/pages/trigger-single-model/index.html"
h = page.read_text(encoding="utf-8")
lead = json.load(open(S / "floodscan_lead.json"))
RP = json.load(open(S / "floodscan_timing.json"))["gu_juba"].get("sfed_rp", 3)

anchor = "    <p>The comparison is thin. Only one to three seasons per window are called a flood by both"
assert h.count(anchor) == 1
h, n_removed = re.subn(r'\s*<p id="lead-to-inundation">.*?(?=    <p>The comparison is thin\. Only one to three seasons per window)', "\n", h, flags=re.S)

NAME = {("juba", "gu"): "Juba Gu", ("juba", "deyr"): "Juba Deyr", ("shabelle", "gu"): "Shabelle Gu", ("shabelle", "deyr"): "Shabelle Deyr"}
GREEN, AMBER, RED = 'rgba(14,138,123,.18)', 'rgba(244,169,59,.18)', 'rgba(179,64,54,.14)'


def shade(ld):
    return f' style="background:{GREEN if ld >= 7 else AMBER if ld > 0 else RED}"'


def lead_cells(o, key_issue, key_lead, key_gauge):
    if o[key_issue] is None:
        return "<td>never</td><td>&ndash;</td><td>&ndash;</td>"
    inun = f"<td{shade(o[key_lead])}>{o[key_lead]:+d} d</td>" if o[key_lead] is not None else "<td>&ndash;</td>"
    return f"<td>{o[key_issue]}</td>{inun}<td>{o[key_gauge]:+d} d</td>"


rows = "".join(
    f"<tr><td>{NAME[(o['river'], o['season'])]} {o['year']}</td><td>{o['second_gauge']}</td>"
    f"<td>{o['inundation'] if o['inundation'] else 'below 1-in-' + str(RP)}</td>"
    f"{lead_cells(o, 'same_issue', 'same_lead', 'same_lead_vs_gauge')}{lead_cells(o, 'either_issue', 'either_lead', 'either_lead_vs_gauge')}</tr>"
    for o in lead)
n = len(lead); with_in = [o for o in lead if o["inundation"] is not None]; n_in = len(with_in)
cnt = lambda key, f: sum(1 for o in with_in if o[key] is not None and f(o[key]))
never = lambda key: sum(1 for o in with_in if o[key] is None)
below = [f"{NAME[(o['river'], o['season'])]} {o['year']}" for o in lead if o["inundation"] is None]

block = f'''    <p id="lead-to-inundation"><strong>Does the trigger catch the inundation seven days ahead?</strong> The question is
      asked only for the seasons the two-gauge rule calls a flood, {n} of them with a forecast
      archive (Google Flood Hub 2016&ndash;2023 in Gu, GloFAS v4 2003&ndash;2023 in Deyr). In {n_in} of
      the {n} FloodScan reached its own 1-in-{RP} level, so an inundation day exists; in the other
      {n - n_in} the whole-river flooded fraction stayed below it, including the severe Shabelle Deyr
      seasons of 2014, 2019 and 2020, which says more about a 10-km product over a narrow river
      than about those floods. For the {n_in}: on the same river the action rule was met seven or
      more days before inundation in <strong>{cnt("same_lead", lambda x: x >= 7)} of {n_in}</strong>, one to six days before
      in {cnt("same_lead", lambda x: 0 < x < 7)}, on or after the day in {cnt("same_lead", lambda x: x <= 0)}, and never in
      {never("same_lead")}. Counting an activation on either river, {cnt("either_lead", lambda x: x >= 7)} of {n_in} had seven
      or more days. The seven-day cases are Deyr on the Juba, 2014 and 2023, on GloFAS v4. In
      Gu, Google met the rule one day before inundation on the Juba in 2018 and eight days after
      it on the Shabelle in 2018 and 2023. The last column of each pair gives the same lead
      measured against the second gauge, the page's usual reference, for comparison.</p>
    <div class="tablewrap">
    <table class="data">
    <thead><tr><th>season</th><th>2nd gauge over 1-in-3</th><th>inundation (FloodScan 1-in-{RP})</th><th>rule met, same river</th><th>lead to inundation</th><th>lead to 2nd gauge</th><th>rule met, either river</th><th>lead to inundation</th><th>lead to 2nd gauge</th></tr></thead>
    <tbody>{rows}</tbody>
    </table>
    </div>
    <p>So, where the ground is seen to flood, the trigger gave a week's warning on the Juba in
      Deyr and did not on the Shabelle or in Gu: on the Shabelle the floodplain was under water
      before the forecasts saw the river reach its levels, and on the Juba in Gu the two came
      within a day. The last column of each pair shows the same forecasts against the second gauge:
      where the second gauge crossed weeks after the first, as on the Juba in Deyr, the
      inundation reference gives the forecasts more credit than the gauge does, and on the
      Shabelle in Gu less.</p>
'''
h = h.replace(anchor, block + anchor)
page.write_text(h, encoding="utf-8")
print("removed", n_removed, "old block(s); inserted;", n, "rows,", n_in, "with inundation; below:", below)
