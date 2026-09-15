"""(Re)build the FloodScan inundation-timing subsection on the analysis page (index.html):
removes any existing version, then inserts the current one before 'Calibrated on the
reanalysis'. Reads floodscan_timing.json (written by floodscan_timing.py)."""
import json
import re
from pathlib import Path

S = Path(__file__).parent
page = S / "wt-trigger/pages/trigger-single-model/index.html"
h = page.read_text(encoding="utf-8")
res = json.load(open(S / "floodscan_timing.json"))
RP = res["gu_juba"].get("sfed_rp", 3)

anchor = '<h3 id="calibrated-on-the-reanalysis-checked-on-the-forecasts">'
assert h.count(anchor) == 1
h, n_removed = re.subn(r'\s*<h3 id="when-does-inundation-follow-the-gauges">.*?(?=<h3 id="calibrated-on-the-reanalysis-checked-on-the-forecasts">)', "\n", h, flags=re.S)

NAME = {"deyr_juba": "Juba Deyr", "deyr_shabelle": "Shabelle Deyr", "gu_juba": "Juba Gu", "gu_shabelle": "Shabelle Gu"}
ORDER = ["gu_juba", "deyr_juba", "gu_shabelle", "deyr_shabelle"]


def med(v):
    v = sorted(v)
    return (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2) if v else None


def cell(vals):
    if not vals:
        return "&ndash;"
    m = med(vals)
    m = f"{m:+.0f}" if m != 0 else "0"
    return f"{m} d ({min(vals):+d} to {max(vals):+d})"


rows = []
for k in ORDER:
    r = res[k]
    v2 = list(r["vs_second"].values()); v1 = list(r["vs_first"].values())
    yrs = ", ".join(str(y) for y in r["years_both"]) or "&ndash;"
    so = ", ".join(str(y) for y in r["sfed_only"]) or "&ndash;"
    go = ", ".join(str(y) for y in r["gauge_only"]) or "&ndash;"
    rows.append(f"<tr><td>{NAME[k]}</td><td>{len(r['years_both'])}</td><td>{cell(v2)}</td><td>{cell(v1)}</td><td>{yrs}</td><td>{so}</td><td>{go}</td></tr>")

block = f'''        <h3 id="when-does-inundation-follow-the-gauges">When does inundation follow the gauges? (FloodScan)</h3>
    <p>The benchmark dates a flood from the day the river's second gauge crosses its own 1-in-3
      level. FloodScan gives an independent date for water on the ground: the daily flooded
      fraction (SFED, about 10 km resolution) averaged over a 10-km buffer of the river's main
      stem, 1998&ndash;2023, the same series the flood-benchmark step validated against the
      gauges. Inundation is dated as the first day of the season on which the buffer's flooded
      fraction reaches its own 1-in-{RP} level (Weibull on seasonal maxima); 1-in-{RP} rather than
      1-in-3 so that ordinary seasonal ponding does not count. The table compares that day with
      the gauge days in the seasons both records call a flood; a positive number means
      inundation showed after the gauge day.</p>
    <div class="tablewrap">
    <table class="data">
    <thead><tr><th>window</th><th>seasons both call a flood</th><th>inundation vs 2nd gauge, median (range)</th><th>vs 1st gauge, median (range)</th><th>seasons</th><th>FloodScan only</th><th>gauges only</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>
    </div>
    <figure><img src="figs/floodscan_timeline.png" alt="Timeline per flood season: first gauge, second gauge and FloodScan inundation"><figcaption>One row per season both records call a flood. Day 0 is the first SWALIM gauge on the river over its own 1-in-3 level; the black dot is the second gauge, the day the benchmark dates the flood; the teal square is the day the FloodScan flooded fraction along the river went over its own 1-in-{RP} level. Points beyond 35 days are drawn at the edge with their value.</figcaption></figure>
    <p><strong>On the Juba the second-gauge date holds.</strong> Inundation follows the second
      gauge by 1 to 13 days in Deyr and lands within three days of it in Gu, where the Juba's
      gauges cross within a day of each other. <strong>On the Shabelle it can come first.</strong>
      In Gu 2018 the floodplain was under water 18 days before the second gauge crossed and two
      days before the first; in Deyr 2023, six days before the second. Three things explain
      inundation ahead of the gauges, and each is a gap in the benchmark rather than in the
      satellite:</p>
    <ul>
      <li><strong>The lower Shabelle has no gauge.</strong> Downstream of Jowhar, from Afgooye
        to Marka, the river overtops at flows that leave the three monitored gauges below their
        levels. In both Shabelle cases the buffer's lower segment flooded first, while Belet
        Weyne, Bulo Burti and Jowhar stood at 71 to 96 per cent of their own 1-in-3 levels. The
        benchmark, and the trigger, date the flood by gauges upstream of where it starts.</li>
      <li><strong>The fitted 1-in-3 levels sit at or above SWALIM's high-risk levels</strong> at
        Bardheere, Belet Weyne and Jowhar, and the river breaks through weak embankments below
        them. Moderate flooding can therefore be under way before a gauge reaches its
        statistical 1-in-3, which the benchmark counts as no flood yet.</li>
      <li><strong>Rain on the floodplain reads as water.</strong> Juba Gu 2010 is dated 61 days
        before the first gauge because the buffer flooded in early March, with the gauges at
        34 to 95 per cent of their levels: rainfall ponding at 10 km resolution, not the river.
        Raising the FloodScan level to 1-in-{RP} removes most such seasons but not this one.</li>
    </ul>
    <p>The comparison is thin. Only one to three seasons per window are called a flood by both
      records, FloodScan calls seasons the gauges do not, and misses several the gauges call
      (Gu Shabelle 2003, 2005, 2010, 2016 and 2020). The flood-benchmark step kept the gauges
      as the benchmark for that reason. What the comparison adds is a direction: where the
      benchmark errs on timing it errs late, on the Shabelle, and the reach it misses is the
      one below the last gauge.</p>
'''
h = h.replace(anchor, block + anchor)
page.write_text(h, encoding="utf-8")
print("removed", n_removed, "old block(s); inserted; rows:", len(rows))
