"""(Re)build the FloodScan inundation subsection on the analysis page (index.html): removes any
existing version (including the seven-day block inside it), then inserts the current one before
'Calibrated on the reanalysis'. Reads floodscan_timing.json (floodscan_timing.py, on the
district-exposure series from floodscan_districts.py). Run patch_index_lead.py afterwards."""
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
    <p>The benchmark dates a flood from the day the river's second SWALIM gauge crosses its own
      1-in-3 level. FloodScan gives an independent date for water on the ground. The series used
      here is flood exposure, people living in flooded cells (FloodScan SFED at about 10 km
      times WorldPop, the team's flood-exposure pipeline), summed each day over the river's
      anticipatory-action districts: Doolow, Luuq, Baardheere, Saakow, Bu'aale and Jilib on the
      Juba; Belet Weyne, Bulo Burto, Jalalaqsi, Jowhar, Balcad, Afgooye, Qoryooley and Marka on
      the Shabelle; 1998&ndash;2023. Inundation is dated as the first day of the season on which
      that exposure reaches its own 1-in-{RP} level (Weibull on seasonal maxima); 1-in-{RP} rather
      than 1-in-3 so that ordinary seasonal ponding does not count.</p>
    <p><strong>Do the benchmark years show high inundation?</strong> On the Juba, yes. Four of the five
      Deyr flood seasons the gauges call are among the eight largest exposure seasons of 26,
      including all three severe ones, with 2023 the largest on record; in Gu, three of six, with
      2018 and 2023 in the top three. On the Shabelle, only partly. Two of the six Deyr flood
      seasons are in the top eight, and the severe 2019 and 2020 seasons rank 15th and 17th; in
      Gu, three of seven, with the severe 2020 season 22nd. The largest Shabelle exposure seasons
      are ones the gauges did not call a flood: Deyr 2017 (179,000 people), Deyr 2015 (169,000),
      Gu 2002 and Gu 2006. So the gauge benchmark and the exposure record agree on the Juba and
      only loosely on the Shabelle, where most of the exposed population lives in the lower
      districts below the last gauge.</p>
    <p>The table compares the inundation day with the SWALIM gauge days in the seasons both records
      call a flood; a positive number means inundation showed after the gauge day.</p>
    <div class="tablewrap">
    <table class="data">
    <thead><tr><th>window</th><th>seasons both call a flood</th><th>inundation vs 2nd SWALIM gauge, median (range)</th><th>vs 1st SWALIM gauge, median (range)</th><th>seasons</th><th>FloodScan only</th><th>SWALIM gauges only</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>
    </div>
    <p><strong>On the Juba the second-gauge date holds.</strong> Inundation lands within two days of
      the second gauge in Gu and 1 to 13 days after it in Deyr. <strong>On the Shabelle in Gu it
      comes first:</strong> 10 and 17 days before the second gauge in 2023 and 2018, a day before
      the first. Three things put inundation ahead of the gauges, and each is a gap in the
      benchmark rather than in the satellite:</p>
    <ul>
      <li><strong>The Shabelle districts flood in a different order from the gauges.</strong> In Gu
        2023 Afgooye and Jowhar crossed their own 1-in-{RP} exposure on 7 April, the day before the
        first gauge; in Gu 2018 Belet Weyne district did on 9 April, twelve days before its gauge
        reached 1-in-3; in Deyr 2023 Bulo Burto did on 7 October, four weeks before the gauges.
        Water reaches people through breaks in the embankments and in the lower reach from
        Afgooye to Marka, which has no gauge, before the monitored gauges reach their levels.</li>
      <li><strong>The fitted 1-in-3 levels sit at or above SWALIM's high-risk levels</strong> at
        Bardheere, Belet Weyne and Jowhar. Moderate flooding can therefore be under way before a
        gauge reaches its statistical 1-in-3, which the benchmark counts as no flood yet.</li>
      <li><strong>Rain on the floodplain reads as water.</strong> In Gu 2010 Doolow and Luuq districts
        crossed their exposure levels on 5 and 21 March, with the gauges at 34 to 95 per cent of
        their levels: rainfall ponding at 10 km resolution, not the river.</li>
    </ul>
'''
h = h.replace(anchor, block + anchor)
# closing paragraph of the subsection (the seven-day block is inserted before it by patch_index_lead.py)
closing = '''    <p>The comparison is thin. Only two or three seasons per window are called a flood by both
      records. The exposure record calls seasons the gauges do not, mostly on the Shabelle, and
      stays below its level in gauge floods there (Deyr 2006, 2008, 2019 and 2020; Gu 2003, 2005,
      2010, 2016 and 2020). At 10 km FloodScan blurs the river with rain on the floodplain, and
      exposure follows population, so the lower Shabelle weighs heavily in it. The
      flood-benchmark step kept the gauges as the benchmark for those reasons. What the
      comparison adds is a direction: where the benchmark errs on timing it errs late, on the
      Shabelle, and the reach it misses is the one below the last gauge.</p>
'''
h = h.replace(anchor, closing + anchor)
page.write_text(h, encoding="utf-8")
print("removed", n_removed, "old block(s); inserted; rows:", len(rows))
