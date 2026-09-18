"""(Re)build the FloodScan inundation subsection on the analysis page (index.html), by season with
both rivers together: removes any existing version (including the seven-day block inside it), then
inserts the current one before 'Calibrated on the reanalysis'. Reads floodscan_rank.json
(floodscan_season.py). Run patch_index_lead.py and then toc_inject.py afterwards."""
import json
import re
from pathlib import Path

S = Path(__file__).parent
page = S.parents[1] / "pages/trigger-single-model/index.html"
h = page.read_text(encoding="utf-8")
rank = json.load(open(S / "floodscan_rank.json"))
RP = rank["deyr"].get("rp", 5)

anchor = '<h3 id="calibrated-on-the-reanalysis-checked-on-the-forecasts">'
assert h.count(anchor) == 1
# the heading id is whatever toc_inject last wrote (it slugs the heading text), so match on
# either id; a non-greedy span from the first match to the anchor removes every earlier copy
h, n_removed = re.subn(r'\s*<h3 id="(?:when-does-inundation-follow-the-gauges|does-the-trigger-catch-the-inundation-floodscan)">.*?(?=<h3 id="calibrated-on-the-reanalysis-checked-on-the-forecasts">)', "\n", h, flags=re.S)
# the contents rail also carries the heading text, so count headings only
assert h.count("Does the trigger catch the inundation? (FloodScan)</h3>") == 0, "an older FloodScan block survived the removal"


def k(v):
    return f"{round(v / 1000):,}k"


rows = []
for season in ("deyr", "gu"):
    r = rank[season]
    top = ", ".join(f"{t['year']}{' (severe)' if t['bench'] == 'severe' else ' (flood)' if t['bench'] == 'flood' else ''} {k(t['people'])}" for t in r["top8"])
    bench = ", ".join(f"{b['year']}{'*' if b['severe'] else ''} rank {b['rank']} ({k(b['people'])})" for b in r["benchmark"])
    rows.append(f"<tr><td>{season.title()}</td><td>{top}</td><td>{bench}</td><td>{r['in_top8']} of {r['n_flood']}</td><td>{r['severe_in_top8']} of {r['n_severe']}</td></tr>")

d, g = rank["deyr"], rank["gu"]
block = f'''        <h3 id="does-the-trigger-catch-the-inundation-floodscan">Does the trigger catch the inundation? (FloodScan)</h3>
    <p>The benchmark dates a flood from the day the river's second SWALIM gauge crosses its own
      1-in-3 level. FloodScan gives an independent record of water on the ground. The series used
      here is flood exposure, people living in flooded cells (FloodScan SFED at about 10 km times
      WorldPop, the team's flood-exposure pipeline), summed each day over the 14
      anticipatory-action districts: Doolow, Luuq, Baardheere, Saakow, Bu'aale and Jilib on the
      Juba; Belet Weyne, Bulo Burto, Jalalaqsi, Jowhar, Balcad, Afgooye, Qoryooley and Marka on
      the Shabelle; 1998&ndash;2023, both rivers together, by season. Inundation is dated as the
      first day of the season on which that exposure reaches its own 1-in-{RP} level (Weibull on
      seasonal maxima), the same convention as the gauges.</p>
    <p><strong>Do the two-gauge flood years show high inundation?</strong> Partly. The largest floods
      do: Deyr 2023, the largest Deyr exposure season on record with {k(d['top8'][0]['people'])} people, Gu 2018,
      the largest Gu season, with {k(g['top8'][0]['people'])}, and Deyr 2014, Deyr 2017 and Gu 2023 all in the top six of 26 seasons. But
      several severe gauge seasons do not: Deyr 2006, 2019 and 2020 rank 11th, 14th and 17th, Gu
      2016 13th and Gu 2020 22nd. And several of the largest exposure seasons were not gauge
      floods: Deyr 2015 ({k(d['top8'][2]['people'])}), Deyr 2004, Gu 2002 ({k(g['top8'][1]['people'])}) and Gu 2006. In all,
      {d['in_top8']} of the {d['n_flood']} Deyr flood years and {g['in_top8']} of the {g['n_flood']} Gu flood years are among the eight largest
      exposure seasons. The gauge benchmark and the exposure record agree on the biggest floods
      and disagree on the moderate ones, and the disagreement is mostly on the Shabelle, where most
      of the exposed population lives in the lower districts below the last gauge.</p>
    <div class="tablewrap">
    <table class="data">
    <thead><tr><th>season</th><th>eight largest exposure seasons, 1998&ndash;2023 (people exposed)</th><th>two-gauge flood years, * severe: rank of 26</th><th>flood years in top 8</th><th>severe in top 8</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>
    </div>
    <p>Where both records agree a season flooded, three things put water on the ground ahead of,
      or apart from, the gauges, and each is a gap in the benchmark rather than in the satellite:</p>
    <ul>
      <li><strong>The Shabelle districts flood in a different order from the gauges.</strong> In Gu
        2023 Jowhar and Jalalaqsi crossed their own 1-in-{RP} exposure on 18 and 19 March, three
        weeks before the first gauge; in Gu 2018 Belet Weyne district did on 9 April, twelve days
        before its gauge reached 1-in-3; in Deyr 2023 Bulo Burto did on 7 October, four weeks
        before the Shabelle gauges.
        Water reaches people through breaks in the embankments and in the lower reach from
        Afgooye to Marka, which has no gauge, before the monitored gauges reach their levels.</li>
      <li><strong>The fitted 1-in-3 levels sit at or above SWALIM's high-risk levels</strong> at
        Bardheere, Belet Weyne and Jowhar. Moderate flooding can therefore be under way before a
        gauge reaches its statistical 1-in-3, which the benchmark counts as no flood yet.</li>
      <li><strong>Rain on the floodplain reads as water.</strong> In Gu 2010 Doolow and Baardheere
        districts crossed their exposure levels on 2 and 5 March, two months before the gauges,
        which then stood at 34 to 95 per cent of their levels: rainfall ponding at 10 km
        resolution, not the river.</li>
    </ul>
'''
h = h.replace(anchor, block + anchor)
closing = '''    <p>The comparison is thin: eleven benchmark seasons have a forecast archive and five of them an
      inundation day. The exposure record calls seasons the gauges do not, mostly on the Shabelle,
      and stays below its level in several gauge floods. At 10 km FloodScan blurs the river with
      rain on the floodplain, and exposure follows population, so the lower Shabelle weighs heavily
      in it. The flood-benchmark step kept the gauges as the benchmark for those reasons. What the
      comparison adds is a direction: where the benchmark errs on timing it errs late, on the
      Shabelle, and the reach it misses is the one below the last gauge.</p>
'''
h = h.replace(anchor, closing + anchor)
page.write_text(h, encoding="utf-8")
print("removed", n_removed, "old block(s); inserted")
