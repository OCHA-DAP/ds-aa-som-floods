"""Does the trigger catch the inundation ahead of time? For every season the two-SWALIM-gauge rule
calls a flood (the benchmark), the FloodScan inundation day (river-buffer flooded fraction over its
own 1-in-5, where reached) against three dated records, each taken on either river:
  * SWALIM gauges: the day the river's second gauge went over its own 1-in-3 level;
  * reanalysis: the first day the window's calibration record (Google retrospective in Gu,
    GloFAS v5 reanalysis in Deyr) met the action rule, n points over their thresholds on one day;
  * forecasts: the first forecast issue on which the window's source (Google in Gu, GloFAS v4 in
    Deyr) met the action rule at leads 1 to 7.
Writes floodscan_lead.json."""
import json
import pandas as pd
import somlib as L
from src.utils import weibull_level

fs = pd.read_parquet("inundation_series.parquet"); fs["date"] = pd.to_datetime(fs["date"])   # people exposed per river's AA districts (floodscan_districts.py)
WINDOWS = [("juba", "gu"), ("juba", "deyr"), ("shabelle", "gu"), ("shabelle", "deyr")]
RULE = {("juba", "deyr"): (4, 3), ("shabelle", "deyr"): (4, 2), ("juba", "gu"): (5, 3), ("shabelle", "gu"): (6, 2)}
CAL = {"deyr": "glofas_v5", "gu": "google_grrr"}
FC = {"deyr": "glofas_v4", "gu": "google_grrr"}
ARCHIVE = {"google_grrr": (2016, 2023), "glofas_v4": (2003, 2023)}
SFED_RP = 5
SPAN = list(range(1999, 2024))
RIVERS = ("juba", "shabelle")


def sfed_onsets(river, season, rp=SFED_RP):
    s = fs[fs.river == river].set_index("date")["value"].sort_index()
    s = s[s.index.month.isin(L.SEASONS[season])]
    lev = weibull_level(s.groupby(s.index.year).max().dropna().values, rp)
    return {y: g[g >= lev].index.min().normalize() for y, g in s.groupby(s.index.year) if (g >= lev).any() and y in SPAN}


def record_rule_dates(model, river, season, rp, n_req):
    m = L.daily_matrix(model, river, season)
    thr = L.model_thresholds(model, river, season, rp)
    cols = [c for c in thr.index if c in m.columns and not pd.isna(thr[c])]
    votes = (m[cols] >= thr[cols]).sum(axis=1)
    hit = votes[votes >= n_req]
    return {y: g.index.min().normalize() for y, g in hit.groupby(hit.index.year) if y in SPAN}


gauge2, rean, fcst = {}, {}, {}
for river, season in WINDOWS:
    rp, n = RULE[(river, season)]
    gauge2[(river, season)] = L.gauge_crossings(river, season, 3, span=SPAN)
    rean[(river, season)] = record_rule_dates(CAL[season], river, season, rp, n)
    fcst[(river, season)] = {y: v[0] for y, v in L.first_issue_dates(FC[season], river, season, rp, n, span=SPAN, leads=(1, 7)).items()}


def either(d, season, y):
    vals = [d[(r, season)][y] for r in RIVERS if y in d[(r, season)]]
    return min(vals) if vals else None


out = []
print(f"{'season':18}{'inundation':>13}{'SWALIM 2nd gauge':>17}{'lead':>6}{'reanalysis rule':>17}{'lead':>6}{'forecast issue':>16}{'lead':>6}")
for river, season in WINDOWS:
    lo, hi = ARCHIVE[FC[season]]
    onsets = sfed_onsets(river, season)
    for y in sorted(gauge2[(river, season)]):
        if not (lo <= y <= hi):
            continue
        d = onsets.get(y)
        rec = {"river": river, "season": season, "year": y, "inundation": str(d.date()) if d is not None else None}
        for key, src in (("gauge", gauge2), ("reanalysis", rean), ("forecast", fcst)):
            dd = either(src, season, y)
            rec[key + "_date"] = str(dd.date()) if dd is not None else None
            rec[key + "_lead"] = (d - dd).days if (d is not None and dd is not None) else None
        out.append(rec)
        f = lambda k: (rec[k + "_date"] or "never")
        g = lambda k: (f"{rec[k + '_lead']:+d} d" if rec[k + "_lead"] is not None else "-")
        print(f"{season.title() + ' ' + river.title() + ' ' + str(y):18}{(rec['inundation'] or 'below 1-in-5'):>13}{f('gauge'):>17}{g('gauge'):>6}{f('reanalysis'):>17}{g('reanalysis'):>6}{f('forecast'):>16}{g('forecast'):>6}")
json.dump(out, open("floodscan_lead.json", "w"), indent=1)
with_in = [o for o in out if o["inundation"]]
print(f"\n{len(out)} two-gauge flood seasons with a forecast archive, {len(with_in)} with a FloodScan 1-in-5 inundation")
for key, lab in (("gauge", "SWALIM gauges"), ("reanalysis", "reanalysis"), ("forecast", "forecasts")):
    v = [o[key + "_lead"] for o in with_in]
    print(f"  {lab:14} 8+ d before: {sum(1 for x in v if x is not None and x >= 8)}  on the day to 7 d before: {sum(1 for x in v if x is not None and 0 <= x < 8)}  after: {sum(1 for x in v if x is not None and x < 0)}  never: {sum(1 for x in v if x is None)}")
