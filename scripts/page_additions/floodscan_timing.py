"""Inundation timing: when FloodScan (SFED, 10-km buffer of the river's main stem) reaches its own
1-in-3 level in a season, against the day the river's first and second gauges crossed theirs.
Levels are Weibull on seasonal maxima, 2000-2023 for gauges (as on the page) and 1998-2023 for SFED
(the archive). Writes floodscan_timing.json."""
import json
import pandas as pd
import somlib as L
from src.utils import weibull_level

fs = pd.read_parquet("inundation_series.parquet"); fs["date"] = pd.to_datetime(fs["date"])   # people exposed per river's AA districts (floodscan_districts.py)
WINDOWS = [("juba", "deyr"), ("shabelle", "deyr"), ("juba", "gu"), ("shabelle", "gu")]
SEG_GAUGE = {("shabelle", "upper"): ["belet_weyne", "bulo_burti"], ("shabelle", "mid"): ["jowhar"], ("juba", "upper"): ["dollow", "luuq", "bardheere"], ("juba", "lower"): ["bualle"]}
SPAN = range(1999, 2024)
SFED_RP = 5          # inundation is dated when the buffer's flooded fraction reaches its own 1-in-5 level (gauges stay at 1-in-3)


def sfed_onsets(river, segment, season, rp=SFED_RP):
    s = fs[fs.river == river].set_index("date")["value"].sort_index()          # segment kept for the call signature only
    s = s[s.index.month.isin(L.SEASONS[season])]
    am = s.groupby(s.index.year).max().dropna()
    lev = weibull_level(am.values, rp)
    out = {}
    for y, g in s.groupby(s.index.year):
        hit = g[g >= lev]
        if len(hit) and y in SPAN:
            out[y] = hit.index.min().normalize()
    return out, float(lev)


res = {}
print(f"FloodScan (whole-river buffer) 1-in-{SFED_RP} onset against the gauges' 1-in-3, days (+ = inundation after the gauge day)")
print(f"{'window':15}{'yrs both':>9}{'vs 2nd gauge: median (range)':>32}{'vs 1st gauge: median (range)':>32}   SFED-only years | gauge-only years")
for river, season in WINDOWS:
    on_s, lev = sfed_onsets(river, "full", season)
    g2 = L.gauge_crossings(river, season, 3, span=list(SPAN)); g1 = L.gauge_crossings(river, season, 3, n_req=1, span=list(SPAN))
    both = sorted(set(on_s) & set(g2))
    d2 = sorted((on_s[y] - g2[y]).days for y in both); d1 = sorted((on_s[y] - g1[y]).days for y in both)
    med = lambda v: (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2) if v else None
    rng = lambda v: f"{v[0]} to {v[-1]}" if v else "-"
    res[f"{season}_{river}"] = {"level": lev, "sfed_rp": SFED_RP, "years_both": both, "vs_second": {y: (on_s[y] - g2[y]).days for y in both}, "vs_first": {y: (on_s[y] - g1[y]).days for y in both},
                                "sfed_only": sorted(set(on_s) - set(g2)), "gauge_only": sorted(set(g2) - set(on_s))}
    print(f"{season.title() + ' ' + river.title():15}{len(both):>9}{str(med(d2)) + ' (' + rng(d2) + ')':>32}{str(med(d1)) + ' (' + rng(d1) + ')':>32}   {res[f'{season}_{river}']['sfed_only']} | {res[f'{season}_{river}']['gauge_only']}")
    for y in both:
        print(f"      {y}: 1st gauge {g1[y].date()}  2nd gauge {g2[y].date()}  SFED onset {on_s[y].date()}  ({(on_s[y]-g2[y]).days:+d} d vs 2nd, {(on_s[y]-g1[y]).days:+d} d vs 1st)")
json.dump(res, open("floodscan_timing.json", "w"), indent=1, default=str)

print("\nBy segment, against the gauge(s) in that segment (first of them to cross):")
for river, season in WINDOWS:
    for (rv, seg), gauges in SEG_GAUGE.items():
        if rv != river:
            continue
        on_s, lev = sfed_onsets(river, seg, season)
        mat = L.gauge_matrix(river, season); lev3 = L.gauge_levels(river, season, 3)
        rows = []
        for y, d in on_s.items():
            g = mat[mat.index.year == y]
            dates = [g[st][g[st] >= lev3[st]].index.min() for st in gauges if st in g.columns and pd.notna(lev3.get(st)) and (g[st] >= lev3[st]).any()]
            if dates:
                rows.append((d - min(dates)).days)
        rows.sort()
        print(f"  {season.title():5} {river.title():9} {seg:6} ({'/'.join(L.NAME[g_] for g_ in gauges)}): n={len(rows)} median {rows[len(rows)//2] if rows else '-'} range {rows[0] if rows else '-'} to {rows[-1] if rows else '-'}")
