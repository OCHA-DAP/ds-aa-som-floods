"""By season, both rivers together. Inundation = FloodScan flood exposure summed over all 14
anticipatory-action districts, dated at its own 1-in-3 (SFED_RP). (1) Do the two-gauge benchmark years rank
high in it? (2) For each benchmark season-year with a forecast archive: the inundation day against
the SWALIM gauges (second gauge over 1-in-3), the reanalysis rule and the first forecast issue, each
the earliest on either river. Writes floodscan_rank.json and floodscan_lead.json."""
import json
import pandas as pd
import somlib as L
from src.utils import weibull_level

S = __import__("pathlib").Path(__file__).parent
ser = pd.read_parquet(S / "inundation_series.parquet"); ser["date"] = pd.to_datetime(ser["date"])
both = ser.groupby("date")["value"].sum().sort_index()
from src.constants import TRIGGER_CONFIG as _TC
RULE = {k: (v["rp"], v["n_req"]) for k, v in _TC.items()}
CAL = {"deyr": "glofas_v5", "gu": "google_grrr"}
FC = {"deyr": "glofas_v4", "gu": "google_grrr"}
ARCHIVE = {"google_grrr": (2016, 2023), "glofas_v4": (2003, 2023)}
SFED_RP = 3
SPAN = list(range(1999, 2024))
RIVERS = ("juba", "shabelle")
m = json.load(open(S / "metrics.json")); win = {w["window"]: w for w in m["windows"]}


def record_rule_dates(model, river, season, rp, n_req):
    mat = L.daily_matrix(model, river, season); thr = L.model_thresholds(model, river, season, rp)
    cols = [c for c in thr.index if c in mat.columns and not pd.isna(thr[c])]
    votes = (mat[cols] >= thr[cols]).sum(axis=1); hit = votes[votes >= n_req]
    return {y: g.index.min().normalize() for y, g in hit.groupby(hit.index.year) if y in SPAN}


def earliest(dicts, y):
    v = [d[y] for d in dicts if y in d]
    return min(v) if v else None


rank_out, lead_out = {}, []
for season in ("deyr", "gu"):
    s = both[both.index.month.isin(L.SEASONS[season])]
    am = s.groupby(s.index.year).max(); am = am[(am.index >= 1998) & (am.index <= 2023)]
    rk = am.rank(ascending=False); rp = (len(am) + 1) / rk
    fl = set().union(*(set(win[f"{season.title()} {r.title()}"]["flood_years"]) for r in RIVERS))
    sv = set().union(*(set(win[f"{season.title()} {r.title()}"]["severe_years"]) for r in RIVERS))
    lev = weibull_level(am.values, SFED_RP)
    onsets = {y: g[g >= lev].index.min().normalize() for y, g in s.groupby(s.index.year) if (g >= lev).any() and y in SPAN}
    top = am.sort_values(ascending=False)
    rank_out[season] = {"level": float(lev), "rp": SFED_RP, "n_seasons": int(len(am)),
                        "top8": [{"year": int(y), "people": int(v), "rp": round(float(rp[y]), 1), "bench": "severe" if y in sv else "flood" if y in fl else ""} for y, v in top.head(8).items()],
                        "benchmark": [{"year": int(y), "rank": int(rk[y]), "people": int(am[y]), "severe": y in sv} for y in sorted(fl) if y in rk.index],
                        "in_top8": sum(1 for y in fl if y in rk.index and rk[y] <= 8), "n_flood": len([y for y in fl if y in rk.index]),
                        "severe_in_top8": sum(1 for y in sv if y in rk.index and rk[y] <= 8), "n_severe": len(sv)}
    print(f"\n{season.title()}, both rivers, seasonal max people exposed in the 14 districts (1-in-{SFED_RP} level {lev/1000:.0f}k)")
    print("  top 8:", ", ".join(f"{y}{' S' if y in sv else ' F' if y in fl else ''} ({v/1000:.0f}k, 1-in-{rp[y]:.0f})" for y, v in top.head(8).items()))
    print("  two-gauge flood years:", ", ".join(f"{y}{' S' if y in sv else ''}: rank {int(rk[y])} ({am[y]/1000:.0f}k)" for y in sorted(fl) if y in rk.index))
    print(f"  in top 8: {rank_out[season]['in_top8']} of {rank_out[season]['n_flood']} flood years; severe {rank_out[season]['severe_in_top8']} of {len(sv)}")
    # records, either river
    g2 = [L.gauge_crossings(r, season, 3, span=SPAN) for r in RIVERS]
    re_ = [record_rule_dates(CAL[season], r, season, *RULE[(r, season)]) for r in RIVERS]
    fc = [{y: v[0] for y, v in L.first_issue_dates(FC[season], r, season, *RULE[(r, season)], span=SPAN, leads=(1, 7)).items()} for r in RIVERS]
    lo, hi = ARCHIVE[FC[season]]
    print(f"  {'season':12}{'inundation':>13}{'SWALIM 2nd gauge':>18}{'lead':>6}{'reanalysis':>13}{'lead':>6}{'forecast':>13}{'lead':>6}")
    for y in sorted(fl):
        if not (lo <= y <= hi):
            continue
        d = onsets.get(y)
        rec = {"season": season, "year": y, "inundation": str(d.date()) if d is not None else None}
        for key, dicts in (("gauge", g2), ("reanalysis", re_), ("forecast", fc)):
            dd = earliest(dicts, y)
            rec[key + "_date"] = str(dd.date()) if dd is not None else None
            rec[key + "_lead"] = (d - dd).days if (d is not None and dd is not None) else None
        lead_out.append(rec)
        f = lambda k: rec[k + "_date"] or "never"; g = lambda k: f"{rec[k + '_lead']:+d} d" if rec[k + "_lead"] is not None else "-"
        print(f"  {season.title() + ' ' + str(y):12}{(rec['inundation'] or 'below 1-in-' + str(SFED_RP)):>13}{f('gauge'):>18}{g('gauge'):>6}{f('reanalysis'):>13}{g('reanalysis'):>6}{f('forecast'):>13}{g('forecast'):>6}")
json.dump(rank_out, open(S / "floodscan_rank.json", "w"), indent=1)
json.dump(lead_out, open(S / "floodscan_lead.json", "w"), indent=1)
with_in = [o for o in lead_out if o["inundation"]]
print(f"\n{len(lead_out)} benchmark season-years with an archive, {len(with_in)} with an inundation day")
for key, lab in (("gauge", "SWALIM gauges"), ("reanalysis", "reanalysis"), ("forecast", "forecasts")):
    v = [o[key + "_lead"] for o in with_in]
    print(f"  {lab:14} 8+ d: {sum(1 for x in v if x is not None and x >= 8)}  0-7 d: {sum(1 for x in v if x is not None and 0 <= x < 8)}  after: {sum(1 for x in v if x is not None and x < 0)}  never: {sum(1 for x in v if x is None)}")
