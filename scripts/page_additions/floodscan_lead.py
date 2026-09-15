"""Does the trigger catch the inundation 7 days ahead? For every season the two-gauge rule calls a
flood (the benchmark), the FloodScan inundation day (river-buffer flooded fraction over its own
1-in-5, where reached), the second-gauge day, and the first forecast
issue on which the window's source met the action rule (leads 1-7), same river and either
river, and the lead to the inundation day. Writes floodscan_lead.json."""
import json
import pandas as pd
import somlib as L
from src.utils import weibull_level

fs = pd.read_parquet("floodscan_daily.parquet"); fs["date"] = pd.to_datetime(fs["date"])
WINDOWS = [("juba", "gu"), ("juba", "deyr"), ("shabelle", "gu"), ("shabelle", "deyr")]
RULE = {("juba", "deyr"): (4, 3), ("shabelle", "deyr"): (4, 2), ("juba", "gu"): (5, 3), ("shabelle", "gu"): (6, 2)}
FC = {"deyr": "glofas_v4", "gu": "google_grrr"}
ARCHIVE = {"google_grrr": (2016, 2023), "glofas_v4": (2003, 2023)}
SFED_RP = 5
SPAN = list(range(1999, 2024))


def sfed_onsets(river, season, rp=SFED_RP):
    s = fs[(fs.river == river) & (fs.segment == "full")].set_index("date")["mean_sfed"].sort_index()
    s = s[s.index.month.isin(L.SEASONS[season])]
    lev = weibull_level(s.groupby(s.index.year).max().dropna().values, rp)
    return {y: g[g >= lev].index.min().normalize() for y, g in s.groupby(s.index.year) if (g >= lev).any() and y in SPAN}


first = {}
for river, season in WINDOWS:
    rp, n = RULE[(river, season)]
    first[(river, season)] = {y: v[0] for y, v in L.first_issue_dates(FC[season], river, season, rp, n, span=SPAN, leads=(1, 7)).items()}

out = []
print(f"{'season':16}{'2nd gauge':>11}{'inundation':>13}{'same-river issue':>18}{'lead':>7}{'either-river issue':>20}{'lead':>7}")
for river, season in WINDOWS:
    lo, hi = ARCHIVE[FC[season]]
    other = "shabelle" if river == "juba" else "juba"
    onsets = sfed_onsets(river, season)
    g2 = L.gauge_crossings(river, season, 3, span=SPAN)                 # the benchmark: second gauge over 1-in-3
    for y in sorted(g2):
        if not (lo <= y <= hi):
            continue
        d = onsets.get(y)                                                # FloodScan inundation day, if its 1-in-5 was reached
        same = first[(river, season)].get(y)
        both = [x for x in (first[(river, season)].get(y), first[(other, season)].get(y)) if x is not None]
        either = min(both) if both else None
        ls = (d - same).days if (d is not None and same is not None) else None; le = (d - either).days if (d is not None and either is not None) else None
        out.append({"river": river, "season": season, "year": y, "second_gauge": str(g2[y].date()), "inundation": str(d.date()) if d is not None else None,
                    "same_issue": str(same.date()) if same is not None else None, "same_lead": ls, "same_lead_vs_gauge": (g2[y] - same).days if same is not None else None,
                    "either_issue": str(either.date()) if either is not None else None, "either_lead": le, "either_lead_vs_gauge": (g2[y] - either).days if either is not None else None})
        f = lambda dd: dd.date().isoformat() if dd is not None else "never"
        print(f"{season.title() + ' ' + river.title() + ' ' + str(y):16}{g2[y].date().isoformat():>11}{(d.date().isoformat() if d is not None else 'below 1-in-5'):>13}{f(same):>18}{(str(ls) + ' d') if ls is not None else '-':>7}{f(either):>20}{(str(le) + ' d') if le is not None else '-':>7}")
json.dump(out, open("floodscan_lead.json", "w"), indent=1)
n = len(out); n_in = sum(1 for o in out if o["inundation"] is not None)
print(f"\n{n} two-gauge flood seasons with a forecast archive, {n_in} with a FloodScan 1-in-5 inundation")
for key, lab in (("same_lead", "same river"), ("either_lead", "either river")):
    v = [o[key] for o in out if o["inundation"] is not None]
    print(f"{lab}: rule met >=7 d before inundation: {sum(1 for x in v if x is not None and x >= 7)}; 1-6 d before: {sum(1 for x in v if x is not None and 0 < x < 7)}; same day or after: {sum(1 for x in v if x is not None and x <= 0)}; never: {sum(1 for x in v if x is None)}")
