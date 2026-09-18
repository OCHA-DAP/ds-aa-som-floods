"""For each SWALIM-matched river-season: activation date and the points over their
return-period thresholds on that day, for (a) the trigger on its own record (Google
reanalysis for Gu, GloFAS v5 reanalysis for Deyr), (b) the GloFAS v4 forecast replay
(ensemble median at leads 1-7, v4-fitted levels, by valid day and by issue date), (c)
the GloFAS v5 reanalysis. Rules and levels as somlib (TRIGGER_CONFIG, levels fitted
on TRIGGER_YEARS, season months only). Writes trigger_detail.json, read by
swalim_section.py. Run with SOM_DATA_REPO set when this file sits in a worktree."""
import json
import sys
from pathlib import Path

import pandas as pd

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
import somlib as L  # noqa: E402
from src.constants import SEASONS, TRIGGER_CONFIG, TRIGGER_STATIONS  # noqa: E402

ROWS = [("juba", "deyr", 2006), ("shabelle", "deyr", 2006), ("juba", "deyr", 2014), ("shabelle", "deyr", 2014),
        ("shabelle", "gu", 2016), ("juba", "deyr", 2019), ("shabelle", "deyr", 2019), ("shabelle", "deyr", 2020),
        ("juba", "gu", 2020), ("shabelle", "gu", 2020), ("juba", "gu", 2021), ("shabelle", "gu", 2021),
        ("shabelle", "gu", 2023), ("juba", "deyr", 2023), ("shabelle", "deyr", 2023)]
NAME = L.NAME
fmt = lambda d: pd.Timestamp(d).strftime("%d %b")


def first_activation(daily, thr, n_req, season, year):
    """daily: date x station. -> (date, [stations over], max points, first date of the max,
    [stations over on that date], number of days at or above n_req)."""
    cols = [c for c in thr.index if c in daily.columns and pd.notna(thr[c])]
    over = daily[cols] >= thr[cols]
    over = over[(over.index.year == year) & over.index.month.isin(SEASONS[season])]
    votes = over.sum(axis=1)
    if not len(votes):
        return None, [], 0, None, [], 0
    mx = int(votes.max()); dmax = votes.idxmax()
    pk = [NAME[c] for c in cols if over.loc[dmax, c]] if mx else []
    pk_date = fmt(dmax) if mx else None
    hit = votes[votes >= n_req]
    if len(hit):
        d = hit.index[0]
        return fmt(d), [NAME[c] for c in cols if over.loc[d, c]], mx, pk_date, pk, int(len(hit))
    return None, [], mx, pk_date, pk, 0


def first_issue(med_iv, thr, n_req, season, year):
    """med_iv: (issued_time, valid_time) x station ensemble medians. The first issue on
    which, for one valid day at leads 1-7, at least n_req points were over their level.
    -> (issue date, valid day, [stations over], max points over the season, issue date of
    the max, valid day of the max, [stations], number of issues at or above n_req)."""
    cols = [c for c in thr.index if c in med_iv.columns and pd.notna(thr[c])]
    over = med_iv[cols] >= thr[cols]
    vd = over.index.get_level_values("valid_time")
    over = over[(vd.year == year) & vd.month.isin(SEASONS[season])]
    if not len(over):
        return None, None, [], 0, None, None, [], 0
    votes = over.sum(axis=1)
    per_issue = votes.groupby(level="issued_time").max()
    mx = int(per_issue.max()); imax = per_issue.idxmax()
    vmax = votes.loc[imax].idxmax() if mx else None
    pk = [NAME[c] for c in cols if over.loc[(imax, vmax), c]] if mx else []
    hit = per_issue[per_issue >= n_req]
    if len(hit):
        i0 = hit.index[0]; v0 = votes.loc[i0].idxmax()
        return fmt(i0), fmt(v0), [NAME[c] for c in cols if over.loc[(i0, v0), c]], mx, fmt(imax), fmt(vmax), pk, int(len(hit))
    return None, None, [], mx, fmt(imax) if mx else None, fmt(vmax) if mx else None, pk, 0


rf = L.reforecast("glofas_v4")
rf = rf[rf.leadtime_days.between(1, 7)]
med_iv = rf.groupby(["issued_time", "valid_time", "station"]).discharge.median().unstack("station")
med = med_iv.groupby(level="valid_time").max()          # most alarming issue per valid day
daily = {m: L.daily(m).pivot_table(index="date", columns="station", values="discharge") for m in ("google_grrr", "glofas_v5")}

out = []
for river, season, year in ROWS:
    cfg = TRIGGER_CONFIG[(river, season)]
    rec = {"river": river, "season": season, "year": year, "n_req": cfg["n_req"],
           "n_points": len(TRIGGER_STATIONS[river]), "rp": cfg["rp"]}
    m = cfg["source"]
    rec["trigger"] = first_activation(daily[m], L.model_thresholds(m, river, season, cfg["rp"]), cfg["n_req"], season, year)
    thr4 = L.model_thresholds("glofas_v4", river, season, cfg["rp"])
    rec["v4_fc"] = first_activation(med, thr4, cfg["n_req"], season, year)
    rec["v4_fc_issue"] = first_issue(med_iv, thr4, cfg["n_req"], season, year)
    rec["thr_v4"] = {NAME[k]: round(float(v), 1) for k, v in thr4.items()}
    rec["v5"] = first_activation(daily["glofas_v5"], L.model_thresholds("glofas_v5", river, season, cfg["rp"]), cfg["n_req"], season, year)
    out.append(rec)
    print(f"{river:9s} {season:5s} {year} | trigger {rec['trigger'][:2]} | v4 issue {rec['v4_fc_issue'][:3]} | v5 {rec['v5'][:2]}")
(S / "trigger_detail.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote trigger_detail.json")
