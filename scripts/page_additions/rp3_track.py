"""Tracking correlation in flood seasons only: for each station, season and model, the Spearman
rank correlation between the model's daily flow and the gauge's daily level, using only the days
of the seasons in which the gauge reached its own 1-in-3 level (2000 to 2023), at the best lag
between -10 and +30 days (positive: the model leads the gauge; the full-year gauge series is
shifted, as in station_metrics.py). Writes rp3_track.json."""
import json
import numpy as np
import pandas as pd
import somlib as L

MODELS = ["google_grrr", "glofas_v5"]
lv = L.levels()
out = []
for river in ("juba", "shabelle"):
    for season in ("deyr", "gu"):
        lev3 = L.gauge_levels(river, season, 3)
        for st in L.TRIGGER_STATIONS[river]:
            g_all = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
            g = g_all[g_all.index.month.isin(L.SEASONS[season]) & (g_all.index.year >= 2000) & (g_all.index.year <= 2023)]
            if np.isnan(lev3[st]) or not len(g):
                continue
            cnt = g.groupby(g.index.year).size(); gmax = g.groupby(g.index.year).max()
            flood_years = sorted(y for y in gmax[gmax >= lev3[st]].index if cnt[y] >= 30)
            rec = {"river": river, "season": season, "station": L.NAME[st], "flood_years": flood_years, "n_flood_seasons": len(flood_years)}
            for m in MODELS:
                s = L.season_series(m, st, season); s = s[s.index.year.isin(flood_years)]
                best, best_lag, n_best = 0.0, 0, 0
                for lag in range(L.MIN_LAG, L.MAX_LAG + 1):
                    gg = g_all.copy(); gg.index = gg.index - pd.Timedelta(days=lag)
                    j = pd.concat([s.rename("m"), gg.rename("g")], axis=1).dropna()
                    if len(j) < L.MIN_OBS:
                        continue
                    r = j["m"].corr(j["g"], method="spearman")
                    if r > best:
                        best, best_lag, n_best = float(r), lag, len(j)
                rec[m] = round(best, 2); rec[m + "_lag"] = best_lag; rec[m + "_n_days"] = n_best
            out.append(rec)
json.dump(out, open("rp3_track.json", "w"), indent=1)
df = pd.DataFrame(out)
print("per station: best-lag daily Spearman over the gauge's own 1-in-3 seasons")
for _, r in df.iterrows():
    print(f"  {r.season:4} {r.river:9} {r.station:12} seasons={r.n_flood_seasons} | Google {r.google_grrr} lag {r.google_grrr_lag:+d} n{r.google_grrr_n_days} | v5 {r.glofas_v5} lag {r.glofas_v5_lag:+d} n{r.glofas_v5_n_days}")
print("per window, median across stations")
for (s, rv), g in df.groupby(["season", "river"]):
    print(f"  {s:4} {rv:9} | Google {g.google_grrr.median():.2f} | v5 {g.glofas_v5.median():.2f}")
for s, g in df.groupby("season"):
    print(f"  {s:4} either river | Google {g.google_grrr.median():.2f} | v5 {g.glofas_v5.median():.2f}")
