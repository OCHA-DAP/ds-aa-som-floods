"""Correlation with the SWALIM gauges in flood seasons only: for each station, season and model,
Spearman rank correlation between the model's seasonal peak and the gauge's seasonal peak,
using only the seasons in which the gauge reached its own 1-in-3 level or rarer."""
import json
import numpy as np
import pandas as pd
import somlib as L

MODELS = ["google_grrr", "glofas_v5", "glofas_v4"]
out = []
for river in ("juba", "shabelle"):
    for season in ("deyr", "gu"):
        lev3 = L.gauge_levels(river, season, 3)
        gm = L.gauge_matrix(river, season)
        for st in L.TRIGGER_STATIONS[river]:
            if st not in gm.columns or np.isnan(lev3[st]):
                continue
            g = gm[st].dropna(); gmax = g.groupby(g.index.year).max()
            gmax = gmax[(gmax.index >= 2000) & (gmax.index <= 2023)]
            flood_years = gmax[gmax >= lev3[st]].index
            rec = {"river": river, "season": season, "station": L.NAME[st], "n_flood_seasons": int(len(flood_years))}
            for m in MODELS:
                s = L.season_series(m, st, season); mmax = s.groupby(s.index.year).max()
                j = pd.concat([gmax.rename("g"), mmax.rename("m")], axis=1).dropna()
                jf = j.loc[j.index.intersection(flood_years)]
                rec[m] = round(float(jf["g"].corr(jf["m"], method="spearman")), 2) if len(jf) >= 4 else None
                rec[m + "_n"] = int(len(jf))
            out.append(rec)
json.dump(out, open("rp3_corr.json", "w"), indent=1)
df = pd.DataFrame(out)
print("per station: Spearman rho of seasonal peaks over the gauge's own 1-in-3+ seasons (n = seasons used)")
for _, r in df.iterrows():
    print(f"  {r.season:4} {r.river:9} {r.station:12} n={r.n_flood_seasons:2d} | Google {r.google_grrr} (n{r.google_grrr_n}) | v5 {r.glofas_v5} (n{r.glofas_v5_n}) | v4 {r.glofas_v4} (n{r.glofas_v4_n})")
print("\nper window: median across stations")
for (s, rv), g in df.groupby(["season", "river"]):
    print(f"  {s:4} {rv:9} | Google {g.google_grrr.median():.2f} | v5 {g.glofas_v5.median():.2f} | v4 {g.glofas_v4.median():.2f}")
print("\nall stations pooled, median:", {m: round(df[m].median(), 2) for m in MODELS})
