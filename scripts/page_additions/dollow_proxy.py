"""Rank the Google gauges near Dollow as proxies for the design's Dollow point: reanalysis
discharge downloaded from the public GRRR store, compared with Dollow's own series (magnitude and
day-to-day correlation over Gu). Writes dollow_proxy.csv."""
import os, sys
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
sys.path.insert(0, r"C:\Users\pauni\Desktop\Work\OCHA\GitHub\ds-aa-som-floods")
import pandas as pd
from src.datasources import grrr

cand = pd.read_csv("dollow_candidates.csv")
ids = [g for g in cand.gauge_id if g != "hybas_1121038740"][:14]
print("downloading", len(ids), "candidate gauges from the GRRR reanalysis store", flush=True)
df = grrr.download_reanalysis(ids)                       # date x gauge_id
dollow = pd.read_parquet(r"C:\Users\pauni\Desktop\Work\OCHA\GitHub\ds-aa-som-floods\data\google\reanalysis_dollow.parquet")
dcol = [c for c in dollow.columns if "discharge" in c.lower() or "value" in c.lower()]
d = dollow.set_index(pd.to_datetime(dollow["date"]))[dcol[0]] if "date" in dollow.columns else dollow.iloc[:, 0]
d = d[(d.index.year >= 1999) & (d.index.year <= 2023)]
rows = []
for g in ids:
    s = df[g]; s.index = pd.to_datetime(s.index); s = s[(s.index.year >= 1999) & (s.index.year <= 2023)]
    j = pd.concat([d.rename("dollow"), s.rename("g")], axis=1).dropna()
    gu = j[j.index.month.isin([3, 4, 5])]
    rows.append({"gauge_id": g, "km": float(cand.set_index("gauge_id").km_from_Dollow[g]),
                 "mean_flow_m3s": round(float(s.mean()), 1), "dollow_mean_m3s": round(float(d.mean()), 1),
                 "ratio_to_dollow": round(float(s.mean() / d.mean()), 2),
                 "corr_daily_gu": round(float(gu["g"].corr(gu["dollow"])), 3),
                 "corr_gu_seasonal_max": round(float(gu.groupby(gu.index.year).max().corr().iloc[0, 1]), 3)})
out = pd.DataFrame(rows).sort_values(["corr_daily_gu"], ascending=False)
out.to_csv("dollow_proxy.csv", index=False)
print(out.to_string(index=False))
