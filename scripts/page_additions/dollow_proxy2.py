"""Top Dollow candidates against the SWALIM Dollow gauge itself, and what the Gu Juba rule would
have done with each candidate standing in for the design's Dollow point. Saves the candidate
series (dollow_candidate_series.parquet) and writes dollow_proxy2.csv."""
import os, sys
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
sys.path.insert(0, r"C:\Users\pauni\Desktop\Work\OCHA\GitHub\ds-aa-som-floods"); sys.path.insert(0, ".")
import pandas as pd, numpy as np
from src.datasources import grrr
from src.utils import weibull_threshold
import somlib as L

CANDS = ["hybas_1121039440", "hybas_1122046190", "hybas_1121037840", "hybas_1121038680", "hybas_1121038740"]   # last = the design's Dollow point
df = grrr.download_reanalysis(CANDS); df.index = pd.to_datetime(df.index); df.to_parquet("dollow_candidate_series.parquet")
lv = L.levels(); g = lv[lv.station == "dollow"].set_index("date")["level_m"].dropna().sort_index()
print(f"SWALIM Dollow gauge: {g.index.min().date()} to {g.index.max().date()}, {len(g)} readings")
rows = []
for c in CANDS:
    s = df[c]
    rec = {"gauge_id": c, "mean_m3s": round(float(s[(s.index.year >= 1999) & (s.index.year <= 2023)].mean()), 1)}
    for season, months in (("gu", [3, 4, 5]), ("deyr", [10, 11, 12])):
        j = pd.concat([s.rename("m"), g.rename("g")], axis=1).dropna(); j = j[j.index.month.isin(months)]
        rec[f"rho_vs_gauge_{season}"] = round(float(j["m"].corr(j["g"], method="spearman")), 3) if len(j) > 60 else None
        rec[f"n_days_{season}"] = int(len(j))
    rows.append(rec)
out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv("dollow_proxy2.csv", index=False)

# Gu Juba rule (3 of 4 over own 1-in-5) with each candidate in Dollow's place
m = L.daily_matrix("google_grrr", "juba", "gu"); thr = L.model_thresholds("google_grrr", "juba", "gu", 5)
base_cols = ["luuq", "bardheere", "bualle"]
print("\nGu Juba, 3 of 4 over own 1-in-5, 1999-2023, with the candidate as the fourth point (design years: 2013, 2016, 2018, 2020):")
for c in CANDS:
    s = df[c]; s = s[s.index.month.isin([3, 4, 5])]; s = s[(s.index.year >= 1999) & (s.index.year <= 2023)]
    t = weibull_threshold(s.groupby(s.index.year).max().dropna().values, 5)
    mm = pd.concat([m[base_cols], s.rename("cand")], axis=1)
    over = pd.concat([(mm[b] >= thr[b]) for b in base_cols] + [(mm["cand"] >= t)], axis=1).fillna(False)
    votes = over.sum(axis=1); yrs = sorted({y for y in votes[votes >= 3].index.year if 1999 <= y <= 2023})
    print(f"  {c}: 1-in-5 level {t:.0f} m3/s, activations {yrs}")
