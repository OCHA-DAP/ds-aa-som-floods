"""Compare GloFAS v4.0 vs v5.0 reanalysis bias against observed discharge.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/compare_glofas_versions.py

Reports, per station with real observed discharge (the four SNRFA stations
scraped during the EF5 work), the magnitude bias of each GloFAS version
alongside GEOGloWS and Google GRRR: median, q95 and annual-maximum ratios
plus rank correlation. Answers whether v5.0 fixes v4.0's Shabelle wet bias
and therefore whether it is worth switching the reanalysis version.

Run scripts/process_data.py first so the v5 series is in the processed
table (it appears as source "glofas_v5" once downloaded).
"""

import json
import sys
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.processing import PROCESSED_DIR, load_processed

# stations that publish real discharge, and their blob JSON slugs
OBS_SLUGS = {
    "belet_weyne": "beletweyne",
    "bulo_burti": "buloburte",
    "luuq": "luuq",
    "bardheere": "bardheere",
}
SOURCES = ["geoglows", "google", "glofas", "glofas_v5"]


def load_observed(slug):
    raw = stratus.load_blob_data(
        f"ds-aa-som-floods/raw/ef5/analysis/{slug}_real_discharge.json", stage="dev"
    )
    s = pd.Series(json.loads(raw), dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def main():
    dd = load_processed("discharge_daily")
    present = [s for s in SOURCES if s in set(dd.source)]
    missing = [s for s in SOURCES if s not in present]
    if missing:
        print(f"note: not in processed table yet, skipping: {missing}")

    rows = []
    for key, slug in OBS_SLUGS.items():
        obs = load_observed(slug)
        for src in present:
            mod = dd[(dd.source == src) & (dd.station == key)].set_index("date")["discharge"]
            j = pd.concat([mod, obs], axis=1, join="inner").dropna()
            j.columns = ["mod", "obs"]
            j = j[j["obs"] > 0]
            if len(j) < 500:
                continue
            am = j.resample("YE").max().dropna()
            rows.append({
                "station": key,
                "source": src,
                "n_days": len(j),
                "median_ratio": j["mod"].median() / j["obs"].median(),
                "q95_ratio": j["mod"].quantile(0.95) / j["obs"].quantile(0.95),
                "annual_max_ratio": (am["mod"] / am["obs"]).median(),
                "spearman": j["mod"].corr(j["obs"], method="spearman"),
            })
    bias = pd.DataFrame(rows)

    for metric in ["median_ratio", "annual_max_ratio", "spearman"]:
        print(f"\n=== {metric} (ratios: >1 model runs high) ===")
        print(bias.pivot(index="station", columns="source", values=metric).round(2).to_string())

    if {"glofas", "glofas_v5"}.issubset(set(bias.source)):
        w = bias.pivot(index="station", columns="source", values="median_ratio")
        print("\n=== v5 vs v4 verdict (median ratio, distance from 1.0) ===")
        for st in w.index:
            v4, v5 = w.loc[st, "glofas"], w.loc[st, "glofas_v5"]
            better = "v5 closer" if abs(np.log(v5)) < abs(np.log(v4)) else "v4 closer"
            print(f"  {st:14s} v4={v4:5.2f} v5={v5:5.2f}  -> {better}")

    out = PROCESSED_DIR / "glofas_version_bias.csv"
    bias.to_csv(out, index=False)
    stratus.upload_csv_to_blob(
        bias, "ds-aa-som-floods/processed/glofas_version_bias.csv", stage="dev"
    )
    print(f"\nSaved {out.name} + blob copy.")


if __name__ == "__main__":
    main()
