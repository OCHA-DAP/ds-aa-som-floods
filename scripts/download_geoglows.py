"""Download GEOGloWS v2 data for all Juba/Shabelle stations.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/download_geoglows.py

Saves to data/geoglows/:
- retro_daily_<station>.parquet      daily retrospective discharge, 1940-present
- return_periods_<station>.parquet   Gumbel return periods (daily max)
- forecast_stats_<station>_<issue>.parquet      latest 15-day forecast stats
- forecast_ensembles_<station>_<issue>.parquet  latest 52-member ensemble

Retrospective/return periods are skipped when the file already exists;
forecasts are skipped only if today's issue was already saved.
"""

import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import STATIONS
from src.datasources import geoglows_data as gg


def main():
    gg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    retro_missing = {
        key: st.geoglows_river_id
        for key, st in STATIONS.items()
        if not (gg.DATA_DIR / f"retro_daily_{key}.parquet").exists()
    }
    if retro_missing:
        print(f"Retrospective: fetching {len(retro_missing)} stations...")
        df = gg.download_retro_daily(list(retro_missing.values()))
        for key, rid in retro_missing.items():
            out = gg.DATA_DIR / f"retro_daily_{key}.parquet"
            df[[rid]].rename(columns={rid: str(rid)}).to_parquet(out)
            print(f"  saved {out.name}")
    else:
        print("Retrospective: all stations already downloaded")

    rp_missing = {
        key: st.geoglows_river_id
        for key, st in STATIONS.items()
        if not (gg.DATA_DIR / f"return_periods_{key}.parquet").exists()
    }
    if rp_missing:
        print(f"Return periods: fetching {len(rp_missing)} stations...")
        df = gg.download_return_periods(list(rp_missing.values()))
        for key, rid in rp_missing.items():
            out = gg.DATA_DIR / f"return_periods_{key}.parquet"
            df[[rid]].rename(columns={rid: str(rid)}).to_parquet(out)
            print(f"  saved {out.name}")
    else:
        print("Return periods: all stations already downloaded")

    print("Forecasts: fetching latest per station...")
    for key, st in STATIONS.items():
        rid = st.geoglows_river_id
        try:
            stats = gg.download_forecast_stats(rid)
        except Exception as e:
            print(f"  {key}: forecast stats FAILED - {e}")
            continue
        issue = pd.Timestamp(stats.index.min()).date().isoformat()
        stats_path = gg.DATA_DIR / f"forecast_stats_{key}_{issue}.parquet"
        if stats_path.exists():
            print(f"  {key}: issue {issue} already saved, skipping")
            continue
        stats.to_parquet(stats_path)
        try:
            ens = gg.download_forecast_ensembles(rid)
            ens.to_parquet(gg.DATA_DIR / f"forecast_ensembles_{key}_{issue}.parquet")
        except Exception as e:
            print(f"  {key}: ensembles FAILED - {e}")
        print(f"  {key}: saved forecast issue {issue}")

    print("Done.")


if __name__ == "__main__":
    main()
