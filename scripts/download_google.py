"""Download Google GRRR (flood-forecasting model) data for all stations.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/download_google.py

Saves to data/google/:
- reanalysis_<station>.parquet       daily discharge, 1980 to end-2023
- return_periods_<station>.parquet   discharge at RP 2-200 years
- reforecast_<station>.parquet       daily issues 2016-2023, leads 0-7 days

Files that already exist are skipped.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import STATIONS
from src.datasources import grrr


def main():
    grrr.DATA_DIR.mkdir(parents=True, exist_ok=True)

    ra_missing = {
        key: st.grrr_gauge_id
        for key, st in STATIONS.items()
        if not (grrr.DATA_DIR / f"reanalysis_{key}.parquet").exists()
    }
    if ra_missing:
        print(f"Reanalysis: fetching {len(ra_missing)} stations...")
        df = grrr.download_reanalysis(list(ra_missing.values()))
        for key, gid in ra_missing.items():
            out = grrr.DATA_DIR / f"reanalysis_{key}.parquet"
            df[[gid]].to_parquet(out)
            print(f"  saved {out.name}")
    else:
        print("Reanalysis: all stations already downloaded")

    rp_missing = {
        key: st.grrr_gauge_id
        for key, st in STATIONS.items()
        if not (grrr.DATA_DIR / f"return_periods_{key}.parquet").exists()
    }
    if rp_missing:
        print(f"Return periods: fetching {len(rp_missing)} stations...")
        df = grrr.download_return_periods(list(rp_missing.values()))
        for key, gid in rp_missing.items():
            out = grrr.DATA_DIR / f"return_periods_{key}.parquet"
            df[[gid]].to_parquet(out)
            print(f"  saved {out.name}")
    else:
        print("Return periods: all stations already downloaded")

    print("Reforecast: fetching per station (issues 2016-2023, leads 0-7d)...")
    for key, st in STATIONS.items():
        out = grrr.DATA_DIR / f"reforecast_{key}.parquet"
        if out.exists():
            print(f"  {key}: already downloaded, skipping")
            continue
        df = grrr.download_reforecast(st.grrr_gauge_id)
        df.to_parquet(out)
        print(f"  saved {out.name} ({len(df)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
