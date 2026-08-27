"""Build the bias-corrected GEOGloWS tables and mirror them to blob.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/build_geoglows_debias.py

Writes, under data/processed/ and blob ds-aa-som-floods/processed/:

* discharge_daily_geoglows_sfdc.parquet   retrospective, SFDC-corrected
                                          (all reaches, no gauge needed)
* reforecast_geoglows_gauge.parquet       forecasts corrected onto observed
                                          discharge, the four gauged stations

Standalone rather than part of process_data.py: it only needs the processed
GEOGloWS tables plus the GEOGloWS SFDC service, and it is slow enough
(one call per forecast issue) to be worth running on its own.
"""

import os
import sys
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
os.chdir(repo)
sys.path.insert(0, str(repo))

import ocha_stratus as stratus
import pandas as pd

from src import debias

PROCESSED = repo / "data" / "processed"
BLOB = "ds-aa-som-floods/processed"


def _load(name):
    return stratus.load_parquet_from_blob(f"{BLOB}/{name}.parquet", stage="dev")


def _save(df, name):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"{name}.parquet"
    df.to_parquet(out, index=False)
    stratus.upload_parquet_to_blob(df, f"{BLOB}/{name}.parquet", stage="dev")
    print(f"  wrote {out.name} ({len(df):,} rows) and mirrored to blob")


def main():
    daily = _load("discharge_daily_geoglows")
    daily["date"] = pd.to_datetime(daily["date"])
    print(f"GEOGloWS retrospective: {len(daily):,} rows, "
          f"{daily.station.nunique()} stations")

    print("\nSFDC-correcting the retrospective (project tables, no gauge needed)")
    corrected, report = debias.sfdc_correct_retrospective(daily)
    print(report.to_string(index=False))
    if len(corrected):
        _save(corrected, "discharge_daily_geoglows_sfdc")

    print("\nCorrecting forecasts onto observed discharge (gauged stations only)")
    fc = _load("reforecast_geoglows")
    for col in ("issued_time", "valid_time"):
        fc[col] = pd.to_datetime(fc[col])
    fc_corrected = debias.gauge_correct_forecast(fc, daily)
    if len(fc_corrected):
        summary = (
            fc_corrected.groupby("station")
            .agg(
                n_issues=("issued_time", "nunique"),
                raw_median=("discharge_raw", "median"),
                corrected_median=("discharge", "median"),
            )
            .round(1)
        )
        print(summary.to_string())
        _save(fc_corrected, "reforecast_geoglows_gauge")
    else:
        print("  no forecast issues corrected (check observed-discharge blobs)")

    print("\nDone.")


if __name__ == "__main__":
    main()
