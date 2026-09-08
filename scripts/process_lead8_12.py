"""Build the GloFAS leads 8-12 readiness-band reforecast table and upload it.

Usage (from repo root):
    .venv/bin/python scripts/process_lead8_12.py

Standalone (not part of process_data.py's full rebuild) because the full
rebuild needs every raw source locally, while this table only needs the
reforecast_som_ext_lead8_12/ zips (scripts/restore_from_blob.py fetches
them). Writes data/processed/reforecast_glofas_v4_lead8_12.parquet and
mirrors it to blob.
"""

import sys
from pathlib import Path

import ocha_stratus as stratus

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import processing
from src.constants import BLOB_PREFIX, BLOB_STAGE

FILENAME = "reforecast_glofas_v4_lead8_12.parquet"


def main():
    df = processing.build_reforecast_lead8_12()
    processing.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = processing.PROCESSED_DIR / FILENAME
    df.to_parquet(path, index=False)
    stratus.upload_parquet_to_blob(df, f"{BLOB_PREFIX}/{FILENAME}", stage=BLOB_STAGE)
    print(f"{len(df):,} rows -> {FILENAME} (local + blob)")
    print(df.groupby("leadtime_days")["issued_time"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
