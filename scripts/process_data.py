"""Build all processed tables and upload them to blob.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/process_data.py

Rebuilds every table in src.processing from whatever raw data exists,
writes them to data/processed/, and uploads each to blob under
ds-aa-som-floods/processed/ (overwriting — the processed layer is always
a full rebuild, unlike the append-only raw layer).

Multi-source tables are split into one file per source, with the product
named in the file (discharge_daily_glofas_v4.parquet etc.) so provenance is
visible in a blob listing. Read them back with
src.processing.load_processed(kind[, source]).
"""

import sys
from pathlib import Path

import ocha_stratus as stratus

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import processing
from src.constants import BLOB_PREFIX, BLOB_STAGE

BUILDERS = {
    "discharge_daily": processing.build_discharge_daily,
    "return_periods": processing.build_return_periods,
    "reforecast": processing.build_reforecast,
    "swalim_levels": processing.build_swalim_levels,
    "swalim_thresholds": processing.build_swalim_thresholds,
}


def write(df, filename):
    path = processing.PROCESSED_DIR / filename
    df.to_parquet(path, index=False)
    stratus.upload_parquet_to_blob(df, f"{BLOB_PREFIX}/{filename}", stage=BLOB_STAGE)
    print(f"  {len(df):>10,} rows -> {filename}")


def main():
    processing.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in BUILDERS.items():
        print(f"=== {name} ===")
        df = build()
        if "source" in df.columns:
            for source, part in df.groupby("source"):
                label = processing.SOURCE_FILE_LABELS.get(source, source)
                write(part.reset_index(drop=True), f"{name}_{label}.parquet")
        else:
            write(df, f"{name}.parquet")
    print("Done.")


if __name__ == "__main__":
    main()
