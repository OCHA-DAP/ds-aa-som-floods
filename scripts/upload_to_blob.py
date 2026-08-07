"""Sync downloaded forecast-source data to Azure Blob via ocha_stratus.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/upload_to_blob.py

Mirrors data/geoglows, data/google and data/glofas into
ds-aa-som-floods/raw/{geoglows,google,glofas}/... on the projects/dev
container, preserving relative paths. Blobs that already exist are
skipped, so this is safe to re-run after each download finishes.
"""

import sys
from pathlib import Path

import ocha_stratus as stratus

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import BLOB_STAGE

DATA_DIR = repo_root / "data"
SOURCES = ["geoglows", "google", "glofas"]
RAW_PREFIX = "ds-aa-som-floods/raw"

CONTENT_TYPES = {
    ".parquet": "application/octet-stream",
    ".nc": "application/x-netcdf",
    ".csv": "text/csv",
}


def main():
    existing = set(
        stratus.list_container_blobs(name_starts_with=f"{RAW_PREFIX}/", stage=BLOB_STAGE)
    )
    n_up = n_skip = 0
    for source in SOURCES:
        src_dir = DATA_DIR / source
        if not src_dir.exists():
            continue
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file() or path.suffix not in CONTENT_TYPES:
                continue
            rel = path.relative_to(DATA_DIR).as_posix()
            blob_name = f"{RAW_PREFIX}/{rel}"
            if blob_name in existing:
                n_skip += 1
                continue
            stratus.upload_blob_data(
                path.read_bytes(),
                blob_name,
                stage=BLOB_STAGE,
                content_type=CONTENT_TYPES[path.suffix],
            )
            print(f"uploaded {blob_name}")
            n_up += 1
    print(f"Done: {n_up} uploaded, {n_skip} already present.")


if __name__ == "__main__":
    main()
