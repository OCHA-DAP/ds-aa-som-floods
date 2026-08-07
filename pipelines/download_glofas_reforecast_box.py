"""Databricks entrypoint: download GloFAS v4.2 reforecast for the SOM box.

Defaults target the readiness-band gap: leads 8-12 days, 2003-2023, Gu +
Deyr months, streamed chunk-by-chunk to blob under

  ds-aa-som-floods/raw/glofas/raw/reforecast_som_ext_lead8_12/

(the layout scripts/upload_to_blob.py mirrors, so a local
`stratus.list_container_blobs` + download restores the exact directory
notebook 10 section C expects). Chunks already in blob are skipped, so the
job is safe to re-kick after an interruption.

Deploy and run on Databricks:
    databricks bundle validate -t dev -p DEFAULT
    databricks bundle deploy   -t dev -p DEFAULT
    databricks bundle run download_glofas_reforecast_box -t dev -p DEFAULT

Smoke test (one year, one month — ~15-30 min):
    databricks bundle run download_glofas_reforecast_box -t dev -p DEFAULT \\
        --python-named-params "start_year=2020,end_year=2020,months=10"

Local run (writes to data/glofas/... unless SOM_DATA_DIR is set; add
blob_sync=true to mirror the Databricks behaviour):
    uv run python pipelines/download_glofas_reforecast_box.py --blob-sync false
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# On Databricks the git checkout is ephemeral (and possibly read-only):
# stage downloads in a temp dir. Must be set before importing the module.
if "DATABRICKS_RUNTIME_VERSION" in os.environ and "SOM_DATA_DIR" not in os.environ:
    os.environ["SOM_DATA_DIR"] = tempfile.mkdtemp(prefix="som_glofas_")

from src.datasources import glofas  # noqa: E402


def _csv_ints(text):
    return [int(v) for v in text.split(",") if v.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Download GloFAS reforecast for the SOM all-stations box."
    )
    p.add_argument("--leads", default="8,9,10,11,12",
                   help="Comma-separated lead times in days (default: 8-12)")
    p.add_argument("--start-year", type=int, default=2003)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--months", default=",".join(glofas.FLOOD_SEASON_MONTHS),
                   help="Comma-separated months (default: Gu + Deyr)")
    p.add_argument("--dir-suffix", default="_lead8_12",
                   help="Suffix of reforecast_som_ext<suffix>/ (default: _lead8_12)")
    p.add_argument("--blob-sync", type=lambda x: x.lower() == "true",
                   default=True, metavar="true|false",
                   help="Upload each chunk to blob and skip chunks already there")
    args = p.parse_args()

    leads = _csv_ints(args.leads)
    years = [str(y) for y in range(args.start_year, args.end_year + 1)]
    months = [str(m).zfill(2) for m in args.months.split(",") if m.strip()]
    print(
        f"GloFAS reforecast box download: leads={leads}, "
        f"years={years[0]}-{years[-1]}, months={months}, "
        f"dir_suffix={args.dir_suffix!r}, blob_sync={args.blob_sync}, "
        f"data_dir={glofas.DATA_DIR}"
    )

    n = glofas.download_reforecast_box(
        years=years,
        months=months,
        leadtime_days=leads,
        dir_suffix=args.dir_suffix,
        blob_sync=args.blob_sync,
    )
    print(f"\nDone: {n} chunks downloaded.")


if __name__ == "__main__":
    main()
