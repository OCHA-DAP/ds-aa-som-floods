"""Download the GEOGloWS archived daily forecasts for all stations.

Usage (from repo root):
    .venv/Scripts/python.exe -u scripts/download_geoglows_forecast_archive.py

GEOGloWS has no multi-decade reforecast like GloFAS, but it does keep every
daily 52-member init as its own zarr on S3 (archive starts 2024-07-01), which
is the only route to its lead-time skill. This pulls each init, averages the
sub-daily steps to daily means, and writes one parquet per init to
data/geoglows/forecast_archive/<YYYYMMDDHH>.parquet. Existing files are
skipped, so the script resumes.

Reads run in a small thread pool: each init is an independent S3 read of two
river chunks (~5 s), so the job is I/O bound and a few workers cut the
wall-clock a lot without hammering the bucket.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import STATIONS
from src.datasources import geoglows_data as gg

WORKERS = 6


def fetch_one(init, river_ids, out_dir):
    out_path = out_dir / f"{init}.parquet"
    if out_path.exists():
        return init, "skipped"
    df = gg.download_forecast_archive_date(init, river_ids)
    df.to_parquet(out_path, index=False)
    return init, f"{len(df)} rows"


def main():
    out_dir = gg.FORECAST_ARCHIVE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    river_ids = [st.geoglows_river_id for st in STATIONS.values()]

    inits = gg.list_forecast_archive_dates()
    todo = [i for i in inits if not (out_dir / f"{i}.parquet").exists()]
    print(f"{len(inits)} archived inits ({inits[0]} -> {inits[-1]}); {len(todo)} to fetch")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, i, river_ids, out_dir): i for i in todo}
        for fut in as_completed(futures):
            init = futures[fut]
            try:
                _, msg = fut.result()
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(f"[{done}/{len(todo)}] {init}: {msg}")
            except Exception as e:
                failed += 1
                print(f"{init}: FAILED {type(e).__name__} {str(e)[:120]}")
    print(f"\nDone: {done} fetched, {failed} failed, {len(inits) - len(todo)} already present.")


if __name__ == "__main__":
    main()
