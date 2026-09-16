"""Step 1 of the daily run: fetch today's forecasts and store them.

- GloFAS operational ensemble at the seven trigger points (EWDS), with the
  system-version guard: the run fails if EWDS has retired v4 to the legacy
  list or the GRIB process identifiers changed, unless
  ALLOW_VERSION_MISMATCH=true. Applying the frozen v4 thresholds to a v5
  forecast would be wrong, so failing loudly is the safe default.
- Google Flood Hub at the six gauges the live API serves (Dollow's HYBAS
  gauge is not in the operational feed).

Rows are written to blob as monitoring/forecasts/<date>.parquet (dev), the store the later steps read.
MONITORING_DATE (YYYY-MM-DD) overrides today for re-runs; Google always
returns its current forecast, so a re-run for a past date carries today's
Google issue and is labelled as such by issued_time.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd  # noqa: E402

from src.monitoring import config as cfg  # noqa: E402
from src.monitoring import etl  # noqa: E402
from src.monitoring.flags import env_flag  # noqa: E402


def fetch_glofas_waiting(monitoring_date):
    """Ask EWDS for the day's own GloFAS issue; while it is not published yet, retry every
    15 minutes until WAIT_FOR_ISSUE_UNTIL_UTC (hour, default 16), then accept the previous
    day's issue. Runs for a past date do not wait."""
    deadline_h = int(os.getenv("WAIT_FOR_ISSUE_UNTIL_UTC", "16"))
    while True:
        try:
            return etl.fetch_glofas(monitoring_date, max_days_back=0)
        except RuntimeError as exc:
            now = datetime.now(timezone.utc)
            if monitoring_date != now.date() or now.hour >= deadline_h:
                print(f"  {monitoring_date} issue not on EWDS ({str(exc)[:80]}); taking the previous issue")
                return etl.fetch_glofas(monitoring_date, max_days_back=1)
            print(f"  {now:%H:%M} UTC: today's issue not on EWDS yet; retrying in 15 min, until {deadline_h}:00 UTC")
            time.sleep(900)


def main():
    monitoring_date = etl.monitoring_date_from_env()
    dry_run = env_flag("DRY_RUN", True)
    print(f"monitoring date {monitoring_date}; operational GloFAS assumed {cfg.GLOFAS_OPERATIONAL}")

    version = etl.check_glofas_version()
    print(f"EWDS legacy versions: {version['legacy_versions']}")
    if version["v4_retired"] and cfg.GLOFAS_OPERATIONAL == "glofas_v4":
        msg = ("EWDS now lists a version_4 entry under Legacy Versions: the operational "
               "forecast has moved on. Flip GLOFAS_OPERATIONAL in src/monitoring/config.py "
               "and re-check the thresholds (pages/glofas-version/).")
        if not env_flag("ALLOW_VERSION_MISMATCH", False):
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg)

    print("fetching GloFAS ...")
    df_glofas, meta = fetch_glofas_waiting(monitoring_date)
    print(f"  issue {meta['issue_date']} ({meta['days_back']} d back), {len(df_glofas)} rows, "
          f"process ids {meta['process_ids']}, expected {cfg.GLOFAS_EXPECTED_PROCESS}")
    if not meta["version_ok"]:
        msg = ("GRIB process identifiers differ from the expected operational system; "
               "the GloFAS version may have changed.")
        if not env_flag("ALLOW_VERSION_MISMATCH", False):
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg)

    print("fetching Google Flood Hub ...")
    df_google, gmeta = etl.fetch_google(monitoring_date)
    print(f"  {gmeta['n_gauges']} gauges, {len(df_google)} rows; missing: {gmeta['missing']}")

    df = pd.concat([df_glofas, df_google], ignore_index=True)
    if dry_run:
        print(f"DRY_RUN: not writing {len(df)} rows")
    else:
        n = etl.save_rows(df, monitoring_date)
        print(f"wrote {n} rows to blob {etl.forecasts_blob(monitoring_date)}")


if __name__ == "__main__":
    main()
    # cfgrib/eccodes can segfault during interpreter teardown on Linux; all
    # work is done and flushed by here.
    sys.stdout.flush()
    os._exit(0)
