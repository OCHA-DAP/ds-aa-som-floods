"""Download the operational GloFAS forecasts for Gu 2024 (out-of-sample check).

The 2024 operational system was v4, so these forecasts pair with the v4
reanalysis thresholds the way the trigger operates. One request per station
and month (tiny 0.1-degree boxes keep every request far below the cost
limit), control plus all perturbed members, leads 1-12 days.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/download_gu2024_forecast.py

Saves to data/glofas/raw/forecast_gu2024/<station>_<month>.nc; files that
already exist are skipped, so re-running is safe.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import TRIGGER_STATIONS  # noqa: E402
from src.datasources import glofas  # noqa: E402

OUT = glofas.DATA_DIR / "raw" / "forecast_gu2024"
LEAD_HOURS = [str(24 * d) for d in range(1, 13)]
STATIONS = [s for v in TRIGGER_STATIONS.values() for s in v]


def main():
    client = glofas._client(wait_until_complete=False)
    coll = client.client.get_collection("cems-glofas-forecast")
    jobs = []
    # April-May issues cover the whole event window (gauge crossings ran
    # 19 Apr - 25 May 2024, and lead-12 readiness for mid-April needs early
    # April issues); the 51-member ensemble alone carries the median
    for st in STATIONS:
        for month in ("04", "05"):
            for prod, tag in [("ensemble_perturbed_forecasts", "ens")]:
                req = {
                    "variable": "river_discharge_in_the_last_24_hours",
                    "system_version": ["operational"],
                    "hydrological_model": ["lisflood"],
                    "product_type": [prod],
                    "year": ["2024"],
                    "month": [month],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "leadtime_hour": LEAD_HOURS,
                    "area": glofas._station_area(st),
                    "data_format": "netcdf",
                }
                for creq, prefix in glofas._plan_chunks(
                        coll, req, f"{st}_2024{month}_{tag}"):
                    jobs.append(("cems-glofas-forecast", creq,
                                 OUT / f"{prefix}.nc"))
    print(f"[gu2024] {len(jobs)} chunks planned")
    glofas._submit_rolling(client, jobs, log_prefix="[gu2024] ")


if __name__ == "__main__":
    main()
