"""Download the operational GloFAS forecast archive for every v4-era season.

Feeds the full-resolution agreement-level checks on pages/ensemble-agreement/:
the operational system has been v4 since July 2023, so these seasons pair
with the v4-fitted levels the way the trigger operates. Seasons before that
ran v3 (May 2021 - Jul 2023) and v2 (to May 2021) operationally; testing them
needs levels fitted on those versions' own climatologies, so they are not
fetched here.

50 perturbed members, daily issues, leads 1-12 days, one tiny box per trigger
point. Jobs are ordered so Deyr 2023 (the flagship missed flood) lands first,
Shabelle before Juba. Files that already exist are skipped: re-running is
safe, and Gu 2024 lives in its own directory from the earlier pull.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/download_operational_seasons.py
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import TRIGGER_STATIONS  # noqa: E402
from src.datasources import glofas  # noqa: E402

OUT = glofas.DATA_DIR / "raw" / "forecast_operational"
LEAD_HOURS = [str(24 * d) for d in range(1, 13)]

# (year, months, stations) in priority order. Gu 2024 was already fetched to
# forecast_gu2024/ (April-May; March was quiet at every gauge).
SEASONS = [
    ("2023", ["10", "11", "12"], TRIGGER_STATIONS["shabelle"]),  # Deyr 2023
    ("2023", ["10", "11", "12"], TRIGGER_STATIONS["juba"]),
    ("2024", ["10", "11", "12"], TRIGGER_STATIONS["shabelle"]),  # Deyr 2024
    ("2024", ["10", "11", "12"], TRIGGER_STATIONS["juba"]),
    ("2025", ["03", "04", "05"], TRIGGER_STATIONS["shabelle"]),  # Gu 2025
    ("2025", ["03", "04", "05"], TRIGGER_STATIONS["juba"]),
    ("2025", ["10", "11", "12"], TRIGGER_STATIONS["shabelle"]),  # Deyr 2025
    ("2025", ["10", "11", "12"], TRIGGER_STATIONS["juba"]),
    ("2026", ["03", "04", "05"], TRIGGER_STATIONS["shabelle"]),  # Gu 2026
    ("2026", ["03", "04", "05"], TRIGGER_STATIONS["juba"]),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client = glofas._client(wait_until_complete=False)
    coll = client.client.get_collection("cems-glofas-forecast")
    jobs = []
    for year, months, stations in SEASONS:
        for st in stations:
            for month in months:
                req = {
                    "variable": "river_discharge_in_the_last_24_hours",
                    "system_version": ["operational"],
                    "hydrological_model": ["lisflood"],
                    "product_type": ["ensemble_perturbed_forecasts"],
                    "year": [year],
                    "month": [month],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "leadtime_hour": LEAD_HOURS,
                    "area": glofas._station_area(st),
                    "data_format": "netcdf",
                }
                for creq, prefix in glofas._plan_chunks(
                        coll, req, f"{st}_{year}{month}_ens"):
                    jobs.append(("cems-glofas-forecast", creq,
                                 OUT / f"{prefix}.nc"))
    done = sum(1 for _, _, p in jobs if p.exists())
    print(f"[ops-archive] {len(jobs)} chunks planned, {done} already on disk")
    glofas._submit_rolling(client, jobs, log_prefix="[ops-archive] ")
    # submission failures inside _submit_rolling are printed, not retried:
    # verify the count on disk before trusting any season's result
    missing = [p.name for _, _, p in jobs if not p.exists()]
    print(f"[ops-archive] done; {len(missing)} chunks missing")
    for name in missing[:20]:
        print("  missing:", name)


if __name__ == "__main__":
    main()
