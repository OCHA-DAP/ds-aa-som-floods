"""Informational step: the last weeks of GloFAS reanalysis, versions 4 and 5, at the seven
trigger stations, for the /monitoring-google/ page.

EWDS serves a near-real-time ("intermediate") reanalysis for both versions, a day or two
behind. Each version is read against its own Deyr/Gu levels on the page, so this is "what each
model says the river has been doing", not a forecast and not part of the trigger. The result is
one small JSON on blob (monitoring/reanalysis_recent.json) that export_monitoring_status.py
embeds in status.json when present. Failures here must not stop the daily run: the workflow
step is continue-on-error and the page shows the last successful fetch.

    MONITORING_DATE=YYYY-MM-DD python pipelines/fetch_recent_reanalysis.py
"""

import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import xarray as xr  # noqa: E402

from src.constants import TRIGGER_STATIONS  # noqa: E402
from src.datasources.glofas import _client  # noqa: E402
from src.monitoring import config as cfg, etl  # noqa: E402
from src.monitoring.flags import env_flag  # noqa: E402

VERSIONS = {"glofas_v4": "version_4_0", "glofas_v5": "version_5_0"}
# EWDS product per version: "intermediate" is the near-real-time stream (v4 serves it a day
# or two behind); set per version once EWDS confirms what v5 offers for the current period
PRODUCT = {"glofas_v4": "intermediate", "glofas_v5": "intermediate"}
DAYS_BACK = 21
BLOB = f"{cfg.PROJECT_PREFIX}/monitoring/reanalysis_recent.json"
STATIONS = [s for river in ("juba", "shabelle") for s in TRIGGER_STATIONS[river]]


def request(key, version, days):
    return {
        "system_version": version,
        "hydrological_model": "lisflood",
        "product_type": PRODUCT[key],
        "variable": "average_river_discharge_in_the_last_24_hours",
        "timespan": "time_mean",
        "year": sorted({str(d.year) for d in days}),
        "month": sorted({f"{d.month:02d}" for d in days}),
        "day": [f"{d:02d}" for d in range(1, 32)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": etl.FORECAST_AREA,
    }


def fetch(key, version, days):
    out = Path(tempfile.mkdtemp()) / f"recent_{version}.nc"
    _client(wait_until_complete=True).retrieve("cems-glofas-historical", request(key, version, days), str(out))
    ds = xr.open_dataset(out)
    var = next(v for v in ds.data_vars if "dis" in v.lower())
    tdim = next(d for d in ds[var].dims if "time" in d)
    series = {}
    for st in STATIONS:
        cell = etl.CELLS[st]
        da = ds[var].sel(latitude=cell["lat"], longitude=cell["lon"], method="nearest")
        s = da.to_series()
        if s.index.nlevels > 1:
            s.index = s.index.get_level_values(tdim)
        s = s.dropna().sort_index()
        s = s[s.index >= str(days[-1])]
        series[st] = {"dates": [t.strftime("%Y-%m-%d") for t in s.index], "value": [round(float(v), 1) for v in s.values]}
    return series


def main():
    monitoring_date = etl.monitoring_date_from_env()
    days = [monitoring_date - dt.timedelta(days=i) for i in range(0, DAYS_BACK + 1)]
    out = {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "monitoring_date": str(monitoring_date),
           "product": "intermediate", "days_back": DAYS_BACK, "versions": {}}
    for key, version in VERSIONS.items():
        try:
            series = fetch(key, version, days)
            last = max((s["dates"][-1] for s in series.values() if s["dates"]), default=None)
            out["versions"][key] = {"ewds_version": version, "product": PRODUCT[key], "last_date": last, "series": series}
            print(f"{key}: {version}, latest day {last}, "
                  f"jowhar last {series['jowhar']['value'][-1] if series['jowhar']['value'] else 'n/a'} m3/s")
        except Exception as exc:  # one version failing must not lose the other
            out["versions"][key] = {"ewds_version": version, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"{key}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
    if env_flag("DRY_RUN", True):
        p = Path("temp"); p.mkdir(exist_ok=True)
        (p / "reanalysis_recent.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("DRY_RUN: written to temp/reanalysis_recent.json, not to blob")
    else:
        etl._container(write=True).upload_blob(BLOB, json.dumps(out, indent=1).encode(), overwrite=True)
        print(f"wrote {BLOB}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
