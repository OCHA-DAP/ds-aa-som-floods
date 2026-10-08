"""Informational step: a season-long log of the GloFAS reanalysis, versions 4 and 5, at the
seven trigger stations, for the /monitoring-google/ page.

EWDS serves a near-real-time ("intermediate") reanalysis a day or two behind. Each day the
last three weeks are fetched and merged into a store that runs from the first day of the
season being monitored (Deyr from 1 September, Gu from 1 February), so the page can chart
the whole season. Past seasons stay in the store under their own key. Each version is read
against its own levels on the page: "what each model says the river has been doing", not a
forecast and not part of the trigger. The store is one JSON on blob
(monitoring/reanalysis_season.json) that export_monitoring_status.py embeds in status.json.
Failures here must not stop the daily run: the workflow step is continue-on-error.

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
# EWDS product per version: "intermediate" is the near-real-time stream. v4 serves it a day or
# two behind (checked 2026-10-08: ~90 s per request, data to two days before today). v5 is a
# fixed 1980-2025 reanalysis release with no near-real-time stream yet: EWDS rejects both
# products for 2026 (checked 2026-10-08). The request is still made daily so v5 appears on the
# page by itself the day ECMWF extends it; until then it fails fast and the page says so.
PRODUCT = {"glofas_v4": "intermediate", "glofas_v5": "intermediate"}
# days fetched each run; REANALYSIS_DAYS_BACK overrides it for a one-off backfill of the season
DAYS_BACK = int(os.environ.get("REANALYSIS_DAYS_BACK", "21"))
BLOB = f"{cfg.PROJECT_PREFIX}/monitoring/reanalysis_season.json"
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


def season_of(day):
    """(key, first day) of the season whose monitoring window holds `day`; Deyr runs
    September to January, Gu February to June. Outside both: a rolling window."""
    for season, months in cfg.MONITORING_OPEN_MONTHS.items():
        if day.month in months:
            start_month = min(m for m in months if m >= 7) if season == "deyr" else min(months)
            year = day.year - 1 if (season == "deyr" and day.month < 7) else day.year
            return f"{season}_{year}", dt.date(year, start_month, 1)
    return f"rolling_{day:%Y%m}", day - dt.timedelta(days=DAYS_BACK)


def load_store():
    from azure.core.exceptions import ResourceNotFoundError
    local = os.environ.get("REANALYSIS_STORE_FILE")
    if local and Path(local).exists():
        return json.loads(Path(local).read_text(encoding="utf-8"))
    if local:
        return {}
    try:
        return json.loads(etl._container().get_blob_client(BLOB).download_blob().readall())
    except ResourceNotFoundError:
        return {}


def merge(store_series, new_series, start):
    """Upsert the fetched days into the season's series and keep only days from `start`."""
    out = {}
    for st in STATIONS:
        old = store_series.get(st, {"dates": [], "value": []})
        merged = dict(zip(old["dates"], old["value"]))
        new = new_series.get(st, {"dates": [], "value": []})
        merged.update(zip(new["dates"], new["value"]))   # the newest fetch wins for a day
        keep = sorted(d for d in merged if d >= str(start))
        out[st] = {"dates": keep, "value": [merged[d] for d in keep]}
    return out


def main():
    monitoring_date = etl.monitoring_date_from_env()
    days = [monitoring_date - dt.timedelta(days=i) for i in range(0, DAYS_BACK + 1)]
    season, start = season_of(monitoring_date)
    store = load_store()
    entry = store.get("seasons", {}).get(season, {"versions": {}})
    entry.update({"season": season, "season_start": str(start), "monitoring_date": str(monitoring_date),
                  "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "product": "intermediate"})
    for key, version in VERSIONS.items():
        prev = entry["versions"].get(key, {})
        try:
            series = merge(prev.get("series", {}), fetch(key, version, days), start)
            last = max((s["dates"][-1] for s in series.values() if s["dates"]), default=None)
            entry["versions"][key] = {"ewds_version": version, "product": PRODUCT[key], "last_date": last, "series": series}
            print(f"{key}: {version}, season {season} from {start}, latest day {last}, "
                  f"{len(series['jowhar']['dates'])} days, jowhar last {series['jowhar']['value'][-1] if series['jowhar']['value'] else 'n/a'} m3/s")
        except Exception as exc:  # one version failing must not lose the other, nor the stored days
            entry["versions"][key] = {**prev, "ewds_version": version, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"{key}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
    store.setdefault("seasons", {})[season] = entry
    store["current"] = season
    if env_flag("DRY_RUN", True):
        p = Path("temp"); p.mkdir(exist_ok=True)
        (p / "reanalysis_season.json").write_text(json.dumps(store, indent=1), encoding="utf-8")
        print("DRY_RUN: written to temp/reanalysis_season.json, not to blob")
    else:
        etl._container(write=True).upload_blob(BLOB, json.dumps(store, indent=1).encode(), overwrite=True)
        print(f"wrote {BLOB}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
