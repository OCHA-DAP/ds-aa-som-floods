"""Fetch the operational forecasts at the seven trigger points and keep them.

Two products, one long table (projects.ds_aa_som_floods_monitoring, dev):

- GloFAS operational ensemble (EWDS cems-glofas-forecast, `operational`
  system version, 51 perturbed members, days 1-12) at the frozen river cells
  in glofas_cells.json. Stored per (station, valid day) as the ensemble
  median plus spread; the median is what the trigger reads.
- Google Flood Hub (Flood Forecasting API, gauges:queryGaugeForecasts) at the
  seven HYBAS gauges; deterministic daily values, latest issue per gauge.

Lead labelling follows each product's own convention, exactly as the
backtest did: GloFAS "day n" (leadtime_days = n) describes calendar day
issue+(n-1); Google leadtime_days = valid day - issue day (the API returns
two days of hindsight, leads -2 and -1, which are kept for the chart).

ocha_stratus reads DB credentials at import time, so load_dotenv() runs
first in every entry point that imports this module.
"""

import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from sqlalchemy import text

import ocha_stratus as stratus
from src.constants import ALL_TRIGGER_STATIONS, STATIONS
from src.datasources.glofas import EWDS_URL, _client
from src.monitoring import config as cfg

HERE = Path(__file__).resolve().parent
CELLS = json.loads((HERE / "glofas_cells.json").read_text())["cells"]
GOOGLE_API = "https://floodforecasting.googleapis.com/v1"
GOOGLE_TIMEOUT = 90

# One box covering the seven cells (N, W, S, E); EWDS cost depends on the
# time dimensions, not the area, so one request serves every station.
FORECAST_AREA = [4.9, 41.9, 1.1, 45.7]


def glofas_raw_blob(monitoring_date):
    return f"{cfg.PROJECT_PREFIX}/raw/glofas/monitoring/glofas_forecast_{monitoring_date}.grib"


def google_raw_blob(monitoring_date):
    return f"{cfg.PROJECT_PREFIX}/raw/google/monitoring/google_forecast_{monitoring_date}.json"


def forecasts_blob(monitoring_date):
    """Processed rows of both sources for one monitoring day (same columns as the DB table)."""
    return f"{cfg.PROJECT_PREFIX}/monitoring/forecasts/{monitoring_date}.parquet"


def status_blob(monitoring_date):
    """The evaluation result (status.json content) for one monitoring day."""
    return f"{cfg.PROJECT_PREFIX}/monitoring/status/{monitoring_date}.json"


def _container(write=False):
    return stratus.get_container_client("projects", cfg.BLOB_STAGE, write=write)


# ----------------------------------------------------------------- GloFAS
def glofas_request(issue_date):
    return {
        "variable": "river_discharge_in_the_last_24_hours",
        "system_version": ["operational"],
        "hydrological_model": ["lisflood"],
        "product_type": ["ensemble_perturbed_forecasts"],
        "year": [f"{issue_date:%Y}"],
        "month": [f"{issue_date:%m}"],
        "day": [f"{issue_date:%d}"],
        "leadtime_hour": [str(24 * d) for d in cfg.GLOFAS_LEADS],
        "area": FORECAST_AREA,
        "data_format": "grib2",
        "download_format": "unarchived",
    }


def _raw_in_blob(issue_date, out_path):
    """Download today's raw GRIB from blob if an earlier run already fetched it."""
    from azure.core.exceptions import ResourceNotFoundError

    try:
        data = _container().get_blob_client(glofas_raw_blob(issue_date)).download_blob().readall()
    except ResourceNotFoundError:
        return False
    Path(out_path).write_bytes(data)
    Path(out_path).with_suffix(".fromblob").touch()
    return True


def download_glofas(issue_date, out_path):
    """Download one day's operational ensemble to out_path (GRIB2)."""
    client = _client(wait_until_complete=True)
    client.retrieve("cems-glofas-forecast", glofas_request(issue_date), str(out_path))
    return Path(out_path)


def grib_process_ids(path):
    """Process identifiers stamped in the first GRIB message (version fingerprint)."""
    import eccodes

    with open(path, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        try:
            return {k: int(eccodes.codes_get(gid, k)) for k in cfg.GLOFAS_EXPECTED_PROCESS}
        finally:
            eccodes.codes_release(gid)


def process_glofas(path, monitoring_date):
    """GRIB -> long frame: one row per (station, valid day), ensemble stats."""
    ds = xr.open_dataset(path, engine="cfgrib")
    var = next(v for v in ds.data_vars if "dis" in v.lower())
    da = ds[var]
    if "number" not in da.dims:
        da = da.expand_dims("number")
    keys = list(CELLS)
    lats = xr.DataArray([CELLS[k]["lat"] for k in keys], dims="station", coords={"station": keys})
    lons = xr.DataArray([CELLS[k]["lon"] for k in keys], dims="station", coords={"station": keys})
    pts = da.sel(latitude=lats, longitude=lons, method="nearest")
    # guard: the nearest cell must be the frozen cell, not a box-edge snap
    for k in keys:
        got = (float(pts.sel(station=k).latitude), float(pts.sel(station=k).longitude))
        if abs(got[0] - CELLS[k]["lat"]) > 0.03 or abs(got[1] - CELLS[k]["lon"]) > 0.03:
            raise ValueError(f"{k}: nearest GloFAS cell {got} is not the frozen cell {CELLS[k]}")
    df = pts.to_dataframe(name="value").reset_index()
    issued = pd.Timestamp(df["time"].iloc[0])
    df["lead_hours"] = (df["step"] / pd.Timedelta(hours=1)).astype(int)
    df["leadtime_days"] = df["lead_hours"] // 24
    # end-of-period stamping: day n describes calendar day issue + (n - 1)
    df["valid_date"] = (issued + pd.to_timedelta(df["lead_hours"], unit="h")
                        - pd.Timedelta(days=1)).dt.date
    g = df.groupby(["station", "valid_date", "leadtime_days"])["value"]
    out = g.agg(value="median", value_min="min", value_max="max", n_members="count",
                value_p25=lambda x: x.quantile(0.25), value_p75=lambda x: x.quantile(0.75)).reset_index()
    if (out["n_members"] < cfg.GLOFAS_ENSEMBLE_MEMBERS_MIN).any():
        raise ValueError(f"truncated ensemble: min members {out['n_members'].min()}")
    out["source"] = "glofas"
    out["river"] = out["station"].map(lambda s: STATIONS[s].river)
    out["issued_time"] = issued.tz_localize("UTC") if issued.tzinfo is None else issued
    out["monitoring_date"] = monitoring_date
    ds.close()
    return out


def check_glofas_version():
    """Has EWDS moved `operational` to a new system version?

    Signal: a version_4_x entry under the forecast dataset's Legacy Versions
    means v4 has been retired from the operational slot (v2.1 and v3.1 sit
    there today). Returns the legacy list so the caller can decide.
    """
    client = _client(wait_until_complete=False)
    form = client.client.get_collection("cems-glofas-forecast").form
    legacy = []
    for w in form:
        if w.get("name") == "system_version":
            for grp in w.get("details", {}).get("groups", []):
                if grp.get("label", "").lower().startswith("legacy"):
                    legacy = list(grp.get("values", []))
    v4_retired = any(v.startswith("version_4") for v in legacy)
    return {"legacy_versions": legacy, "v4_retired": v4_retired}


def fetch_glofas(monitoring_date, max_days_back=1, keep_raw=True):
    """Download today's operational ensemble (falling back to yesterday's issue).

    Returns (frame, meta). meta records which issue was used, the GRIB
    process ids and whether they match the expected operational system.
    """
    tmp = Path(tempfile.mkdtemp())
    last_err = None
    for back in range(0, max_days_back + 1):
        issue = monitoring_date - timedelta(days=back)
        path = tmp / f"glofas_{issue}.grib"
        try:
            if _raw_in_blob(issue, path):
                print(f"  GloFAS issue {issue}: reusing the raw GRIB already in blob")
            else:
                download_glofas(issue, path)
        except Exception as exc:  # EWDS raises generic exceptions on "no data"
            last_err = exc
            print(f"  GloFAS issue {issue} not available: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        ids = grib_process_ids(path)
        df = process_glofas(path, monitoring_date)
        version_ok = ids == cfg.GLOFAS_EXPECTED_PROCESS
        label = f"{cfg.GLOFAS_OPERATIONAL} (gpi{ids['generatingProcessIdentifier']}/bp{ids['backgroundProcess']})"
        df["model_version"] = label
        if keep_raw and not path.with_suffix(".fromblob").exists():
            _container(write=True).upload_blob(glofas_raw_blob(issue), path.read_bytes(), overwrite=True)
        return df, {"issue_date": str(issue), "days_back": back, "process_ids": ids,
                    "version_label": label, "version_ok": version_ok}
    raise RuntimeError(f"no GloFAS operational forecast within {max_days_back} days: {last_err}")


# ----------------------------------------------------------------- Google
def _google_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return key


def _google_query(gauge_ids):
    """One queryGaugeForecasts call; errors never echo the key (it sits in the URL)."""
    params = [("gaugeIds", g) for g in gauge_ids] + [("key", _google_key())]
    r = requests.get(f"{GOOGLE_API}/gauges:queryGaugeForecasts", params=params, timeout=GOOGLE_TIMEOUT)
    if not r.ok:
        raise requests.HTTPError(f"Google Flood Forecasting API {r.status_code} for {gauge_ids}: "
                                 f"{r.text[:200]}")
    return r.json().get("forecasts", {})


def fetch_google(monitoring_date, stations=ALL_TRIGGER_STATIONS, keep_raw=True):
    """Latest Google Flood Hub forecast per trigger gauge.

    One batched call; if the API rejects the batch (it answers 404 for the
    whole request when any id is unknown, as for Dollow), fall back to one
    call per gauge and skip the ids it does not serve.
    """
    by_gauge = {STATIONS[s].grrr_gauge_id: s for s in stations}
    # gauges known to be absent from the live API are probed one by one so the
    # batch does not fail on them, and so they are picked up if they appear
    known_absent = [STATIONS[s].grrr_gauge_id for s in stations if s in cfg.GOOGLE_NOT_SERVED]
    gauge_ids = [g for g in by_gauge if g not in known_absent]
    not_served = []
    for g in known_absent:
        try:
            forecasts_extra = _google_query([g])
        except requests.HTTPError as exc1:
            if " 404 " in str(exc1):
                not_served.append(by_gauge[g]); forecasts_extra = {}
            else:
                raise
    try:
        forecasts = _google_query(gauge_ids)
        for g in known_absent:
            if by_gauge[g] not in not_served:
                forecasts.update(_google_query([g]))
    except requests.HTTPError as exc:
        print(f"  batched Google query failed ({str(exc)[:80]}); querying gauges one by one")
        forecasts, not_served = {}, []
        for g in gauge_ids:
            try:
                forecasts.update(_google_query([g]))
            except requests.HTTPError as exc1:
                if " 404 " in str(exc1):
                    not_served.append(by_gauge[g])
                else:
                    raise
    if keep_raw and forecasts:
        _container(write=True).upload_blob(google_raw_blob(monitoring_date),
                                           json.dumps(forecasts).encode(), overwrite=True)
    rows = []
    for gid, block in forecasts.items():
        issues = block.get("forecasts", [])
        if not issues:
            continue
        latest = max(issues, key=lambda x: x["issuedTime"])
        issued = pd.Timestamp(latest["issuedTime"])
        for rg in latest["forecastRanges"]:
            valid = pd.Timestamp(rg["forecastStartTime"]).date()
            rows.append({
                "station": by_gauge[gid], "river": STATIONS[by_gauge[gid]].river,
                "source": "google", "monitoring_date": monitoring_date,
                "issued_time": issued, "valid_date": valid,
                "leadtime_days": (valid - issued.date()).days,
                "value": float(rg["value"]), "value_min": None, "value_p25": None,
                "value_p75": None, "value_max": None, "n_members": 1,
                "model_version": "google_grrr (Flood Hub API)",
            })
    df = pd.DataFrame(rows)
    missing = sorted(set(stations) - set(df.station)) if len(df) else list(stations)
    return df, {"n_gauges": int(df.station.nunique()) if len(df) else 0, "missing": missing,
                "not_served": not_served}


# ----------------------------------------------------------------- storage
COLUMNS = ["monitoring_date", "source", "station", "valid_date", "river", "issued_time",
           "leadtime_days", "value", "value_min", "value_p25", "value_p75", "value_max",
           "n_members", "model_version"]


def archive_day(df, result, monitoring_date):
    """Keep the day's processed forecast rows and evaluation on blob, next to the raw
    GRIB, the raw Google answer and the chart, so a day can be re-read without the DB
    or the monitoring-status git branch."""
    import io

    buf = io.BytesIO()
    out = df[COLUMNS].copy()
    out["issued_time"] = pd.to_datetime(out["issued_time"], utc=True)
    out.to_parquet(buf, index=False)
    c = _container(write=True)
    c.upload_blob(forecasts_blob(monitoring_date), buf.getvalue(), overwrite=True)
    c.upload_blob(status_blob(monitoring_date), json.dumps(result, indent=1, default=str).encode(), overwrite=True)
    return forecasts_blob(monitoring_date), status_blob(monitoring_date)


def upsert(df):
    df = df[COLUMNS].copy()
    df["issued_time"] = pd.to_datetime(df["issued_time"], utc=True)
    engine = stratus.get_engine(stage=cfg.BLOB_STAGE, write=True)
    df.to_sql(cfg.DB_TABLE, schema=cfg.DB_SCHEMA, con=engine, if_exists="append",
              index=False, method=stratus.postgres_upsert)
    return len(df)


def load_day(monitoring_date):
    engine = stratus.get_engine(stage=cfg.BLOB_STAGE)
    with engine.connect() as con:
        df = pd.read_sql(text(f"select * from {cfg.DB_SCHEMA}.{cfg.DB_TABLE} "
                              "where monitoring_date = :d order by source, station, valid_date"),
                         con, params={"d": monitoring_date})
    if df.empty:
        raise LookupError(f"no monitoring rows for {monitoring_date}")
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["monitoring_date"] = pd.to_datetime(df["monitoring_date"])
    return df


def latest_monitoring_date():
    engine = stratus.get_engine(stage=cfg.BLOB_STAGE)
    with engine.connect() as con:
        d = con.execute(text(f"select max(monitoring_date) from {cfg.DB_SCHEMA}.{cfg.DB_TABLE}")).scalar()
    return d


def monitoring_date_from_env():
    raw = os.getenv("MONITORING_DATE", "").strip()
    return date.fromisoformat(raw) if raw else datetime.now(timezone.utc).date()
