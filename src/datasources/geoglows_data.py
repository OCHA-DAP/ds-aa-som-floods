"""GEOGloWS v2 (ECMWF streamflow) data access for the Somalia stations.

Retrospective simulation (1940-present) and return periods are read straight
from the public S3 zarr stores with anonymous access, pulling all stations'
rivers from one open — the geoglows package's per-river helpers re-open the
store on every call, and its latlon_to_river lookup is broken against the
current metadata table schema, so river IDs were resolved offline (nearest
TDX-Hydro reach with upstream area >= 50,000 km2; see src.constants.Station).

Forecasts (15-day, 52-member ensemble) come from the REST API via the
geoglows package, per river.

The module reads zarr directly (zarr.open_group) rather than through
xarray/dask: on this machine xr.open_zarr with chunks="auto" stalls on these
million-river stores.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "geoglows"

S3_STORAGE_OPTIONS = {"anon": True, "client_kwargs": {"region_name": "us-west-2"}}
RETRO_DAILY_ZARR = "s3://geoglows-v2/retrospective/daily.zarr"
RETURN_PERIODS_ZARR = "s3://geoglows-v2/retrospective/return-periods.zarr"


def _open(url):
    # imported lazily: zarr (+ s3fs) is only needed for live store reads,
    # not for loading already-downloaded data
    import zarr

    return zarr.open_group(url, mode="r", storage_options=S3_STORAGE_OPTIONS)


def _river_indices(group, river_ids):
    """Positions of each river_id in the store's river_id coordinate."""
    store_ids = group["river_id"][:]
    idx = {rid: pos for pos, rid in enumerate(store_ids) if rid in set(river_ids)}
    missing = [r for r in river_ids if r not in idx]
    if missing:
        raise KeyError(f"river_ids not in {group.store}: {missing}")
    return idx


def download_retro_daily(river_ids):
    """Fetch daily retrospective discharge for many rivers in one pass.

    Returns a DataFrame indexed by date with one column per river_id (m3/s).
    """
    g = _open(RETRO_DAILY_ZARR)
    idx = _river_indices(g, river_ids)
    time = pd.to_datetime(g["time"][:], unit="s", origin="1940-01-01")
    q = g["Q"]
    data = {}
    for rid in river_ids:
        data[rid] = q[:, idx[rid]]
        print(f"  retro {rid}: {len(time)} days read")
    return pd.DataFrame(data, index=pd.Index(time, name="time"))


def download_return_periods(river_ids):
    """Fetch Gumbel return-period discharge for many rivers in one pass.

    Returns a DataFrame indexed by return period (years), one column per
    river_id (m3/s, from the daily-max Gumbel fit).
    """
    g = _open(RETURN_PERIODS_ZARR)
    idx = _river_indices(g, river_ids)
    rps = g["return_period"][:]
    gumbel = g["gumbel_daily"]
    data = {rid: gumbel[:, idx[rid]] for rid in river_ids}
    return pd.DataFrame(
        data, index=pd.Index(np.asarray(rps), name="return_period")
    )


def download_forecast_stats(river_id, date=None):
    """Latest (or dated) 15-day forecast stats for one river via REST."""
    # imported lazily: the geoglows package is only needed for live REST
    # downloads, and is not part of the project dependencies
    import geoglows.data

    kwargs = {"river_id": river_id}
    if date:
        kwargs["date"] = date
    return geoglows.data.forecast_stats(**kwargs)


def download_forecast_ensembles(river_id, date=None):
    """Latest (or dated) 52-member forecast ensemble for one river via REST."""
    import geoglows.data

    kwargs = {"river_id": river_id}
    if date:
        kwargs["date"] = date
    return geoglows.data.forecast_ensembles(**kwargs)


FORECAST_ARCHIVE_BUCKET = "geoglows-v2-forecasts"
FORECAST_ARCHIVE_DIR = DATA_DIR / "forecast_archive"


def list_forecast_archive_dates():
    """Archived forecast initialisation dates (YYYYMMDDHH), oldest first.

    GEOGloWS keeps every daily 52-member init as its own zarr; the archive
    currently starts 2024-07-01. This is the only route to GEOGloWS
    lead-time skill — there is no multi-decade reforecast like GloFAS's.
    """
    import s3fs

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
    return sorted(
        x.split("/")[-1].replace(".zarr", "")
        for x in fs.ls(FORECAST_ARCHIVE_BUCKET)
        if x.endswith(".zarr")
    )


def _river_positions(group, river_ids):
    rivid = group["rivid"][:]
    pos = {int(r): int(np.where(rivid == r)[0][0]) for r in river_ids}
    return pos


def download_forecast_archive_date(init, river_ids):
    """One archived init -> daily-mean ensemble forecast for the given rivers.

    Returns a long DataFrame: river_id, member, issued_time, valid_time,
    leadtime_days, discharge. Sub-daily steps are averaged to daily means.
    The 13 Somalia rivers span only 2 of the store's 9,969 river chunks, so
    a contiguous slice across them is far cheaper than 13 separate reads.
    """
    g = zarr.open_group(
        f"s3://{FORECAST_ARCHIVE_BUCKET}/{init}.zarr", mode="r",
        storage_options=S3_STORAGE_OPTIONS,
    )
    pos = _river_positions(g, river_ids)
    lo, hi = min(pos.values()), max(pos.values())
    block = g["Qout"][:, :, lo:hi + 1]          # (member, time, river)
    members = g["ensemble"][:]
    units = dict(g["time"].attrs).get("units", "")
    origin = units.split("since", 1)[1].strip() if "since" in units else init[:8]
    times = pd.to_datetime(g["time"][:], unit="s", origin=pd.Timestamp(origin))

    issued = pd.Timestamp(init[:8])
    frames = []
    for rid, p in pos.items():
        df = pd.DataFrame(block[:, :, p - lo].T, index=times, columns=members)
        daily = df.groupby(df.index.normalize()).mean()
        long = (
            daily.rename_axis("valid_time")
            .melt(ignore_index=False, var_name="member", value_name="discharge")
            .reset_index()
        )
        long["river_id"] = rid
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out["issued_time"] = issued
    out["leadtime_days"] = (out["valid_time"] - issued).dt.days
    return out[out["leadtime_days"] >= 0].reset_index(drop=True)


def load_retro_daily(station_key):
    """Load one station's saved retrospective discharge as a pandas Series."""
    df = pd.read_parquet(DATA_DIR / f"retro_daily_{station_key}.parquet")
    series = df.iloc[:, 0]
    series.name = "discharge"
    return series
