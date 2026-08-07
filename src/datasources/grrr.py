"""Google GRRR (flood-forecasting model) data access for the Somalia stations.

Reads the public model-output zarrs on gs://flood-forecasting (anonymous
access, CC-BY-4.0) for model model_id_8583a5c2_v0:

- reanalysis/streamflow.zarr  — daily discharge, 1980 to end-2023
- reforecast/streamflow.zarr  — daily issues 2016-2023, lead times 0-7 days
- return_periods.zarr         — per-gauge discharge at RP 2-200 years

Gauge IDs are "hybas_" + the HydroBASINS Africa level-12 basin containing
each station (see src.constants.Station); all 13 were verified present in
the reanalysis store.

Like geoglows_data, this reads zarr directly rather than via xarray/dask,
which stalls on the ~1M-gauge stores here.
"""

import os
from pathlib import Path

# Silence gRPC/abseil log spam from google-cloud libs; must be set before
# any grpc/google import happens.
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GLOG_minloglevel", "3")

import pandas as pd
import zarr

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "google"

BASE = "gs://flood-forecasting/hydrologic_predictions/model_id_8583a5c2_v0"
REANALYSIS_ZARR = f"{BASE}/reanalysis/streamflow.zarr"
REFORECAST_ZARR = f"{BASE}/reforecast/streamflow.zarr"
RETURN_PERIODS_ZARR = f"{BASE}/return_periods.zarr"


def _open(url):
    return zarr.open_group(url, mode="r", storage_options={"token": "anon"})


def _gauge_indices(group, gauge_ids):
    # str() per element: gauge_id dtype varies by store (fixed-width U16 vs
    # numpy 2 StringDType), and astype(str) fails on the latter
    wanted = set(gauge_ids)
    idx = {
        str(gid): pos
        for pos, gid in enumerate(group["gauge_id"][:])
        if str(gid) in wanted
    }
    missing = [g for g in gauge_ids if g not in idx]
    if missing:
        raise KeyError(f"gauge_ids not found in store: {missing}")
    return idx


def download_reanalysis(gauge_ids):
    """Fetch daily reanalysis discharge for many gauges in one pass.

    Returns a DataFrame indexed by date with one column per gauge_id (m3/s).
    """
    g = _open(REANALYSIS_ZARR)
    idx = _gauge_indices(g, gauge_ids)
    time = pd.to_datetime(g["time"][:], unit="D", origin="1980-01-01")
    sf = g["streamflow"]
    data = {}
    for gid in gauge_ids:
        data[gid] = sf[idx[gid], :]
        print(f"  reanalysis {gid}: {len(time)} days read")
    return pd.DataFrame(data, index=pd.Index(time, name="time"))


def download_reforecast(gauge_id):
    """Fetch the full reforecast for one gauge (daily issues 2016-2023).

    Returns a long DataFrame with issue_time, leadtime (days), valid_time,
    and streamflow columns.
    """
    g = _open(REFORECAST_ZARR)
    idx = _gauge_indices(g, [gauge_id])
    issue = pd.to_datetime(g["issue_time"][:], unit="D", origin="2016-01-01")
    leads = g["lead_time"][:]
    arr = g["streamflow"][idx[gauge_id], :, :]  # (issue_time, lead_time)
    df = pd.DataFrame(arr, index=issue, columns=leads)
    df = (
        df.rename_axis(index="issue_time", columns="leadtime")
        .stack()
        .rename("streamflow")
        .reset_index()
    )
    df["leadtime"] = df["leadtime"].astype(int)
    df["valid_time"] = df["issue_time"] + pd.to_timedelta(df["leadtime"], unit="D")
    return df


def download_return_periods(gauge_ids):
    """Fetch return-period discharge for many gauges in one pass.

    Returns a DataFrame indexed by return period (years), one column per
    gauge_id (m3/s).
    """
    g = _open(RETURN_PERIODS_ZARR)
    idx = _gauge_indices(g, gauge_ids)
    rps = sorted(
        int(k.split("_")[-1]) for k in g.array_keys() if k.startswith("return_period_")
    )
    data = {
        gid: [float(g[f"return_period_{rp}"][idx[gid]]) for rp in rps]
        for gid in gauge_ids
    }
    return pd.DataFrame(data, index=pd.Index(rps, name="return_period"))


def load_reanalysis(station_key):
    """Load one station's saved reanalysis discharge as a pandas Series."""
    df = pd.read_parquet(DATA_DIR / f"reanalysis_{station_key}.parquet")
    series = df.iloc[:, 0]
    series.name = "discharge"
    return series
