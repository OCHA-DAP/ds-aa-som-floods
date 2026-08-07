"""Build tidy analysis-ready tables from the raw multi-source downloads.

Files are written per data source, so provenance is visible in the name
(matching the swalim_* convention). Written to data/processed/ and mirrored
to blob under ds-aa-som-floods/processed/:

Daily discharge (station, river, source, date, discharge in m3/s):
- discharge_daily_geoglows.parquet    GEOGloWS v2 retrospective, 1940-present
- discharge_daily_google_grrr.parquet Google GRRR reanalysis, 1980-2023
- discharge_daily_glofas_v4.parquet   GloFAS v4 reanalysis, 1999-2023
- discharge_daily_glofas_v5.parquet   GloFAS v5 reanalysis, 1999-2023

Return periods (station, river, source, return_period, discharge):
- return_periods_geoglows.parquet     daily-max Gumbel fit, RP2-100
- return_periods_google_grrr.parquet  GRRR RP2-200
  (EWDS publishes none per cell for GloFAS — fit from its reanalysis.)

Forecasts (station, river, source, issued_time, valid_time, leadtime_days,
member, discharge):
- reforecast_geoglows.parquet     archived daily 52-member inits, 2024-07
      onwards, sub-daily steps averaged to daily, leads 0-7
- reforecast_google_grrr.parquet  deterministic GRRR, 2016-2023, leads 0-7
- reforecast_glofas_v4.parquet    control + 10 perturbed members,
      2003-2023 Gu/Deyr, GloFAS "day 1-7" (see note below)

Observations:
- swalim_levels.parquet     station, date, level_m (capped at Bank Full —
      plateaus mean ">=")
- swalim_thresholds.parquet station, moderate_flood_risk, high_flood_risk,
      bank_full, max_level (m)

GloFAS lead convention: its 24-hour means are stamped at period END, so
"day 1" (lead_hours=24) describes the calendar day of issue. valid_time in
the table is already shifted to the day each value describes, so
valid_time = issued_time + leadtime_days - 1 for GloFAS, while GEOGloWS and
GRRR use valid_time = issued_time + leadtime_days. Score on valid_time.

Every builder includes whatever raw data currently exists and skips
missing station/source combos, so processing can re-run as downloads land.
"""

import re

import pandas as pd

from src.constants import STATIONS, SWALIM_CODE_TO_KEY
from src.datasources import geoglows_data, glofas, grrr, swalim

DATA_DIR = glofas.DATA_DIR.parent  # repo data/
PROCESSED_DIR = DATA_DIR / "processed"

# SWALIM station numbers -> station registry keys (Balcad, Saakow and
# Jilib have no gauge in the SNRFA threshold table)
# SWALIM_CODE_TO_KEY now lives in src.constants, derived from the registry's
# swalim_code fields, so gauge links can't drift from the station definitions


def _long(series, key, source):
    """One station-source daily series -> tidy long frame."""
    df = series.rename("discharge").rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["station"] = key
    df["river"] = STATIONS[key].river
    df["source"] = source
    return df[["station", "river", "source", "date", "discharge"]]


def build_discharge_daily():
    frames = []
    for key, st in STATIONS.items():
        # GEOGloWS retrospective
        try:
            frames.append(_long(geoglows_data.load_retro_daily(key), key, "geoglows"))
        except FileNotFoundError:
            print(f"  {key}: no geoglows retro yet, skipping")
        # Google GRRR reanalysis
        try:
            frames.append(_long(grrr.load_reanalysis(key), key, "google"))
        except FileNotFoundError:
            print(f"  {key}: no google reanalysis yet, skipping")
        # GloFAS reanalysis, one source per system version (v4.0 is "glofas";
        # v5.0 is downloaded alongside for comparison — see glofas.VERSION_DIRS)
        for version, label in [("version_4_0", "glofas"), ("version_5_0", "glofas_v5")]:
            try:
                series = glofas.load_reanalysis(key, version=version)
            except (OSError, StopIteration, FileNotFoundError, KeyError):
                print(f"  {key}: no {label} reanalysis yet, skipping")
                continue
            if len(series):
                frames.append(_long(series, key, label))
    return pd.concat(frames, ignore_index=True)


def build_return_periods():
    frames = []
    for key, st in STATIONS.items():
        for source, path in [
            ("geoglows", geoglows_data.DATA_DIR / f"return_periods_{key}.parquet"),
            ("google", grrr.DATA_DIR / f"return_periods_{key}.parquet"),
        ]:
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            out = df.iloc[:, 0].rename("discharge").rename_axis("return_period").reset_index()
            out["station"] = key
            out["river"] = st.river
            out["source"] = source
            frames.append(out[["station", "river", "source", "return_period", "discharge"]])
    return pd.concat(frames, ignore_index=True)


MAX_LEADTIME_DAYS = 7  # trigger horizon; raw archive keeps leads out to 14

# `source` values stay short and stable (analysis code groups on them); the
# filename labels spell out the product so blob listings are self-describing
SOURCE_FILE_LABELS = {
    "geoglows": "geoglows",
    "google": "google_grrr",
    "glofas": "glofas_v4",
    "glofas_v5": "glofas_v5",
}


def load_processed(kind, source=None):
    """Read a processed table back, concatenating all sources by default.

    load_processed("discharge_daily")            -> every source
    load_processed("discharge_daily", "glofas")   -> just GloFAS v4
    """
    labels = (
        [SOURCE_FILE_LABELS[source]] if source else list(SOURCE_FILE_LABELS.values())
    )
    frames = []
    for label in labels:
        path = PROCESSED_DIR / f"{kind}_{label}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        # single-source tables (swalim_*) carry no source suffix
        return pd.read_parquet(PROCESSED_DIR / f"{kind}.parquet")
    return pd.concat(frames, ignore_index=True)


def _geoglows_forecast_frames(key):
    """Every downloaded geoglows ensemble file for one station -> long daily frames."""
    frames = []
    for path in sorted(geoglows_data.DATA_DIR.glob(f"forecast_ensembles_{key}_*.parquet")):
        issued = pd.Timestamp(re.search(r"(\d{4}-\d{2}-\d{2})", path.stem).group(1))
        ens = pd.read_parquet(path)
        idx = pd.to_datetime(ens.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        daily = ens.groupby(idx.normalize()).mean()
        df = (
            daily.rename_axis("valid_time")
            .melt(ignore_index=False, var_name="member", value_name="discharge")
            .reset_index()
        )
        df["member"] = df["member"].str.replace("ensemble_", "").astype(int)
        df["issued_time"] = issued
        df["leadtime_days"] = (df["valid_time"] - issued).dt.days
        frames.append(df)
    return frames


def _geoglows_archive_frame():
    """The archived daily GEOGloWS inits (2024-07 onwards) as one long frame.

    One parquet per init, keyed by river_id; mapped back to station keys and
    trimmed to the trigger horizon (the raw files keep leads 0-14).
    """
    files = sorted(geoglows_data.FORECAST_ARCHIVE_DIR.glob("*.parquet"))
    if not files:
        return None
    # one reach can serve several stations (TDX reach 110794493 is 115 km long
    # and carries jowhar, shabelle_04 AND mahadey_weyne) — a plain dict keeps
    # only one of them, silently dropping the rest, so expand one-to-many
    reach_map = pd.DataFrame(
        [(st.geoglows_river_id, key, st.river) for key, st in STATIONS.items()],
        columns=["river_id", "station", "river"],
    )
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df[df["leadtime_days"] <= MAX_LEADTIME_DAYS]
    # drop reaches that no longer belong to any registry station (IDs change
    # if a station's coordinates move; mislabeling would be worse than a gap)
    unknown = set(df["river_id"].unique()) - set(reach_map["river_id"])
    if unknown:
        print(f"  geoglows archive: dropping {len(unknown)} river_id(s) not in the registry")
        df = df[~df["river_id"].isin(unknown)]
    if df.empty:
        return None
    df = df.merge(reach_map, on="river_id")
    df["source"] = "geoglows"
    return df.drop(columns="river_id")


def build_reforecast():
    frames = []
    archive = _geoglows_archive_frame()
    if archive is not None:
        frames.append(archive)
        print(f"  geoglows archive: {len(archive):,} rows")
    for key, st in STATIONS.items():
        for df in _geoglows_forecast_frames(key):
            df["station"] = key
            df["river"] = st.river
            df["source"] = "geoglows"
            frames.append(df)
        grrr_path = grrr.DATA_DIR / f"reforecast_{key}.parquet"
        if grrr_path.exists():
            df = pd.read_parquet(grrr_path).rename(
                columns={"streamflow": "discharge", "leadtime": "leadtime_days", "issue_time": "issued_time"}
            )
            df["member"] = 0
            df["station"] = key
            df["river"] = st.river
            df["source"] = "google"
            frames.append(df)

    # GloFAS reforecast: control + 10 perturbed, leads 1-7 (its "day 1" is the
    # issue day — valid_day already carries the day each value describes)
    gf = glofas.load_reforecast_box()
    if gf is not None:
        gf = gf.rename(columns={"valid_day": "valid_time"})
        gf["river"] = gf["station"].map(lambda k: STATIONS[k].river)
        gf["source"] = "glofas"
        frames.append(gf.drop(columns="lead_hours"))
        print(f"  glofas reforecast: {len(gf):,} rows")

    cols = ["station", "river", "source", "issued_time", "valid_time", "leadtime_days", "member", "discharge"]
    return pd.concat(frames, ignore_index=True)[cols]


def build_swalim_levels():
    frames = []
    for blob_name in swalim.list_station_blobs():
        df = swalim.load_station_levels(blob_name)
        if not len(df):
            continue
        code = df["station_number"].iloc[0]
        key = SWALIM_CODE_TO_KEY.get(code)
        if key is None:
            continue
        out = df["level(m)"].rename("level_m").rename_axis("date").reset_index()
        out["station"] = key
        frames.append(out[["station", "date", "level_m"]])
    return pd.concat(frames, ignore_index=True)


def build_swalim_thresholds():
    th = swalim.load_thresholds()
    th = th[th.index.isin(SWALIM_CODE_TO_KEY)].copy()
    th["station"] = th.index.map(SWALIM_CODE_TO_KEY)
    out = th.rename(
        columns={
            "Moderate Flood Risk": "moderate_flood_risk",
            "High Flood Risk": "high_flood_risk",
            "Bank Full": "bank_full",
            "Maximum Depth": "max_level",
        }
    )[["station", "moderate_flood_risk", "high_flood_risk", "bank_full", "max_level"]]
    return out.reset_index(drop=True)
