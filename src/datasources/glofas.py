"""Download and load GloFAS reanalysis and reforecast river discharge.

GloFAS moved off the main Climate Data Store onto the Early Warning Data
Store (EWDS) — same account/API key as CDS, different host and (for the
reforecast) a different product_type spelling than older tooling expects
(`ensemble_perturbed_reforecast`, singular).

EWDS enforces a per-request "cost" limit tighter than the old CDS: reanalysis
must be chunked by year (a 10-year request is already rejected), and
reforecast (which multiplies by ensemble members and lead times) only fits
~8 lead times per station-month. Both download functions submit all needed
chunks asynchronously up front, then poll — mirroring the pattern used in
the org's pa-aa-toolbox — rather than blocking on one request at a time.
"""

import json
import os
import time
import zipfile
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr

from src.constants import STATIONS as _REGISTRY

EWDS_URL = "https://ewds.climate.copernicus.eu/api"

# SOM_DATA_DIR overrides the repo-local data root on ephemeral runners
# (Databricks pulls the code via git_source into a read-only checkout)
DATA_DIR = (
    Path(os.getenv("SOM_DATA_DIR") or Path(__file__).resolve().parents[2] / "data")
    / "glofas"
)


def _snap(v):
    """Snap a coordinate to the nearest GloFAS v4 0.05deg cell center (x.x25/x.x75)."""
    return round((v - 0.025) / 0.05) * 0.05 + 0.025


# GloFAS-grid-snapped coordinates (0.05deg resolution, v4 grid).
# luuq and belet_weyne keep their previously verified river cells (belet_weyne
# sits one cell north of the plain snap of the EF5 coordinate); the rest are
# plain snaps — the download box (+/-0.1deg) covers neighboring cells either
# way, so the exact river cell can still be picked at load time.
_VERIFIED = {
    "luuq": {"lat": 3.725, "lon": 42.525},
    "belet_weyne": {"lat": 4.725, "lon": 45.225},
}
STATIONS = {
    key: _VERIFIED.get(key, {"lat": _snap(st.lat), "lon": _snap(st.lon)})
    for key, st in _REGISTRY.items()
}

REANALYSIS_YEARS = [str(y) for y in range(1999, 2024)]

# One box covering every station (N, W, S, E). EWDS request cost scales with
# time dimensions only (year x month x day), not area, so a whole-domain
# yearly request costs the same as a single-station one (365 vs the 500
# limit) and needs far fewer queue slots.
#
# Extended south to -0.1 on 2026-08-03 to reach the lower-Juba gauges
# (southernmost 42.675, 0.025). The original box stopped at S 0.2, so those
# points silently snapped ~0.2deg north via method="nearest" — hence the
# extended download lives in its own directory: never mix extents in one
# open_mfdataset call.
SOM_AREA = [4.9, 41.9, -0.1, 45.9]
SOM_AREA_LEGACY = [4.9, 41.9, 0.2, 45.9]
ALL_MONTHS = [str(m).zfill(2) for m in range(1, 13)]
ALL_DAYS = [str(d).zfill(2) for d in range(1, 32)]

# Confirmed working ceiling is ~8 leadtime values per station-month request
LEADTIME_DAYS = [1, 3, 7, 11, 15, 21, 30, 42]

POLL_INTERVAL_SECONDS = 60


def _station_area(station_key, buffer=0.1):
    """Small bounding box around one station: [lat_max, lon_min, lat_min, lon_max]."""
    s = STATIONS[station_key]
    return [
        s["lat"] + buffer,
        s["lon"] - buffer,
        s["lat"] - buffer,
        s["lon"] + buffer,
    ]


def _key():
    # Databricks Job Compute injects CDSAPI_KEY from the dsci secret scope
    key = os.getenv("CDSAPI_KEY")
    if key:
        return key
    rc = Path.home() / ".cdsapirc"
    lines = dict(
        line.strip().split(": ", 1) for line in rc.read_text().splitlines() if ":" in line
    )
    return lines["key"]


def _client(wait_until_complete=True):
    return cdsapi.Client(
        url=EWDS_URL, key=_key(), wait_until_complete=wait_until_complete
    )


def _unwrap(raw_path):
    """CDS returns a zip when a request spans >1 output file; unwrap it."""
    raw_path = Path(raw_path)
    if zipfile.is_zipfile(raw_path):
        extract_dir = raw_path.parent / (raw_path.stem + "_extracted")
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(raw_path) as z:
            names = z.namelist()
            z.extractall(extract_dir)
        return [extract_dir / n for n in names]
    return [raw_path]


def _submit_and_download_all(jobs, log_prefix=""):
    """jobs: dict of key -> (collection_id, request, out_path). Blocks until all done.

    Submits every job asynchronously up front, then polls all of them
    together so they process concurrently on the CDS side instead of
    queuing one at a time.
    """
    client = _client(wait_until_complete=False)
    remotes = {}
    for key, (collection_id, request, out_path) in jobs.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"{log_prefix}{key}: already downloaded, skipping")
            continue
        remotes[key] = client.retrieve(collection_id, request)
        print(f"{log_prefix}{key}: submitted")

    out_paths = {key: jobs[key][2] for key in remotes}
    downloaded = {k: p for k, p in ((k, jobs[k][2]) for k in jobs) if p.exists()}
    pending = dict(remotes)
    while pending:
        for key, remote in list(pending.items()):
            remote.update()
            status = remote.status
            if status == "successful":
                remote.download(str(out_paths[key]))
                downloaded[key] = out_paths[key]
                del pending[key]
                print(f"{log_prefix}{key}: downloaded")
            elif status == "failed":
                print(f"{log_prefix}{key}: FAILED - {remote.get_receipt()}")
                del pending[key]
        if pending:
            time.sleep(POLL_INTERVAL_SECONDS)
    return downloaded


def download_reanalysis(station_key, years=None):
    """Download GloFAS reanalysis for one station, chunked by year."""
    years = years or REANALYSIS_YEARS
    raw_dir = DATA_DIR / "raw" / "reanalysis" / station_key
    jobs = {}
    for year in years:
        # EWDS restructured this dataset on 2026-07-30: hyear/hmonth/hday
        # became year/month/day, the variable gained an "average_" prefix,
        # and a timespan field was added. The reforecast dataset (below)
        # still uses the old spelling.
        query = {
            "system_version": "version_4_0",
            "hydrological_model": "lisflood",
            "product_type": "consolidated",
            "variable": "average_river_discharge_in_the_last_24_hours",
            "timespan": "time_mean",
            "year": year,
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": _station_area(station_key),
        }
        jobs[year] = (
            "cems-glofas-historical",
            query,
            raw_dir / f"{year}.nc",
        )
    return _submit_and_download_all(jobs, log_prefix=f"[reanalysis/{station_key}] ")


# EWDS offers v4.0 and v5.0 for the reanalysis, but the reforecast archive
# stops at v4.0 — so v4.0 stays the default (thresholds fitted on the
# reanalysis must come from the same model version as the forecasts they are
# applied to). v5.0 is downloaded alongside for comparison.
DEFAULT_VERSION = "version_4_0"
# extended-box downloads (SOM_AREA); the legacy 0.2-south boxes remain in
# reanalysis_som / reanalysis_som_v5 and are used only as a fallback
VERSION_DIRS = {
    "version_4_0": "reanalysis_som_ext",
    "version_5_0": "reanalysis_som_v5_ext",
}
VERSION_DIRS_LEGACY = {
    "version_4_0": "reanalysis_som",
    "version_5_0": "reanalysis_som_v5",
}


def _box_dir(version=DEFAULT_VERSION):
    return DATA_DIR / "raw" / VERSION_DIRS[version]


def download_reanalysis_box(years=None, version=DEFAULT_VERSION):
    """Download GloFAS reanalysis for the all-stations box, chunked by year.

    Preferred over per-station downloads: same per-request cost, one
    request per year for the whole domain. Files land in
    data/glofas/raw/<VERSION_DIRS[version]>/<year>.nc and are picked up by
    load_reanalysis for every station.
    """
    years = years or REANALYSIS_YEARS
    raw_dir = _box_dir(version)
    jobs = {}
    for year in years:
        query = {
            "system_version": version,
            "hydrological_model": "lisflood",
            "product_type": "consolidated",
            "variable": "average_river_discharge_in_the_last_24_hours",
            "timespan": "time_mean",
            "year": year,
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": SOM_AREA,
        }
        jobs[year] = ("cems-glofas-historical", query, raw_dir / f"{year}.nc")
    return _submit_and_download_all(
        jobs, log_prefix=f"[reanalysis/som-box/{version}] "
    )


SPLIT_PRIORITY = ["hyear", "year", "hmonth", "month", "hday", "day", "leadtime_hour"]
MAX_IN_FLIGHT = 20


def _plan_chunks(coll, request, out_prefix):
    """Bisect a request on time dimensions until each piece fits the live cost limit.

    Returns a flat list of (request, out_prefix) leaves; submits nothing.
    """
    try:
        est = coll.estimate_costs(request)
        cost, limit = est.get("cost"), est.get("limit")
    except Exception as e:
        print(f"  cost estimate failed for {out_prefix} ({e}); using as-is")
        return [(request, out_prefix)]
    if cost is not None and limit is not None and cost > limit:
        splittable = [
            k for k in SPLIT_PRIORITY
            if isinstance(request.get(k), list) and len(request[k]) > 1
        ]
        if not splittable:
            print(f"  {out_prefix}: cost {cost} > limit {limit}, cannot split; skipping")
            return []
        key = splittable[0]
        vals = request[key]
        mid = len(vals) // 2
        chunks = []
        for i, part in enumerate((vals[:mid], vals[mid:])):
            chunks += _plan_chunks(coll, {**request, key: part}, f"{out_prefix}_{key}{i}")
        return chunks
    return [(request, out_prefix)]


def _submit_rolling(client, jobs, log_prefix="", skip=None, on_downloaded=None):
    """Run jobs through EWDS with a rolling window of in-flight requests.

    jobs: list of (collection_id, request, out_path). Existing out_paths are
    skipped, so this is resumable. Keeping only MAX_IN_FLIGHT submitted at a
    time avoids flooding the shared per-user queue while still overlapping
    enough requests to saturate the processing slots.

    skip: optional predicate on out_path marking a chunk already done
    elsewhere (e.g. present in blob when local disk is ephemeral).
    on_downloaded: optional callback on out_path after each finished chunk
    (e.g. upload to blob as the run goes, so partial runs persist).
    """
    pending = []
    for collection_id, request, out_path in jobs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() or (skip and skip(out_path)):
            print(f"{log_prefix}{out_path.stem}: already downloaded, skipping")
            continue
        pending.append((collection_id, request, out_path))

    total = len(pending)
    done = 0
    failed = []
    in_flight = {}
    while pending or in_flight:
        while pending and len(in_flight) < MAX_IN_FLIGHT:
            collection_id, request, out_path = pending.pop(0)
            try:
                in_flight[out_path] = client.retrieve(collection_id, request)
                print(f"{log_prefix}{out_path.stem}: submitted ({len(pending)} queued locally)")
            except Exception as e:
                print(f"{log_prefix}{out_path.stem}: submission FAILED - {e}")
                failed.append(out_path.stem)
        for out_path, remote in list(in_flight.items()):
            try:
                remote.update()
                status = remote.status
            except Exception as e:
                print(f"{log_prefix}{out_path.stem}: poll failed ({e}); retrying")
                continue
            if status == "successful":
                remote.download(str(out_path))
                del in_flight[out_path]
                done += 1
                print(f"{log_prefix}{out_path.stem}: downloaded [{done}/{total}]")
                if on_downloaded:
                    on_downloaded(out_path)
            elif status == "failed":
                print(f"{log_prefix}{out_path.stem}: FAILED - {remote.get_receipt()}")
                del in_flight[out_path]
                failed.append(out_path.stem)
        if pending or in_flight:
            time.sleep(POLL_INTERVAL_SECONDS)
    if failed:
        # Fail loudly AFTER draining the queue: everything downloadable was
        # downloaded, but a green run must mean a complete one (an EWDS
        # schema flip mid-run once 400'd 28 chunks behind a 0 exit code).
        raise RuntimeError(
            f"{log_prefix}{len(failed)}/{total} chunks failed: {', '.join(failed)}"
        )
    return done


# Somalia flood seasons: Gu (Mar-Jun) and Deyr (Oct-Dec), matching the
# seasons used in the EF5 hindcasts
FLOOD_SEASON_MONTHS = ["03", "04", "05", "06", "10", "11", "12"]
# EWDS reforecast is frozen at GloFAS v4.2 and ends 2023-11-25
REFORECAST_YEARS = [str(y) for y in range(2003, 2024)]


def _reforecast_schema():
    """Detect the live EWDS reforecast schema; return (base_fields, valid_days).

    EWDS has flipped cems-glofas-reforecast between two schemas: the
    2026-08-06 restructure dropped system_version and put the v4.2 set
    under hydrological_model=global_lisflood pinned to the forecast cycle
    year=2022/month=10/day=01; on 2026-08-11 it reverted mid-flight to the
    old system_version=version_4_0 + hydrological_model=lisflood scheme
    (requests in the other spelling get 400s). Inspect constraints.json and
    build requests to match whichever is live. Under both schemas hday is
    strictly validated against the twice-weekly reforecast dates, which
    vary by month and year — valid_days maps (hyear, hmonth) to them.
    """
    import requests

    url = ("https://ewds.climate.copernicus.eu/api/catalogue/v1/"
           "collections/cems-glofas-reforecast/constraints.json")
    blocks = requests.get(url, timeout=60).json()
    if any("global_lisflood" in b.get("hydrological_model", []) for b in blocks):
        base = {"hydrological_model": "global_lisflood", **REFORECAST_CYCLE}

        def keep(b):
            return "global_lisflood" in b.get("hydrological_model", [])
    else:
        base = {"system_version": "version_4_0", "hydrological_model": "lisflood"}

        def keep(b):
            return "version_4_0" in b.get("system_version", [])

    days = {}
    for b in blocks:
        if not keep(b) or not all(k in b for k in ("hyear", "hmonth", "hday")):
            continue
        for y in b["hyear"]:
            for mo in b["hmonth"]:
                days.setdefault((y, mo), set()).update(b["hday"])
    return base, {k: sorted(v) for k, v in days.items()}


# forecast-cycle pin selecting the frozen GloFAS v4.2 reforecast set
REFORECAST_CYCLE = {"year": ["2022"], "month": ["10"], "day": ["01"]}


def download_reforecast_box(years=None, months=None, leadtime_days=(1, 2, 3, 4, 5, 6, 7),
                            dir_suffix="", blob_sync=False):
    """Download GloFAS v4.2 reforecast for the all-stations box at daily leads.

    Defaults: 2003-2023, Gu + Deyr months, leads 1-7 days, control plus
    perturbed ensemble members. One request per (year, month) with the
    constraint-valid hday list (the new EWDS schema rejects invalid days),
    bisected further by _plan_chunks only if it exceeds the live cost limit.

    dir_suffix separates lead-band downloads (the readiness band 8-12 d
    lands in reforecast_som_ext_lead8_12/).

    blob_sync=True uploads each chunk to blob as it finishes and skips
    chunks already in blob — required on ephemeral runners (Databricks),
    where nothing on local disk survives the run. Blob names mirror the
    scripts/upload_to_blob.py convention:
    ds-aa-som-floods/raw/glofas/raw/reforecast_som_ext{dir_suffix}/<file>.

    API note: EWDS has flipped the reforecast request schema twice
    (2026-08-06 and back on 2026-08-11); requests follow whichever schema
    is live, see _reforecast_schema.
    """
    years = list(years or REFORECAST_YEARS)
    months = list(months or FLOOD_SEASON_MONTHS)
    raw_dir = DATA_DIR / "raw" / f"reforecast_som_ext{dir_suffix}"

    skip = on_downloaded = None
    if blob_sync:
        import ocha_stratus as stratus

        from src.constants import BLOB_STAGE

        blob_prefix = f"ds-aa-som-floods/raw/glofas/raw/reforecast_som_ext{dir_suffix}/"
        existing = set(
            stratus.list_container_blobs(name_starts_with=blob_prefix, stage=BLOB_STAGE)
        )
        print(f"[reforecast/som-box] blob sync on: {len(existing)} chunks already in blob")

        def skip(out_path):
            return blob_prefix + out_path.name in existing

        def on_downloaded(out_path):
            stratus.upload_blob_data(
                out_path.read_bytes(),
                blob_prefix + out_path.name,
                stage=BLOB_STAGE,
                content_type="application/zip",
            )
            out_path.unlink()  # keep the ephemeral disk from filling
            print(f"[reforecast/som-box] {out_path.name}: uploaded to blob")

    client = _client(wait_until_complete=False)
    coll = client.client.get_collection("cems-glofas-reforecast")
    schema_base, valid_days = _reforecast_schema()
    base = {
        **schema_base,
        "product_type": ["control_reforecast", "ensemble_perturbed_reforecast"],
        "variable": "river_discharge_in_the_last_24_hours",
        "leadtime_hour": [str(int(d) * 24) for d in leadtime_days],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": SOM_AREA,
    }

    jobs = []
    for year in years:
        for month in months:
            days = valid_days.get((year, month))
            if not days:
                print(f"[reforecast/som-box] {year}-{month}: no valid reforecast days, skipping")
                continue
            req = {**base, "hyear": [year], "hmonth": [month], "hday": days}
            chunks = _plan_chunks(coll, req, f"rf_{year}_m{month}")
            for creq, prefix in chunks:
                jobs.append(("cems-glofas-reforecast", creq, raw_dir / f"{prefix}.zip"))
    print(f"[reforecast/som-box] {len(jobs)} chunks planned across {len(years)} years")
    return _submit_rolling(
        client, jobs, log_prefix="[reforecast/som-box] ",
        skip=skip, on_downloaded=on_downloaded,
    )
def download_reforecast_months(station_key, year_months):
    """Download GloFAS reforecast for one station, for specific (year, month) pairs.

    Each request covers one station-month at LEADTIME_DAYS lead times (both
    control and ensemble members) - the largest chunk confirmed to fit
    under the EWDS cost limit. Use this to target reforecast downloads
    around specific known flood events rather than the full calendar.
    """
    raw_dir = DATA_DIR / "raw" / "reforecast" / station_key
    jobs = {}
    for year, month in year_months:
        month = str(month).zfill(2)
        query = {
            "system_version": "version_4_0",
            "hydrological_model": "lisflood",
            "product_type": ["control_reforecast", "ensemble_perturbed_reforecast"],
            "variable": "river_discharge_in_the_last_24_hours",
            "hyear": str(year),
            "hmonth": month,
            "hday": ALL_DAYS,
            "leadtime_hour": [str(d * 24) for d in LEADTIME_DAYS],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": _station_area(station_key),
        }
        jobs[f"{year}-{month}"] = (
            "cems-glofas-reforecast",
            query,
            raw_dir / f"{year}-{month}.zip",
        )
    return _submit_and_download_all(jobs, log_prefix=f"[reforecast/{station_key}] ")


CHANNEL_SEARCH_DEG = 0.15


def _channel_cells_path(version=DEFAULT_VERSION):
    suffix = "" if version == DEFAULT_VERSION else f"_{version.split('_', 1)[1]}"
    return DATA_DIR / f"channel_cells{suffix}.json"


def _reanalysis_files(station_key, version=DEFAULT_VERSION):
    """Extended-box files if present, else the legacy box, else per-station.

    Never mixes directories: their spatial extents differ, and
    open_mfdataset would align them into a padded grid full of NaNs.
    """
    files = sorted(_box_dir(version).glob("*.nc"))
    if files:
        return files
    files = sorted((DATA_DIR / "raw" / VERSION_DIRS_LEGACY[version]).glob("*.nc"))
    if files:
        return files
    if version == DEFAULT_VERSION:
        return sorted((DATA_DIR / "raw" / "reanalysis" / station_key).glob("*.nc"))
    return []


def channel_cells(force=False, version=DEFAULT_VERSION):
    """Map each station to the GloFAS river cell nearest its gauge.

    The registry coordinates are snapped to the HydroSHEDS 30" channel used
    by EF5, which does not always coincide with GloFAS's own 0.05deg river
    network — at Jowhar the plain nearest cell is dry (0 m3/s) while the
    modelled channel is ~0.14deg away. For each station this picks the cell
    with the highest long-term mean discharge within CHANNEL_SEARCH_DEG,
    which is the channel. Result is cached to channel_cells.json; delete it
    or pass force=True to recompute after downloading more years.
    """
    path = _channel_cells_path(version)
    if path.exists() and not force:
        return json.loads(path.read_text())

    files = _reanalysis_files(next(iter(_REGISTRY)), version)
    if not files:
        raise FileNotFoundError(f"no GloFAS {version} reanalysis files downloaded yet")
    ds = xr.open_mfdataset(files, combine="by_coords")
    var = next(v for v in ds.data_vars if "dis" in v.lower())
    mean = ds[var].mean("valid_time").compute()

    # Reuse the default version's cells where they are also on-channel in this
    # version, so versions are compared at the SAME point. v4 and v5 share the
    # same river network here; only magnitudes differ, and because v5's flow
    # decreases downstream while v4's is nearly flat, an independent argmax
    # picks a different cell on the same channel and would confound the
    # version comparison with a location change.
    base = {}
    base_path = _channel_cells_path(DEFAULT_VERSION)
    if version != DEFAULT_VERSION and base_path.exists():
        base = json.loads(base_path.read_text())

    lat_v, lon_v = mean.latitude.values, mean.longitude.values
    half = 0.025  # half a 0.05deg cell

    cells = {}
    for key, st in _REGISTRY.items():
        # Refuse stations outside the downloaded extent. sel(method="nearest")
        # would silently snap them to the nearest edge cell — that is how the
        # lower-Juba points read plausible values from a box that stopped at
        # S 0.2. Omitting them makes the gap visible downstream instead.
        if not (lat_v.min() - half <= st.lat <= lat_v.max() + half
                and lon_v.min() - half <= st.lon <= lon_v.max() + half):
            print(f"  {key}: ({st.lon}, {st.lat}) outside GloFAS extent "
                  f"lat {lat_v.min():.3f}..{lat_v.max():.3f} "
                  f"lon {lon_v.min():.3f}..{lon_v.max():.3f} — skipped")
            continue
        win = mean.sel(
            latitude=slice(st.lat + CHANNEL_SEARCH_DEG, st.lat - CHANNEL_SEARCH_DEG),
            longitude=slice(st.lon - CHANNEL_SEARCH_DEG, st.lon + CHANNEL_SEARCH_DEG),
        )
        win_max = float(win.max())

        # Prefer the station's exact cell whenever it is on-channel: the
        # registry coordinates are user-verified GloFAS cell centres, and a
        # window argmax drifts downstream wherever discharge grows along the
        # window (it put juba_07's cell 0.15deg south, on juba_08). The
        # window search remains only as a rescue for off-channel points.
        at_exact = float(
            mean.sel(latitude=st.lat, longitude=st.lon, method="nearest")
        )
        if win_max > 0 and at_exact >= 0.5 * win_max:
            cells[key] = {
                "lat": st.lat, "lon": st.lon,
                "mean_discharge": round(at_exact, 2),
                "cell_from": "exact",
            }
            continue

        reused = False
        if key in base:
            at_base = float(
                mean.sel(latitude=base[key]["lat"], longitude=base[key]["lon"],
                         method="nearest")
            )
            # on-channel if it carries a substantial share of the local maximum
            if win_max > 0 and at_base >= 0.5 * win_max:
                cells[key] = {
                    "lat": base[key]["lat"],
                    "lon": base[key]["lon"],
                    "mean_discharge": round(at_base, 2),
                    "cell_from": DEFAULT_VERSION,
                }
                reused = True
        if not reused:
            idx = win.argmax(dim=["latitude", "longitude"])
            cells[key] = {
                "lat": float(win.latitude[idx["latitude"]]),
                "lon": float(win.longitude[idx["longitude"]]),
                "mean_discharge": round(win_max, 2),
                "cell_from": version,
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cells, indent=2))
    return cells


def load_reanalysis(station_key, version=DEFAULT_VERSION):
    """Load one station's reanalysis discharge as a pandas Series indexed by date.

    Extracted at the GloFAS channel cell for the station (see channel_cells),
    not the plain nearest cell. Prefers the all-stations box files
    (reanalysis_som/) when present, falling back to the older per-station
    downloads; the two cover different extents and are never mixed.
    """
    files = _reanalysis_files(station_key, version)
    cells = channel_cells(version=version)
    if station_key not in cells:
        raise KeyError(
            f"{station_key} has no GloFAS channel cell for {version} — it lies "
            f"outside the downloaded box extent (see channel_cells)"
        )
    cell = cells[station_key]
    ds = xr.open_mfdataset(files, combine="by_coords")
    # discharge variable name differs between pre- and post-2026-07 EWDS
    # downloads (dis24 vs. avg_dis24-style names) — pick it by pattern
    var = next(v for v in ds.data_vars if "dis" in v.lower())
    da = ds[var].sel(latitude=cell["lat"], longitude=cell["lon"], method="nearest")
    series = da.to_series()
    if series.index.nlevels > 1:
        time_level = next(n for n in series.index.names if n and "time" in n)
        series.index = series.index.get_level_values(time_level)
    # GloFAS stamps each 24-hour mean at the END of its period: a year=1999
    # request returns 365 values stamped 1999-01-02..2000-01-01. Shift back
    # one day so the index is the day each value actually describes, matching
    # GEOGloWS/GRRR/SWALIM. (Rank correlation can't resolve this on these
    # smooth rivers, but it matters for peak timing and event matching.)
    series.index = pd.to_datetime(series.index) - pd.Timedelta(days=1)
    series.name = "discharge"
    return series.sort_index().dropna()


def load_reforecast_box(version=DEFAULT_VERSION, dir_suffix=None):
    """Load the all-stations reforecast as one long DataFrame.

    Columns: station, issued_time, lead_hours, leadtime_days, valid_day,
    member, discharge. Members are 0 (control) plus 1-10 (perturbed).

    dir_suffix selects a lead-band download made by download_reforecast_box
    (e.g. "_lead8_12" reads reforecast_som_ext_lead8_12/, the readiness
    band). With the default None, keeps the historical behaviour: extended
    box if complete enough, else the legacy leads-1-7 box.

    Two lead conventions are kept deliberately:
      * leadtime_days = lead_hours / 24, i.e. GloFAS's own labelling
        ("day 1" = the 24 h ending 24 h after the 00Z issue).
      * valid_day = the calendar day each value actually describes
        (valid_time - 1 day, same end-of-period stamping as the reanalysis).
    So GloFAS "day 1..7" describes calendar days issue+0 .. issue+6; score
    against observations on valid_day.
    """
    cells = channel_cells(version=version)

    if dir_suffix is not None:
        rf_dir = DATA_DIR / "raw" / f"reforecast_som_ext{dir_suffix}"
        n_chunks = len(list(rf_dir.glob("*.zip")))
        print(f"  reforecast source: {rf_dir.name} ({n_chunks} chunks)")
        if not n_chunks:
            return None
    else:
        # Prefer the extended box only once it has at least as many chunks as
        # the legacy one — a partially downloaded _ext dir must not silently
        # replace a complete legacy archive (ties go to _ext, which covers
        # juba_07/08).
        ext_dir = DATA_DIR / "raw" / "reforecast_som_ext"
        leg_dir = DATA_DIR / "raw" / "reforecast_som"
        n_ext, n_leg = len(list(ext_dir.glob("*.zip"))), len(list(leg_dir.glob("*.zip")))
        rf_dir = ext_dir if n_ext >= max(n_leg, 1) else leg_dir
        print(f"  reforecast source: {rf_dir.name} ({max(n_ext, n_leg)} chunks; ext={n_ext}, legacy={n_leg})")

    # Extent guard, same as the reanalysis path: channel cells can lie outside
    # this file set's box (the legacy zips stop at S 0.2), and
    # sel(method="nearest") would silently snap them to edge cells — juba_08
    # would get a dry cell full of zeros. Skip them visibly instead.
    probe = xr.open_dataset(_unwrap(sorted(rf_dir.glob("*.zip"))[0])[0])
    lat_v, lon_v = probe.latitude.values, probe.longitude.values
    probe.close()
    half = 0.025
    outside = [
        k for k, c in cells.items()
        if not (lat_v.min() - half <= c["lat"] <= lat_v.max() + half
                and lon_v.min() - half <= c["lon"] <= lon_v.max() + half)
    ]
    if outside:
        print(f"  reforecast: stations outside {rf_dir.name} extent, skipped: {outside}")
        cells = {k: c for k, c in cells.items() if k not in outside}

    station_keys = list(cells)
    lats = xr.DataArray([cells[k]["lat"] for k in station_keys], dims="station",
                        coords={"station": station_keys})
    lons = xr.DataArray([cells[k]["lon"] for k in station_keys], dims="station",
                        coords={"station": station_keys})

    frames = []
    for zip_path in sorted(rf_dir.glob("*.zip")):
        for part in _unwrap(zip_path):
            ds = xr.open_dataset(part)
            var = next(v for v in ds.data_vars if "dis" in v.lower())
            da = ds[var].sel(latitude=lats, longitude=lons, method="nearest")
            if "number" not in da.dims:
                da = da.expand_dims("number")
            df = da.to_dataframe(name="discharge").reset_index()
            keep = ["station", "number", "forecast_reference_time",
                    "forecast_period", "discharge"]
            df = df[[c for c in keep if c in df.columns]].rename(
                columns={"number": "member", "forecast_reference_time": "issued_time"}
            )
            df["lead_hours"] = (df["forecast_period"] / pd.Timedelta(hours=1)).astype(int)
            frames.append(df.drop(columns="forecast_period"))
            ds.close()
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["leadtime_days"] = out["lead_hours"] // 24
    out["valid_day"] = (
        out["issued_time"] + pd.to_timedelta(out["lead_hours"], unit="h")
        - pd.Timedelta(days=1)
    ).dt.normalize()
    return out.dropna(subset=["discharge"]).reset_index(drop=True)


def load_reforecast(station_key):
    """Load one station's reforecast ensemble as a single combined xarray DataArray.

    Dims: number (0=control, 1-10=perturbed), forecast_reference_time
    (issue date), forecast_period (lead time).
    """
    raw_dir = DATA_DIR / "raw" / "reforecast" / station_key
    s = STATIONS[station_key]
    month_parts = []
    for zip_path in sorted(raw_dir.glob("*.zip")):
        files = _unwrap(zip_path)
        cf_pf_parts = []
        for f in files:
            ds = xr.load_dataset(f)
            if "number" not in ds["dis24"].dims:
                ds = ds.expand_dims("number")
            cf_pf_parts.append(ds)
        month_parts.append(xr.concat(cf_pf_parts, dim="number").sortby("number"))
    combined = xr.concat(month_parts, dim="forecast_reference_time").sortby(
        "forecast_reference_time"
    )
    da = combined["dis24"].sel(
        latitude=s["lat"], longitude=s["lon"], method="nearest"
    )
    return da
