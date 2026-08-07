import ocha_stratus as stratus
import pandas as pd

from src.constants import BLOB_STAGE, SWALIM_RAW_PREFIX, SWALIM_THRESHOLD_FILENAME


def _parse_meters(value):
    """Parse strings like '5.5m' to float meters; '-'/NaN -> NaN."""
    if pd.isna(value):
        return float("nan")
    value = str(value).strip()
    if value in ("-", ""):
        return float("nan")
    return float(value.rstrip("m"))


def load_thresholds():
    """Load SWALIM station metadata and flood-risk thresholds (in meters)."""
    df = stratus.load_csv_from_blob(
        f"{SWALIM_RAW_PREFIX}/{SWALIM_THRESHOLD_FILENAME}", stage=BLOB_STAGE
    )
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    for col in [
        "Level",
        "Moderate Flood Risk",
        "High Flood Risk",
        "Bank Full",
        "Maximum Depth",
        "Maximum Width",
        "Maximum Flow",
    ]:
        df[col] = df[col].apply(_parse_meters)
    df["Station Number"] = df["Station Number"].str.lower()
    return df.set_index("Station Number")


def list_station_blobs():
    """List SWALIM per-station level-data blobs (excludes the threshold file)."""
    blobs = stratus.list_container_blobs(
        name_starts_with=f"{SWALIM_RAW_PREFIX}/", stage=BLOB_STAGE
    )
    return [b for b in blobs if SWALIM_THRESHOLD_FILENAME not in b]


def load_station_levels(blob_name):
    """Load one SWALIM station's daily water-level time series.

    A handful of station files are header-only (no rows) for gauges that
    were never activated; callers should check `len(df)` before use.
    """
    df = stratus.load_csv_from_blob(blob_name, stage=BLOB_STAGE)
    # date format is inconsistent across station files (ISO vs. dd/mm/yyyy);
    # each file uses one format throughout, so detect and parse explicitly
    # rather than using pandas "mixed" mode, which misparses ISO dates here
    date_format = (
        "%d/%m/%Y" if df["date"].str.contains("/", na=False).any() else "%Y-%m-%d"
    )
    df["date"] = pd.to_datetime(df["date"], format=date_format)
    df["station_number"] = df["station_number"].str.lower()
    df = df.drop_duplicates(subset="date").sort_values("date")
    return df.set_index("date")
