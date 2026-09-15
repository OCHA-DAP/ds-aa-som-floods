"""Per-station trigger thresholds for every candidate climatology.

Exactly the estimator the trigger analysis used (scripts/page_additions/
somlib.model_threshold): for one model, station and season, take the
seasonal maximum of the model's own daily record in each year of
TRIGGER_YEARS and read the return level off the Weibull plotting position
(log-log interpolation, no extrapolation). Readiness thresholds are fitted
the same way on the readiness-band forecast series (per valid day, the
ensemble median at each lead, then the most alarming lead in the band), as
the trigger page describes.

`build()` needs the processed parquets in data/processed/ (restore with
scripts/restore_from_blob.py) and is run by
scripts/build_monitoring_thresholds.py, which freezes the result to
thresholds.json next to this file. The pipeline only ever reads the JSON.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (ALL_TRIGGER_STATIONS, SEASONS, STATIONS,
                           TRIGGER_YEARS)
from src.utils import weibull_threshold

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "thresholds.json"
PROCESSED = HERE.parents[1] / "data" / "processed"

REANALYSIS_SOURCES = ["google_grrr", "glofas_v4", "glofas_v5"]
RPS = [3, 4, 5, 6]
READINESS_BAND = (8, 12)  # the archived band is 8-12 (reforecast_glofas_v4_lead8_12)


def _season_annual_max(series, season, years=TRIGGER_YEARS):
    s = series[series.index.month.isin(SEASONS[season])]
    s = s[(s.index.year >= years[0]) & (s.index.year <= years[1])]
    return s.groupby(s.index.year).max().dropna()


def fit(series, season, rp, years=TRIGGER_YEARS):
    """Return level at `rp` for one daily series, one season."""
    am = _season_annual_max(series, season, years)
    if len(am) < 2:
        return np.nan
    return float(weibull_threshold(am.values, rp))


def _daily(source):
    d = pd.read_parquet(PROCESSED / f"discharge_daily_{source}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d


def readiness_series(source="glofas_v4", band=READINESS_BAND):
    """Daily series per station: ensemble median per lead, max over the band."""
    name = {"glofas_v4": "reforecast_glofas_v4_lead8_12"}[source]
    rf = pd.read_parquet(PROCESSED / f"{name}.parquet")
    rf["valid_time"] = pd.to_datetime(rf["valid_time"])
    rf = rf[rf.leadtime_days.between(*band) & rf.station.isin(ALL_TRIGGER_STATIONS)]
    med = (rf.groupby(["station", "valid_time", "leadtime_days"])["discharge"].median()
             .groupby(level=["station", "valid_time"]).max()
             .reset_index().rename(columns={"valid_time": "date"}))
    return med


def build():
    rows = []
    for source in REANALYSIS_SOURCES:
        d = _daily(source)
        for st in ALL_TRIGGER_STATIONS:
            s = d[d.station == st].set_index("date")["discharge"].sort_index()
            for season in SEASONS:
                am = _season_annual_max(s, season)
                for rp in RPS:
                    rows.append({"basis": "reanalysis", "source": source, "station": st,
                                 "river": STATIONS[st].river, "season": season, "rp": rp,
                                 "threshold": fit(s, season, rp), "n_years": int(len(am)),
                                 "years": f"{TRIGGER_YEARS[0]}-{TRIGGER_YEARS[1]}"})
    # readiness band, GloFAS v4 reforecast (the only archive at leads 8-12)
    med = readiness_series("glofas_v4")
    y0, y1 = int(med.date.dt.year.min()), int(med.date.dt.year.max())
    for st in ALL_TRIGGER_STATIONS:
        s = med[med.station == st].set_index("date")["discharge"].sort_index()
        for season in SEASONS:
            am = _season_annual_max(s, season, (y0, y1))
            for rp in RPS:
                rows.append({"basis": "readiness_band", "source": "glofas_v4", "station": st,
                             "river": STATIONS[st].river, "season": season, "rp": rp,
                             "threshold": fit(s, season, rp, (y0, y1)), "n_years": int(len(am)),
                             "years": f"{y0}-{y1}"})
    return pd.DataFrame(rows)


def save(df, path=JSON_PATH):
    recs = df.replace({np.nan: None}).to_dict(orient="records")
    path.write_text(json.dumps({"generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
                                "estimator": "Weibull plotting position, log-log interpolation, seasonal annual maxima",
                                "rows": recs}, indent=1) + "\n")


def load(path=JSON_PATH):
    return pd.DataFrame(json.loads(path.read_text())["rows"])


def lookup(df, source, season, rp, stations, basis="reanalysis"):
    """station -> threshold for one (source, season, rp)."""
    sub = df[(df.basis == basis) & (df.source == source) & (df.season == season) & (df.rp == rp)]
    out = sub.set_index("station")["threshold"].reindex(stations)
    missing = out[out.isna()].index.tolist()
    if missing:
        raise KeyError(f"no {basis} threshold for {source} {season} RP{rp} at {missing}")
    return out.astype(float).to_dict()
