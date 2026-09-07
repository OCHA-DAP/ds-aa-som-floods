"""Shared helpers for the trigger-page additions: local data, the two-gauge
benchmark, per-window activations (reanalysis flow dates and forecast issue
dates) and gauge crossing dates. Mirrors scripts/envelope_search.py and
scripts/model_selection.py in the repo so numbers agree with the page."""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\pauni\Desktop\Work\OCHA\GitHub\ds-aa-som-floods")
sys.path.insert(0, str(REPO))
from src.constants import (REFERENCE_GAUGE, SEASONS, SEVERE_RP, TRIGGER_CONFIG,  # noqa: E402
                           TRIGGER_STATIONS, TRIGGER_YEARS)
from src.utils import weibull_level, weibull_threshold  # noqa: E402

P = REPO / "data" / "processed"
Y0, Y1 = TRIGGER_YEARS
SPAN = list(range(Y0, Y1 + 1))
WINDOWS = [("juba", "deyr"), ("shabelle", "deyr"), ("juba", "gu"), ("shabelle", "gu")]
MODELS = ["google_grrr", "glofas_v5", "glofas_v4"]
NICE = {"google_grrr": "Google", "glofas_v5": "GloFAS v5", "glofas_v4": "GloFAS v4"}
NAME = {"belet_weyne": "Belet Weyne", "bulo_burti": "Bulo Burti", "jowhar": "Jowhar", "dollow": "Dollow",
        "luuq": "Luuq", "bardheere": "Bardheere", "bualle": "Bualle"}
BENCH_GAUGES = 2
MIN_LAG, MAX_LAG, MIN_OBS = -10, 30, 60  # as scripts/model_selection.py


def wname(river, season):
    return f"{season.title()} {river.title()}"


@lru_cache(None)
def daily(model):
    d = pd.read_parquet(P / f"discharge_daily_{model}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d


@lru_cache(None)
def levels():
    lv = pd.read_parquet(P / "swalim_levels.parquet")
    lv["date"] = pd.to_datetime(lv["date"])
    return lv


@lru_cache(None)
def swalim_thresholds():
    return pd.read_parquet(P / "swalim_thresholds.parquet").set_index("station")


@lru_cache(None)
def reforecast(model):
    rf = pd.read_parquet(P / f"reforecast_{model}.parquet")
    rf["issued_time"] = pd.to_datetime(rf["issued_time"])
    rf["valid_time"] = pd.to_datetime(rf["valid_time"])
    return rf


def season_series(model, station, season, span=(Y0, Y1)):
    d = daily(model)
    s = d[d.station == station].set_index("date")["discharge"].sort_index()
    s = s[s.index.month.isin(SEASONS[season])]
    return s[(s.index.year >= span[0]) & (s.index.year <= span[1])]


def model_threshold(model, station, season, rp):
    s = season_series(model, station, season)
    am = s.groupby(s.index.year).max().dropna()
    return float(weibull_threshold(am.values, rp)) if len(am) else np.nan


def model_thresholds(model, river, season, rp):
    return pd.Series({st: model_threshold(model, st, season, rp) for st in TRIGGER_STATIONS[river]})


def daily_matrix(model, river, season):
    """date x station discharge for the window's points, all years."""
    d = daily(model)
    d = d[d.station.isin(TRIGGER_STATIONS[river]) & d.date.dt.month.isin(SEASONS[season])]
    return d.pivot_table(index="date", columns="station", values="discharge")


def activation_years(model, river, season, rp, n_req, span=SPAN):
    """Years the window rule activates on the model's reanalysis: set of years."""
    m = daily_matrix(model, river, season)
    thr = model_thresholds(model, river, season, rp)
    cols = [c for c in thr.index if c in m.columns and not np.isnan(thr[c])]
    over = m[cols] >= thr[cols]
    mx = over.sum(axis=1).groupby(over.index.year).max()
    return set(mx[mx >= n_req].index) & set(span)


def first_crossing_dates(model, river, season, rp, n_req, span=SPAN):
    """Year -> first flow date on which >= n_req points are over their RP level."""
    m = daily_matrix(model, river, season)
    thr = model_thresholds(model, river, season, rp)
    cols = [c for c in thr.index if c in m.columns and not np.isnan(thr[c])]
    votes = (m[cols] >= thr[cols]).sum(axis=1)
    hit = votes[votes >= n_req]
    out = {}
    for d in hit.index:
        if d.year in span and d.year not in out:
            out[d.year] = d.normalize()
    return out


def first_issue_dates(model, river, season, rp, n_req, span=SPAN, leads=(1, 7)):
    """Year -> (first issue date, valid day) on which the ensemble median at leads 1-7
    put >= n_req points over their RP level (levels from the model's reanalysis)."""
    rf = reforecast(model)
    rf = rf[rf.station.isin(TRIGGER_STATIONS[river]) & rf.leadtime_days.between(*leads)
            & rf.valid_time.dt.month.isin(SEASONS[season])]
    med = rf.groupby(["issued_time", "valid_time", "station"]).discharge.median().unstack("station")
    thr = model_thresholds(model, river, season, rp)
    cols = [c for c in thr.index if c in med.columns and not np.isnan(thr[c])]
    votes = (med[cols] >= thr[cols]).sum(axis=1)
    hit = votes[votes >= n_req].reset_index()
    hit.columns = ["issued", "valid", "votes"]
    hit["year"] = hit["valid"].dt.year
    out = {}
    for y, g in hit.groupby("year"):
        if y in span:
            g = g.sort_values(["issued", "valid"])
            out[y] = (g.iloc[0]["issued"].normalize(), g.iloc[0]["valid"].normalize())
    return out


def gauge_levels(river, season, rp):
    """Station -> post-2000 Weibull level at rp (NaN where the record is short)."""
    lv = levels()
    out = {}
    for st in TRIGGER_STATIONS[river]:
        s = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
        s = s[s.index.month.isin(SEASONS[season])]
        modern = s[(s.index.year >= 2000) & (s.index.year <= Y1)]
        am = modern.groupby(modern.index.year).max().dropna()
        out[st] = float(weibull_level(am.values, rp)) if len(am) else np.nan
    return pd.Series(out)


def gauge_matrix(river, season):
    lv = levels()
    d = lv[lv.station.isin(TRIGGER_STATIONS[river]) & lv.date.dt.month.isin(SEASONS[season])]
    return d.pivot_table(index="date", columns="station", values="level_m")


def gauge_crossings(river, season, rp, n_req=BENCH_GAUGES, span=SPAN):
    """Year -> first date on which >= n_req gauges have crossed their own RP level that
    season (cumulative: a gauge counts once it has crossed)."""
    m = gauge_matrix(river, season)
    lev = gauge_levels(river, season, rp)
    cols = [c for c in lev.index if c in m.columns and not np.isnan(lev[c])]
    out = {}
    for y, g in m[cols].groupby(m.index.year):
        if y not in span:
            continue
        crossed = (g >= lev[cols]).cummax()
        n = crossed.sum(axis=1)
        hit = n[n >= n_req]
        if len(hit):
            out[y] = hit.index[0].normalize()
    return out


def assessable_years(river, season, span=SPAN, min_days=30):
    """Years in which at least BENCH_GAUGES gauges of the river report in season."""
    m = gauge_matrix(river, season)
    cnt = m.notna().groupby(m.index.year).sum()
    ok = (cnt >= min_days).sum(axis=1)
    return set(ok[ok >= BENCH_GAUGES].index) & set(span)


def benchmark(river, season, span=SPAN):
    """(flood years at two-gauge RP3, severe years at two-gauge RP5, assessable years)."""
    flood = set(gauge_crossings(river, season, 3, span=span))
    severe = set(gauge_crossings(river, season, SEVERE_RP, span=span))
    return flood, severe, assessable_years(river, season, span)


def contingency(active, positive, universe):
    """Counts and scores for a set of activation years against positive years."""
    active, positive, universe = set(active) & set(universe), set(positive) & set(universe), set(universe)
    a = len(active & positive)                 # hits
    b = len(active - positive)                 # false alarms
    c = len(positive - active)                 # misses
    d = len(universe - active - positive)      # correct negatives
    n = a + b + c + d
    div = lambda x, y: (x / y) if y else float("nan")
    pod = div(a, a + c); pofd = div(b, b + d); far = div(b, a + b)
    exp_correct = div((a + b) * (a + c) + (c + d) * (b + d), n)
    return {
        "hits": a, "false_alarms": b, "misses": c, "correct_negatives": d, "n": n,
        "POD": pod, "FAR": far, "POFD": pofd,
        "CSI": div(a, a + b + c), "bias": div(a + b, a + c),
        "PSS": (pod - pofd) if not (np.isnan(pod) or np.isnan(pofd)) else float("nan"),
        "HSS": div(a + d - exp_correct, n - exp_correct),
        "F1": div(2 * a, 2 * a + b + c), "accuracy": div(a + d, n),
        "precision": div(a, a + b),
    }


def auc(points):
    """Trapezoid area under (POFD, POD) points, closed at (0,0) and (1,1)."""
    pts = sorted(set(points) | {(0.0, 0.0), (1.0, 1.0)})
    return float(sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:])))
