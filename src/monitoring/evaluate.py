"""Apply the four river-season window rules to one monitoring day's forecasts.

A window's rule (src.constants.TRIGGER_CONFIG, via config.ACTION_RULES and
READINESS_RULES): on any single forecast valid day, at least n_req of the
river's points are at or above their own return-period threshold, reading
the ensemble median for GloFAS and the deterministic value for Google.
Action reads the window's source at leads 1-7; readiness reads GloFAS at
leads 8-12 with the readiness thresholds. Only valid days inside the
window's season months count, so a late-September forecast can already
carry Deyr votes.

The result is a plain dict of scalars, lists and dicts so the same object
feeds the email template, the chart header and status.json; the status page
therefore cannot disagree with the last email.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from src.constants import SEASONS, TRIGGER_STATIONS
from src.monitoring import config as cfg
from src.monitoring import thresholds as thr


def _window_open(monitoring_date, season):
    """Emails go out for a window while a forecast valid day can fall in it:
    from 12 days before the season's first month to its last day."""
    months = SEASONS[season]
    horizon = [monitoring_date + timedelta(days=d) for d in range(0, cfg.READINESS_LEADS[1] + 1)]
    return any(d.month in months for d in horizon)


def _leg(df, source, stations, levels, leads, months, n_req):
    sub = df[(df.source == source) & df.station.isin(stations)
             & df.leadtime_days.between(*leads) & df.valid_date.dt.month.isin(months)]
    per_station = {}
    live = set(df[df.source == source].station)  # served by the product at all
    for st in stations:
        s = sub[sub.station == st]
        if s.empty:
            per_station[st] = {"threshold": round(levels[st], 1), "max_value": None, "max_date": None,
                               "pct_of_threshold": None, "exceeds": False, "reporting": st in live}
            continue
        i = s["value"].idxmax()
        mx = float(s.loc[i, "value"])
        per_station[st] = {"threshold": round(levels[st], 1), "max_value": round(mx, 1),
                           "max_date": s.loc[i, "valid_date"].strftime("%Y-%m-%d"),
                           "pct_of_threshold": round(100 * mx / levels[st], 1),
                           "exceeds": bool(mx >= levels[st]), "reporting": True}
    if sub.empty:
        votes = pd.Series(dtype=int)
    else:
        over = sub.assign(over=sub["value"] >= sub["station"].map(levels))
        votes = over.groupby("valid_date")["over"].sum().astype(int)
    max_votes = int(votes.max()) if len(votes) else 0
    return {
        "n_req": n_req, "n_of": len(stations),
        "votes_by_day": {d.strftime("%Y-%m-%d"): int(v) for d, v in votes.items()},
        "max_votes": max_votes,
        "max_votes_date": votes.idxmax().strftime("%Y-%m-%d") if max_votes else None,
        "activated": bool(max_votes >= n_req),
        "n_reporting": int(sum(1 for v in per_station.values() if v["reporting"])),
        "stations": per_station,
    }


def evaluate(df, monitoring_date=None, levels_df=None):
    levels_df = thr.load() if levels_df is None else levels_df
    if monitoring_date is None:
        monitoring_date = pd.Timestamp(df.monitoring_date.iloc[0]).date()
    assert df.monitoring_date.nunique() == 1, "one monitoring day at a time"
    windows = {}
    for w in cfg.WINDOWS:
        river, season = w
        stations = TRIGGER_STATIONS[river]
        months = SEASONS[season]
        a = cfg.ACTION_RULES[w]
        r = cfg.READINESS_RULES[w]
        a_levels = thr.lookup(levels_df, cfg.threshold_source(a["source"]), season, a["rp"], stations)
        r_levels = thr.lookup(levels_df, cfg.GLOFAS_OPERATIONAL, season, r["rp"], stations,
                              basis="readiness_band")
        action = _leg(df, a["source"], stations, a_levels, cfg.ACTION_LEADS, months, a["n_req"])
        action.update({"source": a["source"], "rp": a["rp"], "leads": list(cfg.ACTION_LEADS)})
        ready = _leg(df, r["source"], stations, r_levels, cfg.READINESS_LEADS, months, r["n_req"])
        ready.update({"source": r["source"], "rp": r["rp"], "leads": list(cfg.READINESS_LEADS)})
        is_open = _window_open(monitoring_date, season)
        windows[cfg.WINDOW_KEY[w]] = {
            "river": river, "season": season, "title": cfg.WINDOW_TITLE[w],
            "open": is_open, "in_season": monitoring_date.month in months,
            "action": action, "readiness": ready,
        }
    open_w = [k for k, v in windows.items() if v["open"]]
    action_hit = [k for k in open_w if windows[k]["action"]["activated"]]
    ready_hit = [k for k in open_w if windows[k]["readiness"]["activated"]]
    glofas = df[df.source == "glofas"]
    google = df[df.source == "google"]
    return {
        "monitoring_date": str(monitoring_date),
        "open_windows": open_w,
        "action": bool(action_hit), "action_windows": action_hit,
        "readiness": bool(ready_hit), "readiness_windows": ready_hit,
        "status": ("ACTION TRIGGER REACHED" if action_hit else
                   "READINESS TRIGGER REACHED" if ready_hit else
                   "NOT ACTIVATED" if open_w else "OUT OF SEASON"),
        "glofas_version": glofas.model_version.iloc[0] if len(glofas) else None,
        "glofas_issue": (pd.Timestamp(glofas.issued_time.iloc[0]).strftime("%Y-%m-%d")
                         if len(glofas) else None),
        "google_issue": (pd.Timestamp(google.issued_time.max()).strftime("%Y-%m-%d %H:%M UTC")
                         if len(google) else None),
        "n_glofas_points": int(glofas.station.nunique()),
        "n_google_gauges": int(google.station.nunique()),
        "windows": windows,
    }


def status_label(result):
    return result["status"]
