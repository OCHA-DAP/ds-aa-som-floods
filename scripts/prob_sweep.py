"""Which ensemble-exceedance probability beats the median vote rule?

Current rule: a point votes when the ENSEMBLE MEDIAN crosses its level,
which with the 11-member reforecast equals "at least 6 of 11 members over".
Sweep k = 1..11 ("at least k members over, share k/11") on the GloFAS v4
reforecast, levels fitted on the v4 reanalysis at each leg's return period
(frequency matching), and score each k against the two-gauge benchmark,
2003-2023. Feeds pages/ensemble-vote/.

Per (station, valid day): the most alarming signal in the leg's lead band,
i.e. max over issues of the member-exceedance count, same "best lead per
valid day" convention as the operational backtest. A window activates in a
season-year when >= n_req points vote on the SAME valid day.

Run from the repo root:
    .venv/Scripts/python.exe scripts/prob_sweep.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.constants import (  # noqa: E402
    SEASONS, SEVERE_RP, TRIGGER_CONFIG, TRIGGER_STATIONS, TRIGGER_YEARS,
)
from src.datasources import glofas  # noqa: E402
from src.utils import weibull_threshold  # noqa: E402
import envelope_search  # noqa: E402

N_MEMBERS = 11
Y0, Y1 = 2003, 2023  # reforecast archive span

# Readiness rules as published in the mechanism table.
READINESS = {
    ("juba", "gu"): {"rp": 5, "n_req": 3},
    ("juba", "deyr"): {"rp": 4, "n_req": 3},
    ("shabelle", "gu"): {"rp": 5, "n_req": 2},
    ("shabelle", "deyr"): {"rp": 4, "n_req": 2},
}


def fit_thresholds():
    """Windows dict with per-leg levels fitted on the v4 reanalysis."""
    dd = envelope_search.load("discharge_daily_glofas_v4")
    dd["date"] = pd.to_datetime(dd["date"])
    fy0, fy1 = TRIGGER_YEARS
    windows = {}
    for (river, season), cfg in TRIGGER_CONFIG.items():
        w = {"stations": TRIGGER_STATIONS[river],
             "action": {"rp": cfg["rp"], "n_req": cfg["n_req"],
                        "thresholds": {}},
             "readiness": {**READINESS[(river, season)], "thresholds": {}}}
        for st in TRIGGER_STATIONS[river]:
            s = dd[dd.station == st].set_index("date")["discharge"]
            s = s[s.index.month.isin(SEASONS[season])]
            s = s[(s.index.year >= fy0) & (s.index.year <= fy1)]
            am = s.groupby(s.index.year).max().dropna()
            w["action"]["thresholds"][st] = weibull_threshold(am.values,
                                                              cfg["rp"])
            w["readiness"]["thresholds"][st] = weibull_threshold(
                am.values, READINESS[(river, season)]["rp"])
        windows[f"{river}_{season}"] = w
    return windows


TH = fit_thresholds()


def leg_band_frame(leg):
    """Member-level reforecast rows for the leg's lead band, trigger stations."""
    if leg == "action":
        df = glofas.load_reforecast_box(version="version_4_0")
        df = df[df.leadtime_days.between(1, 7)]
    else:
        parts = [glofas.load_reforecast_box(version="version_4_0",
                                            dir_suffix="_lead8_12")]
        d7 = glofas.load_reforecast_box(version="version_4_0")
        parts.append(d7[d7.leadtime_days == 7])
        df = pd.concat(parts, ignore_index=True)
    keep = {st for river in TRIGGER_STATIONS for st in TRIGGER_STATIONS[river]}
    df = df[df.station.isin(keep)]
    return df


def daily_exceed_counts(df, leg):
    """{window: DataFrame valid_day x station -> max members over level}."""
    out = {}
    for wkey, w in TH.items():
        river, season = wkey.split("_")
        sub = df[df.station.isin(w["stations"])].copy()
        sub = sub[sub.valid_day.dt.month.isin(SEASONS[season])]
        rows = []
        for st in w["stations"]:
            thr = w[leg]["thresholds"][st]
            s = sub[sub.station == st]
            # members over the level, per (issue, valid_day); then the most
            # alarming issue per valid day
            over = (s.assign(hit=s.discharge >= thr)
                     .groupby(["issued_time", "valid_day"])["hit"].sum()
                     .groupby("valid_day").max())
            rows.append(over.rename(st))
        out[wkey] = pd.concat(rows, axis=1)
    return out


def score(counts, leg):
    """Long results: window, k, activation years, POD severe, false, F1."""
    lv = envelope_search.load("swalim_levels")
    lv["date"] = pd.to_datetime(lv["date"])
    res = []
    for wkey, mat in counts.items():
        river, season = wkey.split("_")
        w = TH[wkey]
        n_req = w[leg]["n_req"]
        span = set(range(Y0, Y1 + 1))
        floods = envelope_search.gauge_consensus_years(lv, river, season, 3) & span
        severe = envelope_search.gauge_consensus_years(
            lv, river, season, SEVERE_RP) & span
        for k in range(1, N_MEMBERS + 1):
            votes = (mat >= k).sum(axis=1)
            days = votes[votes >= n_req]
            years = {d.year for d in days.index}
            hits = years & severe
            false = years - floods
            pod = len(hits) / len(severe) if severe else np.nan
            prec = len(years & floods) / len(years) if years else np.nan
            f1 = (0 if not years or not severe else
                  2 * len(hits) / (len(years) + len(severe))
                  if (len(years) + len(severe)) else np.nan)
            res.append({
                "window": wkey, "leg": leg, "k": k,
                "share": k / N_MEMBERS,
                "n_years": len(years),
                "years": sorted(years),
                "pod_severe": pod, "false_act": len(false),
                "f1_severe": round(f1, 3),
                "severe_n": len(severe),
            })
    return pd.DataFrame(res)


def main():
    all_res = []
    for leg in ("action", "readiness"):
        print(f"loading {leg} band ...")
        df = leg_band_frame(leg)
        print(f"  {len(df):,} rows | leads {sorted(df.leadtime_days.unique())} "
              f"| members {df.member.nunique()} | "
              f"{df.valid_day.dt.year.min()}-{df.valid_day.dt.year.max()}")
        counts = daily_exceed_counts(df, leg)
        all_res.append(score(counts, leg))
    res = pd.concat(all_res, ignore_index=True)
    res.to_json(OUT / "prob_sweep_results.json", orient="records")
    for leg in ("action", "readiness"):
        print(f"\n=== {leg.upper()} (k of 11 members; median rule = k=6)")
        for wkey in TH:
            d = res[(res.window == wkey) & (res.leg == leg)]
            base = d[d.k == 6].iloc[0]
            print(f"\n{wkey}  (needs {TH[wkey][leg]['n_req']} points, "
                  f"RP{TH[wkey][leg]['rp']}, severe n={base.severe_n})")
            for _, r in d.iterrows():
                mark = " <- median" if r.k == 6 else ""
                print(f"  k={r.k:2d} ({r.share:.0%})  acts={r.n_years:2d}  "
                      f"POD={r.pod_severe if not np.isnan(r.pod_severe) else float('nan'):.2f}  "
                      f"false={r.false_act}  F1={r.f1_severe:.2f}{mark}")


if __name__ == "__main__":
    main()
