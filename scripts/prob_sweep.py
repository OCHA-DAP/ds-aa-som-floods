"""Which ensemble-exceedance probability beats the median vote rule?

Live rule: a point votes when the ENSEMBLE MEDIAN crosses its level, which
with the 11-member reforecast equals "at least 6 of 11 members over". Sweep
k = 1..11 ("at least k members over, share k/11") on the GloFAS v4
reforecast, action leg at leads 1-7 and readiness leg at leads 8-12, both
against the reanalysis levels the live pipeline reads
(src/monitoring/thresholds.json, the window's return period, readiness capped
at READINESS_RP_CAP), and score each k against the two-gauge benchmark,
2003-2023. Feeds pages/ensemble-agreement/ (figure and table via
scripts/page_additions/ensemble_agreement_figs.py).

Per (station, issue, valid day): the count of members over the level. A
window activates in a season-year when >= n_req points have >= k members
over on the SAME issue and valid day, exactly as src/monitoring/evaluate.py
counts votes, so the k = 6 row is the live rule.

Run from the repo root:
    .venv/Scripts/python.exe scripts/prob_sweep.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "scripts" / "page_additions"
# the processed parquets live in the main checkout; point SOM_DATA_REPO at it
# when running from a worktree (same convention as page_additions/somlib.py)
PROCESSED = Path(os.environ.get("SOM_DATA_REPO", REPO)) / "data" / "processed"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.constants import (  # noqa: E402
    SEASONS, SEVERE_RP, TRIGGER_CONFIG, TRIGGER_STATIONS,
)
from src.monitoring import config as cfg  # noqa: E402
from src.monitoring import thresholds as thr  # noqa: E402
import envelope_search  # noqa: E402

N_MEMBERS = 11
Y0, Y1 = 2003, 2023  # reforecast archive span
LEADS = {"action": cfg.ACTION_LEADS, "readiness": cfg.READINESS_LEADS}
ARCHIVE = {"action": "reforecast_glofas_v4.parquet",
           "readiness": "reforecast_glofas_v4_lead8_12.parquet"}
RULES = {"action": cfg.ACTION_RULES, "readiness": cfg.READINESS_RULES}


def windows(leg):
    """{window key: stations, rp, n_req, levels} for one leg, live levels."""
    lv = thr.load()
    out = {}
    for (river, season), rule in RULES[leg].items():
        out[f"{river}_{season}"] = {
            "river": river, "season": season, "stations": TRIGGER_STATIONS[river],
            "rp": rule["rp"], "n_req": rule["n_req"],
            "levels": thr.lookup(lv, cfg.GLOFAS_OPERATIONAL, season, rule["rp"],
                                 TRIGGER_STATIONS[river]),
        }
    return out


def member_frame(leg):
    """Member-level reforecast rows in the leg's lead band, trigger stations."""
    df = pd.read_parquet(PROCESSED / ARCHIVE[leg])
    lo, hi = LEADS[leg]
    df = df[df.leadtime_days.between(lo, hi) & df.station.isin(sum(TRIGGER_STATIONS.values(), []))]
    df["issued_time"] = pd.to_datetime(df["issued_time"])
    df["valid_day"] = pd.to_datetime(df["valid_time"]).dt.normalize()
    return df


def over_counts(df, w):
    """DataFrame (issue, valid_day) x station: members over the level."""
    sub = df[df.station.isin(w["stations"]) & df.valid_day.dt.month.isin(SEASONS[w["season"]])]
    lev = sub.station.map(w["levels"])
    return (sub.assign(hit=sub.discharge >= lev)
               .groupby(["issued_time", "valid_day", "station"])["hit"].sum()
               .unstack("station").reindex(columns=w["stations"]))


def score(counts_by_window, leg, ws):
    lv = pd.read_parquet(PROCESSED / "swalim_levels.parquet")
    lv["date"] = pd.to_datetime(lv["date"])
    span = set(range(Y0, Y1 + 1))
    res = []
    for wkey, mat in counts_by_window.items():
        w = ws[wkey]
        floods = envelope_search.gauge_consensus_years(lv, w["river"], w["season"], 3) & span
        severe = envelope_search.gauge_consensus_years(lv, w["river"], w["season"], SEVERE_RP) & span
        for k in range(1, N_MEMBERS + 1):
            votes = (mat >= k).sum(axis=1)
            days = votes[votes >= w["n_req"]]
            years = {d.year for d in days.index.get_level_values("valid_day")} & span
            hits = years & severe
            pod = len(hits) / len(severe) if severe else np.nan
            f1 = (2 * len(hits) / (len(years) + len(severe))
                  if (len(years) + len(severe)) else np.nan)
            res.append({"window": wkey, "leg": leg, "k": k, "share": k / N_MEMBERS,
                        "rp": w["rp"], "n_req": w["n_req"], "n_of": len(w["stations"]),
                        "n_years": len(years), "years": sorted(years),
                        "pod_severe": pod, "false_act": len(years - floods),
                        "f1_severe": round(f1, 3), "severe_n": len(severe),
                        "severe_years": sorted(severe), "flood_years": sorted(floods)})
    return pd.DataFrame(res)


def main():
    all_res = []
    for leg in ("action", "readiness"):
        ws = windows(leg)
        df = member_frame(leg)
        print(f"{leg}: {len(df):,} rows | leads {sorted(df.leadtime_days.unique())} | "
              f"members {df.member.nunique()} | issues {df.issued_time.dt.year.min()}-{df.issued_time.dt.year.max()}")
        counts = {wkey: over_counts(df, w) for wkey, w in ws.items()}
        all_res.append(score(counts, leg, ws))
    res = pd.concat(all_res, ignore_index=True)
    res.to_json(OUT / "prob_sweep_results.json", orient="records")
    for leg in ("action", "readiness"):
        print(f"\n=== {leg.upper()} (k of 11 members; median rule = k=6; leads {LEADS[leg]})")
        for wkey in res[res.leg == leg].window.unique():
            d = res[(res.window == wkey) & (res.leg == leg)]
            base = d[d.k == 6].iloc[0]
            print(f"\n{wkey}  ({base.n_req} of {base.n_of} points over 1-in-{base.rp}, severe n={base.severe_n}: {base.severe_years})")
            for _, r in d.iterrows():
                mark = " <- median" if r.k == 6 else ""
                print(f"  k={r.k:2d} ({r.share:.0%})  acts={r.n_years:2d} {r.years}  "
                      f"POD={r.pod_severe:.2f}  false={r.false_act}  F1={r.f1_severe:.2f}{mark}")


if __name__ == "__main__":
    main()
