"""Old vs new Google model (GRRR) under the adopted Gu windows.

Google released a new version of its Runoff Reanalysis & Reforecast in
November 2026 (gs://flood-forecasting/hydrologic_predictions/
distributed_model_2026_11, a semi-distributed model on the HydroBASINS river
graph). The Gu windows were designed on the earlier version
(model_id_8583a5c2_v0, data/processed/*_google_grrr.parquet). This module
writes the new version into the same processed schema under the source name
``google_grrr_2026_11`` and replays the adopted rules on both with the method
held fixed (scripts/page_additions/somlib.py, unchanged): same seven points,
Gu months Mar-May, levels fitted on 1999-2023 seasonal maxima by Weibull
plotting position, Juba 3 of 4 over 1-in-5, Shabelle 2 of 3 over 1-in-6,
forecast leads 1-7 with valid_time = issued_time + leadtime.

    .venv/bin/python scripts/google_model_update.py fetch   # once, ~1 min
    .venv/bin/python scripts/google_model_update.py         # print the comparison
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GRPC_VERBOSITY", "NONE")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "page_additions"))

import somlib as L  # noqa: E402
from src.constants import (ALL_TRIGGER_STATIONS, SEASONS, STATIONS,  # noqa: E402
                           TRIGGER_CONFIG, TRIGGER_STATIONS)
from src.utils import weibull_threshold  # noqa: E402

OLD, NEW = "google_grrr", "google_grrr_2026_11"
NEW_STORE = "gs://flood-forecasting/hydrologic_predictions/distributed_model_2026_11"
GU = [("juba", "gu"), ("shabelle", "gu")]
DEYR = [("juba", "deyr"), ("shabelle", "deyr")]
EXT = list(range(L.Y0, 2026))  # the new record runs to 2025; levels stay fitted on 1999-2023


def _open(url):
    import zarr

    return zarr.open_group(url, mode="r", storage_options={"token": "anon"})


def _decode_time(arr):
    unit, origin = arr.attrs["units"].split(" since ")
    return pd.to_datetime(arr[:], unit={"days": "D"}[unit.strip()], origin=pd.Timestamp(origin.strip()))


def fetch():
    """Write the new version for the trigger points in the processed schema.

    The new store names the same HydroBASINS level-12 basins subhybas_<id>
    (outlet coordinates identical to the old hybas_<id>)."""
    sid = {st: STATIONS[st].grrr_gauge_id.replace("hybas_", "subhybas_") for st in ALL_TRIGGER_STATIONS}
    g = _open(f"{NEW_STORE}/reanalysis/streamflow.zarr")
    pos = {str(x): i for i, x in enumerate(g["gauge_id"][:]) if str(x) in set(sid.values())}
    time = _decode_time(g["time"])
    daily = pd.concat([pd.DataFrame({"station": st, "river": STATIONS[st].river, "source": "google",
                                     "date": time, "discharge": g["streamflow"][pos[s], :]})
                       for st, s in sid.items()], ignore_index=True)
    g = _open(f"{NEW_STORE}/reforecast/streamflow.zarr")
    issue, leads = _decode_time(g["issue_time"]), np.asarray(g["lead_time"][:]).astype(int)
    frames = []
    for st, s in sid.items():
        df = (pd.DataFrame(g["streamflow"][pos[s], :, :], index=pd.Index(issue, name="issued_time"),
                           columns=pd.Index(leads, name="leadtime_days"))
              .stack(future_stack=True).rename("discharge").reset_index())
        df["valid_time"] = df["issued_time"] + pd.to_timedelta(df["leadtime_days"], unit="D")
        df["station"], df["river"], df["source"], df["member"] = st, STATIONS[st].river, "google", 0
        frames.append(df[["station", "river", "source", "issued_time", "valid_time", "leadtime_days",
                          "member", "discharge"]])
    out = L.P
    daily.to_parquet(out / f"discharge_daily_{NEW}.parquet")
    pd.concat(frames, ignore_index=True).to_parquet(out / f"reforecast_{NEW}.parquet")
    print(f"wrote {out}/discharge_daily_{NEW}.parquet, reforecast_{NEW}.parquet")


def votes(model, river, season, rp):
    """Daily count of the window's points at or over their own level (all years)."""
    m = L.daily_matrix(model, river, season)
    thr = L.model_thresholds(model, river, season, rp)
    cols = [c for c in thr.index if c in m.columns and not np.isnan(thr[c])]
    return (m[cols] >= thr[cols]).sum(axis=1)


def margin(model, river, season, rp, n_req):
    """Daily flow of the n_req-th highest point as a share of its own level:
    the window's rule is met on days this reaches 1."""
    m = L.daily_matrix(model, river, season)
    thr = L.model_thresholds(model, river, season, rp)
    ratio = m[thr.index] / thr
    return ratio.apply(lambda r: np.sort(r.dropna().values)[::-1][n_req - 1]
                       if r.notna().sum() >= n_req else np.nan, axis=1)


def forecast_years(model, river, season, rp, n_req, span):
    """Years in which one forecast (leads 1-7) put n_req points over their level."""
    return sorted(L.first_issue_dates(model, river, season, rp, n_req, span=span))


def best_lag_rho(model_s, gauge_s):
    """Copied from scripts/page_additions/station_metrics.py (which runs its whole
    table on import): best Spearman over lags L.MIN_LAG..L.MAX_LAG, shifting the
    gauge's FULL-YEAR record by calendar days."""
    best = 0.0
    for lag in range(L.MIN_LAG, L.MAX_LAG + 1):
        g = gauge_s.copy()
        g.index = g.index - pd.Timedelta(days=lag)
        j = pd.concat([model_s, g], axis=1, join="inner").dropna()
        if len(j) < L.MIN_OBS:
            continue
        r = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
        if pd.notna(r) and abs(r) > abs(best):
            best = float(r)
    return best


def tracking(model, river, season):
    """Best-lag rank correlation of each point with its SWALIM gauge in the
    season, exactly as in the trigger page's station table
    (scripts/page_additions/station_metrics.py)."""
    lv = L.levels()
    out = {}
    for st in TRIGGER_STATIONS[river]:
        g_all = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
        s = L.season_series(model, st, season)
        if len(s) < 100 or not len(g_all):
            continue
        out[st] = best_lag_rho(s, g_all)
    return out


def mismatch_years(river, season, rp, n_req):
    """What the live window would do if Google switched the API to the new model
    before the frozen (old-model) levels in src/monitoring/thresholds.json were
    replaced: old levels on new flows, every year of the new record."""
    m = L.daily_matrix(NEW, river, season)
    thr = L.model_thresholds(OLD, river, season, rp)
    mx = (m[thr.index] >= thr).sum(axis=1)
    mx = mx.groupby(mx.index.year).max()
    return sorted(set(mx[mx >= n_req].index) & set(EXT))


def season_means(model, river, season):
    """Mean in-season flow per point over the calibration years."""
    return pd.Series({st: L.season_series(model, st, season).mean() for st in TRIGGER_STATIONS[river]})


def run():
    res = {}
    span_rf_old = [y for y in L.SPAN if y >= 2016]
    for w in GU:
        c = TRIGGER_CONFIG[w]
        r = {}
        for model in (OLD, NEW):
            r[model] = {
                "levels": L.model_thresholds(model, *w, c["rp"]),
                "years": sorted(L.activation_years(model, *w, c["rp"], c["n_req"])),
                "years_ext": sorted(L.activation_years(model, *w, c["rp"], c["n_req"], span=EXT)),
                "first_cross": L.first_crossing_dates(model, *w, c["rp"], c["n_req"], span=EXT),
                "first_issue": L.first_issue_dates(model, *w, c["rp"], c["n_req"], span=EXT),
                "votes": votes(model, *w, c["rp"]),
                "margin": margin(model, *w, c["rp"], c["n_req"]),
                "tracking": tracking(model, *w),
                "sweep": {rp: sorted(L.activation_years(model, *w, rp, c["n_req"])) for rp in (3, 4, 5, 6)},
                "means": season_means(model, *w),
            }
        r["bench"] = L.benchmark(*w)
        # the gauge benchmark past the calibration record (levels still fitted 2000-2023)
        r["bench_ext"] = (set(L.gauge_crossings(*w, 3, span=EXT)), set(L.gauge_crossings(*w, 5, span=EXT)),
                          L.assessable_years(*w, span=EXT))
        r["mismatch"] = mismatch_years(*w, c["rp"], c["n_req"])
        res[w] = r
    # Deyr windows run on GloFAS (design: v5) and do not change
    res["deyr"] = {w: sorted(L.activation_years(TRIGGER_CONFIG[w]["source"], *w, TRIGGER_CONFIG[w]["rp"],
                                                TRIGGER_CONFIG[w]["n_req"])) for w in DEYR}
    res["deyr_bench"] = {w: L.benchmark(*w) for w in DEYR}
    res["rf_span_old"] = span_rf_old
    # the old replay must reproduce the adopted Gu years
    assert res[("juba", "gu")][OLD]["years"] == [2013, 2016, 2018, 2020]
    assert res[("shabelle", "gu")][OLD]["years"] == [2013, 2016, 2018, 2020]
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
        sys.stdout.flush()
        os._exit(0)
    res = run()
    fmt = lambda d: {y: (str(a.date()) if not isinstance(a, tuple) else (str(a[0].date()), str(a[1].date()))) for y, a in d.items()}
    for w in GU:
        r = res[w]
        print(f"\n=== {w} rule {TRIGGER_CONFIG[w]}  bench flood/severe/assessable: {sorted(r['bench'][0])} / {sorted(r['bench'][1])} / n={len(r['bench'][2])}")
        for m in (OLD, NEW):
            x = r[m]
            print(f"  {m}: levels {x['levels'].round(0).to_dict()}")
            print(f"     years {x['years']}  ext {x['years_ext']}")
            print(f"     first cross {fmt(x['first_cross'])}")
            print(f"     first issue {fmt(x['first_issue'])}")
            print(f"     tracking {({k: round(v, 2) for k, v in x['tracking'].items()})} median {np.median(list(x['tracking'].values())):.2f}")
            print(f"     sweep {x['sweep']}")
            print(f"     mean flow {x['means'].round(0).to_dict()}")
            mx = x['votes'].groupby(x['votes'].index.year).max()
            print(f"     max votes by year {mx.loc[1999:].to_dict()}")
    print("\nDeyr (GloFAS, unchanged):", res["deyr"])
    for w in GU:
        b = res[w]["bench_ext"]
        print(w, "MISMATCH old levels on new flows:", res[w]["mismatch"],
              "| bench 2024-25 flood", sorted(y for y in b[0] if y > 2023), "severe", sorted(y for y in b[1] if y > 2023),
              "assessable", sorted(y for y in b[2] if y > 2023))
    sys.stdout.flush()
    os._exit(0)
