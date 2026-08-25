"""Bake the indicator-explorer data for the Pages site.

Usage (from repo root):
    .venv/bin/python scripts/export_explorer_data.py

Writes pages/explorer/data/{juba,shabelle}.json: per basin and year
(1999-2023), daily series of
  - the SWALIM reference-gauge level (full year),
  - per source, the count of the season's selected stations whose
    discharge exceeds their own seasonal 1-in-6 Weibull threshold
    (season months only - the trigger only exists in season), and
  - per source, discharge at the reference station divided by that
    source's own seasonal 1-in-6 threshold there.

Every source - GEOGloWS included, via its retrospective - is thresholded
against ITS OWN historical values (1999-2023 seasonal maxima), the same
convention as the mechanism itself: we assume bias between reanalysis
and forecast (and between models) for ALL sources, so absolute
magnitudes are never compared across products.

Inputs are the local processed tables (scripts/restore_from_blob.py).
The station sets and consensus requirements mirror
processed/workflow/som_ms_trigger_config.parquet.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import SEASONS  # noqa: E402
from src.utils import weibull_threshold  # noqa: E402

DATA = repo_root / "data" / "processed"
OUT = repo_root / "pages" / "explorer" / "data"
YEARS = (1999, 2023)
RP = 6
SOURCES = ["geoglows", "glofas_v5", "google_grrr"]
REF_GAUGE = {"shabelle": "belet_weyne", "juba": "luuq"}
GAUGE_BM = {"shabelle": "swalim_belet_weyne", "juba": "swalim_luuq"}


def main():
    cfg = pd.read_parquet(DATA / "workflow" / "som_ms_trigger_config.parquet")
    legs = pd.read_parquet(DATA / "workflow" / "som_ms_trigger_legs.parquet")
    bench = pd.read_parquet(DATA / "workflow" / "som_flood_benchmark_seasonal.parquet")
    lv = pd.read_parquet(DATA / "swalim_levels.parquet")
    lv["date"] = pd.to_datetime(lv["date"])
    sw_th = pd.read_parquet(DATA / "swalim_thresholds.parquet").set_index("station")
    dd = pd.concat(
        [pd.read_parquet(DATA / f"discharge_daily_{s}.parquet").assign(file_source=s)
         for s in SOURCES],
        ignore_index=True,
    )
    dd["date"] = pd.to_datetime(dd["date"])

    OUT.mkdir(parents=True, exist_ok=True)
    for basin in ["juba", "shabelle"]:
        out = {
            "basin": basin,
            "ref_gauge": REF_GAUGE[basin],
            "swalim_thresholds": {
                "moderate": round(float(sw_th.loc[REF_GAUGE[basin], "moderate_flood_risk"]), 2),
                "high": round(float(sw_th.loc[REF_GAUGE[basin], "high_flood_risk"]), 2),
            },
            "seasons": {s: SEASONS[s] for s in ("gu", "deyr")},
            "rp": RP,
            "sources": SOURCES,
        }
        # station sets, requirements and adopted sources per season
        sets, n_req, adopted_src, act_years = {}, {}, {}, {}
        for season in ("gu", "deyr"):
            c = cfg[(cfg.river == basin) & (cfg.season == season)]
            # the config is (station, model) pairs — the explorer shows counts
            # per model over the window's DISTINCT stations
            sets[season] = list(dict.fromkeys(c.station))
            n_req[season] = int(c.n_pairs_required.iloc[0])
            leg = legs[(legs.river == basin) & (legs.season == season)].iloc[0]
            # mixed-model windows: the tick line uses the window's majority model
            adopted_src[season] = c.source.value_counts().idxmax()
            ys = str(leg.activation_years)
            act_years[season] = sorted({int(y) for y in ys.split(",") if y.strip().isdigit()})
        out.update(stations=sets, n_req=n_req, adopted_source=adopted_src,
                   activation_years=act_years)

        b = bench[(bench.river == basin) & (bench.benchmark == GAUGE_BM[basin])]
        out["benchmark"] = {
            "moderate_years": sorted(int(y) for y in b[b.flood_3yr == 1].year.unique()
                                     if YEARS[0] <= y <= YEARS[1]),
            "severe_years": sorted(int(y) for y in b[b.flood_5yr == 1].year.unique()
                                   if YEARS[0] <= y <= YEARS[1]),
        }

        # per (source, season, station): daily series + own RP6 threshold
        daily, ths = {}, {}
        for season in ("gu", "deyr"):
            months = SEASONS[season]
            for src in SOURCES:
                for stn in sets[season]:
                    s = dd[(dd.file_source == src) & (dd.station == stn)].set_index("date")["discharge"]
                    s = s[s.index.month.isin(months)]
                    s = s[(s.index.year >= YEARS[0]) & (s.index.year <= YEARS[1])]
                    am = s.groupby(s.index.year).max().dropna()
                    daily[(src, season, stn)] = s
                    ths[(src, season, stn)] = (
                        weibull_threshold(am.values, RP) if len(am) else np.nan
                    )
        out["ref_thresholds"] = {
            src: {season: (None if np.isnan(ths[(src, season, REF_GAUGE[basin])])
                           else round(float(ths[(src, season, REF_GAUGE[basin])]), 1))
                  for season in ("gu", "deyr")}
            for src in SOURCES
        }

        sw = lv[lv.station == REF_GAUGE[basin]].set_index("date")["level_m"]

        years_out = {}
        for year in range(YEARS[0], YEARS[1] + 1):
            idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            swv = sw.reindex(idx)

            def season_of(ts):
                for season in ("gu", "deyr"):
                    if ts.month in SEASONS[season]:
                        return season
                return None

            counts = {src: [] for src in SOURCES}
            ratio = {src: [] for src in SOURCES}
            for src in SOURCES:
                # pre-index the year's values per station for speed
                cache = {}
                for season in ("gu", "deyr"):
                    for stn in sets[season]:
                        s = daily[(src, season, stn)]
                        cache[(season, stn)] = s[s.index.year == year]
                for ts in idx:
                    season = season_of(ts)
                    if season is None:
                        counts[src].append(None)
                        ratio[src].append(None)
                        continue
                    n = 0
                    seen = False
                    for stn in sets[season]:
                        t = ths[(src, season, stn)]
                        v = cache[(season, stn)].get(ts, np.nan)
                        if not np.isnan(v) and not np.isnan(t):
                            seen = True
                            if v >= t:
                                n += 1
                    counts[src].append(n if seen else None)
                    rg = REF_GAUGE[basin]
                    t = ths[(src, season, rg)]
                    v = cache[(season, rg)].get(ts, np.nan)
                    ratio[src].append(
                        None if (np.isnan(v) or np.isnan(t) or t == 0)
                        else round(float(v / t), 3)
                    )
            years_out[str(year)] = {
                "swalim": [None if np.isnan(v) else round(float(v), 2) for v in swv],
                "counts": counts,
                "ratio": ratio,
            }
        out["years"] = years_out

        path = OUT / f"{basin}.json"
        path.write_text(json.dumps(out, separators=(",", ":")))
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
