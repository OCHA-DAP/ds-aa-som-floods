"""Choose the configuration whose ENVELOPE fires about once every three years.

The 1-in-3 target belongs to the envelope, not to each river-season window:
the money is released whenever any window fires, so that union is what the
budget is sized on. Four windows each calibrated to 1-in-3 give a union of
roughly 1-in-1.5, which is why the per-window settings have to be searched
against the union rather than set individually.

  objective   with the envelope pinned near 1-in-3, catch as many severe
              years as possible (severe = two or more of the river's gauges
              recorded a 1-in-5 or rarer season)
  tie-breaks  fewer activations in years with no recorded flood, then the
              lower vote requirement
  constraint  every window needs at least two gauges to agree (no single
              gauge releases the money) and never all of them (one quiet
              gauge cannot block it); station return periods stay at or
              below a quarter of the record length

Usage (from repo root):
    .venv/Scripts/python.exe scripts/envelope_search.py

Writes nothing. The chosen configuration is recorded in src/constants.py
(TRIGGER_CONFIG), which notebook 09 and the summary page both read.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import ocha_stratus as stratus  # noqa: E402

from src.constants import (  # noqa: E402
    ENVELOPE_TARGET_RP,
    REFERENCE_GAUGE,
    WINDOW_MODEL,
    SEASONS,
    SEVERE_RP,
    TRIGGER_CONFIG,
    TRIGGER_STATIONS,
    TRIGGER_YEARS,
)
from src.utils import weibull_level, weibull_threshold  # noqa: E402

PREFIX, STAGE = "ds-aa-som-floods/processed", "dev"
# Station return periods are capped at a quarter of the record: a 1-in-10
# threshold fitted to 25 annual maxima is extrapolation, not estimation.
RPS = [3, 4, 5, 6]
# At least two gauges must agree, so no single gauge can release the money,
# and never all of them, so one quiet gauge cannot block it.
MIN_VOTES = 2
Y0, Y1 = TRIGGER_YEARS
SPAN = set(range(Y0, Y1 + 1))
N = len(SPAN)
WINDOWS = [(r, s) for r in TRIGGER_STATIONS for s in SEASONS]


def load(name):
    return stratus.load_parquet_from_blob(f"{PREFIX}/{name}.parquet", stage=STAGE)


# ----------------------------------------------------- the gauge benchmark
# A river-season counts as a flood when at least BENCH_GAUGES of that river's
# gauges cross their own post-2000 return level in the same season (directive
# 2026-08-27). One reference gauge deciding the benchmark put too much weight
# on a single record.
BENCH_GAUGES = 2


# Benchmark validation, 2026-08-28. The two-gauge rule was checked against gauge
# availability and the EM-DAT / CERF record:
#   the five costliest years (2006, 2018, 2019, 2020, 2023) are all severe
#   1999-2001 cannot be assessed: the Juba had no gauge reporting and the
#     Shabelle one, so 22 of the 25 years are assessable
#   2021 is missed although EM-DAT records 400,000 affected: only Bardheere and
#     Belet Weyne crossed, and Belet Weyne's peak reads exactly bank full
#     (8.30 m), where the record is censored
#   2016 is flagged although EM-DAT records nobody affected
#   smaller single-gauge cases: 2013 (50,000 affected) and 2012 (32,200)
#   possible remedy for the censoring: count a reading at bank full as a
#     crossing at any return period, since the gauge cannot read higher
def gauge_consensus_years(lv, river, season, rp, n_req=BENCH_GAUGES):
    """Years in which at least n_req of the river's gauges crossed their own RP."""
    counts = {}
    for st in TRIGGER_STATIONS[river]:
        s = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
        s = s[s.index.month.isin(SEASONS[season])]
        modern = s[s.index.year >= 2000]
        am = modern.groupby(modern.index.year).max().dropna()
        if not len(am):
            continue
        level = weibull_level(am.values, rp)
        if np.isnan(level):
            continue
        for y in set(am[am >= level].index) & SPAN:
            counts[y] = counts.get(y, 0) + 1
    return {y for y, n in counts.items() if n >= n_req}


def benchmark_years_from_gauges(lv, flood_rp=3, severe_rp=SEVERE_RP):
    """(flood years, severe years) across every window, on the consensus rule."""
    flood, severe = set(), set()
    for river, season in WINDOWS:
        flood |= gauge_consensus_years(lv, river, season, flood_rp)
        severe |= gauge_consensus_years(lv, river, season, severe_rp)
    return flood, severe


def window_options(dd, bench):
    """{(river, season): {(station_rp, votes): frozenset(activation years)}}."""
    out = {}
    for river, season in WINDOWS:
        model, months = WINDOW_MODEL[(river, season)], SEASONS[season]
        opts, series = {}, {}
        for st in TRIGGER_STATIONS[river]:
            s = dd[(dd.src == model) & (dd.station == st)].set_index("date")["discharge"]
            s = s[s.index.month.isin(months)]
            s = s[(s.index.year >= Y0) & (s.index.year <= Y1)].sort_index()
            if len(s) >= 100:
                series[st] = s
        for rp in RPS:
            cols = []
            for st, s in series.items():
                am = s.groupby(s.index.year).max().dropna()
                t = weibull_threshold(am.values, rp)
                if not np.isnan(t):
                    cols.append((s >= t).rename(st))
            if not cols:
                continue
            mat = pd.concat(cols, axis=1).fillna(False)
            mx = mat.sum(axis=1).groupby(mat.index.year).max()
            for n in range(1, len(cols) + 1):
                opts[(rp, n)] = frozenset(set(mx[mx >= n].index) & SPAN)
        out[(river, season)] = opts
    return out


def benchmark_years(bench):
    """(years with any RP3 flood, years with a severe flood) across the windows."""
    any_flood, severe = set(), set()
    for river, season in WINDOWS:
        b = bench[(bench.river == river) & (bench.season == season)
                  & (bench.benchmark == f"swalim_{REFERENCE_GAUGE[river]}")]
        any_flood |= set(b[b.flood_3yr == 1].year) & SPAN
        severe |= set(b[b.rp >= SEVERE_RP].year) & SPAN
    return any_flood, severe


def evaluate(options, combo, keys, any_flood, severe):
    union = set()
    for k, setting in zip(keys, combo):
        union |= options[k][setting]
    return {
        "fires": len(union),
        "env_rp": round((N + 1) / len(union), 1) if union else None,
        "severe_caught": len(union & severe),
        "severe_missed": sorted(severe - union),
        "no_flood_years": len(union - any_flood),
        "votes": sum(n for _, n in combo),
        "years": sorted(union),
    }


def search(options, any_flood, severe, robust=True):
    """Best configuration at every activation count."""
    keys = list(options)

    def allowed(key, setting):
        n_gauges = len(TRIGGER_STATIONS[key[0]])
        if not robust:
            return True
        return MIN_VOTES <= setting[1] <= n_gauges - 1

    best = {}
    for combo in itertools.product(
        *[[o for o in options[k] if allowed(k, o)] for k in keys]
    ):
        r = evaluate(options, combo, keys, any_flood, severe)
        if not r["fires"]:
            continue
        score = (r["severe_caught"], -r["no_flood_years"], r["votes"],
                 -sum(rp for rp, _ in combo))
        if r["fires"] not in best or score > best[r["fires"]][0]:
            best[r["fires"]] = (score, combo, r)
    return keys, best


def main():
    print("loading ...")
    bench = load("workflow/som_flood_benchmark_seasonal")
    models = sorted(set(WINDOW_MODEL.values()))
    dd = pd.concat([load(f"discharge_daily_{m}").assign(src=m) for m in models],
                   ignore_index=True)
    dd["date"] = pd.to_datetime(dd["date"])

    options = window_options(dd, bench)
    any_flood, severe = benchmark_years(bench)
    keys, best = search(options, any_flood, severe)

    print(f"\n{N} years ({Y0}-{Y1}) | {len(any_flood)} with a flood at RP3 or rarer "
          f"| {len(severe)} severe (RP{SEVERE_RP}+): {sorted(severe)}")
    print(f"envelope target 1-in-{ENVELOPE_TARGET_RP} = about "
          f"{(N + 1) / ENVELOPE_TARGET_RP:.1f} activations in {N} years\n")

    print("frontier: the best the envelope can do at each activation rate")
    print(f"{'fires':>5} {'env RP':>7} {'severe':>8} {'no-flood':>9}  configuration")
    for k in sorted(best):
        _, combo, r = best[k]
        if not (1.4 <= r["env_rp"] <= 6.5):
            continue
        cfg = " | ".join(
            f"{river[:4]}-{season[:2]} RP{rp} {n}of{len(TRIGGER_STATIONS[river])}"
            for (river, season), (rp, n) in zip(keys, combo)
        )
        print(f"{k:>5} {r['env_rp']:>7} {r['severe_caught']:>4}/{len(severe):<3} "
              f"{r['no_flood_years']:>9}  {cfg}")

    near = [k for k in best if abs((N + 1) / k - ENVELOPE_TARGET_RP) <= 0.45]
    pick = min(near, key=lambda k: (abs((N + 1) / k - ENVELOPE_TARGET_RP), -k))
    _, combo, r = best[pick]
    print(f"\nchosen: 1-in-{r['env_rp']} envelope, "
          f"{r['severe_caught']} of {len(severe)} severe years caught, "
          f"{r['no_flood_years']} activations in years with no recorded flood")
    for (river, season), (rp, n) in zip(keys, combo):
        yrs = sorted(options[(river, season)][(rp, n)])
        print(f"  {river:9s} {season:5s} station RP {rp:>2} | {n} of "
              f"{len(TRIGGER_STATIONS[river])} gauges | fires {yrs}")
    print(f"  envelope years: {r['years']}")
    print(f"  severe years missed: {r['severe_missed']}")

    # a uniform rule is easier to operate: report what it costs
    print("\nsimpler alternatives, for comparison")
    for name, spec in {
        "uniform RP5, majority of gauges": {r_: (5, len(TRIGGER_STATIONS[r_]) - 1)
                                            for r_ in TRIGGER_STATIONS},
        "uniform RP6, majority of gauges": {r_: (6, len(TRIGGER_STATIONS[r_]) - 1)
                                            for r_ in TRIGGER_STATIONS},
        "uniform RP5, two gauges": {r_: (5, 2) for r_ in TRIGGER_STATIONS},
    }.items():
        combo = tuple(spec[river] for river, _ in keys)
        if any(c not in options[k] for k, c in zip(keys, combo)):
            continue
        r = evaluate(options, combo, keys, any_flood, severe)
        print(f"  {name:34s} 1-in-{r['env_rp']:<4} "
              f"severe {r['severe_caught']}/{len(severe)} | "
              f"no-flood {r['no_flood_years']} | years {r['years']}")

    # and what the configuration currently in constants.py does
    stored = tuple((TRIGGER_CONFIG[k]["rp"], TRIGGER_CONFIG[k]["n_req"])
                   for k in keys if k in TRIGGER_CONFIG)
    if len(stored) == len(keys) and all(c in options[k] for k, c in zip(keys, stored)):
        r = evaluate(options, stored, keys, any_flood, severe)
        print(f"\nstored in constants.py: 1-in-{r['env_rp']} envelope | "
              f"severe {r['severe_caught']}/{len(severe)} | "
              f"no-flood years {r['no_flood_years']}")


if __name__ == "__main__":
    main()
