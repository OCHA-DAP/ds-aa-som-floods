"""Pick ONE model per window, following the trigger page's selection logic.

The published pages/trigger design lets three models vote inside a single
window, so the same station can cast two or three votes and three providers
each hold a veto. This keeps that page's logic (rank the stations by how well
the model tracks the reference gauge, take the strongest few, require a
consensus of them over their own return-period threshold, then judge the
union of the four windows) but forbids mixing: each window runs on exactly
one model.

Search space per window (river, season):
    model    google_grrr | glofas_v4 | glofas_v5 | geoglows
    stations the top k by best-lag rank correlation, k = 2 .. all
    threshold station return period 3, 4, 5 or 6
    votes    2 .. k

The four windows are then searched jointly, because the envelope (any window
fires) is what the 1-in-3 target applies to. Two variants are reported:

    free       a model chosen per river-season: 4 choices
    per-river  one model per river, used in both its seasons: 2 choices

Usage (from repo root):
    .venv/Scripts/python.exe scripts/model_selection.py

Writes nothing: it prints the selection so the choice can be recorded in
src/constants.py deliberately.
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
    SEASONS,
    SEVERE_RP,
    TRIGGER_STATIONS,
    TRIGGER_YEARS,
)
from src.utils import weibull_threshold  # noqa: E402

PREFIX, STAGE = "ds-aa-som-floods/processed", "dev"
MODELS = ["google_grrr", "glofas_v4", "glofas_v5", "geoglows"]
# No model is excluded for want of a forecast archive (directive 2026-08-27).
# GloFAS runs operationally as v5, so v5 reanalysis is the right basis for its
# thresholds and the v4 reforecast supplies the lead-time evidence. What each
# choice does carry is a note on where its thresholds come from and what can
# be verified at lead time.
NOTES = {
    "glofas_v5": "thresholds from v5 reanalysis, operational forecast is v5, "
                 "lead-time evidence from the v4 reforecast",
    "glofas_v4": "thresholds and lead-time evidence both from v4, which is no "
                 "longer the operational version",
    "google_grrr": "thresholds from the retrospective, 2016-2023 reforecast, "
                   "7-day horizon",
    "geoglows": "forecasts run below its own retrospective, so thresholds must "
                "be refitted on the forecast archive before operating",
}
RPS = [3, 4, 5, 6]
MIN_VOTES = 2
# A window that fires once in 25 years is fitted to that one event, not
# calibrated: it cannot be distinguished from luck and it tells an operator
# nothing about the years in between. Every window must fire at least twice.
MIN_WINDOW_FIRES = 2
MIN_LAG, MAX_LAG, MIN_OBS = -10, 30, 60
Y0, Y1 = TRIGGER_YEARS
SPAN = set(range(Y0, Y1 + 1))
N_YEARS = len(SPAN)
WINDOWS = [(r, s) for r in TRIGGER_STATIONS for s in SEASONS]
NICE = {"google_grrr": "Google", "glofas_v4": "GloFAS v4",
        "glofas_v5": "GloFAS v5", "geoglows": "GEOGloWS"}


def load(name):
    return stratus.load_parquet_from_blob(f"{PREFIX}/{name}.parquet", stage=STAGE)


def season_series(dd, model, station, months, span=None):
    span = span or SPAN
    s = dd[(dd.src == model) & (dd.station == station)].set_index("date")["discharge"]
    s = s[s.index.month.isin(months)]
    return s[s.index.year.isin(span)].sort_index()


def best_lag_rho(model_series, gauge_series):
    """Strongest rank correlation over plausible travel times."""
    best = 0.0
    for lag in range(MIN_LAG, MAX_LAG + 1):
        # CORRECTION (branch fix/corrected-benchmark-and-lag): .shift(-lag) moves by
        # ROW POSITION, and the series has already been filtered to season months, so
        # every non-zero lag pulled rows from the adjacent year and always scored
        # worse. That forced best-lag to 0 almost everywhere. Shift by calendar days.
        shifted = gauge_series.copy()
        shifted.index = shifted.index - pd.Timedelta(days=lag)
        j = pd.concat([model_series, shifted], axis=1, join="inner").dropna()
        if len(j) < MIN_OBS:
            continue
        r = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
        if pd.notna(r) and abs(r) > abs(best):
            best = float(r)
    return best


def benchmark_years(bench, river, season):
    b = bench[(bench.river == river) & (bench.season == season)
              & (bench.benchmark == f"swalim_{REFERENCE_GAUGE[river]}")]
    return (set(b[b.flood_3yr == 1].year) & SPAN, set(b[b.rp >= SEVERE_RP].year) & SPAN)


def window_candidates(dd, lv, river, season, models=None, span=None, rps=None):
    """Every single-model configuration for one window, keyed by activation set.

    Stations are ranked by best-lag correlation for that model, as the trigger
    page does, and only the top k are used. Configurations that activate in
    exactly the same years are collapsed, keeping the most defensible one:
    an operable model, then fewer votes required relative to the pool, then
    the lower station return period.
    """
    models = models or MODELS
    span = span or SPAN
    rps = rps or RPS
    months = SEASONS[season]
    ref = lv[lv.station == REFERENCE_GAUGE[river]].set_index("date")["level_m"]
    ref = ref[ref.index.month.isin(months)].sort_index()
    pool = TRIGGER_STATIONS[river]
    out = {}
    for model in models:
        series, rho = {}, {}
        for st in pool:
            s = season_series(dd, model, st, months, span=span)
            if len(s) < 100:
                continue
            series[st] = s
            rho[st] = best_lag_rho(s, ref)
        if len(series) < MIN_VOTES:
            continue
        # every gauge in the pool is monitored (directive 2026-08-27): the
        # four Juba points and the three Shabelle points, ranked here only so
        # the reported order runs strongest first
        ranked = sorted(series, key=lambda st: -rho[st])
        for k in [len(ranked)]:
            chosen = ranked[:k]
            for rp in rps:
                cols = []
                for st in chosen:
                    s = series[st]
                    am = s.groupby(s.index.year).max().dropna()
                    t = weibull_threshold(am.values, rp)
                    if not np.isnan(t):
                        cols.append((s >= t).rename(st))
                if len(cols) < MIN_VOTES:
                    continue
                mat = pd.concat(cols, axis=1).fillna(False)
                mx = mat.sum(axis=1).groupby(mat.index.year).max()
                # never all of them: one quiet gauge must not block the trigger
                for n in range(MIN_VOTES, max(MIN_VOTES, len(cols) - 1) + 1):
                    years = frozenset(set(mx[mx >= n].index) & span)
                    cand = {
                        "model": model,
                        "stations": list(chosen),
                        "rp": rp,
                        "n_req": n,
                        "years": years,
                        "rho": round(float(np.mean([rho[st] for st in chosen])), 2),
                        "unanimous": n == len(cols),
                    }
                    key = years
                    rank = (cand["unanimous"], rp, -cand["rho"])
                    if key not in out or rank < out[key][0]:
                        out[key] = (rank, cand)
    return [c for _, c in out.values()]


def evaluate(combo, any_flood, severe, span=None):
    span = span or SPAN
    n = len(span)
    union = set().union(*[set(c["years"]) for c in combo])
    return {
        "fires": len(union),
        "n_years": n,
        "env_rp": round((n + 1) / len(union), 1) if union else None,
        "severe_caught": len(union & severe & span),
        "severe_missed": sorted((severe & span) - union),
        "no_flood_years": sorted(union - any_flood),
        "years": sorted(union),
        "n_severe": len(severe & span),
    }


def choose(cands, any_flood, severe, per_river=False, span=None, min_rp=None):
    """Joint search over the four windows for an envelope near the target."""
    span = span or SPAN
    keys = list(cands)
    best = None
    frontier = {}
    for combo in itertools.product(*[cands[k] for k in keys]):
        if any(len(c["years"]) < MIN_WINDOW_FIRES for c in combo):
            continue
        if per_river:
            by_river = {}
            ok = True
            for (river, _), c in zip(keys, combo):
                if by_river.setdefault(river, c["model"]) != c["model"]:
                    ok = False
                    break
            if not ok:
                continue
        r = evaluate(combo, any_flood, severe, span=span)
        if not r["fires"]:
            continue
        # frontier: the best severe-year coverage at each activation count
        score = (r["severe_caught"], -len(r["no_flood_years"]),
                 -sum(c["unanimous"] for c in combo),
                 # tracking correlation breaks remaining ties: an indifferent
                 # envelope must not hand a window to the worst tracker
                 round(sum(c["rho"] for c in combo), 2),
                 -sum(c["rp"] for c in combo))
        if r["fires"] not in frontier or score > frontier[r["fires"]][0]:
            frontier[r["fires"]] = (score, combo, r)
        # min_rp: accept only rates at or rarer than this, for when the target
        # must not be exceeded (the record quantises the achievable rates)
        near = abs(r["env_rp"] - ENVELOPE_TARGET_RP) <= 0.45
        if min_rp is not None:
            near = r["env_rp"] >= min_rp - 0.01 and r["env_rp"] <= min_rp + 0.6
        if near:
            pick_score = (r["severe_caught"], -len(r["no_flood_years"]),
                          -abs(r["env_rp"] - ENVELOPE_TARGET_RP),
                          -sum(c["unanimous"] for c in combo))
            if best is None or pick_score > best[0]:
                best = (pick_score, combo, r)
    return keys, best, frontier


def report(title, keys, best, severe, n_severe):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if best is None:
        print("  no configuration lands within 0.45 of the target rate")
        return
    _, combo, r = best
    print(f"envelope 1-in-{r['env_rp']} ({r['fires']} of {N_YEARS} years) | "
          f"severe {r['severe_caught']}/{n_severe} | "
          f"fired with no recorded flood: {r['no_flood_years'] or 'never'}")
    for (river, season), c in zip(keys, combo):
        print(f"  {river:9s} {season:5s} {NICE[c['model']]:11s} "
              f"{c['n_req']} of {len(c['stations'])} at 1-in-{c['rp']} | "
              f"rho {c['rho']} | {', '.join(c['stations'])}")
        print(f"  {'':9s} {'':5s} fires {sorted(c['years'])}")
    print(f"  envelope years: {r['years']}")
    print(f"  severe missed:  {r['severe_missed']}")
    per_river_rp = {}
    for (river, _), c in zip(keys, combo):
        per_river_rp.setdefault(river, set()).update(c["years"])
    for river, yrs in per_river_rp.items():
        print(f"  {river} basin: 1-in-{(N_YEARS + 1) / len(yrs):.1f} ({len(yrs)} years)"
              if yrs else f"  {river}: never")


def ties(cands, any_flood, severe, per_river=False, span=None):
    """Every model assignment that reaches the best score, to show the slack.

    If many assignments tie, the model choice is not what the result rests on,
    and that is worth knowing before defending one of them.
    """
    span = span or SPAN
    keys = list(cands)
    best_score, by_assignment = None, {}
    for combo in itertools.product(*[cands[k] for k in keys]):
        if any(len(c["years"]) < MIN_WINDOW_FIRES for c in combo):
            continue
        if per_river:
            seen = {}
            if any(seen.setdefault(r, c["model"]) != c["model"]
                   for (r, _), c in zip(keys, combo)):
                continue
        r = evaluate(combo, any_flood, severe, span=span)
        if not r["fires"] or abs(r["env_rp"] - ENVELOPE_TARGET_RP) > 0.45:
            continue
        score = (r["severe_caught"], -len(r["no_flood_years"]))
        if best_score is None or score > best_score:
            best_score, by_assignment = score, {}
        if score == best_score:
            if per_river:
                assign = tuple(sorted({(river, c["model"])
                                       for (river, _), c in zip(keys, combo)}))
            else:
                assign = tuple((f"{river}-{season}", c["model"])
                               for (river, season), c in zip(keys, combo))
            by_assignment.setdefault(assign, []).append(r["env_rp"])
    return best_score, by_assignment


def main():
    print("loading ...")
    bench = load("workflow/som_flood_benchmark_seasonal")
    lv = load("swalim_levels")
    lv["date"] = pd.to_datetime(lv["date"])
    dd = pd.concat([load(f"discharge_daily_{m}").assign(src=m) for m in MODELS],
                   ignore_index=True)
    dd["date"] = pd.to_datetime(dd["date"])

    any_flood, severe = set(), set()
    for river, season in WINDOWS:
        af, sv = benchmark_years(bench, river, season)
        any_flood |= af
        severe |= sv

    print("building single-model candidates per window ...")
    cands = {}
    for river, season in WINDOWS:
        cands[(river, season)] = window_candidates(dd, lv, river, season)
        print(f"  {river:9s} {season:5s} {len(cands[(river, season)]):3d} distinct "
              "configurations")

    print(f"\n{N_YEARS} years | {len(severe)} severe (1-in-{SEVERE_RP}+): "
          f"{sorted(severe)}")
    print(f"target: envelope 1-in-{ENVELOPE_TARGET_RP}")

    for title, per_river in [
        ("A. one model chosen per river-season: 4 choices", False),
        ("B. one model per river, used in both seasons: 2 choices", True),
    ]:
        keys, best, _ = choose(cands, any_flood, severe, per_river=per_river)
        report(title, keys, best, severe, len(severe))
        score, assigns = ties(cands, any_flood, severe, per_river=per_river)
        if score:
            print(f"  model assignments reaching {score[0]}/{len(severe)} severe with "
                  f"{-score[1]} no-flood activations: {len(assigns)}")
            for a in sorted(assigns)[:8]:
                shown = " | ".join(f"{k}: {NICE[m]}" for k, m in a)
                print(f"    {shown}  (rates {sorted(set(assigns[a]))})")
        if best is not None:
            used = {c["model"] for c in best[1]}
            print("  where the thresholds come from:")
            for m in sorted(used):
                print(f"    {NICE[m]:11s} {NOTES[m]}")


if __name__ == "__main__":
    main()
