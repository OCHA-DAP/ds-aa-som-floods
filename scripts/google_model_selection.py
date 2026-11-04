"""Re-run the design's model choice with the new Google model as a candidate.

The adopted design (src/constants.py TRIGGER_CONFIG) came from a joint search
over the four river-season windows (scripts/model_selection.py candidates:
per window one model, station return period 1-in-3 to 1-in-6, 2 .. k-1
votes, every window firing at least twice), judged on the two-gauge
benchmark (scripts/envelope_search.py), with GEOGloWS excluded (directive
2026-08-28), the envelope held to no more often than 1-in-3 (directive
2026-08-27: 8 years of 25), and Shabelle Deyr moved to 1-in-5 on 2026-09-18 so
the allocation is released in 8 river-seasons rather than 10.

This module reconstructs that as one search and checks it: among all
combinations with at most MAX_YEARS activation years and at most MAX_SEASONS
released river-seasons, take the most activation years (closest to the
target from below), then the design's ranking: severe years caught, fewer
activations outside a flood season, fewer unanimous rules, higher tracking
(the candidates' own correlation with the river's reference gauge), lower
return periods. Run on the old models (Google model_id_8583a5c2_v0, GloFAS
v4, GloFAS v5) it must return TRIGGER_CONFIG exactly. ms.choose() is not
used: its pick has no tracking tie-break and depends on candidate order.

    .venv/bin/python scripts/google_model_selection.py
"""

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "page_additions"))

import envelope_search as es  # noqa: E402
import model_selection as ms  # noqa: E402
import somlib as L  # noqa: E402
from src.constants import SEASONS, TRIGGER_CONFIG, TRIGGER_STATIONS  # noqa: E402

OLD_G, NEW_G = "google_grrr", "google_grrr_2026_11"
OLD_SET = [OLD_G, "glofas_v4", "glofas_v5"]
NEW_SET = [NEW_G, "glofas_v4", "glofas_v5"]
MAX_YEARS = 8     # no more often than 1-in-3 on 25 years: (25 + 1) / 8 = 3.25
MAX_SEASONS = 8   # released river-seasons, the 2026-09-18 adjustment
P = Path(os.environ.get("SOM_DATA_REPO", REPO)) / "data" / "processed"
WINDOWS = ms.WINDOWS  # juba gu, juba deyr, shabelle gu, shabelle deyr


def daily(models):
    d = pd.concat([pd.read_parquet(P / f"discharge_daily_{m}.parquet").assign(src=m) for m in models],
                  ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d


def levels():
    lv = pd.read_parquet(P / "swalim_levels.parquet")
    lv["date"] = pd.to_datetime(lv["date"])
    return lv


def released_seasons(combo):
    return {(y, w[1]) for w, c in zip(WINDOWS, combo) for y in c["years"]}


def search(cands, any_flood, severe, max_years=MAX_YEARS, max_seasons=MAX_SEASONS, years_exact=None):
    """The design's pick under the caps (None = no cap): (rank, combo, result)."""
    best = None
    for combo in itertools.product(*[cands[w] for w in WINDOWS]):
        if any(len(c["years"]) < ms.MIN_WINDOW_FIRES for c in combo):
            continue
        r = ms.evaluate(combo, any_flood, severe)
        n_seasons = len(released_seasons(combo))
        if not r["fires"] or (max_years and r["fires"] > max_years) or (max_seasons and n_seasons > max_seasons):
            continue
        if years_exact and r["fires"] != years_exact:
            continue
        rank = (r["fires"], r["severe_caught"], -len(r["no_flood_years"]), -sum(c["unanimous"] for c in combo),
                round(sum(c["rho"] for c in combo), 2), -sum(c["rp"] for c in combo))
        if best is None or rank > best[0]:
            r["seasons"] = n_seasons
            best = (rank, combo, r)
    return best


def tied_assignments(cands, any_flood, severe, pick, max_years=MAX_YEARS, max_seasons=MAX_SEASONS):
    """Distinct model assignments that match the pick on the backtest alone
    (activation years, severe caught, activations outside a flood season),
    i.e. before the tracking tie-break."""
    target = (pick[2]["fires"], pick[2]["severe_caught"], len(pick[2]["no_flood_years"]))
    out = set()
    for combo in itertools.product(*[cands[w] for w in WINDOWS]):
        if any(len(c["years"]) < ms.MIN_WINDOW_FIRES for c in combo):
            continue
        r = ms.evaluate(combo, any_flood, severe)
        if (max_years and r["fires"] > max_years) or (max_seasons and len(released_seasons(combo)) > max_seasons):
            continue
        if (r["fires"], r["severe_caught"], len(r["no_flood_years"])) == target:
            out.add(tuple(c["model"] for c in combo))
    return sorted(out)


def config_of(combo):
    return {w: {"source": c["model"], "rp": c["rp"], "n_req": c["n_req"]} for w, c in zip(WINDOWS, combo)}


def tracking(cands):
    """Per window and model: the search's tracking measure (mean best-lag
    Spearman of the window's points against the river's reference gauge)."""
    out = {}
    for w in WINDOWS:
        for c in cands[w]:
            out.setdefault(w, {})[c["model"]] = c["rho"]
    return out


def forecast_lead(window, models, rp, n_req, years):
    """Per year: the gauges' crossing day (two gauges over their own 1-in-3)
    and, per model, the first issue whose forecast (leads 1-7, ensemble median)
    met the window's rule against that model's own levels."""
    gx = L.gauge_crossings(*window, 3, span=years)
    out = {}
    for y in sorted(gx):
        row = {"gauges": gx[y]}
        for m in models:
            fi = L.first_issue_dates(m, *window, rp, n_req, span=years)
            row[m] = fi[y][0] if y in fi else None
        out[y] = row
    return out


def run():
    lv = levels()
    any_flood, severe = es.benchmark_years_from_gauges(lv)
    out = {"any_flood": any_flood, "severe": severe}
    for tag, models in (("old", OLD_SET), ("new", NEW_SET)):
        dd = daily(models)
        cands = {w: ms.window_candidates(dd, lv, *w, models=models) for w in WINDOWS}
        out[tag] = {"cands": cands, "tracking": tracking(cands),
                    "capped": search(cands, any_flood, severe),
                    "years_cap_only": search(cands, any_flood, severe, max_seasons=None),
                    "seven": search(cands, any_flood, severe, years_exact=7)}
        out[tag]["ties"] = tied_assignments(cands, any_flood, severe, out[tag]["capped"])
    # gate: on the old models the reconstructed search returns the adopted design
    got = config_of(out["old"]["capped"][1])
    assert got == {w: TRIGGER_CONFIG[w] for w in WINDOWS}, got
    assert out["old"]["capped"][2]["years"] == [2006, 2013, 2014, 2016, 2018, 2019, 2020, 2023]
    # forecast side: same rule (the adopted settings) on each forecast archive
    gu_years = list(range(2016, 2024))    # Gu seasons all three archives cover
    deyr_years = list(range(2016, 2023))  # the old Google archive ends June 2023
    out["lead"] = {w: forecast_lead(w, [OLD_G, NEW_G, "glofas_v4"], TRIGGER_CONFIG[w]["rp"], TRIGGER_CONFIG[w]["n_req"],
                                    gu_years if w[1] == "gu" else deyr_years) for w in WINDOWS}
    out["lead_years"] = {"gu": gu_years, "deyr": deyr_years}
    rf = L.reforecast("glofas_v4")
    out["glofas_issues_per_year"] = float(rf.groupby(rf.issued_time.dt.year).issued_time.nunique().median())
    return out


def describe(combo, r):
    return " | ".join(f"{w[0][:4]}-{w[1][:2]} {c['model']} 1-in-{c['rp']} {c['n_req']}of{len(c['stations'])}"
                      for w, c in zip(WINDOWS, combo)) + f" || {r['fires']} yrs {r.get('seasons')} seasons " \
        f"severe {r['severe_caught']}/{r['n_severe']} outside {r['no_flood_years']} {r['years']}"


if __name__ == "__main__":
    res = run()
    for tag in ("old", "new"):
        print(f"\n=== {tag}")
        print("tracking", {f"{w[0]}-{w[1]}": {m: round(v, 2) for m, v in t.items()} for w, t in res[tag]["tracking"].items()})
        for k in ("capped", "years_cap_only", "seven"):
            b = res[tag][k]
            print(f"  {k}: {describe(b[1], b[2]) if b else None}")
        print("  ties at the capped pick's backtest score:", len(res[tag]["ties"]), res[tag]["ties"])
    print("\nlead (adopted settings):")
    for w, rows in res["lead"].items():
        print(" ", w, {y: {k: (str(v.date()) if v is not None else None) for k, v in row.items()} for y, row in rows.items()})
    print("GloFAS v4 reforecast issues per year (median):", res["glofas_issues_per_year"])
    sys.stdout.flush()
    os._exit(0)
