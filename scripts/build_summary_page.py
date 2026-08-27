"""Regenerate the data-source review page directly from the source data.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/build_summary_page.py

Self-contained by design: it reads only the PRIMARY processed tables (the
gauge record, each model's daily discharge, the seasonal flood benchmarks)
and recomputes every headline figure with the same helpers the notebooks
use. The page is therefore current whether or not anybody has re-run a
notebook.

The adopted design lives in src/constants.py (TRIGGER_STATIONS, RIVER_MODEL,
TRIGGER_CONFIG, TRIGGER_YEARS), which notebook 09 reads as well, so the page
and the notebooks cannot drift apart.

Writes pages/summary/index.html and pages/summary/summary.json. Narrative
prose is static; numbers, tables and the timestamp are generated.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import ocha_stratus as stratus  # noqa: E402

from src.constants import (  # noqa: E402
    BENCHMARK_RP,
    ENVELOPE_TARGET_RP,
    SEVERE_RP,
    REFERENCE_GAUGE,
    WINDOW_MODEL,
    SEASONS,
    TRIGGER_CONFIG,
    TRIGGER_STATIONS,
    TRIGGER_YEARS,
)
from src.utils import episodes, hits, weibull_level, weibull_threshold  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import envelope_search  # noqa: E402
import model_selection  # noqa: E402
import page_chrome  # noqa: E402
import summary_figures  # noqa: E402

OUT = REPO / "pages" / "summary"
PREFIX = "ds-aa-som-floods/processed"
STAGE = "dev"
# every product the selection search considers (scripts/model_selection.py),
# so the comparison charts and the tie count cover the same field the choice
# was made in
MODELS = ["glofas_v5", "glofas_v4", "google_grrr", "geoglows"]
# Product and version on every label: GloFAS appears as two versions, so
# naming the others without a version made four columns look like four
# unrelated products.
NICE = {
    "google_grrr": "Google GRRR",
    "glofas_v5": "GloFAS v5",
    "glofas_v4": "GloFAS v4",
    "geoglows": "GEOGloWS v2",
}
GAUGE_BM = {r: f"swalim_{g}" for r, g in REFERENCE_GAUGE.items()}
MIN_LAG, MAX_LAG, MIN_OBS = -10, 30, 60


def load(name):
    return stratus.load_parquet_from_blob(f"{PREFIX}/{name}.parquet", stage=STAGE)


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def em(x):
    return f"<em>{x}</em>"


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        '<div class="tablewrap">\n<table class="data">\n'
        f"<thead><tr>{head}</tr></thead>\n<tbody>{body}</tbody>\n</table>\n</div>"
    )


def load_reforecast():
    """Forecast archives for the lead-time chart, tagged with a source key."""
    parts = []
    for name, key in [("reforecast_google_grrr", "google_grrr"),
                      ("reforecast_glofas_v4", "glofas_v4")]:
        try:
            d = load(name)
        except Exception as exc:
            print(f"  ! {name} unavailable: {type(exc).__name__}")
            continue
        d["valid_time"] = pd.to_datetime(d["valid_time"])
        parts.append(d.assign(source_key=key))
    return pd.concat(parts, ignore_index=True) if parts else None


def load_exposure():
    """District population exposure, from the same table notebook 01 queries."""
    import os
    import tempfile

    import geopandas as gpd

    raw = stratus.load_blob_data(
        f"{PREFIX}/workflow/som_river_buffers.gpkg", stage=STAGE
    )
    tmp = os.path.join(tempfile.gettempdir(), "som_river_buffers.gpkg")
    with open(tmp, "wb") as fh:
        fh.write(raw)
    bufs = gpd.read_file(tmp)
    adm2 = stratus.codab.load_codab_from_blob("som", admin_level=2)
    pcodes = sorted({
        p for _, buf in bufs.iterrows()
        for p in adm2[adm2.geometry.intersects(buf.geometry)]["ADM2_PCODE"]
    })
    if not pcodes:
        return None
    q = (
        "SELECT valid_date, pcode, sum FROM app.floodscan_exposure "
        "WHERE iso3='SOM' AND adm_level='2' AND pcode IN ("
        + ",".join(f"'{p}'" for p in pcodes)
        + ")"
    )
    exp = pd.read_sql(q, stratus.get_engine(stage="prod"))
    exp["valid_date"] = pd.to_datetime(exp["valid_date"])
    return exp


def forecast_daily():
    """A daily series per station built from the FORECAST archives alone.

    For every valid date, the ensemble median at each lead 1-7, then the
    strongest of those leads: what a trigger watching the coming week would
    have seen. Thresholds are fitted on this series, so nothing is inherited
    from a retrospective and the two kinds of evidence are never mixed.
    """
    fc = load_reforecast()
    if fc is None or fc.empty:
        return None, {}
    fc = fc[(fc.leadtime_days >= 1) & (fc.leadtime_days <= 7)]
    med = (
        fc.groupby(["source_key", "station", "valid_time", "leadtime_days"])["discharge"]
        .median()
        .groupby(level=["source_key", "station", "valid_time"])
        .max()
        .reset_index()
    )
    med = med.rename(columns={"source_key": "src", "valid_time": "date"})
    spans = {
        src: set(range(int(g.date.dt.year.min()), int(g.date.dt.year.max()) + 1))
        for src, g in med.groupby("src")
    }
    return med, spans

# ---------------------------------------------------------------- primary data
print("reading primary tables and recomputing the figures ...")
lv = load("swalim_levels")
lv["date"] = pd.to_datetime(lv["date"])
bench = load("workflow/som_flood_benchmark_seasonal")
th = load("swalim_thresholds").set_index("station")
dd = pd.concat(
    [load(f"discharge_daily_{m}").assign(src=m) for m in MODELS], ignore_index=True
)
dd["date"] = pd.to_datetime(dd["date"])

Y0, Y1 = TRIGGER_YEARS
span = set(range(Y0, Y1 + 1))
data = {
    "generated": date.today().isoformat(),
    "years": [Y0, Y1],
    "stations": TRIGGER_STATIONS,
    "window_model": {f"{r}_{s}": m for (r, s), m in WINDOW_MODEL.items()},
}


def model_season(model, station, months):
    s = dd[(dd.src == model) & (dd.station == station)].set_index("date")["discharge"]
    s = s[s.index.month.isin(months)]
    return s[(s.index.year >= Y0) & (s.index.year <= Y1)].sort_index()


def flood_years(river, season, rp=BENCHMARK_RP):
    b = bench[
        (bench.river == river)
        & (bench.season == season)
        & (bench.benchmark == GAUGE_BM[river])
    ]
    return set(b[b[f"flood_{rp}yr"] == 1].year) & span


def severe_window_years(river, season):
    """Years the river's reference gauge recorded a 1-in-SEVERE_RP or rarer season."""
    b = bench[
        (bench.river == river)
        & (bench.season == season)
        & (bench.benchmark == GAUGE_BM[river])
    ]
    return set(b[b.rp >= SEVERE_RP].year) & span


# ---- the adopted configuration, scored window by window
win_rows, per_window = [], {}
for (river, season), spec in TRIGGER_CONFIG.items():
    model, months = WINDOW_MODEL[(river, season)], SEASONS[season]
    cols = []
    for st in TRIGGER_STATIONS[river]:
        s = model_season(model, st, months)
        if len(s) < 100:
            continue
        am = s.groupby(s.index.year).max().dropna()
        t = weibull_threshold(am.values, spec["rp"])
        if not np.isnan(t):
            cols.append((s >= t).rename(st))
    if not cols:
        print(f"  ! no usable stations for {river} {season}")
        continue
    mat = pd.concat(cols, axis=1).fillna(False)
    mx = mat.sum(axis=1).groupby(mat.index.year).max()
    act = sorted(set(mx[mx >= spec["n_req"]].index) & span)
    ev = flood_years(river, season)
    tp, fp, fn = len(set(act) & ev), len(set(act) - ev), len(ev - set(act))
    per_window[(river, season)] = act
    win_rows.append(
        {
            "river": river,
            "season": season,
            "source": model,
            "rp": spec["rp"],
            "n_req": spec["n_req"],
            "n_of": len(cols),
            "pod": round(tp / (tp + fn), 2) if tp + fn else None,
            "far": round(fp / (tp + fp), 2) if tp + fp else None,
            "act_rp": round((len(span) + 1) / len(act), 1) if act else None,
            "activations": act,
            "missed": sorted(ev - set(act)),
        }
    )
data["windows"] = win_rows

# ---- the envelope: released whenever any window activates
union = {y for a in per_window.values() for y in a}
counts = {}
for a in per_window.values():
    for y in a:
        counts[y] = counts.get(y, 0) + 1
data["envelope"] = {
    "n_years_in_span": len(span),
    "n_fires": len(union),
    "rp": round((len(span) + 1) / len(union), 1) if union else None,
    "years": sorted(union),
    "multi_window_years": sorted(y for y, n in counts.items() if n >= 2),
}

# ---- model-choice evidence: best-lag rank correlation against the reference gauge
# rho is kept per station as well as averaged: the averages decide the model,
# the per-station values are what the correlation heatmap draws.
choice, rho = [], {}
for river, stns in TRIGGER_STATIONS.items():
    ref = (
        lv[lv.station == REFERENCE_GAUGE[river]]
        .set_index("date")["level_m"]
        .sort_index()
    )
    for model in MODELS:
        per_season = {}
        for season, months in SEASONS.items():
            rhos = []
            for st in stns:
                m = model_season(model, st, months)
                if not len(m):
                    continue
                best = 0.0
                for lag in range(MIN_LAG, MAX_LAG + 1):
                    j = pd.concat([m, ref.shift(-lag)], axis=1, join="inner").dropna()
                    if len(j) < MIN_OBS:
                        continue
                    r = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
                    if pd.notna(r) and abs(r) > abs(best):
                        best = float(r)
                rho[(river, st, model, season)] = round(best, 2)
                rhos.append(best)
            if rhos:
                per_season[season] = float(np.mean(rhos))
        if len(per_season) == 2:
            choice.append(
                {
                    "river": river,
                    "source": model,
                    "gu": round(per_season["gu"], 2),
                    "deyr": round(per_season["deyr"], 2),
                    "worst": round(min(per_season.values()), 2),
                }
            )
data["model_choice"] = choice

# ---- do the rivers flood in the same seasons?
cooc = []
for rp in (3, 4, 5):
    entry = {"rp": rp}
    for season in ("gu", "deyr"):
        sets = {r: flood_years(r, season, rp) for r in TRIGGER_STATIONS}
        both = sets["juba"] & sets["shabelle"]
        entry[season] = {
            "both": len(both),
            "of": max(len(sets["juba"]), len(sets["shabelle"])),
        }
    cooc.append(entry)
data["cooccurrence"] = cooc

# ---- per-station model scores against the computed RP3 baseline
station_scores = []
for river, stns in TRIGGER_STATIONS.items():
    for st in stns:
        obs = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
        modern = obs[obs.index.year >= 2000]
        am = modern.groupby(modern.index.year).max().dropna()
        base = weibull_level(am.values, BENCHMARK_RP)
        if np.isnan(base):
            continue
        obs_ev = episodes(obs >= base)
        if not obs_ev:
            continue
        for model in MODELS:
            m = dd[(dd.src == model) & (dd.station == st)].set_index("date")["discharge"]
            m = m[(m.index.year >= 2002) & (m.index.year <= 2023)].sort_index()
            if len(m) < 500:
                continue
            am_m = m.groupby(m.index.year).max().dropna()
            t = weibull_threshold(am_m.values, BENCHMARK_RP)
            if np.isnan(t):
                continue
            # score on days both series cover, as notebook 02 does
            j = pd.concat([m.rename("mod"), obs.rename("obs")], axis=1,
                          join="inner").dropna()
            if len(j) < 500:
                continue
            obs_ev_j = episodes(j["obs"] >= base)
            if not obs_ev_j:
                continue
            mod_ev = episodes(j["mod"] >= t)
            pod = hits(obs_ev_j, mod_ev) / len(obs_ev_j)
            far = (1 - hits(mod_ev, obs_ev_j) / len(mod_ev)) if mod_ev else None
            station_scores.append(
                {
                    "station": st,
                    "river": river,
                    "source": model,
                    "n_events": len(obs_ev_j),
                    "POD": round(pod, 2),
                    "FAR": round(far, 2) if far is not None else None,
                }
            )
data["station_scores"] = station_scores

# ---- the envelope frontier: what the union can buy at each activation rate
# Computed with scripts/envelope_search.py, so the page and the search cannot
# give different answers. The adopted configuration is marked on the chart.
print("searching the envelope frontier ...")
_opts = envelope_search.window_options(dd, bench)
_any_flood, _severe = envelope_search.benchmark_years(bench)
_keys, _best = envelope_search.search(_opts, _any_flood, _severe)
frontier = []
for _k in sorted(_best):
    _, _combo, _r = _best[_k]
    if 1.4 <= _r["env_rp"] <= 6.5:
        frontier.append({"env_rp": _r["env_rp"], "fires": _r["fires"],
                         "severe_caught": _r["severe_caught"],
                         "no_flood_years": _r["no_flood_years"]})
_stored = tuple((TRIGGER_CONFIG[k]["rp"], TRIGGER_CONFIG[k]["n_req"]) for k in _keys)
adopted = envelope_search.evaluate(_opts, _stored, _keys, _any_flood, _severe)
data["severe_years"] = sorted(_severe)
data["frontier"] = frontier
data["adopted"] = adopted
print(f"  adopted configuration: 1-in-{adopted['env_rp']} | "
      f"severe {adopted['severe_caught']}/{len(_severe)} | "
      f"no-flood years {adopted['no_flood_years']}")

# ---- four selections, evidence never mixed, Google in or out of each
# The forecast field excludes GEOGloWS (its forecasts run below its own
# retrospective) and GloFAS v5 (no reforecast). Return periods are capped at a
# quarter of each field's record: a 1-in-5 fitted to 8 years of Google
# reforecast would be extrapolation.
print("building the four selections ...")
fc_daily, fc_spans = forecast_daily()
variants = {}
FIELDS = [
    ("reanalysis", MODELS, dd, span),
    ("forecast", ["google_grrr", "glofas_v4"], fc_daily, None),
]
for basis, field, frame, basis_span in FIELDS:
    if frame is None:
        continue
    for with_google in (True, False):
        off_target = False
        models_here = [m for m in field if with_google or m != "google_grrr"]
        if not models_here:
            continue
        if basis == "forecast":
            common = set.intersection(*[fc_spans[m] for m in models_here
                                        if m in fc_spans]) if models_here else set()
            use_span = common & span
        else:
            use_span = basis_span
        if len(use_span) < 8:
            variants[(basis, with_google)] = {
                "error": (f"only {len(use_span)} years of overlapping record, too "
                          "short to fit a return-period threshold")
            }
            continue
        # 1-in-3 is the floor for a gauge threshold (directive 2026-08-27), and
        # the ceiling stays a quarter of the record. A field that cannot support
        # a 1-in-3 threshold is reported as such rather than dropped to 1-in-2.
        rp_cap = len(use_span) // 4
        rps_here = [r for r in [3, 4, 5, 6] if r <= rp_cap]
        if not rps_here:
            variants[(basis, with_google)] = {
                "error": (f"{len(use_span)} years of overlapping record "
                          f"({min(use_span)}-{max(use_span)}) cannot support a "
                          "1-in-3 gauge threshold: fitting one needs at least 12 "
                          "years, and gauge thresholds may not go below 1-in-3")
            }
            continue
        cands = {
            k: model_selection.window_candidates(
                frame, lv, k[0], k[1], models=models_here, span=use_span, rps=rps_here
            )
            for k in TRIGGER_CONFIG
        }
        if basis == "reanalysis" and with_google:
            # the adopted configuration itself, not a re-search: with 154
            # assignments tying, a search returns an arbitrary one of them
            keys = list(TRIGGER_CONFIG)
            combo = []
            for k in keys:
                spec = TRIGGER_CONFIG[k]
                match = [c for c in cands[k]
                         if c["model"] == spec["source"] and c["rp"] == spec["rp"]
                         and c["n_req"] == spec["n_req"]]
                if not match:
                    combo = None
                    break
                combo.append(match[0])
            best = None if combo is None else (
                None, tuple(combo),
                model_selection.evaluate(tuple(combo), _any_flood, _severe,
                                         span=use_span),
            )
        else:
            keys, best, _frontier = model_selection.choose(
                cands, _any_flood, _severe, span=use_span
            )
            # nothing on target is a finding, not a dead end: show the closest
            # rate this evidence can actually support, labelled as such
            if best is None and _frontier:
                closest = min(
                    _frontier.values(),
                    key=lambda t: abs(t[2]["env_rp"] - ENVELOPE_TARGET_RP),
                )
                best, off_target = closest, True
        if best is None:
            variants[(basis, with_google)] = {
                "error": (f"no configuration lands near 1-in-{ENVELOPE_TARGET_RP} on "
                          f"{len(use_span)} years of overlapping record "
                          f"({min(use_span)}-{max(use_span)}), where thresholds cannot "
                          f"go rarer than 1-in-{rp_cap}")
            }
            continue
        _, combo, res = best
        variants[(basis, with_google)] = {
            "years_from": min(use_span),
            "years_to": max(use_span),
            "n_years": len(use_span),
            "rp_cap": rp_cap,
            "off_target": off_target,
            "envelope": res,
            "windows": [
                {"river": r, "season": sn, "source": c["model"], "rp": c["rp"],
                 "n_req": c["n_req"], "n_of": len(c["stations"]),
                 "stations": c["stations"], "years": sorted(c["years"])}
                for (r, sn), c in zip(keys, combo)
            ],
        }
        print(f"  {basis:10s} google={'yes' if with_google else 'no ':3s} "
              f"{len(use_span)} yrs, RP<= {rp_cap} -> "
              f"1-in-{res['env_rp']} | severe {res['severe_caught']}/{res['n_severe']}")
data["variants"] = {f"{b}_{'google' if g else 'nogoogle'}": v
                    for (b, g), v in variants.items()}

# ---- the same per-gauge test, but on the forecast archives
# Thresholds fitted on each product's own forecast series, so a forecast is
# never judged against a retrospective-fitted number. GloFAS v5 has no
# reforecast and GEOGloWS's forecasts sit below its retrospective, so only
# Google and GloFAS v4 can be scored this way.
fc_scores, fc_spans_used = [], {}
if fc_daily is not None:
    for river, stns in TRIGGER_STATIONS.items():
        for st in stns:
            obs = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
            modern = obs[obs.index.year >= 2000]
            am_o = modern.groupby(modern.index.year).max().dropna()
            base = weibull_level(am_o.values, BENCHMARK_RP)
            if np.isnan(base):
                continue
            for src in sorted(fc_daily.src.unique()):
                m = fc_daily[(fc_daily.src == src) & (fc_daily.station == st)]
                if m.empty:
                    continue
                m = m.set_index("date")["discharge"].sort_index()
                am_m = m.groupby(m.index.year).max().dropna()
                if len(am_m) < 8:
                    continue
                t = weibull_threshold(am_m.values, BENCHMARK_RP)
                if np.isnan(t):
                    continue
                j = pd.concat([m.rename("mod"), obs.rename("obs")], axis=1,
                              join="inner").dropna()
                if len(j) < 300:
                    continue
                oev = episodes(j["obs"] >= base)
                mev = episodes(j["mod"] >= t)
                if not oev:
                    continue
                pod = hits(oev, mev) / len(oev)
                far = (1 - hits(mev, oev) / len(mev)) if mev else None
                fc_spans_used.setdefault(
                    src, (int(m.index.year.min()), int(m.index.year.max()))
                )
                fc_scores.append({
                    "station": st, "river": river, "source": src,
                    "n_events": len(oev), "POD": round(pod, 2),
                    "FAR": round(far, 2) if far is not None else None,
                })
data["station_scores_forecast"] = fc_scores
print(f"  forecast-basis station scores: {len(fc_scores)}")

# ---- charts: the same figures the review deck carried
print("drawing the figures ...")
figs = summary_figures.build(
    {
        "lv": lv,
        "th": th,
        "dd": dd,
        "Y0": Y0,
        "Y1": Y1,
        "span": span,
        "model_season": model_season,
        "flood_years": flood_years,
        "severe_window_years": severe_window_years,
        "station_scores": station_scores,
        "rho": rho,
        "per_window": per_window,
        "reference_gauge": REFERENCE_GAUGE,
        "window_model": WINDOW_MODEL,
        "frontier": frontier,
        "adopted_point": adopted,
        "n_severe": len(_severe),
        "target_rp": ENVELOPE_TARGET_RP,
        "reforecast": load_reforecast,
        "exposure": load_exposure,
    },
    OUT / "figures",
)
data["figures"] = sorted(figs)

def figure(name):
    """Embed one generated chart, or nothing if it could not be drawn."""
    f = figs.get(name)
    if not f:
        return ""
    return (
        f'    <figure class="chart">\n'
        f'      <img src="{f["src"]}" alt="{esc(f["caption"])}">\n'
        f'      <figcaption>{esc(f["caption"])}</figcaption>\n'
        f"    </figure>"
    )


OUT.mkdir(parents=True, exist_ok=True)

# how many model assignments reach the same result: if many do, the backtest
# is not what the model choice rests on, and the page should say so
_cands = {
    (river, season): model_selection.window_candidates(dd, lv, river, season)
    for river, season in TRIGGER_CONFIG
}
_score, _assigns = model_selection.ties(_cands, _any_flood, _severe)
n_tied = len(_assigns)
data["n_tied_assignments"] = n_tied

# what the trigger looks like with Google off the table: same search, smaller
# field. Google is one provider with a 7-day horizon and a 2016-2023 reforecast,
# so the cost of dropping it is worth stating.
_no_google = {k: [c for c in v if c["model"] != "google_grrr"]
              for k, v in _cands.items()}
_ng_keys, _ng_best, _ = model_selection.choose(_no_google, _any_flood, _severe)
no_google = None
if _ng_best is not None:
    _, _ng_combo, _ng_r = _ng_best
    no_google = {
        "envelope": _ng_r,
        "windows": [
            {"river": river, "season": season, "source": c["model"], "rp": c["rp"],
             "n_req": c["n_req"], "n_of": len(c["stations"]),
             "years": sorted(c["years"])}
            for (river, season), c in zip(_ng_keys, _ng_combo)
        ],
    }
    print(f"  without Google: 1-in-{_ng_r['env_rp']} | "
          f"severe {_ng_r['severe_caught']}/{len(_severe)}")
data["no_google"] = no_google
print(f"  model assignments tying the adopted result: {n_tied}")

# ---- the four selections, as tabs
TAB_LABELS = {
    ("reanalysis", True): "Reanalysis",
    ("reanalysis", False): "Reanalysis, no Google",
    ("forecast", True): "Forecasts only",
    ("forecast", False): "Forecasts only, no Google",
}
BASIS_NOTE = {
    "reanalysis": ("Thresholds and activation years from the retrospective "
                   "simulations. Every product is in the field: GloFAS v5, "
                   "GloFAS v4, Google and GEOGloWS."),
    "forecast": ("Thresholds and activation years from the forecast archives "
                 "themselves, so nothing is inherited from a retrospective. "
                 "GEOGloWS is out, its forecasts run below its own "
                 "retrospective; GloFAS v5 is out, it publishes no reforecast."),
}
tab_buttons, tab_panels = [], []
for i, (key, lab) in enumerate(TAB_LABELS.items()):
    v = variants.get(key)
    slug = f"{key[0]}-{'google' if key[1] else 'nogoogle'}"
    active = " active" if i == 0 else ""
    tab_buttons.append(
        f'      <button class="tab{active}" data-group="variant" '
        f'data-panel="{slug}">{esc(lab)}</button>'
    )
    if not v or "error" in v:
        why = v.get("error", "not available") if v else "not available"
        tab_panels.append(
            f'      <div class="panel{active}" id="{slug}">\n'
            f"        <p>{BASIS_NOTE[key[0]]}</p>\n"
            f'        <p class="muted">No selection to show: {esc(why)}.</p>\n'
            f"      </div>"
        )
        continue
    e = v["envelope"]
    rows = [
        (
            w["river"].capitalize(),
            w["season"].capitalize(),
            NICE.get(w["source"], w["source"]),
            f"1-in-{w['rp']}",
            f"{w['n_req']} of {w['n_of']}",
            ", ".join(str(y) for y in w["years"]) or em("never"),
        )
        for w in v["windows"]
    ]
    body = table(
        ["River", "Season", "Forecast", "Gauge threshold", "Gauges that must agree",
         "Would have activated in"],
        rows,
    )
    srcs = ", ".join(sorted({NICE.get(w["source"], w["source"]) for w in v["windows"]}))
    cap = ""
    if v["rp_cap"] < 5:
        cap = (f' Thresholds are capped at 1-in-{v["rp_cap"]} here, a quarter of the '
               f'{v["n_years"]}-year record: anything rarer would be extrapolation.')
    tab_panels.append(
        f'      <div class="panel{active}" id="{slug}">\n'
        f"        <p>{BASIS_NOTE[key[0]]} Calibrated on "
        f'{v["years_from"]}-{v["years_to"]}, {v["n_years"]} years.{cap}</p>\n'
        f"{body}\n"
        f'        <p><strong>Envelope: 1-in-{e["env_rp"]}</strong>'
        f'{" (the closest this evidence supports; nothing lands on the target)" if v.get("off_target") else ""} '
        f'({e["fires"]} of {v["n_years"]} years), catching '
        f'{e["severe_caught"]} of the {e["n_severe"]} severe years, on {srcs}. '
        f'Activated with no recorded flood: '
        f'{", ".join(str(y) for y in e["no_flood_years"]) or "never"}. '
        f'Severe years missed: '
        f'{", ".join(str(y) for y in e["severe_missed"]) or "none"}.</p>\n'
        f"      </div>"
    )
tabs_html = (
    '    <div class="tabbar">\n' + "\n".join(tab_buttons) + "\n    </div>\n"
    '    <div class="panels">\n' + "\n".join(tab_panels) + "\n    </div>"
)

# ---------------------------------------------------------------------- render
# the trigger, stated as a rule: this is what leads the page
trigger_rows = []
for w in win_rows:
    trigger_rows.append(
        (
            w["river"].capitalize(),
            w["season"].capitalize(),
            NICE.get(w["source"], w["source"]),
            f"1-in-{w['rp']}",
            f"{w['n_req']} of {w['n_of']}",
            ", ".join(str(y) for y in w["activations"]) or em("never"),
        )
    )
trigger_html = table(
    ["River", "Season", "Forecast", "Gauge threshold", "Gauges that must agree",
     "Would have activated in"],
    trigger_rows,
)
severe_missed = ", ".join(str(y) for y in adopted["severe_missed"]) or "none"

if no_google:
    ng_html = table(
        ["River", "Season", "Forecast", "Gauge threshold", "Gauges that must agree",
         "Would have activated in"],
        [
            (
                w["river"].capitalize(),
                w["season"].capitalize(),
                NICE.get(w["source"], w["source"]),
                f"1-in-{w['rp']}",
                f"{w['n_req']} of {w['n_of']}",
                ", ".join(str(y) for y in w["years"]) or em("never"),
            )
            for w in no_google["windows"]
        ],
    )
    ng = no_google["envelope"]
    ng_delta = ng["severe_caught"] - adopted["severe_caught"]
    ng_verdict = (
        "the same severe-year coverage" if ng_delta == 0 else
        f"{abs(ng_delta)} fewer severe year{'s' if abs(ng_delta) != 1 else ''}"
        if ng_delta < 0 else
        f"{ng_delta} more severe year{'s' if ng_delta != 1 else ''}"
    )
    ng_sources = ", ".join(
        sorted({NICE.get(w["source"], w["source"]) for w in no_google["windows"]})
    )
    # caveats that follow from what the variant actually selected
    _ng_srcs = {w["source"] for w in no_google["windows"]}
    _ng_notes = []
    if len(_ng_srcs) == 1:
        _ng_notes.append("every window would run on a single provider")
    elif _ng_srcs <= {"glofas_v4", "glofas_v5"}:
        _ng_notes.append(
            "every window would run on GloFAS, but on two different versions: "
            "thresholds and operational forecasts should come from the same one, "
            "so both would have to be refitted on v5"
        )
    if "geoglows" in _ng_srcs:
        _ng_notes.append(
            "GEOGloWS forecasts run below its own retrospective, so its thresholds "
            "would have to be refitted on the forecast archive before it could run"
        )
    if "glofas_v5" in _ng_srcs:
        _ng_notes.append(
            "there is no v5 reforecast, so the lead-time evidence comes from v4"
        )
    ng_caveats = "; ".join(_ng_notes)
else:
    ng_html, ng, ng_verdict, ng_sources = "", None, "", ""
# the whole section, so it simply disappears if the variant cannot be built
no_google_html = ""
if no_google:
    no_google_html = f"""    <h2><span class="num">6</span>Without Google</h2>
    <p>Google is a single provider with a seven-day horizon and a reforecast that starts
      in 2016, so it is worth knowing what the trigger looks like if it is set aside. The
      same search, over the remaining products only:</p>
{ng_html}
    <p>That envelope is <strong>1-in-{ng['env_rp']}</strong>, catching
      {ng['severe_caught']} of the {len(data['severe_years'])} severe years, which is
      {ng_verdict} than the adopted design, on {ng_sources}. So dropping Google costs
      nothing measurable in the backtest, which is the same lesson as the tie count
      above: the activation years do not separate the products. The cost is
      operational: {ng_caveats}.</p>
"""


# --------------------------------------------------------------------- render
# how often the strongest hindsight model matches or beats the adopted one
_sc = pd.DataFrame(station_scores)
n_v5_ahead, n_gauges = 0, 0
if not _sc.empty:
    for st in _sc.station.unique():
        a = _sc[(_sc.station == st) & (_sc.source == "glofas_v5")]
        b = _sc[(_sc.station == st) & (_sc.source == "google_grrr")]
        if a.empty or b.empty:
            continue
        n_gauges += 1
        pa, pb = a.POD.iloc[0], b.POD.iloc[0]
        fa, fb = a.FAR.iloc[0], b.FAR.iloc[0]
        if pa > pb or (pa == pb and fa is not None and fb is not None and fa <= fb):
            n_v5_ahead += 1

n_juba, n_shab = len(TRIGGER_STATIONS["juba"]), len(TRIGGER_STATIONS["shabelle"])
models_used = sorted({w["source"] for w in win_rows})
model_tile = " + ".join(NICE.get(m, m) for m in models_used)
env = data["envelope"]
act_rps = [w["act_rp"] for w in win_rows if w["act_rp"]]

tiles = [
    (
        f"1-in-{adopted['env_rp']}",
        f"envelope activation rate (target 1-in-{ENVELOPE_TARGET_RP})",
        "",
    ),
    (
        f"{adopted['severe_caught']} of {len(data['severe_years'])}",
        f"severe years caught (1-in-{SEVERE_RP} or rarer at the gauge)",
        "",
    ),
    (f"{n_juba} + {n_shab}", "gauges monitored: Juba and Shabelle", ""),
    ("7 days", f"lead time | sources: {model_tile}", ""),
]
tiles_html = "\n".join(
    f'      <div class="stat{" " + c if c else ""}"><span class="v">{esc(v)}</span>\n'
    f'        <span class="l">{esc(l)}</span></div>'
    for v, l, c in tiles
)

config_html = table(
    ["River", "Season", "Source", "RP", "Stations", "POD", "FAR", "Activates"],
    [
        (
            w["river"].capitalize(),
            w["season"].capitalize(),
            NICE.get(w["source"], w["source"]),
            w["rp"],
            f"{w['n_req']} of {w['n_of']}",
            em(w["pod"]) if (w["pod"] or 0) >= 0.8 else w["pod"],
            em(w["far"]) if (w["far"] is not None and w["far"] <= 0.2) else w["far"],
            f"1-in-{w['act_rp']}",
        )
        for w in win_rows
    ],
)

missed_html = "; ".join(
    f"{w['river'].capitalize()} {w['season'].capitalize()} "
    + (", ".join(str(y) for y in w["missed"]) if w["missed"] else "none")
    for w in win_rows
)

choice_rows = []
for river in TRIGGER_STATIONS:
    rows = sorted(
        [c for c in choice if c["river"] == river], key=lambda r: -r["worst"]
    )
    cells = [
        f"{NICE.get(r['source'], r['source'])} "
        + (em(r["worst"]) if i == 0 else str(r["worst"]))
        for i, r in enumerate(rows)
    ]
    choice_rows.append((f"{river.capitalize()} (worse season)", *cells))
choice_html = table(["River", "Best", "Second", "Third"], choice_rows)

cooc_html = table(
    ["Level", "Gu: both rivers flood", "Deyr: both rivers flood"],
    [
        (
            f"{c['rp']}-year",
            f"{c['gu']['both']} of {c['gu']['of']} seasons",
            f"{c['deyr']['both']} of {c['deyr']['of']}",
        )
        for c in cooc
    ],
)

station_html = table(
    ["River", "Gauges used"],
    [
        (r.capitalize(), ", ".join(em(s) for s in v))
        for r, v in TRIGGER_STATIONS.items()
    ],
)

if station_scores:
    sc = pd.DataFrame(station_scores)
    rows = []
    for st, g in sc.groupby("station", sort=False):
        cells = []
        for model in MODELS:
            r = g[g.source == model]
            cells.append(
                f"{r.iloc[0]['POD']:.2f} / {r.iloc[0]['FAR']:.2f}" if len(r) else "-"
            )
        rows.append((st, int(g.n_events.iloc[0]), *cells))
    scores_html = table(
        ["Station", "Events", *[NICE[m] for m in MODELS]], rows
    )
else:
    scores_html = (
        '<div class="callout">No station had enough overlapping gauge and model '
        "data to score.</div>"
    )

# ---- section 4 as two panels: reanalysis and forecasts, never mixed
def score_table(scores, sources, labels):
    if not scores:
        return ('<div class="callout">Nothing to score on this basis.</div>')
    sc = pd.DataFrame(scores)
    rows = []
    for st, g in sc.groupby("station", sort=False):
        cells = []
        for src in sources:
            r = g[g.source == src]
            cells.append(
                f"{r.iloc[0]['POD']:.2f} / {r.iloc[0]['FAR']:.2f}" if len(r) else "-"
            )
        rows.append((st, int(g.n_events.iloc[0]), *cells))
    return table(["Gauge", "Events", *labels], rows)


fc_sources = sorted(fc_spans_used)
basis_panels = (
    '    <div class="tabbar">\n'
    '      <button class="tab active" data-group="basis" data-panel="basis-rean">'
    "Reanalysis</button>\n"
    '      <button class="tab" data-group="basis" data-panel="basis-fc">'
    "Forecasts</button>\n"
    "    </div>\n"
    '    <div class="panels">\n'
    '      <div class="panel active" id="basis-rean">\n'
    "        <p>Each product's retrospective simulation against the events recorded at "
    f"the gauges, {Y0 + 3} to {Y1}, scored at its own 1-in-{BENCHMARK_RP} discharge so "
    "it is judged on timing rather than on scale.</p>\n"
    f"{scores_html}\n"
    f"{figure('skill')}\n"
    "      </div>\n"
    '      <div class="panel" id="basis-fc">\n'
    "        <p>The same gauge events, but each product scored on its own forecast "
    "archive: the ensemble median at leads 1 to 7, with thresholds fitted on that "
    "forecast series rather than inherited from a retrospective. Only two products can "
    "be tested this way. GloFAS v5 publishes no reforecast, and GEOGloWS's forecasts "
    "run below its own retrospective, so a retrospective-fitted threshold would not "
    "transfer.</p>\n"
    "        <p>Read these rates with care: an archive covering 8 years holds only a "
    "handful of gauge events, so one hit or miss moves POD a long way. The event count "
    "per gauge is in the second column.</p>\n"
    + score_table(
        fc_scores,
        fc_sources,
        [f"{NICE.get(m, m)} ({fc_spans_used[m][0]}-{fc_spans_used[m][1]})"
         for m in fc_sources],
    )
    + "\n      </div>\n    </div>"
)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Somalia floods: data-source review &mdash; Somalia Riverine Flood Trigger</title>
<meta name="description" content="Review of the data sources behind the Somalia riverine flood trigger: the SWALIM gauge record, how each model performs against it, the stations selected, and the resulting configuration.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css">
<style>
{page_chrome.STYLE}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <div class="inner">
      <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / data-source review</p>
      <h1>Somalia floods: data-source review</h1>
      <p>The evidence behind the trigger: the SWALIM gauge record, how each model
        performs against it, the stations selected, and the configuration that follows.
        Generated {data['generated']} from the source data.</p>
    </div>
  </header>

  <article>

    <div class="stats">
{tiles_html}
    </div>

    <h2><span class="num">1</span>Key takeaways</h2>
    <ul class="takeaways">
      <li><strong>Model performance differs by river and season.</strong> GloFAS leads
        the Shabelle and both Deyr windows; Google leads the Juba in Gu.</li>
      <li><strong>Accuracy holds from one to seven days ahead,</strong> so a one-week
        action window is defensible. Models still miss a share of floods, so SWALIM
        guidance belongs in the trigger design.</li>
      <li><strong>The gauge record is the reference,</strong> and only the reporting
        gauges can serve a live trigger: {n_juba} on the Juba, {n_shab} on the
        Shabelle. Thresholds are fitted on post-2000 data, because flooding has been
        recorded far more often in recent decades.</li>
      <li><strong>A flood is defined by the computed 1-in-3-year level at the
        gauge,</strong> not by an official SWALIM mark: it is frequency-calibrated, so
        it means the same thing at every gauge.</li>
    </ul>

    <h2><span class="num">2</span>SWALIM thresholds</h2>
    <p>SWALIM publishes three risk levels per station: moderate, high and bank full.
      Each station's 1-in-3-year level is estimated from its own annual maxima and
      compared against them. Where the 1-in-3-year level sits above the moderate mark,
      that mark is crossed more often than once every three years.</p>
{figure("thresholds")}

    <h2><span class="num">3</span>SWALIM events</h2>
    <p>An event is a spell in which the level sits at or above the baseline. The level
      must stay below it for at least 14 days before a new crossing counts separately,
      so a brief dip mid-flood does not split one flood in two. Readings stop rising
      once a gauge reaches bank full, so the largest events are understated.</p>
{figure("map")}
{figure("backtest")}

    <h2><span class="num">4</span>SWALIM risk level crossings</h2>
{figure("crossings")}

    <h2><span class="num">5</span>SWALIM against flood exposure</h2>
{figure("exposure")}

    <h2><span class="num">6</span>Reanalysis performance</h2>
    <p>Each model's 1-in-3-year signal against the events recorded at the gauges, 2002
      to 2023, within a 7-day window. POD is the share of events caught; FAR the share
      of alarms that were false.</p>
{figure("skill")}
{scores_html}

    <h2><span class="num">7</span>Model correlation</h2>
    <p>Daily tracking rather than event detection: the best-lag rank correlation between
      each model and the river's reference gauge, allowing for travel time.</p>
{figure("correlation")}
{choice_html}

    <h2><span class="num">8</span>Forecast correlation</h2>
    <p>The same test on the forecasts rather than the hindsight simulations, leads 1 to
      7.</p>
{figure("forecast")}

    <h2><span class="num">9</span>Selected stations</h2>
{station_html}
    <p>Gauges whose records stop in 2008 or earlier can calibrate but cannot monitor.
      Upstream stations see the flood before the downstream reference gauge: Dollow leads
      Luuq by about five days in Deyr, and Belet Weyne leads by about six in Gu.</p>

    <h2><span class="num">10</span>Grid search</h2>
    <p>Every combination of station return period and number of gauges that must agree,
      scored against the gauge benchmark. The outlined cell is the one adopted.</p>
{figure("grid")}

    <h2><span class="num">11</span>Best configuration</h2>
{trigger_html}
    <p>The full amount is released whenever the trigger is reached along either river in
      either season, so the combined rate across all four windows is what the budget must
      be sized on. As configured it would have released in {adopted['fires']} of
      {env['n_years_in_span']} years, once every {adopted['env_rp']} years, catching
      {adopted['severe_caught']} of the {len(data['severe_years'])} severe years.</p>
{figure("activation")}

  </article>
</div>
</body>
</html>
"""

(OUT / "summary.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
print(f"wrote {OUT / 'summary.json'}")
(OUT / "index.html").write_text(HTML, encoding="utf-8")
print(f"wrote {OUT / 'index.html'}")
print(
    f"  {len(win_rows)} windows | envelope {env['n_fires']}/{env['n_years_in_span']}"
    f" years (1-in-{env['rp']}) | {len(station_scores)} station scores"
)
