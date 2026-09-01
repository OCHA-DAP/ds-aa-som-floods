"""Rebuild the multi-source trigger document under one-model-per-window rules.

Starts from pages/trigger/index.html, keeps its structure, styling, provider
toggle and section order, and redoes the analysis under the conditions set on
2026-08-27:

  stations    all four Juba points and all three Shabelle points, none dropped
  models      one source per river-season window, never mixed inside a window,
              so four model choices
  thresholds  1-in-3 is the floor for every leg including readiness, and a
              quarter of the record is the ceiling; 1-in-2 is never used
  basis       reanalysis is the calibration basis, with a forecasts-only view;
              the two are never combined in one table

Every section of the source document is kept. Three carry diagnostics that
compare models rather than rules (seasonal peaks, the metric plane, each
model against its own reanalysis); they are marked as carried over unchanged.

DO NOT REGENERATE BLINDLY (2026-09-01). The published page at
pages/trigger-single-model/index.html carries hand-applied edits this script
does not reproduce: the review notes R1-R12 and their toggle, the two-state
provider switch (the third state was removed, review note R3), and several
correction passes. Rerunning this script would discard them. Treat the page as
the current source of truth for prose; use this script's draw_* functions to
regenerate figures (see scratchpad make_figs pattern: exec the script up to
draw_model_choice and call the draw functions directly).

Usage (from repo root):
    .venv/Scripts/python.exe scripts/build_multisource_variant.py

Writes pages/trigger-single-model/. The published pages/trigger is untouched.
"""

import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import ocha_stratus as stratus  # noqa: E402

import envelope_search  # noqa: E402
import model_selection  # noqa: E402
import summary_figures  # noqa: E402

from src.constants import (  # noqa: E402
    BENCHMARK_RP,
    ENVELOPE_TARGET_RP,
    INK,
    REFERENCE_GAUGE,
    SEASONS,
    SEVERE_RP,
    SOURCE_COLORS,
    TRIGGER_CONFIG,
    TRIGGER_STATIONS,
    TRIGGER_YEARS,
    WINDOW_MODEL,
)
from src.plots import style_ax  # noqa: E402
from src.utils import (  # noqa: E402
    episodes,
    hits,
    weibull_level,
    weibull_threshold,
)

SRC_PAGE = REPO / "pages" / "trigger" / "index.html"
SRC_FIGS = REPO / "pages" / "trigger" / "figs"
OUT = REPO / "pages" / "trigger-single-model"
FIGS = OUT / "figs"
PREFIX, STAGE = "ds-aa-som-floods/processed", "dev"

MODELS = ["glofas_v5", "glofas_v4", "google_grrr", "geoglows"]
# Which models each switch state puts on display in the comparison
# figures. GloFAS v4 is in neither (2026-08-31): v5 supersedes it on
# magnitude and timing, and v4 survives only as the readiness reforecast.
SET_MODELS = {
    "base": ["glofas_v5", "google_grrr"],          # adopted set, on load
    "all": ["glofas_v5", "google_grrr", "geoglows"],  # data-alt-src
}

FC_MODELS = ["google_grrr", "glofas_v4"]
READINESS_MODEL = "glofas_v4"
READINESS_LEADS = (7, 12)
ACTION_LEADS = (1, 7)
RP_FLOOR = 3
NICE = {"google_grrr": "Google GRRR", "glofas_v5": "GloFAS v5",
        "glofas_v4": "GloFAS v4", "geoglows": "GEOGloWS v2"}
Y0, Y1 = TRIGGER_YEARS
SPAN = set(range(Y0, Y1 + 1))
N_YEARS = len(SPAN)
WINDOWS = list(TRIGGER_CONFIG)
WLABEL = {k: f"{k[0].capitalize()} {k[1].capitalize()}" for k in WINDOWS}

# Which provider set fills which role on the page. The default excludes
# GEOGloWS, because its return periods cannot yet be fitted on its own
# forecasts (archive begins July 2024).
BASE_SET = "nogeoglows"      # shown on load
ALT_SET = False              # data-alt: all providers
ALT2_SET = True              # data-alt2: without Google




def load(name):
    return stratus.load_parquet_from_blob(f"{PREFIX}/{name}.parquet", stage=STAGE)


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def data_table(headers, rows, cls="data"):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<div class="tablewrap">\n<table class="{cls}">\n'
            f"<thead><tr>{head}</tr></thead>\n<tbody>{body}</tbody>\n</table>\n</div>")


def rp_choices(n):
    return [r for r in [3, 4, 5, 6] if RP_FLOOR <= r <= n // 4]


def rp_of(count, n=N_YEARS):
    return f"{(n + 1) / count:.1f} yr" if count else "never"


# --------------------------------------------------------------------- inputs
print("loading ...")
lv = load("swalim_levels")
lv["date"] = pd.to_datetime(lv["date"])
bench = load("workflow/som_flood_benchmark_seasonal")
th = load("swalim_thresholds").set_index("station")
dd = pd.concat([load(f"discharge_daily_{m}").assign(src=m) for m in MODELS],
               ignore_index=True)
dd["date"] = pd.to_datetime(dd["date"])
# a flood year needs two or more of the river's gauges, not one reference
any_flood, severe = envelope_search.benchmark_years_from_gauges(lv)


def reforecast_band(lead_lo, lead_hi, sources):
    """Daily series per station over a lead band: ensemble median, best lead."""
    parts = []
    for name, key in [("reforecast_google_grrr", "google_grrr"),
                      ("reforecast_glofas_v4", "glofas_v4"),
                      ("reforecast_glofas_v4_lead8_12", "glofas_v4")]:
        if key not in sources:
            continue
        try:
            d = load(name)
        except Exception as exc:
            print(f"  ! {name}: {type(exc).__name__}")
            continue
        d["valid_time"] = pd.to_datetime(d["valid_time"])
        parts.append(d.assign(src=key))
    if not parts:
        return None, {}
    fc = pd.concat(parts, ignore_index=True)
    fc = fc[(fc.leadtime_days >= lead_lo) & (fc.leadtime_days <= lead_hi)]
    if fc.empty:
        return None, {}
    med = (fc.groupby(["src", "station", "valid_time", "leadtime_days"])["discharge"]
           .median()
           .groupby(level=["src", "station", "valid_time"]).max()
           .reset_index().rename(columns={"valid_time": "date"}))
    spans = {s: set(range(int(g.date.dt.year.min()), int(g.date.dt.year.max()) + 1))
             for s, g in med.groupby("src")}
    return med, spans


print("building forecast series ...")
fc_action, fc_action_spans = reforecast_band(*ACTION_LEADS, FC_MODELS)
fc_ready, fc_ready_spans = reforecast_band(*READINESS_LEADS, [READINESS_MODEL])


# ------------------------------------------------------------- the action leg
def action_leg(basis, drop_google=False, drop_geoglows=False):
    """One model per window, all points, thresholds at or above 1-in-3."""
    if basis == "reanalysis":
        frame, models, span, excluded = dd, list(MODELS), set(SPAN), []
    else:
        if fc_action is None:
            return {"error": "no forecast archive available"}
        frame = fc_action
        models = [m for m in FC_MODELS
                  if len(fc_action_spans.get(m, set()) & SPAN) >= 4 * RP_FLOOR]
        excluded = [m for m in FC_MODELS if m not in models]
        if not models:
            return {"error": f"no forecast archive can carry a 1-in-{RP_FLOOR}"}
        span = set.intersection(*[fc_action_spans[m] & SPAN for m in models])
    if drop_google:
        models = [m for m in models if m != "google_grrr"]
        if not models:
            return {"error": "nothing left once Google is removed"}
    if drop_geoglows:
        models = [m for m in models if m != "geoglows"]
        if not models:
            return {"error": "nothing left once GEOGloWS is removed"}
    rps = rp_choices(len(span))
    if not rps:
        return {"error": (f"{len(span)} years ({min(span)}-{max(span)}) cannot carry "
                          f"a 1-in-{RP_FLOOR} threshold, and 1-in-2 is not allowed")}
    cands = {k: model_selection.window_candidates(
        frame, lv, k[0], k[1], models=models, span=span, rps=rps) for k in WINDOWS}
    pinned = False  # one model per RIVER is searched, not read from constants
    off_target = False
    if pinned:
        combo = []
        for k in WINDOWS:
            spec = TRIGGER_CONFIG[k]
            hit = [c for c in cands[k] if c["model"] == spec["source"]
                   and c["rp"] == spec["rp"] and c["n_req"] == spec["n_req"]]
            if not hit:
                pinned, combo = False, None
                break
            combo.append(hit[0])
        if combo is not None:
            res = model_selection.evaluate(tuple(combo), any_flood, severe, span=span)
    if not pinned:
        # one source per river AND season (directive 2026-08-27): four choices,
        # never mixed inside a window
        # never more often than the target: the record quantises the
        # achievable rates, so 1-in-3.2 is the nearest at or rarer than 1-in-3
        _, best, frontier = model_selection.choose(
            cands, any_flood, severe, span=span,
            min_rp=float(ENVELOPE_TARGET_RP))
        if best is None and frontier:
            best = min(frontier.values(),
                       key=lambda t: abs(t[2]["env_rp"] - ENVELOPE_TARGET_RP))
            off_target = True
        if best is None:
            return {"error": "no configuration could be scored"}
        combo, res = list(best[1]), best[2]
    return {
        "basis": basis, "drop_google": drop_google, "excluded": excluded,
        "years": [min(span), max(span)], "n_years": len(span),
        "rps_available": rps, "off_target": off_target, "pinned": pinned,
        "envelope": res,
        "windows": {k: {"source": c["model"], "rp": c["rp"], "n_req": c["n_req"],
                        "n_of": len(c["stations"]), "stations": c["stations"],
                        "rho": c["rho"], "years": sorted(c["years"])}
                    for k, c in zip(WINDOWS, combo)},
    }


print("action leg ...")
action = {
    ("reanalysis", False): action_leg("reanalysis"),
    ("reanalysis", True): action_leg("reanalysis", drop_google=True),
    ("forecast", False): action_leg("forecast"),
    ("forecast", True): action_leg("forecast", drop_google=True),
}
# the third provider set: GEOGloWS out, so every window sits on a model whose
# return periods can be fitted on a forecast archive
action[("reanalysis", "nogeoglows")] = action_leg("reanalysis", drop_geoglows=True)
action[("forecast", "nogeoglows")] = action_leg("forecast", drop_geoglows=True)
for k, v in action.items():
    if "error" in v:
        print(f"  {k}: {v['error']}")
    else:
        e = v["envelope"]
        print(f"  {k[0]:10s} drop_google={k[1]!s:5s} 1-in-{e['env_rp']} | "
              f"severe {e['severe_caught']}/{e['n_severe']}")


# ---------------------------------------------------------- the readiness leg
def readiness_leg(action_ref):
    """Leads 7-12 on GloFAS v4, carrying the action window's own rule shape.

    Readiness is not required to precede activation (directive 2026-08-27), so
    nothing is tuned to cover the action years: the window keeps its votes and
    return period, the thresholds are refitted on the readiness series, and how
    often it happens to lead an activation is simply reported.
    """
    if fc_ready is None:
        return {"error": "no reforecast covers leads 7-12"}
    span = fc_ready_spans.get(READINESS_MODEL, set()) & SPAN
    rps = rp_choices(len(span))
    if not rps:
        return {"error": (f"{len(span)} years at leads 7-12 cannot carry a "
                          f"1-in-{RP_FLOOR} threshold")}
    rows = {}
    for k in WINDOWS:
        river, season = k
        months = SEASONS[season]
        aw = action_ref["windows"][k]
        act_years = [y for y in aw["years"] if y in span]
        cols = []
        for st in TRIGGER_STATIONS[river]:
            s = fc_ready[(fc_ready.src == READINESS_MODEL) & (fc_ready.station == st)]
            if s.empty:
                continue
            s = s.set_index("date")["discharge"].sort_index()
            s = s[s.index.month.isin(months) & s.index.year.isin(span)]
            if len(s) < 60:
                continue
            am = s.groupby(s.index.year).max().dropna()
            # the action window's return period, or the rarest this record allows
            rp = aw["rp"] if aw["rp"] in rps else max(rps)
            t = weibull_threshold(am.values, rp)
            if not np.isnan(t):
                cols.append((s >= t).rename(st))
        if len(cols) < 2:
            rows[k] = {"error": "too few points with a usable readiness record"}
            continue
        rp = aw["rp"] if aw["rp"] in rps else max(rps)
        n_req = max(2, min(aw["n_req"], len(cols) - 1))
        mat = pd.concat(cols, axis=1).fillna(False)
        mx = mat.sum(axis=1).groupby(mat.index.year).max()
        fires = sorted(set(mx[mx >= n_req].index) & span)
        rows[k] = {
            "rp": rp, "n_req": n_req, "n_of": len(cols), "fires": fires,
            "covers": [y for y in act_years if y in fires],
            "n_action": len(act_years),
            "severe_caught": len(set(fires) & severe),
        }
    return {"model": READINESS_MODEL, "leads": list(READINESS_LEADS),
            "years": [min(span), max(span)], "n_years": len(span),
            "rps_available": rps, "windows": rows}


print("readiness leg ...")
ready = {
    False: readiness_leg(action[("reanalysis", False)]),
    True: readiness_leg(action[("reanalysis", True)]),
    "nogeoglows": readiness_leg(action[("reanalysis", "nogeoglows")]),
}
for g, v in ready.items():
    if "error" in v:
        print(f"  drop_google={g}: {v['error']}")
    else:
        for k, w in v["windows"].items():
            if "error" in w:
                print(f"  {WLABEL[k]:16s} {w['error']}")
            else:
                print(f"  {WLABEL[k]:16s} 1-in-{w['rp']} {w['n_req']}/{w['n_of']} | "
                      f"fires {len(w['fires'])} | covers {len(w['covers'])} of "
                      f"{w['n_action']} action years")


# ------------------------------------------------------------------- figures
def flood_years(river, season, rp=BENCHMARK_RP):
    return envelope_search.gauge_consensus_years(lv, river, season, rp)


def severe_window_years(river, season):
    return envelope_search.gauge_consensus_years(lv, river, season, SEVERE_RP)


def model_season(model, station, months):
    s = dd[(dd.src == model) & (dd.station == station)].set_index("date")["discharge"]
    s = s[s.index.month.isin(months)]
    return s[(s.index.year >= Y0) & (s.index.year <= Y1)].sort_index()


def ctx_for(act):
    return {
        "lv": lv, "th": th, "dd": dd, "Y0": Y0, "Y1": Y1, "span": SPAN,
        "model_season": model_season, "flood_years": flood_years,
        "severe_window_years": severe_window_years, "window_model": WINDOW_MODEL,
        "per_window": {k: v["years"] for k, v in act["windows"].items()},
        "reference_gauge": REFERENCE_GAUGE, "station_scores": [], "rho": {},
        "reforecast": lambda: None, "exposure": lambda: None,
        "frontier": [], "adopted_point": None, "n_severe": len(severe),
        "target_rp": ENVELOPE_TARGET_RP,
    }


def as_png(svg_name, png_name):
    """summary_figures writes SVG; this page's figs are PNG."""
    src = FIGS / f"{svg_name}.svg"
    if src.exists():
        src.unlink()


OBS_SLUGS = {"belet_weyne": "beletweyne", "bulo_burti": "buloburte",
             "luuq": "luuq", "bardheere": "bardheere"}


def tail_ratios():
    """{(station, source): {rp: model return level / observed return level}}."""
    out = {}
    for key, slug in OBS_SLUGS.items():
        try:
            raw = stratus.load_blob_data(
                f"ds-aa-som-floods/raw/ef5/analysis/{slug}_real_discharge.json",
                stage="dev")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! observed discharge for {key}: {type(exc).__name__}")
            continue
        obs = pd.Series(json.loads(raw), dtype=float)
        obs.index = pd.to_datetime(obs.index)
        obs = obs.sort_index().dropna()
        obs = obs[obs > 0]
        for src in MODELS:
            mod = dd[(dd.src == src) & (dd.station == key)].set_index("date")["discharge"]
            j = pd.concat([mod, obs], axis=1, join="inner").dropna()
            j.columns = ["mod", "obs"]
            if len(j) < 500:
                continue
            am = j.groupby(j.index.year).max()
            am = am[[(j.index.year == y).sum() >= 200 for y in am.index]]
            vals = {}
            for rp in (3, 4, 5, 6):
                tm = weibull_threshold(am["mod"].values, rp)
                to = weibull_threshold(am["obs"].values, rp)
                if to and to > 0 and not np.isnan(tm):
                    vals[rp] = tm / to
            if vals:
                out[(key, src)] = vals
    return out


def draw_tail(path, models=None):
    """Model over observed discharge at RP3 to RP6, one panel per gauge."""
    models = models or MODELS
    tr = tail_ratios()
    if not tr:
        return None
    stns = [s for s in OBS_SLUGS if any(k[0] == s for k in tr)]
    fig, axes = plt.subplots(1, len(stns), figsize=(3.2 * len(stns), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, stns):
        ax.axhline(1.0, color="#5C6B7A", lw=1.2, zorder=1)
        for src in models:
            v = tr.get((st, src))
            if not v:
                continue
            xs = sorted(v)
            ax.plot(xs, [v[x] for x in xs], marker="o", ms=4.5, lw=2,
                    color=SOURCE_COLORS[src], label=NICE.get(src, src), zorder=3)
        ax.set_yscale("log")
        ax.set_xticks([3, 4, 5, 6])
        ax.set_xlabel("return period (years)")
        ax.set_title(st.replace("_", " ").title(), fontsize=11)
        ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 5, 10])
        ax.set_yticklabels(["0.1x", "0.25x", "0.5x", "1x", "2x", "5x", "10x"])
        style_ax(ax, grid="y")
    axes[0].set_ylabel("model / observed discharge")
    axes[-1].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path.name


def rp3_detection():
    """[{station, source, n_events, POD, FAR}] on observed RP3 crossings."""
    out = []
    for river in TRIGGER_STATIONS:
        for st in TRIGGER_STATIONS[river]:
            obs = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
            modern = obs[obs.index.year >= 2000]
            am = modern.groupby(modern.index.year).max().dropna()
            base = weibull_level(am.values, RP_FLOOR)
            if np.isnan(base):
                continue
            for model in MODELS:
                m = dd[(dd.src == model) & (dd.station == st)].set_index("date")["discharge"]
                m = m[(m.index.year >= 2002) & (m.index.year <= Y1)].sort_index()
                if len(m) < 500:
                    continue
                t = weibull_threshold(
                    m.groupby(m.index.year).max().dropna().values, RP_FLOOR)
                if np.isnan(t):
                    continue
                j = pd.concat([m.rename("mod"), obs.rename("obs")], axis=1,
                              join="inner").dropna()
                if len(j) < 500:
                    continue
                oe, me = episodes(j["obs"] >= base), episodes(j["mod"] >= t)
                if not oe:
                    continue
                out.append({"station": st, "source": model, "n_events": len(oe),
                            "POD": hits(oe, me) / len(oe),
                            "FAR": (1 - hits(me, oe) / len(me)) if me else np.nan})
    return out


def draw_detection(path, models=None):
    """Hit rate and false-alarm rate per gauge on RP3 events, per provider set."""
    models = models or MODELS
    sc = pd.DataFrame(rp3_detection())
    if sc.empty:
        return None
    ylab, pod, far = [], [], []
    for river in TRIGGER_STATIONS:
        for st in TRIGGER_STATIONS[river]:
            sub = sc[sc.station == st]
            if sub.empty:
                continue
            ylab.append(f"{st.replace('_', ' ').title()}  n={int(sub.n_events.iloc[0])}")
            pod.append([float(sub[sub.source == m].POD.iloc[0])
                        if len(sub[sub.source == m]) else np.nan for m in models])
            far.append([float(sub[sub.source == m].FAR.iloc[0])
                        if len(sub[sub.source == m]) else np.nan for m in models])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 0.52 * len(ylab) + 2.2))
    summary_figures.heat(axes[0], pod, [NICE[m] for m in models], ylab)
    axes[0].set_title("Hit rate on RP3 events at that gauge\n(dark = better)",
                      fontsize=10.5)
    summary_figures.heat(axes[1], far, [NICE[m] for m in models], [""] * len(ylab),
                         reverse=True)
    axes[1].set_title(
        "Share of the model's own alarms that were false\n(dark = better)",
        fontsize=10.5)
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path.name


def event_mask(station, season, pad_days=10):
    """Days inside an observed RP3-or-rarer event at this gauge, widened by pad."""
    obs = lv[lv.station == station].set_index("date")["level_m"].dropna().sort_index()
    seas = obs[obs.index.month.isin(SEASONS[season])]
    modern = seas[seas.index.year >= 2000]
    am = modern.groupby(modern.index.year).max().dropna()
    base = weibull_level(am.values, RP_FLOOR)
    if np.isnan(base):
        return None, None
    ev = episodes(seas >= base)
    if not ev:
        return None, None
    pad = pd.Timedelta(days=pad_days)
    keep = pd.Series(False, index=obs.index)
    for a, b in ev:
        keep |= (obs.index >= a - pad) & (obs.index <= b + pad)
    return obs[keep], len(ev)


def draw_model_choice(path, models=None, chosen_from=None):
    """Which model tracks the gauges best on RP3+ events, per river and season."""
    models = models or MODELS
    # "base"/"all" both adopt the same configuration; the stored action
    # dict is keyed by the computation basis, so mark from the adopted run
    chosen_from = "nogeoglows"
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.7), sharey=True)
    for ax, k in zip(axes, WINDOWS):
        river, season = k
        vals, pods, n_ev = {}, {}, 0
        for m in models:
            rhos, hit = [], []
            for st in TRIGGER_STATIONS[river]:
                obs_ev, n = event_mask(st, season)
                if obs_ev is None:
                    continue
                n_ev = max(n_ev, n)
                mod = model_season(m, st, SEASONS[season])
                if not len(mod):
                    continue
                sub = mod[mod.index.isin(obs_ev.index)]
                if len(sub) >= 60:
                    rhos.append(model_selection.best_lag_rho(sub, obs_ev))
                # and how many of those events the model was above its own RP3 for
                am = mod.groupby(mod.index.year).max().dropna()
                t = weibull_threshold(am.values, RP_FLOOR)
                if np.isnan(t):
                    continue
                full = lv[lv.station == st].set_index("date")["level_m"].dropna()
                full = full[full.index.month.isin(SEASONS[season])].sort_index()
                fam = full[full.index.year >= 2000]
                fam = fam.groupby(fam.index.year).max().dropna()
                base = weibull_level(fam.values, RP_FLOOR)
                if np.isnan(base):
                    continue
                j = pd.concat([mod.rename("mod"), full.rename("obs")], axis=1,
                              join="inner").dropna()
                if len(j) < 200:
                    continue
                oe, me = episodes(j["obs"] >= base), episodes(j["mod"] >= t)
                if oe:
                    hit.append(hits(oe, me) / len(oe))
            if rhos:
                vals[m] = float(np.mean(rhos))
                pods[m] = float(np.mean(hit)) if hit else np.nan
        if not vals:
            continue
        chosen = action[("reanalysis", "nogeoglows")]["windows"][k]["source"]
        names = list(vals)
        ax.bar(range(len(names)), [vals[m] for m in names],
               color=[SOURCE_COLORS.get(m, "#1C7293") for m in names], width=0.62)
        for i, m in enumerate(names):
            if m == chosen:
                ax.plot(i, vals[m] + 0.04, marker="v", color=INK, markersize=8)
            ax.text(i, max(vals[m] - 0.07, 0.03), f"{vals[m]:.2f}", ha="center",
                    fontsize=8.5, color="white")
            if not np.isnan(pods.get(m, np.nan)):
                ax.text(i, 0.03, f"hit {pods[m]:.2f}", ha="center", fontsize=7.5,
                        color="white")
        ax.set_xticks(range(len(names)),
                      [NICE[m].replace(" ", "\n") for m in names], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{WLABEL[k]}  ({n_ev} events)", fontsize=10.5)
        style_ax(ax, grid="y")
    axes[0].set_ylabel("mean rho on RP3+ events")
    fig.suptitle("Which model works best per river and season, judged on 1-in-3 or "
                 "rarer events only (marker = adopted)",
                 x=0.06, ha="left", fontweight="bold", fontsize=11.5, color=INK)
    fig.subplots_adjust(top=0.78, wspace=0.08)
    fig.savefig(path, format="png", dpi=115, bbox_inches="tight")
    plt.close(fig)


# one copy of each comparison figure per switch state
draw_model_choice(FIGS / "a_selection.png", SET_MODELS["base"], "base")
draw_model_choice(FIGS / "a_selection_all.png", SET_MODELS["all"], "all")
draw_tail(FIGS / "g_tail.png", SET_MODELS["base"])
draw_tail(FIGS / "g_tail_all.png", SET_MODELS["all"])
draw_detection(FIGS / "h_detection.png", SET_MODELS["base"])
draw_detection(FIGS / "h_detection_all.png", SET_MODELS["all"])
for stale in ["_alt", "_altpng", "_ng", "_ngpng"]:
    shutil.rmtree(FIGS / stale, ignore_errors=True)
for svg in FIGS.glob("*.svg"):
    svg.unlink()
# figures inherited unchanged from the source document
for keep in ["f_own_skill.png", "i_peaks_reanalysis.png", "i_peaks_reforecast.png",
             "i_metric_plane.png", "e_anticipation.png", "j2_impact_shares.png"]:
    if (SRC_FIGS / keep).exists():
        shutil.copy2(SRC_FIGS / keep, FIGS / keep)
print("  figs:", sorted(p.name for p in FIGS.glob("*.png")))


# --------------------------------------------------- the JSON-driven tables
def write_selection_detail():
    """Per-point tracking correlation for every model, with the pick marked.

    Same table the published document shows, but "selected" is now the single
    source that carries the whole window, not a per-station pick.
    """
    # still reporting means reporting now, not merely past the calibration
    # window: Bardheere ends 2023-11-30 and Bualle 2024-03-14, so five remain
    last = lv.dropna(subset=["level_m"]).groupby("station")["date"].max()
    live = set(last[last.dt.year >= last.max().year].index)
    out = {}
    for k in WINDOWS:
        river, season = k
        ref = lv[lv.station == REFERENCE_GAUGE[river]].set_index("date")["level_m"]
        ref = ref[ref.index.month.isin(SEASONS[season])].sort_index()
        chosen = std["windows"][k]["source"]
        rows = []
        for st in TRIGGER_STATIONS[river]:
            row = {"station": st, "gauge_active": st in live,
                   "selected": [chosen]}
            for m in MODELS:
                ser = model_season(m, st, SEASONS[season])
                if not len(ser):
                    continue
                best, best_lag = 0.0, 0
                for lag in range(-10, 31):
                    j = pd.concat([ser, ref.shift(-lag)], axis=1,
                                  join="inner").dropna()
                    if len(j) < 60:
                        continue
                    r = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
                    if pd.notna(r) and abs(r) > abs(best):
                        best, best_lag = float(r), lag
                if best:
                    row[m] = {"rho": round(best, 3), "lag": best_lag}
            rows.append(row)
        out[f"{river}_{season}"] = rows
    (OUT / "selection_detail.json").write_text(json.dumps(out, indent=1),
                                               encoding="utf-8")
    return out


def write_activation_impact():
    """Activations against the impact record, year by year.

    The EM-DAT and CERF entries are taken verbatim from the published
    document: they are observations, not model output. Everything else, the
    per-window counts and whether the window activated, is recomputed here for
    both provider sets.
    """
    src_path = SRC_PAGE.parent / "activation_impact.json"
    impact = {}
    meta_note = ""
    if src_path.exists():
        old = json.loads(src_path.read_text(encoding="utf-8"))
        meta_note = old.get("meta", {}).get("note", "")
        for row in old.get("rows", []):
            for river, seasons in row.get("rivers", {}).items():
                for season, w in seasons.items():
                    impact[(row["year"], river, season)] = (w.get("emdat"),
                                                            w.get("cerf"))

    def counts(act, k):
        """Peak simultaneous points over threshold, per year, for one window."""
        river, season = k
        w = act["windows"][k]
        cols = []
        for st in TRIGGER_STATIONS[river]:
            ser = model_season(w["source"], st, SEASONS[season])
            if len(ser) < 100:
                continue
            am = ser.groupby(ser.index.year).max().dropna()
            t = weibull_threshold(am.values, w["rp"])
            if not np.isnan(t):
                cols.append((ser >= t).rename(st))
        if not cols:
            return {}
        mat = pd.concat(cols, axis=1).fillna(False)
        return mat.sum(axis=1).groupby(mat.index.year).max().to_dict()

    n_std = {k: counts(std, k) for k in WINDOWS}
    n_alt = {k: counts(alt, k) for k in WINDOWS}
    rows = []
    for year in range(Y0, Y1 + 1):
        rivers = {}
        for river in TRIGGER_STATIONS:
            seasons = {}
            for season in SEASONS:
                k = (river, season)
                em, cerf = impact.get((year, river, season), (None, None))
                sev = year in severe_window_years(river, season)
                mod = year in flood_years(river, season)
                seasons[season] = {
                    "n": int(n_std[k].get(year, 0)),
                    "nv": int(n_alt[k].get(year, 0)),
                    "adopted": year in std["windows"][k]["years"],
                    "adopted_v": year in alt["windows"][k]["years"],
                    "bench": "severe" if sev else ("moderate" if mod else ""),
                    "emdat": em,
                    "cerf": cerf,
                }
            rivers[river] = seasons
        rows.append({"year": year, "basins": rivers})

    def meta_for(act):
        return {
            f"{r}_{sn}": {
                "label": f"{NICE[w['source']]} at {w['n_of']} points",
                "n_req": w["n_req"], "pool": w["n_of"], "rp": w["rp"],
            }
            for (r, sn), w in act["windows"].items()
        }

    payload = {
        "meta": {
            "note": ("n = peak simultaneous count of monitored points over their "
                     "thresholds in that season, at the adopted return period; "
                     + (meta_note.split(";", 1)[1].strip() if ";" in meta_note
                        else "EM-DAT and CERF records carried over unchanged")),
            "windows": meta_for(std),
            "windows_v": meta_for(alt),
        },
        "rows": rows,
    }
    (OUT / "activation_impact.json").write_text(json.dumps(payload, indent=1),
                                                encoding="utf-8")
    return payload


# -------------------------------------------------------------------- sections
def vswap(std, alt, alt2=None):
    """The document's provider swap: adopted, without Google, without GEOGloWS."""
    extra = "" if alt2 is None else f' data-alt2="{esc(alt2)}"'
    return f'<td class="vswap" data-alt="{esc(alt)}"{extra}>{std}</td>'


def glance_table():
    std, alt = action[("reanalysis", BASE_SET)], action[("reanalysis", ALT_SET)]
    ng = action[("reanalysis", ALT2_SET)]
    rst, ralt = ready[BASE_SET], ready[ALT_SET]
    rng = ready[ALT2_SET]
    rows = []
    for k in WINDOWS:
        w = std["windows"][k]
        wa = alt["windows"][k] if "error" not in alt else w
        wg = ng["windows"][k] if "error" not in ng else w
        rd = rst["windows"].get(k, {}) if "error" not in rst else {}
        rda = ralt["windows"].get(k, {}) if "error" not in ralt else {}
        rdg = rng["windows"].get(k, {}) if "error" not in rng else {}
        act_std = (f"{NICE[w['source']]}: {w['n_req']} of {w['n_of']} points over "
                   f"their 1-in-{w['rp']}-yr thresholds")
        act_alt = (f"{NICE[wa['source']]}: {wa['n_req']} of {wa['n_of']} points over "
                   f"their 1-in-{wa['rp']}-yr thresholds")
        act_ng = (f"{NICE[wg['source']]}: {wg['n_req']} of {wg['n_of']} points over "
                  f"their 1-in-{wg['rp']}-yr thresholds")
        rd_ng = (f"{NICE[READINESS_MODEL]} ens-median: {rdg['n_req']} of "
                 f"{rdg['n_of']} over 1-in-{rdg['rp']}-yr" if "rp" in rdg else "-")
        rd_std = (f"{NICE[READINESS_MODEL]} ens-median: {rd['n_req']} of {rd['n_of']} "
                  f"over 1-in-{rd['rp']}-yr" if "rp" in rd else "no rule fits")
        rd_alt = (f"{NICE[READINESS_MODEL]} ens-median: {rda['n_req']} of "
                  f"{rda['n_of']} over 1-in-{rda['rp']}-yr" if "rp" in rda
                  else rd_std)
        rows.append(
            "<tr><td>" + WLABEL[k] + "</td>"
            + vswap(act_std, act_alt, act_ng)
            + vswap(rp_of(len(w["years"])), rp_of(len(wa["years"])),
                    rp_of(len(wg["years"])))
            + vswap(rd_std, rd_alt, rd_ng)
            + vswap(rp_of(len(rd.get("fires", [])), rst.get("n_years", N_YEARS))
                    if "rp" in rd else "-",
                    rp_of(len(rda.get("fires", [])), ralt.get("n_years", N_YEARS))
                    if "rp" in rda else "-")
            + "</tr>"
        )
    head = ("<th>window</th><th>action trigger (leads 1-7 d)</th><th>leg RP</th>"
            "<th>readiness (leads 7-12 d)</th><th>readiness RP</th>")
    return ('<div class="tablewrap">\n<table class="data">\n'
            f"<thead><tr>{head}</tr></thead>\n<tbody>"
            + "".join(rows) + "</tbody>\n</table>\n</div>")


def bookkeeping_table():
    std, alt = action[("reanalysis", BASE_SET)], action[("reanalysis", ALT_SET)]
    ng = action[("reanalysis", ALT2_SET)]
    rows = []
    for k in WINDOWS:
        w, wa = std["windows"][k], alt["windows"][k]
        wg = ng["windows"][k] if "error" not in ng else w
        rows.append(
            f"<tr><td>individual</td><td>{WLABEL[k]} action</td>"
            + vswap(f"{len(w['years'])} &mdash; "
                    + ", ".join(str(y) for y in w["years"]),
                    f"{len(wa['years'])} &mdash; "
                    + ", ".join(str(y) for y in wa["years"]),
                    f"{len(wg['years'])} &mdash; "
                    + ", ".join(str(y) for y in wg["years"]))
            + vswap(rp_of(len(w["years"])), rp_of(len(wa["years"])),
                    rp_of(len(wg["years"])))
            + "</tr>"
        )
    for river in TRIGGER_STATIONS:
        yrs = sorted({y for k, w in std["windows"].items() if k[0] == river
                      for y in w["years"]})
        yrs_a = sorted({y for k, w in alt["windows"].items() if k[0] == river
                        for y in w["years"]})
        yrs_g = sorted({y for k, w in ng["windows"].items() if k[0] == river
                        for y in w["years"]}) if "error" not in ng else yrs
        rows.append(
            f"<tr><td>river</td><td>{river.capitalize()} (Gu or Deyr)</td>"
            + vswap(str(len(yrs)), str(len(yrs_a)), str(len(yrs_g)))
            + vswap(rp_of(len(yrs)), rp_of(len(yrs_a)), rp_of(len(yrs_g)))
            + "</tr>"
        )
    e, ea = std["envelope"], alt["envelope"]
    eg = ng["envelope"] if "error" not in ng else e
    rows.append(
        "<tr><td>overall</td><td>action, either river</td>"
        + vswap(str(e["fires"]), str(ea["fires"]), str(eg["fires"]))
        + vswap(f"{e['env_rp']} yr", f"{ea['env_rp']} yr", f"{eg['env_rp']} yr")
        + "</tr>"
    )
    head = ("<th>level</th><th>trigger</th><th>activations (1999-2023)</th>"
            "<th>RP</th>")
    return ('<div class="tablewrap">\n<table class="data">\n'
            f"<thead><tr>{head}</tr></thead>\n<tbody>"
            + "".join(rows) + "</tbody>\n</table>\n</div>")


def forecast_panel():
    v = action[("forecast", False)]
    vg = action[("forecast", True)]
    if "error" in v and "error" in vg:
        return (f'<p class="muted">A forecasts-only calibration is not available: '
                f'{esc(v["error"])}.</p>')
    use = v if "error" not in v else vg
    rows = [
        (WLABEL[k], NICE[w["source"]], f"1-in-{w['rp']}",
         f"{w['n_req']} of {w['n_of']}",
         ", ".join(str(y) for y in w["years"]) or "never")
        for k, w in use["windows"].items()
    ]
    e = use["envelope"]
    note = ("" if not use["off_target"] else
            " This is the closest rate the forecast record supports; nothing lands "
            f"on 1-in-{ENVELOPE_TARGET_RP}.")
    excl = ""
    if use.get("excluded"):
        excl = (" Out of the field: "
                + ", ".join(NICE[m] for m in use["excluded"])
                + f", whose archive is too short to carry a 1-in-{RP_FLOOR} "
                "threshold, so it can only be calibrated on the retrospective.")
    return (
        f"<p>Refitting the same rule on the forecast archives themselves, "
        f"{use['years'][0]}&ndash;{use['years'][1]}, thresholds from 1-in-"
        f"{min(use['rps_available'])} to 1-in-{max(use['rps_available'])}.{excl}</p>"
        + data_table(["window", "source", "gauge threshold", "points that must agree",
                      "would have activated in"], rows)
        + f"<p><strong>Envelope 1-in-{e['env_rp']}</strong> ({e['fires']} of "
        f"{use['n_years']} years), catching {e['severe_caught']} of the "
        f"{e['n_severe']} severe years.{note}</p>"
    )


std = action[("reanalysis", BASE_SET)]
alt = action[("reanalysis", ALT_SET)]
env, env_alt = std["envelope"], alt["envelope"]

SECTIONS = {
    "The mechanism at a glance": f"""
    <p>Four river-season windows, each running on <strong>one</strong> forecast source
      rather than a mixture. Inside a window, every monitored point's flow is compared
      with its own return-period threshold and the window activates when enough points are
      over their thresholds <strong>on the same day</strong>. A season where the points
      cross on different days does not count. This rarely matters in practice: flood flows
      stay high for weeks, and in every past activation the points crossed within 0 to 5
      days of each other, upstream gauges first (see "How far apart do the points cross?"
      below). At least two points must agree, so no single point releases the
      money, and never all of them, so one quiet point cannot block it.</p>
    <p>All seven reporting-era points are monitored: four on the Juba (Luuq, Dollow,
      Bardheere, Bualle) and three on the Shabelle (Belet Weyne, Bulo Burti, Jowhar). No
      threshold, on either leg, sits below 1-in-3.</p>
{glance_table()}
    <div class="callout">
      <strong>The envelope.</strong> The full amount is released whenever any window
      activates, so the union is what the 1-in-3 target applies to. As configured it would
      have released in <strong>{env['fires']} of {N_YEARS} years, once every
      {env['env_rp']} years</strong>, catching {env['severe_caught']} of the
      {env['n_severe']} years in which two or more of a river's gauges recorded a 1-in-{SEVERE_RP} or rarer season,
      and never activating in a year with no recorded flood.
    </div>
    <div class="callout warn altonly">
      <strong>GEOGloWS carries a window here, with conditions.</strong> It is the only model
      in the field that cannot be operated exactly as calibrated. Its forecasts run
      below its own retrospective, so a threshold fitted on the retrospective sits too
      low on the live forecast and would activate too often: the threshold has to be
      refitted on the forecast archive before use. That archive only begins in
      July 2024, so there is no way to backtest a GEOGloWS window at lead time yet, and
      its per-gauge event detection was the weakest of the four models on the
      reanalysis. Bias correcting it does not fix this: the SFDC correction lowers
      detection at every gauge. Treat Juba Gu as provisional, and revisit it once two
      or three Gu seasons of GEOGloWS forecasts exist.
    </div>
    <div class="callout warn">
      <strong>What the two-gauge benchmark sees, and what it misses.</strong> A
      river-season counts as a flood only when two or more of that river's gauges cross
      their own level, which keeps one record from deciding the benchmark but has three
      consequences worth stating.
      <ul>
        <li><strong>It agrees with the impact record on the big years.</strong> All five
          costliest years in EM-DAT and CERF terms &mdash; 2006, 2018, 2019, 2020 and
          2023 &mdash; are severe under the rule.</li>
        <li><strong>1999 to 2001 cannot be assessed.</strong> The Juba had no gauge
          reporting and the Shabelle only one, so no consensus is possible and those
          years read as quiet rather than as unknown. Nothing activates before 2005, so
          no year is wrongly scored as a miss, but the benchmark effectively starts in
          2002.</li>
        <li><strong>2021 is a genuine gap.</strong> EM-DAT records 400,000 people
          affected, yet only Bardheere on the Juba and Belet Weyne on the Shabelle
          crossed, so the rule reads no flood. Belet Weyne's peak that year sits exactly
          at bank full (8.30 m), where the gauge record is censored and the true level is
          unknown, so consensus can be understated in exactly the years that matter
          most.</li>
        <li><strong>2016 is the reverse case.</strong> Three of the four Juba gauges and
          all three Shabelle gauges crossed their levels in Gu, but EM-DAT records nobody
          affected. Gauge levels and recorded impact are not the same measurement.</li>
      </ul>
    </div>
    <h3>What happens without Google Flood Hub</h3>
    <p>Use the provider switch above to see it. With Google removed the windows run on
      GloFAS, the envelope sits at <strong>1-in-{env_alt['env_rp']}</strong>
      ({env_alt['fires']} of {N_YEARS} years) and severe-year coverage is
      {env_alt['severe_caught']} of {env_alt['n_severe']}: the same coverage, so on this
      evidence dropping Google costs nothing measurable.</p>
    <figure><img src="figs/map_stations.png" alt="The seven monitored points">
      <figcaption>The seven points the trigger watches, four on the Juba and three on
        the Shabelle.</figcaption></figure>
""",



    "The readiness leg (7–12 days)": f"""
    <p>Readiness runs on {NICE[READINESS_MODEL]} ensemble-median forecasts at leads 7 to
      12, the only archive covering that band, over the same full set of points, with
      thresholds refitted on that series. It releases only the mobilisation share and is
      held to the same floor: <strong>no threshold below 1-in-3</strong>, which is the
      change from the published mechanism, where readiness sat on 1-in-2 levels.</p>
{data_table(
    ["window", "readiness rule", "readiness years", "covers action years"],
    [
        (
            WLABEL[k],
            (f"{w['n_req']} of {w['n_of']} points over 1-in-{w['rp']}"
             if "rp" in w else w.get("error", "-")),
            ", ".join(str(y) for y in w.get("fires", [])) or "-",
            (f"{len(w['covers'])} of {w['n_action']}" if "covers" in w else "-"),
        )
        for k, w in (ready[False].get("windows", {}) or {}).items()
    ],
)}
    <p>Readiness carries each window's own rule, the same votes and the same return
      period, refitted on the 7-to-12-day series. It is not tuned to precede activation:
      an action trigger may activate with no readiness phase ahead of it, which is accepted
      (decision 2026-08-27). The last column reports how often readiness did lead an
      activation, as an observation rather than a requirement.</p>
""",
    "Return-period bookkeeping": f"""
{bookkeeping_table()}
    <p>Under all-in funding, where any activation releases the full envelope, the
      effective return period is the overall row: <strong>1-in-{env['env_rp']}</strong>
      with Google, 1-in-{env_alt['env_rp']} without. The individual windows are set
      rarer than that on purpose, because four windows each calibrated to 1-in-3 give a
      union of roughly 1-in-1.5.</p>
""",
    "Open items before a trigger report": f"""
    <ul>
      <li><strong>Impact years are still undefined.</strong> The trigger is scored
        against gauge levels, not against recorded humanitarian impact. Until impact
        years exist, severe-year coverage is a proxy.</li>
      <li><strong>The reanalysis cannot pick the model.</strong> Many assignments tie,
        so the model per window rests on lead-time skill and on operational
        considerations, not on the backtest.</li>
      <li><strong>Google cannot be calibrated on its own forecasts.</strong> Its
        reforecast starts in 2016, and a 1-in-3 threshold needs about 12 years, so a
        Google window is necessarily calibrated on the retrospective and only checked at
        lead time.</li>
      <li><strong>Two Juba points can no longer be verified.</strong> Bardheere's gauge
        record ends 2023-11-30 and Bualle's 2024-03-14. Both can still be forecast at,
        but neither can be checked against observations from here on.</li>
      <li><strong>Readiness coverage is uneven by season,</strong> which is accepted
        rather than solved: at 7 to 12 days GloFAS v4 leads Gu activations more reliably
        than Deyr ones, so a Deyr activation may arrive with no readiness phase.</li>
    </ul>
""",
}
# Every section is kept. Three carry diagnostics from the published study that
# do not depend on the trigger rule (seasonal peak comparisons, the selection
# metric plane, each model's skill against its own reanalysis); they are marked
# as carried over rather than silently reused.
DROP = set()
CARRIED_OVER = {
    "Seasonal peaks: model vs gauge",
    "Forecast skill against each model's own reanalysis",
}
CARRY_NOTE = (
    '    <p class="muted"><em>Carried over unchanged from the published '
    "multi-source study: this diagnostic compares models, so it does not "
    "depend on how many points vote or where the thresholds sit.</em></p>\n"
)

FORECAST_PANEL_TOKEN = forecast_panel()
OPERATIONAL_TABLE_TOKEN = data_table(
    ["window", "calibrated on", "run on", "reproduces its calibration years?"],
    [
        (
            WLABEL[k],
            f"{NICE[w['source']]} reanalysis",
            (f"{NICE[w['source']]} forecasts, leads 1-7"
             if w["source"] in FC_MODELS
             else f"{NICE[w['source']]} operationally; lead-time evidence from "
                  f"{NICE['glofas_v4']}"),
            ("yes, same archive"
             if w["source"] in FC_MODELS else
             "cannot be shown directly: no reforecast for this version"),
        )
        for k, w in std["windows"].items()
    ],
)
OPERATIONAL_NOTE_TOKEN = (
    '<div class="callout warn">\n<strong>What the calibration cannot settle.</strong> '
    "Google's reforecast starts in 2016, so a Google window cannot be calibrated on "
    "its own forecasts at a 1-in-3 threshold, which needs about 12 years: it is "
    "calibrated on the retrospective and only checked at lead time. GloFAS v5 has no "
    "reforecast at all, so its window inherits its lead-time evidence from v4. "
    "Readiness is not tuned to precede activation, so an action trigger may fire with "
    "no readiness phase ahead of it.\n</div>"
)

# Sections that keep the template's markup and have only the passages whose
# claims changed rewritten. Each entry is a list of (pattern, replacement)
# applied with re.sub; anything unmatched is left exactly as it was, so
# subsections, figures and script anchors cannot be lost.
SECTION_EDITS = {
    "How the pairs were chosen": [
        (r"<h2([^>]*)>\s*How the pairs were chosen\s*</h2>",
         r"<h2\1>How the model was chosen, one per river and season</h2>"),
        (r"<p>\s*Every combination of the.*?</p>\s*<ul>.*?</ul>",
         "<p>One source per river and season, so four choices. Four steps:</p>\n"
         "<ol>\n"
         "<li><strong>Build the candidate rules.</strong> For a window and a candidate "
         "model, put a threshold at every one of the river's points, taken from that "
         "model's own annual maxima (1-in-3 to 1-in-6). The window activates in a year "
         "when at least N points cross in that season.</li>\n"
         "<li><strong>Score the envelope, not the window.</strong> The money is "
         "released when any of the four windows activates, so candidates are judged on "
         "that union: how often it activates, how many of the severe years it "
         "catches, and how often it activates in a year with no recorded flood.</li>\n"
         "<li><strong>Apply the constraints.</strong> Thresholds never below 1-in-3 nor "
         "above a quarter of the record; all seven points monitored; at least two must "
         "agree but never all of them; every window must activate at least twice in 25 "
         "years.</li>\n"
         "<li><strong>Pick.</strong> Nearest 1-in-3 overall with the most severe years "
         "caught. Ties break on fewer no-flood activations, then on tracking "
         "correlation.</li>\n"
         "</ol>"),
        (r"<p>\s*<strong>Multi-model representation.*?</p>",
         "<p><strong>Why correlation is only a tie-break.</strong> Many model "
         "assignments reproduce the same activation years: with 25 years and 8 severe "
         "events, the threshold and the vote count absorb the difference between "
         "models. Correlation decides only when the envelope cannot.</p>"),
        (r"      <figcaption>The candidate pool\..*?</figcaption>",
         "      <figcaption>Mean best-lag correlation between each model and the "
         "river's reference gauge, per window, across that window's points. The marker "
         "shows the source chosen for that window.</figcaption>"),
        (r'alt="Scatter plots of correlation versus lag[^"]*"',
         'alt="Mean best-lag correlation per window and model"'),
    ],
    "Thresholds and calibration": [
        # the backtest strip swaps by provider set: add the third source
        (r'data-alt-src="figs/c_backtest_strip_nogoogle\.png"',
         'data-alt-src="figs/c_backtest_strip_all.png" '
         'data-alt2-src="figs/c_backtest_strip_nogoogle.png"'),
        (r"<h3[^>]*>\s*Are SWALIM.s official flood-risk levels trustworthy\?\s*</h3>",
         "<h3>How the official SWALIM levels compare with the fitted ones</h3>"),
        (r"shows they imply very different frequencies from gauge to gauge",
         "shows they imply very different frequencies from gauge to gauge"),
        (r"<p>\s*Each selected pair gets its threshold.*?</p>",
         "<p>Each monitored point gets its threshold from its own model's "
         "climatology: the Weibull plotting position of that point's seasonal "
         "maxima, so a model is judged on timing rather than on scale. "
         "<strong>1-in-3 is the floor and a quarter of the record is the "
         "ceiling</strong>, which on 25 years allows 1-in-3 to 1-in-6.</p>"),
        (r"(<h3[^>]*>\s*The tuning surface)",
         """    <h3>How far apart do the points cross?</h3>
    <p>A window activates only when enough points are above their thresholds
      <strong>on the same day</strong>, not just in the same season. How tight are the
      crossings in practice?</p>
    <p>Across every activation, the first and last point to cross fall <strong>0 to 5
      days apart</strong>, and they cross in downstream order: Dollow, Luuq, Bardheere,
      Bualle on the Juba; Belet Weyne, Bulo Burti, Jowhar on the Shabelle. Shabelle Gu
      2018 had all three within a single day, Juba Gu 2018 four points inside three days.
      Because high flows persist for weeks, same-day overlap is reached even where the
      first crossings are a week or more apart, as in Juba Deyr 2023 (11 days between
      Dollow and Bualle, yet four points above at once).</p>
    <p>The exception is instructive. Juba Deyr 2015 had Dollow crossing on 24 October and
      the other three between 10 and 13 November, 20 days later: two separate pulses
      rather than one flood wave. Same-day overlap never reached three, so the window did
      not activate, which is the behaviour we want.</p>
    <div class="callout">
      <strong>What a tolerance window would change.</strong> If crossings within 5 days
      of each other counted together instead of requiring the same day, Juba Deyr would
      add 2018, 2019, 2015 and 2004, and Juba Gu would add 2010. The envelope would move
      from <strong>1-in-3.2 to 1-in-2.4</strong> (8 activations to 11) and activations in
      years two gauges did not record a flood would go from one (2013) to three (2004,
      2013, 2015). Severe-year coverage would <em>not</em> improve, staying at 7 of 8,
      because 2018 and 2019 are already covered by the Shabelle windows. Widening beyond
      5 days changes nothing further: crossings are either within 5 days or 20 days
      apart. The same-day rule is therefore kept, and a tolerance is a lever for later if
      the activation rate is allowed to rise.
    </div>
\\1"""),
        (r"(<h3[^>]*>\s*The tuning surface)",
         "<h3>Calibrated on the reanalysis, checked on the forecasts</h3>\n"
         + FORECAST_PANEL_TOKEN + "\n\\1"),
    ],
    "Would it have worked operationally?": [
        (r"<div class=\"tablewrap\">.*?</div>\s*(?=<div class=\"callout)",
         OPERATIONAL_TABLE_TOKEN),
        (r"<div class=\"callout warn\">\s*<strong>Two operational tuning items.*?</div>",
         OPERATIONAL_NOTE_TOKEN),
    ],
}


def apply_edits(title, html):
    for pattern, repl in SECTION_EDITS.get(title, []):
        html, n = re.subn(pattern, repl, html, count=1, flags=re.S)
        if not n:
            print(f"    ! {title}: no match for {pattern[:46]}")
    return html


# ------------------------------------------------------------------- assemble
print("assembling ...")
template = SRC_PAGE.read_text(encoding="utf-8")
parts = re.split(r"(?=<h2)", template)
head, tail_scripts = parts[0], ""
m = re.search(r"(<script.*)$", parts[-1], re.S)
if m:
    tail_scripts = m.group(1)
    parts[-1] = parts[-1][: m.start()]

kept = []
for p in parts[1:]:
    t = re.search(r"<h2[^>]*>(.*?)</h2>", p, re.S)
    title = re.sub(r"<[^>]+>", "", t.group(1)).strip() if t else ""
    title_key = title.replace("–", "–")
    if title in DROP:
        print(f"  dropped: {title}")
        continue
    replacement = SECTIONS.get(title) or SECTIONS.get(title_key)
    if replacement is None:
        for key in SECTIONS:
            if title.startswith(key[:18]):
                replacement = SECTIONS[key]
                break
    if replacement is None and title in SECTION_EDITS:
        print(f"  edited in place: {title}")
        kept.append(apply_edits(title, p))
        continue
    if replacement is None:
        print(f"  kept as-is: {title}")
        if title in CARRIED_OVER:
            hp = re.search(r"</h2>", p)
            p = p[: hp.end()] + "\n" + CARRY_NOTE + p[hp.end():] if hp else p
        kept.append(p)
        continue
    print(f"  rebuilt: {title}")
    heading = t.group(0) if t else f"<h2>{title}</h2>"
    if replacement.lstrip().startswith("<h2"):
        kept.append(replacement)
    else:
        kept.append(heading + replacement)

body = head + "".join(kept)

# every script is kept: both fetch-driven tables are back, and their JSON is
# regenerated below. The model list the selection table iterates has to grow,
# since GloFAS v4 is now part of the field.
tail_scripts = tail_scripts.replace(
    'var SRC = ["geoglows", "glofas_v5", "google_grrr"];',
    'var SRC = ["geoglows", "glofas_v5", "glofas_v4", "google_grrr"];',
)
tail_scripts = tail_scripts.replace(
    'var SRC_SHORT = { geoglows: "GEO", glofas_v5: "GLO5", google_grrr: "GOO" };',
    'var SRC_SHORT = { geoglows: "GEO", glofas_v5: "GLO5", glofas_v4: "GLO4",'
    ' google_grrr: "GOO" };',
)
tail_scripts = tail_scripts.replace(
    'var SRC_COLOR = { geoglows: "#8E5FA8", glofas_v5: "#B34036", '
    'google_grrr: "#2A78D6" };',
    'var SRC_COLOR = { geoglows: "#8E5FA8", glofas_v5: "#B34036", '
    'glofas_v4: "#EB6834", google_grrr: "#2A78D6" };',
)
# The template ships its own two-state controller. Strip it BEFORE appending,
# otherwise both it and the controller below end up live and fight over the DOM
# (that bug shipped: the published page carried two controllers and a stray
# </body></html> between them).
_sw = tail_scripts.find("// ---- provider-set switch")
if _sw != -1:
    _last = tail_scripts.rfind("})();")
    tail_scripts = tail_scripts[:_sw] + "})();" + tail_scripts[_last + 5:]
body += tail_scripts
body += """
<script>
(function () {
  // three provider sets: 0 all, 1 without Google, 2 without GEOGloWS.
  // Every .vswap element holds the adopted text in the DOM, the no-Google text
  // in data-alt and the no-GEOGloWS text in data-alt2; images use data-alt-src
  // and data-alt2-src. The original is stashed on first use so switching back
  // is exact.
  var MODES = [
    { btn: "vAll", cls: null,
      note: "Adopted: GloFAS + Google, one source per window" },
    { btn: "vNog", cls: "variant",
      note: "All providers: GEOGloWS then carries a window" },
    { btn: "vNoGeo", cls: "variant2",
      note: "Without Google Flood Hub: recalibrated from scratch" }
  ];
  function apply(mode) {
    var m = MODES[mode];
    document.body.classList.toggle("variant", m.cls === "variant");
    document.body.classList.toggle("variant2", m.cls === "variant2");
    var bar = document.getElementById("vbar");
    if (bar) { bar.classList.toggle("alt", mode !== 0); }
    MODES.forEach(function (x, i) {
      var b = document.getElementById(x.btn);
      if (!b) { return; }
      b.classList.toggle("on", i === mode);
      b.setAttribute("aria-pressed", String(i === mode));
    });
    var note = document.getElementById("vbarNote");
    if (note) { note.textContent = m.note; }
    document.querySelectorAll(".vswap").forEach(function (el) {
      if (el.dataset.orig === undefined) { el.dataset.orig = el.innerHTML; }
      var alt = mode === 1 ? el.dataset.alt : mode === 2 ? el.dataset.alt2 : null;
      el.innerHTML = alt === undefined || alt === null ? el.dataset.orig : alt;
    });
    document.querySelectorAll("img[data-alt-src], img[data-alt2-src]").forEach(
      function (img) {
        if (img.dataset.origSrc === undefined) { img.dataset.origSrc = img.src; }
        var src = mode === 1 ? img.dataset.altSrc
                : mode === 2 ? img.dataset.alt2Src : null;
        img.src = src || img.dataset.origSrc;
      }
    );
  }
  MODES.forEach(function (m, i) {
    var b = document.getElementById(m.btn);
    if (b) { b.addEventListener("click", function () { apply(i); }); }
  });
  apply(0);
})();
</script>
"""
body += "\n</body>\n</html>\n"

# a third provider set on the switch: GEOGloWS out
body = body.replace(
    '<button type="button" id="vNog" aria-pressed="false">Without Google Flood Hub'
    "</button>",
    '<button type="button" id="vNog" aria-pressed="false">All providers'
    "</button>\n"
    '        <button type="button" id="vNoGeo" aria-pressed="false">Without Google '
    "Flood Hub</button>",
    1,
)
# a note for the third state, alongside the template's no-Google one
_ng_note = ""
if "error" not in _ng:
    _nge = _ng["envelope"]
    _ng_note = (
        '    <div class="callout warn altonly" style="margin-top:18px">\n'
        "      <strong>You are viewing all providers.</strong> GEOGloWS then carries a "
        "window, and its return periods cannot yet be fitted on its own forecasts: that "
        "archive begins in July 2024. Every window "
        "then sits on a model whose return periods can be fitted on a forecast "
        "archive, which GEOGloWS's cannot yet be: its forecast record begins in "
        "July 2024. The whole report below is recalibrated with GEOGloWS excluded, "
        f"under the same rules. The envelope becomes 1-in-{_nge['env_rp']} "
        f"({_nge['fires']} of {N_YEARS} years), catching {_nge['severe_caught']} of the "
        f"{_nge['n_severe']} severe years"
        + " What changes with the provider set is the configuration: the mechanism table, the return periods and the backtest. The comparison sections further down (tracking correlation, seasonal peaks, per-point detail) still show every candidate model including GEOGloWS, because they are evidence about the models rather than a statement of what was chosen."
        + (f", and it activates in {', '.join(str(y) for y in _nge['no_flood_years'])}, "
           "where two gauges did not record a flood"
           if _nge["no_flood_years"] else ", with no activation in a year that recorded "
           "no flood")
        + ".\n    </div>\n"
    )
body = body.replace('    <div class="stats">', _ng_note + '    <div class="stats">', 1)

# the head is the template's, so its hero blurb, provider note and stat tiles
# all describe the mixed-model mechanism and have to be restated
per_basin = {}
for river in TRIGGER_STATIONS:
    for tag, a in (("std", std), ("alt", alt)):
        yrs = {y for k, w in a["windows"].items() if k[0] == river for y in w["years"]}
        per_basin[(river, tag)] = (N_YEARS + 1) / len(yrs) if yrs else 0

# the hero paragraph carries no class, so anchor on its opening words
body = re.sub(
    r"<p>Proposed trigger for anticipatory action.*?</p>",
    "<p>Proposed trigger for anticipatory action against riverine flooding on the "
    "Juba and Shabelle: a station consensus in which each river-season window runs "
    "on a single forecast source rather than a mixture, monitoring all seven points, "
    "with no threshold below 1-in-3. Calibrated against SWALIM river gauges. "
    "August 2026.</p>",
    body,
    count=1,
    flags=re.S,
)

# the provider-set label and the switch's own note text
body = body.replace(
    "Adopted mechanism &mdash; GloFAS v5 + Google GRRR + GEOGloWS",
    "Adopted: one source per window &mdash; "
    + " | ".join(f"{WLABEL[k]}: {NICE[w['source']]}"
                 for k, w in std["windows"].items()),
)
body = body.replace(
    "Adopted mechanism \u2014 GloFAS v5 + Google GRRR + GEOGloWS",
    "Adopted: one source per window",
)
body = re.sub(
    r"(mechanism table, return periods, backtest and the year-by-year table)",
    "mechanism table, return periods and backtest",
    body,
    count=1,
)

# the rules named in that note are the published study's pair-selection guards
body = re.sub(
    r"using\s+identical rules \(.*?\)",
    "using identical rules (all seven points, one source per window, thresholds at "
    "or above 1-in-3, and the envelope judged jointly)",
    body,
    count=1,
    flags=re.S,
)
body = re.sub(
    r"Narrative sections that do not depend\s+on the provider set \(.*?\)",
    "Narrative sections that do not depend on the provider set (how the model is "
    "chosen, the threshold floor, the readiness band)",
    body,
    count=1,
    flags=re.S,
)
body = body.replace("All three providers", "GloFAS + Google")

stats_new = (
    '<div class="stats">\n'
    '      <div class="stat"><span class="v vswap" '
    f'data-alt="1-in-{env_alt["env_rp"]}">1-in-{env["env_rp"]}</span>'
    '<span class="l">overall action return period (either river)</span></div>\n'
    '      <div class="stat"><span class="v vswap" '
    f'data-alt="1-in-{per_basin[("shabelle", "alt")]:.1f} / '
    f'1-in-{per_basin[("juba", "alt")]:.1f}">'
    f'1-in-{per_basin[("shabelle", "std")]:.1f} / '
    f'1-in-{per_basin[("juba", "std")]:.1f}</span>'
    '<span class="l">per river, Shabelle / Juba</span></div>\n'
    '      <div class="stat"><span class="v">'
    f'{len(TRIGGER_STATIONS["juba"])} + {len(TRIGGER_STATIONS["shabelle"])}</span>'
    '<span class="l">points monitored, Juba and Shabelle</span></div>\n'
    '      <div class="stat"><span class="v">7&ndash;12 d</span>'
    '<span class="l">readiness lead time (action at 1&ndash;7 days)</span></div>\n'
    '    </div>'
)
body = re.sub(r'<div class="stats">.*?\n\s*</div>', stats_new, body, count=1,
              flags=re.S)

# the template's no-Google note belongs to the third state now
body = body.replace(
    'class="callout warn altonly" style="margin-top:18px">',
    'class="callout warn alt2only" style="margin-top:18px">',
    1,
)

# the note explaining the no-Google view lists the published study's rules
body = re.sub(
    r'(using identical rules \().*?(\))',
    r'\1all seven points, one source per window, thresholds at or above 1-in-3, '
    r'and the envelope judged jointly\2',
    body,
    count=1,
    flags=re.S,
)

# retitle, and point the crumb back at the site root
body = body.replace("The multi-source trigger mechanism",
                    "Trigger mechanism: one model per window")
body = re.sub(r"<title>.*?</title>",
              "<title>Trigger mechanism: one model per window &mdash; Somalia "
              "Riverine Flood Trigger</title>", body, count=1, flags=re.S)
body = body.replace(
    "</head>",
    "<style>\nbody.variant .alt2only { display:none; }\n"
    ".alt2only { display:none; }\n"
    "body.variant2 .alt2only { display:block; }\n"
    "body.variant2 .altonly { display:none; }\n"
    "body.variant2 .stdonly { display:none; }\n"
    "figure { margin:24px 0 30px; }\n"
    "figure img { width:100%; height:auto; display:block; }\n"
    "figure figcaption { font-size:12.5px; color:#55606d; line-height:1.5;\n"
    "  margin-top:8px; }\n.altonly { display:none; }\n"
    ".muted { color:#6b7683; }\n</style>\n</head>",
    1,
)

OUT.mkdir(parents=True, exist_ok=True)
sel_payload = write_selection_detail()
ai_payload = write_activation_impact()
print(f"wrote {OUT / 'selection_detail.json'}")
print(f"wrote {OUT / 'activation_impact.json'}")

# Embed both payloads and let fetch be the fallback, so the page also works
# opened straight from disk, where a browser refuses to fetch sibling files.
embedded = (
    "<script>\n"
    "window.__PAYLOADS__ = {\n"
    '  "selection_detail.json": ' + json.dumps(sel_payload).replace("</", "<\\/")
    + ",\n"
    '  "activation_impact.json": ' + json.dumps(ai_payload).replace("</", "<\\/")
    + "\n};\n"
    "window.__data__ = function (name) {\n"
    "  var d = window.__PAYLOADS__[name];\n"
    "  if (d) { return Promise.resolve({ json: function () { return d; } }); }\n"
    "  return fetch(name);\n"
    "};\n"
    "</script>\n"
)
body = body.replace("</head>", embedded + "</head>", 1)
for name in ("selection_detail.json", "activation_impact.json"):
    body = body.replace(f'fetch("{name}")', f'window.__data__("{name}")')
(OUT / "index.html").write_text(body, encoding="utf-8")
SET_KEY = {False: "google", True: "nogoogle", "nogeoglows": "nogeoglows"}


def jsonable(obj):
    """Window keys are (river, season) tuples; JSON needs strings."""
    if isinstance(obj, dict):
        return {("_".join(k) if isinstance(k, tuple) else str(k)): jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return obj


(OUT / "trigger.json").write_text(
    json.dumps(
        {
            "generated": date.today().isoformat(),
            "conditions": {
                "stations": TRIGGER_STATIONS,
                "rp_floor": RP_FLOOR,
                "one_model_per": "river-season window",
                "action_leads": list(ACTION_LEADS),
                "readiness_leads": list(READINESS_LEADS),
            },
            # three provider sets, so the key cannot be a boolean
            "action": {f"{b}_{SET_KEY[g]}": jsonable(a)
                       for (b, g), a in action.items()},
            "readiness": {SET_KEY[g]: jsonable(r) for g, r in ready.items()},
            "severe_years": sorted(severe),
        },
        indent=1,
        default=str,
    ),
    encoding="utf-8",
)
print(f"\nwrote {OUT / 'index.html'}")
print(f"wrote {OUT / 'trigger.json'}")
