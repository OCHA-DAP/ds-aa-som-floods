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

Sections whose argument was specific to the pair-based selection (the seasonal
peak and metric-plane comparisons, the impact tilt) are dropped rather than
left in place describing a design this page does not use.

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
from src.utils import weibull_threshold  # noqa: E402

SRC_PAGE = REPO / "pages" / "trigger" / "index.html"
SRC_FIGS = REPO / "pages" / "trigger" / "figs"
OUT = REPO / "pages" / "trigger-single-model"
FIGS = OUT / "figs"
PREFIX, STAGE = "ds-aa-som-floods/processed", "dev"

MODELS = ["glofas_v5", "glofas_v4", "google_grrr", "geoglows"]
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
any_flood, severe = envelope_search.benchmark_years(bench)


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
def action_leg(basis, drop_google=False):
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
    rps = rp_choices(len(span))
    if not rps:
        return {"error": (f"{len(span)} years ({min(span)}-{max(span)}) cannot carry "
                          f"a 1-in-{RP_FLOOR} threshold, and 1-in-2 is not allowed")}
    cands = {k: model_selection.window_candidates(
        frame, lv, k[0], k[1], models=models, span=span, rps=rps) for k in WINDOWS}
    pinned = basis == "reanalysis" and not drop_google
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
        _, best, frontier = model_selection.choose(cands, any_flood, severe, span=span)
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
for k, v in action.items():
    if "error" in v:
        print(f"  {k}: {v['error']}")
    else:
        e = v["envelope"]
        print(f"  {k[0]:10s} drop_google={k[1]!s:5s} 1-in-{e['env_rp']} | "
              f"severe {e['severe_caught']}/{e['n_severe']}")


# ---------------------------------------------------------- the readiness leg
def readiness_leg(action_ref):
    """Leads 7-12 on GloFAS v4, thresholds at or above 1-in-3.

    Readiness releases only the mobilisation share, so it may fire more often
    than action, but it is held to the same threshold floor: no 1-in-2.
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
        act_years = [y for y in action_ref["windows"][k]["years"] if y in span]
        best = None
        for rp in rps:
            cols = []
            for st in TRIGGER_STATIONS[river]:
                s = fc_ready[(fc_ready.src == READINESS_MODEL)
                             & (fc_ready.station == st)]
                if s.empty:
                    continue
                s = s.set_index("date")["discharge"].sort_index()
                s = s[s.index.month.isin(months) & s.index.year.isin(span)]
                if len(s) < 60:
                    continue
                am = s.groupby(s.index.year).max().dropna()
                t = weibull_threshold(am.values, rp)
                if not np.isnan(t):
                    cols.append((s >= t).rename(st))
            if len(cols) < 2:
                continue
            mat = pd.concat(cols, axis=1).fillna(False)
            mx = mat.sum(axis=1).groupby(mat.index.year).max()
            for n in range(2, len(cols)):
                fires = sorted(set(mx[mx >= n].index) & span)
                if not fires or len(fires) < len(act_years):
                    continue
                if len(act_years) and len(fires) > 2 * len(act_years):
                    continue
                covers = [y for y in act_years if y in fires]
                score = (len(covers), len(set(fires) & severe), -len(fires), -rp)
                if best is None or score > best[0]:
                    best = (score, {"rp": rp, "n_req": n, "n_of": len(cols),
                                    "fires": fires, "covers": covers,
                                    "n_action": len(act_years),
                                    "severe_caught": len(set(fires) & severe)})
        rows[k] = best[1] if best else {"error": "no rule fits the frequency band"}
    return {"model": READINESS_MODEL, "leads": list(READINESS_LEADS),
            "years": [min(span), max(span)], "n_years": len(span),
            "rps_available": rps, "windows": rows}


print("readiness leg ...")
ready = {
    False: readiness_leg(action[("reanalysis", False)]),
    True: readiness_leg(action[("reanalysis", True)]),
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
    b = bench[(bench.river == river) & (bench.season == season)
              & (bench.benchmark == f"swalim_{REFERENCE_GAUGE[river]}")]
    return set(b[b[f"flood_{rp}yr"] == 1].year) & SPAN


def severe_window_years(river, season):
    b = bench[(bench.river == river) & (bench.season == season)
              & (bench.benchmark == f"swalim_{REFERENCE_GAUGE[river]}")]
    return set(b[b.rp >= SEVERE_RP].year) & SPAN


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


def draw_model_choice(path):
    """Mean best-lag rho per window and product, with the choice ringed."""
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5), sharey=True)
    for ax, k in zip(axes, WINDOWS):
        river, season = k
        ref = lv[lv.station == REFERENCE_GAUGE[river]].set_index("date")["level_m"]
        ref = ref[ref.index.month.isin(SEASONS[season])].sort_index()
        vals = {}
        for m in MODELS:
            rhos = []
            for st in TRIGGER_STATIONS[river]:
                s = model_season(m, st, SEASONS[season])
                if len(s):
                    rhos.append(model_selection.best_lag_rho(s, ref))
            if rhos:
                vals[m] = float(np.mean(rhos))
        chosen = action[("reanalysis", False)]["windows"][k]["source"]
        names = list(vals)
        ax.bar(range(len(names)), [vals[m] for m in names],
               color=[SOURCE_COLORS.get(m, "#1C7293") for m in names], width=0.66)
        for i, m in enumerate(names):
            if m == chosen:
                ax.plot(i, vals[m] + 0.03, marker="v", color=INK, markersize=8)
            ax.text(i, vals[m] - 0.06, f"{vals[m]:.2f}", ha="center", fontsize=8.5,
                    color="white")
        ax.set_xticks(range(len(names)),
                      [NICE[m].replace(" ", "\n") for m in names], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(WLABEL[k], fontsize=10.5)
        style_ax(ax, grid="y")
    axes[0].set_ylabel("mean best-lag rho")
    fig.suptitle("One model per window: mean tracking correlation across that window's "
                 "points (marker = chosen)",
                 x=0.06, ha="left", fontweight="bold", fontsize=11.5, color=INK)
    fig.subplots_adjust(top=0.8, wspace=0.08)
    fig.savefig(path, format="png", dpi=115, bbox_inches="tight")
    plt.close(fig)


print("figures ...")
FIGS.mkdir(parents=True, exist_ok=True)
figs_built = summary_figures.build(ctx_for(action[("reanalysis", False)]), FIGS)
figs_alt = {}
if "error" not in action[("reanalysis", True)]:
    alt_dir = FIGS / "_alt"
    figs_alt = summary_figures.build(ctx_for(action[("reanalysis", True)]), alt_dir)

# convert the SVGs this page needs into PNGs at the document's own filenames,
# by redrawing them: matplotlib is the source, so no rasteriser is needed
summary_figures.os.environ["SUMMARY_FIG_PNG"] = str(FIGS)
figs_built = summary_figures.build(ctx_for(action[("reanalysis", False)]), FIGS)
name_map = {"map": "map_stations", "thresholds": "a2_swalim_rp",
            "grid": "c_grid_heatmaps", "activation": "c_backtest_strip",
            "crossings": "c_crossings"}
for svg, png in name_map.items():
    src = FIGS / f"{svg}.png"
    if src.exists():
        shutil.move(str(src), str(FIGS / f"{png}.png"))
if figs_alt:
    summary_figures.os.environ["SUMMARY_FIG_PNG"] = str(FIGS / "_altpng")
    summary_figures.build(ctx_for(action[("reanalysis", True)]), FIGS / "_alt")
    alt = FIGS / "_altpng" / "activation.png"
    if alt.exists():
        shutil.move(str(alt), str(FIGS / "c_backtest_strip_nogoogle.png"))
summary_figures.os.environ.pop("SUMMARY_FIG_PNG", None)
draw_model_choice(FIGS / "a_selection.png")
for stale in ["_alt", "_altpng"]:
    shutil.rmtree(FIGS / stale, ignore_errors=True)
for svg in FIGS.glob("*.svg"):
    svg.unlink()
# figures inherited unchanged from the source document
for keep in ["f_own_skill.png"]:
    if (SRC_FIGS / keep).exists():
        shutil.copy2(SRC_FIGS / keep, FIGS / keep)
print("  figs:", sorted(p.name for p in FIGS.glob("*.png")))


# -------------------------------------------------------------------- sections
def vswap(std, alt):
    """The document's own with/without-Google swap."""
    return f'<td class="vswap" data-alt="{esc(alt)}">{std}</td>'


def glance_table():
    std, alt = action[("reanalysis", False)], action[("reanalysis", True)]
    rst, ralt = ready[False], ready[True]
    rows = []
    for k in WINDOWS:
        w = std["windows"][k]
        wa = alt["windows"][k] if "error" not in alt else w
        rd = rst["windows"].get(k, {}) if "error" not in rst else {}
        rda = ralt["windows"].get(k, {}) if "error" not in ralt else {}
        act_std = (f"{NICE[w['source']]}: {w['n_req']} of {w['n_of']} points over "
                   f"their 1-in-{w['rp']}-yr thresholds")
        act_alt = (f"{NICE[wa['source']]}: {wa['n_req']} of {wa['n_of']} points over "
                   f"their 1-in-{wa['rp']}-yr thresholds")
        rd_std = (f"{NICE[READINESS_MODEL]} ens-median: {rd['n_req']} of {rd['n_of']} "
                  f"over 1-in-{rd['rp']}-yr" if "rp" in rd else "no rule fits")
        rd_alt = (f"{NICE[READINESS_MODEL]} ens-median: {rda['n_req']} of "
                  f"{rda['n_of']} over 1-in-{rda['rp']}-yr" if "rp" in rda
                  else rd_std)
        rows.append(
            "<tr><td>" + WLABEL[k] + "</td>"
            + vswap(act_std, act_alt)
            + vswap(rp_of(len(w["years"])), rp_of(len(wa["years"])))
            + vswap(rd_std, rd_alt)
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
    std, alt = action[("reanalysis", False)], action[("reanalysis", True)]
    rows = []
    for k in WINDOWS:
        w, wa = std["windows"][k], alt["windows"][k]
        rows.append(
            f"<tr><td>individual</td><td>{WLABEL[k]} action</td>"
            + vswap(f"{len(w['years'])} &mdash; "
                    + ", ".join(str(y) for y in w["years"]),
                    f"{len(wa['years'])} &mdash; "
                    + ", ".join(str(y) for y in wa["years"]))
            + vswap(rp_of(len(w["years"])), rp_of(len(wa["years"])))
            + "</tr>"
        )
    for river in TRIGGER_STATIONS:
        yrs = sorted({y for k, w in std["windows"].items() if k[0] == river
                      for y in w["years"]})
        yrs_a = sorted({y for k, w in alt["windows"].items() if k[0] == river
                        for y in w["years"]})
        rows.append(
            f"<tr><td>basin</td><td>{river.capitalize()} (Gu or Deyr)</td>"
            + vswap(str(len(yrs)), str(len(yrs_a)))
            + vswap(rp_of(len(yrs)), rp_of(len(yrs_a))) + "</tr>"
        )
    e, ea = std["envelope"], alt["envelope"]
    rows.append(
        "<tr><td>overall</td><td>action, either basin</td>"
        + vswap(str(e["fires"]), str(ea["fires"]))
        + vswap(f"{e['env_rp']} yr", f"{ea['env_rp']} yr") + "</tr>"
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
                      "would have fired in"], rows)
        + f"<p><strong>Envelope 1-in-{e['env_rp']}</strong> ({e['fires']} of "
        f"{use['n_years']} years), catching {e['severe_caught']} of the "
        f"{e['n_severe']} severe years.{note}</p>"
    )


std = action[("reanalysis", False)]
alt = action[("reanalysis", True)]
env, env_alt = std["envelope"], alt["envelope"]

SECTIONS = {
    "The mechanism at a glance": f"""
    <p>Four river-season windows, each running on <strong>one</strong> forecast source
      rather than a mixture. Inside a window, every monitored point's flow is compared
      with its own return-period threshold and the window fires when enough points cross
      in the same season. At least two points must agree, so no single point releases the
      money, and never all of them, so one quiet point cannot block it.</p>
    <p>All seven reporting-era points are monitored: four on the Juba (Luuq, Dollow,
      Bardheere, Bualle) and three on the Shabelle (Belet Weyne, Bulo Burti, Jowhar). No
      threshold, on either leg, sits below 1-in-3.</p>
{glance_table()}
    <div class="callout">
      <strong>The envelope.</strong> The full amount is released whenever any window
      fires, so the union is what the 1-in-3 target applies to. As configured it would
      have released in <strong>{env['fires']} of {N_YEARS} years, once every
      {env['env_rp']} years</strong>, catching {env['severe_caught']} of the
      {env['n_severe']} years the reference gauge recorded as 1-in-{SEVERE_RP} or rarer,
      and never firing in a year with no recorded flood.
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
    "How the pairs were chosen": f"""
    <h2>How the model was chosen, one per window</h2>
    <p>The published mechanism let (station, provider) pairs compete freely, so one
      station could cast two or three votes and three providers each held a veto. Here
      the unit is the point, not the pair: each window takes a single source, and the
      only things searched are the threshold and how many points must agree.</p>
    <p>Products are compared on best-lag Spearman correlation between their reanalysis
      discharge and the river's reference gauge level (Luuq for the Juba, Belet Weyne for
      the Shabelle), averaged over that window's points.</p>
    <figure><img src="figs/a_selection.png" alt="Mean tracking correlation per window">
      <figcaption>Mean best-lag correlation per window and product; the marker shows the
        source adopted for that window.</figcaption></figure>
    <div class="callout warn">
      <strong>The correlation does not settle it.</strong> Many product assignments
      reproduce the same activation years, because with 25 years and
      {len(severe)} severe events the threshold and vote count carry enough freedom to
      absorb the difference. Correlation ranks the products; what separates them
      operationally is skill at lead time and whether their archive can carry the
      threshold at all.
    </div>
""",
    "Thresholds and calibration": f"""
    <p>Every threshold is a return period on the product's own climatology at that point,
      so a product is judged on timing rather than on scale. <strong>1-in-3 is the floor
      and a quarter of the record is the ceiling</strong>: on 25 years that allows 1-in-3
      to 1-in-6, and 1-in-2 is not used anywhere, on either leg.</p>
    <figure><img src="figs/c_grid_heatmaps.png" alt="Threshold and vote grid">
      <figcaption>POD, FAR and F1 across the threshold and vote grid for each window,
        against the reference gauge's 1-in-3 flood definition. The outlined cell is
        adopted.</figcaption></figure>
    <figure><img src="figs/a2_swalim_rp.png" alt="Where the thresholds sit">
      <figcaption>Where each point's 1-in-3 level sits against SWALIM's official Moderate
        to High band, per season.</figcaption></figure>
    <h3>Calibrated on the reanalysis, checked on the forecasts</h3>
    {forecast_panel()}
""",
    "Would it have worked operationally?": f"""
    <p>Year by year, against the flood record at the reference gauges. Severe years are
      those the gauge recorded as 1-in-{SEVERE_RP} or rarer; ordinary floods are 1-in-3
      or rarer and are deliberately not all covered at this activation rate.</p>
    <figure><img src="figs/c_backtest_strip.png" alt="Backtest by year">
      <figcaption>Each window and the envelope, {Y0} to {Y1}.</figcaption></figure>
    <figure class="altonly"><img src="figs/c_backtest_strip_nogoogle.png"
      alt="Backtest by year without Google">
      <figcaption>The same backtest with Google removed.</figcaption></figure>
    <p>Severe years missed: {", ".join(str(y) for y in env["severe_missed"]) or "none"}.
      Years the envelope fired with no flood recorded at either reference gauge:
      {", ".join(str(y) for y in env["no_flood_years"]) or "none"}.</p>
""",
    "The readiness leg (7–12 days)": f"""
    <p>Readiness runs on {NICE[READINESS_MODEL]} ensemble-median forecasts at leads 7 to
      12, the only archive covering that band, over the same full set of points, with
      thresholds fitted on that series. It releases only the mobilisation share and may
      fire more often than action, but it is held to the same floor:
      <strong>no threshold below 1-in-3</strong>, which is the change from the published
      mechanism, where readiness sat on 1-in-2 levels.</p>
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
    <p>Readiness is sized to fire at least as often as the action leg and at most twice as
      often, so it buys preparation time without becoming routine. The Gu windows lead
      every action year; the Deyr windows cover about half, which is the honest limit of
      what GloFAS v4 sees at 7 to 12 days in that season.</p>
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
      <li><strong>The reanalysis cannot pick the product.</strong> Many assignments tie,
        so the model per window rests on lead-time skill and on operational
        considerations, not on the backtest.</li>
      <li><strong>Google cannot be calibrated on its own forecasts.</strong> Its
        reforecast starts in 2016, and a 1-in-3 threshold needs about 12 years, so a
        Google window is necessarily calibrated on the retrospective and only checked at
        lead time.</li>
      <li><strong>Two Juba points can no longer be verified.</strong> Bardheere's gauge
        record ends 2023-11-30 and Bualle's 2024-03-14. Both can still be forecast at,
        but neither can be checked against observations from here on.</li>
      <li><strong>Deyr readiness is weak.</strong> At 7 to 12 days GloFAS v4 covers only
        about half the Deyr action years, so a Deyr activation may arrive with little or
        no readiness phase.</li>
    </ul>
""",
}
DROP = {
    "Seasonal peaks: model vs gauge",
    "Activations, impact and response, year by year",
    "Forecast skill against each model's own reanalysis",
}

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
    if replacement is None:
        print(f"  kept as-is: {title}")
        kept.append(p)
        continue
    print(f"  rebuilt: {title}")
    heading = t.group(0) if t else f"<h2>{title}</h2>"
    if replacement.lstrip().startswith("<h2"):
        kept.append(replacement)
    else:
        kept.append(heading + replacement)

body = head + "".join(kept)

# only the provider-set switch survives: the two fetch-driven tables belonged
# to sections this variant does not carry, and their JSON is not written here
sw = tail_scripts.find("// ---- provider-set switch")
last = tail_scripts.rfind("})();")
if sw != -1 and last != -1:
    body += "<script>\n(function () {\n" + tail_scripts[sw:last + 5] + "\n</script>"
body += "\n</body>\n</html>\n"

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
body = body.replace("All three providers", "All providers")

stats_new = (
    '<div class="stats">\n'
    '      <div class="stat"><span class="v vswap" '
    f'data-alt="1-in-{env_alt["env_rp"]}">1-in-{env["env_rp"]}</span>'
    '<span class="l">overall action return period (either basin)</span></div>\n'
    '      <div class="stat"><span class="v vswap" '
    f'data-alt="1-in-{per_basin[("shabelle", "alt")]:.1f} / '
    f'1-in-{per_basin[("juba", "alt")]:.1f}">'
    f'1-in-{per_basin[("shabelle", "std")]:.1f} / '
    f'1-in-{per_basin[("juba", "std")]:.1f}</span>'
    '<span class="l">per basin, Shabelle / Juba</span></div>\n'
    '      <div class="stat"><span class="v">'
    f'{len(TRIGGER_STATIONS["juba"])} + {len(TRIGGER_STATIONS["shabelle"])}</span>'
    '<span class="l">points monitored, Juba and Shabelle</span></div>\n'
    '      <div class="stat"><span class="v">7&ndash;12 d</span>'
    '<span class="l">readiness lead time (action at 1&ndash;7 days)</span></div>\n'
    '    </div>'
)
body = re.sub(r'<div class="stats">.*?\n\s*</div>', stats_new, body, count=1,
              flags=re.S)

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
    "<style>\nfigure { margin:24px 0 30px; }\n"
    "figure img { width:100%; height:auto; display:block; }\n"
    "figure figcaption { font-size:12.5px; color:#55606d; line-height:1.5;\n"
    "  margin-top:8px; }\n.altonly { display:none; }\n"
    ".muted { color:#6b7683; }\n</style>\n</head>",
    1,
)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "index.html").write_text(body, encoding="utf-8")
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
            "action": {f"{b}_{'nogoogle' if g else 'google'}": jsonable(a)
                       for (b, g), a in action.items()},
            "readiness": {("nogoogle" if g else "google"): jsonable(r)
                          for g, r in ready.items()},
            "severe_years": sorted(severe),
        },
        indent=1,
        default=str,
    ),
    encoding="utf-8",
)
print(f"\nwrote {OUT / 'index.html'}")
print(f"wrote {OUT / 'trigger.json'}")
