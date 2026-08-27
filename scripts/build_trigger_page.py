"""Build a second trigger page: one model per window, no mixing.

A variant of the published pages/trigger design, which lets three models vote
inside one window so the same station can cast two or three votes. Here every
window runs on exactly one source, and the restrictions are:

  stations    all four Juba points and all three Shabelle points, none dropped
  thresholds  1-in-3 is the floor, a quarter of the record is the ceiling,
              1-in-2 is never used
  models      one per river-season, so four choices, never mixed in a window
  basis       reanalysis by default, with a forecasts-only view alongside;
              the two are never combined in one table
  legs        an action leg at leads 1-7 and a readiness leg at leads 8-12,
              the latter on GloFAS v4, the only archive covering those leads

Usage (from repo root):
    .venv/Scripts/python.exe scripts/build_trigger_page.py

Writes pages/trigger-single-model/{index.html,trigger.json,figures/}. The
published pages/trigger is not touched.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import ocha_stratus as stratus  # noqa: E402

import model_selection  # noqa: E402
import summary_figures  # noqa: E402

from src.constants import (  # noqa: E402
    BENCHMARK_RP,
    ENVELOPE_TARGET_RP,
    REFERENCE_GAUGE,
    SEASONS,
    SEVERE_RP,
    TRIGGER_CONFIG,
    TRIGGER_STATIONS,
    TRIGGER_YEARS,
    WINDOW_MODEL,
)
from src.utils import episodes, hits, weibull_level, weibull_threshold  # noqa: E402

OUT = REPO / "pages" / "trigger-single-model"
PREFIX, STAGE = "ds-aa-som-floods/processed", "dev"
MODELS = ["glofas_v5", "glofas_v4", "google_grrr", "geoglows"]
FC_MODELS = ["google_grrr", "glofas_v4"]  # the archives that exist
READINESS_MODEL = "glofas_v4"
READINESS_LEADS = (8, 12)
ACTION_LEADS = (1, 7)
RP_FLOOR = 3
NICE = {"google_grrr": "Google GRRR", "glofas_v5": "GloFAS v5",
        "glofas_v4": "GloFAS v4", "geoglows": "GEOGloWS v2"}
Y0, Y1 = TRIGGER_YEARS
SPAN = set(range(Y0, Y1 + 1))
WINDOWS = list(TRIGGER_CONFIG)


def load(name):
    return stratus.load_parquet_from_blob(f"{PREFIX}/{name}.parquet", stage=STAGE)


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return ('<div class="tablewrap">\n<table class="data">\n'
            f"<thead><tr>{head}</tr></thead>\n<tbody>{body}</tbody>\n</table>\n</div>")


def rp_choices(n_years):
    """Return periods this record can carry: floor 1-in-3, ceiling a quarter."""
    return [r for r in [3, 4, 5, 6] if RP_FLOOR <= r <= n_years // 4]


# ------------------------------------------------------------------ load once
print("loading ...")
lv = load("swalim_levels")
lv["date"] = pd.to_datetime(lv["date"])
bench = load("workflow/som_flood_benchmark_seasonal")
th = load("swalim_thresholds").set_index("station")
dd = pd.concat([load(f"discharge_daily_{m}").assign(src=m) for m in MODELS],
               ignore_index=True)
dd["date"] = pd.to_datetime(dd["date"])


def reforecast_daily(lead_lo, lead_hi, sources):
    """Daily series per station from the forecast archives, over a lead band.

    Ensemble median at each lead, then the strongest lead in the band: what a
    forecaster watching that horizon would have seen.
    """
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
    med = (
        fc.groupby(["src", "station", "valid_time", "leadtime_days"])["discharge"]
        .median()
        .groupby(level=["src", "station", "valid_time"])
        .max()
        .reset_index()
        .rename(columns={"valid_time": "date"})
    )
    spans = {s: set(range(int(g.date.dt.year.min()), int(g.date.dt.year.max()) + 1))
             for s, g in med.groupby("src")}
    return med, spans


print("building the forecast series ...")
fc_action, fc_action_spans = reforecast_daily(*ACTION_LEADS, FC_MODELS)
fc_ready, fc_ready_spans = reforecast_daily(*READINESS_LEADS, [READINESS_MODEL])

# model_selection reports per window; envelope_search reports the union
import envelope_search  # noqa: E402

any_flood, severe = envelope_search.benchmark_years(bench)


# --------------------------------------------------------------- the selection
def select(basis):
    """One model per window on the given basis, judged on the envelope."""
    if basis == "reanalysis":
        frame, models, span = dd, MODELS, SPAN
    else:
        if fc_action is None:
            return {"error": "no forecast archive available"}
        # a product qualifies only if its own archive can carry a 1-in-3
        # threshold, which needs about 12 years. Google's starts in 2016, so it
        # cannot be calibrated on its own forecasts and drops out here.
        frame = fc_action
        models = [m for m in FC_MODELS
                  if len(fc_action_spans.get(m, set()) & SPAN) >= 4 * RP_FLOOR]
        excluded = [m for m in FC_MODELS if m not in models]
        if not models:
            return {"error": ("no forecast archive is long enough to carry a "
                              f"1-in-{RP_FLOOR} threshold")}
        span = set.intersection(*[fc_action_spans[m] & SPAN for m in models])
    rps = rp_choices(len(span))
    if not rps:
        return {"error": (f"{len(span)} overlapping years "
                          f"({min(span)}-{max(span)}) cannot carry a 1-in-{RP_FLOOR} "
                          "threshold, and 1-in-2 is not permitted")}
    cands = {
        k: model_selection.window_candidates(
            frame, lv, k[0], k[1], models=models, span=span, rps=rps
        )
        for k in WINDOWS
    }
    off_target = False
    if basis == "reanalysis":
        excluded = []
        # the adopted configuration, so this page and src/constants.py agree
        combo = []
        for k in WINDOWS:
            spec = TRIGGER_CONFIG[k]
            m = [c for c in cands[k] if c["model"] == spec["source"]
                 and c["rp"] == spec["rp"] and c["n_req"] == spec["n_req"]]
            if not m:
                combo = None
                break
            combo.append(m[0])
        if combo is None:
            return {"error": "the configuration in constants is not reachable here"}
        res = model_selection.evaluate(tuple(combo), any_flood, severe, span=span)
    else:
        keys, best, frontier = model_selection.choose(
            cands, any_flood, severe, span=span
        )
        if best is None and frontier:
            best = min(frontier.values(),
                       key=lambda t: abs(t[2]["env_rp"] - ENVELOPE_TARGET_RP))
            off_target = True
        if best is None:
            return {"error": "no configuration could be scored"}
        combo, res = list(best[1]), best[2]
    return {
        "basis": basis,
        "excluded": excluded,
        "years": [min(span), max(span)],
        "n_years": len(span),
        "rps_available": rps,
        "off_target": off_target,
        "envelope": res,
        "windows": [
            {"river": r, "season": s, "source": c["model"], "rp": c["rp"],
             "n_req": c["n_req"], "n_of": len(c["stations"]),
             "stations": c["stations"], "rho": c["rho"],
             "years": sorted(c["years"])}
            for (r, s), c in zip(WINDOWS, combo)
        ],
    }


print("selecting one model per window ...")
selections = {b: select(b) for b in ("reanalysis", "forecast")}
for b, v in selections.items():
    if "error" in v:
        print(f"  {b:10s} not available: {v['error']}")
    else:
        e = v["envelope"]
        print(f"  {b:10s} {v['n_years']}y RP{v['rps_available']} -> "
              f"1-in-{e['env_rp']}, severe {e['severe_caught']}/{e['n_severe']}")


# ------------------------------------------------------------- readiness leg
def readiness():
    """Leads 8-12 on GloFAS v4: fires at least as often as the action leg.

    Readiness buys preparation time, so it should not be rarer than the action
    leg, and it should not be so frequent that it means nothing. The rule kept
    is the one catching the most severe years while firing between once and
    twice as often as the action leg fires.
    """
    if fc_ready is None:
        return {"error": "no reforecast covers leads 8-12"}
    span = fc_ready_spans.get(READINESS_MODEL, set()) & SPAN
    rps = rp_choices(len(span))
    if not rps:
        return {"error": (f"{len(span)} years at leads 8-12 cannot carry a "
                          f"1-in-{RP_FLOOR} threshold")}
    action = selections["reanalysis"]
    if "error" in action:
        return {"error": "no action leg to size readiness against"}
    out = []
    for (river, season), aw in zip(WINDOWS, action["windows"]):
        months = SEASONS[season]
        n_action = len([y for y in aw["years"] if y in span])
        pool = TRIGGER_STATIONS[river]
        best = None
        for rp in rps:
            cols = []
            for st in pool:
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
            for n in range(2, len(cols)):  # never all: one quiet gauge cannot block
                fires = sorted(set(mx[mx >= n].index) & span)
                if not fires or (n_action and len(fires) < n_action):
                    continue
                if n_action and len(fires) > 2 * n_action:
                    continue
                sev = len(set(fires) & severe & span)
                lead_in = len([y for y in aw["years"] if y in fires])
                score = (sev, lead_in, -len(fires), -rp)
                if best is None or score > best[0]:
                    best = (score, {"rp": rp, "n_req": n, "n_of": len(cols),
                                    "fires": fires, "severe_caught": sev,
                                    "precedes": lead_in})
        row = {"river": river, "season": season, "n_action": n_action}
        row.update(best[1] if best else {"error": "no rule fits the frequency band"})
        out.append(row)
    return {"model": READINESS_MODEL, "leads": list(READINESS_LEADS),
            "years": [min(span), max(span)], "n_years": len(span), "windows": out}


print("sizing the readiness leg ...")
ready = readiness()
if "error" in ready:
    print(f"  not available: {ready['error']}")
else:
    for w in ready["windows"]:
        if "error" in w:
            print(f"  {w['river']:9s} {w['season']:5s} {w['error']}")
        else:
            print(f"  {w['river']:9s} {w['season']:5s} 1-in-{w['rp']} "
                  f"{w['n_req']} of {w['n_of']} | fires {len(w['fires'])}, "
                  f"precedes {w['precedes']} of {w['n_action']} action years")


# --------------------------------------------------------------- the figures
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


print("drawing the figures ...")
rean = selections["reanalysis"]
per_window = {(w["river"], w["season"]): w["years"] for w in rean["windows"]} \
    if "error" not in rean else {}
figs = summary_figures.build(
    {
        "lv": lv, "th": th, "dd": dd, "Y0": Y0, "Y1": Y1, "span": SPAN,
        "model_season": model_season, "flood_years": flood_years,
        "severe_window_years": severe_window_years,
        "window_model": WINDOW_MODEL, "per_window": per_window,
        "reference_gauge": REFERENCE_GAUGE,
        "station_scores": [], "rho": {},
        "reforecast": lambda: None, "exposure": lambda: None,
        "frontier": [], "adopted_point": None,
        "n_severe": len(severe), "target_rp": ENVELOPE_TARGET_RP,
    },
    OUT / "figures",
)


def figure(name, caption=None):
    f = figs.get(name)
    if not f:
        return ""
    cap = caption or f["caption"]
    return (f'    <figure class="chart">\n'
            f'      <img src="{f["src"]}" alt="{esc(cap)}">\n'
            f"      <figcaption>{esc(cap)}</figcaption>\n    </figure>")


# ------------------------------------------------------------------- render
def selection_panel(key, label, slug, active):
    v = selections[key]
    head = (f'      <div class="panel{" active" if active else ""}" id="{slug}">\n')
    if "error" in v:
        return head + f'        <p class="muted">Not available: {esc(v["error"])}.</p>\n      </div>'
    e = v["envelope"]
    rows = [
        (w["river"].capitalize(), w["season"].capitalize(),
         NICE.get(w["source"], w["source"]), f"1-in-{w['rp']}",
         f"{w['n_req']} of {w['n_of']}", f"{w['rho']:.2f}",
         ", ".join(str(y) for y in w["years"]) or "never")
        for w in v["windows"]
    ]
    note = ""
    if v["off_target"]:
        note = (" This is the closest rate this evidence supports; nothing lands on "
                f"1-in-{ENVELOPE_TARGET_RP}.")
    return (
        head
        + f"        <p>Calibrated on {v['years'][0]} to {v['years'][1]}, "
        f"{v['n_years']} years, with thresholds available from 1-in-"
        f"{min(v['rps_available'])} to 1-in-{max(v['rps_available'])}.</p>\n"
        + table(["River", "Season", "Source", "Gauge threshold",
                 "Gauges that must agree", "Tracking rho", "Would have fired in"], rows)
        + f"\n        <p><strong>Envelope: 1-in-{e['env_rp']}</strong> "
        f"({e['fires']} of {v['n_years']} years), catching {e['severe_caught']} of "
        f"the {e['n_severe']} severe years. Fired with no recorded flood: "
        f"{', '.join(str(y) for y in e['no_flood_years']) or 'never'}.{note}</p>\n"
        "      </div>"
    )


tabs = (
    '    <div class="tabbar">\n'
    '      <button class="tab active" data-group="basis" data-panel="sel-rean">'
    "Reanalysis</button>\n"
    '      <button class="tab" data-group="basis" data-panel="sel-fc">'
    "Forecasts only</button>\n    </div>\n"
    '    <div class="panels">\n'
    + selection_panel("reanalysis", "Reanalysis", "sel-rean", True) + "\n"
    + selection_panel("forecast", "Forecasts only", "sel-fc", False) + "\n"
    "    </div>"
)

if "error" in ready:
    ready_html = f'    <p class="muted">Not available: {esc(ready["error"])}.</p>'
else:
    ready_html = table(
        ["River", "Season", "Gauge threshold", "Gauges that must agree",
         "Readiness years", "Precedes action years"],
        [
            (w["river"].capitalize(), w["season"].capitalize(),
             f"1-in-{w['rp']}" if "rp" in w else esc(w.get("error", "-")),
             f"{w['n_req']} of {w['n_of']}" if "n_req" in w else "-",
             ", ".join(str(y) for y in w["fires"]) if "fires" in w else "-",
             f"{w['precedes']} of {w['n_action']}" if "precedes" in w else "-")
            for w in ready["windows"]
        ],
    )

station_html = table(
    ["River", "Points monitored"],
    [(r.capitalize(), f"{len(v)}: " + ", ".join(v))
     for r, v in TRIGGER_STATIONS.items()],
)

data = {
    "generated": date.today().isoformat(),
    "restrictions": {
        "stations": TRIGGER_STATIONS,
        "rp_floor": RP_FLOOR,
        "one_model_per": "river-season",
        "action_leads": list(ACTION_LEADS),
        "readiness_leads": list(READINESS_LEADS),
    },
    "selections": selections,
    "readiness": ready,
    "severe_years": sorted(severe),
    "figures": sorted(figs),
}

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Single-model trigger variant &mdash; Somalia Riverine Flood Trigger</title>
<meta name="description" content="A variant of the Somalia riverine flood trigger in which every river-season window runs on one model rather than a mixture, with all seven gauge points monitored and no threshold below 1-in-3.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css">
<style>
.hero.hero-sub {{ padding:40px 44px 34px; }}
.hero.hero-sub h1 {{ font-size:28px; }}
.crumb {{ font-size:12px; margin:0 0 14px; }}
.crumb a {{ color:#cfe0ef; }}
figure.chart {{ margin:24px 0 30px; }}
figure.chart img {{ width:100%; height:auto; display:block; }}
figure.chart figcaption {{ font-size:13px; color:#55606d; line-height:1.55;
  margin-top:10px; }}
.tabbar {{ display:flex; flex-wrap:wrap; gap:6px; margin:18px 0 0; }}
.tab {{ font:inherit; font-size:13px; padding:7px 13px; cursor:pointer;
  border:1px solid #d3d9e0; background:#f7f9fb; color:#3a4552;
  border-radius:6px; }}
.tab:hover {{ background:#eef2f6; }}
.tab.active {{ background:#1f5c8b; border-color:#1f5c8b; color:#fff;
  font-weight:500; }}
.panel {{ display:none; padding-top:6px; }}
.panel.active {{ display:block; }}
.muted {{ color:#6b7683; }}
h2 .num {{ display:inline-block; min-width:1.6em; color:#8a97a5; }}
</style>
</head>
<body>
  <header class="hero hero-sub">
    <p class="crumb"><a href="../">Somalia riverine flood trigger</a></p>
    <h1>Single-model trigger variant</h1>
    <p>Every river-season window runs on one source rather than a mixture of
      three, all seven gauge points are monitored, and no threshold sits below
      1-in-3. Generated {data['generated']} from the source data.</p>
  </header>

  <article>

    <h2><span class="num">1</span>The trigger</h2>
    <p>Four windows, one source each, so a window is one product plus one rule. Inside a
      window every point's flow is compared with its own return-period threshold and the
      window fires when enough points cross in the same season. At least two must agree,
      never all of them. The reanalysis view is the calibration basis; the forecasts view
      refits the same rule on the forecast archives, and the two are never combined.</p>
{tabs}

    <h2><span class="num">2</span>Readiness leg</h2>
    <p>Readiness runs at leads 8 to 12 on {NICE[READINESS_MODEL]}, the only archive that
      covers those leads, with thresholds fitted on that series and never below 1-in-3.
      It is sized to fire at least as often as the action leg and at most twice as often,
      so it buys preparation time without becoming routine.</p>
{ready_html}

    <h2><span class="num">3</span>Points monitored</h2>
{station_html}
{figure("map")}

    <h2><span class="num">4</span>Where the thresholds sit</h2>
{figure("thresholds")}
{figure("crossings")}

    <h2><span class="num">5</span>The grid the rule came from</h2>
{figure("grid")}

    <h2><span class="num">6</span>Backtest</h2>
{figure("activation")}

    <h2><span class="num">7</span>What this variant does not settle</h2>
    <ul>
      <li><strong>The reanalysis cannot choose the model.</strong> Many assignments reach
        the same activation years, so the model per window rests on skill at lead time,
        not on this backtest.</li>
      <li><strong>Google cannot be calibrated on its own forecasts.</strong> Its archive
        starts in 2016, and 1-in-3 needs about 12 years, so a Google window is
        necessarily calibrated on the retrospective and only checked at lead time.</li>
      <li><strong>Two Juba points can no longer be verified.</strong> Bardheere's gauge
        record ends in 2023 and Bualle's in 2024, so both can be forecast at but not
        checked against observations from here on.</li>
    </ul>

    <p class="muted">The mixed-model design remains published at
      <a href="../trigger/">trigger</a>, and the full data-source review at
      <a href="../summary/">summary</a>.</p>

  </article>

  <script>
    document.querySelectorAll(".tab").forEach(function (btn) {{
      btn.addEventListener("click", function () {{
        var group = btn.dataset.group || "basis";
        document.querySelectorAll('.tab[data-group="' + group + '"]').forEach(
          function (b) {{
            b.classList.remove("active");
            var p = document.getElementById(b.dataset.panel);
            if (p) {{ p.classList.remove("active"); }}
          }}
        );
        btn.classList.add("active");
        var panel = document.getElementById(btn.dataset.panel);
        if (panel) {{ panel.classList.add("active"); }}
      }});
    }});
  </script>
</body>
</html>
"""

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "trigger.json").write_text(json.dumps(data, indent=1, default=str),
                                  encoding="utf-8")
(OUT / "index.html").write_text(HTML, encoding="utf-8")
print(f"\nwrote {OUT / 'trigger.json'}")
print(f"wrote {OUT / 'index.html'}")
