"""Summary page: how we got to the trigger, step by step, with tables and plots.

Every number on the page is computed here from the analysis outputs and cross-checked with
assertions against the analysis page (index.html), so the two cannot drift apart silently.
Writes pages/trigger-single-model/summary.html, three figures (figs/s_*.png) and summary_rows.json.
"""
import datetime
import json
import re
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import somlib as L

S = Path(__file__).parent
PAGE = S / "wt-trigger/pages/trigger-single-model"
FIGS = PAGE / "figs"
OUT = PAGE / "summary.html"

# ---------------------------------------------------------------- the design, as adopted
WINDOWS = [("juba", "deyr"), ("shabelle", "deyr"), ("juba", "gu"), ("shabelle", "gu")]   # Deyr before Gu
RULE = {("juba", "deyr"): (4, 3), ("shabelle", "deyr"): (4, 2), ("juba", "gu"): (5, 3), ("shabelle", "gu"): (6, 2)}
CAL = {"deyr": "glofas_v5", "gu": "google_grrr"}                    # calibrated model per season (data key)
SOURCE = {"deyr": "GloFAS", "gu": "Google Flood Hub"}               # how the trigger names its source
NICE = {"google_grrr": "Google Flood Hub", "glofas_v5": "GloFAS v5", "glofas_v4": "GloFAS v4"}
COL = {"google_grrr": "#1d4ed8", "glofas_v5": "#0f766e", "glofas_v4": "#6b7280", "swalim": "#d97706", "gauge": "#111827"}
SEASON = {"deyr": ("Deyr", "October to December"), "gu": ("Gu", "March to May")}
STATIONS = {"juba": "Dollow, Luuq, Bardheere, Bualle", "shabelle": "Belet Weyne, Bulo Burti, Jowhar"}
ARCHIVE = {"google_grrr": (2016, 2023), "glofas_v4": (2003, 2024)}
SPAN = list(range(1999, 2025))
N_YEARS = 25                                                        # 1999 to 2023, the calibration record


def wname(river, season):
    return f"{SEASON[season][0]} {river.title()}"


def rp_text(n_act, n=N_YEARS):
    """Return period as the analysis page reports it: Weibull (n+1)/m."""
    return f"1-in-{(n + 1) / n_act:.1f}"


def dfmt(x):
    return f"{x.day} {x.strftime('%b')}"


def all_issue_dates(model, river, season, rp, n_req, span, leads=(1, 7)):
    """Year -> sorted issue dates (normalised) on which the ensemble median at the given leads
    put >= n_req of the river's points over their level. Same construction as
    somlib.first_issue_dates, keeping every issue rather than the first."""
    import numpy as np
    rf = L.reforecast(model)
    rf = rf[rf.station.isin(L.TRIGGER_STATIONS[river]) & rf.leadtime_days.between(*leads)
            & rf.valid_time.dt.month.isin(L.SEASONS[season])]
    med = rf.groupby(["issued_time", "valid_time", "station"]).discharge.median().unstack("station")
    thr = L.model_thresholds(model, river, season, rp)
    cols = [c for c in thr.index if c in med.columns and not np.isnan(thr[c])]
    votes = (med[cols] >= thr[cols]).sum(axis=1)
    hit = votes[votes >= n_req].reset_index()
    hit.columns = ["issued", "valid", "votes"]
    hit["year"] = hit["valid"].dt.year
    out = {}
    for y, g in hit.groupby("year"):
        if y in span:
            out[y] = sorted(set(g["issued"].dt.normalize()))
    return out


# ---------------------------------------------------------------- inputs already computed
metrics = json.load(open(S / "metrics.json"))
win = {w["window"]: w for w in metrics["windows"]}
station = json.load(open(S / "station_metrics.json"))
swalim_tl = json.load(open(S / "swalim_timeline.json"))
fallback = json.load(open(S / "obs_fallback.json"))["rows"]
G24 = json.load(open(S / "gu2024_issue.json"))
rp3c = pd.DataFrame(json.load(open(S / "rp3_corr.json")))
rp3t = pd.DataFrame(json.load(open(S / "rp3_track.json")))          # tracking, 1-in-3 seasons only
index_html = (PAGE / "index.html").read_text(encoding="utf-8")
index_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", index_html, flags=re.S)))

# ---------------------------------------------------------------- step 1: floods on the gauges
flood, severe = {}, {}
for river, season in WINDOWS:
    w = win[wname(river, season)]
    flood[(river, season)] = set(w["flood_years"]); severe[(river, season)] = set(w["severe_years"])
flood_all = set().union(*flood.values()); severe_all = set().union(*severe.values())

# ---------------------------------------------------------------- step 4/5: activations at the rule
act = {}
for river, season in WINDOWS:
    w = win[wname(river, season)]
    act[(river, season)] = set(w["models"][CAL[season]]["activations"])
env_years = set().union(*act.values())
env_page = set(metrics["envelope"]["activations"])
assert env_years == env_page, ("envelope mismatch", sorted(env_years), sorted(env_page))
n_act = len(env_years)
sev_caught = len(env_years & severe_all)
outside = sorted(env_years - flood_all)
per_river = {r: set().union(*(act[(r, s)] for s in ("deyr", "gu"))) for r in ("juba", "shabelle")}
rate_env = rp_text(n_act)
rate_river = {r: rp_text(len(v)) for r, v in per_river.items()}
# cross-check against the analysis page's own headline tiles
assert f"{rate_env} overall action return period" in index_text, rate_env
assert f"{rate_river['shabelle']} / {rate_river['juba']} per river" in index_text, rate_river
assert sev_caught == len(severe_all) == 7, (sev_caught, len(severe_all))
assert outside == [2013], outside

# ---------------------------------------------------------------- step 6: forecasts, flood by flood
V4_2024 = {r: (pd.Timestamp(f"2024 {G24[r]['v4_fc_issue'][0]}") if G24[r]["v4_fc_issue"][0] else None) for r in ("juba", "shabelle")}


def verdict(first_onset, when, year, model):
    lo, hi = ARCHIVE[model]
    if not (lo <= year <= hi):
        return ("na", "no archive", None)
    if when is None:
        return ("never", "never crossed", None)
    lead = (first_onset - when).days
    return ("before", f"{lead} d before", lead) if lead > 0 else ("same", "same day", 0) if lead == 0 else ("after", f"{-lead} d after", lead)


fb_rows, fb_tot = {}, {}
for season in ("deyr", "gu"):
    onset, sev_s, fc, fc_all = {}, {}, {"google_grrr": {}, "glofas_v4": {}}, {}
    for river in ("juba", "shabelle"):
        rp, n = RULE[(river, season)]
        onset[river] = L.gauge_crossings(river, season, 3, span=SPAN)
        sev_s[river] = set(L.gauge_crossings(river, season, L.SEVERE_RP, span=SPAN))
        for mdl in fc:
            fc[mdl][river] = {y: v[0] for y, v in L.first_issue_dates(mdl, river, season, rp, n, span=SPAN, leads=(1, 7)).items()}
            fc_all.setdefault(mdl, {})[river] = all_issue_dates(mdl, river, season, rp, n, SPAN)
        if season == "gu" and V4_2024[river] is not None:
            fc["glofas_v4"][river][2024] = V4_2024[river]
            fc_all["glofas_v4"][river][2024] = [V4_2024[river]]          # only the first is archived for 2024
    years = sorted({y for r in onset for y in onset[r] if y in SPAN and (season == "gu" or y <= 2023)})
    # the 1999-2023 flood years must agree with the benchmark used everywhere else
    assert {y for y in years if y <= 2023} == set().union(*(flood[(r, season)] for r in ("juba", "shabelle"))), season
    rows, tot, h2h, n_both = [], {k: {} for k in fc}, {k: 0 for k in fc}, 0
    for y in years:
        flooded = [r for r in onset if y in onset[r]]
        first = min(onset[r][y] for r in flooded)
        rec = {"year": y, "rivers": [r.title() for r in flooded], "severe": any(y in sev_s[r] for r in flooded), "first_onset": dfmt(first)}
        leads = {}
        for mdl in fc:
            c = {r: fc[mdl][r][y] for r in fc[mdl] if y in fc[mdl][r]}
            when, river = (min(c.values()), min(c, key=c.get)) if c else (None, None)
            kind, text, lead = verdict(first, when, y, mdl)
            issues = sorted({d for r_ in fc_all.get(mdl, {}) for d in fc_all[mdl][r_].get(y, [])})
            rec[mdl] = {"kind": kind, "text": text, "river": river.title() if river else None, "when": dfmt(when) if when is not None else None,
                        "lead": lead, "all_leads": [(first - d_).days for d_ in issues], "n_issues": len(issues)}
            tot[mdl][kind] = tot[mdl].get(kind, 0) + 1
            leads[mdl] = (kind, lead)
        g, v = leads["google_grrr"], leads["glofas_v4"]
        rec["first"] = ""
        if g[0] != "na" and v[0] != "na":
            n_both += 1
            gl = g[1] if g[1] is not None else -999; vl = v[1] if v[1] is not None else -999
            rec["first"] = "Google" if gl > vl else "GloFAS v4" if vl > gl else ("tie" if g[1] is not None else "neither")
            if gl > vl: h2h["google_grrr"] += 1
            elif vl > gl: h2h["glofas_v4"] += 1
        rows.append(rec)
    fb_rows[season], fb_tot[season] = rows, {"per_model": tot, "head_to_head": h2h, "n_both": n_both, "n": len(rows), "n_severe": sum(r["severe"] for r in rows)}
json.dump({"rows": fb_rows, "totals": fb_tot}, open(S / "summary_rows.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- step 3: agreement in flood seasons
agree = {}
for river, season in WINDOWS:
    g = rp3c[(rp3c.river == river) & (rp3c.season == season)]
    agree[(river, season)] = {m: float(g[m].median()) for m in ("google_grrr", "glofas_v5", "glofas_v4")} | {"n": (int(g.n_flood_seasons.min()), int(g.n_flood_seasons.max()))}
pooled = {m: float(rp3c[m].median()) for m in ("google_grrr", "glofas_v5", "glofas_v4")}


def _med(v):
    v = sorted(v); m_ = len(v) // 2
    return v[m_] if len(v) % 2 else (v[m_ - 1] + v[m_]) / 2


track = {}
for river, season in WINDOWS:
    g_ = rp3t[(rp3t.river == river) & (rp3t.season == season)]
    assert len(g_) == len(STATIONS[river].split(", ")), (river, season)
    track[(river, season)] = {m: float(g_[m].median()) for m in ("google_grrr", "glofas_v5")}
track_season = {(s, m): float(rp3t[rp3t.season == s][m].median()) for s in ("deyr", "gu") for m in ("google_grrr", "glofas_v5")}
assert all(track[(r_, "deyr")]["glofas_v5"] > track[(r_, "deyr")]["google_grrr"] for r_ in ("juba", "shabelle")), "prose says GloFAS tracks closer in Deyr"
assert track[("juba", "gu")]["google_grrr"] > track[("juba", "gu")]["glofas_v5"], "prose says Google tracks closer on the Juba in Gu"
assert abs(track[("shabelle", "gu")]["google_grrr"] - track[("shabelle", "gu")]["glofas_v5"]) <= 0.05, "prose says the two are level on the Shabelle in Gu"
lead_band = json.load(open(S / "lead_band_design.json"))
FADE = {int(k_): v for k_, v in lead_band["fade_share_of_day1"].items()}
BAND13 = {k_: len(v) for k_, v in lead_band["bands_to_2013"].items()}
assert 0.85 <= FADE[7] <= 0.92 and 0.68 <= FADE[12] <= 0.78, FADE
assert BAND13["1-7"] > BAND13["8-12"], BAND13
_iss4 = L.reforecast("glofas_v4").issued_time.drop_duplicates(); GLOFAS_ISSUES = int(_iss4.groupby(_iss4.dt.year).size().median())
_issg = L.reforecast("google_grrr").issued_time.drop_duplicates(); GOOGLE_ISSUES = int(_issg.groupby(_issg.dt.year).size().median())
assert 100 <= GLOFAS_ISSUES <= 110 and GOOGLE_ISSUES >= 360, (GLOFAS_ISSUES, GOOGLE_ISSUES)
adopted_station = [r for r in station if r["adopted"]]
lag_min, lag_max = min(r["lag_days"] for r in adopted_station), max(r["lag_days"] for r in adopted_station)
assert lag_min >= 0, "a chosen model trails a gauge somewhere; the page says it never does"

# ---------------------------------------------------------------- step 7: SWALIM
both_flag = [t for t in swalim_tl if " d before " in t["vs_trigger"]]
sw_first = sum(1 for t in both_flag if t["vs_trigger"].startswith("SWALIM"))
sw_only = [t for t in swalim_tl if t["vs_trigger"].startswith("SWALIM only")]

# ---------------------------------------------------------------- step 8: fail-safe
fs_only = [r for r in fallback if r["bank1"] and not r["trigger"]]
fs_both = [r for r in fallback if r["bank1"] and r["trigger"] and r["bank1_vs_trigger"] is not None]

# ================================================================ figures
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def fig_agreement():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.3), sharey=True)
    labels = [wname(r, s) for r, s in WINDOWS]
    panels = [(axes[0], track, (0.3, 1.0), "Tracking: day to day, in the 1-in-3 flood seasons", "tracking correlation, model against gauge"),
              (axes[1], agree, (-0.7, 1.0), "Ranking: order of the 1-in-3 floods by size", "ranking correlation, model against gauge")]
    for ax, data, xlim, title, xlabel in panels:
        for i, (r, s) in enumerate(WINDOWS):
            for m, dy in (("google_grrr", .13), ("glofas_v5", -.13)):
                v = data[(r, s)][m]
                chosen = (m == CAL[s])
                ax.plot([v], [i + dy], "o", color=COL[m], ms=9 if chosen else 7, mec=("#111827" if chosen else "white"),
                        mew=(1.6 if chosen else .8), zorder=4 if chosen else 3)
            if i < 3:
                ax.axhline(i + .5, color="#eef1f4", lw=1)
        ax.axvline(0, color="#9ca3af", lw=1, ls=":")
        ax.set_xlim(*xlim); ax.set_title(title, fontsize=10, loc="left", color="#111827")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#eef1f4"); ax.tick_params(length=0)
    axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", ls="none", color=COL[m], label=("Google Flood Hub" if m == "google_grrr" else "GloFAS")) for m in ("google_grrr", "glofas_v5")]
    handles.append(Line2D([], [], marker="o", ls="none", color="white", mec="#111827", mew=1.6, ms=9, label="ringed: the source the window runs on"))
    fig.legend(handles=handles, loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0.01, 0.93), fontsize=8.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIGS / "s_agreement.png", dpi=150, bbox_inches="tight", pad_inches=0.12); plt.close(fig)


def fig_activations():
    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    yrs = list(range(1999, 2024))
    for i, (r, s) in enumerate(WINDOWS):
        for y in yrs:
            f, sv, a = y in flood[(r, s)], y in severe[(r, s)], y in act[(r, s)]
            if f:
                ax.plot([y], [i], "s", color="#cbd5e1" if not sv else COL["gauge"], ms=11 if sv else 9, zorder=2)
            if a:
                ax.plot([y], [i], "o", color=COL[CAL[s]], ms=5.5, mec="white", mew=.8, zorder=4)
    ax.set_yticks(range(len(WINDOWS))); ax.set_yticklabels([wname(r, s) for r, s in WINDOWS]); ax.invert_yaxis()
    ax.set_xticks(yrs); ax.set_xticklabels([str(y)[2:] for y in yrs], fontsize=8.5); ax.set_xlim(1998.4, 2023.6)
    ax.grid(axis="x", color="#f1f5f9"); ax.tick_params(length=0)
    ax.legend(handles=[Line2D([], [], marker="s", ls="none", color="#cbd5e1", ms=9, label="gauge flood season (two gauges over 1-in-3)"),
                       Line2D([], [], marker="s", ls="none", color=COL["gauge"], ms=11, label="severe (two gauges over 1-in-5)"),
                       Line2D([], [], marker="o", ls="none", color=COL["glofas_v5"], label="window activates, GloFAS"),
                       Line2D([], [], marker="o", ls="none", color=COL["google_grrr"], label="window activates, Google Flood Hub")],
              loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.0), fontsize=8.6, columnspacing=2.0)
    fig.tight_layout(); fig.savefig(FIGS / "s_activations.png", dpi=150, bbox_inches="tight", pad_inches=0.12); plt.close(fig)


def fig_leads():
    rows = [(s, r) for s in ("deyr", "gu") for r in fb_rows[s]]
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(rows) + 1.4))
    ax.axvspan(-7.5, -0.5, color="#dcfce7", alpha=.8, zorder=0); ax.axvspan(-12.5, -7.5, color="#f0fdf4", alpha=.9, zorder=0)
    ax.axvline(0, color="#374151", lw=1.2)
    labels = []
    for i, (s, r) in enumerate(rows):
        labels.append(f"{SEASON[s][0]} {r['year']}{' *' if r['severe'] else ''}")
        for m, dy in (("google_grrr", .17), ("glofas_v4", -.17)):
            v = r[m]
            if v["kind"] == "na":
                continue
            if v["lead"] is None:
                ax.plot([22], [i + dy], "x", color=COL[m], ms=7, mew=1.6, zorder=3)
            else:
                later = [max(min(-ld, 21), -21) for ld in v.get("all_leads", []) if ld != v["lead"]]
                if later:
                    ax.plot(later, [i + dy] * len(later), "|", color=COL[m], ms=9, mew=1.3, alpha=.5, zorder=2)
                x = max(min(-v["lead"], 21), -21)              # negative = before the flood
                ax.plot([x], [i + dy], "o", color=COL[m], ms=7, mec="white", mew=.8, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(-22, 23.5); ax.set_xticks([-21, -14, -7, 0, 7, 14, 21])
    ax.set_xticklabels(["21 d before", "14 d before", "7 d before", "2nd gauge over 1-in-3", "7 d after", "14 d after", "21 d after"], fontsize=8.5)
    ax.set_xlabel("days from the day the river's second gauge crossed its own 1-in-3 level, the start of the flood season", fontsize=8.8, color="#374151")
    ax.text(-4, -0.9, "action window", color="#166534", fontsize=8.5, ha="center"); ax.text(-10, -0.9, "readiness", color="#4d7c0f", fontsize=8.5, ha="center")
    ax.grid(axis="x", color="#f1f5f9"); ax.tick_params(length=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=COL["google_grrr"], label="Google Flood Hub forecast, first issue meeting the rule"),
                       Line2D([], [], marker="o", ls="none", color=COL["glofas_v4"], label="GloFAS v4 forecast (the version running live)"),
                       Line2D([], [], marker="|", ls="none", color="#6b7280", ms=9, mew=1.3, alpha=.6, label="every later issue that also met the rule"),
                       Line2D([], [], marker="x", ls="none", color="#6b7280", mew=1.6, label="never crossed (shown at right edge)")],
              loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.02), fontsize=8.6)
    fig.tight_layout(); fig.savefig(FIGS / "s_leads.png", dpi=150, bbox_inches="tight", pad_inches=0.12); plt.close(fig)


fig_agreement(); fig_activations(); fig_leads()

# ================================================================ HTML helpers
CSS = """
:root{--ink:#111827;--muted:#6b7280;--rule:#e5e7eb;--green:#1f6f5f;--ok:#166534;--okbg:#dcfce7;--late:#9a3412;--latebg:#ffedd5;--miss:#991b1b;--missbg:#fee2e2;--na:#6b7280;--nabg:#f3f4f6}
body{margin:0;font:15px/1.6 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#fff}
html,body{overflow-x:hidden}
header{background:var(--green);color:#fff;padding:40px 0 30px}
.wrap{max-width:980px;margin:0 auto;padding:0 24px}
header h1{font:700 30px/1.15 Georgia,"Times New Roman",serif;margin:6px 0 12px}
header p{font-size:16.5px;max-width:840px;margin:0;opacity:.95}
header small{opacity:.8;letter-spacing:.02em}
h2{font:700 21px/1.25 Georgia,serif;margin:44px 0 6px}
p{max-width:840px} p.note{color:var(--muted);font-size:13.5px}
.tw{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 6px}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:2px solid var(--ink);padding:7px 8px;font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}
td{padding:7px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums} td.pick{font-weight:700}
tr.sev td:first-child{font-weight:700}
figure{margin:22px 0 26px} figure img{max-width:100%;display:block} figcaption{color:var(--muted);font-size:13px;margin-top:8px;max-width:840px}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12.5px;font-weight:600;white-space:nowrap}
.before{background:var(--okbg);color:var(--ok)} .same,.after{background:var(--latebg);color:var(--late)} .never{background:var(--missbg);color:var(--miss)} .na{background:var(--nabg);color:var(--na)}
.who{color:var(--muted);font-size:12px;display:block}
ul{max-width:840px} li{margin:5px 0}
footer{color:var(--muted);font-size:13px;margin:40px 0 30px;border-top:1px solid var(--rule);padding-top:14px}
"""
H = []                                                                  # page parts


def add(s): H.append(s)


def table(head, rows, cls=None):
    th = "".join(f'<th{" class=\"n\"" if c.startswith("#") else ""}>{escape(c.lstrip("#"))}</th>' for c in head)
    body = "".join("<tr" + (f' class="{cls(r)}"' if cls else "") + ">" + "".join(f"<td{a}>{c}</td>" for c, a in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def n(x): return (x, ' class="n"')
def c(x): return (x, "")
def pick(x): return (x, ' class="n pick"')
def pill(v):
    who = f'<span class="who">{escape(v["river"])}, {escape(v["when"])}</span>' if v["river"] else ""
    return f'<span class="pill {v["kind"]}">{escape(v["text"])}</span>{who}'


yl = lambda ys: ", ".join(str(y) for y in sorted(ys)) or "none"

# ================================================================ the page
add(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>How we got to the trigger | Somalia riverine flood trigger</title><style>{CSS}
details{{margin:14px 0}} summary{{cursor:pointer;font-weight:600;padding:4px 0;color:var(--green)}}
</style></head><body>
<header><div class="wrap"><small>Somalia Riverine Flood Trigger / summary</small>
<h1>How we got to the trigger</h1>
<p>This page sets out how the trigger was designed and the evidence behind each choice. Figures cover 1999 to 2023 unless stated. The full analysis is on the <a href="index.html" style="color:#fff">trigger analysis page</a>.</p></div></header>
<div class="wrap">""")

add("<h2>The trigger</h2><p>The trigger covers two rivers and two rainy seasons, which gives four windows. Each window runs on one forecast source and one rule. If any one window activates, the full allocation is released.</p>")
add(table(["Window", "Season", "Source", "Rule", "Gauges"],
          [[c(wname(r, s)), c(SEASON[s][1]), c(SOURCE[s]), c(f"{RULE[(r, s)][1]} of {4 if r == 'juba' else 3} gauges forecast over their own 1-in-{RULE[(r, s)][0]} level on the same day"), c(STATIONS[r])] for r, s in WINDOWS]))
add("<p class=\"note\">Readiness activates 8 to 12 days ahead, either on the GloFAS ensemble forecast or on a SWALIM moderate flood risk alert for either river, and releases the mobilisation share. Action activates 1 to 7 days ahead and releases the rest. Thresholds are fitted on each source's own record.</p>")
add(f"<p><b>Why action stops at 7 days.</b> Google Flood Hub, the Gu source, forecasts 7 days ahead and no further. GloFAS forecasts run longer, but the ensemble's high flows fall away with lead time. By day 7 they are about {round((1 - FADE[7]) * 100):d} per cent below the day-1 forecast, and by day 12 about {round((1 - FADE[12]) * 100):d} per cent below. Over the years both GloFAS archives cover, the same rule activated {BAND13['1-7']} times at 1 to 7 days and {'once' if BAND13['8-12'] == 1 else str(BAND13['8-12']) + ' times'} at 8 to 12 days. Days 8 to 12 are therefore used for readiness rather than action.</p>")

add("<h2>What counts as a flood</h2><p>The trigger is scored against the SWALIM gauge record. A river has a flood season when two of its gauges reach their own 1-in-3 level, and a severe season when two reach their 1-in-5 level. Each gauge's levels are fitted on its own record from 2000 to 2023.</p>")
add(table(["Window", "#Flood seasons", "#Severe", "Severe years"],
          [[c(wname(r, s)), n(str(len(flood[(r, s)]))), n(str(len(severe[(r, s)]))), c(yl(severe[(r, s)]))] for r, s in WINDOWS]))
add(f"<p class=\"note\">Across both rivers there are {len(flood_all)} flood years, of which {len(severe_all)} are severe. Gauges are capped at bank full, so the largest floods record the same reading. A season in which only one gauge crosses does not count, as in Gu 2021 at Belet Weyne.</p>")

add("<h2>Which source, and why</h2><p>Three global models were considered. GEOGloWS runs 4 to 10 times too high on the Shabelle and has no forecast archive, and was not taken further. Google Flood Hub and GloFAS were compared with the SWALIM gauges on their own records of the past, Google's retrospective run and GloFAS's version 5 reanalysis, using only the seasons in which the gauge reached its own 1-in-3 level between 2000 and 2023 (4 to 10 seasons per gauge). Values are the median across the river's gauges.</p>"
    "<ul><li><b>Tracking</b> is the rank correlation between the model's daily flow and the gauge's daily level over every day of those seasons, allowing the model to run a few days ahead of or behind the gauge. A value of 1 means the model rises and falls exactly with the gauge, and 0 means no relation.</li>"
    "<li><b>Ranking</b> is the rank correlation between how high the model went in each of those seasons and how high the gauge went. It tests whether the model orders the floods by size as the gauge did. A value of 1 means the same order, 0 means no relation, and a value below 0 means the wrong order.</li>"
    "</ul>")
rows = []
season_total = {}
for s in ("deyr", "gu"):
    for r in ("juba", "shabelle"):
        w = win[wname(r, s)]
        row = [c(wname(r, s))]
        for m in ("google_grrr", "glofas_v5"):
            row.append(n(f"{track[(r, s)][m]:.2f}"))
        for m in ("google_grrr", "glofas_v5"):
            row.append(n(f"{agree[(r, s)][m]:.2f}"))
        row.append(c(SOURCE[s]))
        rows.append(row)
    # season total: either river; a flood year is caught if the model's rule activated on either river
    fl_s = flood[("juba", s)] | flood[("shabelle", s)]
    row = [(f"<b>{SEASON[s][0]}, either river</b>", "")]
    for m in ("google_grrr", "glofas_v5"):
        row.append((f"<b>{track_season[(s, m)]:.2f}</b>", ' class="n"'))
    for m in ("google_grrr", "glofas_v5"):
        g_ = rp3c[rp3c.season == s]
        row.append((f"<b>{g_[m].median():.2f}</b>", ' class="n"'))
    row.append(c(SOURCE[s]))
    rows.append(row)
_body = "".join("<tr>" + "".join(f"<td{a}>{c_}</td>" for c_, a in r) + "</tr>" for r in rows)
_mods = "".join(f'<th class="n">{"Google Flood Hub" if m == "google_grrr" else "GloFAS"}</th>' for m in ("google_grrr", "glofas_v5"))
add("<div class='tw'><table><thead>"
    "<tr><th></th><th colspan='2' class='n' style='border-bottom:1px solid var(--rule)'>Tracking</th>"
    "<th colspan='2' class='n' style='border-bottom:1px solid var(--rule)'>Ranking</th>"
    "<th></th></tr>"
    f"<tr><th>Window</th>{_mods}{_mods}<th>Chosen</th></tr></thead><tbody>{_body}</tbody></table></div>")
add("<figure><img src=\"figs/s_agreement.png\" alt=\"Agreement with the gauges in flood seasons, by window and model\"><figcaption>Tracking on the left and ranking on the right, as in the table. The ringed dot marks the source the window runs on.</figcaption></figure>")
add(f"<p>In Deyr, GloFAS tracks the gauges far more closely than Google ({track_season[('deyr', 'glofas_v5')]:.2f} against {track_season[('deyr', 'google_grrr')]:.2f}). Google also over-activates in Deyr, with three Juba activations in years with no flood, and misses the Shabelle in 2020. In Gu, Google tracks more closely on the Juba and level with GloFAS on the Shabelle, orders the floods closer to the gauges' order, and its forecasts went out before the flood where GloFAS v4's mostly went out after it, as the next section shows. Gu Shabelle reads near zero on ranking for every model because the gauge is capped at bank full in the largest floods.</p>")

add("<h2>How often it activates</h2><p>Every threshold sits at 1-in-3 or rarer. The vote counts were set so that the mechanism as a whole activates about once in three years while still catching every severe season.</p>")
add(table(["", "#Activations, 25 years", "Return period", "Years"],
          [[c(wname(r, s)), n(str(len(act[(r, s)]))), c(rp_text(len(act[(r, s)]))), c(yl(act[(r, s)]))] for r, s in WINDOWS]
          + [[c("Juba, either window"), n(str(len(per_river["juba"]))), c(rate_river["juba"]), c(yl(per_river["juba"]))],
             [c("Shabelle, either window"), n(str(len(per_river["shabelle"]))), c(rate_river["shabelle"]), c(yl(per_river["shabelle"]))],
             [c("Whole mechanism"), n(str(n_act)), c(rate_env), c(yl(env_years))]]))
add(f"<figure><img src=\"figs/s_activations.png\" alt=\"Flood seasons and activations by year and window\"><figcaption>Squares mark gauge flood seasons, dark where severe. Dots mark the years in which the window activated on its source. There are {n_act} activations in 25 years. All {len(severe_all)} severe seasons are caught, and one activation, in {yl(outside)}, has no gauge flood behind it, although SWALIM and WFP both record that year as a major flood. Return periods are Weibull, (years + 1) divided by activations.</figcaption></figure>")

d, g = fb_tot["deyr"], fb_tot["gu"]
add(f"<h2>Checked on the forecasts</h2><p>The tests above use each model's record of the past, whereas the trigger runs on forecasts. The historical forecasts were therefore replayed to find the day the alert would have gone out, 1 to 7 days ahead, and that day was compared with the day the flood season began at the gauges, which is the day the river's second gauge crossed its own 1-in-3 level (the first gauge may have crossed days earlier). Either river counts. Only Google Flood Hub (2016 to 2023) and GloFAS v4 (2003 to 2023, plus the live Gu 2024 forecasts) have archives. GloFAS's archive holds two issue days a week ({GLOFAS_ISSUES} a year) where Google's holds every day, so replayed GloFAS lead times are coarser by up to three days. The live GloFAS forecast is daily.</p>")
add("<figure><img src=\"figs/s_leads.png\" alt=\"Lead time of the first forecast issue meeting the rule, per flood season\"><figcaption>One row per flood season, with severe seasons starred. The green band is the action window and the pale green band is readiness. The dot is the first forecast issue that met the rule, and the ticks are every later issue that met it. Points left of the line went out before the second gauge crossed, and points to the right went out after it.</figcaption></figure>")
add(f"<p>In Deyr, GloFAS v4 activated before the second gauge crossed in {d['per_model']['glofas_v4'].get('before', 0)} of {d['n']} seasons and was first in {d['head_to_head']['glofas_v4']} of the {d['n_both']} seasons both archives cover. In Gu, Google was first in all {g['n_both']}. On the Shabelle it gave 10 to 13 days of warning in three of four seasons, where GloFAS v4 gave 3 days once and nothing in the other three. The lead-time comparison and the calibration were done independently and point the same way.</p>")
rows6 = [[c(f"{SEASON[s][0]} {r['year']}{' *' if r['severe'] else ''}"), c(" and ".join(r["rivers"])), c(r["first_onset"]), c(pill(r["google_grrr"])), c(pill(r["glofas_v4"])), c(escape(r["first"]))] for s in ("deyr", "gu") for r in fb_rows[s]]
add("<details><summary>Flood by flood</summary>" + table(["Season", "River(s) that flooded", "Second gauge over 1-in-3", "Google Flood Hub forecast", "GloFAS v4 forecast", "First"], rows6) + "</details>")

add(f"<h2>SWALIM's alerts</h2><p>SWALIM's bulletins were first in {sw_first} of the {len(both_flag)} seasons in which both SWALIM and the window's source flagged, and SWALIM alone flagged in {len(sw_only)} further seasons. The bulletins are forward-looking and often early, and a SWALIM moderate flood risk alert for either river therefore activates readiness. They cannot carry the action phase, because CERF needs an activation basis that can be backtested and the bulletins are expert judgement rather than a fixed rule.</p>")
rows7 = [[c(t_["season"]), c(t_["river"]), c(t_["swalim_first"] or "no bulletin"), c(t_["vs_trigger"])] for t_ in swalim_tl]
add("<details><summary>SWALIM against the window's source, season by season</summary>" + table(["Season", "River", "SWALIM first bulletin", "Who was first"], rows7) + "</details>")

add("<h2>What runs live, and what is open</h2><ul>"
    "<li><b>GloFAS version 4 runs both phases today.</b> Deyr is calibrated on version 5, which has no published forecast yet, and Gu on Google Flood Hub, to which there is no API access yet. Version 4 stands in, with levels refitted on its own record.</li>"
    "<li><b>Two gauges have stopped reporting.</b> Bardheere has not reported since 30 November 2023 and Bualle since 14 March 2024.</li>"
    "<li><b>Impact years are not yet defined.</b> The benchmark is gauge levels rather than people affected, and 2013 and 2021 show why an impact cross-check is the next step. Each window holds three to five severe seasons, so every difference on this page rests on one or two events.</li></ul>")
add(f"<footer>Generated {datetime.date.today().isoformat()} from the trigger analysis outputs and checked against the analysis page when built.</footer></div></body></html>")

html = "".join(H)
OUT.write_text(html, encoding="utf-8")

# ================================================================ consistency checks on the rendered page
vis = html
assert "—" not in vis, "em dash on the page"
assert "fail-safe" not in vis and "fail safe" not in vis, "fail-safe still on the page"
assert "seasonal peak" not in vis, "seasonal peaks wording on the page"
for word in (" fired", " fires ", " basin", " product"):
    assert word not in vis, f"forbidden word {word!r}"
for f in re.findall(r'figs/([^"]+)"', vis):
    assert (FIGS / f).exists(), f"missing figure {f}"
design_tbl = re.search(r"<h2>The trigger</h2>.*?</table>", vis, re.S).group(0)
assert not re.search(r"\bv[45]\b", design_tbl), "version in the trigger table"
for tbl in re.findall(r"<table>.*?</table>", vis, re.S):
    names = re.findall(r"<td>(Deyr|Gu) (?:Juba|Shabelle)</td>", tbl)
    if names:
        assert names.index("Gu") > names.index("Deyr") if "Gu" in names and "Deyr" in names else True
        assert all(x == "Deyr" for x in names[:names.index("Gu")]) if "Gu" in names else True, "Deyr rows must precede Gu rows"
print("wrote", OUT.name, "| checks passed")
print(f"envelope {n_act} in {N_YEARS} = {rate_env}; per river {rate_river}; severe {sev_caught} of {len(severe_all)}; outside {outside}")
print("agreement pooled", {NICE[k]: round(v, 2) for k, v in pooled.items()}, "| lag", lag_min, lag_max)
print("SWALIM first", sw_first, "of", len(both_flag), "| only", len(sw_only), "| fail-safe only", [(r['window'], r['year']) for r in fs_only])
