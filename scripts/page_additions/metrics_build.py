"""Skill scores beyond POD, FAR and F1 for each window rule on each model, plus
ROC curves and AUC from sweeping the station return period at the adopted point
count. Writes figs/l_roc.png, metrics.json and metrics_tables.html."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import somlib as L
from src.constants import TRIGGER_CONFIG

S = Path(__file__).parent
FIGS = PAGE_DIR / "figs"
COLOR = {"google_grrr": "#1d4ed8", "glofas_v5": "#0f766e", "glofas_v4": "#9ca3af"}
RP_GRID = np.round(np.geomspace(1.3, 12, 28), 2)
UNIVERSE = set(L.SPAN)

out = {"windows": [], "envelope": {}}
roc = {}
env_years = set(); any_flood = set(); any_severe = set()
for river, season in L.WINDOWS:
    cfg = TRIGGER_CONFIG[(river, season)]
    flood, severe, ok = L.benchmark(river, season)
    any_flood |= flood; any_severe |= severe
    w = {"window": L.wname(river, season), "river": river, "season": season, "rp": cfg["rp"], "n_req": cfg["n_req"],
         "adopted_model": cfg["source"], "flood_years": sorted(flood), "severe_years": sorted(severe),
         "not_assessable": sorted(UNIVERSE - ok), "models": {}}
    for model in L.MODELS:
        yrs = L.activation_years(model, river, season, cfg["rp"], cfg["n_req"])
        if model == cfg["source"]:
            env_years |= yrs
        rec = {"activations": sorted(yrs),
               "vs_flood": L.contingency(yrs, flood, UNIVERSE),
               "vs_severe": L.contingency(yrs, severe, UNIVERSE)}
        # ROC: sweep the station return period at the adopted count
        pts_f, pts_s = [], []
        for rp in RP_GRID:
            a = L.activation_years(model, river, season, float(rp), cfg["n_req"])
            cf, cs = L.contingency(a, flood, UNIVERSE), L.contingency(a, severe, UNIVERSE)
            pts_f.append((rp, cf["POFD"], cf["POD"])); pts_s.append((rp, cs["POFD"], cs["POD"]))
        rec["roc_flood"] = pts_f; rec["roc_severe"] = pts_s
        rec["auc_flood"] = L.auc([(x, y) for _, x, y in pts_f])
        rec["auc_severe"] = L.auc([(x, y) for _, x, y in pts_s])
        w["models"][model] = rec
        roc[(river, season, model)] = rec
        print(f"{w['window']:14s} {L.NICE[model]:10s} act {len(yrs):2d} | severe POD {rec['vs_severe']['POD']:.2f} "
              f"FAR {rec['vs_severe']['FAR']:.2f} AUC {rec['auc_severe']:.2f} | flood POD {rec['vs_flood']['POD']:.2f} "
              f"FAR {rec['vs_flood']['FAR']:.2f} AUC {rec['auc_flood']:.2f}")
    out["windows"].append(w)

out["envelope"] = {"activations": sorted(env_years), "flood_years": sorted(any_flood), "severe_years": sorted(any_severe),
                   "vs_flood": L.contingency(env_years, any_flood, UNIVERSE),
                   "vs_severe": L.contingency(env_years, any_severe, UNIVERSE)}
print("envelope:", out["envelope"]["activations"], "severe", out["envelope"]["vs_severe"])
(S / "metrics.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

# ---- ROC figure: rows = benchmark, columns = windows -------------------------
fig, axes = plt.subplots(2, 4, figsize=(12.5, 7.2), sharex=True, sharey=True)
for j, (river, season) in enumerate(L.WINDOWS):
    cfg = TRIGGER_CONFIG[(river, season)]
    for i, (key, label) in enumerate((("roc_severe", "severe years (two-gauge 1-in-5)"),
                                      ("roc_flood", "flood years (two-gauge 1-in-3)"))):
        ax = axes[i, j]
        ax.plot([0, 1], [0, 1], color="#e5e7eb", lw=1, zorder=0)
        for model in L.MODELS:
            rec = roc[(river, season, model)]
            pts = sorted(set((x, y) for _, x, y in rec[key]) | {(0, 0), (1, 1)})
            xs, ys = zip(*pts)
            a = rec["auc_severe" if key == "roc_severe" else "auc_flood"]
            lw = 2.2 if model == cfg["source"] else 1.4
            ax.plot(xs, ys, color=COLOR[model], lw=lw, marker="o", ms=3, zorder=3 if model == cfg["source"] else 2,
                    label=f"{L.NICE[model]}  AUC {a:.2f}")
            # the adopted setting
            ad = [(x, y) for rp, x, y in rec[key] if abs(rp - cfg["rp"]) < 1e-6]
            if not ad:
                c = rec["vs_severe" if key == "roc_severe" else "vs_flood"]
                ad = [(c["POFD"], c["POD"])]
            if model == cfg["source"]:
                ax.plot(*ad[0], marker="D", color=COLOR[model], ms=9, mec="white", mew=1.2, zorder=6)
        ax.legend(loc="lower right", fontsize=7.6, frameon=False)
        if i == 0:
            ax.set_title(f"{L.wname(river, season)}\n{cfg['n_req']} of {len(L.TRIGGER_STATIONS[river])} points, "
                         f"adopted 1-in-{cfg['rp']}", fontsize=9, fontweight="bold", loc="left")
        if j == 0:
            ax.set_ylabel(f"POD vs {label}", fontsize=9)
        if i == 1:
            ax.set_xlabel("POFD (false alarms / non-flood years)", fontsize=9)
        ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
        ax.grid(color="#f3f4f6", lw=.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, length=0)
fig.suptitle("ROC by window: the station return period swept from 1-in-1.3 to 1-in-12 at the adopted point count; "
             "the diamond is the adopted model at its adopted return period", fontsize=9.5, x=.02, ha="left", y=.995)
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(FIGS / "l_roc.png", dpi=150)
print("wrote", FIGS / "l_roc.png")

# ---- HTML tables -------------------------------------------------------------
def f2(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def table(key, auc_key, title):
    rows = []
    for w in out["windows"]:
        for model in L.MODELS:
            r = w["models"][model]; c = r[key]
            bold = model == w["adopted_model"]
            name = f"<strong>{L.NICE[model]}</strong>" if bold else L.NICE[model]
            rows.append(f"<tr><td>{w['window']}</td><td>{name}</td>"
                        f"<td>{c['hits']}</td><td>{c['misses']}</td><td>{c['false_alarms']}</td><td>{c['correct_negatives']}</td>"
                        f"<td>{f2(c['POD'])}</td><td>{f2(c['FAR'])}</td><td>{f2(c['POFD'])}</td><td>{f2(c['CSI'])}</td>"
                        f"<td>{f2(c['bias'])}</td><td>{f2(c['PSS'])}</td><td>{f2(c['HSS'])}</td><td>{f2(c['F1'])}</td>"
                        f"<td>{f2(r[auc_key])}</td></tr>")
    e = out["envelope"][key]
    rows.append(f"<tr style='border-top:2px solid #cbd5e1'><td><strong>Envelope</strong> (any window)</td><td>adopted assignment</td>"
                f"<td>{e['hits']}</td><td>{e['misses']}</td><td>{e['false_alarms']}</td><td>{e['correct_negatives']}</td>"
                f"<td>{f2(e['POD'])}</td><td>{f2(e['FAR'])}</td><td>{f2(e['POFD'])}</td><td>{f2(e['CSI'])}</td>"
                f"<td>{f2(e['bias'])}</td><td>{f2(e['PSS'])}</td><td>{f2(e['HSS'])}</td><td>{f2(e['F1'])}</td><td></td></tr>")
    return (f'<p class="muted" style="margin:14px 0 4px"><em>{title}</em></p>\n<div class="tablewrap">\n'
            '<table class="data" style="font-size:12px">\n<thead><tr><th>window</th><th>model</th>'
            '<th>hits</th><th>misses</th><th>false alarms</th><th>correct negatives</th>'
            '<th>POD</th><th>FAR</th><th>POFD</th><th>CSI</th><th>bias</th><th>PSS</th><th>HSS</th><th>F1</th><th>AUC</th>'
            '</tr></thead>\n<tbody>' + "".join(rows) + "</tbody>\n</table>\n</div>")


html = (table("vs_severe", "auc_severe", "Against the severe years: two gauges of the river over their 1-in-5 level in the season.")
        + "\n" + table("vs_flood", "auc_flood", "Against the flood years: two gauges over their 1-in-3 level in the season."))
(S / "metrics_tables.html").write_text(html, encoding="utf-8")
print("wrote metrics_tables.html")
