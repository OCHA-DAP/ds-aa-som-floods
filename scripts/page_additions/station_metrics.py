"""Station by station: how each model tracks each gauge, per season. Rank
correlation at the best lag against the gauge's own level record, and whether the
model's own 1-in-3 / 1-in-5 crossing reproduces the gauge's own crossing, year by
year. Writes station_metrics.json and station_metrics_table.html."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import somlib as L
from src.constants import TRIGGER_CONFIG
from src.utils import weibull_level

S = Path(__file__).parent


def best_lag_rho(model_s, gauge_s):
    best, best_lag = 0.0, 0
    for lag in range(L.MIN_LAG, L.MAX_LAG + 1):
        g = gauge_s.copy(); g.index = g.index - pd.Timedelta(days=lag)
        j = pd.concat([model_s, g], axis=1, join="inner").dropna()
        if len(j) < L.MIN_OBS:
            continue
        r = j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
        if pd.notna(r) and abs(r) > abs(best):
            best, best_lag = float(r), lag
    return best, best_lag


lv = L.levels()
out = []
for river in ("juba", "shabelle"):
    for st in L.TRIGGER_STATIONS[river]:
        g_all = lv[lv.station == st].set_index("date")["level_m"].dropna().sort_index()
        for season in ("deyr", "gu"):
            g = g_all[g_all.index.month.isin(L.SEASONS[season])]
            g_mod = g[(g.index.year >= 2000) & (g.index.year <= L.Y1)]
            am_g = g_mod.groupby(g_mod.index.year).max()
            cnt = g_mod.groupby(g_mod.index.year).size()
            ok_years = set(cnt[cnt >= 30].index)
            lev = {rp: float(weibull_level(am_g.dropna().values, rp)) if len(am_g.dropna()) else np.nan for rp in (3, 5)}
            g_years = {rp: set(am_g[am_g >= lev[rp]].index) & ok_years if not np.isnan(lev[rp]) else set() for rp in (3, 5)}
            adopted = TRIGGER_CONFIG[(river, season)]["source"]
            for model in L.MODELS:
                s = L.season_series(model, st, season)
                if len(s) < 100:
                    continue
                rho, lag = best_lag_rho(s, g)
                am_m = s.groupby(s.index.year).max().dropna()
                rec = {"river": river, "station": L.NAME[st], "season": season.title(), "model": L.NICE[model],
                       "adopted": model == adopted, "rho": round(rho, 2), "lag_days": lag,
                       "gauge_years": len(ok_years), "gauge_level_rp3": lev[3], "gauge_level_rp5": lev[5]}
                for rp in (3, 5):
                    thr = L.model_threshold(model, st, season, rp)
                    m_years = set(am_m[am_m >= thr].index) & ok_years
                    c = L.contingency(m_years, g_years[rp], ok_years)
                    rec[f"rp{rp}"] = {"gauge_events": len(g_years[rp]), "hits": c["hits"], "misses": c["misses"],
                                      "false_alarms": c["false_alarms"], "POD": c["POD"], "FAR": c["FAR"], "CSI": c["CSI"]}
                out.append(rec)
                print(f"{river:9s}{L.NAME[st]:12s}{season:5s}{L.NICE[model]:10s} rho {rho:5.2f} lag {lag:3d} | "
                      f"RP3 {c['hits']}/{len(g_years[3])} hits FA {rec['rp3']['false_alarms']} | RP5 {rec['rp5']['hits']}/{len(g_years[5])} FA {rec['rp5']['false_alarms']} | yrs {len(ok_years)}")
json.dump(out, open(S / "station_metrics.json", "w"), indent=1, default=float)


def f2(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def table(river):
    trs = []
    last = None
    for r in [o for o in out if o["river"] == river]:
        key = (r["station"], r["season"])
        first = key != last
        last = key
        name = f"<strong>{r['model']}</strong>" if r["adopted"] else r["model"]
        style = " style='border-top:2px solid #cbd5e1'" if first else ""
        trs.append(f"<tr{style}><td>{r['station'] if first else ''}</td><td>{r['season'] if first else ''}</td><td>{name}</td>"
                   f"<td>{f2(r['rho'])}</td><td>{r['lag_days']:+d}</td><td>{r['gauge_years']}</td>"
                   f"<td>{r['rp3']['hits']} of {r['rp3']['gauge_events']}</td><td>{r['rp3']['false_alarms']}</td>"
                   f"<td>{f2(r['rp3']['POD'])}</td><td>{f2(r['rp3']['FAR'])}</td><td>{f2(r['rp3']['CSI'])}</td>"
                   f"<td>{r['rp5']['hits']} of {r['rp5']['gauge_events']}</td><td>{r['rp5']['false_alarms']}</td>"
                   f"<td>{f2(r['rp5']['POD'])}</td><td>{f2(r['rp5']['FAR'])}</td><td>{f2(r['rp5']['CSI'])}</td></tr>")
    return ('<div class="tablewrap">\n<table class="data" style="font-size:12px">\n<thead>'
            '<tr><th rowspan="2">station</th><th rowspan="2">season</th><th rowspan="2">model</th>'
            '<th rowspan="2">rank corr. (best lag)</th><th rowspan="2">lag, days</th><th rowspan="2">gauge years</th>'
            '<th colspan="5">gauge 1-in-3 crossings</th><th colspan="5">gauge 1-in-5 crossings</th></tr>'
            '<tr><th>caught</th><th>false alarms</th><th>POD</th><th>FAR</th><th>CSI</th>'
            '<th>caught</th><th>false alarms</th><th>POD</th><th>FAR</th><th>CSI</th></tr></thead>\n<tbody>'
            + "".join(trs) + "</tbody>\n</table>\n</div>")


html = (f'<h3>Juba</h3>\n{table("juba")}\n<h3>Shabelle</h3>\n{table("shabelle")}')
(S / "station_metrics_table.html").write_text(html, encoding="utf-8")
print("wrote station_metrics_table.html", len(out), "rows")
