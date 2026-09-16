"""Build pages/balanced/data.json - the open multi-model trigger with basin AND
season balance imposed as explicit constraints rather than left to fall out of
the search.

Candidates are the five still-reporting SWALIM gauges with all three products
competing freely (this is the variant that keeps GEOGloWS), filtered by the same
stage-1 rules as the adopted design: lag >= -3 days, rho >= 0.50, then everything
within 0.10 rho of the window's best.  The search maximises severe-year coverage
subject to the envelope staying at 1-in-3 or rarer, tie-broken on how level the
basin and season activation counts come out.

Run with the project interpreter from the corrected-benchmark checkout - see the
guard, and the note in scripts/build_comparison_page.py for why it matters:

    .venv/bin/python scripts/build_balanced_page.py
"""
import inspect, sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd, numpy as np
from src.constants import SEASONS
from src.utils import weibull_threshold
import envelope_search as es

if "max(SPAN)" not in inspect.getsource(es.gauge_consensus_years):
    sys.exit(
        "REFUSING TO RUN: scripts/envelope_search.py still fits return levels on an\n"
        "open-ended window, so the benchmark would carry a lookahead.\n"
        "Check out fix/corrected-benchmark-and-lag (or merge it) and rerun."
    )

D = ROOT / "data/processed"
ACTIVE = ["belet_weyne", "bulo_burti", "jowhar", "luuq", "dollow"]
SRCS = ["geoglows", "glofas_v5", "google_grrr"]
NICE = {"geoglows": "GEOGloWS", "glofas_v5": "GloFAS v5", "google_grrr": "Google GRRR"}
W = [("juba", "gu"), ("juba", "deyr"), ("shabelle", "gu"), ("shabelle", "deyr")]
Y0, Y1 = 1999, 2023
N = Y1 - Y0 + 1
YEARS = list(range(Y0, Y1 + 1))

lv = pd.read_parquet(D / "swalim_levels.parquet"); lv["date"] = pd.to_datetime(lv["date"])
flood, sev = es.benchmark_years_from_gauges(lv)
corr = pd.read_parquet(D / "workflow/som_gauge_correlations.parquet")
cand = corr[(corr.benchmark == "swalim") & (corr.best_lag >= -3) & (corr.best_rho >= 0.5)
            & corr.station.isin(ACTIVE) & corr.source.isin(SRCS)]
pools = {}
for (r, s), g in cand.groupby(["river", "season"]):
    g = g.sort_values("best_rho", ascending=False)
    pools[(r, s)] = g[g.best_rho >= g.best_rho.max() - 0.10].reset_index(drop=True)
dd = {m: pd.read_parquet(D / f"discharge_daily_{m}.parquet") for m in SRCS}
for m in dd:
    dd[m]["date"] = pd.to_datetime(dd[m]["date"])


def counts(r, s, rp):
    """peak simultaneous over-threshold pair count, per year"""
    cols = []
    for _, x in pools[(r, s)].iterrows():
        ser = dd[x.source][dd[x.source].station == x.station].set_index("date")["discharge"]
        ser = ser[ser.index.month.isin(SEASONS[s])]
        ser = ser[(ser.index.year >= Y0) & (ser.index.year <= Y1)]
        if len(ser) < 100:
            continue
        am = ser.groupby(ser.index.year).max().dropna()
        t = weibull_threshold(am.values, rp)
        if not np.isnan(t):
            cols.append((ser >= t).rename(f"{x.station}|{x.source}"))
    m = pd.concat(cols, axis=1).fillna(False)
    return m.sum(axis=1).groupby(m.index.year).max()


CNT = {(r, s, rp): counts(r, s, rp) for r, s in W for rp in (3, 4, 5, 6)}


def acts(r, s, rp, n):
    c = CNT[(r, s, rp)]
    return frozenset(c[c >= n].index)


grids = {w: [(rp, n) for rp in (3, 4, 5, 6) for n in range(2, len(pools[w]) + 1)] for w in W}
best = None
for cg in grids[W[0]]:
    jg = acts(*W[0], *cg)
    for cd in grids[W[1]]:
        jd = acts(*W[1], *cd); J = jg | jd
        for sg in grids[W[2]]:
            sgy = acts(*W[2], *sg); GU = jg | sgy
            for sd in grids[W[3]]:
                sdy = acts(*W[3], *sd); S = sgy | sdy
                U = J | S
                if not U or (N + 1) / len(U) < 3.0:
                    continue
                DE = jd | sdy
                # final tie-break prefers non-unanimous rules (a rule that needs
                # every unit in its pool has no tolerance for one feed going down)
                unan = sum(c[1] == len(pools[w]) for w, c in zip(W, (cg, cd, sg, sd)))
                key = (len(U & sev), -(abs(len(J) - len(S)) + abs(len(GU) - len(DE))),
                       -len(U - flood), -abs(len(J) - len(S)), -unan, len(U))
                if best is None or key > best[0]:
                    best = (key, dict(zip([f"{r}_{s}" for r, s in W], [cg, cd, sg, sd])),
                            dict(zip([f"{r}_{s}" for r, s in W], [jg, jd, sgy, sdy])),
                            J, S, GU, DE, U)
_, cfg, legs, J, S, GU, DE, U = best


def label(pool):
    order, cnt = [], {}
    for src in pool.source:
        if src not in cnt:
            order.append(src); cnt[src] = 0
        cnt[src] += 1
    return " + ".join(NICE[s] + (f" &times;{cnt[s]}" if cnt[s] > 1 else "") for s in order)


meta, table = {}, []
for r, s in W:
    wk = f"{r}_{s}"; rp, n = cfg[wk]; pool = pools[(r, s)]
    meta[wk] = {
        "rp": rp, "n_req": n, "pool": len(pool), "label": label(pool),
        "pairs": [{"station": x.station, "source": NICE[x.source],
                   "rho": round(float(x.best_rho), 3), "lag": int(x.best_lag)}
                  for _, x in pool.iterrows()],
        "years": sorted(legs[wk]),
        "leg_rp": round((N + 1) / max(len(legs[wk]), 1), 1),
        "unanimous": n == len(pool),
    }
    c = CNT[(r, s, rp)]
    for y in YEARS:
        table.append({"year": y, "window": wk, "n": int(c.get(y, 0)), "fire": y in legs[wk]})

out = {
    "years": YEARS, "windows": [f"{r}_{s}" for r, s in W], "meta": meta, "table": table,
    "bench": {f"{r}_{s}": {"sev": sorted(es.gauge_consensus_years(lv, r, s, 5)),
                           "mod": sorted(es.gauge_consensus_years(lv, r, s, 3))} for r, s in W},
    "basins": {"juba": {"n": len(J), "rp": round((N + 1) / len(J), 1), "years": sorted(J),
                        "unique": sorted(J - S)},
               "shabelle": {"n": len(S), "rp": round((N + 1) / len(S), 1), "years": sorted(S),
                            "unique": sorted(S - J)}},
    "seasons": {"gu": {"n": len(GU), "rp": round((N + 1) / len(GU), 1)},
                "deyr": {"n": len(DE), "rp": round((N + 1) / len(DE), 1)}},
    "envelope": {"years": sorted(U), "n": len(U), "rp": round((N + 1) / len(U), 1),
                 "severe_caught": len(U & sev), "n_severe": len(sev), "false": sorted(U - flood)},
}
(ROOT / "pages/balanced").mkdir(exist_ok=True)
(ROOT / "pages/balanced/data.json").write_text(json.dumps(out, indent=1))
e = out["envelope"]
print(f"envelope {e['n']} 1-in-{e['rp']}  severe {e['severe_caught']}/{e['n_severe']}  false {e['false']}")
print(f"basins Juba {len(J)} / Shabelle {len(S)}   seasons Gu {len(GU)} / Deyr {len(DE)}")
for wk, m in meta.items():
    print(f"  {wk:15s} RP{m['rp']} {m['n_req']} of {m['pool']}  {m['years']}")
print("wrote pages/balanced/data.json")
