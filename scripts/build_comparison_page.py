"""Build pages/comparison/data.json - mixed models vs one model per basin-season.

Both setups run the full two-stage design algorithm on the five still-reporting
SWALIM gauges, over GloFAS v5 + Google only (GEOGloWS is excluded: its forecast
archive is too short to fit or validate thresholds on).

  Stage 1, select the best station-models.  Drop a unit whose signal trails the
  reference gauge by more than 3 days, or whose rank correlation against the
  gauge is below 0.50, then keep everything within 0.10 rho of the best.  For the
  mixed setup that relative floor runs model-blind across every (station,
  product) pair; for the one-model setup it runs WITHIN each product, so a
  product is judged on its own best stations rather than being crowded out of a
  window another product dominates.  That asymmetry lets the one-model setup
  reach units the mixed setup rejects - it bites only in Shabelle Deyr, where the
  search declines the option anyway.

  Stage 2, calibrate.  Search the return-period threshold and the number of units
  required, per window, for the balanced and most accurate combination that keeps
  the envelope between 1-in-3.0 and 1-in-3.6.

Run it with the project interpreter, from a checkout that carries the corrected
benchmark:

    git checkout fix/corrected-benchmark-and-lag
    .venv/bin/python scripts/build_comparison_page.py

The guard below enforces that.  It matters: this script imports
scripts/envelope_search.py, and the uncorrected copy fits return levels on
2000-2026 while counting crossings inside 1999-2023.  That lookahead alone moves
the truth set from 7 severe years to 8 and silently changes every score on the
page - it produced a spurious mixed F1 of 0.75 once already.
"""
import inspect, sys, json, itertools, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd, numpy as np
from collections import Counter
from src.constants import SEASONS, TRIGGER_STATIONS
from src.utils import weibull_threshold, weibull_level
import envelope_search as es

if "max(SPAN)" not in inspect.getsource(es.gauge_consensus_years):
    sys.exit(
        "REFUSING TO RUN: scripts/envelope_search.py still fits return levels on an\n"
        "open-ended window, so the benchmark would carry a lookahead (8 severe years\n"
        "instead of 7) and every score on the page would be wrong.\n"
        "Check out fix/corrected-benchmark-and-lag (or merge it) and rerun."
    )

D = str(ROOT / "data/processed")
SRCS=["glofas_v5","google_grrr"]; ACTIVE={"belet_weyne","bulo_burti","jowhar","luuq","dollow"}
LAG_GUARD, MIN_RHO, REL_TOL = -3, 0.5, 0.10
W=[("juba","gu"),("juba","deyr"),("shabelle","gu"),("shabelle","deyr")]
NICE={"glofas_v5":"GloFAS v5","google_grrr":"Google GRRR"}
SHORT={"glofas_v5":"GloFAS v5","google_grrr":"Google"}
REF={"juba":"luuq","shabelle":"belet_weyne"}
YEARS=list(range(1999,2024))

lv=pd.read_parquet(f"{D}/swalim_levels.parquet"); lv["date"]=pd.to_datetime(lv["date"])
flood,sev=es.benchmark_years_from_gauges(lv)
TG=json.load(open(ROOT/"scripts/comparison_targets.json"))
targets={k:set(v) for k,v in TG["targets"].items()}
corr=pd.read_parquet(f"{D}/workflow/som_gauge_correlations.parquet")
cand=corr[(corr.benchmark=="swalim")&corr.station.isin(ACTIVE)&corr.source.isin(SRCS)
          &(corr.best_lag>=LAG_GUARD)&(corr.best_rho>=MIN_RHO)]
dd={m:pd.read_parquet(f"{D}/discharge_daily_{m}.parquet") for m in SRCS}
for m in dd: dd[m]["date"]=pd.to_datetime(dd[m]["date"])
dd4=pd.read_parquet(f"{D}/discharge_daily_glofas_v4.parquet"); dd4["date"]=pd.to_datetime(dd4["date"])
rf={"google_grrr":pd.read_parquet(f"{D}/reforecast_google_grrr.parquet"),
    "glofas_v4":pd.read_parquet(f"{D}/reforecast_glofas_v4.parquet")}
for k in rf: rf[k]["valid_time"]=pd.to_datetime(rf[k]["valid_time"])

def floor_pool(g):
    """keep everything within REL_TOL rho of the group's best"""
    if not len(g): return g
    return g[g.best_rho>=g.best_rho.max()-REL_TOL].sort_values("best_rho",ascending=False)

def pools_for(setup, r, s):
    """-> {tag: [(station, source, rho, lag), ...]}"""
    g=cand[(cand.river==r)&(cand.season==s)]
    if setup=="mixed":
        p=floor_pool(g)
        return {"mix":[(x.station,x.source,round(float(x.best_rho),3),int(x.best_lag)) for _,x in p.iterrows()]}
    out={}
    for m in SRCS:
        p=floor_pool(g[g.source==m])
        if len(p)>=2:
            out[m]=[(x.station,x.source,round(float(x.best_rho),3),int(x.best_lag)) for _,x in p.iterrows()]
    return out

def ser(st,src,s):
    x=dd[src]; v=x[x.station==st].set_index("date")["discharge"]
    v=v[v.index.month.isin(SEASONS[s])]; return v[(v.index.year>=1999)&(v.index.year<=2023)]
def over(st,src,s,rp):
    v=ser(st,src,s)
    if len(v)<100: return None
    am=v.groupby(v.index.year).max().dropna(); t=weibull_threshold(am.values,rp)
    return None if np.isnan(t) else (v>=t).rename(f"{st}|{src}")
def counts(units,s,rp):
    cols=[c for st,src,_,_ in units if (c:=over(st,src,s,rp)) is not None]
    if not cols: return None,0
    m=pd.concat(cols,axis=1).fillna(False); return m.sum(axis=1).groupby(m.index.year).max(), len(cols)

def search(setup):
    P={w:pools_for(setup,*w) for w in W}
    opts={}
    for w in W:
        for tag,units in P[w].items():
            for rp in (3,4,5,6):
                c,pool=counts(units,w[1],rp)
                if c is None or pool<2: continue
                for n in range(2,pool+1):
                    opts[(w,(tag,rp,n))]=(frozenset(c[c>=n].index),pool,units[:pool])
    keys={w:[k for (ww,k) in opts if ww==w] for w in W}
    best=None
    for combo in itertools.product(*[keys[w] for w in W]):
        L=[opts[(w,k)][0] for w,k in zip(W,combo)]; cnt=[len(x) for x in L]
        U=set().union(*L)
        if not U or not (3.0<=26/len(U)<=3.6): continue
        unan=sum(1 for w,k in zip(W,combo) if k[2]==opts[(w,k)][1])
        key=(-(max(cnt)-min(cnt)), len(U&sev), -len(U-flood), -unan)
        if best is None or key>best[0]: best=(key,combo,L,U)
    return best,opts,P

def gauge_cross(r,s,y,rp=3):
    x=lv[lv.station==REF[r]].set_index("date")["level_m"].dropna().sort_index()
    x=x[x.index.month.isin(SEASONS[s])]
    m=x[(x.index.year>=2000)&(x.index.year<=2023)]
    L=weibull_level(m.groupby(m.index.year).max().dropna().values,rp)
    yr=x[x.index.year==y]; hit=yr[yr>=L]
    return hit.index.min() if len(hit) else None

def leads(units,r,s,rp,n):
    cnt=Counter()
    for st,src,_,_ in units:
        fcm="google_grrr" if src=="google_grrr" else "glofas_v4"
        base=(dd4[dd4.station==st].set_index("date")["discharge"] if fcm=="glofas_v4" else ser(st,src,s))
        if fcm=="glofas_v4":
            base=base[base.index.month.isin(SEASONS[s])]; base=base[(base.index.year>=1999)&(base.index.year<=2023)]
        am=base.groupby(base.index.year).max().dropna()
        if not len(am): continue
        t=weibull_threshold(am.values,rp)
        d=rf[fcm]; d=d[(d.station==st)&(d.leadtime_days>=1)&(d.leadtime_days<=6)
                       &(d.valid_time.dt.month.isin(SEASONS[s]))]
        if not len(d): continue
        agg=(d.groupby(["issued_time","valid_time"])["discharge"].median() if fcm=="glofas_v4"
             else d.groupby(["issued_time","valid_time"])["discharge"].max()).reset_index()
        cnt.update(set(agg[agg.discharge>=t].valid_time.dt.normalize()))
    fire=sorted({d for d,k in cnt.items() if k>=n})
    by={}
    for d in fire: by.setdefault(d.year,[]).append(d)
    out=[]
    for y,ds in by.items():
        if y not in YEARS: continue
        gx=gauge_cross(r,s,y)
        if gx is not None: out.append(int((gx-min(ds)).days))
    return out

def prf(truth,acts):
    tp=len(acts&truth); fp=len(acts-truth); fn=len(truth-acts)
    tpr=tp/(tp+fn) if tp+fn else float("nan"); ppv=tp/(tp+fp) if tp+fp else float("nan")
    f1=(2*tpr*ppv/(tpr+ppv)) if tp and tpr+ppv else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"tpr":round(tpr,2),"ppv":round(ppv,2),"f1":round(f1,2)}

sevw={f"{r}_{s}":es.gauge_consensus_years(lv,r,s,5) for r,s in W}
out={"years":YEARS,"windows":[f"{r}_{s}" for r,s in W],
     "targets":{k:sorted(v) for k,v in targets.items()},
     "target_rule":f"CERF flood allocation, or EM-DAT >= {TG['threshold']:,} affected",
     "selection":{"lag_guard":LAG_GUARD,"min_rho":MIN_RHO,"rel_tol":REL_TOL},"setups":{}}
for setup in ("mixed","one"):
    (key,combo,L,U),opts,P=search(setup)
    meta={}; table=[]; allleads=[]
    for w,k,years in zip(W,combo,L):
        wk=f"{w[0]}_{w[1]}"; tag,rp,n=k
        yrs,pool,units=opts[(w,k)]
        c,_=counts(units,w[1],rp)
        srcs=sorted({s for _,s,_,_ in units})
        meta[wk]={"model":" + ".join(NICE[x] for x in srcs),"rp":rp,"n_req":n,"pool":pool,
                  "years":sorted(years),"leg_rp":round(26/max(len(years),1),1),
                  "units":[{"station":st.replace("_"," ").title(),"model":SHORT[sc],
                            "rho":rho,"lag":lag} for st,sc,rho,lag in units],
                  "candidates":len(cand[(cand.river==w[0])&(cand.season==w[1])]),
                  "vs_gauge":prf(sevw[wk],set(years)),"vs_target":prf(targets[wk],set(years))}
        ld=leads(units,w[0],w[1],rp,n); allleads+=ld
        meta[wk]["lead_days"]=round(float(np.mean(ld)),1) if ld else None
        for y in YEARS: table.append({"year":y,"window":wk,"n":int(c.get(y,0)),"fire":y in years})
    allt=set().union(*targets.values())
    out["setups"][setup]={"meta":meta,"table":table,"windows_counts":[len(x) for x in L],
      "envelope":{"years":sorted(U),"n":len(U),"rp":round(26/len(U),1),
                  "vs_gauge":prf(sev,U),"vs_target":prf(allt,U),
                  "lead_days":round(float(np.mean(allleads)),1) if allleads else None}}
out["bench"]={f"{r}_{s}":{"sev":sorted(es.gauge_consensus_years(lv,r,s,5)),
                          "mod":sorted(es.gauge_consensus_years(lv,r,s,3))} for r,s in W}
prev=json.loads((ROOT/"pages/comparison/data.json").read_text())
out["impact"]=prev["impact"]
(ROOT/"pages/comparison/data.json").write_text(json.dumps(out,indent=1))
for k,v in out["setups"].items():
    e=v["envelope"]
    print(f"\n{k.upper()}  windows {v['windows_counts']} | env {e['n']} 1-in-{e['rp']} | "
          f"gauge F1 {e['vs_gauge']['f1']} (tpr {e['vs_gauge']['tpr']} ppv {e['vs_gauge']['ppv']}) | "
          f"target F1 {e['vs_target']['f1']} | lead {e['lead_days']}d")
    for w,m in v["meta"].items():
        u=", ".join(f"{x['station']}·{x['model']}" for x in m["units"])
        print(f"   {w:15s} {m['n_req']}/{m['pool']} RP{m['rp']}  (from {m['candidates']} candidates)  {u}")
