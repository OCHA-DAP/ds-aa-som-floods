"""Why the action window stops at 7 days: the design rule on the GloFAS v4 reforecast,
scored by lead band. Same construction as somlib.first_issue_dates (ensemble median,
levels from the model's own record, design return period and vote count per window,
votes counted on one issue and one valid day), run once on leads 1-7 and once on leads
8-12, and lead by lead, over the 21 years both processed tables cover (2003-2023).
Writes lead_band_design.json. Run with SOM_DATA_REPO pointing at the checkout that holds
data/processed when this file sits in a worktree."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
import somlib as L  # noqa: E402  (puts the repo root on sys.path and resolves data/processed)

SPAN = set(range(2003, 2024))
WINDOWS = [("juba", "deyr"), ("shabelle", "deyr"), ("juba", "gu"), ("shabelle", "gu")]
from src.constants import TRIGGER_CONFIG as _TC
RULE = {k: (v["rp"], v["n_req"]) for k, v in _TC.items()}

a = L.reforecast("glofas_v4")                       # the processed table the page already uses for leads 1-7
b = pd.read_parquet(L.P / "reforecast_glofas_v4_lead8_12.parquet")  # the processed 8-12 table the readiness leg reads
print("leads 1-7 table", sorted(a.leadtime_days.unique()), str(a.issued_time.min())[:10], str(a.issued_time.max())[:10])
print("leads 8-12 table", sorted(b.leadtime_days.unique()), str(b.issued_time.min())[:10], str(b.issued_time.max())[:10])
rf = pd.concat([a[["issued_time", "valid_time", "station", "leadtime_days", "discharge"]], b[["issued_time", "valid_time", "station", "leadtime_days", "discharge"]]], ignore_index=True)
for col in ("issued_time", "valid_time"):
    rf[col] = pd.to_datetime(rf[col])
keep = {st for r in L.TRIGGER_STATIONS for st in L.TRIGGER_STATIONS[r]}
rf = rf[rf.station.isin(keep)]
med = rf.groupby(["issued_time", "valid_time", "station", "leadtime_days"]).discharge.median().reset_index()

flood, severe = set(), set()
for r, s in WINDOWS:
    fl, sv, _ = L.benchmark(r, s); flood |= fl & SPAN; severe |= sv & SPAN


def years(river, season, leads):
    rp, n_req = RULE[(river, season)]
    thr = L.model_thresholds("glofas_v4", river, season, rp)
    m = med[med.station.isin(L.TRIGGER_STATIONS[river]) & med.leadtime_days.between(*leads) & med.valid_time.dt.month.isin(L.SEASONS[season])]
    w = m.pivot_table(index=["issued_time", "valid_time"], columns="station", values="discharge")
    cols = [c for c in thr.index if c in w.columns and not np.isnan(thr[c])]
    votes = (w[cols] >= thr[cols]).sum(axis=1)
    hit = votes[votes >= n_req].reset_index()
    return set(hit["valid_time"].dt.year) & SPAN


def score(u):
    return {"n": len(u), "years": sorted(u), "severe": len(u & severe), "flood": len(u & flood), "noflood": sorted(u - flood), "missed_severe": sorted(severe - u)}


out = {"span": [2003, 2023], "severe": sorted(severe), "flood": sorted(flood), "bands": {}, "by_lead": {}, "cumulative": {}}
for name, leads in (("1-7", (1, 7)), ("8-12", (8, 12)), ("1-12", (1, 12))):
    per = {f"{r}_{s}": sorted(years(r, s, leads)) for r, s in WINDOWS}
    u = set().union(*(set(v) for v in per.values()))
    out["bands"][name] = score(u) | {"windows": per}
    print(f"leads {name:5}: {len(u):2d} activations, 1-in-{22 / max(len(u), 1):.1f}, severe {len(u & severe)} of {len(severe)}, no-flood {sorted(u - flood)}, missed severe {sorted(severe - u)}")
    for k, v in per.items():
        print(f"    {k:14} {v}")
for ld in range(1, 13):
    u = set().union(*(years(r, s, (ld, ld)) for r, s in WINDOWS))
    out["by_lead"][ld] = score(u)
    uc = set().union(*(years(r, s, (1, ld)) for r, s in WINDOWS))
    out["cumulative"][ld] = score(uc)
    print(f"lead {ld:2d} alone: {len(u):2d} acts, severe {len(u & severe)}, no-flood {len(u - flood)} | window 1-{ld:2d}: {len(uc):2d} acts, severe {len(uc & severe)}, no-flood {len(uc - flood)}")
# how the ensemble's high flows fade with lead, over the 21 years both tables cover
both = med[med.issued_time.dt.year.isin(SPAN)]
q = both.groupby(["station", "leadtime_days"]).discharge.quantile(.98).unstack("leadtime_days")
share = q.div(q[1], axis=0)
out["fade_years"] = [int(both.issued_time.dt.year.min()), int(both.issued_time.dt.year.max())]
out["fade_share_of_day1"] = {int(ld): round(float(share[ld].median()), 3) for ld in share.columns}
print("fade (median station share of day-1 98th percentile):", out["fade_share_of_day1"])
json.dump(out, open(SCRATCH / "lead_band_design.json", "w"), indent=1)
