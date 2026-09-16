"""Why the action window stops at 7 days: the design rule on the GloFAS v4 reforecast,
scored by lead band. Same construction as somlib.first_issue_dates (ensemble median,
levels from the model's own record, design return period and vote count per window),
run once on leads 1-7 and once on leads 8-12, and lead by lead. Writes lead_band_design.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\pauni\Desktop\Work\OCHA\GitHub\ds-aa-som-floods")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCRATCH))
from src.datasources import glofas  # noqa: E402
import somlib as L  # noqa: E402

SPAN = set(range(2003, 2024))
WINDOWS = [("juba", "deyr"), ("shabelle", "deyr"), ("juba", "gu"), ("shabelle", "gu")]
RULE = {("juba", "deyr"): (4, 3), ("shabelle", "deyr"): (4, 2), ("juba", "gu"): (5, 3), ("shabelle", "gu"): (6, 2)}

a = L.reforecast("glofas_v4")                       # the processed table the page already uses for leads 1-7
b = glofas.load_reforecast_box(version="version_4_0", dir_suffix="_lead8_12").rename(columns={"valid_day": "valid_time"})
print("leads 1-7 table", sorted(a.leadtime_days.unique()), sorted(a.station.unique())); print("8-12 raw", sorted(b.leadtime_days.unique()), sorted(b.station.unique()))
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
# how the ensemble's high flows fade with lead, over the years both archives cover (the 8-12 archive stops in 2013)
both = med[med.issued_time.dt.year <= 2013]
q = both.groupby(["station", "leadtime_days"]).discharge.quantile(.98).unstack("leadtime_days")
share = q.div(q[1], axis=0)
out["fade_years"] = [2003, int(both.issued_time.dt.year.max())]
out["fade_share_of_day1"] = {int(ld): round(float(share[ld].median()), 3) for ld in share.columns}
out["bands_to_2013"] = {k: sorted(y for y in v["years"] if y <= 2013) for k, v in out["bands"].items()}
print("fade (median station share of day-1 98th percentile):", out["fade_share_of_day1"])
print("activations 2003-2013 by band:", out["bands_to_2013"])
json.dump(out, open(SCRATCH / "lead_band_design.json", "w"), indent=1)
