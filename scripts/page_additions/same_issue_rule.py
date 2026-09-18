"""Price the alternative vote rule: points counted within one forecast issue (any valid day in
leads 1-7) instead of on the same valid day. On the model records the analogue is a rolling
7-day window. Reports activations, severe seasons caught, no-flood activations, and the first
issue meeting the rule against the second and the first gauge crossing."""
import pandas as pd
import somlib as L

WINDOWS = [("juba", "deyr"), ("shabelle", "deyr"), ("juba", "gu"), ("shabelle", "gu")]
from src.constants import TRIGGER_CONFIG as _TC
RULE = {k: (v["rp"], v["n_req"]) for k, v in _TC.items()}
CAL = {"deyr": "glofas_v5", "gu": "google_grrr"}
FC = {"deyr": "glofas_v4", "gu": "google_grrr"}
SPAN = set(range(1999, 2024))
name = lambda r, s: f"{s.title()} {r.title()}"

# ---- records: same-day count (design) vs any-day-within-7 count (variant)
def record_years(model, river, season, rp, n_req, window_days):
    thr = L.model_thresholds(model, river, season, rp)
    cols = []
    for st in L.TRIGGER_STATIONS[river]:
        s = L.season_series(model, st, season)
        if not len(s) or pd.isna(thr[st]):
            continue
        if window_days > 1:
            s = s[::-1].rolling(window_days, min_periods=1).max()[::-1]     # max over [d, d+window-1]
        cols.append((s >= thr[st]).rename(st))
    m = pd.concat(cols, axis=1).fillna(False)
    votes = m.sum(axis=1)
    return {y for y in votes[votes >= n_req].index.year if y in SPAN}

flood_all, severe_all = set(), set()
fl, sv = {}, {}
for r, s in WINDOWS:
    fl[(r, s)], sv[(r, s)], _ = L.benchmark(r, s)
    flood_all |= fl[(r, s)] & SPAN; severe_all |= sv[(r, s)] & SPAN

print("MODEL RECORDS: same-day count (design) vs any day within 7 (variant)")
env = {"design": set(), "variant": set()}
for r, s in WINDOWS:
    rp, n = RULE[(r, s)]
    a = record_years(CAL[s], r, s, rp, n, 1); b = record_years(CAL[s], r, s, rp, n, 7)
    env["design"] |= a; env["variant"] |= b
    print(f"  {name(r, s):15} design {len(a):2d} {sorted(a)}")
    print(f"  {'':15} variant {len(b):2d} {sorted(b)}   added: {sorted(b - a)}")
for k, v in env.items():
    print(f"  envelope {k:8}: {len(v)} activations in 25 yr (1-in-{26 / len(v):.1f}), severe {len(v & severe_all)} of {len(severe_all)}, no-flood {sorted(v - flood_all)}")

# ---- forecasts: first issue meeting the rule, same valid day vs any valid day within the issue
def first_issue_any_day(model, river, season, rp, n_req, leads=(1, 7)):
    rf = L.reforecast(model)
    rf = rf[rf.station.isin(L.TRIGGER_STATIONS[river]) & rf.leadtime_days.between(*leads) & rf.valid_time.dt.month.isin(L.SEASONS[season])]
    med = rf.groupby(["issued_time", "valid_time", "station"]).discharge.median().reset_index()
    thr = L.model_thresholds(model, river, season, rp)
    med["over"] = med.apply(lambda x: x.discharge >= thr.get(x.station, float("nan")), axis=1)
    votes = med[med.over].groupby(["issued_time"]).station.nunique()
    hit = votes[votes >= n_req]
    out = {}
    for iss in sorted(hit.index):
        y = iss.year if iss.month in L.SEASONS[season] or True else None
        # season-year: the valid days are in season months; take the year of the issue's first in-season valid day
        vy = med[(med.issued_time == iss)].valid_time.min().year
        out.setdefault(vy, iss.normalize())
    return out

print("\nFORECASTS: first issue meeting the rule, lead in days relative to the second gauge (and the first gauge) crossing 1-in-3")
print(f"  {'season':14}{'gap':>4}   {'same valid day':>16}   {'any day in issue':>18}   {'gain':>5}")
for s in ("deyr", "gu"):
    onset2 = {r: L.gauge_crossings(r, s, 3, span=SPAN) for r in ("juba", "shabelle")}
    onset1 = {r: L.gauge_crossings(r, s, 3, n_req=1, span=SPAN) for r in ("juba", "shabelle")}
    base, var = {}, {}
    for r in ("juba", "shabelle"):
        rp, n = RULE[(r, s)]
        for y, v in L.first_issue_dates(FC[s], r, s, rp, n, span=SPAN, leads=(1, 7)).items():
            base[y] = min(base.get(y, v[0]), v[0])
        for y, v in first_issue_any_day(FC[s], r, s, rp, n).items():
            var[y] = min(var.get(y, v), v)
    years = sorted({y for r in onset2 for y in onset2[r] if y in SPAN})
    lo, hi = (2016, 2023) if FC[s] == "google_grrr" else (2003, 2023)
    for y in years:
        if not (lo <= y <= hi):
            continue
        flooded = [r for r in onset2 if y in onset2[r]]
        r0 = min(flooded, key=lambda r_: onset2[r_][y]); d2 = onset2[r0][y]; d1 = onset1[r0][y]
        gap = (d2 - d1).days
        def fmt(d):
            if d is None: return "never"
            l2 = (d2 - d).days; l1 = (d1 - d).days
            return f"{l2:+d} d ({l1:+d} vs 1st)"
        b_, v_ = base.get(y), var.get(y)
        gain = (b_ - v_).days if (b_ is not None and v_ is not None) else ("new" if v_ is not None else "")
        sev = "*" if any(y in sv[(r, s)] for r in flooded) else " "
        print(f"  {s + ' ' + str(y) + sev:14}{gap:>4}   {fmt(b_):>16}   {fmt(v_):>18}   {str(gain):>5}")
