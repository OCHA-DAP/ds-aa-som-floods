"""Every score behind pages/google-model-review/: Google's November 2026 river model and the three
sources the trigger knows (Google's 2024 model, GloFAS v4, GloFAS v5) against the SWALIM gauges,
by season, river and station.

Families of scores:
- tracking(): day-to-day agreement with the gauge level (rank correlation on the same day, with the
  seasonal rise and fall taken out, and at the shift with the highest value), with ranges from
  resampling seasons; days between model and gauge at the peak and at the 1-in-3 crossing;
- station_events(rp): flood seasons at one station against its own gauge (the four counts and the
  scores built on them, with ranges), how the model orders the seasons (rank correlation of season
  peaks, AUC), and the same counts with every level fitted without the season being scored;
- river_rule(rp): the trigger's vote per river against the two-gauge flood, on the reanalysis, on
  the reanalysis restricted to the forecast seasons, on the forecast archives, and with levels
  fitted without the season being scored;
- by_lead(): the forecast archives day by day ahead, including how much of a coming rise the
  forecast already shows;
- discharge(): size of the flow against SWALIM's discharge series where it exists (four stations);
- gauge_notes(): how often each gauge stops at one top reading.

Reuses the data plumbing of build_google_model_2026_page.py (the new model wired into somlib).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import build_google_model_2026_page as G
from src.utils import weibull_level, weibull_threshold

L = G.L
NEW = G.NEW
MODELS = [NEW, "google_grrr", "glofas_v4", "glofas_v5"]
FC_MODELS = [NEW, "google_grrr", "glofas_v4"]
STATIONS = G.STATIONS
RIVER_OF = G.RIVER_OF
RIVERS = G.RIVERS
SEASON_LIST = ("deyr", "gu")
SEASONS = G.SEASONS
Y0, Y1 = 1999, 2023                 # the years every reanalysis shares
FC0, FC1 = 2016, 2023               # the years every forecast archive shares
MIN_DAYS = G.MIN_DAYS
LAGS = range(-10, 31)
N_BOOT = 400
RNG = np.random.default_rng(20261117)
OBS_Q_DIR = os.environ.get("SOM_OBS_Q_DIR")
OBS_Q_FILES = {"luuq": "luuq_real_discharge.json", "bardheere": "bardheere_real_discharge.json",
               "belet_weyne": "beletweyne_real_discharge.json", "bulo_burti": "buloburte_real_discharge.json"}


# ------------------------------------------------------------------------------- series
_c = {}


def gauge_all(st):
    """Gauge level, every day with a reading above zero (a zero is a gap in the record)."""
    k = ("ga", st)
    if k not in _c:
        lv = L.levels()
        s = lv[lv.station == st].set_index("date").level_m.dropna().sort_index()
        s.index = pd.DatetimeIndex(s.index).normalize()
        s = s[~s.index.duplicated()]
        _c[k] = s[s > 0]
    return _c[k]


def gauge_daily(st, season):
    """Gauge level on the season's days, 1999 to 2025, seasons with at least MIN_DAYS readings."""
    k = ("g", st, season)
    if k not in _c:
        s = gauge_all(st)
        s = s[s.index.month.isin(SEASONS[season]) & (s.index.year >= Y0) & (s.index.year <= 2025)]
        n = s.groupby(s.index.year).size()
        _c[k] = s[s.index.year.isin(n[n >= MIN_DAYS].index)]
    return _c[k]


def model_all(m, st):
    k = ("ma", m, st)
    if k not in _c:
        d = L.daily(m)
        s = d[d.station == st].set_index("date").discharge.dropna().sort_index()
        s.index = pd.DatetimeIndex(s.index).normalize()
        _c[k] = s[~s.index.duplicated()]
    return _c[k]


def model_daily(m, st, season, y1=Y1):
    s = model_all(m, st)
    return s[s.index.month.isin(SEASONS[season]) & (s.index.year >= Y0) & (s.index.year <= y1)]


def gauge_level(st, season, rp):
    """The framework's gauge level: Weibull on the 2000 to 2023 season peaks, to the centimetre."""
    return float(L.gauge_levels(RIVER_OF[st], season, rp)[st])


def model_level(m, st, season, rp):
    return float(L.model_threshold(m, st, season, rp))


def season_max(s):
    return s.groupby(s.index.year).max().dropna()


def first_over(s, level):
    """Year -> first day at or over the level."""
    x = s[s >= level]
    if x.empty:
        return pd.Series(dtype="datetime64[ns]")
    return x.groupby(x.index.year).apply(lambda v: v.index[0])


def peak_day(s):
    if s.empty:
        return pd.Series(dtype="datetime64[ns]")
    return s.groupby(s.index.year).apply(lambda v: v.idxmax())


# ------------------------------------------------------------------------------- scores
def contingency(sim, obs, universe):
    """The 2 x 2 table and every score read from it. sim, obs, universe are sets of years."""
    sim, obs, universe = set(sim) & set(universe), set(obs) & set(universe), set(universe)
    a = len(sim & obs)                 # flood season, threshold reached
    b = len(sim - obs)                 # threshold reached, no flood season
    c = len(obs - sim)                 # flood season, threshold not reached
    d = len(universe - sim - obs)      # neither
    n = a + b + c + d
    div = lambda x, y: (x / y) if y else np.nan
    pod, far, pofd = div(a, a + c), div(b, a + b), div(b, b + d)
    exp = div((a + b) * (a + c) + (c + d) * (b + d), n) if n else np.nan
    hss = div((a + d) - exp, n - exp) if n and not np.isnan(exp) else np.nan
    return dict(hits=a, false_alarms=b, misses=c, correct_negatives=d, n=n, n_floods=a + c, n_reached=a + b,
                POD=pod, FAR=far, POFD=pofd, CSI=div(a, a + b + c), F1=div(2 * a, 2 * a + b + c), freq_bias=div(a + b, a + c),
                accuracy=div(a + d, n), HSS=hss, PSS=(pod - pofd) if not (np.isnan(pod) or np.isnan(pofd)) else np.nan)


def boot_events(sim, obs, universe, n_boot=1000):
    """5th and 95th percentiles of POD, FAR and F1 when the seasons are drawn again with replacement."""
    yrs = sorted(universe)
    if len(yrs) < 5:
        return {}
    s = np.array([y in sim for y in yrs])
    o = np.array([y in obs for y in yrs])
    idx = np.random.default_rng(20261117).integers(0, len(yrs), (n_boot, len(yrs)))   # same seasons drawn for every row
    a, b, c = (s & o)[idx].sum(1), (s & ~o)[idx].sum(1), (~s & o)[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pod, far, f1 = a / (a + c), b / (a + b), 2 * a / (2 * a + b + c)
    q = lambda v: (float(np.nanpercentile(v, 5)), float(np.nanpercentile(v, 95))) if np.isfinite(v).any() else (np.nan, np.nan)
    (pl, ph), (fl, fh), (gl, gh) = q(pod), q(far), q(f1)
    return dict(POD_lo=pl, POD_hi=ph, FAR_lo=fl, FAR_hi=fh, F1_lo=gl, F1_hi=gh)


def auc(score, obs_years):
    """Chance that a flood season carries a higher model peak than a season without one."""
    pos = [v for y, v in score.items() if y in obs_years]
    neg = [v for y, v in score.items() if y not in obs_years]
    if not pos or not neg:
        return np.nan
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def spearman(x, y, min_n=5):
    j = pd.concat([x, y], axis=1, join="inner").dropna()
    return float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")) if len(j) >= min_n else np.nan


def lag_rho(model, gauge_any, lag, min_n=60):
    """Rank correlation with the model on day t set against the gauge on day t + lag. gauge_any is
    the gauge on every day of its record, so the days used are the model's season days at every lag."""
    g = gauge_any.copy()
    g.index = g.index - pd.Timedelta(days=lag)
    j = pd.concat([model, g], axis=1, join="inner").dropna()
    return (float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")), len(j)) if len(j) >= min_n else (np.nan, len(j))


def best_lag(model, gauge_any):
    """(highest rank correlation, its lag, pairs used); (nan, None, 0) when no lag has enough pairs."""
    best = (np.nan, None, 0)
    for k in LAGS:
        r, n = lag_rho(model, gauge_any, k)
        if not np.isnan(r) and (np.isnan(best[0]) or r > best[0]):
            best = (r, k, n)
    return best


def deseason(s):
    """The series with its usual value for the calendar day taken out (the middle value over the years)."""
    key = s.index.strftime("%m-%d")
    return s - s.groupby(key).transform("median")


# ----------------------------------------------------------------------------- tracking
def tracking():
    """Per season, station and model: daily rank correlation with the gauge on the same day (with a
    range from drawing the seasons again), the same with the usual seasonal rise and fall taken out,
    the highest value over shifts of the model against the gauge, the same on flood seasons; days
    between the model's and the gauge's peak, and between their first days at the 1-in-3."""
    rows = []
    for season in SEASON_LIST:
        for st in STATIONS:
            g = gauge_daily(st, season)
            g = g[g.index.year <= Y1]
            ga = gauge_all(st)
            years = sorted(set(g.index.year))
            g3 = gauge_level(st, season, 3)
            gmax, gpk = season_max(g), peak_day(g)
            gx = first_over(g, g3) if not np.isnan(g3) else pd.Series(dtype="datetime64[ns]")
            flood = set(gmax[gmax >= g3].index) if not np.isnan(g3) else set()
            draws = RNG.integers(0, len(years), (N_BOOT, len(years)))
            pairs, base = {}, {}
            for m in MODELS:
                x = model_daily(m, st, season)
                j = pd.concat([x.rename("x"), g.rename("g")], axis=1, join="inner").dropna()
                pairs[m] = {y: v[["x", "g"]].to_numpy() for y, v in j.groupby(j.index.year)}
            boot = {m: np.full(N_BOOT, np.nan) for m in MODELS}
            for b in range(N_BOOT):
                ys = [years[i] for i in draws[b]]
                for m in MODELS:
                    arr = [pairs[m][y] for y in ys if y in pairs[m]]
                    if arr:
                        a_ = np.vstack(arr)
                        boot[m][b] = pd.Series(a_[:, 0]).corr(pd.Series(a_[:, 1]), method="spearman")
            for m in MODELS:
                x = model_daily(m, st, season)
                x = x[x.index.year.isin(years)]
                rho0, n0 = lag_rho(x, g, 0)
                rho_anom = spearman(deseason(x), deseason(g), min_n=60)
                bl = best_lag(x, ga)
                m3 = model_level(m, st, season, 3)
                xmax = season_max(x)
                ev = flood | set(xmax[xmax >= m3].index)
                bl_ev = best_lag(x[x.index.year.isin(ev)], ga) if ev else (np.nan, None, 0)
                xpk, xx = peak_day(x), first_over(x, m3)
                dpk = [(gpk[y] - xpk[y]).days for y in flood if y in xpk.index]
                dx = [(gx[y] - xx[y]).days for y in flood if y in xx.index and y in gx.index]
                d_new = boot[NEW] - boot[m]
                rows.append(dict(season=season, river=RIVER_OF[st], station=st, model=m, n_seasons=len(years), n_days=n0,
                                 rho_same_day=rho0, rho_lo=float(np.nanpercentile(boot[m], 5)), rho_hi=float(np.nanpercentile(boot[m], 95)),
                                 new_minus_this_lo=float(np.nanpercentile(d_new, 5)) if m != NEW else np.nan,
                                 new_minus_this_hi=float(np.nanpercentile(d_new, 95)) if m != NEW else np.nan,
                                 rho_cycle_removed=rho_anom,
                                 rho_best_shift=bl[0], best_shift_days=bl[1], n_days_best_shift=bl[2],
                                 rho_flood_seasons=bl_ev[0], shift_flood_seasons=bl_ev[1], n_flood_seasons=len(ev),
                                 peak_days_ahead_median=float(np.median(dpk)) if dpk else np.nan, n_peaks=len(dpk),
                                 crossing_days_ahead_median=float(np.median(dx)) if dx else np.nan, n_crossings=len(dx)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- station events
def _gauge_fit_max(st, season):
    """The season peaks the framework fits a gauge's levels on: 2000 to 2023, every season with a reading."""
    lv = L.levels()
    s = lv[lv.station == st].set_index("date").level_m.dropna().sort_index()
    s = s[s.index.month.isin(SEASONS[season]) & (s.index.year >= 2000) & (s.index.year <= Y1)]
    return s.groupby(s.index.year).max().dropna()


def _loo_gauge_level(fit, y, rp):
    v = weibull_level(fit[fit.index != y].values, rp)
    return round(v, 2) if not np.isnan(v) else np.nan


def station_events(rp):
    """Per season, station and model: seasons in which the model's peak reaches its own 1-in-rp
    level against seasons in which the gauge's peak reaches its own 1-in-rp level. The columns
    ending in _held_out repeat the counts with both levels fitted without the season being scored."""
    rows = []
    for season in SEASON_LIST:
        for st in STATIONS:
            g = gauge_daily(st, season)
            gmax = season_max(g[g.index.year <= Y1])
            glev = gauge_level(st, season, rp)
            if np.isnan(glev) or len(gmax) < 8:
                continue
            obs = set(gmax[gmax >= glev].index)
            fit = _gauge_fit_max(st, season)
            obs_h = {y for y in gmax.index if gmax[y] >= _loo_gauge_level(fit, y, rp)}
            for m in MODELS:
                xmax = season_max(model_daily(m, st, season))
                uni = set(gmax.index) & set(xmax.index)
                mlev = model_level(m, st, season, rp)
                sim = set(xmax[xmax >= mlev].index)
                sim_h = {y for y in uni if xmax[y] >= weibull_threshold(xmax[xmax.index != y].values, rp)}
                c = contingency(sim, obs, uni)
                ch = contingency(sim_h, obs_h, uni)
                rows.append(dict(season=season, river=RIVER_OF[st], station=st, model=m, rp=rp, **c, **boot_events(sim, obs, uni),
                                 AUC=auc({y: xmax[y] for y in uni}, obs),
                                 rho_season_peaks=spearman(xmax[xmax.index.isin(uni)], gmax[gmax.index.isin(uni)]),
                                 hit_years=sorted(sim & obs & uni), false_years=sorted((sim - obs) & uni), missed_years=sorted((obs - sim) & uni),
                                 **{f"{k}_held_out": ch[k] for k in ("hits", "false_alarms", "misses", "n_floods", "POD", "FAR", "F1")}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- river rule
def fc_years(m, season):
    if m not in G.FC_SPAN:
        return set()
    return set(range(G.FC_SPAN[m][0], G.FC_END.get((m, season), G.FC_SPAN[m][1]) + 1))


def _river_held_out(m, river, season, rp, n_req, years):
    """The rule and the benchmark for each year with every level fitted without that year."""
    mm = L.daily_matrix(m, river, season)
    mm = mm[(mm.index.year >= Y0) & (mm.index.year <= Y1)]
    gm = L.gauge_matrix(river, season)
    sim, obs = set(), set()
    mx = {st: mm[st].groupby(mm.index.year).max().dropna() for st in mm.columns}
    fits = {st: _gauge_fit_max(st, season) for st in gm.columns}
    for y in years:
        day = mm[mm.index.year == y]
        thr = pd.Series({st: weibull_threshold(mx[st][mx[st].index != y].values, rp) for st in mm.columns})
        if len(day) and ((day >= thr).sum(axis=1) >= n_req).any():
            sim.add(y)
        gy = gm[gm.index.year == y]
        lev = pd.Series({st: _loo_gauge_level(fits[st], y, rp) for st in gm.columns})
        cols = [c for c in gm.columns if not np.isnan(lev[c])]
        if len(gy) and cols and ((gy[cols] >= lev[cols]).cummax().sum(axis=1) >= L.BENCH_GAUGES).any():
            obs.add(y)
    return sim, obs


def river_rule(rp):
    """Per season, river, model and record: the trigger's vote (3 of 4 Juba stations, 2 of 3 Shabelle
    stations over their own 1-in-rp on the same day) against the two-gauge 1-in-rp flood."""
    rows, dates = [], {}
    for season in SEASON_LIST:
        for river in RIVERS:
            n_req = G.TRIGGER_CONFIG[(river, season)]["n_req"]
            assess = L.assessable_years(river, season, span=G.YEARS)
            gx = L.gauge_crossings(river, season, rp, span=G.YEARS)
            obs = set(gx)
            for m in MODELS:
                ra = L.first_crossing_dates(m, river, season, rp, n_req, span=range(Y0, G.RA_SPAN[m][1] + 1))
                all_ra = set(range(Y0, Y1 + 1))
                recs = [("reanalysis", ra, all_ra, obs)]
                if m in G.FC_SPAN:
                    fy = fc_years(m, season)
                    common = fy & set(range(FC0, FC1 + 1))
                    fc = G.first_issue_in_season(m, river, season, rp, n_req, span=fy)
                    recs.append(("reanalysis, forecast seasons", ra, common, obs))
                    recs.append(("forecasts", fc, common, obs))
                    if max(fy) > FC1:
                        recs.append((f"forecasts to {max(fy)}", fc, fy, obs))
                sim_h, obs_h = _river_held_out(m, river, season, rp, n_req, sorted(all_ra & assess))
                recs.append(("reanalysis, levels held out", {y: ra.get(y) for y in sim_h}, all_ra, obs_h))
                for basis, d, yrs, ob in recs:
                    uni = yrs & assess
                    c = contingency(set(d), ob, uni)
                    lead = [(gx[y] - d[y]).days for y in set(d) & ob & uni if d.get(y) is not None and y in gx]
                    rows.append(dict(season=season, river=river, model=m, record=basis, rp=rp, **c, **boot_events(set(d), ob, uni),
                                     lead_days_median=float(np.median(lead)) if lead else np.nan,
                                     hit_years=sorted(set(d) & ob & uni), false_years=sorted((set(d) - ob) & uni),
                                     missed_years=sorted((ob - set(d)) & uni)))
                    if "held out" not in basis:
                        dates[(season, river, m, basis, rp)] = {int(y): v for y, v in d.items()}
            dates[(season, river, "gauge", "gauge", rp)] = {int(y): v for y, v in gx.items()}
    return pd.DataFrame(rows), dates


# ------------------------------------------------------------------------------ by lead
def lead_series(m, st, season, lead):
    """The forecast for each day made `lead` days ahead, where 1 is the day the forecast is issued
    (GloFAS: the middle member). Google's 2024 archive stamps its forecast days one day later than
    its reanalysis days, so it is moved back a day to line up with its own reanalysis."""
    k = ("rf", m)
    if k not in _c:
        rf = L.reforecast(m)
        rf = rf[rf.station.isin(STATIONS)]
        s = rf.groupby(["station", "leadtime_days", "valid_time"]).discharge.median()
        _c[k] = s
    try:
        s = _c[k].loc[(st, lead)]
    except KeyError:
        return pd.Series(dtype=float)
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).normalize() - pd.Timedelta(days=1 if m == "google_grrr" else 0)
    return s[s.index.month.isin(SEASONS[season])]


def by_lead():
    """Per season, station, model and days ahead: forecast peak over reanalysis peak on the same
    days, rank correlation with the gauge and with the model's own reanalysis, the share of a coming
    rise that the forecast already shows, and the flood-season counts at 1-in-3."""
    rows = []
    for m in FC_MODELS:
        leads = range(1, 11) if m == NEW else range(1, 8)
        for season in SEASON_LIST:
            yrs = fc_years(m, season) & set(range(FC0, FC1 + 1))
            for st in STATIONS:
                g = gauge_daily(st, season)
                g = g[g.index.year.isin(yrs)]
                gmax = season_max(g)
                glev, mlev = gauge_level(st, season, 3), model_level(m, st, season, 3)
                if np.isnan(glev) or gmax.empty:
                    continue
                obs = set(gmax[gmax >= glev].index)
                ra_all = model_all(m, st)
                ra = model_daily(m, st, season, y1=G.RA_SPAN[m][1])
                for lead in leads:
                    f = lead_series(m, st, season, lead)
                    f = f[f.index.year.isin(yrs)]
                    if f.empty:
                        continue
                    j = pd.concat([f.rename("f"), ra.rename("r")], axis=1, join="inner").dropna()
                    pk = (j.groupby(j.index.year).f.max() / j.groupby(j.index.year).r.max()) if len(j) else pd.Series(dtype=float)
                    # the rise over the `lead` days up to each day, in the reanalysis and as the forecast had it
                    base = ra_all.copy()
                    base.index = base.index + pd.Timedelta(days=lead)
                    r3 = pd.concat([f.rename("f"), ra_all.rename("r"), base.rename("b")], axis=1, join="inner").dropna()
                    r3["actual"], r3["fore"] = r3.r - r3.b, r3.f - r3.b
                    big = r3[(r3.actual > 0) & (r3.actual >= r3.actual.quantile(0.9))]
                    fmax = season_max(f)
                    uni = set(gmax.index) & set(fmax.index)
                    c = contingency(set(fmax[fmax >= mlev].index), obs, uni)
                    rows.append(dict(season=season, river=RIVER_OF[st], station=st, model=m, lead=lead,
                                     peak_ratio_median=float(pk.median()) if len(pk) else np.nan,
                                     rho_vs_gauge=lag_rho(f, g, 0)[0], rho_vs_reanalysis=spearman(j.f, j.r) if len(j) else np.nan,
                                     shift_vs_reanalysis=best_lag(f, ra_all)[1] if lead == 1 else None,
                                     rise_share_median=float((big.fore / big.actual).median()) if len(big) >= 10 else np.nan, n_rise_days=len(big),
                                     n_seasons=len(uni), **{k: c[k] for k in ("hits", "false_alarms", "misses", "correct_negatives")}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- discharge
def obs_discharge(st):
    if not OBS_Q_DIR or st not in OBS_Q_FILES or not (Path(OBS_Q_DIR) / OBS_Q_FILES[st]).exists():
        return None
    d = json.loads((Path(OBS_Q_DIR) / OBS_Q_FILES[st]).read_text())
    s = pd.Series({pd.Timestamp(k).normalize(): v for k, v in d.items() if v is not None}, dtype=float).dropna().sort_index()
    s = s[~s.index.duplicated()]
    return s[(s.index.year >= Y0) & (s.index.year <= Y1)]


def discharge():
    """Per station, model and period (each season, and the whole year): model flow against SWALIM's
    discharge series on the same days."""
    rows = []
    for st in OBS_Q_FILES:
        o_all = obs_discharge(st)
        if o_all is None or o_all.empty:
            continue
        for m in MODELS:
            x_all = model_all(m, st)
            for period, months in (("deyr", SEASONS["deyr"]), ("gu", SEASONS["gu"]), ("year", range(1, 13))):
                j = pd.concat([x_all.rename("s"), o_all.rename("o")], axis=1, join="inner").dropna()
                j = j[j.index.month.isin(list(months)) & (j.o > 0)]
                if len(j) < 200:
                    continue
                r = float(j.s.corr(j.o))
                alpha, beta = float(j.s.std() / j.o.std()), float(j.s.mean() / j.o.mean())
                kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
                nse = 1 - float(((j.s - j.o) ** 2).sum() / ((j.o - j.o.mean()) ** 2).sum())
                lo = j[(j.s > 0)]
                lnse = 1 - float(((np.log(lo.s) - np.log(lo.o)) ** 2).sum() / ((np.log(lo.o) - np.log(lo.o).mean()) ** 2).sum())
                ys = j.groupby(j.index.year).agg(s=("s", "max"), o=("o", "max"), n=("s", "size"))
                ys = ys[ys.n >= (MIN_DAYS if period != "year" else 200)]
                rows.append(dict(station=st, river=RIVER_OF[st], model=m, period=period, n_days=len(j), n_years=len(ys),
                                 first_day=j.index.min().strftime("%Y-%m-%d"), last_day=j.index.max().strftime("%Y-%m-%d"),
                                 ratio_median=float(j.s.median() / j.o.median()), ratio_mean=beta,
                                 ratio_p90=float(j.s.quantile(0.9) / j.o.quantile(0.9)),
                                 ratio_peak_median=float((ys.s / ys.o).median()) if len(ys) else np.nan,
                                 pearson_r=r, spearman_rho=float(j.s.corr(j.o, method="spearman")),
                                 variability_ratio=alpha, KGE=float(kge), NSE=nse, log_NSE=lnse))
    return pd.DataFrame(rows)


def discharge_notes():
    """Facts about the SWALIM discharge series themselves."""
    out = {}
    for st in OBS_Q_FILES:
        o = obs_discharge(st)
        if o is None or o.empty:
            continue
        j = pd.concat([o.rename("q"), gauge_all(st).rename("h")], axis=1, join="inner").dropna()
        out[st] = dict(days_at_top=int((o >= o.max() * 0.999).sum()), top=float(o.max()), median=float(o.median()),
                       rank_corr_with_level=float(j.q.corr(j.h, method="spearman")) if len(j) > 100 else np.nan)
    a, b = obs_discharge("luuq"), obs_discharge("bardheere")
    if a is not None and b is not None:
        j = pd.concat([a.rename("luuq"), b.rename("bardheere")], axis=1, join="inner").dropna()
        out["bardheere_over_luuq"] = dict(swalim=float(j.bardheere.median() / j.luuq.median()),
                                          **{m: float(model_all(m, "bardheere").loc[str(Y0):str(Y1)].median() / model_all(m, "luuq").loc[str(Y0):str(Y1)].median()) for m in MODELS})
    return out


# -------------------------------------------------------------------------- gauge notes
def gauge_notes():
    """Per season and station: seasons on record, seasons at or over the 1-in-3 level, the most
    common season peak and how many seasons share it, and seasons that sit at their peak for five
    days or more. A level meant for one season in three that is reached in many more is a sign that
    the gauge stops at a top reading."""
    rows = []
    for season in SEASON_LIST:
        for st in STATIONS:
            g = gauge_daily(st, season)
            g = g[g.index.year <= Y1]
            mx = season_max(g)
            l3 = gauge_level(st, season, 3)
            top = mx.round(2).value_counts()
            flat = g.groupby(g.index.year).apply(lambda v: int((v >= v.max() - 0.005).sum()))
            rows.append(dict(season=season, river=RIVER_OF[st], station=st, n_seasons=len(mx), level_1in3=l3,
                             n_at_or_over=int((mx >= l3).sum()) if not np.isnan(l3) else None,
                             common_peak=float(top.index[0]), n_common_peak=int(top.iloc[0]),
                             n_flat_top=int((flat >= 5).sum()), record_max=float(mx.max()),
                             level_differs_when_short_seasons_dropped=bool(
                                 not np.isnan(l3) and abs(round(weibull_level(mx[mx.index >= 2000].values, 3), 2) - l3) > 0.005)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- worst years
def worst_years(season, n=5):
    """The gauges' n largest seasons per river (second-highest gauge as a share of its 1-in-3),
    merged across the two rivers, largest first."""
    best = {}
    for river in RIVERS:
        P, Rk = G.rank_table(season, river)
        for y in G.top_seasons(Rk)[:n + 2]:
            if Rk.loc[y, "gauge"] <= n:
                best[int(y)] = max(best.get(int(y), 0), float(P.loc[y, "gauge"]))
    return [y for y, _ in sorted(best.items(), key=lambda t: -t[1])]


def compute():
    out = dict(tracking=tracking(), station_3=station_events(3), station_5=station_events(5), by_lead=by_lead(),
               discharge=discharge(), discharge_notes=discharge_notes(), gauges=gauge_notes())
    for rp in (3, 5):
        df, dates = river_rule(rp)
        out[f"river_{rp}"] = df
        out[f"river_dates_{rp}"] = dates
    out["worst"] = {s: worst_years(s) for s in SEASON_LIST}
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    M = compute()
    nm = {NEW: "new", "google_grrr": "cur", "glofas_v4": "v4", "glofas_v5": "v5"}
    T = M["tracking"].assign(model=lambda d: d.model.map(nm))
    for col in ("rho_same_day", "rho_lo", "rho_hi", "new_minus_this_lo", "rho_cycle_removed", "rho_best_shift", "best_shift_days"):
        print("==", col); print(T.pivot_table(index=["season", "river", "station"], columns="model", values=col, sort=False).reindex(columns=["new", "cur", "v4", "v5"]).round(2))
    E = M["station_3"].assign(model=lambda d: d.model.map(nm))
    print("== station 1-in-3, in sample vs levels held out (pooled over stations)")
    print(E.groupby(["season", "model"])[["hits", "n_floods", "false_alarms", "n_reached", "hits_held_out", "n_floods_held_out", "false_alarms_held_out"]].sum())
    Rv = M["river_3"].assign(model=lambda d: d.model.map(nm))
    print("== river rule 1-in-3"); print(Rv[["season", "river", "model", "record", "hits", "false_alarms", "misses", "n", "POD", "POD_lo", "POD_hi", "FAR", "FAR_lo", "FAR_hi", "lead_days_median"]].round(2).to_string(index=False))
    Bl = M["by_lead"].assign(model=lambda d: d.model.map(nm))
    print("== shift of the lead-1 forecast against the reanalysis (0 = lined up)"); print(Bl[Bl.lead == 1].groupby("model").shift_vs_reanalysis.agg(["min", "max"]))
    g = Bl.groupby(["model", "season", "river", "lead"]).agg(peak=("peak_ratio_median", "median"), rise=("rise_share_median", "median"), rho=("rho_vs_gauge", "mean")).reset_index()
    print("== by lead"); print(g[g.lead.isin([1, 3, 5, 7, 10])].round(2).to_string(index=False))
    print("== gauges"); print(M["gauges"].to_string(index=False))
    print("== discharge notes"); print(json.dumps(M["discharge_notes"], indent=1, default=float))
