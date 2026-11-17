"""Every score behind pages/google-model-review/: Google's November 2026 river model and the three
sources the trigger knows (Google's 2024 model, GloFAS v4, GloFAS v5) against the SWALIM gauges,
by season, river and station.

Four families:
- tracking(): day-to-day agreement with the gauge level (rank correlation, lag, peak and crossing timing);
- station_events(rp): flood seasons at one station against its own gauge (contingency table and the
  scores built on it), plus how the model orders the seasons (rank correlation of season peaks, AUC);
- river_rule(rp): the trigger's vote per river against the two-gauge flood, on the reanalysis and on
  the forecast archives;
- by_lead(): the forecast archives lead by lead;
- discharge(): magnitude against SWALIM's measured discharge where it exists (four stations).

Reuses the data plumbing of build_google_model_2026_page.py (the new model wired into somlib).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import build_google_model_2026_page as G

L = G.L
NEW = G.NEW
MODELS = [NEW, "google_grrr", "glofas_v4", "glofas_v5"]
STATIONS = G.STATIONS
RIVER_OF = G.RIVER_OF
RIVERS = G.RIVERS
SEASON_LIST = ("deyr", "gu")
SEASONS = G.SEASONS
Y0, Y1 = 1999, 2023                 # the years every reanalysis shares
FC0, FC1 = 2016, 2023               # the years every forecast archive shares
MIN_DAYS = G.MIN_DAYS
LAGS = range(-10, 31)
OBS_Q_DIR = os.environ.get("SOM_OBS_Q_DIR")
OBS_Q_FILES = {"luuq": "luuq_real_discharge.json", "bardheere": "bardheere_real_discharge.json",
               "belet_weyne": "beletweyne_real_discharge.json", "bulo_burti": "buloburte_real_discharge.json"}


# ------------------------------------------------------------------------------- series
_c = {}


def gauge_daily(st, season):
    """Gauge level on the season's days, 1999 to 2025, seasons with at least MIN_DAYS readings."""
    k = ("g", st, season)
    if k not in _c:
        lv = L.levels()
        s = lv[lv.station == st].set_index("date").level_m.dropna().sort_index()
        s = s[s > 0]                      # a reading of zero is a gap in the record, not a level
        s = s[s.index.month.isin(SEASONS[season]) & (s.index.year >= Y0) & (s.index.year <= 2025)]
        n = s.groupby(s.index.year).size()
        _c[k] = s[s.index.year.isin(n[n >= MIN_DAYS].index)]
    return _c[k]


def model_daily(m, st, season, y1=Y1):
    k = ("m", m, st, season, y1)
    if k not in _c:
        _c[k] = L.season_series(m, st, season, span=(Y0, y1))
    return _c[k]


def gauge_level(st, season, rp):
    return float(L.gauge_levels(RIVER_OF[st], season, rp)[st])


def model_level(m, st, season, rp):
    return float(L.model_threshold(m, st, season, rp))


def season_max(s):
    return s.groupby(s.index.year).max()


def first_over(s, level):
    """Year -> first day at or over the level."""
    x = s[s >= level]
    return x.groupby(x.index.year).apply(lambda v: v.index[0])


def peak_day(s):
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
    csi = div(a, a + b + c)
    f1 = div(2 * a, 2 * a + b + c)
    exp = div((a + b) * (a + c) + (c + d) * (b + d), n) if n else np.nan
    hss = div((a + d) - exp, n - exp) if n and not np.isnan(exp) else np.nan
    return dict(hits=a, false_alarms=b, misses=c, correct_negatives=d, n=n, n_floods=a + c, n_reached=a + b,
                POD=pod, FAR=far, POFD=pofd, CSI=csi, F1=f1, freq_bias=div(a + b, a + c),
                accuracy=div(a + d, n), HSS=hss, PSS=(pod - pofd) if not (np.isnan(pod) or np.isnan(pofd)) else np.nan)


def auc(score, obs_years):
    """Chance that a flood season carries a higher model peak than a season without one."""
    pos = [v for y, v in score.items() if y in obs_years]
    neg = [v for y, v in score.items() if y not in obs_years]
    if not pos or not neg:
        return np.nan
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def spearman(x, y):
    j = pd.concat([x, y], axis=1, join="inner").dropna()
    return float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")) if len(j) >= 5 else np.nan


def lag_rho(model, gauge, lag):
    """Rank correlation with the model on day t set against the gauge on day t + lag."""
    g = gauge.copy()
    g.index = g.index - pd.Timedelta(days=lag)
    j = pd.concat([model, g], axis=1, join="inner").dropna()
    return (float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")), len(j)) if len(j) >= 60 else (np.nan, len(j))


# ----------------------------------------------------------------------------- tracking
def tracking():
    """Per season, station and model: daily rank correlation with the gauge (same day, and at the
    lag with the highest value), on every season with a gauge record and on the 1-in-3 seasons;
    days between the model's peak and the gauge's peak; days between the model and the gauge
    first reaching their own 1-in-3."""
    rows = []
    for season in SEASON_LIST:
        for st in STATIONS:
            g = gauge_daily(st, season)
            g = g[g.index.year <= Y1]
            g3 = gauge_level(st, season, 3)
            gmax, gpk, gx = season_max(g), peak_day(g), first_over(g, g3) if not np.isnan(g3) else pd.Series(dtype="datetime64[ns]")
            flood = set(gmax[gmax >= g3].index) if not np.isnan(g3) else set()
            for m in MODELS:
                x = model_daily(m, st, season)
                x = x[x.index.year.isin(set(g.index.year))]
                rho0, n0 = lag_rho(x, g, 0)
                best = max(((lag_rho(x, g, k)[0], k) for k in LAGS), key=lambda t: (-np.inf if np.isnan(t[0]) else t[0]))
                m3 = model_level(m, st, season, 3)
                xmax = season_max(x)
                ev = flood | set(xmax[xmax >= m3].index)
                xe, ge = x[x.index.year.isin(ev)], g[g.index.year.isin(ev)]
                best_ev = max(((lag_rho(xe, ge, k)[0], k) for k in LAGS), key=lambda t: (-np.inf if np.isnan(t[0]) else t[0])) if ev else (np.nan, None)
                xpk, xx = peak_day(x), first_over(x, m3)
                dpk = [(xpk[y] - gpk[y]).days for y in flood if y in xpk.index]
                dx = [(gx[y] - xx[y]).days for y in flood if y in xx.index and y in gx.index]
                rows.append(dict(season=season, river=RIVER_OF[st], station=st, model=m, n_seasons=int(g.index.year.nunique()), n_days=n0,
                                 rho_same_day=rho0, rho_best_lag=best[0], best_lag_days=best[1],
                                 rho_flood_seasons=best_ev[0], lag_flood_seasons=best_ev[1], n_flood_seasons=len(ev),
                                 peak_days_late_median=float(np.median(dpk)) if dpk else np.nan,
                                 peak_days_abs_median=float(np.median(np.abs(dpk))) if dpk else np.nan, n_peaks=len(dpk),
                                 crossing_days_early_median=float(np.median(dx)) if dx else np.nan, n_crossings=len(dx)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- station events
def station_events(rp):
    """Per season, station and model: seasons in which the model's peak reaches its own 1-in-rp
    level against seasons in which the gauge's peak reaches its own 1-in-rp level."""
    rows = []
    for season in SEASON_LIST:
        for st in STATIONS:
            g = gauge_daily(st, season)
            gmax = season_max(g[g.index.year <= Y1])
            glev = gauge_level(st, season, rp)
            if np.isnan(glev) or len(gmax) < 8:
                continue
            obs = set(gmax[gmax >= glev].index)
            for m in MODELS:
                xmax = season_max(model_daily(m, st, season))
                uni = set(gmax.index) & set(xmax.index)
                sim = set(xmax[xmax >= model_level(m, st, season, rp)].index)
                c = contingency(sim, obs, uni)
                rows.append(dict(season=season, river=RIVER_OF[st], station=st, model=m, rp=rp, **c,
                                 AUC=auc({y: xmax[y] for y in uni}, obs),
                                 rho_season_peaks=spearman(xmax[xmax.index.isin(uni)], gmax[gmax.index.isin(uni)]),
                                 hit_years=sorted(sim & obs & uni), false_years=sorted((sim - obs) & uni), missed_years=sorted((obs - sim) & uni)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- river rule
def fc_years(m, season):
    if m not in G.FC_SPAN:
        return set()
    return set(range(G.FC_SPAN[m][0], G.FC_END.get((m, season), G.FC_SPAN[m][1]) + 1))


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
                recs = [("reanalysis", ra, set(range(Y0, Y1 + 1)))]
                if m in G.FC_SPAN:
                    fy = fc_years(m, season)
                    fc = G.first_issue_in_season(m, river, season, rp, n_req, span=fy)
                    recs.append(("forecasts", fc, fy & set(range(FC0, FC1 + 1))))
                    if max(fy) > FC1:
                        recs.append((f"forecasts to {max(fy)}", fc, fy))
                for basis, d, yrs in recs:
                    uni = yrs & assess
                    c = contingency(set(d), obs, uni)
                    lead = [(gx[y] - d[y]).days for y in set(d) & obs & uni]
                    rows.append(dict(season=season, river=river, model=m, record=basis, rp=rp, **c,
                                     lead_days_median=float(np.median(lead)) if lead else np.nan,
                                     lead_days_mean=float(np.mean(lead)) if lead else np.nan,
                                     hit_years=sorted(set(d) & obs & uni), false_years=sorted((set(d) - obs) & uni),
                                     missed_years=sorted((obs - set(d)) & uni)))
                    dates[(season, river, m, basis, rp)] = {int(y): v for y, v in d.items()}
            dates[(season, river, "gauge", "gauge", rp)] = {int(y): v for y, v in gx.items()}
    return pd.DataFrame(rows), dates


# ------------------------------------------------------------------------------ by lead
def lead_series(m, st, season, lead):
    """The forecast for each valid day issued `lead` days ahead (GloFAS: the middle member)."""
    k = ("rf", m)
    if k not in _c:
        rf = L.reforecast(m)
        rf = rf[rf.station.isin(STATIONS)]
        _c[k] = rf.groupby(["station", "leadtime_days", "valid_time"]).discharge.median()
    try:
        s = _c[k].loc[(st, lead)]
    except KeyError:
        return pd.Series(dtype=float)
    s = s[s.index.month.isin(SEASONS[season])]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s


def by_lead():
    """Per season, station, model and lead: forecast peak over reanalysis peak on the same days,
    rank correlation with the gauge, and the flood-season table at 1-in-3 (reanalysis-fitted level)."""
    rows = []
    for m in [NEW, "google_grrr", "glofas_v4"]:
        leads = range(1, 11) if m == NEW else range(1, 8)
        for season in SEASON_LIST:
            yrs = fc_years(m, season) & set(range(FC0, FC1 + 1))
            for st in STATIONS:
                g = gauge_daily(st, season)
                g = g[g.index.year.isin(yrs)]
                gmax = season_max(g)
                glev, mlev = gauge_level(st, season, 3), model_level(m, st, season, 3)
                obs = set(gmax[gmax >= glev].index) if not np.isnan(glev) else set()
                ra = model_daily(m, st, season, y1=G.RA_SPAN[m][1])
                for lead in leads:
                    f = lead_series(m, st, season, lead)
                    f = f[f.index.year.isin(yrs)]
                    if f.empty:
                        continue
                    j = pd.concat([f.rename("f"), ra.rename("r")], axis=1, join="inner").dropna()
                    pk = (j.groupby(j.index.year).f.max() / j.groupby(j.index.year).r.max()) if len(j) else pd.Series(dtype=float)
                    fmax = season_max(f)
                    uni = set(gmax.index) & set(fmax.index)
                    c = contingency(set(fmax[fmax >= mlev].index), obs, uni)
                    rows.append(dict(season=season, river=RIVER_OF[st], station=st, model=m, lead=lead,
                                     peak_ratio_median=float(pk.median()) if len(pk) else np.nan,
                                     rho_vs_gauge=lag_rho(f, g, 0)[0], rho_vs_reanalysis=spearman(j.f, j.r) if len(j) else np.nan,
                                     n_seasons=len(uni), **{k: c[k] for k in ("hits", "false_alarms", "misses", "correct_negatives")}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- discharge
def obs_discharge(st):
    if not OBS_Q_DIR or st not in OBS_Q_FILES:
        return None
    d = json.loads((Path(OBS_Q_DIR) / OBS_Q_FILES[st]).read_text())
    s = pd.Series({pd.Timestamp(k): v for k, v in d.items() if v is not None}, dtype=float).dropna().sort_index()
    return s[(s.index.year >= Y0) & (s.index.year <= Y1)]


def discharge():
    """Per station, model and period (each season, and the whole year): model flow against SWALIM's
    measured discharge on the same days."""
    rows = []
    for st in OBS_Q_FILES:
        o_all = obs_discharge(st)
        if o_all is None or o_all.empty:
            continue
        for m in MODELS:
            d = L.daily(m)
            x_all = d[d.station == st].set_index("date").discharge.sort_index()
            x_all.index = pd.DatetimeIndex(x_all.index).normalize()
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
                                 ratio_median=float(j.s.median() / j.o.median()), ratio_mean=beta,
                                 ratio_p90=float(j.s.quantile(0.9) / j.o.quantile(0.9)),
                                 ratio_peak_median=float((ys.s / ys.o).median()) if len(ys) else np.nan,
                                 pearson_r=r, spearman_rho=float(j.s.corr(j.o, method="spearman")),
                                 variability_ratio=alpha, KGE=float(kge), NSE=nse, log_NSE=lnse))
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
    out = dict(tracking=tracking(), station_3=station_events(3), station_5=station_events(5), by_lead=by_lead(), discharge=discharge())
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
    for col in ("rho_same_day", "rho_best_lag", "best_lag_days", "rho_flood_seasons", "peak_days_late_median", "crossing_days_early_median"):
        print("==", col); print(T.pivot_table(index=["season", "river", "station"], columns="model", values=col, sort=False)[["new", "cur", "v4", "v5"]].round(2))
    for rp in (3, 5):
        E = M[f"station_{rp}"].assign(model=lambda d: d.model.map(nm))
        for col in ("POD", "FAR", "CSI", "AUC", "rho_season_peaks"):
            print(f"== station 1-in-{rp}", col); print(E.pivot_table(index=["season", "river", "station"], columns="model", values=col, sort=False)[["new", "cur", "v4", "v5"]].round(2))
        Rv = M[f"river_{rp}"].assign(model=lambda d: d.model.map(nm))
        print(f"== river rule 1-in-{rp}"); print(Rv[["season", "river", "model", "record", "hits", "false_alarms", "misses", "n", "POD", "FAR", "CSI", "F1", "lead_days_median", "hit_years", "false_years", "missed_years"]].round(2).to_string(index=False))
    Dq = M["discharge"].assign(model=lambda d: d.model.map(nm))
    print("== discharge"); print(Dq[["station", "period", "model", "n_days", "ratio_median", "ratio_mean", "ratio_p90", "ratio_peak_median", "pearson_r", "variability_ratio", "KGE", "NSE", "log_NSE"]].round(2).to_string(index=False))
    Bl = M["by_lead"].assign(model=lambda d: d.model.map(nm))
    g = Bl.groupby(["model", "season", "river", "lead"]).agg(peak=("peak_ratio_median", "median"), rho=("rho_vs_gauge", "mean"), a=("hits", "sum"), b=("false_alarms", "sum"), c=("misses", "sum")).reset_index()
    g["POD"], g["FAR"] = g.a / (g.a + g.c), g.b / (g.a + g.b)
    print("== by lead"); print(g.round(2).to_string(index=False))
    print("worst", M["worst"])
