"""GEOGloWS bias correction, using the project's own methods.

Two routes, both from `geoglows.bias`:

* `sfdc_correct_retrospective()` applies `sfdc_bias_correction`, which uses
  the GEOGloWS project's precomputed SFDC tables (Riley Hales' SABER
  method). It needs no observed discharge, so it works at every reach in the
  registry: this is the route for the fourteen ungauged stations.
* `gauge_correct_forecast()` applies `correct_forecast` per forecast issue,
  which maps a forecast onto the observed-discharge scale. It needs observed
  discharge, so it is limited to the four stations that have it.

Two gotchas handled here so callers do not hit them:

* the processed parquets store discharge as float32 while the corrected
  values come back float64; recent pandas raises `TypeError` on the in-place
  update, so inputs are cast to float64 first.
* `correct_forecast` builds its mapping from a single month (the first day of
  the frame it is given), so it must be applied one forecast issue at a
  time, never to a multi-year series.

Known limitation, matching the open item in the trigger report: correcting
forecasts at the *ungauged* stations is not covered by either route. That
needs the forecast climatology mapped onto the retrospective first, fitted
on the 2024+ archive as it grows.
"""

import json

import numpy as np
import pandas as pd

from src.constants import STATIONS

OBSERVED_SLUGS = {
    "belet_weyne": "beletweyne",
    "bulo_burti": "buloburte",
    "luuq": "luuq",
    "bardheere": "bardheere",
}


def load_observed_discharge(station, stage="dev"):
    """Scraped SNRFA observed discharge for one station, as a Series."""
    import ocha_stratus as stratus

    slug = OBSERVED_SLUGS.get(station)
    if slug is None:
        return None
    raw = json.loads(
        stratus.load_blob_data(
            f"ds-aa-som-floods/raw/ef5/analysis/{slug}_real_discharge.json",
            stage=stage,
        )
    )
    s = pd.Series(raw, dtype="float64")
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def _as_frame(series):
    """Single-column float64 frame with a datetime index, as geoglows wants."""
    return series.astype("float64").to_frame("q").sort_index()


def sfdc_correct_retrospective(daily, stations=None):
    """SFDC-correct the GEOGloWS retrospective for each station.

    `daily` is the long-format discharge table (station | date | discharge)
    filtered to GEOGloWS, or the full table (other sources are ignored).
    Returns a long-format frame with the corrected series, plus a per-station
    report of what succeeded.
    """
    from geoglows import bias as gg_bias

    geo = daily[daily["source"] == "geoglows"] if "source" in daily else daily
    keys = stations or sorted(geo["station"].unique())
    out, report = [], []
    for key in keys:
        reach = getattr(STATIONS.get(key), "geoglows_river_id", None)
        sim = (
            geo[geo["station"] == key]
            .set_index("date")["discharge"]
            .sort_index()
        )
        if reach is None or not len(sim):
            report.append({"station": key, "status": "no reach or no data"})
            continue
        try:
            corrected = gg_bias.sfdc_bias_correction(_as_frame(sim), reach)
        except Exception as exc:  # network or missing SFDC table
            report.append(
                {"station": key, "status": f"{type(exc).__name__}: {exc}"}
            )
            continue
        cs = corrected.iloc[:, 0].rename("discharge")
        out.append(
            pd.DataFrame(
                {
                    "station": key,
                    "river": STATIONS[key].river,
                    "source": "geoglows_sfdc",
                    "date": cs.index,
                    "discharge": cs.values,
                }
            )
        )
        report.append(
            {
                "station": key,
                "status": "ok",
                "n_days": len(cs),
                "median_ratio_vs_raw": round(
                    float(cs.median() / sim.median()), 3
                ),
            }
        )
    frame = (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame(
            columns=["station", "river", "source", "date", "discharge"]
        )
    )
    return frame, pd.DataFrame(report)


def _quantile_map(values, fit_from, fit_to, n_q=99):
    """Monotonic empirical quantile map from one climatology onto another.

    Used for the first of the two steps below: putting forecast values on the
    scale of the model's own retrospective. Tails are scaled by the edge
    ratio rather than extrapolated.
    """
    values = np.asarray(values, dtype="float64")
    a = np.asarray(fit_from, dtype="float64")
    b = np.asarray(fit_to, dtype="float64")
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 30 or len(b) < 30:
        return values
    qs = np.linspace(1 / (n_q + 1), n_q / (n_q + 1), n_q)
    aq, bq = np.quantile(a, qs), np.quantile(b, qs)
    out = np.interp(values, aq, bq)
    hi = bq[-1] / aq[-1] if aq[-1] > 0 else 1.0
    lo = bq[0] / aq[0] if aq[0] > 0 else 1.0
    out[values > aq[-1]] = values[values > aq[-1]] * hi
    out[values < aq[0]] = values[values < aq[0]] * lo
    return out


def forecast_to_retro_scale(forecast_daily, retro):
    """Map a forecast series onto the model's own retrospective climatology.

    GEOGloWS forecasts run well below its retrospective (roughly half the
    median), so feeding them straight into `correct_forecast` - which assumes
    the input shares the retrospective climatology - penalises them twice and
    leaves them too dry to ever reach a flood threshold. Fitting this step on
    the days the two series share fixes that.
    """
    j = pd.concat(
        [forecast_daily.rename("fc"), retro.rename("retro")],
        axis=1,
        join="inner",
    ).dropna()
    if len(j) < 60:
        return forecast_daily.astype("float64")
    mapped = _quantile_map(
        forecast_daily.values, j["fc"].values, j["retro"].values
    )
    return pd.Series(mapped, index=forecast_daily.index, dtype="float64")


def gauge_correct_forecast(forecast, daily, stations=None, max_lead=7,
                           stage="dev"):
    """Correct GEOGloWS forecasts onto the observed scale, per issue.

    `forecast` is the archive table (station | issued_time | valid_time |
    leadtime_days | member | discharge); the ensemble median is taken per
    valid day, which is the 50%-of-members point used elsewhere in this
    analysis. Only stations with observed discharge can be corrected.
    """
    from geoglows import bias as gg_bias

    geo = daily[daily["source"] == "geoglows"] if "source" in daily else daily
    keys = stations or [k for k in OBSERVED_SLUGS if k in set(geo["station"])]
    out = []
    for key in keys:
        obs = load_observed_discharge(key, stage=stage)
        sim = (
            geo[geo["station"] == key]
            .set_index("date")["discharge"]
            .sort_index()
        )
        sub = forecast[
            (forecast["station"] == key)
            & (forecast["leadtime_days"].between(1, max_lead))
        ]
        if obs is None or not len(sim) or not len(sub):
            continue
        sim_f, obs_f = _as_frame(sim), _as_frame(obs)
        # step 1 fitted once per station: forecast -> retrospective scale
        fc_daily = (
            sub.groupby("valid_time")["discharge"].median().sort_index()
        )
        step1_full = forecast_to_retro_scale(fc_daily, sim)
        for issued, grp in sub.groupby("issued_time"):
            fc = (
                grp.groupby("valid_time")["discharge"]
                .median()
                .astype("float64")
                .to_frame("q")
                .sort_index()
            )
            # step 2: retrospective scale -> observed scale
            fc_step1 = (
                step1_full.reindex(fc.index).to_frame("q").astype("float64")
            )
            try:
                cf_one = gg_bias.correct_forecast(fc, sim_f, obs_f)
                cf = gg_bias.correct_forecast(fc_step1, sim_f, obs_f)
            except Exception:
                continue
            out.append(
                pd.DataFrame(
                    {
                        "station": key,
                        "river": STATIONS[key].river,
                        "source": "geoglows_gauge_corrected",
                        "issued_time": issued,
                        "valid_time": cf.index,
                        "leadtime_days": (
                            cf.index - pd.Timestamp(issued)
                        ).days,
                        "discharge_raw": fc["q"].reindex(cf.index).values,
                        "discharge_onestep": cf_one.iloc[:, 0].values,
                        "discharge": cf.iloc[:, 0].values,
                    }
                )
            )
    if not out:
        return pd.DataFrame(
            columns=[
                "station",
                "river",
                "source",
                "issued_time",
                "valid_time",
                "leadtime_days",
                "discharge_raw",
                "discharge_onestep",
                "discharge",
            ]
        )
    return pd.concat(out, ignore_index=True)


def ratio_report(model, observed):
    """Median / q95 / annual-max ratios of a model series against observed."""
    j = pd.concat(
        [model.rename("mod"), observed.rename("obs")], axis=1, join="inner"
    ).dropna()
    j = j[j["obs"] > 0]
    if not len(j):
        return {"n_days": 0}
    am = j.resample("YE").max().dropna()
    return {
        "n_days": len(j),
        "median": round(float(j["mod"].median() / j["obs"].median()), 2),
        "q95": round(
            float(j["mod"].quantile(0.95) / j["obs"].quantile(0.95)), 2
        ),
        "annual_max": round(float((am["mod"] / am["obs"]).median()), 2),
    }
