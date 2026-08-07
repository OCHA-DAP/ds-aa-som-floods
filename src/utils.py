import numpy as np
import pandas as pd


def annual_maxima(series):
    """Annual maximum value per calendar year (NaNs dropped first)."""
    s = series.dropna()
    return s.groupby(s.index.year).max()


def return_period_for_level(annual_max, level):
    """Interpolate the return period (years) at which `level` is reached.

    Uses the same Weibull plotting-position convention as return-level
    calculations elsewhere (rank i, ascending, has non-exceedance
    probability i/(n+1)); this is its inverse — given a level, find the
    continuous rank via linear interpolation, then convert to a return
    period. Returns NaN if `level` exceeds every annual maximum on record
    (i.e. the return period can't be estimated without extrapolating
    beyond the length of record).
    """
    values = np.sort(annual_max.dropna().to_numpy())
    n = len(values)
    if n == 0 or level > values[-1]:
        return float("nan")
    if level <= values[0]:
        return (n + 1) / n
    ranks = np.arange(1, n + 1)
    r = np.interp(level, values, ranks)
    return (n + 1) / (n + 1 - r)


def compute_threshold_stats(levels, thresholds, threshold_col, min_recession_days=14):
    """Per-station exceedance stats + flood events for one threshold column.

    `levels` is {station_number: DataFrame} with a "level(m)" column and a
    date index (as loaded by `src.datasources.swalim.load_station_levels`).
    `thresholds` is the DataFrame from `load_thresholds()`, indexed by
    station number, with a `threshold_col` column (e.g. "Moderate Flood
    Risk" or "High Flood Risk") in meters.

    Returns (summary, station_events, station_exceed_dates):
      - summary: DataFrame indexed by station_number with threshold, record
        length, exceedance day/event counts, and events_per_year.
      - station_events: {station_number: DataFrame[start, end]} flood events.
      - station_exceed_dates: {station_number: DatetimeIndex} raw exceedance
        days (only for stations with a threshold and at least one event).
    """
    station_events = {}
    station_exceed_dates = {}
    rows = []

    for station_number, df in levels.items():
        if len(df) == 0 or station_number not in thresholds.index:
            continue
        threshold = thresholds.loc[station_number, threshold_col]
        label = thresholds.loc[station_number, "Label"]

        obs = df["level(m)"].dropna()
        if len(obs) == 0 or pd.isna(threshold):
            n_obs = len(obs)
            n_years = float("nan")
            n_exceed_days = 0
            pct_exceed = float("nan")
            n_events = 0
            events_per_year = float("nan")
        else:
            n_obs = len(obs)
            n_years = (obs.index.max() - obs.index.min()).days / 365.25
            exceed = obs[obs >= threshold]
            n_exceed_days = len(exceed)
            pct_exceed = 100 * n_exceed_days / n_obs
            periods = event_periods(exceed.index, min_recession_days=min_recession_days)
            station_events[station_number] = periods
            station_exceed_dates[station_number] = exceed.index
            n_events = len(periods)
            events_per_year = n_events / n_years if n_years > 0 else float("nan")

        rows.append(
            {
                "station_number": station_number,
                "station_name": label,
                "threshold_m": threshold,
                "n_obs_days": n_obs,
                "record_years": n_years,
                "n_exceed_days": n_exceed_days,
                "pct_days_exceed": pct_exceed,
                "n_events": n_events,
                "events_per_year": events_per_year,
            }
        )

    summary = pd.DataFrame(rows).set_index("station_number")
    summary = summary.sort_values("events_per_year", ascending=False)
    return summary, station_events, station_exceed_dates


def return_periods_table(levels, thresholds, threshold_col, station_events, summary):
    """Per-station return period (annual-max + empirical) for one threshold."""
    rows = []
    for station_number, periods in station_events.items():
        threshold = thresholds.loc[station_number, threshold_col]
        label = thresholds.loc[station_number, "Label"]
        obs = levels[station_number]["level(m)"].dropna()
        am = annual_maxima(obs)
        rp_annual_max = return_period_for_level(am, threshold)
        pct_years = 100 * (am >= threshold).mean()
        rp_empirical = (
            summary.loc[station_number, "record_years"]
            / summary.loc[station_number, "n_events"]
        )
        rows.append(
            {
                "station_number": station_number,
                "station_name": label,
                "threshold_m": threshold,
                "n_years_with_data": len(am),
                "return_period_annual_max": rp_annual_max,
                "pct_years_at_or_above": pct_years,
                "return_period_empirical": rp_empirical,
            }
        )
    result = pd.DataFrame(rows).set_index("station_number")
    return result.sort_values("return_period_annual_max")


def system_events_table(
    station_events, station_exceed_dates, thresholds, min_recession_days=14, tolerance_days=14
):
    """Pool per-station events into systemwide events, with station membership.

    A systemwide event is the union of every station's exceedance days,
    re-collapsed with the same recession rule; each systemwide event is then
    matched back to whichever individual stations had an event within
    `tolerance_days` of its window.
    """
    if not station_exceed_dates:
        return pd.DataFrame(columns=["start", "end", "stations", "n_stations", "station_names"])

    all_exceed_dates = pd.DatetimeIndex(
        sorted(set().union(*station_exceed_dates.values()))
    )
    system_events = event_periods(all_exceed_dates, min_recession_days=min_recession_days)

    def stations_in_window(start, end):
        lo = start - pd.Timedelta(days=tolerance_days)
        hi = end + pd.Timedelta(days=tolerance_days)
        return [
            sn
            for sn, periods in station_events.items()
            if ((periods["start"] <= hi) & (periods["end"] >= lo)).any()
        ]

    system_events["stations"] = [
        stations_in_window(row.start, row.end) for row in system_events.itertuples()
    ]
    system_events["n_stations"] = system_events["stations"].apply(len)
    system_events["station_names"] = system_events["stations"].apply(
        lambda sns: ", ".join(sorted(thresholds.loc[sn, "Label"] for sn in sns))
    )
    return system_events.sort_values("start")


def event_periods(dates, min_recession_days=14):
    """Group exceedance dates into distinct flood events (start/end per event).

    A new event starts only once the water has been below threshold for at
    least `min_recession_days` — i.e. the gap since the *previous* exceedance
    date, not since the current event's start — so one long continuous
    exceedance run is always a single event, however long it runs.
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates)))
    if len(dates) == 0:
        return pd.DataFrame(columns=["start", "end"])
    starts = [dates[0]]
    ends = [dates[0]]
    for prev, d in zip(dates[:-1], dates[1:]):
        if (d - prev).days > min_recession_days:
            starts.append(d)
            ends.append(d)
        else:
            ends[-1] = d
    return pd.DataFrame({"start": starts, "end": ends})


def yT(T):
    """Gumbel reduced variate for return period T."""
    return -np.log(-np.log(1 - 1 / T))


def gumbel_rp(series, T):
    """Discharge at return period T from a Gumbel fit to annual maxima.

    Used for GloFAS, which publishes no return periods per cell.
    """
    am = series.resample("YE").max().dropna()
    beta = am.std(ddof=1) * np.sqrt(6) / np.pi
    return (am.mean() - 0.5772 * beta) + beta * yT(T)


def weibull_level(am, target_rp):
    """Water level at `target_rp` via Weibull plotting position (log-log interp).

    Returns NaN with fewer than 8 annual maxima or when `target_rp` exceeds
    the record length (no extrapolation).
    """
    am = np.sort(np.asarray(am, dtype=float))[::-1]
    n = len(am)
    if n < 8:
        return np.nan
    rps = (n + 1) / np.arange(1, n + 1)
    if target_rp > rps[0]:
        return np.nan
    lo = np.log(np.maximum(am[::-1], 1e-3))
    return float(np.exp(np.interp(np.log(target_rp), np.log(rps[::-1]), lo)))


def weibull_threshold(am, target_rp):
    """Discharge at `target_rp` via Weibull plotting position (log-log interp).

    Same estimator as `weibull_level` but with no minimum-record guard,
    matching the trigger-grid usage on model series.
    """
    am = np.sort(np.asarray(am))[::-1]
    n = len(am)
    rps = (n + 1) / np.arange(1, n + 1)
    if target_rp > rps[0]:
        return np.nan
    return float(
        np.exp(
            np.interp(
                np.log(target_rp),
                np.log(rps[::-1]),
                np.log(np.maximum(am[::-1], 1e-6)),
            )
        )
    )


def auc(score, label):
    """Rank-based ROC AUC (Mann-Whitney); NaN if labels are one-class."""
    r = pd.Series(score).rank()
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def episodes(mask, gap_days=14):
    """Group True days into (start, end) episodes; gaps > gap_days split."""
    days = mask.index[mask]
    if not len(days):
        return []
    out, s, p = [], days[0], days[0]
    for d in days[1:]:
        if (d - p).days > gap_days:
            out.append((s, p))
            s = d
        p = d
    out.append((s, p))
    return out


def hits(a, b, tol=7):
    """Count episodes in `a` overlapped by any episode in `b` within tol days."""
    t = pd.Timedelta(days=tol)
    return sum(any(bs <= e + t and be >= s - t for bs, be in b) for s, e in a)
