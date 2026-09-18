"""Figures and numbers for pages/ensemble-agreement/.

    .venv/Scripts/python.exe scripts/page_additions/ensemble_agreement_figs.py sweep
        -> figs/j_prob_sweep.png and ensemble_years_table.html from
           prob_sweep_results.json (written by scripts/prob_sweep.py)
    .venv/Scripts/python.exe scripts/page_additions/ensemble_agreement_figs.py operational
        -> figs/j_deyr2023_operational.png and, printed, the member counts over
           the live levels for Deyr 2023 (both rivers) and Gu 2024 (Shabelle)
           on the operational 50-member archive (data/glofas/raw/forecast_*).

Levels are the live ones (src/monitoring/thresholds.json, glofas_v4
reanalysis): the window's return period on the action leg, capped at
READINESS_RP_CAP on the readiness leg. Valid day = issue + lead - 1 day, the
GloFAS end-of-period stamp, as in src/monitoring/etl.py.
"""
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
DATA_REPO = Path(os.environ.get("SOM_DATA_REPO", REPO))
FIGS = REPO / "pages" / "ensemble-agreement" / "figs"

from src.constants import BODY, FAINT, GRID, INK, SEASONS, TRIGGER_STATIONS  # noqa: E402
from src.monitoring import config as cfg  # noqa: E402
from src.monitoring import thresholds as thr  # noqa: E402

ST = cfg.STATION_TITLE
WTITLE = {"juba_deyr": "Juba Deyr", "shabelle_deyr": "Shabelle Deyr",
          "juba_gu": "Juba Gu", "shabelle_gu": "Shabelle Gu"}
ORDER = ["juba_deyr", "shabelle_deyr", "juba_gu", "shabelle_gu"]
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"],
                     "axes.edgecolor": GRID, "axes.labelcolor": BODY, "xtick.color": FAINT,
                     "ytick.color": FAINT, "text.color": INK, "axes.titlelocation": "left",
                     "axes.titleweight": "bold", "axes.titlesize": 11})


# ------------------------------------------------------------------ sweep
def sweep():
    res = pd.read_json(HERE / "prob_sweep_results.json")
    ks = list(range(1, 12))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), dpi=150, sharex=True)
    panels = ((axes[0], "readiness", f"Readiness leg (leads {cfg.READINESS_LEADS[0]}-{cfg.READINESS_LEADS[1]} d)"),
              (axes[1], "action", f"Action leg (leads {cfg.ACTION_LEADS[0]}-{cfg.ACTION_LEADS[1]} d)"))
    for ax, leg, title in panels:
        d = res[res.leg == leg]
        mat = np.array([[float(d[(d.window == w) & (d.k == k)].f1_severe.iloc[0]) for k in ks] for w in ORDER])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(ORDER)):
            for j in range(len(ks)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if mat[i, j] > 0.45 else INK)
        n_sev = {w: int(d[d.window == w].severe_n.iloc[0]) for w in ORDER}
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([f"{WTITLE[w]}  (severe n={n_sev[w]})" for w in ORDER], fontsize=9)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"{k}/11\n{k / 11:.0%}" for k in ks], fontsize=8.5)
        ax.set_xticks([x - 0.5 for x in range(len(ks) + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(len(ORDER) + 1)], minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_patch(plt.Rectangle((5 - 0.5, -0.5), 1, len(ORDER), fill=False,
                                   edgecolor="#B34036", lw=2.2, zorder=4))
        ax.set_title(title, fontsize=10)
    axes[1].set_xlabel("members that must cross the level for a point to count (red box: the median rule)")
    fig.suptitle("Severe-year F1 of each window rule vs the ensemble agreement level, "
                 "GloFAS v4 reforecast 2003-2023", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.005,
             "Levels from the v4 reanalysis at each leg's return period (readiness capped at 1-in-5); a point "
             "counts when at least k of 11 members cross on one issue and valid day; the window activates when "
             "the rule's point count is reached on that same issue and valid day, as the live pipeline counts "
             "votes. The median rule is k = 6.", fontsize=7.5, color=FAINT)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIGS / "j_prob_sweep.png", bbox_inches="tight")
    plt.close(fig)

    # years table: every year that activates at any k on either leg, plus every severe year
    rows = []
    for w in ORDER:
        d = res[res.window == w]
        severe = set(d.severe_years.iloc[0])
        flood = set(d.flood_years.iloc[0])
        years = set(severe)
        for _, r in d.iterrows():
            years |= set(r.years)
        for y in sorted(years):
            cells = []
            for leg in ("readiness", "action"):
                dl = d[d.leg == leg]
                kmax = max([int(r.k) for _, r in dl.iterrows() if y in set(r.years)], default=0)
                cells.append("at every level" if kmax == 11 else ("no" if kmax == 0 else f"up to {kmax} of 11"))
            bench = ("<em>severe (1-in-5)</em>" if y in severe
                     else ("flood (1-in-3)" if y in flood else "below 1-in-3"))
            rows.append((w, y, bench, cells[0], cells[1]))
    out, prev = [], None
    for w, y, bench, rd, ac in rows:
        label = WTITLE[w] if w != prev else ""
        out.append(f"<tr><td>{label}</td><td>{y}</td><td>{bench}</td><td>{rd}</td><td>{ac}</td></tr>")
        prev = w
    (HERE / "ensemble_years_table.html").write_text("<tbody>" + "".join(out) + "</tbody>", encoding="utf-8")
    print("wrote", FIGS / "j_prob_sweep.png", "and ensemble_years_table.html")
    for leg in ("readiness", "action"):
        for w in ORDER:
            r = res[(res.window == w) & (res.leg == leg) & (res.k == 6)].iloc[0]
            print(f"  {leg:9s} {WTITLE[w]:14s} k=6 years {r.years}  severe {r.severe_years}  flood {r.flood_years}")


# ------------------------------------------------------------ operational
RAW = {"deyr": DATA_REPO / "data" / "glofas" / "raw" / "forecast_operational",
       "gu": DATA_REPO / "data" / "glofas" / "raw" / "forecast_gu2024"}


def members(station, year, months, season):
    frames = []
    for m in months:
        for f in sorted(RAW[season].glob(f"{station}_{year}{m:02d}_ens_*.nc")):
            ds = xr.open_dataset(f)
            da = ds["dis24"]
            for dim in ("latitude", "longitude"):
                da = da.isel({dim: da.sizes[dim] // 2})
            frames.append(da.to_dataframe().reset_index()[
                ["number", "forecast_period", "forecast_reference_time", "dis24"]])
            ds.close()
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["number", "forecast_period", "forecast_reference_time"])
    d["lead"] = (d.forecast_period.dt.total_seconds() / 86400).round().astype(int)
    d["issued"] = pd.to_datetime(d.forecast_reference_time).dt.normalize()
    d["valid"] = d.issued + pd.to_timedelta(d.lead - 1, unit="D")
    return d[["number", "lead", "issued", "valid", "dis24"]]


def season_check(river, season, year):
    lv = thr.load()
    stations = TRIGGER_STATIONS[river]
    months = SEASONS[season]
    frames = {st: members(st, year, months, season) for st in stations}
    frames = {k: v for k, v in frames.items() if v is not None}
    n_members = max(v.number.nunique() for v in frames.values())
    print(f"\n=== {river} {season} {year}: operational v4, {n_members} members")
    for st, d in frames.items():
        print(f"  {st:12s} issues {d.issued.min().date()}..{d.issued.max().date()} ({d.issued.nunique()} issue days)")
    out = {}
    legs = [("readiness", cfg.READINESS_LEADS, cfg.READINESS_RULES[(river, season)])]
    if cfg.ACTION_RULES[(river, season)]["source"] == "glofas":
        legs.insert(0, ("action", cfg.ACTION_LEADS, cfg.ACTION_RULES[(river, season)]))
    for leg, (lo, hi), rule in legs:
        levels = thr.lookup(lv, cfg.GLOFAS_OPERATIONAL, season, rule["rp"], stations)
        print(f"\n--- {leg} (leads {lo}-{hi}), {rule['n_req']} of {len(stations)} over 1-in-{rule['rp']}: "
              + ", ".join(f"{ST[s]} {levels[s]:,.0f}" for s in stations))
        med, share = {}, {}
        for st, d in frames.items():
            b = d[d.lead.between(lo, hi) & d.valid.dt.month.isin(months)]
            med[st] = b.groupby(["issued", "valid"]).dis24.median()
            share[st] = b.assign(hit=b.dis24 >= levels[st]).groupby(["issued", "valid"]).hit.mean()
            top = b.dis24.max()
            when = ""
            if share[st].max() > 0:
                iss, val = share[st].idxmax()
                when = f" on {val.date()} (issued {iss.date()})"
            print(f"  {ST[st]:12s} top member {top:7.0f} ({top / levels[st]:.0%} of level) | median peak "
                  f"{med[st].max():7.0f} ({med[st].max() / levels[st]:.0%}) | max member share over level "
                  f"{share[st].max():.0%}{when}")
        M = pd.DataFrame(med)
        votes = (M >= pd.Series(levels)[M.columns]).sum(axis=1)
        act = votes[votes >= rule["n_req"]]
        if len(act):
            iss, val = act.index[0]
            print(f"  MEDIAN RULE reached: issued {iss.date()} for {val.date()}; max votes {votes.max()}")
        else:
            print(f"  median rule not reached; max same-issue-and-day points over level: {votes.max()}")
        Sh = pd.DataFrame(share).fillna(0)
        kth = Sh.apply(lambda r: sorted(r, reverse=True)[rule["n_req"] - 1], axis=1)
        top_txt = ""
        if kth.max() > 0:
            iss, val = kth.idxmax()
            top_txt = f" (issued {iss.date()} for {val.date()})"
        print(f"  highest member share at >= {rule['n_req']} points on one issue and day: {kth.max():.0%}{top_txt}")
        for p in (0.02, 0.1, 0.2, 0.3, 0.5):
            days = kth[kth >= p - 1e-9]
            if len(days):
                iss, val = days.index[0]
                print(f"    agreement level {p:.0%}: reached, issued {iss.date()} for {val.date()}")
            else:
                print(f"    agreement level {p:.0%}: not reached")
        out[leg] = {"levels": levels, "frames": frames, "lo": lo, "hi": hi}
    return out


def deyr2023_figure(check, gauge_dates):
    """Shabelle Deyr 2023: 50-member range across action leads per valid day, the most
    alarming issue's median, the live 1-in-5 levels, SWALIM's alert and the gauge crossings."""
    lo, hi = check["action"]["lo"], check["action"]["hi"]
    levels, frames = check["action"]["levels"], check["action"]["frames"]
    stations = TRIGGER_STATIONS["shabelle"]
    colors = {"belet_weyne": "#0E8A7B", "bulo_burti": "#2A78D6", "jowhar": "#8E5FA8"}
    rp = cfg.ACTION_RULES[("shabelle", "deyr")]["rp"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), dpi=150, sharex=True)
    for ax, st in zip(axes, stations):
        d = frames[st]
        b = d[d.lead.between(lo, hi)]
        rng = b.groupby("valid").dis24.agg(["min", "max"])
        medi = b.groupby(["issued", "valid"]).dis24.median().groupby("valid").max()
        ax.fill_between(rng.index, rng["min"], rng["max"], color=colors[st], alpha=0.22, lw=0,
                        label=f"all 50 members, leads {lo}-{hi}")
        ax.plot(medi.index, medi.values, color=colors[st], lw=1.8, label="ensemble median, most alarming issue")
        ax.axhline(levels[st], color="#B34036", ls="--", lw=1.2)
        ax.text(pd.Timestamp("2023-12-31"), levels[st] + 15, f"1-in-{rp} level {levels[st]:,.0f}",
                ha="right", va="bottom", fontsize=8.5, color="#B34036")
        marks = ((pd.Timestamp("2023-10-20"), "SWALIM alert,\nAA call", INK),
                 (gauge_dates[3], "gauges 1-in-3", FAINT),
                 (gauge_dates[5], "gauges 1-in-5", "#B34036"))
        for x, lab, c in marks:
            ax.axvline(x, color=c, ls=":", lw=1)
            if st == stations[0]:
                ax.text(x, 1290, lab, ha="center", va="top", fontsize=8.5, color=c)
        ax.set_ylim(0, 1300)
        ax.set_ylabel(f"{ST[st]}\nm³/s")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False, bbox_to_anchor=(1, 0.88))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].set_xlim(pd.Timestamp("2023-10-01"), pd.Timestamp("2024-01-05"))
    fig.suptitle("Deyr 2023 on the operational GloFAS v4 ensemble: no member reaches the level on the Shabelle",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.005,
             "Shaded: the full 50-member range across leads 1-7 for each valid day, all issues. Line: the ensemble "
             "median of the most alarming issue for that day. Dashed: the live 1-in-5 level from the v4 reanalysis. "
             f"Dotted: SWALIM's Flood Alert (20 Oct) and the day the second Shabelle gauge crossed its own 1-in-3 "
             f"({gauge_dates[3]:%d %b}) and 1-in-5 ({gauge_dates[5]:%d %b}).", fontsize=7.5, color=FAINT)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIGS / "j_deyr2023_operational.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / "j_deyr2023_operational.png")


def gauge_crossing_dates():
    """Day the second Shabelle gauge reached its own 1-in-3 / 1-in-5 level in Deyr 2023 (somlib)."""
    sys.path.insert(0, str(HERE))
    import somlib as L
    return {rp: pd.Timestamp(L.gauge_crossings("shabelle", "deyr", rp)[2023]) for rp in (3, 5)}


def operational():
    sh = season_check("shabelle", "deyr", 2023)
    season_check("juba", "deyr", 2023)
    season_check("shabelle", "gu", 2024)
    dates = gauge_crossing_dates()
    print("\nShabelle Deyr 2023 two-gauge crossings:", {k: str(v.date()) for k, v in dates.items()})
    deyr2023_figure(sh, dates)


if __name__ == "__main__":
    {"sweep": sweep, "operational": operational}[sys.argv[1]]()
