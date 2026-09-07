"""An observational fallback: SWALIM's official bank-full (and high-risk) levels at
the gauges as a trigger of last resort when the forecast windows have not
activated. Per window and year: when a gauge first read bank full, when two did,
how that sits against the forecast trigger's first day and the gauges' own 1-in-3
and 1-in-5 crossings, and what the fallback would add or cost.
Writes figs/l_obs_fallback.png, obs_fallback.json and obs_fallback_table.html."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import somlib as L
from src.constants import TRIGGER_CONFIG

S = Path(__file__).parent
FIGS = PAGE_DIR / "figs"
TH = L.swalim_thresholds()


def first_level_crossings(river, season, level_col, n_req):
    """Year -> (first date on which >= n_req gauges have read at or above their official
    level that season, [stations that had crossed by then])."""
    m = L.gauge_matrix(river, season)
    lev = TH[level_col].reindex(L.TRIGGER_STATIONS[river]).dropna()
    cols = [c for c in lev.index if c in m.columns]
    out = {}
    for y, g in m[cols].groupby(m.index.year):
        if y not in L.SPAN:
            continue
        crossed = (g >= lev[cols]).cummax()
        n = crossed.sum(axis=1)
        hit = n[n >= n_req]
        if len(hit):
            d = hit.index[0]
            out[y] = (d.normalize(), [L.NAME[c] for c in cols if crossed.loc[d, c]])
    return out


def days(a, b):
    return None if a is None or b is None else int((a - b).days)


fmt = lambda d: d.strftime("%d %b").lstrip("0") if d is not None else ""
rows, summary = [], []
matrix = {}
for river, season in L.WINDOWS:
    cfg = TRIGGER_CONFIG[(river, season)]
    flood, severe, ok = L.benchmark(river, season)
    onset3 = L.gauge_crossings(river, season, 3)
    onset5 = L.gauge_crossings(river, season, 5)
    trig = L.first_crossing_dates(cfg["source"], river, season, cfg["rp"], cfg["n_req"])
    bank1 = first_level_crossings(river, season, "bank_full", 1)
    bank2 = first_level_crossings(river, season, "bank_full", 2)
    high1 = first_level_crossings(river, season, "high_flood_risk", 1)
    high2 = first_level_crossings(river, season, "high_flood_risk", 2)
    levels_txt = ", ".join(f"{L.NAME[st]} {TH.loc[st, 'bank_full']:.1f} m" for st in L.TRIGGER_STATIONS[river]
                           if st in TH.index and not np.isnan(TH.loc[st, "bank_full"]))
    years = sorted(set(flood) | set(trig) | set(bank1) | set(high2))
    for y in years:
        bench = "severe" if y in severe else ("flood" if y in flood else "none")
        t = trig.get(y); b1 = bank1.get(y); b2 = bank2.get(y); h2 = high2.get(y); h1 = high1.get(y)
        o3 = onset3.get(y); o5 = onset5.get(y)
        rows.append({
            "window": L.wname(river, season), "river": river, "season": season, "year": y, "benchmark": bench,
            "trigger": fmt(t), "onset3": fmt(o3), "onset5": fmt(o5),
            "high2": fmt(h2[0]) if h2 else "", "high2_st": ", ".join(h2[1]) if h2 else "",
            "bank1": fmt(b1[0]) if b1 else "", "bank1_st": ", ".join(b1[1]) if b1 else "",
            "bank2": fmt(b2[0]) if b2 else "", "bank2_st": ", ".join(b2[1]) if b2 else "",
            "bank1_vs_trigger": days(b1[0], t) if b1 else None,       # + means bank full came after the trigger
            "bank1_vs_onset3": days(b1[0], o3) if b1 else None,
            "bank1_vs_onset5": days(b1[0], o5) if b1 else None,
            "high2_vs_trigger": days(h2[0], t) if h2 else None,
        })
    # standalone and combined scores
    U = set(L.SPAN)
    trig_years = set(trig)
    def sc(active):
        return {"activations": sorted(active), "vs_flood": L.contingency(active, flood, U),
                "vs_severe": L.contingency(active, severe, U)}
    summary.append({
        "window": L.wname(river, season), "river": river, "season": season, "levels": levels_txt,
        "flood_years": sorted(flood), "severe_years": sorted(severe),
        "forecast_trigger": sc(trig_years),
        "bank_full_any_gauge": sc(set(bank1)),
        "bank_full_two_gauges": sc(set(bank2)),
        "high_risk_two_gauges": sc(set(high2)),
        "trigger_or_bank_full": sc(trig_years | set(bank1)),
        "trigger_or_high_two": sc(trig_years | set(high2)),
        "recovered_by_bank_full": sorted((set(bank1) - trig_years) & flood),
        "added_outside_benchmark_by_bank_full": sorted(set(bank1) - trig_years - flood),
        "bank_full_lag_vs_onset5_days": sorted(int((bank1[y][0] - onset5[y]).days) for y in bank1 if y in onset5),
        "bank_full_lag_vs_onset3_days": sorted(int((bank1[y][0] - onset3[y]).days) for y in bank1 if y in onset3),
    })
    matrix[(river, season)] = {"flood": flood, "severe": severe, "trigger": trig_years, "bank1": set(bank1),
                               "bank2": set(bank2), "high2": set(high2)}
    print(f"{L.wname(river, season):14s} trigger {sorted(trig_years)}\n   bank full any {sorted(bank1)}\n   bank full two {sorted(bank2)}"
          f"\n   high two {sorted(high2)}\n   recovered {summary[-1]['recovered_by_bank_full']} | outside benchmark {summary[-1]['added_outside_benchmark_by_bank_full']}"
          f"\n   bank-full lag vs 1-in-5 onset (d): {summary[-1]['bank_full_lag_vs_onset5_days']}")

json.dump({"rows": rows, "summary": summary}, open(S / "obs_fallback.json", "w"), indent=1, default=str)

# ---- figure: year matrix per window --------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(11.5, 7.8), sharex=True)
TRACKS = [("flood", "gauges: two over 1-in-3", "#9ca3af", "s"), ("severe", "gauges: two over 1-in-5", "#111827", "s"),
          ("trigger", "forecast trigger (adopted model)", "#1d4ed8", "D"),
          ("high2", "two gauges at the high-risk level", "#f59e0b", "o"),
          ("bank1", "one gauge at bank full", "#b45309", "^"), ("bank2", "two gauges at bank full", "#7c2d12", "^")]
for ax, (river, season) in zip(axes, L.WINDOWS):
    mx = matrix[(river, season)]
    for k, (key, label, c, mk) in enumerate(TRACKS):
        y = len(TRACKS) - 1 - k
        ys = sorted(mx[key])
        ax.scatter(ys, [y] * len(ys), color=c, marker=mk, s=42, zorder=3, label=label if ax is axes[0] else None)
    ax.set_yticks(range(len(TRACKS))); ax.set_yticklabels([t[1] for t in TRACKS][::-1], fontsize=8.2)
    ax.set_ylim(-.6, len(TRACKS) - .4)
    ax.set_title(L.wname(river, season), loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#f3f4f6"); ax.tick_params(length=0, labelsize=8.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axes[-1].set_xticks(range(L.Y0, L.Y1 + 1)); axes[-1].set_xticklabels([str(y)[2:] for y in range(L.Y0, L.Y1 + 1)])
axes[-1].set_xlabel("year (19xx / 20xx)", fontsize=9)
fig.tight_layout(h_pad=1.2)
fig.savefig(FIGS / "l_obs_fallback.png", dpi=150)
print("wrote", FIGS / "l_obs_fallback.png")

# ---- HTML table -----------------------------------------------------------------
def lagtxt(n, what):
    if n is None:
        return ""
    if n == 0:
        return f"same day as {what}"
    return f"{abs(n)} d {'after' if n > 0 else 'before'} {what}"


trs = []
for r in rows:
    if not (r["bank1"] or r["high2"] or r["benchmark"] != "none" or r["trigger"]):
        continue
    trs.append(f"<tr><td>{r['window']}</td><td>{r['year']}</td><td>{r['benchmark']}</td>"
               f"<td>{r['onset3'] or 'no'}<br><span style='color:var(--n7)'>1-in-5: {r['onset5'] or 'no'}</span></td>"
               f"<td>{r['trigger'] or 'never'}</td>"
               f"<td>{(r['high2'] + ' (' + r['high2_st'] + ')') if r['high2'] else 'no'}</td>"
               f"<td>{(r['bank1'] + ' (' + r['bank1_st'] + ')') if r['bank1'] else 'no'}"
               + (f"<br><span style='color:var(--n7)'>{lagtxt(r['bank1_vs_trigger'], 'the trigger') if r['trigger'] else 'trigger never crossed'}"
                  f"; {lagtxt(r['bank1_vs_onset5'], '1-in-5') if r['onset5'] else 'no 1-in-5 crossing'}</span>" if r["bank1"] else "")
               + f"</td><td>{(r['bank2'] + ' (' + r['bank2_st'] + ')') if r['bank2'] else 'no'}</td></tr>")
html = ('<div class="tablewrap">\n<table class="data" style="font-size:12px">\n<thead><tr>'
        '<th>window</th><th>year</th><th>benchmark</th><th>gauges: two over 1-in-3</th><th>forecast trigger first day</th>'
        '<th>two gauges at the high-risk level</th><th>first gauge at bank full</th><th>two gauges at bank full</th>'
        '</tr></thead>\n<tbody>' + "".join(trs) + "</tbody>\n</table>\n</div>")
(S / "obs_fallback_table.html").write_text(html, encoding="utf-8")
print("rows:", len(trs))
