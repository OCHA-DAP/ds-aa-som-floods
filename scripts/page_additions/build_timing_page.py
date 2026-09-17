"""Build pages/activation-timing/: when the trigger activates, against the gauges and the
flooding, for both seasons. Deyr first, then Gu."""
import os
import shutil
import sys

sys.path.insert(0, "C:/Users/pauni/AppData/Local/Temp/claude/C--Users-pauni-Desktop-Work-OCHA-GitHub/f11e6aed-1a18-4740-beb3-3ee68ff96279/scratchpad")
import numpy as np
import season_charts as m

OUT = "C:/Users/pauni/AppData/Local/Temp/claude/C--Users-pauni-Desktop-Work-OCHA-GitHub/f11e6aed-1a18-4740-beb3-3ee68ff96279/scratchpad/wt-trigger/pages/activation-timing/"
FIGS = OUT + "figs/"
os.makedirs(FIGS, exist_ok=True)
SOURCE = {"deyr": "GloFAS version 4", "gu": "Google Flood Hub"}
ARCH = {"deyr": "2003 to 2023, about two issues a week", "gu": "2016 to 2023, daily"}
f = lambda d: d.strftime("%d %b") if d is not None else None


def collect(season):
    out = []
    for y in m.seasons_of_interest(season):
        acts = m.activations(season, y)
        bench = m.benchmark(season, y)
        fd, nd = m.flood_date(season, y)
        png, _, _, _ = m.chart(season, y)
        shutil.copy(png, FIGS + os.path.basename(png))
        first = [d for d, _, _ in acts.values() if d is not None]
        out.append(dict(year=int(y), acts=acts, bench=bench, flood=fd, nd=nd,
                        fig=f"figs/{os.path.basename(png)}",
                        first=min(first) if first else None,
                        has_arch=all(h for _, _, h in acts.values())))
    return out


def sections(season, data):
    out = []
    for r in sorted(data, key=lambda x: -x["year"]):
        a = r["acts"]
        bits = []
        for riv in ("Juba", "Shabelle"):
            if a[riv][0] is not None:
                bits.append(f"{riv} {f(a[riv][0])}")
        act = ", ".join(bits) or "no activation"
        b = ", ".join(f"{k} {f(v)}" for k, v in r["bench"].items()) or "fewer than two gauges crossed"
        out.append(f"""<section class="yr" id="{season}{r['year']}">
<h3>{m.SEASON_TITLE[season]} {r['year']}</h3>
<dl><dt>Activation</dt><dd>{act}</dd>
<dt>Two-gauge benchmark onset</dt><dd>{b}</dd></dl>
<figure><img src="{r['fig']}" alt="{m.SEASON_TITLE[season]} {r['year']}" loading="lazy"></figure>
</section>""")
    return "\n".join(out)


def summary(season, data):
    leads = [(r["flood"] - r["first"]).days for r in data if r["first"] is not None and r["flood"] is not None]
    bl = [(list(r["bench"].values())[0] - r["first"]).days for r in data
          if r["first"] is not None and r["bench"]]
    fired = sum(1 for r in data if r["first"] is not None)
    arch = sum(1 for r in data if r["has_arch"])
    flooded = sum(1 for r in data if r["flood"] is not None)
    caught = sum(1 for r in data if r["first"] is not None and r["flood"] is not None)
    med = lambda xs: f"{np.median(xs):+.0f} d" if xs else "&mdash;"
    return (f"<p>Seasons shown: {len(data)}, of which {arch} have a forecast archive. The trigger is met in "
            f"{fired}. Exposure reaches the three-district level in {flooded}, and in {caught} of those the "
            f"trigger is also met. Median days from the trigger being met to that exposure date: {med(leads)}. "
            f"To the second gauge reaching its 1-in-3: {med(bl)}.</p>")


def build():
    parts = {}
    for season in ("deyr", "gu"):
        data = collect(season)
        parts[season] = (data, sections(season, data), summary(season, data))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Timing of activations &mdash; Somalia Riverine Flood Trigger</title>
<meta name="description" content="When the Somalia riverine flood trigger activates, season by season, against the river gauges and the flood exposure that followed. Deyr and Gu.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css">
<style>
.hero.hero-sub {{ padding:40px 44px 34px; }}
.hero.hero-sub h1 {{ font-size:28px; }}
.crumb {{ font-size:12px; margin:0 0 14px; }}
.crumb a {{ color:rgba(255,255,255,.85); text-decoration:none; }}
.crumb a:hover {{ text-decoration:underline; }}
article {{ padding:8px 44px 40px; max-width:980px; }}
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:21px; color:var(--n9); margin:40px 0 10px; }}
article h3 {{ font-family:'Merriweather',Georgia,serif; font-size:17px; color:var(--n9); margin:34px 0 8px; }}
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.65; }}
article a {{ color:var(--b6); }}
.muted {{ color:var(--n7); }}
.yr {{ border-top:1px solid #e9eded; padding-top:6px; margin-top:26px; }}
.yr dl {{ display:grid; grid-template-columns:230px 1fr; gap:2px 16px; margin:8px 0 12px; font-size:13px; }}
.yr dt {{ color:var(--n7); }} .yr dd {{ margin:0; color:var(--n8); }}
figure {{ margin:12px 0 4px; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid #e2e7e7; border-radius:4px; }}
.key {{ background:var(--n05); border:1px solid #e7ecec; border-radius:5px; padding:14px 16px; margin:18px 0 6px; font-size:13.5px; }}
.key b {{ color:var(--n9); }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero hero-sub">
    <p class="crumb"><a href="../">Somalia Riverine Flood Trigger</a> / timing of activations</p>
    <h1>Timing of activations</h1>
    <p>Dates on which the trigger is met in each season, shown alongside the river gauges and
       flood exposure in the 14 anticipatory-action districts. Activation runs on GloFAS in Deyr and
       Google Flood Hub in Gu. Either river can meet the trigger.</p>
  </header>
  <article>
    <div class="key">
      <p style="margin:0 0 6px"><b>How to read the charts.</b> Shaded bands are the window's own activation
      source, discharge stacked by station. Lines are observed SWALIM river levels on the right axis,
      with each gauge's own 1-in-3 dashed. The lower panel is flood exposure across the 14
      anticipatory-action districts. Red marks the activation.</p>
      <p style="margin:0 0 6px"><b>What each marker is.</b> The forecast date is the first issue on which
      the activation rule is met at leads 1 to 7 days. The reanalysis date is the first day the same
      rule is met in the model's historical run. The two-gauge benchmark onset is the day the second
      gauge on a river reaches its own 1-in-3 level.</p>
      <p style="margin:0"><b>Reading the exposure panel.</b> FloodScan derives flood extent from
      satellite observations, and single-day spikes that fall back immediately can occur. Exposure
      sustained over several days is the more reliable signal.</p>
    </div>

    <h2>Deyr</h2>
    <p>Activation reads {SOURCE[('deyr')]} at leads 1 to 7 days, three of four points on the Juba and
       two of three on the Shabelle, each against its own 1-in-4. Forecast archive: {ARCH['deyr']}.</p>
    {parts['deyr'][2]}
{parts['deyr'][1]}

    <h2>Gu</h2>
    <p>Activation reads {SOURCE[('gu')]} at leads 1 to 7 days, three of four points on the Juba against
       its own 1-in-5 and two of three on the Shabelle against its own 1-in-6. Forecast archive: {ARCH['gu']},
       so seasons before 2016 show the reanalysis, gauges and exposure only.</p>
    {parts['gu'][2]}
{parts['gu'][1]}
  </article>
</div>
</body>
</html>"""
    open(OUT + "index.html", "w", encoding="utf-8").write(html)
    n = sum(len(parts[s][0]) for s in parts)
    print(f"wrote {OUT}index.html with {n} seasons")


if __name__ == "__main__":
    build()
