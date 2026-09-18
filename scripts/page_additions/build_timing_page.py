"""Build pages/activation-timing/: the dates the trigger is met in each season, with the
river gauges and flood exposure. Deyr first, then Gu.

Usage (from repo root; set SOM_DATA_REPO when running from a worktree):
    .venv/Scripts/python.exe scripts/page_additions/build_timing_page.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import season_charts as sc  # noqa: E402
import somlib as L  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "pages" / "activation-timing"
ARCH = {"deyr": "2003 to 2023, about two issues a week", "gu": "2016 to 2023, daily"}
f = lambda d: d.strftime("%d %b") if d is not None else None


def collect(season):
    out = []
    for y in sc.seasons_of_interest(season):
        png, acts, (fd, nd), bench = sc.chart(season, y)
        first = [d for d, _, _ in acts.values() if d is not None]
        out.append(dict(year=y, acts=acts, bench=bench, flood=fd, nd=nd,
                        fig=f"figs/{png.name}", first=min(first) if first else None,
                        has_arch=all(h for _, _, h in acts.values())))
    return out


def sections(season, data):
    out = []
    for r in sorted(data, key=lambda x: -x["year"]):
        a = r["acts"]
        act = ", ".join(f"{riv.title()} {f(a[riv][0])}" for riv in sc.RIVERS if a[riv][0] is not None) or "no activation"
        b = ", ".join(f"{k.title()} {f(v)}" for k, v in r["bench"].items()) or "fewer than two gauges crossed"
        out.append(f"""<section class="yr" id="{season}{r['year']}">
<h3>{sc.SEASON_TITLE[season]} {r['year']}</h3>
<dl><dt>Activation</dt><dd>{act}</dd>
<dt>Two-gauge benchmark onset</dt><dd>{b}</dd></dl>
<figure><img src="{r['fig']}" alt="{sc.SEASON_TITLE[season]} {r['year']}" loading="lazy"></figure>
</section>""")
    return "\n".join(out)


def summary(season, data):
    leads = [(r["flood"] - r["first"]).days for r in data if r["first"] is not None and r["flood"] is not None]
    bl = [(list(r["bench"].values())[0] - r["first"]).days for r in data if r["first"] is not None and r["bench"]]
    fired = sum(1 for r in data if r["first"] is not None)
    arch = sum(1 for r in data if r["has_arch"])
    flooded = sum(1 for r in data if r["flood"] is not None)
    caught = sum(1 for r in data if r["first"] is not None and r["flood"] is not None)
    med = lambda xs: f"{np.median(xs):+.0f} d" if xs else "&mdash;"
    return (f"<p>Seasons shown: {len(data)}, of which {arch} have a forecast archive. The trigger is met in "
            f"{fired}. The three-district level is reached in {flooded}, and in {caught} of those the trigger "
            f"is also met. Median days from the trigger being met to that date: {med(leads)}. To the second "
            f"gauge reaching its 1-in-3: {med(bl)}.</p>")


def build():
    parts = {s: (lambda d: (d, sections(s, d), summary(s, d)))(collect(s)) for s in ("deyr", "gu")}
    deyr_src = sc.SOURCE_TITLE[sc.rule("juba", "deyr")[0]]
    gu_src = sc.SOURCE_TITLE[sc.rule("juba", "gu")[0]]
    rp = {(riv, se): sc.rule(riv, se)[1] for riv in ("juba", "shabelle")
          for se in ("deyr", "gu")}
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Timing of activations &mdash; Somalia Riverine Flood Trigger</title>
<meta name="description" content="Dates on which the Somalia riverine flood trigger is met in each season, with the river gauges and flood exposure in the 14 anticipatory-action districts. Deyr and Gu.">
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
       flood exposure in the 14 anticipatory-action districts. Activation runs on GloFAS in Deyr
       and Google Flood Hub in Gu. Either river can meet the trigger.</p>
  </header>
  <article>
    <div class="key">
      <p style="margin:0 0 6px"><b>How to read the charts.</b> Filled areas are the window's own
      activation source, each station as a share of its own return-period level. Lines are the
      observed SWALIM river levels, each as a share of that gauge's own 1-in-3. The dashed line is
      100%, so anything above it is over its own level. The lower panel is flood exposure across
      the 14 anticipatory-action districts. Red marks the activation.</p>
      <p style="margin:0 0 6px"><b>What each marker is.</b> The forecast date is the first issue on
      which the activation rule is met at leads 1 to 7 days. The reanalysis date is the first day the
      same rule is met in the model's historical run. The two-gauge benchmark onset is the day the
      second gauge on a river reaches its own 1-in-3 level. The three-district level is the day the
      third of the 14 districts reaches its own 1-in-5 seasonal flood exposure, exposure being
      FloodScan flood extent combined with WorldPop population, and each district's level fitted by
      Weibull plotting position on its own seasonal maxima.</p>
      <p style="margin:0"><b>Reading the exposure panel.</b> FloodScan derives flood extent from
      satellite observations, and single-day spikes that fall back immediately can occur. Exposure
      sustained over several days is the more reliable signal.</p>
    </div>

    <h2>Deyr</h2>
    <p>Activation reads {deyr_src} at leads 1 to 7 days, three of four points on the Juba
       against its own 1-in-{rp[('juba', 'deyr')]} and two of three on the Shabelle against its
       own 1-in-{rp[('shabelle', 'deyr')]}. Forecast archive: {ARCH['deyr']}.</p>
    <p>The Deyr rules were calibrated on the GloFAS version 5 reanalysis, and the operational system
       has been version 4 since July 2023. The dates here are the version 4 ones, which is what the
       live system reads, so the count of Deyr activations differs from the
       <a href="../trigger-single-model/summary.html">trigger summary</a>, which reports the version 5
       figures. <a href="../glofas-version/">GloFAS version switch</a> sets out the difference.</p>
    {parts['deyr'][2]}
{parts['deyr'][1]}

    <h2>Gu</h2>
    <p>Activation reads {gu_src} at leads 1 to 7 days, three of four points on the Juba against
       its own 1-in-{rp[('juba', 'gu')]} and two of three on the Shabelle against its own
       1-in-{rp[('shabelle', 'gu')]}. Forecast archive:
       {ARCH['gu']}, so seasons before 2016 show the reanalysis, gauges and exposure only.</p>
    {parts['gu'][2]}
{parts['gu'][1]}
  </article>
</div>
</body>
</html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} with {sum(len(parts[s][0]) for s in parts)} seasons")


if __name__ == "__main__":
    build()
