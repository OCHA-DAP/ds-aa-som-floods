"""Two illustrative charts for the monitoring page's "How to read this chart" section:
what a readiness trigger looks like, and what an activation trigger looks like.

Built from one stored monitoring day with the Juba GloFAS values scaled up inside one lead
band, so the chart code, labels and bands are exactly the live ones. Each image carries an
"illustration" stamp and the figure date is blanked. Re-run after any change to
src/monitoring/plot.py so the examples match the live chart.

    .venv/Scripts/python.exe scripts/build_chart_examples.py [YYYY-MM-DD]
Output: pages/monitoring/examples/{readiness,activation}.png
"""
import datetime as dt
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring import config as cfg, etl, evaluate, plot  # noqa: E402

OUT = Path("pages/monitoring/examples")
JUBA = ["dollow", "luuq", "bardheere", "bualle"]


def example(df, day, leads, factor, name, hold=False):
    d = df.copy()
    # one smooth flood wave: a bell-shaped scaling that peaks at the factor mid-band and
    # fades to 1 outside it. With hold=True the flow stays high after the peak, so the
    # readiness band is over its level as well (an activation normally comes with that).
    centre = (leads[0] + leads[1]) / 2
    for lead in range(1, 13):
        x = min(lead, centre) if hold else lead
        f = 1 + (factor - 1) * math.exp(-((x - centre) ** 2) / (2 * 1.8 ** 2))
        m = (d.source == "glofas") & d.station.isin(JUBA) & (d.leadtime_days == lead)
        d.loc[m, "value"] = d.loc[m, "value"] * f
    res = evaluate.evaluate(d, day)
    res["glofas_issue"] = None                      # the header then says "no GloFAS forecast"
    fig = plot.monitoring_chart(d, res)
    fig.text(0.5, 0.5, "ILLUSTRATION", fontsize=54, color="#B34036", alpha=0.12, ha="center", va="center",
             rotation=20, fontweight="bold", transform=fig.transFigure)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=110, bbox_inches="tight")
    print(f"{name}: {res['status']}  (Juba scaled x{factor} at leads {leads[0]} to {leads[1]})")


def main():
    day = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else etl.latest_monitoring_date()
    df = etl.load_day(day)
    # readiness: Juba over its level only in the 8 to 12 day band
    example(df, day, cfg.READINESS_LEADS, 1.9, "readiness")
    # activation: Juba over its level inside the 1 to 7 day band and staying high after it
    example(df, day, cfg.ACTION_LEADS, 1.75, "activation", hold=True)


if __name__ == "__main__":
    main()
