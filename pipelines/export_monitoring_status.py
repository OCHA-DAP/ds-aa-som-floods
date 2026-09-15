"""Step 4 of the daily run: write status.json + latest.png for the public page.

Re-uses evaluate.evaluate on the latest stored day, so the dashboard can
never disagree with the last email. Output goes to STATUS_OUT_DIR (default
pages/monitoring/), which the workflow pushes to the orphan
`monitoring-status` branch; deploy-pages.yml overlays that branch onto the
site so /monitoring/ reads it next to index.html.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd  # noqa: E402

from src.constants import TRIGGER_STATIONS  # noqa: E402
from src.monitoring import config as cfg  # noqa: E402
from src.monitoring import etl, evaluate, plot  # noqa: E402
from src.monitoring import thresholds as thr  # noqa: E402

OUT_DIR = Path(os.environ.get("STATUS_OUT_DIR", cfg.STATUS_DIR))


def series_for_page(df):
    """Compact per-point forecast series for the dashboard's own rendering."""
    out = {}
    for (source, st), g in df.groupby(["source", "station"]):
        g = g.sort_values("valid_date")
        out.setdefault(source, {})[st] = {
            "issued": pd.Timestamp(g.issued_time.iloc[0]).strftime("%Y-%m-%dT%H:%MZ"),
            "dates": [d.strftime("%Y-%m-%d") for d in g.valid_date],
            "leads": [int(x) for x in g.leadtime_days],
            "value": [round(float(x), 1) for x in g.value],
            "p25": [None if pd.isna(x) else round(float(x), 1) for x in g.value_p25],
            "p75": [None if pd.isna(x) else round(float(x), 1) for x in g.value_p75],
        }
    return out


def levels_for_page(levels_df):
    out = {}
    for w in cfg.WINDOWS:
        river, season = w
        stations = TRIGGER_STATIONS[river]
        a, r = cfg.ACTION_RULES[w], cfg.READINESS_RULES[w]
        out[cfg.WINDOW_KEY[w]] = {
            "action": thr.lookup(levels_df, cfg.threshold_source(a["source"]), season, a["rp"], stations),
            "readiness": thr.lookup(levels_df, cfg.GLOFAS_OPERATIONAL, season, r["rp"], stations, basis="readiness_band"),
        }
    return out


def main():
    latest = etl.latest_monitoring_date()
    df = etl.load_day(latest)
    levels_df = thr.load()
    result = evaluate.evaluate(df, latest, levels_df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart = plot.load_chart(latest)
    chart_stale = chart is None
    if chart is not None:
        (OUT_DIR / "latest.png").write_bytes(chart)
    elif not (OUT_DIR / "latest.png").exists():
        print("  warning: no chart in blob and no previous latest.png")
    status = {
        **result,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chart": "latest.png", "chart_stale": chart_stale,
        "glofas_operational": cfg.GLOFAS_OPERATIONAL,
        "lead_bands": {"action": list(cfg.ACTION_LEADS), "readiness": list(cfg.READINESS_LEADS)},
        "levels": levels_for_page(levels_df),
        "series": series_for_page(df),
        "stations": {r: list(s) for r, s in TRIGGER_STATIONS.items()},
        "titles": {"station": cfg.STATION_TITLE, "river": cfg.RIVER_TITLE, "source": cfg.SOURCE_TITLE,
                   "season": cfg.SEASON_TITLE},
        "swalim_note": ("Readiness may also be activated by a SWALIM moderate flood risk alert for "
                        "either river. SWALIM bulletins are not read automatically."),
    }
    (OUT_DIR / "status.json").write_text(json.dumps(status, indent=1, default=str) + "\n")
    print(f"wrote {OUT_DIR / 'status.json'} for {latest}: {result['status']}; chart_stale={chart_stale}")


if __name__ == "__main__":
    main()
