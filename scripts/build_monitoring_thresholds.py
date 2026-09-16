"""Freeze the per-station trigger thresholds to src/monitoring/thresholds.json.

Fits every candidate climatology (Google retrospective, GloFAS v4 and v5
reanalyses; GloFAS v4 readiness-band reforecast) at RP 3-6 for the seven
trigger points and both seasons, using the estimator from the trigger
analysis. Re-run after the processed parquets change; the monitoring
pipeline reads only the JSON.

    .venv/bin/python scripts/build_monitoring_thresholds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring import thresholds  # noqa: E402

if __name__ == "__main__":
    df = thresholds.build()
    thresholds.save(df)
    print(f"wrote {thresholds.JSON_PATH} ({len(df)} rows)")
    show = df[(df.rp.isin([4, 5])) & (df.season == "deyr") & (df.basis == "reanalysis")]
    print(show.pivot_table(index=["river", "station"], columns=["rp", "source"], values="threshold").round(0).to_string())
