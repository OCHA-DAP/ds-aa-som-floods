"""Step 2 of the daily run: evaluate the windows, chart the day, and keep the
evaluation on blob."""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.monitoring import etl, evaluate, plot  # noqa: E402
from src.monitoring.flags import env_flag  # noqa: E402


def main():
    monitoring_date = etl.monitoring_date_from_env()
    df = etl.load_day(monitoring_date)
    result = evaluate.evaluate(df, monitoring_date)
    print(f"{monitoring_date}: {result['status']}; open windows {result['open_windows']}")
    fig = plot.monitoring_chart(df, result)
    if env_flag("DRY_RUN", True):
        out = Path("temp"); out.mkdir(exist_ok=True)
        fig.savefig(out / f"{monitoring_date}.png", dpi=150, bbox_inches="tight")
        print(f"DRY_RUN: chart written to temp/{monitoring_date}.png, not to blob")
    else:
        plot.save_chart(fig, monitoring_date)
        print(f"chart uploaded to {plot.chart_blob_name(monitoring_date)}")
        print(f"evaluation saved to blob {etl.save_status(result, monitoring_date)}")


if __name__ == "__main__":
    main()
