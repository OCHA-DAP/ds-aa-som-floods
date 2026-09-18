"""Step 3 of the daily run: send the day's email through Listmonk.

Cadence: while at least one window is open, an informational email every
Monday and an immediate email the day a readiness or activation trigger is
reached. When no window is open nothing is sent. Campaigns render inside the Listmonk
instance's base_campaign template (branding, footer, unsubscribe); the Jinja
templates here are the content fragment, and the chart is hosted in the
Listmonk media library.

Flags (src/monitoring/flags.py): TEST_EMAIL routes to som:test and tags
the campaign name [test] (the template's red banner); DRY_RUN renders but
does not upload or send; SIMULATE_TRIGGER forces an action activation on
the first open window (or Deyr Shabelle) and tags [SIM]. A simulation to a
real list additionally needs ALLOW_REAL_SIMULATION=true.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402
from ocha_relay.listmonk import ListmonkClient  # noqa: E402

from src.monitoring import config as cfg  # noqa: E402
from src.monitoring import etl, evaluate, plot  # noqa: E402
from src.monitoring.flags import env_flag, mode  # noqa: E402

TEMPLATES = ROOT / "src" / "monitoring" / "email" / "templates"
DESIGN_URL = cfg.PAGES_URL + "trigger-single-model/summary.html"
DASHBOARD_URL = cfg.PAGES_URL + "monitoring/"
SOURCE_URL = {"glofas": "https://global-flood.emergency.copernicus.eu/", "google": "https://sites.research.google/floods/"}


def resolve_list_id(client, list_type):
    tag = cfg.LISTMONK_LISTS[list_type]["tag"]
    for lst in client.fetch_all_lists(tag=cfg.LISTMONK_PROJECT_TAG):
        if tag in lst.get("tags", []):
            return lst["id"]
    raise RuntimeError(f"No Listmonk list tagged {tag!r}; run pipelines/setup_som_listmonk_lists.py")


def simulate(result):
    key = result["open_windows"][0] if result["open_windows"] else "shabelle_deyr"
    w = result["windows"][key]
    w["open"] = True
    leg = w["action"]
    leg["activated"], leg["max_votes"] = True, leg["n_req"]
    leg["max_votes_date"] = leg["max_votes_date"] or result["monitoring_date"]
    for st in list(leg["stations"])[: leg["n_req"]]:
        v = leg["stations"][st]
        v.update({"exceeds": True, "max_value": v["max_value"] or v["threshold"] * 1.2,
                  "max_date": v["max_date"] or result["monitoring_date"],
                  "pct_of_threshold": max(v["pct_of_threshold"] or 0, 120.0), "reporting": True})
    result.update({"open_windows": sorted(set(result["open_windows"]) | {key}),
                   "action": True, "action_windows": [key], "status": "ACTIVATION TRIGGER REACHED"})
    return result


def _long_date(x):
    """2026-09-17 -> 17 September 2026 (no platform-specific %-d)."""
    if not x:
        return None
    d = pd.Timestamp(str(x)[:10])
    return f"{d.day} {d:%B %Y}"


def _days_from(valid, issue):
    """'8 days' / '1 day' between a forecast issue and the day it describes."""
    if not valid or not issue:
        return None
    n = (pd.Timestamp(str(valid)[:10]) - pd.Timestamp(str(issue)[:10])).days
    return f"{n} day" + ("" if n == 1 else "s")


def render(result, template_name, chart_url):
    env = Environment(loader=FileSystemLoader(TEMPLATES))
    env.filters["longdate"] = _long_date
    env.filters["daysfrom"] = _days_from
    open_sources = []                      # the action source of each open window, then GloFAS (readiness)
    for k in result["open_windows"]:
        src = result["windows"][k]["action"]["source"]
        if src not in open_sources:
            open_sources.append(src)
    if "glofas" not in open_sources:
        open_sources.append("glofas")
    ctx = dict(
        pub_date=result["monitoring_date"], trigger_status=result["status"], chart_url=chart_url,
        glofas_issue=result.get("glofas_issue"),
        season_title={k: cfg.SEASON_TITLE[v["season"]].split(" (")[0] for k, v in result["windows"].items()},
        google_issue=(result.get("google_issue") or "")[:10] if any(result["windows"][k]["action"]["source"] == "google" for k in result["open_windows"]) else None,
        windows=result["windows"], open_windows=result["open_windows"],
        open_titles=[result["windows"][k]["title"] for k in result["open_windows"]],
        action_windows=result["action_windows"],
        action_titles=[result["windows"][k]["title"] for k in result["action_windows"]],
        readiness_windows=result["readiness_windows"],
        readiness_titles=[result["windows"][k]["title"] for k in result["readiness_windows"]],
        source_title=cfg.SOURCE_TITLE, river_title=cfg.RIVER_TITLE, station_title=cfg.STATION_TITLE,
        open_sources=open_sources, source_url=SOURCE_URL,
        open_season=" and ".join(sorted({result["windows"][k]["season"].capitalize() for k in result["open_windows"]})) or "no season",
        contact_name=cfg.CONTACT_NAME, contact_email=cfg.CONTACT_EMAIL,
        dashboard_url=DASHBOARD_URL, repo_url=cfg.REPO_URL, design_url=DESIGN_URL,
    )
    return env.get_template(f"{template_name}.html").render(**ctx)


def main():
    flags = mode()
    monitoring_date = etl.monitoring_date_from_env()
    df = etl.load_day(monitoring_date)
    result = evaluate.evaluate(df, monitoring_date)
    if flags["SIMULATE_TRIGGER"]:
        if not flags["TEST_EMAIL"] and not flags["DRY_RUN"] and not env_flag("ALLOW_REAL_SIMULATION", False):
            raise SystemExit("Refusing to send a simulated activation to a real list without ALLOW_REAL_SIMULATION=true")
        result = simulate(result)
    print(f"{monitoring_date}: {result['status']}; open {result['open_windows']}")

    if not result["open_windows"]:
        print("no window open: no email")
        return
    is_monday = monitoring_date.weekday() == 0

    # each trigger is announced once per season. Readiness is also skipped when activation has
    # already been reached, whether today or earlier: the activation email supersedes it.
    season = result["windows"][result["open_windows"][0]]["season"]
    year = etl.season_year(monitoring_date, season)
    notified = etl.load_notified(season, year)
    leg = None
    if result["action"] and not notified.get("action"):
        template, email_type, leg = "action", "trigger", "action"
    elif result["readiness"] and not (notified.get("readiness") or notified.get("action") or result["action"]):
        template, email_type, leg = "readiness", "info", "readiness"
    elif is_monday:
        template, email_type = "informational", "info"
    else:
        for name in ("action", "readiness"):
            if result[name] and notified.get(name):
                print(f"{name} already announced this {season} season on {notified[name]}: no email today")
                return
        if result["readiness"] and notified.get("action"):
            print(f"activation already announced on {notified['action']}: no readiness email")
            return
        print("no new trigger and not Monday: no email today")
        return

    tags = ("[TEST] " if flags["TEST_EMAIL"] else "") + ("[SIM] " if flags["SIMULATE_TRIGGER"] else "")
    headline = result["status"].capitalize()
    # the subject carries the day the email goes out; the forecast issue date is in the body
    subject = f"{tags}{cfg.EMAIL_SUBJECT_PREFIX} - {headline} | {_long_date(datetime.now(timezone.utc).date())}"
    name = (f"{cfg.LISTMONK_PROJECT_TAG} {template} {monitoring_date} "
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M}"
            + (" [test]" if flags["TEST_EMAIL"] else "") + (" [sim]" if flags["SIMULATE_TRIGGER"] else ""))

    if flags["DRY_RUN"]:
        body = render(result, template, chart_url="chart.png")
        out = Path("temp"); out.mkdir(exist_ok=True)
        (out / f"email_{template}_{monitoring_date}.html").write_text(body, encoding="utf-8")
        print(f"DRY_RUN: {template} email rendered to temp/, subject {subject!r}; not sent")
        return

    client = ListmonkClient.from_env()
    chart_url = None
    if template != "action":
        chart = plot.load_chart(monitoring_date)
        if chart is None:
            raise RuntimeError("chart not in blob; run save_plots.py first")
        chart_url = client.upload_media(chart, f"som_flood_monitoring_{monitoring_date}.png")
    body = render(result, template, chart_url)
    list_id = resolve_list_id(client, "test" if flags["TEST_EMAIL"] else email_type)
    campaign_id = client.create_campaign(name=name, subject=subject, body=body, list_ids=[list_id])
    client.send_campaign(campaign_id, skip_confirmation=True)
    print(f"sent campaign {campaign_id} ({name}) to list {list_id}")
    if leg and not flags["TEST_EMAIL"] and not flags["SIMULATE_TRIGGER"]:
        etl.record_notified(season, year, leg, monitoring_date)
        print(f"recorded {leg} as announced for the {season} {year} season")


if __name__ == "__main__":
    main()
