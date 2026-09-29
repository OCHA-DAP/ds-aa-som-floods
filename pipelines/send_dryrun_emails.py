"""Send the simulated readiness and activation emails to one chosen list, as a dry run.

Both legs are simulated exactly as pipelines/render_dryrun_emails.py does, from the
stored forecast of MONITORING_DATE (default: the latest stored day). Campaigns are
tagged [TEST] in the subject and [test] in the name (the template's test banner).
No notification state is written. The live list (config.LIVE_LIST_IDS) is refused.

Usage (from repo root):
    MONITORING_DATE=2026-09-28 .venv/Scripts/python.exe pipelines/send_dryrun_emails.py <list_id> [readiness|action|both]
"""

import copy, datetime as dt, os, sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, "."); sys.path.insert(0, "pipelines")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from ocha_relay.listmonk import ListmonkClient  # noqa: E402

import send_emails as se  # noqa: E402
from src.monitoring import config as cfg, etl, evaluate, plot  # noqa: E402


def main():
    list_id = int(sys.argv[1])
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    if list_id in cfg.LIVE_LIST_IDS:
        raise SystemExit(f"Refusing: list {list_id} is the live list and never receives a dry run")

    raw = os.environ.get("MONITORING_DATE", "").strip()
    d = dt.date.fromisoformat(raw) if raw else etl.latest_monitoring_date()
    readiness_day = str(d + dt.timedelta(days=12))
    action_day = str(d + dt.timedelta(days=7))
    base = evaluate.evaluate(etl.load_day(d), d)

    rd = copy.deepcopy(base); k = rd["open_windows"][0]; leg = rd["windows"][k]["readiness"]
    leg.update({"activated": True, "max_votes": leg["n_req"], "max_votes_date": readiness_day})
    for st in list(leg["stations"])[: leg["n_req"]]:
        leg["stations"][st].update({"exceeds": True, "max_value": leg["stations"][st]["threshold"] * 1.12,
                                    "max_date": readiness_day, "pct_of_threshold": 112.0})
    rd.update({"readiness": True, "readiness_windows": [k], "status": "READINESS TRIGGER REACHED"})

    ac = se.simulate(copy.deepcopy(base))
    for w in ac["windows"].values():
        if w["action"]["activated"]:
            w["action"]["max_votes_date"] = action_day
            for v in w["action"]["stations"].values():
                if v["exceeds"]:
                    v["max_date"] = action_day

    client = ListmonkClient.from_env()
    lists = {l["id"]: l for l in client.fetch_all_lists()}
    target = lists[list_id]
    print(f"target list {list_id}: {target['name']} ({target['subscriber_count']} subscribers)")

    chart_url = None
    if which in ("readiness", "both"):
        chart = plot.load_chart(d)
        if chart is None:
            raise RuntimeError("chart not in blob for that day")
        chart_url = client.upload_media(chart, f"som_flood_dryrun_{d}.png")

    stamp = dt.datetime.now(dt.timezone.utc)
    for name, res in (("readiness", rd), ("action", ac)):
        if which not in (name, "both"):
            continue
        body = se.render(res, name, chart_url)
        subject = f"[TEST] {cfg.EMAIL_SUBJECT_PREFIX} - {res['status'].capitalize()} | {se._long_date(stamp.date())}"
        cname = f"{cfg.LISTMONK_PROJECT_TAG} dryrun {name} {d} {stamp:%Y%m%dT%H%M} [test] [sim]"
        cid = client.create_campaign(name=cname, subject=subject, body=body, list_ids=[list_id])
        client.send_campaign(cid, skip_confirmation=True)
        print(f"sent {name} campaign {cid} to list {list_id}: {subject}")


if __name__ == "__main__":
    main()
