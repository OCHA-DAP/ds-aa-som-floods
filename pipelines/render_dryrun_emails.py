"""Render the readiness and activation emails for a stored day, wrapped in the real Listmonk
template, so the wording can be reviewed before a trigger is ever met.

Both legs are simulated at the far end of their own lead band, from the latest stored forecast:
the forecast issue date, the send date and the subject date are the same day, as they are in a
real run. Previews are produced through draft campaigns that are deleted immediately; nothing is
sent and no notification state is written.

Usage (from repo root):
    .venv/Scripts/python.exe pipelines/render_dryrun_emails.py
Output: pages/temp/dryrun_{readiness,action}[_wrapped].html
"""

import copy, datetime as dt, os, re, sys, requests
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "pipelines")
import send_emails as se
from src.monitoring import etl, evaluate, config as cfg

raw = os.environ.get("MONITORING_DATE", "").strip()
d = dt.date.fromisoformat(raw) if raw else etl.latest_monitoring_date()
READINESS_DAY = str(d + dt.timedelta(days=12))  # far end of the 8-12 readiness band
ACTION_DAY = str(d + dt.timedelta(days=7))      # far end of the 1-7 activation band
base = evaluate.evaluate(etl.load_day(d), d)
rd = copy.deepcopy(base); k = rd["open_windows"][0]; leg = rd["windows"][k]["readiness"]
leg.update({"activated": True, "max_votes": leg["n_req"], "max_votes_date": READINESS_DAY})
for st in list(leg["stations"])[: leg["n_req"]]:
    v = leg["stations"][st]
    v.update({"exceeds": True, "max_value": v["max_value"] or v["threshold"] * 1.12,
              "max_date": v["max_date"] or READINESS_DAY, "pct_of_threshold": max(v["pct_of_threshold"] or 0, 112.0)})
rd.update({"readiness": True, "readiness_windows": [k], "status": "READINESS TRIGGER REACHED"})
ac = se.simulate(copy.deepcopy(base))
for w in ac["windows"].values():
    if w["action"]["activated"]:
        w["action"]["max_votes_date"] = ACTION_DAY
        for v in w["action"]["stations"].values():
            if v["exceeds"]:
                v["max_date"] = ACTION_DAY
b = (os.environ.get("DSCI_LISTMONK_API_URL") or os.environ.get("DSCI_LISTMONK_BASE_URL")).rstrip("/")
if not b.endswith("/api"):
    b += "/api"
auth = (os.environ["DSCI_LISTMONK_API_USERNAME"], os.environ["DSCI_LISTMONK_API_KEY"])
lists = requests.get(f"{b}/lists", params={"query": "Somalia", "per_page": 50}, auth=auth, timeout=60).json()["data"]["results"]
test = [l for l in lists if "som:test" in l["tags"]][0]["id"]
today = se._long_date(d)   # the run sends on the day it reads the forecast
made = []
try:
    for name, res, chart in (("readiness", rd, "../monitoring/latest.png"), ("action", ac, None)):
        body = se.render(res, name, chart)
        Path(f"pages/temp/dryrun_{name}.html").write_text(
            '<meta charset="utf-8"><body style="background:#fff;margin:0;padding:24px;max-width:760px">'
            + body + "</body>", encoding="utf-8")
        subj = f"[TEST] {cfg.EMAIL_SUBJECT_PREFIX} - {res['status'].capitalize()} | {today}"
        r = requests.post(f"{b}/campaigns", json={"name": f"[test] dry run {name}", "subject": subj,
                          "lists": [test], "type": "regular", "content_type": "html",
                          "body": body, "template_id": 8}, auth=auth, timeout=60)
        r.raise_for_status()
        cid = r.json()["data"]["id"]; made.append(cid)
        Path(f"pages/temp/dryrun_{name}_wrapped.html").write_text(
            requests.get(f"{b}/campaigns/{cid}/preview", auth=auth, timeout=60).text, encoding="utf-8")
        print(f"{name}: {subj}")
finally:
    for cid in made:
        requests.delete(f"{b}/campaigns/{cid}", auth=auth, timeout=60)
    print("drafts deleted")
