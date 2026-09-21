"""One-off, idempotent: create the three Somalia Listmonk lists and seed them.

Lists are found by tag (ds-aa-som-floods + som:info / som:trigger /
som:test) and created only if missing. Subscribers passed with
--subscribe are created if unknown (attribs.type = mailing_list, which
suppresses the unsubscribe link; preconfirmed) and added to every list.
Needs the admin credentials (DSCI_LISTMONK_ADMIN_API_USERNAME / _KEY): the
send-scoped key cannot write subscribers.

    python pipelines/setup_som_listmonk_lists.py --dry-run
    python pipelines/setup_som_listmonk_lists.py --subscribe a@un.org b@un.org
"""

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from ocha_relay.listmonk import ListmonkClient  # noqa: E402

from src.monitoring import config as cfg  # noqa: E402


def admin_client():
    return ListmonkClient(base_url=os.environ["DSCI_LISTMONK_BASE_URL"].rstrip("/"),
                          username=os.environ["DSCI_LISTMONK_ADMIN_API_USERNAME"],
                          password=os.environ["DSCI_LISTMONK_ADMIN_API_KEY"])


def http():
    s = requests.Session()
    s.auth = (os.environ["DSCI_LISTMONK_ADMIN_API_USERNAME"], os.environ["DSCI_LISTMONK_ADMIN_API_KEY"])
    return s, os.environ["DSCI_LISTMONK_BASE_URL"].rstrip("/")


def resolve_or_create(client, dry_run):
    have = {}
    tagged = {k: c for k, c in cfg.LISTMONK_LISTS.items() if "tag" in c}
    for lst in client.fetch_all_lists(tag=cfg.LISTMONK_PROJECT_TAG):
        for cfg_ in tagged.values():
            if cfg_["tag"] in lst.get("tags", []):
                have[cfg_["tag"]] = lst["id"]
    for key, c in cfg.LISTMONK_LISTS.items():
        if "id" in c:                     # a list given by id is never created or tagged here
            print(f"  {key}: fixed id {c['id']} ({c['name']})")
            continue
        if c["tag"] in have:
            print(f"  {key}: exists (id {have[c['tag']]})")
            continue
        tags = [cfg.LISTMONK_PROJECT_TAG, c["tag"], *c.get("extra_tags", [])]
        if dry_run:
            print(f"  {key}: would create {c['name']!r} tags {tags}")
            continue
        have[c["tag"]] = client.create_list(name=c["name"], tags=tags)
        print(f"  {key}: created id {have[c['tag']]}")
    return have


def subscribe(emails, list_ids, dry_run):
    s, base = http()
    for email in emails:
        r = s.get(f"{base}/subscribers", params={"query": f"subscribers.email = '{email}'"}).json()
        hits = r["data"]["results"]
        if dry_run:
            print(f"  {email}: {'exists' if hits else 'would create'}; would add to {list_ids}")
            continue
        if hits:
            sid = hits[0]["id"]
            s.put(f"{base}/subscribers/lists", json={"ids": [sid], "action": "add",
                                                    "target_list_ids": list_ids, "status": "confirmed"}).raise_for_status()
            print(f"  {email}: added existing subscriber {sid} to {list_ids}")
        else:
            r = s.post(f"{base}/subscribers", json={"email": email, "name": email.split("@")[0],
                                                   "status": "enabled", "lists": list_ids,
                                                   "attribs": {"type": "mailing_list"},
                                                   "preconfirm_subscriptions": True})
            r.raise_for_status()
            print(f"  {email}: created and subscribed to {list_ids}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--subscribe", nargs="*", default=[], help="emails to add to all three lists")
    args = ap.parse_args()
    ids = resolve_or_create(admin_client(), args.dry_run)
    if args.subscribe:
        subscribe(args.subscribe, sorted(ids.values()), args.dry_run)
