"""Fail if any published page disagrees with src/constants.py.

Reads the VISIBLE text of each page, with data-alt attributes and scripts
stripped, so a value hidden in an attribute cannot pass the check. Run it
before opening a pull request that touches pages/ or the trigger config:

    .venv/Scripts/python.exe scripts/check_pages_match_config.py
"""
import html
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.constants import TRIGGER_CONFIG  # noqa: E402
from src.monitoring.config import ACTION_LEADS, READINESS_LEADS  # noqa: E402

RP = {(r, s): c["rp"] for (r, s), c in TRIGGER_CONFIG.items()}
# pages name a window either way round, so both spellings are checked
LABEL = {k: (f"{k[1].title()} {k[0].title()}", f"{k[0].title()} {k[1].title()}")
         for k in TRIGGER_CONFIG}
PAGES = REPO / "pages"
failures = []


def visible(path):
    t = path.read_text(encoding="utf-8", errors="replace")
    t = re.sub(r'data-alt2?="[^"]*"', " ", t)
    t = re.sub(r"<(script|style)\b.*?</\1>", " ", t, flags=re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", t)).split())


def fail(where, msg):
    failures.append(f"{where}: {msg}")


# every page that states a window's rule must state the configured return period
for page in sorted(PAGES.rglob("*.html")):
    rel = page.relative_to(REPO).as_posix()
    if "/temp/" in rel:
        continue
    text = visible(page)
    for key, rp in RP.items():
        # only the sentences that state the RULE, never an activation rate or a
        # gauge benchmark level, which are different numbers by design
        names = "|".join(re.escape(n) for n in LABEL[key])
        for pat in (rf"(?:{names})[^.]{{0,90}}?points over their 1-in-(\d+)-yr",
                    rf"(?:{names})[^.]{{0,90}}?gauges forecast over their own 1-in-(\d+) level",
                    rf"(?:{names})[^.]{{0,40}}?(?:GloFAS v5|Google GRRR) 1-in-(\d+) \d of \d"):
            for m in re.finditer(pat, text):
                if int(m.group(1)) != rp:
                    fail(rel, f"{LABEL[key][0]} rule stated as 1-in-{m.group(1)}, "
                              f"config says 1-in-{rp}")
    # the lead bands are configuration, not prose
    stale = f"{READINESS_LEADS[0] - 1}-{READINESS_LEADS[1]}"
    for bad in (f"{stale} d", f"{stale.replace('-', chr(8211))} d",
                f"{READINESS_LEADS[0] - 1} to {READINESS_LEADS[1]} days"):
        if bad in text:
            fail(rel, f"readiness band written as {bad}, config says "
                      f"{READINESS_LEADS[0]} to {READINESS_LEADS[1]}")

# the live status must carry the configured rule on both legs
status = PAGES / "monitoring" / "status.json"
if status.exists():
    s = json.loads(status.read_text(encoding="utf-8"))
    for name, w in s.get("windows", {}).items():
        key = (w["river"], w["season"])
        if w["action"]["rp"] != RP[key]:
            fail("status.json", f"{name} action rp{w['action']['rp']}, config rp{RP[key]}")
        if tuple(w["readiness"]["leads"]) != tuple(READINESS_LEADS):
            fail("status.json", f"{name} readiness leads {w['readiness']['leads']}, "
                                f"config {list(READINESS_LEADS)}")
        if tuple(w["action"]["leads"]) != tuple(ACTION_LEADS):
            fail("status.json", f"{name} action leads {w['action']['leads']}, "
                                f"config {list(ACTION_LEADS)}")

# emails state the rule in words and must not print a return period
for tpl in sorted((REPO / "src/monitoring/email/templates").glob("*.html")):
    if "1-in-" in tpl.read_text(encoding="utf-8"):
        fail(tpl.name, "prints a return period; emails state the rule without one")

for f in failures:
    print("FAIL " + f)
print(f"\n{len(failures)} disagreement(s) with src/constants.py" if failures
      else "\nevery page agrees with src/constants.py")
sys.exit(1 if failures else 0)
