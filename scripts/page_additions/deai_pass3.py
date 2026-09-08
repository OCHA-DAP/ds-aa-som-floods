"""Third prose pass: the last em dashes sit beside literal Unicode symbols."""
import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model" / "index.html"
D = r"(?:&mdash;|\u2014)"
PAIRS = [
    ("peak Spearman \u03c1 @ reanalysis", "peak Spearman \u03c1, reanalysis"),
    ("peak \u03c1 @ reforecast", "peak \u03c1, reforecast"),
    ("ratio \u2248 1.00 @ the licence for calibrating", "ratio about 1.00, which is what licenses calibrating"),
    ("(readiness band, shaded) @ its 7\u201312-day thresholds should anticipate that.",
     "(readiness band, shaded), so its 8-to-12-day thresholds should anticipate that."),
    ("(2024\u201326) @ retrospective-fitted thresholds", "(2024 to 2026), so retrospective-fitted thresholds"),
    ("per river \u00d7 season here @ the same recipe as", "per river and season here, the same recipe as"),
    ("only 8\u20139 seasons (gauge online 2015) @ read it loosely.",
     "only eight or nine seasons (gauge online 2015), so read it loosely."),
    ("The calibration record is reanalysis @ a model's afterwards-view of its own past.",
     "The calibration record is reanalysis, a model's afterwards-view of its own past."),
]


def pattern(old):
    return re.compile(r"\s+".join(re.escape(p) for p in old.split()).replace(re.escape("@"), D))


t = PAGE.read_text(encoding="utf-8")
applied, missing = 0, []
for old, new in PAIRS:
    rx = pattern(old)
    if rx.search(t):
        t = rx.sub(lambda m: new, t, count=1)
        applied += 1
    else:
        missing.append(old[:70])
PAGE.write_text(t, encoding="utf-8")
print("applied", applied)
if missing:
    print("NOT FOUND:", *missing, sep="\n   - ")
    sys.exit(1)
