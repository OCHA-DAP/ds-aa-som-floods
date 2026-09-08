"""Fill the explanation gaps the flow audit found: acronyms expanded at first use,
the envelope defined where it is first met, the valid-day and issue-date distinction
glossed where it is first used, and the GloFAS v4 contradiction resolved.

Whitespace-tolerant and asserted. A pair may carry a third element, an anchor: a short
phrase that proves the edit has landed even if a later pair has since reworded the rest.
Without it the whole replacement string has to match, which fails when two pairs touch
the same sentence."""
import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model" / "index.html"

PAIRS = [
    # the hero summary: one dense clause replaced by what the rule actually does
    ("Proposed trigger for anticipatory action against riverine flooding on the Juba and Shabelle: a station consensus in which each river-season window runs on a single forecast source rather than a mixture, monitoring all seven points, with no threshold below 1-in-3. Calibrated against SWALIM river gauges. August 2026.",
     "Proposed trigger for anticipatory action against riverine flooding on the Juba and "
     "Shabelle. Each of the four river-season windows runs on one forecast model, not a "
     "mixture. A window activates when enough of the seven monitored points are over their "
     "own return-period thresholds on the same day, each threshold set at 1-in-3 or rarer. "
     "Calibrated against SWALIM river gauges. August 2026.",
     "Each of the four river-season windows runs on one forecast model"),
    # reframe the threshold floor as a positive statement (was: "no threshold sits below")
    ("on the same day, and no threshold sits below 1-in-3.",
     "on the same day, each threshold set at 1-in-3 or rarer."),
    # the meta description should say the same thing
    ("Proposed anticipatory-action trigger for the Juba and Shabelle rivers: multi-model forecast consensus, calibration against SWALIM gauges, backtests and return periods.",
     "Proposed anticipatory-action trigger for the Juba and Shabelle rivers. One forecast model "
     "per river-season window, a consensus of monitored points crossing on the same day, "
     "calibrated against SWALIM gauges, with backtests, skill scores and return periods."),
    # POD and FAR are first met here, before the section that defines them
    ("POD counts <strong>severe</strong> years caught, the objective the envelope is sized on. FAR counts activations with no RP3 flood behind them:",
     "POD (probability of detection) counts <strong>severe</strong> years caught, the objective "
     "the envelope is sized on. FAR (false alarm ratio) counts activations with no 1-in-3 flood "
     "behind them. Skill scores below defines both in full, with the rest of the contingency "
     "table:"),
    # v4 does appear later, in the scores and the forecast-issue comparisons
    ("v4 is not shown as a candidate anywhere on this page. It appears only as the readiness leg's reforecast, the sole archive covering the 8-to-12-day readiness leads (its band is cut at 7 to 12 days).",
     "v4 is not a candidate for the action leg. It appears as the readiness leg's reforecast, the "
     "sole archive covering the 8-to-12-day readiness leads (its band is cut at 7 to 12 days), and "
     "in the skill scores and the forecast-issue comparisons, because it is the model that runs live."),
    # the envelope, defined where the reader first meets the four windows
    ("All seven reporting-era points are monitored: four on the Juba (Luuq, Dollow, Bardheere, Bualle) and three on the Shabelle (Belet Weyne, Bulo Burti, Jowhar). No threshold, on either leg, sits below 1-in-3.",
     "All seven reporting-era points are monitored: four on the Juba (Luuq, Dollow, Bardheere, "
     "Bualle) and three on the Shabelle (Belet Weyne, Bulo Burti, Jowhar). No threshold, on either "
     "leg, sits below 1-in-3. The four windows together are called the envelope: the full amount is "
     "released whenever any one of them activates, so the envelope is what the budget is sized on "
     "and what the 1-in-3 ceiling applies to."),
    # valid day against issue date, at first use
    ("The operational test replays the historical forecasts: per pair and valid day, the most alarming",
     "The operational test replays the historical forecasts. A forecast has two dates: the issue "
     "date, when it was published, and the valid day, the day it describes. Per pair and valid day, "
     "the most alarming"),
    # EM-DAT and CERF expanded at first use
    ("Ranked on raw EM-DAT totals, though,",
     "Ranked on raw totals from EM-DAT, the international disaster database, though,"),
    ("(blue) the flood allocation (US$), shaded by magnitude",
     "(blue) the allocation from the UN Central Emergency Response Fund (US$), shaded by magnitude"),
]


def pattern(old):
    return re.compile(r"\s+".join(re.escape(p) for p in old.split()))


t = PAGE.read_text(encoding="utf-8")
applied, already, missing = 0, 0, []
for pair in PAIRS:
    old, new = pair[0], pair[1]
    anchor = pair[2] if len(pair) > 2 else new
    rx = pattern(old)
    if rx.search(t):
        t = rx.sub(lambda m: new, t, count=1)
        applied += 1
    elif pattern(anchor).search(t):
        already += 1
    else:
        missing.append(old[:80])
PAGE.write_text(t, encoding="utf-8")
print(f"applied {applied}, already {already}")
if missing:
    print("NOT FOUND:", *missing, sep="\n   - ")
    sys.exit(1)
