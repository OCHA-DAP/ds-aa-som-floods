"""Prose pass over the trigger page: replace em dashes with colons, commas or
sentence breaks, and drop figurative and filler phrasing. Each replacement is
asserted, so a miss is reported rather than silently skipped. Idempotent: a pair
whose 'old' has already gone is reported as done."""
import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model" / "index.html"
DASHES = ("&mdash;", "—")

# "@" stands for an em dash in either encoding
PAIRS = [
    # --- review-note index list and callouts
    ("The whole report below @ mechanism table, return periods and backtest @ is",
     "The whole report below, including the mechanism table, return periods and backtest, is"),
    ("Nothing in the trigger design has been changed @ the configuration, the numbers and the figures are exactly as generated.",
     "Nothing in the trigger design has been changed: the configuration, the numbers and the figures are exactly as generated."),
    # --- model-choice section
    ("so the only available rule is 2 of 2 @ unanimous, which this design&#39;s own &ldquo;never all of them&rdquo; constraint forbids @ and it adds a false activation in 2015",
     "so the only available rule is 2 of 2, which is unanimous and forbidden by this design&#39;s own &ldquo;never all of them&rdquo; constraint, and it adds a false activation in 2015"),
    ("the seven-point set is load-bearing, and two points are unverifiable",
     "the seven-point set cannot be reduced, and two points are unverifiable"),
    ("The seven-point set is load-bearing, and two of the points can no longer be checked.",
     "The seven-point set cannot be reduced, and two of the points can no longer be checked."),
    ("It is worth recording that it is cost-free on this record.",
     "On this record it costs nothing measurable."),
    ("The per-window search does not beat a rule with no search in it @ partly answered.",
     "The per-window search does not beat a rule with no search in it: partly answered."),
    ("without breaching the 1-in-3 ceiling @ GloFAS v5 everywhere and Google everywhere both land at 1-in-2.6 @ so the mixed assignment is doing real work",
     "without breaching the 1-in-3 ceiling, since GloFAS v5 everywhere and Google everywhere both land at 1-in-2.6, so the mixed assignment is doing real work"),
    ("A uniform rule at different settings @ Google GRRR, 1-in-6, 3 of 4 in every window @ still reproduces the adopted envelope exactly.",
     "A uniform rule at different settings, Google GRRR at 1-in-6 with 3 of 4 in every window, still reproduces the adopted envelope exactly."),
    ("Both are true: the assignment is justified, while the tuning around it is not what earns the result.",
     "Both are true: the assignment is justified, and the tuning around it is not what produces the result."),
    ("the requirement can be honoured without giving anything up @ and it buys a real operational simplification, since",
     "the requirement can be honoured without giving anything up, and it simplifies operations, since"),
    ("One source per river-season window, never a mixture @ requested by WFP",
     "One source per river-season window, never a mixture: requested by WFP"),
    ("GEOGloWS excluded @ working group, leaning that way",
     "GEOGloWS excluded: working group decision, leaning that way"),
    ("The envelope activates no more often than 1-in-3 @ directive of 2026-08-27",
     "The envelope activates no more often than 1-in-3: directive of 2026-08-27"),
    ("and the reasoning is sound @ see the callout above for the quantified version.",
     "and the reasoning is sound; the callout above gives the quantified version."),
    ("but no activation outside the benchmark @ it does not activate in 2013.",
     "but no activation outside the benchmark: it does not activate in 2013."),
    # --- GEOGloWS exclusion rationale
    ("Its forecasts run below its own retrospective @ a median ratio of 0.89 across leads 1&ndash;7 @ so a threshold fitted on the retrospective sits too high",
     "Its forecasts run below its own retrospective, at a median ratio of 0.89 across leads 1 to 7, so a threshold fitted on the retrospective sits too high"),
    ("The team&#39;s Nepal technical note is the cautionary precedent @ SFDC did not rescue event detection there.",
     "The team&#39;s Nepal technical note records the same result: SFDC did not rescue event detection there."),
    # --- benchmark and impact
    ("the 2008 peak is 5.840 m @ a 9 mm margin on a level that moves 38 mm purely from adding future years.",
     "the 2008 peak is 5.840 m, a 9 mm margin on a level that moves 38 mm purely from adding future years."),
    ("which the rule does not call a flood at all @ correctly, since that event was in Galgaduud",
     "which the rule does not call a flood at all, correctly, since that event was in Galgaduud"),
    ("one hit moves coverage by ~12&ndash;15 points @ differences under ~0.15 are noise.",
     "one hit moves coverage by about 12 to 15 points, so differences under about 0.15 are noise."),
    # --- supporting-evidence summaries
    ("Supporting evidence @ seasonal peaks, model vs gauge", "Supporting evidence: seasonal peaks, model vs gauge"),
    ("Supporting evidence @ forecast skill against each model&#39;s own reanalysis",
     "Supporting evidence: forecast skill against each model&#39;s own reanalysis"),
    ("Supporting evidence @ the tuning surface", "Supporting evidence: the tuning surface"),
    ("Supporting evidence @ official SWALIM levels vs the fitted ones",
     "Supporting evidence: official SWALIM levels vs the fitted ones"),
    # --- seasonal peaks
    ("against the observed seasonal peak at the reference gauge @ does the model rank the years correctly?",
     "against the observed seasonal peak at the reference gauge, which shows whether the model ranks the years in the right order."),
    ("makes its own threshold estimate wide @ treat its vertical placement, not its ranking, with caution",
     "makes its own threshold estimate wide, so treat its vertical placement with caution rather than its ranking"),
    # --- own-skill section
    ("own reanalysis or retrospective @ rank correlation (does it track itself?) and the median forecast/reanalysis ratio (is it biased against its own climatology?)",
     "own reanalysis or retrospective, by rank correlation (whether it tracks itself) and the median forecast-to-reanalysis ratio (whether it is biased against its own climatology)"),
    ("ratio &asymp; 1.00 @ the licence for calibrating their thresholds on reanalysis.",
     "ratio about 1.00, which is what licenses calibrating their thresholds on reanalysis."),
    ("(readiness band, shaded) @ its 7&ndash;12-day thresholds should anticipate that.",
     "(readiness band, shaded), so its 8-to-12-day thresholds should anticipate that."),
    ("on only ~2 years of archive (2024&ndash;26) @ retrospective-fitted thresholds will activate less often than intended",
     "on only about two years of archive (2024 to 2026), so retrospective-fitted thresholds will activate less often than intended"),
    ("cannot be tested until one ships @ the single most important gap.",
     "cannot be tested until one ships, which is the largest remaining gap."),
    # --- tuning surface
    ("per river &times; season here @ the same recipe as", "per river and season here, the same recipe as"),
    ("The adopted cell is not always the best single-window score @ it is the best choice subject to the river-level constraints",
     "The adopted cell is not always the best single-window score: it is the best choice subject to the river-level constraints"),
    ("one hit moves POD by ~12 points @ broad plateaus, not sharp optima, are the honest reading.",
     "one hit moves POD by about 12 points, so the surface shows broad plateaus rather than sharp optima."),
    # --- SWALIM levels check
    ("not by SWALIM&#39;s published Moderate/High flood-risk levels @ and the check below is why.",
     "not by SWALIM&#39;s published moderate and high flood-risk levels. The check below is why."),
    ("engineering levels of uncertain provenance, not a consistent severity scale @ the same conclusion",
     "engineering levels of uncertain provenance rather than a consistent severity scale, the same conclusion"),
    ("rests on only 8&ndash;9 seasons (gauge online 2015) @ read it loosely.",
     "rests on only eight or nine seasons (gauge online 2015), so read it loosely."),
    # --- operational section
    ("The calibration record is reanalysis @ a model&#39;s afterwards-view of its own past.",
     "The calibration record is reanalysis, a model&#39;s afterwards-view of its own past."),
    ("(max over issue dates @ with mixed providers in one window, a monitoring day combines each provider&#39;s latest forecast)",
     "(max over issue dates; with mixed providers in one window, a monitoring day combines each provider&#39;s latest forecast)"),
    ("the retrospective as a lead-0 stand-in (hindsight @ flagged per window)",
     "the retrospective as a lead-0 stand-in (hindsight, flagged per window)"),
    ("That forecast skill barely decays from lead 1 to 7 @ initial-condition-driven on these slow rivers @ is",
     "That forecast skill barely decays from lead 1 to 7 on these slow rivers, where the flow is driven by initial conditions, is"),
    ("misses Deyr 2023 on the Shabelle @ the largest flood in the record @ even though the v5 reanalysis flags it clearly.",
     "misses Deyr 2023 on the Shabelle, the largest flood in the record, even though the v5 reanalysis flags it clearly."),
    # --- crossings and activation table
    ("How tight are the crossings in practice?", "The crossings are tight."),
    ("The exception is instructive.", "One season breaks the pattern."),
    ("simultaneously over their thresholds that season-year @ the cell fills red when it",
     "simultaneously over their thresholds that season-year. The cell fills red when it"),
    ("maximum level at the SWALIM reference gauge @ 5yr =", "maximum level at the SWALIM reference gauge: 5yr ="),
    ("per the threshold check above @ not the official flood-risk levels)",
     "per the threshold check above, not the official flood-risk levels)"),
    ("counted under both rivers (not split @ hover for deaths and the multi-river flag)",
     "counted under both rivers (not split; hover for deaths and the multi-river flag)"),
]


def variants(s):
    """Every encoding of the dash placeholder that might be in the file."""
    if "@" not in s:
        return [s]
    return [s.replace("@", d) for d in DASHES]


def pattern(old):
    """Whitespace-tolerant regex: the source wraps sentences across lines."""
    parts = [re.escape(p) for p in old.split()]
    body = r"\s+".join(parts)
    return re.compile(body.replace(re.escape("@"), r"(?:&mdash;|—)"))


t = PAGE.read_text(encoding="utf-8")
applied, already, missing = 0, 0, []
for old, new in PAIRS:
    rx = pattern(old)
    if rx.search(t):
        t = rx.sub(lambda m: new.replace("@", "&mdash;"), t, count=1)
        applied += 1
    elif any(re.search(pattern(n), t) for n in variants(new)):
        already += 1
    else:
        missing.append(old[:90])

# the review-note index list: "R7 &mdash; text" becomes "R7: text"
before = t
t = re.sub(r"(>R\d+</a>)\s*(?:&mdash;|—)\s*", r"\1: ", t)
n_index = len(re.findall(r"(?:&mdash;|—)", before)) - len(re.findall(r"(?:&mdash;|—)", t))

PAGE.write_text(t, encoding="utf-8")
print(f"applied {applied}, already done {already}, review-index dashes converted {n_index}")
if missing:
    print("NOT FOUND:")
    for m in missing:
        print("   -", m)
    sys.exit(1)
