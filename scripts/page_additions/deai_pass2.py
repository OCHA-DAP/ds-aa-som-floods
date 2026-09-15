"""Second prose pass: the remaining em dashes sit inside spans that carry markup,
so these edits are dash-local and keep the tags. Whitespace-tolerant, asserted."""
import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model" / "index.html"
D = r"(?:&mdash;|—)"

PAIRS = [
    ("</strong> @ a median ratio of 0.89 across leads 1&ndash;7 @ so a threshold",
     "</strong>, at a median ratio of 0.89 across leads 1 to 7, so a threshold"),
    ("technical note is the cautionary precedent @ SFDC did not rescue event detection there.",
     "technical note records the same result: SFDC did not rescue event detection there."),
    ("never a mixture</strong> @ requested by", "never a mixture</strong>: requested by"),
    ("<strong>GEOGloWS excluded</strong> @ working group, leaning that way",
     "<strong>GEOGloWS excluded</strong>: working group decision, leaning that way"),
    ("than 1-in-3</strong> @ directive of 2026-08-27", "than 1-in-3</strong>: directive of 2026-08-27"),
    ("at the reference gauge @ does the model rank the <em>years</em> correctly?",
     "at the reference gauge, which shows whether the model ranks the <em>years</em> in the right order."),
    ("peak Spearman &rho; @ reanalysis", "peak Spearman &rho;, reanalysis"),
    ("peak &rho; @ reforecast", "peak &rho;, reforecast"),
    ("reanalysis or retrospective @ rank correlation (does it track itself?)",
     "reanalysis or retrospective, by rank correlation (whether it tracks itself)"),
    ("(is it biased against its own climatology?)", "(whether it is biased against its own climatology)"),
    ("ratio &asymp; 1.00 @ the licence for calibrating", "ratio about 1.00, which is what licenses calibrating"),
    ("(readiness band, shaded) @ its 7&ndash;12-day thresholds should anticipate that.",
     "(readiness band, shaded), so its 8-to-12-day thresholds should anticipate that."),
    ("(2024&ndash;26) @ retrospective-fitted thresholds", "(2024 to 2026), so retrospective-fitted thresholds"),
    ("per river &times; season here @ the same recipe as", "per river and season here, the same recipe as"),
    ("single-window score @ it is the best choice subject to", "single-window score: it is the best choice subject to"),
    ("flood-risk levels @ and the check below is why.", "flood-risk levels. The check below is why."),
    ("only 8&ndash;9 seasons (gauge online 2015) @ read it loosely.",
     "only eight or nine seasons (gauge online 2015), so read it loosely."),
    ("The calibration record is reanalysis @ a model&#39;s afterwards-view of its own past.",
     "The calibration record is reanalysis, a model&#39;s afterwards-view of its own past."),
    ("(max over issue dates @ with mixed providers in one window",
     "(max over issue dates; with mixed providers in one window"),
    ("that season-year @ the cell fills", "that season-year. The cell fills"),
    ("SWALIM reference gauge @ <strong>5yr</strong> =", "SWALIM reference gauge: <strong>5yr</strong> ="),
    ("(not split @ hover for deaths and the multi-river flag)",
     "(not split; hover for deaths and the multi-river flag)"),
]


def pattern(old):
    parts = [re.escape(p) for p in old.split()]
    return re.compile(r"\s+".join(parts).replace(re.escape("@"), D))


t = PAGE.read_text(encoding="utf-8")
applied, already, missing = 0, 0, []
for old, new in PAIRS:
    rx = pattern(old)
    if rx.search(t):
        t = rx.sub(lambda m: new, t, count=1)
        applied += 1
    elif re.search(pattern(new.replace("@", "&mdash;")), t):
        already += 1
    else:
        missing.append(old[:80])

# count-then-years cells: "4 &mdash; 2013, 2016" becomes "4: 2013, 2016", in the cell
# text and in the provider-switch data attributes that must stay in sync
t, n_cells = re.subn(r"(\d)\s*&mdash;\s*(\d{4})", r"\1: \2", t)
t, n_attrs = re.subn(r"(\d)\s*&amp;mdash;\s*(\d{4})", r"\1: \2", t)
# the JS-built window header "Gu — >=3/4" (plain replaces: the source may use either form)
n_js = 0
for old_js in ('" &mdash; &ge;"', '" — &ge;"', '" — ≥"', '" &mdash; ≥"'):
    if old_js in t:
        t = t.replace(old_js, old_js.replace(" &mdash; ", ": ").replace(" — ", ": "))
        n_js += 1
n_js2 = 0

PAGE.write_text(t, encoding="utf-8")
print(f"applied {applied}, already {already}, table cells {n_cells}, switch attrs {n_attrs}, js {n_js + n_js2}")
if missing:
    print("NOT FOUND:")
    for m in missing:
        print("   -", m)
    sys.exit(1)
