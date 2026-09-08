"""Per-station SWALIM record: for every archived bulletin, the reading it reports at
each gauge (in metres) and the risk level that implies on SWALIM's own official
thresholds. Sources: alerts_text/ (Drive archive) and forecasts_2024_text/ (the
weekly bulletins' river section). Writes swalim_station_records.csv."""
import re
import sys
from pathlib import Path

import pandas as pd

S = Path(__file__).parent
sys.path.insert(0, str(S))
import somlib as L  # noqa: E402

TH = L.swalim_thresholds()
ALIASES = {
    "belet_weyne": r"bele[dt]\s*[- ]?weyne|beletweyne|belad\s*weyne|belet\s*wayne",
    "bulo_burti": r"bulo\s*bur(?:ti|te|de)|bula\s*burde|bulo\s*burti",
    "jowhar": r"jowhar|johwar",
    "dollow": r"doll?ow|doolow",
    "luuq": r"luuq|luq\b",
    "bardheere": r"ba+rdhee?re|bardhere|baardheere",
    "bualle": r"bu'?aale|bualle|buale",
}
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
RISK_WORDS = [
    (3, r"bank\s*-?\s*full|bankful|overflow|overtop|burst|over its banks|inundat|submerg|flooded"),
    (2, r"high[- ]?(?:flood )?risk|high risk|surpassed the high|past the critical|beyond the high"),
    (1, r"moderate[- ]?(?:flood )?risk|moderate risk|reached the moderate|beyond the moderate|above the moderate"),
]


def issue_date(name):
    n = name.lower().replace("noveember", "november")
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", n)          # yyyymmdd
    if m:
        y, mo, d = m.groups()
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return pd.Timestamp(int(y), int(mo), int(d))
    m = re.search(r"(?<!\d)(\d{2})(\d{2})(20\d{2})(?!\d)", n)          # ddmmyyyy
    if m:
        d, mo, y = m.groups()
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return pd.Timestamp(int(y), int(mo), int(d))
    m = re.search(r"(\d{1,2})[._-](\d{2})[._-](\d{4})", n)
    if m:
        d, mo, y = m.groups()
        if 1 <= int(mo) <= 12:
            return pd.Timestamp(int(y), int(mo), int(d))
    m = re.search(r"(\d{1,2})[ _]?(" + "|".join(MONTHS) + r")[a-z]*[ _]?(\d{4})", n)
    if m:
        d, mo, y = m.groups()
        return pd.Timestamp(int(y), MONTHS[mo], int(d))
    m = re.search(r"(" + "|".join(MONTHS) + r")[a-z]*[ _](\d{4})", n)
    if m:
        return pd.Timestamp(int(m.group(2)), MONTHS[m.group(1)], 15)
    return None


def clean(txt):
    for a, b in (("\ufb02", "fl"), ("\ufb01", "fi"), ("\u0198", "tt"), ("\u019f", "ti"),
                 ("\u2019", "'"), ("\u2013", "-"), ("\u2014", "-")):
        txt = txt.replace(a, b)
    return re.sub(r"\s+", " ", txt)


def plausible(st, val):
    """A published gauge reading, not a rainfall total or a margin in metres."""
    if st not in TH.index:
        return 0.5 <= val <= 12
    t = TH.loc[st]
    lo = (t["moderate_flood_risk"] - 3.0) if pd.notna(t["moderate_flood_risk"]) else 0.5
    hi = (t["bank_full"] + 1.5) if pd.notna(t["bank_full"]) else 12
    return lo <= val <= hi


def level_class(st, val):
    if st not in TH.index or val is None:
        return None
    t = TH.loc[st]
    for lvl, col in ((3, "bank_full"), (2, "high_flood_risk"), (1, "moderate_flood_risk")):
        if pd.notna(t[col]) and val >= t[col] - 1e-9:
            return lvl
    return 0


NEGATED = re.compile(r"(below|under|beneath|short of|away from|to reach|before reaching)\W*$", re.I)


def word_class(text):
    """Highest level stated in words, ignoring 'x m below the high-risk level'."""
    best = None
    for lvl, w in RISK_WORDS:
        for m in re.finditer(w, text, re.I):
            if NEGATED.search(text[max(0, m.start() - 32):m.start()]):
                continue
            if best is None or lvl > best:
                best = lvl
        if best is not None:
            return best
    return best


def reading_after(pat, text, st):
    """Largest plausible gauge reading published for this station in this text."""
    best = None
    for m in re.finditer(pat, text, re.I):
        tail = text[m.end():m.end() + 90]
        v = re.search(r"(?:is|at|of|reached|recorded|level)?\D{0,18}?(\d\.\d{1,2})\s*m\b", tail)
        if not v:
            v = re.search(r"\((\d\.\d{1,2})\s*m\)", tail)
        if v:
            val = float(v.group(1))
            if plausible(st, val) and (best is None or val > best):
                best = val
    return best


rows = []
for folder in ("alerts_text", "forecasts_2024_text"):
    for p in sorted((S / folder).glob("*.txt")):
        name = p.stem
        low = name.lower()
        if any(k in low for k in ("drought", "wind", "tropical")):
            continue
        d = issue_date(name)
        if d is None:
            print("no date:", name)
            continue
        txt = clean(p.read_text(encoding="utf-8", errors="ignore"))
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", txt)
        for st, pat in ALIASES.items():
            best_val, best_risk, quote = None, None, None
            for s in sents:
                if not re.search(pat, s, re.I):
                    continue
                # readings: a number in metres within ~90 chars after the station name
                val = reading_after(pat, s, st)
                if val is not None and (best_val is None or val > best_val):
                    best_val, quote = val, s[:260]
                # skip sentences that look back at another year ("During the Gu 2023 ...")
                other_year = [int(v) for v in re.findall(r"(?<!\d)(20[0-2]\d)(?!\d)", s) if int(v) != d.year]
                # the weekly bulletins publish a reading for every station they discuss and
                # their prose is full of look-backs and forecasts, so take readings only
                r = None if (other_year or folder == "forecasts_2024_text") else word_class(s)
                if r is not None and (best_risk is None or r > best_risk):
                    best_risk = r
                    if quote is None:
                        quote = s[:260]
            if best_val is None and best_risk is None:
                continue
            rows.append({"file": name, "date": d, "station": st, "reading_m": best_val,
                         "class_from_reading": level_class(st, best_val),
                         "class_from_words": best_risk, "quote": quote})

# Deyr 2006 and Deyr 2014: the ReliefWeb bulletins, whose statements carry the readings
ext = S / "swalim_external.csv"
if ext.exists():
    for _, r in pd.read_csv(ext).iterrows():
        d = pd.Timestamp(r["date"])
        txt = clean(str(r["statement"]))
        for st, pat in ALIASES.items():
            if not re.search(pat, txt, re.I):
                continue
            best_val = reading_after(pat, txt, st)
            other_year = [int(v) for v in re.findall(r"(?<!\d)(20[0-2]\d)(?!\d)", txt) if int(v) != d.year]
            best_risk = None if other_year else word_class(txt)
            if best_val is None and best_risk is None:
                continue
            rows.append({"file": "reliefweb " + str(r["kind"]), "date": d, "station": st, "reading_m": best_val,
                         "class_from_reading": level_class(st, best_val),
                         "class_from_words": best_risk, "quote": txt[:260]})

df = pd.DataFrame(rows).sort_values(["date", "station"])
df["level_class"] = df["class_from_reading"].where(df["class_from_reading"].notna(), df["class_from_words"])
df["from_reading"] = df["class_from_reading"].notna()
df.to_csv(S / "swalim_station_records.csv", index=False, encoding="utf-8")
print(f"{len(df)} station-bulletin records, {df.date.min().date()} to {df.date.max().date()}")
print(df.groupby("station").size().to_string())
print("\nwith a reading:", df.reading_m.notna().sum(), "| class 3/2/1/0:",
      {int(k): int(v) for k, v in df.level_class.value_counts().sort_index(ascending=False).items()})
print(df[df.reading_m.notna()].head(14)[["date", "station", "reading_m", "level_class"]].to_string(index=False))
