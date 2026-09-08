"""Rewrite the SWALIM section of the trigger page: a timeline chart and a short
who-was-first table up front, the full record folded into a details block.
Narrative cells (what happened, SWALIM's three risk steps by bulletin date) are
written here from the bulletin texts; activation dates, points over, peaks and the
v4 forecast / v5 reanalysis columns come from trigger_detail.json
(trigger_detail.py) and, for Gu 2024, gu2024_issue.json (gu2024_issue.py); the
who-was-first verdicts come from swalim_timeline.json (swalim_timeline.py)."""
import json
from datetime import datetime
from pathlib import Path

S = Path(__file__).parent
PAGE_DIR = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model"
PAGE = PAGE_DIR / "index.html"
DET = {(d["river"], d["season"], d["year"]): d
       for d in json.loads((S / "trigger_detail.json").read_text(encoding="utf-8"))}
G24 = json.loads((S / "gu2024_issue.json").read_text(encoding="utf-8"))
TL = {(t["season"], t["river"]): t for t in json.loads((S / "swalim_timeline.json").read_text(encoding="utf-8"))}

# (river, season, year): what happened | SWALIM moderate | SWALIM high | SWALIM bank full or overflow
# | ISO date of SWALIM's first bulletin flagging risk at any level (for the lag columns).
NARR = {
    ("juba", "deyr", 2006): (
        "Severe flood; 1-in-5 at two gauges by 30 Oct",
        "not archived (bulletin no. 1 of 3 Oct saw no risk; nos. 2 to 7 missing)",
        "31 Oct, high risk in the riverine areas",
        "31 Oct, river over its banks from Luuq to Jamame",
        "2006-10-31"),
    ("shabelle", "deyr", 2006): (
        "Severe flood; 1-in-3 on 2 Nov, 1-in-5 on 18 Nov",
        "not archived (nos. 2 to 7 missing)",
        "31 Oct, severe risk downstream of Jowhar",
        "31 Oct, bank breakages in the lower Shabelle after abrupt rises at Belet Weyne, Bulo Burti and Jowhar",
        "2006-10-31"),
    ("juba", "deyr", 2014): (
        "Severe flood; 1-in-3 on 22 Oct, 1-in-5 on 24 Oct",
        "15 Oct, both rivers",
        "21 Oct, both rivers",
        "28 Oct, floods at Dollow, Jilib and Jamame (24 Oct: Luuq 1 m below bank full)",
        "2014-10-15"),
    ("shabelle", "deyr", 2014): (
        "Severe flood; 1-in-3 on 20 Oct, 1-in-5 on 29 Oct",
        "15 Oct, Belet Weyne 6.10 m",
        "20 Oct, Flood Alert: Belet Weyne 7.00 m past the critical level",
        "24 Oct, Belet Weyne 7.30 m, the flooding level; 28 Oct floods at Belet Weyne and breakages in Middle Shabelle",
        "2014-10-15"),
    ("shabelle", "gu", 2016): (
        "Severe flood; 1-in-3 on 11 May, 1-in-5 on 18 May",
        "no bulletin archived", "no bulletin archived",
        "no bulletin archived (June flood map only)",
        None),
    ("juba", "deyr", 2019): (
        "No two-gauge crossing; local flooding reported",
        "not archived; the 25 Oct update says the moderate level was passed in the two weeks before 22 Oct",
        "22 Oct, Bardheere",
        "22 Oct, Bardheere bank full, flooding at Luuq and Bardheere; 25 Oct also Dollow and Bualle",
        "2019-10-22"),
    ("shabelle", "deyr", 2019): (
        "Severe flood; 1-in-3 on 14 Oct, 1-in-5 on 24 Oct",
        "not archived (first bulletin 22 Oct)",
        "22 Oct, Belet Weyne and Jowhar (Jowhar at the high level since late August)",
        "22 Oct, Jowhar near bank full with two breakages; 25 Oct Belet Weyne town flooded by overflow",
        "2019-10-22"),
    ("shabelle", "deyr", 2020): (
        "1-in-3 on 1 Oct, 1-in-5 on 6 Oct (a separate flood at Belet Weyne in September)",
        "no bulletin for the October rise",
        "8 Sep, Bulo Burti 7.00 m (the September flood)",
        "8 Sep, Belet Weyne bank full with overbank spillage",
        None),
    ("juba", "gu", 2020): (
        "Severe flood; 1-in-3 on 22 Apr, 1-in-5 on 12 May",
        "not archived (first bulletin 27 Apr)",
        "27 Apr, Bardheere passed the high level that morning",
        "27 Apr, flooding reported at Dollow, Luuq and Bardheere",
        "2020-04-27"),
    ("shabelle", "gu", 2020): (
        "Severe flood; 1-in-3 on 6 May, 1-in-5 on 12 May",
        "27 Apr, high risk foreseen with Belet Weyne still 0.50 m below the moderate level; 4 May, Belet Weyne 7.20 m past it",
        "4 May, Jowhar at the high level for four days; 11 May, Belet Weyne 8.10 m over it",
        "18 May, Belet Weyne bank full since 12 May; 11 May Jowhar 0.20 m below",
        "2020-04-27"),
    ("juba", "gu", 2021): (
        "No two-gauge crossing",
        "10 May, Dollow over the moderate level, Luuq 0.30 m below",
        "not reached (high risk foreseen on 10 May; risk down to minimal on 19 May)",
        "no",
        "2021-05-10"),
    ("shabelle", "gu", 2021): (
        "Belet Weyne flooded; gauge censored at bank full, no two-gauge crossing",
        "10 May, Belet Weyne 6.50 m",
        "19 May, Belet Weyne 7.60 m (13 May: 6.60 m with high risk stated)",
        "25 May, Belet Weyne 8.25 m about to reach bank full; 19 May breakage flooding upstream reported",
        "2021-05-10"),
    ("shabelle", "gu", 2023): (
        "Severe flood; 1-in-3 on 17 Apr, 1-in-5 on 23 May",
        "no bulletin archived",
        "8 May advisory: high level at Belet Weyne, passed on 2 May",
        "8 May advisory: 0.40 m below bank full, overflow judged very likely",
        "2023-05-08"),
    ("juba", "deyr", 2023): (
        "Severe flood; 1-in-3 and 1-in-5 on 25 Oct",
        "20 Oct, Dollow and Luuq (Flood Alert with a call for anticipatory action)",
        "23 Oct update: high level passed at Luuq on 21 Oct and Bardheere on 22 Oct; 29 Oct all three over it",
        "13 Nov advisory: overflow along the whole river for more than a week",
        "2023-10-20"),
    ("shabelle", "deyr", 2023): (
        "Largest flood on record; 1-in-3 on 7 Nov, 1-in-5 on 20 Nov",
        "29 Oct, moderate risk along the river with Belet Weyne 0.10 m below the level (21 Oct advisory: anticipatory action called for Hiraan)",
        "2 Nov, high risk projected at Belet Weyne in 3 to 4 days",
        "13 Nov advisory: Belet Weyne overflowing, most of the town flooded; 20 Nov Bulo Burti 0.32 m over the high level",
        "2023-10-21"),
    ("juba", "gu", 2024): (
        "1-in-3 flood on 10 May (Luuq 7 May, Dollow 10 May); Luuq alone reached 1-in-5 on 8 May",
        "9 May weekly bulletin: Luuq over the moderate level (8 May reading)",
        "9 May weekly bulletin: Dollow over the high level on 6 May after very heavy rain, back below it by 9 May",
        "no (14 May: Dollow below the flood levels)",
        "2024-05-09"),
    ("shabelle", "gu", 2024): (
        "Severe flood; 1-in-3 on 8 May (Jowhar 19 Apr, Belet Weyne 8 May), 1-in-5 on 20 May",
        "1 May weekly bulletin: Belet Weyne 6.50 m",
        "19 Apr weekly bulletin: Jowhar 0.23 m below bank full; 9 May: Belet Weyne 7.48 m",
        "22 May weekly bulletin: Belet Weyne 8.30 m bank full with riverine flooding reported; 14 May: breakage floods at three villages on 12 May",
        "2024-04-19"),
}
ORDER = list(NARR)

NA = "not available: record ends 2023"
NONE6 = (None, [], 0, None, [], 0)
HAND = {
    ("juba", "gu", 2024): {"n_points": 4, "trigger": (NA,) + NONE6[1:], "v5": (NA,) + NONE6[1:],
                           "v4_fc_issue": G24["juba"]["v4_fc_issue"]},
    ("shabelle", "gu", 2024): {"n_points": 3, "trigger": (NA,) + NONE6[1:], "v5": (NA,) + NONE6[1:],
                               "v4_fc_issue": G24["shabelle"]["v4_fc_issue"]},
}
MUTED = "<span style='color:var(--n7)'>"


def d0(s):
    return s.lstrip("0") if s else s


def iso_short(s):
    return datetime.fromisoformat(s).strftime("%d %b").lstrip("0") if s else None


def cell(act, n_points):
    date, pts, mx, pk_date, pk, ndays = act
    if date == NA:
        return f"<td>{NA}</td>"
    if date is None:
        if mx == 0:
            return f"<td>no<br>{MUTED}no point over its level</span></td>"
        return f"<td>no<br>{MUTED}most: {mx} of {n_points} on {d0(pk_date)}: {', '.join(pk)}</span></td>"
    first = f"{d0(date)}, {len(pts)} of {n_points}: {', '.join(pts)}"
    peak = (f"peak {mx} of {n_points} on {d0(pk_date)}: {', '.join(pk)}" if mx > len(pts)
            else f"peak {mx} of {n_points} that day")
    days = "; 1 day at or above the count" if ndays == 1 else f"; {ndays} days at or above the count"
    return f"<td>{first}<br>{MUTED}{peak}{days}</span></td>"


def cell_issue(act, n_points, unit):
    idate, vdate, pts, mx, pk_i, pk_v, pk, n_iss = act
    if idate is None:
        if mx == 0:
            return f"<td>no<br>{MUTED}no point over its level in any issue</span></td>"
        return (f"<td>no<br>{MUTED}most: {mx} of {n_points}, issued {d0(pk_i)} for {d0(pk_v)}: "
                f"{', '.join(pk)}</span></td>")
    first = f"issued {d0(idate)} for {d0(vdate)}, {len(pts)} of {n_points}: {', '.join(pts)}"
    peak = (f"peak {mx} of {n_points} issued {d0(pk_i)} for {d0(pk_v)}: {', '.join(pk)}" if mx > len(pts)
            else f"peak {mx} of {n_points} in that issue")
    n = f"; 1 {unit[:-1]} at or above the count" if n_iss == 1 else f"; {n_iss} {unit} at or above the count"
    return f"<td>{first}<br>{MUTED}{peak}{n}</span></td>"


def lag(swalim_iso, act_date_str, year):
    if swalim_iso is None or act_date_str in (None, NA):
        return ""
    a = datetime.strptime(f"{act_date_str} {year}", "%d %b %Y")
    b = datetime.fromisoformat(swalim_iso)
    n = (a - b).days
    return f"{n:+d}" if n else "0"


# ---- the short table: who flagged first ------------------------------------
short_rows = []
for key in ORDER:
    season = f"{key[1].title()} {key[2]}"
    t = TL[(season, key[0].title())]
    d = DET.get(key) or HAND[key]
    trig = d["trigger"][0]
    trig_txt = "record ends 2023" if trig == NA else ("never crossed" if trig is None else d0(trig))
    v4 = d["v4_fc_issue"][0]
    v4_txt = "no issue crossed" if v4 is None else f"{d0(v4)} (for {d0(d['v4_fc_issue'][1])})"
    sw = iso_short(t["swalim_first"]) or "no bulletin"
    short_rows.append(f"<tr><td>{season}</td><td>{key[0].title()}</td><td>{sw}</td>"
                      f"<td>{trig_txt}</td><td>{t['vs_trigger']}</td>"
                      f"<td>{v4_txt}</td><td>{t['vs_v4']}</td></tr>")
short_table = ('<div class="tablewrap">\n<table class="data" style="font-size:12.5px">\n<thead><tr>'
               '<th>season</th><th>river</th><th>SWALIM first bulletin</th><th>model first day (GloFAS v5 in Deyr, Google in Gu)</th>'
               '<th>who was first</th><th>GloFAS v4 forecast first issue</th><th>who was first</th>'
               '</tr></thead>\n<tbody>' + "".join(short_rows) + "</tbody>\n</table>\n</div>")

# ---- the full record --------------------------------------------------------
rows = []
for key in ORDER:
    d = DET.get(key) or HAND[key]
    what, mod, high, bank, first_iso = NARR[key]
    season = f"{key[1].title()} {key[2]}"
    trig = d["trigger"]
    v4i = d["v4_fc_issue"]
    unit = "daily issues" if key[2] == 2024 else "issues"
    rows.append(f"<tr><td>{season}</td><td>{key[0].title()}</td><td>{what}</td>"
                f"<td>{mod}</td><td>{high}</td><td>{bank}</td>"
                + cell(trig, d["n_points"])
                + f"<td>{lag(first_iso, trig[0], key[2])}</td>"
                + cell_issue(v4i, d["n_points"], unit)
                + f"<td>{lag(first_iso, v4i[0], key[2])}</td>"
                + cell(d["v5"], d["n_points"]) + "</tr>")
full_table = ('<div class="tablewrap">\n<table class="data" style="font-size:12px">\n<thead><tr>'
              '<th>season</th><th>river</th><th>what happened at the gauges</th>'
              '<th>SWALIM: moderate risk</th><th>SWALIM: high risk</th><th>SWALIM: bank full or overflow</th>'
              '<th>trigger: first day and points over their RP</th><th>days SWALIM\'s first bulletin led the trigger</th>'
              '<th>GloFAS v4 forecast, by issue date</th><th>days SWALIM\'s first bulletin led the v4 issue</th>'
              '<th>GloFAS v5 reanalysis</th></tr></thead>\n<tbody>'
              + "".join(rows) + "</tbody>\n</table>\n</div>")

section = """    <h3>SWALIM's alerts against the trigger</h3>
    <p>SWALIM issues its own flood bulletins from the gauge readings and the rainfall
      forecast, grading river flood risk as moderate, then high, then bank full or overflow.
      The question here is who flagged first in each flood season, SWALIM or the trigger,
      and by how many days. Every date is the date the information was available: the issue
      date of the SWALIM bulletin, the first day the trigger crossed on its reanalysis
      (Google for Gu, GloFAS v5 for Deyr), and the issue date of the first GloFAS v4
      forecast (the model that runs live) whose ensemble median had enough points over
      their thresholds at some lead of 1 to 7 days. The grey band marks the gauges' own
      two-gauge 1-in-3 and 1-in-5 crossings.</p>
    <figure><img src="figs/k_swalim_timeline.png?v=202609071" alt="Timeline of SWALIM bulletins, trigger and GloFAS v4 forecast per flood season">
      <figcaption>Each row is one river-season. Upper track: SWALIM's first bulletin flagging
        risk (open circle) and the bulletins that first reported the moderate, high and bank
        full steps. Lower track: the first day the window's model crossed on its reanalysis
        (diamond: GloFAS v5 in Deyr, Google in Gu) and the GloFAS v4 forecast's first issue
        with enough points over (triangle). The text at right gives the days between
        SWALIM's first bulletin and the model, and in grey between SWALIM and the v4
        issue. Gu 2024 is beyond the Google and v5 records, so only the v4 forecast is
        compared there. Deyr 2020's SWALIM bulletin concerned the September flood at Belet
        Weyne, before the Deyr window; Deyr 2006's SWALIM issues 2 to 7 are not on
        ReliefWeb, so its first flag may have been earlier than shown.</figcaption></figure>
    <p>The action window is 1 to 7 days before the flood and readiness 8 to 12. The chart
      below places each flag against the onset of the flood at the gauges, the day the
      two-gauge 1-in-3 level was first crossed, not the peak. The trigger's date is the day
      the modelled flow crossed; in operation the forecast would have added up to 7 days to
      it. The GloFAS v4 issue date is the operational measure.</p>
    <figure><img src="figs/k_swalim_window.png?v=202609071" alt="Lead time of each flag before the gauges' 1-in-3 crossing">
      <figcaption>Days between each flag and the gauges' first two-gauge 1-in-3 crossing, one
        row per flood season with a gauge event (Deyr 2019 Juba and Gu 2021 have none). Green:
        the action window, 1 to 7 days before onset; pale green: readiness, 8 to 12 days.
        The black tick is the 1-in-5 crossing. Deyr 2006 Juba's onset is the 1-in-5 date, since
        the 1-in-3 date is not in the record.</figcaption></figure>
    <p>SWALIM's first bulletin fell inside the action window in five of the twelve seasons
      with a bulletin and a gauge event: Deyr 2006 and Deyr 2014 on the Shabelle, Deyr 2014
      and Deyr 2023 on the Juba, and Gu 2024 on the Juba, at 1 to 7 days before onset. It
      was earlier than the window in three (Gu 2020 Shabelle 9 days, Deyr 2023 Shabelle 17
      days, Gu 2024 Shabelle 19 days) and on or after onset in four (Deyr 2006 Juba, Deyr
      2019 Shabelle, Gu 2020 Juba, Gu 2023 Shabelle). The GloFAS v4 issue was in the window
      in Gu 2016, Deyr 2023 Juba and Gu 2024 Juba, earlier than it in Deyr 2014 and Deyr 2019
      on the Shabelle, and late or absent everywhere else; the reanalysis trigger's own day
      was inside the window in Deyr 2014, Deyr 2019 and Gu 2020 on the Shabelle (and a day
      before the 1-in-5 date in Deyr 2006 on the Juba) and on or after onset in six seasons,
      which is the gap a 1 to 7 day forecast has to close.</p>
    <p>Where both flagged, SWALIM was first in six river-seasons and the window's model in
      two. SWALIM's lead was 1 to 3 days in Deyr 2006, Deyr 2014 and Gu 2020, and 9 and 19
      days in Deyr 2023, when its 20 October alert asked for anticipatory action while
      GloFAS v5 waited for the rivers themselves. GloFAS v5 led in Deyr 2019 on the Shabelle
      by 11 days and in Deyr 2006 on the Juba by 2 days, where SWALIM's earlier issues are
      missing. In four flood seasons SWALIM flagged and the model never crossed: GloFAS v5 in
      Deyr 2014 and Deyr 2019 on the Juba, Google in Gu 2021 on both rivers and Gu 2023 on
      the Shabelle. Gu 2016 has no SWALIM bulletin at all.</p>
    <p>The GloFAS v4 forecast, which is what would run live, is earlier than SWALIM where
      it crosses at all: by 8 days in Deyr 2014 and 20 days in Deyr 2019 on the Shabelle,
      and by 4 days in Gu 2024 on the Juba, with SWALIM ahead only in Deyr 2014 on the Juba
      (8 days), Gu 2020 on the Juba (4 days) and Deyr 2023 on the Juba (1 day). Its failure
      mode is the Shabelle: no v4 issue had a point over its level in Gu 2020, Deyr 2023 or
      Gu 2024, three seasons in which SWALIM reported bank full at Belet Weyne. Up to 2021
      SWALIM's archived bulletins are Flood Updates written once a river was already at the
      high level or bank full; the ladder-style alerts with a moderate step start in 2023.</p>
    <h4>The same comparison station by station</h4>
    <p>The charts above work at river level, where SWALIM's earliest flag anywhere on the
      river meets a rule that counts points. SWALIM's bulletins name individual gauges and
      publish their readings, so the same comparison can be made gauge by gauge. The two
      charts below give one row per station and season: the bulletin on which SWALIM first
      reported that gauge at each of its own official levels, that gauge's own
      return-period crossings, and the first day each model crossed that station's own
      threshold. A level counts as reported when the reading SWALIM published for the gauge
      is at or above its official moderate, high or bank-full level. Hollow markers are
      seasons where the bulletin stated a risk for the station without publishing a
      reading, which is most of 2006 and 2019 to 2021. The 2024 weekly bulletins are read
      from their published readings only, because their prose mixes forecasts and
      look-backs to earlier years.</p>
    <figure><img src="figs/k_swalim_station_shabelle.png?v=202609081" alt="Shabelle: SWALIM's reported levels at each gauge against the gauge record and the models">
      <figcaption>Shabelle. Upper track: SWALIM's reported levels for that gauge. Lower
        track: the models over that station's own threshold. Grey band: that gauge's own
        1-in-3 to 1-in-5 crossings. Deyr 2020's bulletins concern the September flood at
        Belet Weyne, before the Deyr window opens.</figcaption></figure>
    <figure><img src="figs/k_swalim_station_juba.png?v=202609081" alt="Juba: SWALIM's reported levels at each gauge against the gauge record and the models">
      <figcaption>Juba, same layout.</figcaption></figure>
    <p>Of the 55 station-seasons, 39 have a bulletin naming that gauge and 21 record it at
      bank full. Belet Weyne and Jowhar are named in 8 of their 9 seasons, Bulo Burti in 6;
      on the Juba, Luuq in 6 of 7, Dollow and Bardheere in 4, Bualle in 3. Where the gauge
      crossed its own 1-in-3 and no bulletin named the station, the gap falls on Bulo Burti
      and Bardheere twice each and on Belet Weyne and Bualle once.</p>
    <p>The station view separates two things the river view combines. SWALIM is early at
      Belet Weyne, flagging it before or on the day the gauge crossed its own 1-in-3 in
      seven of eight seasons and 8 days late in the eighth. It is late at Luuq in all five
      seasons with both dates, by 1 to 16 days, and at Bulo Burti in four of six. Jowhar is
      the exception to the whole comparison: in three of its eight flagged seasons the gauge
      never crossed its own 1-in-3, which matches the bulletins describing flooding there
      from breakages at Baarey, Moyko and Mandheere rather than from the river topping its
      banks.</p>
    <p>Against SWALIM's first flag for the same gauge, over the 34 station-seasons up to
      2023, Google crossed that station's threshold earlier in 15, on the same day in 2,
      later in 8 and never in 9. GloFAS v5 was earlier in 12, same day in 2, later in 13 and
      never in 7. The GloFAS v4 forecast issue was earlier in 7, same day in 3, later in 7
      and absent in 17. At station level the models therefore lead SWALIM about as often as
      they trail it, and the trigger's advantage at river level comes from requiring several
      points rather than from any one gauge being called early.</p>
    <details class="supp"><summary>Full record: the bulletins, the points over and the peaks</summary>
    <div class="supp-body">
    <p>For each river-season the table gives what happened at the gauges, the bulletin that
      first reported each step of SWALIM's ladder (with the station and level the bulletin
      gives; where the bulletin dates the crossing to an earlier day, that day follows the
      level), the trigger's first day with the points over their return-period thresholds,
      the GloFAS v4 forecast's first issue with enough points over and the valid day it
      pointed at ("issued 20 Oct for 21 Oct"), and the same rule on the GloFAS v5
      reanalysis for every window. Each model cell then gives the season's peak count and
      how many days (or forecast issues) stayed at or above the required count; where the
      rule never crossed, the most points over on any one day. The lag columns count days
      from SWALIM's first bulletin flagging any level (positive when SWALIM was earlier).</p>
    <p>Sources: the 52 bulletins in the SWALIM archive on blob (<code>raw/swalim/alerts/</code>),
      which carry river bulletins for 2016 and 2019 to 2023; SWALIM's Flood Watch, Flood
      Alert and Flood Update bulletins for Deyr 2006 and Deyr 2014 retrieved from ReliefWeb;
      and for Gu 2024 the river-level section of SWALIM's weekly weather bulletins, since no
      separate flood alert was issued that season. The 2008 and 2018 flood seasons have no
      bulletins yet. The v4 forecast column uses the twice-weekly 11-member reforecast up to
      2023 and, for Gu 2024, the daily 50-member operational forecasts archived for that
      season.</p>
""" + full_table + """
    </div></details>
"""

t = PAGE.read_text(encoding="utf-8")
i = t.find("    <h3>SWALIM's alerts against the trigger</h3>")
j = t.find("    <h3>SWALIM's bulletins in seasons the gauges did not call a flood</h3>", i)
assert i > 0 and j > i
t = t[:i] + section + t[j:]
PAGE.write_text(t, encoding="utf-8")
print("section rewritten:", len(rows), "rows")
