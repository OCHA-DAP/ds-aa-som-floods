from dataclasses import dataclass
from typing import Optional

# Azure Blob (via ocha_stratus): container="projects", stage="dev"
BLOB_PREFIX = "ds-aa-som-floods/processed"
BLOB_STAGE = "dev"

# SWALIM river gauge data (raw)
SWALIM_RAW_PREFIX = "ds-aa-som-floods/raw/swalim"
SWALIM_THRESHOLD_FILENAME = "Search Data  SNRFA.csv"


@dataclass(frozen=True)
class Station:
    """One monitoring point on the Juba or Shabelle and its IDs per source.

    lat/lon are GloFAS v4 grid-cell centres (0.05deg, x.x25/x.x75), supplied
    by the user 2026-08-03 as the canonical point set: 7 on the Shabelle,
    8 on the Juba. Every point was verified to sit on the GloFAS modelled
    channel.

    geoglows_river_id is the TDX-Hydro LINKNO of the nearest main-stem
    stream *line* (>50,000 km2), matched against the actual VPU-103 stream
    geometry in EPSG:3857 — all matches are within 2.1 km. Do NOT match on
    the metadata table's representative points: that put Jowhar 24 km and
    Awdheegle 34 km from their assigned reach and produced spurious
    duplicates. Drainage areas increase monotonically downstream on both
    rivers, which is the check that catches tributary mis-snaps.

    grrr_gauge_id is "hybas_" + the HydroBASINS Africa level-12 basin
    containing the point (polygon containment, so no snapping issue).

    swalim_code links to the SNRFA gauge where identified; None where the
    point has no confirmed gauge (its observed levels cannot be joined).
    """

    name: str
    river: str  # "shabelle" or "juba"
    lat: float
    lon: float
    geoglows_river_id: int
    grrr_gauge_id: str
    swalim_code: Optional[str] = None
    canonical: bool = True  # part of the user's 7 + 8 point set
    note: str = ""


STATIONS = {
    # ---- Shabelle, upstream to downstream (canonical set of 7) ----
    "belet_weyne": Station(
        "Belet Weyne", "shabelle", 4.725, 45.225, 110895473, "hybas_1121024250",
        "sh001"
    ),
    "bulo_burti": Station(
        "Bulo Burti", "shabelle", 3.825, 45.575, 110858761, "hybas_1121049060",
        "sh002"
    ),
    "jowhar": Station(
        "Jowhar", "shabelle", 2.775, 45.525, 110794493, "hybas_1122059030",
        "sh004",
        note="shares GEOGloWS reach 110794493 (115 km long) with "
             "mahadey_weyne and shabelle_04 — identical GEOGloWS series",
    ),
    "shabelle_04": Station(
        "Shabelle km 2.575N", "shabelle", 2.575, 45.475, 110794493,
        "hybas_1121081050", None,
        note="gauge unidentified; shares GEOGloWS reach with jowhar",
    ),
    "shabelle_05": Station(
        "Shabelle km 2.375N", "shabelle", 2.375, 45.375, 110664625,
        "hybas_1121087720", None,
        note="gauge unidentified (Balad/sh005 is the likely candidate but its "
             "coordinates are not published in the SNRFA table)",
    ),
    "afgooye": Station(
        "Afgooye", "shabelle", 2.125, 45.125, 110682995, "hybas_1122063960",
        "sh006"
    ),
    "awdheegle": Station(
        "Awdheegle", "shabelle", 1.975, 44.825, 110713175, "hybas_1122066080",
        "sh007"
    ),
    # ---- Shabelle, retained from the earlier EF5-derived registry ----
    "mahadey_weyne": Station(
        "Mahadey Weyne", "shabelle", 2.975, 45.525, 110794493,
        "hybas_1122057530", "sh003", canonical=False,
        note="SWALIM gauge, not in the canonical 7; shares GEOGloWS reach "
             "with jowhar and shabelle_04",
    ),
    # ---- Juba, upstream to downstream (canonical set of 8) ----
    "luuq": Station(
        "Luuq", "juba", 3.725, 42.525, 110835148, "hybas_1122050700", "jb001"
    ),
    "bardheere": Station(
        "Bardheere", "juba", 2.325, 42.225, 110673811, "hybas_1121087010",
        "jb002"
    ),
    "bualle": Station(
        "Bualle", "juba", 1.225, 42.575, 110547888, "hybas_1121116000", "jb010"
    ),
    "kaitoi": Station(
        "Kaitoi", "juba", 0.775, 42.675, 110494108, "hybas_1121127010", "jb004"
    ),
    "juba_05": Station(
        "Juba km 0.525N", "juba", 0.525, 42.775, 110483618, "hybas_1121133480",
        None, note="gauge unidentified",
    ),
    "jilib": Station(
        "Jilib", "juba", 0.425, 42.725, 110439014, "hybas_1121137050", None,
        note="matches the earlier Jilib point (3 km); no SNRFA gauge",
    ),
    "juba_07": Station(
        "Juba km 0.175N", "juba", 0.175, 42.775, 110398348, "hybas_1121139500",
        None,
        note="gauge unidentified; outside the legacy GloFAS box (needs the "
             "extended S=-0.1 download). GEOGloWS reach set explicitly to "
             "110398348 (Juba main stem, 210,208 km2, 1.2 km away): the point "
             "sits at the Juba-Shabelle confluence and nearest-line matching "
             "picks the lower Shabelle (110402284, 297,648 km2) instead, which "
             "joins the Juba two reaches downstream. Same trap as Dollow.",
    ),
    "juba_08": Station(
        "Juba km 0.025N", "juba", 0.025, 42.675, 110450833, "hybas_1121144270",
        None, note="gauge unidentified (Jamaame is ~9 km away); outside the "
                   "legacy GloFAS box (needs the extended S=-0.1 download)",
    ),
    # ---- Juba, retained from the earlier registry ----
    "dollow": Station(
        "Dollow", "juba", 4.175, 42.075, 110896800, "hybas_1121038740",
        "jb009", canonical=False,
        note="SWALIM gauge; best lead-time skill of any station. GEOGloWS "
             "reach set explicitly to 110896800 (141,124 km2) = the Juba "
             "immediately below the Genale-Dawa confluence, since both "
             "tributary reaches (110895488 Dawa 58,895; 110894176 Genale "
             "82,224) drain into it. Nearest-line matching alone picks the "
             "Dawa tributary and halves the catchment.",
    ),
    "saakow": Station(
        "Saakow", "juba", 1.925, 42.275, 110705309, "hybas_1121098410", None,
        canonical=False, note="retained from the earlier registry; no gauge",
    ),
}

CANONICAL_STATIONS = {k: v for k, v in STATIONS.items() if v.canonical}

# SNRFA gauge code -> registry key, derived from the registry itself
SWALIM_CODE_TO_KEY = {
    st.swalim_code: key for key, st in STATIONS.items() if st.swalim_code
}


# ---------------------------------------------------------------- seasons
# Analysis season windows (months). EXP_SEASONS extends Gu to June because
# population-exposure peaks lag the gauge crossing as water spreads.
SEASONS = {"gu": [3, 4, 5], "deyr": [10, 11, 12]}
SEASON_MONTHS = {**SEASONS, "any": list(range(1, 13))}
EXP_SEASONS = {"gu": [3, 4, 5, 6], "deyr": [10, 11, 12]}

# ---------------------------------------------------------- chart palette
INK = "#1A2733"  # headline text
BODY = "#3A4552"  # axis label text
FAINT = "#6B7683"  # tick text
GRID = "#E4E8EC"  # gridlines
C_MOD = "#F4A93B"  # severity | Moderate (amber)
C_HIGH = "#B34036"  # severity | High (deep red)
C_GU = "#2A78D6"  # season | Gu (blue)
C_DEYR = "#EB6834"  # season | Deyr (orange)
C_JUBA = "#2A78D6"  # river | Juba (blue)
C_SHAB = "#0E8A7B"  # river | Shabelle (teal)
C_MAIN = "#1C7293"  # single-series default (teal-blue)
C_ISOL = "#065A82"  # isolated events (deep blue)
C_JOINT = "#F4A93B"  # joint events (amber)
C_BAND = "#E8E2D4"  # official Moderate-to-High band (sand)
C_BAND_EDGE = "#B9AF9B"  # band outline
C_REF = "#5C6B7A"  # reference marks (slate)
C_GEOGLOWS = "#8E5FA8"  # source | GEOGloWS (purple)
C_GEOGLOWS_SFDC = "#5B3B70"  # source | GEOGloWS, SFDC bias corrected
C_GOOGLE = "#2A78D6"  # source | Google GRRR (blue)
C_GLOFAS4 = "#EB6834"  # source | GloFAS v4 (orange)
C_GLOFAS5 = "#B34036"  # source | GloFAS v5 (deep red)
C_SWALIM = INK  # benchmark | SWALIM observed levels
C_SFED = "#0E8A7B"  # benchmark | FloodScan SFED (teal)
SOURCE_COLORS = {
    "geoglows": C_GEOGLOWS,
    "geoglows_sfdc": C_GEOGLOWS_SFDC,
    "google": C_GOOGLE,
    "google_grrr": C_GOOGLE,
    "glofas": C_GLOFAS4,
    "glofas_v4": C_GLOFAS4,
    "glofas_v5": C_GLOFAS5,
    "swalim": C_SWALIM,
    "sfed": C_SFED,
}


# ------------------------------------------------- trigger station restriction
# Decision 2026-08-26: the trigger is built ONLY on SWALIM gauges that are
# still reporting, and ONE model carries each river (not one per season).
#
# Four gauges on the Juba and three on the Shabelle carry the trigger. Only
# five still report daily (Luuq, Dollow, Belet Weyne, Bulo Burti, Jowhar):
# Bardheere ends 2023-11-30 and Bualle 2024-03-14, so those two can still be
# forecast at, but no longer verified against observations. The other
# SWALIM stations are excluded for lack of a usable record, not for lack of
# skill: Kaitoi, Afgoi and Audegle stop in 2008 and Mahadey Weyne in 1990,
# while Jamaame, Mareere, Kamsuma, Mogambo and Balad have SNRFA files with no
# readings at all.
# Listed upstream to downstream, the direction the water travels, so tables and
# the map read in the same order.
TRIGGER_STATIONS = {
    "juba": ["dollow", "luuq", "bardheere", "bualle"],
    "shabelle": ["belet_weyne", "bulo_burti", "jowhar"],
}
ALL_TRIGGER_STATIONS = [s for v in TRIGGER_STATIONS.values() for s in v]

RIVER_MODEL = {"juba": "google_grrr", "shabelle": "glofas_v5"}

# Google's forecast horizon is 7 days, so a 7-12 day readiness leg cannot run
# on it. Readiness therefore stays on GloFAS, whose reforecast covers those
# leads (notebook 10).
READINESS_MODEL = "glofas_v4"


# The adopted trigger: (station return period, stations that must agree) per
# river-season window, and the years the calibration runs over. Kept here so
# notebook 09 and the summary-page generator cannot drift apart.
# Calibrated on the ENVELOPE, not window by window (see ENVELOPE_TARGET_RP
# above and scripts/envelope_search.py): these settings put the union at
# 1-in-2.9 and catch 8 of the 10 severe years, with no activation in a year
# that recorded no flood at all. Majority consensus everywhere, so no single
# gauge can release the money and no single quiet gauge can block it.
# One model per WINDOW (directive 2026-08-27), chosen by
# scripts/model_selection.py: rank the gauges by how well that model tracks the
# reference gauge, require a consensus of them over their own return-period
# thresholds, and judge the union of the four windows. Models are never mixed
# inside a window, so a window is one product plus one rule.
#
# GEOGloWS is excluded from the adopted design (directive 2026-08-28): its
# return periods cannot yet be fitted on its own forecasts, whose archive
# begins in July 2024. Google carries Gu, GloFAS v5 carries Deyr.
#
# Calibrated to activate no more often than 1-in-3 (directive 2026-08-27): 8
# activations in 25 years, 1-in-3.2, catching 8 of the 10 severe years. The
# 9-activation alternative sits at 1-in-2.9 with the same severe coverage;
# this one drops 2010, which was not a severe year.
#
# The backtest does not identify the model on its own: 161 assignments across
# the four windows reach the same 8-of-10 severe-year coverage near 1-in-3.
# Where the envelope is indifferent, the forecast side decides, which is why
# Shabelle Deyr runs on GloFAS: it leads the Shabelle at lead time in both
# seasons, its thresholds come from the v5 reanalysis that matches the
# operational v5 forecast, and the v4 reforecast supplies the lead-time
# evidence.
TRIGGER_CONFIG = {
    ("juba", "gu"): {"source": "google_grrr", "rp": 5, "n_req": 3},
    ("juba", "deyr"): {"source": "glofas_v5", "rp": 4, "n_req": 3},
    ("shabelle", "gu"): {"source": "google_grrr", "rp": 6, "n_req": 2},
    ("shabelle", "deyr"): {"source": "glofas_v5", "rp": 4, "n_req": 2},
}
# The operative source per window, which is what the trigger, the envelope
# search and the summary page all read.
WINDOW_MODEL = {k: v["source"] for k, v in TRIGGER_CONFIG.items()}
# The 1-in-3 target applies to the ENVELOPE, not to each window (directive
# 2026-08-27): the full amount is released whenever any window fires, so the
# union of the four windows is what the budget is sized on. Four windows each
# calibrated to 1-in-3 give a union of about 1-in-1.5, so the per-window
# settings are searched against the union instead (scripts/envelope_search.py).
ENVELOPE_TARGET_RP = 3

# A year counts as severe when the river's reference gauge recorded a 1-in-5
# or rarer season. These are the years the envelope is judged on: at a 1-in-3
# activation rate it cannot catch every RP3 flood, so it should catch the
# worst ones.
SEVERE_RP = 5

TRIGGER_YEARS = (1999, 2023)
BENCHMARK_RP = 3  # a flood year at the reference gauge
REFERENCE_GAUGE = {"juba": "luuq", "shabelle": "belet_weyne"}
