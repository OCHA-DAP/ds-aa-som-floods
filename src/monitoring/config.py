"""Monitoring settings that are not part of the trigger definition.

The trigger (windows, sources, return periods, votes) is TRIGGER_CONFIG in
src.constants. Everything here is operational plumbing: which GloFAS system
version is live, lead bands, storage locations, Listmonk lists.
"""

from src.constants import SEASONS, TRIGGER_CONFIG, TRIGGER_STATIONS

PROJECT_PREFIX = "ds-aa-som-floods"
BLOB_STAGE = "dev"
DB_SCHEMA = "projects"
DB_TABLE = "ds_aa_som_floods_monitoring"

# ------------------------------------------------------------ GloFAS version
# The trigger analysis fitted the Deyr thresholds on the GloFAS v5 reanalysis
# on the assumption that v5 was the live system. Checked 2026-09-14: EWDS
# still labels version_4_0 "Operational" and version_5_0 "Pre-operational",
# ECMWF's release wiki lists v4.5 (16 Apr 2026) as the latest operational
# release, and the forecast dataset offers no legacy v4 stream (only 2.1 and
# 3.1), so the `operational` forecast IS v4. Thresholds must come from the
# climatology of the model that produces the forecast they are applied to,
# so this is the one switch: flip it to "glofas_v5" the day EWDS lists a
# version_4_x entry under Legacy Versions (see etl.check_glofas_version).
GLOFAS_OPERATIONAL = "glofas_v4"
GLOFAS_REANALYSIS_VERSION = {"glofas_v4": "version_4_0", "glofas_v5": "version_5_0"}
# Process identifiers stamped in the operational GRIB on 2026-09-13 (v4.5).
# A change means the system version changed; the run then fails loudly.
GLOFAS_EXPECTED_PROCESS = {"generatingProcessIdentifier": 5, "backgroundProcess": 21}
GLOFAS_LEADS = list(range(1, 13))  # days 1..12, GloFAS labelling
GLOFAS_ENSEMBLE_MEMBERS_MIN = 40  # 51 expected; fewer means a truncated download

# Dollow's HYBAS gauge exists in the Google retrospective the Gu levels were
# fitted on but is not served by the live Flood Hub API (404, checked
# 2026-09-14). Probed separately each day so it is picked up if it appears.
GOOGLE_NOT_SERVED = ["dollow"]

# ------------------------------------------------------------- lead bands
# GloFAS "day n" describes calendar day issue+(n-1); Google lead n describes
# issue+n. Both are filtered on each product's own leadtime_days labelling,
# exactly as the backtest did (scripts/page_additions/somlib.first_issue_dates).
ACTION_LEADS = (1, 7)
READINESS_LEADS = (8, 12)

# ------------------------------------------------------- monitoring windows
# Calendar months in which each season's windows are open, i.e. the status is
# evaluated and emails go out: Deyr from September to January, Gu from
# February to June (decided 2026-09-15). The trigger itself still counts only
# forecast days inside the season months (SEASONS in src.constants), so an
# open window with no in-season forecast day yet reads "not activated".
MONITORING_OPEN_MONTHS = {"deyr": (9, 10, 11, 12, 1), "gu": (2, 3, 4, 5, 6)}

# ------------------------------------------------------------- windows
WINDOWS = [("juba", "gu"), ("juba", "deyr"), ("shabelle", "gu"), ("shabelle", "deyr")]
WINDOW_KEY = {w: f"{w[0]}_{w[1]}" for w in WINDOWS}
WINDOW_TITLE = {
    ("juba", "gu"): "Gu Juba",
    ("juba", "deyr"): "Deyr Juba",
    ("shabelle", "gu"): "Gu Shabelle",
    ("shabelle", "deyr"): "Deyr Shabelle",
}
SEASON_TITLE = {"gu": "Gu (March to May)", "deyr": "Deyr (October to December)"}
RIVER_TITLE = {"juba": "Juba", "shabelle": "Shabelle"}
STATION_TITLE = {
    "dollow": "Dollow", "luuq": "Luuq", "bardheere": "Bardheere", "bualle": "Bualle",
    "belet_weyne": "Belet Weyne", "bulo_burti": "Bulo Burti", "jowhar": "Jowhar",
}
SOURCE_TITLE = {"google": "Google Flood Hub", "glofas": "GloFAS"}


def operational_source(analysis_source):
    """Map a source name from the trigger analysis to the live product.

    The analysis names model versions (glofas_v5, google_grrr); the pipeline
    stores products (glofas, google). GloFAS thresholds are looked up under
    GLOFAS_OPERATIONAL, whatever version the analysis assumed.
    """
    if analysis_source.startswith("glofas"):
        return "glofas"
    if analysis_source.startswith("google"):
        return "google"
    raise ValueError(analysis_source)


def threshold_source(product):
    """The climatology a live product's thresholds are fitted on."""
    return {"glofas": GLOFAS_OPERATIONAL, "google": "google_grrr"}[product]


# The readiness leg carries each window's own votes and return period, but
# with GloFAS at leads 8-12 for every window and the return period capped at
# 1-in-5 where the 21-year reforecast archive cannot resolve rarer levels
# (trigger-single-model page, "The readiness leg"). Thresholds are refitted
# on the readiness-band series of the operational GloFAS version.
READINESS_RP_CAP = 5
READINESS_RULES = {
    w: {
        "source": "glofas",
        "rp": min(TRIGGER_CONFIG[w]["rp"], READINESS_RP_CAP),
        "n_req": TRIGGER_CONFIG[w]["n_req"],
        "n_of": len(TRIGGER_STATIONS[w[0]]),
    }
    for w in WINDOWS
}
ACTION_RULES = {
    w: {
        "source": operational_source(TRIGGER_CONFIG[w]["source"]),
        "rp": TRIGGER_CONFIG[w]["rp"],
        "n_req": TRIGGER_CONFIG[w]["n_req"],
        "n_of": len(TRIGGER_STATIONS[w[0]]),
    }
    for w in WINDOWS
}


def season_of(month):
    for season, months in SEASONS.items():
        if month in months:
            return season
    return None


# ------------------------------------------------------------- Listmonk
LISTMONK_PROJECT_TAG = "ds-aa-som-floods"
LISTMONK_LISTS = {
    "info": {"name": "[AA framework] Somalia riverine flooding (informational)", "tag": "som:info"},
    "trigger": {"name": "[AA framework] Somalia riverine flooding (trigger)", "tag": "som:trigger"},
    "test": {"name": "[TEST] Somalia riverine flooding", "tag": "som:test", "extra_tags": ["TEST"]},
}
EMAIL_SUBJECT_PREFIX = "Somalia AA: Riverine Flooding"
CONTACT_NAME = "Tristan Downing"
CONTACT_EMAIL = "tristan.downing@un.org"

# ------------------------------------------------------------- outputs
STATUS_DIR = "pages/monitoring"  # on the orphan monitoring-status branch too
STATUS_BRANCH = "monitoring-status"
PAGES_URL = "https://ocha-dap.github.io/ds-aa-som-floods/"
REPO_URL = "https://github.com/OCHA-DAP/ds-aa-som-floods"
