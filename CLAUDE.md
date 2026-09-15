# ds-aa-som-floods

Somalia riverine flood anticipatory-action trigger: analysis (notebooks, `scripts/`, Pages
site under `pages/`) and, since September 2026, the daily **monitoring pipeline**
(`src/monitoring/`, `pipelines/`, `.github/workflows/monitoring.yml`).

## The trigger (do not restate; read `src/constants.py`)

`TRIGGER_CONFIG` is the adopted mechanism: four river-season windows, one source and one rule
each. Gu (Mar–May) on Google Flood Hub, Deyr (Oct–Dec) on GloFAS; Juba needs 3 of 4 points over
their own return-period level on the same forecast day, Shabelle 2 of 3. Any window activating
releases the allocation. Action reads leads 1–7 d on the window's source; readiness reads GloFAS
at leads 8–12 d for every window (`src/monitoring/config.py: READINESS_RULES`). Design pages:
`pages/trigger-single-model/summary.html` (summary) and `pages/trigger-single-model/` (analysis).

## Monitoring pipeline (runbook)

Daily GHA `monitoring.yml` at 16:00 UTC (first full hour after GloFAS lands on EWDS, ~14:15-15:30 UTC), four steps, all from repo root:

1. `pipelines/check_forecasts.py` — GloFAS operational ensemble at the 7 frozen cells
   (`src/monitoring/glofas_cells.json`) from EWDS, Google Flood Hub for the 7 gauges, written to
   blob `monitoring/forecasts/<date>.parquet` (dev), the store every later step reads. No database.
   Ends with `os._exit(0)` (cfgrib teardown segfault on Linux).
2. `pipelines/save_plots.py` — `evaluate.evaluate` + chart → blob
   `projects/ds-aa-som-floods/monitoring/{date}.png` (dev).
3. `pipelines/send_emails.py` — Listmonk via ocha-relay. Lists resolved by tag
   (`ds-aa-som-floods` + `som:info` / `som:trigger` / `som:test`; created by
   `pipelines/setup_som_listmonk_lists.py`, ids 122/123/124 as observed, never hardcoded).
   Cadence: while a window is open, Monday informational + immediate on readiness/action.
   **When no window is open nothing is sent** (the pipeline still runs and the page still updates).
   Windows are open by calendar month (`config.MONITORING_OPEN_MONTHS`): Deyr September to
   January, Gu February to June; every forecast day inside the window counts, so Deyr monitoring starts in September.
4. `pipelines/export_monitoring_status.py` — `status.json` + `latest.png` to the orphan
   `monitoring-status` branch under `pages/monitoring/`; `deploy-pages.yml` (16:45 UTC cron)
   overlays them so <https://ocha-dap.github.io/ds-aa-som-floods/monitoring/> reads them.

Blob (container `projects`, dev, prefix `ds-aa-som-floods/`): raw GloFAS GRIB `raw/glofas/monitoring/`, raw Google
answer `raw/google/monitoring/`, processed rows `monitoring/forecasts/<date>.parquet`, evaluation
`monitoring/status/<date>.json`, chart `monitoring/<date>.png`. There is no database table.

Run modes (KB `infrastructure/email-testing.md`): `TEST_EMAIL`, `DRY_RUN` default **true**;
production needs repo vars `TEST_EMAIL=false`, `DRY_RUN=false`. `SIMULATE_TRIGGER=true` forces an
action activation (tags `[SIM]`; a real-list simulation additionally needs
`ALLOW_REAL_SIMULATION=true`). `MONITORING_DATE=YYYY-MM-DD` re-runs a day (Google always returns
its current forecast).

Secrets: org-level `DSCI_AZ_BLOB_DEV_SAS(_WRITE)`, `DSCI_AZ_DB_DEV_*`, `DSCI_LISTMONK_API_URL`
(mapped to `DSCI_LISTMONK_BASE_URL`), `DSCI_LISTMONK_API_USERNAME/KEY`; repo-level
`GOOGLE_API_KEY`, `CDSAPI_KEY`, `CDSAPI_URL` (= `https://ewds.climate.copernicus.eu/api`).

Local: `.env` (gitignored) with `GOOGLE_API_KEY`; DSCI_* and CDSAPI_* from the shell.
Dependencies for the runner are pinned in `requirements-monitoring.txt` (the analysis extras
in `pyproject.toml`, geoglows → hydrostats, do not build on a clean 3.12 runner).

### The GloFAS version guard — read before touching thresholds

The analysis fitted the Deyr levels on the **GloFAS v5** reanalysis assuming v5 was live. As of
2026-09-14 the operational forecast is **v4** (v4.5, April 2026; v5 pre-operational on EWDS).
`GLOFAS_OPERATIONAL` in `src/monitoring/config.py` selects the climatology; both v4 and v5 levels
(plus the v4 readiness-band levels) are frozen in `src/monitoring/thresholds.json` by
`scripts/build_monitoring_thresholds.py`. `check_forecasts.py` fails (no email) when EWDS lists a
`version_4*` entry under Legacy Versions or the GRIB process ids
(`GLOFAS_EXPECTED_PROCESS`) change — that is the day to flip the constant, rebuild
`pages/glofas-version/` (`scripts/build_glofas_version_page.py`) and re-check the design.
v4 runs ~2–2.5× v5 here; on v4 the adopted Deyr rules over-activate (envelope 1-in-1.9 vs 1-in-3.2)
and never register Deyr 2006/2023 on the Shabelle — see the page.

### Live-feed caveats

- Dollow's Google point is the Juba main-stem gauge `hybas_1121039440` (since 2026-09-15). The
  design's gauge `hybas_1121038740` is the Dawa branch and is **not in the Flood Hub API** (404).
  The Gu levels for Dollow in `src/monitoring/thresholds.json` are fitted on the main-stem gauge's
  retrospective; the analysis pages still use the old point.
- Google's live horizon is issue−2 … issue+5 days, so the action leg reads Google at leads 1–5.
- SWALIM moderate flood risk alerts (readiness) are **not automated**; emails and the page point
  readers to FAO SWALIM.

## Conventions

- Feature branches → PR to `main`; `feat/multisource-trigger` is Pauline's shared branch (fast
  forward only). Schedules only fire from `main`.
- `data/` is gitignored; `scripts/restore_from_blob.py` restores the processed layer needed by
  the threshold and page builders.
- Pages: one site, landing page `pages/index.html` with a card per product; nested pages link
  back via the hero crumb. Palette in `src/constants.py`; site CSS `pages/assets/site.css`.
