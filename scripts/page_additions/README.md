# Trigger-page additions (2026-09-07)

Generators for the sections added to `pages/trigger-single-model/index.html` on 7 September 2026.
Run from this directory with the repo venv (`../../.venv/Scripts/python.exe`), in this order:

1. `metrics_build.py` -> `metrics.json`, `metrics_tables.html`, `figs/l_roc.png` (skill scores, ROC, AUC)
2. `station_metrics.py` -> `station_metrics.json`, `station_metrics_table.html`
3. `model_timeline_all.py` -> `model_timeline_all.json`, `figs/l_years_<season>_<river>.png`
4. `obs_trigger.py` -> `obs_fallback.json`, `obs_fallback_table.html`, `figs/l_obs_fallback.png`
5. `page_assemble.py` (inserts or replaces the four sections between HTML comment markers; moves the
   station-level studies once) then `toc_inject.py` (ids on every h2 and the contents list).

`swalim_station_parse.py` extracts the per-gauge readings from the bulletin texts
(`swalim_station_records.csv`, not in git) and `swalim_station_timeline.py` draws the
station-level charts; when the csv is absent it reads `swalim_station_steps.json`, the saved
per-station record of SWALIM's steps. `swalim_timeline.py` and `swalim_window.py` draw the
river-level SWALIM charts (SWALIM bulletin dates hand-read; model, forecast and gauge dates
computed from `somlib`). `trigger_detail.py` writes `trigger_detail.json` (activation dates and
points over, per SWALIM-matched season) and `gu2024_issue.json` holds the Gu 2024 operational
replay; `swalim_section.py` writes the SWALIM section of the trigger page from these and checks
the gauge dates in its narrative cells against `somlib.gauge_crossings`. Run `toc_inject.py`
after it. `lead_band_design.py` scores the design rule by lead band on the two processed GloFAS
v4 tables (leads 1-7 and 8-12, 2003-2023) for the summary page.
`ensemble_agreement_figs.py`, with `scripts/prob_sweep.py`, makes the figures and the years table for
`pages/ensemble-agreement/`. `swalim_seasons.json` is the per-season SWALIM flood-risk level record
(2006-2023), kept as data; the summary page no longer quotes a return period from it.
`somlib.py` holds the shared conventions: local parquet under `data/processed/`, the two-gauge
benchmark as `scripts/envelope_search.py` computes it (levels fitted 2000-2023), activation dates
by flow day and by forecast issue date. Intermediate JSON/HTML land next to the scripts.

The data under `data/processed/` is not in git. When running from a worktree without it, set
`SOM_DATA_REPO` to a checkout that has the data (the scripts read only; they write figures and
JSON next to themselves and into `pages/trigger-single-model/`).

`explain_pass.py` fills the explanation gaps the flow audit found (acronyms expanded at
first use, the envelope defined, issue date against valid day). `deai_pass.py`,
`deai_pass2.py` and `deai_pass3.py` are one-off prose passes (em dashes to
colons or commas, figurative phrasing removed). They are asserted and idempotent: rerunning
reports every pair as already applied.
