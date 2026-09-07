# Trigger-page additions (2026-09-07)

Generators for the sections added to `pages/trigger-single-model/index.html` on 7 September 2026.
Run from this directory with the repo venv (`../../.venv/Scripts/python.exe`), in this order:

1. `metrics_build.py` -> `metrics.json`, `metrics_tables.html`, `figs/l_roc.png` (skill scores, ROC, AUC)
2. `station_metrics.py` -> `station_metrics.json`, `station_metrics_table.html`
3. `model_timeline.py` -> `model_timeline.json`, `figs/l_model_timeline.png`
4. `obs_trigger.py` -> `obs_fallback.json`, `obs_fallback_table.html`, `figs/l_obs_fallback.png`
5. `page_assemble.py` (inserts or replaces the four sections between HTML comment markers; moves the
   station-level studies once) then `toc_inject.py` (ids on every h2 and the contents list).

`swalim_timeline.py` and `swalim_window.py` draw the SWALIM comparison charts; the SWALIM section
itself was written from bulletin texts (see the page) and its generator depends on session files.
`somlib.py` holds the shared conventions: local parquet under `data/processed/`, the two-gauge
benchmark as `scripts/envelope_search.py` computes it (levels fitted 2000-2023), activation dates
by flow day and by forecast issue date. Intermediate JSON/HTML land next to the scripts.

The data under `data/processed/` is not in git. When running from a worktree without it, set
`SOM_DATA_REPO` to a checkout that has the data (the scripts read only; they write figures and
JSON next to themselves and into `pages/trigger-single-model/`).
