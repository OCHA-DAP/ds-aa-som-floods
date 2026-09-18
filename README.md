# Somalia Flooding Trigger Framework

This repository contains the analytical framework developed to support anticipatory action (AA) for flooding in Somalia.

## Objective

Develop and evaluate flood trigger thresholds to activate preparedness and early response activities ahead of significant flood events in Somalia.

## Folder Structure

- `analysis/` — Jupyter notebooks for exploratory analysis and trigger simulations
- `data/` — Cached local data files (folder structure tracked, contents gitignored)
- `src/` — Data-source clients, constants, and utility functions
- `src/monitoring/`, `pipelines/` — the daily monitoring pipeline (forecast fetch, trigger evaluation, chart, Listmonk email, status export); see `CLAUDE.md` for the runbook
- `pages/` — the GitHub Pages site: trigger design, live monitoring status (`/monitoring/`) and the GloFAS version note (`/glofas-version/`)

## Usage

1. Clone the repository.
2. Install dependencies with `uv sync` (project uses `pyproject.toml` and `uv.lock`).
3. Set `DSCI_AZ_BLOB_DEV_SAS` (read) and, if running upload steps, `DSCI_AZ_BLOB_DEV_SAS_WRITE` for Azure blob access via `ocha_stratus`.
4. Run the notebooks in `analysis/` for exploratory work.
