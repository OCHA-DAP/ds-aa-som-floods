"""Download GloFAS v4 reforecast (leads 1-7 days) for all Juba/Shabelle stations.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/download_glofas_reforecast.py

Downloads the control + perturbed ensemble reforecast at daily lead times
1-7 days for the Gu (Mar-Jun) and Deyr (Oct-Dec) seasons, 2003-2023 (EWDS
reforecast is frozen at GloFAS v4.2 and ends 2023-11-25), as one
all-stations box per chunk. Chunks are sized against the live EWDS cost
limit and submitted with a rolling window; files already on disk are
skipped, so the script is safe to re-run and to resume.

This is the lead-time counterpart to the reanalysis in download_glofas.py:
reanalysis measures how well the model reproduces the river given observed
rainfall (a skill ceiling), whereas the reforecast is what an operational
trigger would actually have seen at 1-7 days notice.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.datasources import glofas


def main():
    glofas.download_reforecast_box()
    print("\nDone.")


if __name__ == "__main__":
    main()
