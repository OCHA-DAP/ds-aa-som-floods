"""Download GloFAS reanalysis for all Juba/Shabelle stations.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/download_glofas.py                    # v4.0
    .venv/Scripts/python.exe scripts/download_glofas.py version_5_0        # v5.0

v4.0 is the default because the reforecast archive stops there, and
thresholds fitted on a reanalysis must come from the same model version as
the forecasts they will be applied to. v5.0 goes to its own directory
(see glofas.VERSION_DIRS) and appears as source "glofas_v5" in the
processed tables, so the two can be compared side by side.

Downloads river discharge reanalysis (1999-2023) as one all-stations box
per year (EWDS cost scales with time dimensions only, so a whole-domain
request costs the same as a single station and needs 25 queue slots
instead of ~325) into data/glofas/raw/reanalysis_som/<year>.nc. Years
already on disk are skipped, so the script is safe to re-run. Requires
~/.cdsapirc. Reforecast downloads stay event-targeted; use
src.datasources.glofas.download_reforecast_months for those.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.datasources import glofas


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else glofas.DEFAULT_VERSION
    if version not in glofas.VERSION_DIRS:
        raise SystemExit(f"unknown version {version!r}; choose from {list(glofas.VERSION_DIRS)}")
    glofas.download_reanalysis_box(version=version)
    print("\nDone.")


if __name__ == "__main__":
    main()
