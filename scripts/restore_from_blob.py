"""Restore blob-mirrored data to the local data/ tree (inverse of upload_to_blob).

Usage (from repo root):
    .venv/bin/python scripts/restore_from_blob.py [blob_prefix ...]

Blob names follow the upload_to_blob.py convention: ds-aa-som-floods/raw/<x>
maps to data/<x>, and ds-aa-som-floods/processed/<x> to data/processed/<x>.
Files that already exist locally are skipped, so this is safe to re-run.

With no arguments, restores the prefixes the analysis notebooks need that
scripts in this repo can't rebuild from scratch locally.
"""

import sys
from pathlib import Path

import ocha_stratus as stratus

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import BLOB_STAGE  # noqa: E402

DATA_DIR = repo_root / "data"
PROJECT_PREFIX = "ds-aa-som-floods/"

DEFAULT_PREFIXES = [
    "ds-aa-som-floods/processed/",
    "ds-aa-som-floods/raw/glofas/raw/reanalysis_som_ext/",
    "ds-aa-som-floods/raw/glofas/raw/reforecast_som_ext_lead8_12/",
]


def local_path(blob_name):
    rel = blob_name[len(PROJECT_PREFIX):]
    if rel.startswith("raw/"):
        rel = rel[len("raw/"):]
    return DATA_DIR / rel


def main(prefixes):
    cc = stratus.get_container_client(stage=BLOB_STAGE, container_name="projects")
    n_dl = n_skip = 0
    for prefix in prefixes:
        blobs = list(cc.list_blobs(name_starts_with=prefix))
        names = {b.name for b in blobs}
        for blob in blobs:
            # zero-byte "directory marker" blobs shadow real directories
            if blob.size == 0 and any(n.startswith(blob.name + "/") for n in names):
                continue
            dest = local_path(blob.name)
            if dest.exists() and dest.stat().st_size == blob.size:
                n_skip += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(cc.download_blob(blob.name).readall())
            n_dl += 1
            print(f"restored {blob.name} ({blob.size / 1e6:.1f} MB)")
    print(f"Done: {n_dl} restored, {n_skip} already present.")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_PREFIXES)
