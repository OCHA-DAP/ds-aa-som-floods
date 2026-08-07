"""Cancel EWDS queue jobs that no local process will ever download.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/cancel_orphaned_ewds_jobs.py <keep_log> [...]

Stopping a download script locally does NOT cancel its already-submitted
EWDS requests — they stay queued server-side, occupying the per-user
processing slots ahead of any newly submitted work. This sweeps the
account queue: every accepted/running job whose request ID does not
appear in one of the given log files (as "Request ID is <uuid>") is
deleted. Pass the log files of every RUN THAT IS STILL ALIVE.
"""

import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.datasources.glofas import _client


def main():
    keep = set()
    for log_path in sys.argv[1:]:
        text = Path(log_path).read_text(encoding="utf8", errors="ignore")
        keep |= set(re.findall(r"Request ID is ([0-9a-f-]{36})", text))
    print(f"Keeping {len(keep)} request IDs from {len(sys.argv) - 1} live log(s)")

    client = _client(wait_until_complete=False).client
    jobs = client.get_jobs(limit=1000).json
    rows = jobs.get("jobs", jobs.get("results", []))
    print(f"Account has {len(rows)} listed jobs")

    n_del = n_keep = 0
    for r in rows:
        uid = r.get("jobID") or r.get("request_uid") or r.get("id")
        status = r.get("status")
        if status not in ("accepted", "running"):
            continue
        if uid in keep:
            n_keep += 1
            continue
        try:
            client.delete(uid)
            n_del += 1
        except Exception as e:
            print(f"  delete failed for {uid}: {e}")
    print(f"Cancelled {n_del} orphaned jobs; left {n_keep} live ones queued.")


if __name__ == "__main__":
    main()
