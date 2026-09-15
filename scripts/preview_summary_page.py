"""Write a single self-contained copy of the summary page, for viewing.

The published page pulls the shared stylesheet and its charts from sibling
files, which is right for GitHub Pages but means the file cannot be opened
or sent on its own. This inlines the stylesheet and every chart so one file
renders anywhere.

Usage (from repo root, after scripts/build_summary_page.py):
    .venv/Scripts/python.exe scripts/preview_summary_page.py [out.html]
"""

import base64
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGES = REPO / "pages"
DEFAULT_OUT = REPO / "pages" / "summary" / "standalone.html"


def build(out_path):
    html = (PAGES / "summary" / "index.html").read_text(encoding="utf-8")
    css = (PAGES / "assets" / "site.css").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="../assets/site.css">', f"<style>\n{css}\n</style>"
    )
    html = re.sub(r'<script src="\.\./assets/hero\.js"></script>\s*', "", html)

    def embed(match):
        raw = (PAGES / "summary" / match.group(1)).read_bytes()
        return f'src="data:image/svg+xml;base64,{base64.b64encode(raw).decode()}"'

    html, n = re.subn(r'src="(figures/[^"]+)"', embed, html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"{n} charts inlined | {out_path.stat().st_size / 1024:.0f} KB")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
