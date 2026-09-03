"""Write a single self-contained copy of a generated trigger page.

A page under pages/ loads its stylesheet, its figures and (for the two
JS-driven tables) its JSON from sibling files. Opened as a local file that
last part fails: a file:// page cannot fetch its neighbours, so the
selection-detail and activation-impact tables render empty. This inlines all
three so one file behaves like the served page.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/preview_trigger_page.py \\
        pages/trigger-single-model [out.html]
"""

import base64
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml"}


def build(page_dir, out_path):
    page_dir = Path(page_dir)
    html = (page_dir / "index.html").read_text(encoding="utf-8")
    css = (REPO / "pages" / "assets" / "site.css").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="../assets/site.css">',
                        f"<style>\n{css}\n</style>")

    # the JSON the page fetches, handed over as resolved promises instead
    payloads = {}
    for name in re.findall(r'fetch\("([^"]+\.json)"\)', html):
        f = page_dir / name
        if f.exists():
            payloads[name] = json.loads(f.read_text(encoding="utf-8"))
    for name, payload in payloads.items():
        html = html.replace(
            f'fetch("{name}")',
            "Promise.resolve({json:function(){return "
            + json.dumps(payload).replace("</", "<\\/")
            + ";}})",
        )

    def embed(match):
        f = page_dir / match.group(1)
        if not f.exists():
            return match.group(0)
        mime = MIME.get(f.suffix.lower(), "application/octet-stream")
        return f'src="data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}"'

    html, n_figs = re.subn(r'src="((?:figs|figures)/[^"]+)"', embed, html)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"{n_figs} figures and {len(payloads)} payload(s) inlined | "
          f"{out_path.stat().st_size / 1024:.0f} KB")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    page = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else Path(page) / "standalone.html"
    build(page, out)
