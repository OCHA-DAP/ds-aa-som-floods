"""Give every h2 on the trigger page an id and (re)build a table of contents just
before the first h2. Idempotent: existing ids and a previous TOC are replaced."""
import html as H
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model" / "index.html"
t = PAGE.read_text(encoding="utf-8")

# drop a previous TOC
t = re.sub(r'\n?<nav class="toc"[^>]*>.*?</nav>\n?', "\n", t, flags=re.S)


def slug(s):
    s = H.unescape(re.sub(r"<[^>]+>", "", s)).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60]


entries = []
seen = set()


def add_id(m):
    inner = m.group(2)
    sid = slug(inner)
    while sid in seen:
        sid += "-2"
    seen.add(sid)
    entries.append((sid, H.unescape(re.sub(r"<[^>]+>", "", inner)).strip()))
    return f'<h2 id="{sid}"{m.group(1)}>{inner}</h2>'


t = re.sub(r'<h2(?: id="[^"]*")?([^>]*)>(.*?)</h2>', add_id, t, flags=re.S)

items = "".join(f'<li><a href="#{sid}">{H.escape(txt)}</a></li>' for sid, txt in entries)
toc = ('<nav class="toc" aria-label="Contents" style="margin:22px 0 8px;padding:14px 18px;border:1px solid #e2e7e7;'
       'border-radius:5px;background:var(--n05)">\n'
       '  <p style="margin:0 0 6px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--n7)">Contents</p>\n'
       '  <ol style="margin:0;padding-left:20px;columns:2;column-gap:28px;font-size:13.5px;line-height:1.7">' + items + '</ol>\n</nav>\n')
first_h2 = t.find("<h2 ")
t = t[:first_h2] + toc + t[first_h2:]
PAGE.write_text(t, encoding="utf-8")
print("toc entries:", len(entries))
for sid, txt in entries:
    print("  ", sid, "|", txt)
