"""Give every h2 on the trigger page an id and (re)build the contents panel: a
floating panel at the right of the article on wide screens, an inline list before
the first h2 on narrow ones, with the current section highlighted while scrolling.
Idempotent: existing ids, panel, style and script are replaced."""
import html as H
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model" / "index.html"
t = PAGE.read_text(encoding="utf-8")

# drop a previous panel, style and script
t = re.sub(r'\n?<nav class="toc"[^>]*>.*?</nav>\n?', "\n", t, flags=re.S)
t = re.sub(r'\n?<style id="toc-style">.*?</style>\n?', "\n", t, flags=re.S)
t = re.sub(r'\n?<script id="toc-script">.*?</script>\n?', "\n", t, flags=re.S)


def slug(s):
    s = H.unescape(re.sub(r"<[^>]+>", "", s)).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60]


entries, seen = [], set()


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
style = """<style id="toc-style">
.toc { margin:22px 0 8px; padding:14px 18px; border:1px solid #e2e7e7; border-radius:5px; background:var(--n05); }
.toc .toc-h { margin:0 0 6px; font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--n7); }
.toc ol { margin:0; padding-left:20px; columns:2; column-gap:28px; font-size:13.5px; line-height:1.7; }
.toc a { color:var(--n8); text-decoration:none; }
.toc a:hover { text-decoration:underline; }
.toc li.on > a { color:var(--b6); font-weight:600; }
/* wide screens: the .wrap is 1080px centred and the article 860px at its left, leaving a
   220px gutter on the right for a panel that stays put while the page scrolls */
@media (min-width:1100px) {
  .toc { position:fixed; top:64px; right:calc(50% - 540px + 14px); width:190px; max-height:calc(100vh - 90px);
         overflow:auto; margin:0; padding:12px 14px; background:#fff; box-shadow:0 1px 6px rgba(26,39,51,.06); z-index:30; }
  .toc ol { columns:1; font-size:12.5px; line-height:1.55; padding-left:16px; }
  .toc li { margin:3px 0; }
}
@media print { .toc { position:static; } }
</style>
"""
nav = ('<nav class="toc" aria-label="Contents">\n  <p class="toc-h">Contents</p>\n'
       '  <ol>' + items + '</ol>\n</nav>\n')
script = """<script id="toc-script">
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a[href^="#"]'));
  var heads = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
  function update() {
    var y = window.scrollY + 90, cur = -1;
    for (var i = 0; i < heads.length; i++) { if (heads[i] && heads[i].offsetTop <= y) cur = i; }
    links.forEach(function (a, i) { a.parentNode.classList.toggle('on', i === cur); });
  }
  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
})();
</script>
"""
first_h2 = t.find("<h2 ")
t = t[:first_h2] + style + nav + t[first_h2:]
t = t.replace("</body>", script + "</body>", 1) if "</body>" in t else t + script
PAGE.write_text(t, encoding="utf-8")
print("toc entries:", len(entries))
