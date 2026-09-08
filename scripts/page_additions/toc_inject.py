"""Give every h2 on the trigger page an id and (re)build the contents rail: on wide
screens a sticky left column beside the article, below the hero, with the section in
view highlighted; on narrow screens an inline list above the article.
Idempotent: existing ids, rail, wrapper, style and script are replaced."""
import html as H
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "pages" / "trigger-single-model" / "index.html"
t = PAGE.read_text(encoding="utf-8")

# undo a previous run: rail, style, script, and the body-grid wrapper around the article
t = re.sub(r'\n?<nav class="toc"[^>]*>.*?</nav>\n?', "\n", t, flags=re.S)
t = re.sub(r'\n?<style id="toc-style">.*?</style>\n?', "\n", t, flags=re.S)
t = re.sub(r'\n?<script id="toc-script">.*?</script>\n?', "\n", t, flags=re.S)
t = t.replace('<div class="body-grid">\n', "", 1).replace("\n</div><!-- /body-grid -->", "", 1)


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
    entries.append((sid, H.unescape(re.sub(r"<[^>]+>", "", inner)).strip(), 2))
    return f'<h2 id="{sid}"{m.group(1)}>{inner}</h2>'


def in_details(pos):
    """True when this heading sits inside a <details> block (collapsed by default)."""
    before = t[:pos]
    return before.count("<details") - before.count("</details>") > 0


def add_id_h3(m):
    inner = m.group(2)
    sid = slug(inner)
    while sid in seen:
        sid += "-2"
    seen.add(sid)
    if not in_details(m.start()):
        entries.append((sid, H.unescape(re.sub(r"<[^>]+>", "", inner)).strip(), 3))
    return f'<h3 id="{sid}"{m.group(1)}>{inner}</h3>'


t = re.sub(r'<h2(?: id="[^"]*")?([^>]*)>(.*?)</h2>', add_id, t, flags=re.S)
t = re.sub(r'<h3(?: id="[^"]*")?([^>]*)>(.*?)</h3>', add_id_h3, t, flags=re.S)
entries.sort(key=lambda e: t.find(f'id="{e[0]}"'))

items = "".join(
    f'<li class="lv{lvl}"><a href="#{sid}">{H.escape(txt, quote=False)}</a></li>' for sid, txt, lvl in entries)
style = """<style id="toc-style">
/* clicking a contents entry stops below the sticky provider bar, so the heading shows */
article h2[id], article h3[id] { scroll-margin-top:76px; }
html { scroll-behavior:smooth; }
@media (prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } }
/* contents: inline list above the article on narrow screens */
.toc { margin:22px 44px 0; padding:14px 18px 10px; border:1px solid #e2e7e7; border-radius:5px; background:var(--n05); }
.toc .toc-h { margin:0 0 6px; font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--n7); }
.toc ul { margin:0; padding:0; list-style:none; columns:2; column-gap:28px; font-size:13.5px; line-height:1.7; }
.toc a { color:var(--n8); text-decoration:none; }
.toc a:hover { color:var(--b6); }
.toc li.on > a { color:var(--b6); font-weight:600; }
.toc li.lv3 > a { padding-left:14px; font-size:12.5px; color:var(--n7); }
/* wide screens: a sticky rail in a left column beside the article, starting below the
   hero and the provider bar; the hero text and the bar shift right to line up with the text */
@media (min-width:1100px) {
  .body-grid { display:grid; grid-template-columns:220px minmax(0,860px); align-items:start; }
  .hero.hero-sub .inner, .vbar-in { margin-left:220px; }
  .vbar-in { max-width:860px; }
  .toc { position:sticky; top:54px; margin:22px 0 0 22px; padding:8px 0 8px 0; border:0; border-radius:0;
         background:transparent; max-height:calc(100vh - 70px); overflow:auto; }
  .toc .toc-h { margin:0 0 8px 12px; }
  .toc ul { columns:1; font-size:12.5px; line-height:1.4; border-left:2px solid #e2e7e7; }
  .toc li { margin:0; }
  .toc a { display:block; padding:5px 10px 5px 12px; margin-left:-2px; border-left:2px solid transparent; color:var(--n7); }
  .toc a:hover { color:var(--n9); }
  .toc li.on > a { border-left-color:var(--b6); color:var(--b6); font-weight:600; }
}
@media print { .toc { position:static; } }
</style>
"""
nav = ('<nav class="toc" aria-label="Contents">\n  <p class="toc-h">Contents</p>\n'
       '  <ul>' + items + '</ul>\n</nav>\n')
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
# the rail and the article share a grid wrapper
i = t.find("  <article>")
j = t.find("</article>") + len("</article>")
assert 0 < i < j, "article element not found"
t = (t[:i] + '<div class="body-grid">\n' + nav + t[i:j] + "\n</div><!-- /body-grid -->" + t[j:])
# style goes into the head, script before </body>
t = t.replace("</head>", style + "</head>", 1) if "</head>" in t else style + t
t = t.replace("</body>", script + "</body>", 1) if "</body>" in t else t + script
PAGE.write_text(t, encoding="utf-8")
print("toc entries:", len(entries))
