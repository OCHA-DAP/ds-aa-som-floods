# The Pages site

This directory is the repo's one GitHub Pages site (team convention: one repo, one
Pages site with a landing page at the root — see `methods/static-data-apps.md` in
`ds-knowledge-base`). `.github/workflows/deploy-pages.yml` publishes `pages/` verbatim
as the site root on every push to `main` that touches it; there is no build step.

The repo is private, so the site lives on a private `*.pages.github.io` subdomain and
is visible to OCHA-DAP organisation members only.

Layout:

- `index.html` — the landing page: one card per product. Styling is the HDX v2 tokens +
  particle hero, copied from `ds-seas5-skill` (lineage:
  `ds-geospatial-impact-estimates`), so team sites read as one system.
- `assets/` — `site.css` (shared tokens) and `hero.js` (decorative particle hero).
- `trigger/` — the multi-source trigger analysis report. Its figures (`trigger/figs/`)
  and `trigger/summary.json` are written by `analysis/11_multisource_trigger.ipynb`;
  re-run the notebook after changing the analysis, then update the prose in
  `trigger/index.html` to match.

## Adding a page

Create a directory under `pages/` with an `index.html`, then add a card to
`pages/index.html` — copy an existing `<a class="k">` block and change the href, title,
blurb and foot. If contributions ever start overlapping, switch to the manifest-driven
variant (`ds-storm-impact-harmonisation/pages/README.md` § "Reusing this in another
repo") instead of growing the hand-edited list.
