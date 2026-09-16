"""Shared page chrome for the generated pages under pages/.

Both generated pages (the data-source review and the single-model trigger
variant) sit inside the site's own stylesheet, which supplies the wrap and
hero containers. Everything specific to these pages, the section headings,
tables, tabs, callouts and figures, lives here so the two cannot drift
apart in appearance.
"""

STYLE = '''.hero.hero-sub { padding:40px 44px 34px; }
figure.chart { margin:24px 0 30px; }
.tabbar { display:flex; flex-wrap:wrap; gap:6px; margin:18px 0 0; }
.tab { font:inherit; font-size:13px; padding:7px 13px; cursor:pointer;
  border:1px solid #d3d9e0; background:#f7f9fb; color:#3a4552;
  border-radius:6px; }
.tab:hover { background:#eef2f6; }
.tab.active { background:#1f5c8b; border-color:#1f5c8b; color:#fff;
  font-weight:500; }
.panel { display:none; padding-top:6px; }
.panel.active { display:block; }
.muted { color:#6b7683; }
figure.chart img { width:100%; height:auto; display:block; }
figure.chart figcaption { font-size:13px; color:#55606d; line-height:1.55;
  margin-top:10px; padding-left:2px; }
.hero.hero-sub h1 { font-size:28px; }
.crumb { font-size:12px; margin:0 0 14px; }
.crumb a { color:rgba(255,255,255,.85); text-decoration:none; }
article { padding:8px 44px 24px; max-width:880px; }
article h2 { font-family:'Merriweather',Georgia,serif; font-size:21px; color:var(--n9);
             margin:40px 0 10px; line-height:1.25; }
article h2 .num { color:var(--b5); font-size:15px; margin-right:8px; }
article p, article li { font-size:14.5px; color:var(--n8); line-height:1.65; }
article ul { padding-left:22px; }
article strong { color:var(--n9); }
article a { color:var(--b6); }
article code { background:var(--n05); border:1px solid #e2e7e7; padding:1px 5px;
               border-radius:3px; font-size:12.5px; color:var(--b7); }
table.data { border-collapse:collapse; width:100%; margin:16px 0; font-size:13px; }
table.data th { text-align:left; font-size:11px; text-transform:uppercase;
                letter-spacing:.07em; color:var(--n7); font-weight:700;
                border-bottom:2px solid var(--b1); padding:7px 10px 6px; }
table.data td { border-bottom:1px solid #eef1f1; padding:7px 10px; color:var(--n8); }
table.data td em { font-style:normal; color:var(--b6); font-weight:600; }
.tablewrap { overflow-x:auto; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
         gap:14px; margin:26px 0 6px; }
.stat { background:#fff; border:1px solid #e2e7e7; border-top:3px solid var(--b5);
        border-radius:5px; padding:14px 16px 12px; }
.stat.flag { border-top-color:#e0a04b; }
.stat .v { font-family:'Merriweather',Georgia,serif; font-size:25px; font-weight:700;
           color:var(--b6); line-height:1.1; }
.stat.flag .v { color:#b0722b; }
.stat .l { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
           color:var(--n7); margin-top:6px; line-height:1.45; }
.callout { margin:20px 0; padding:13px 17px; border-radius:4px; background:var(--b05);
           border-left:5px solid var(--b5); font-size:13.5px; color:var(--n8);
           line-height:1.6; }
.callout.warn { background:#fdf3e7; border-left-color:#e0a04b; }
.takeaways { margin:18px 0 6px; padding:0; list-style:none; }
.takeaways li { border-left:3px solid var(--b1); padding:0 0 0 14px; margin:0 0 16px; }
.provenance { margin:34px 44px 40px; padding-top:14px; border-top:1px solid #e2e7e7;
              font-size:12px; color:var(--n7); line-height:1.7; }
.provenance a { color:var(--b6); }
.updated { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
           color:var(--n7); margin:24px 0 0; }
@media (max-width:640px) {
  article { padding:4px 22px 16px; }
  .hero.hero-sub { padding:32px 22px 28px; }
  .provenance { margin:28px 22px 32px; }
}'''
