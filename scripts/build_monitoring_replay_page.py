"""Build pages/monitoring-replay/: the live monitoring page re-run on a past season.

For every archived GloFAS issue of the season, the pipeline's own evaluation
(evaluate.evaluate) and chart (plot.monitoring_chart) are run unchanged, against
today's rules and thresholds. index.html is generated from
pages/monitoring/index.html: that page's two scripts are kept and fed one issue at
a time, so the replay shows what the live page would have shown that day. Re-run
`build` after editing the monitoring page, the trigger config or the thresholds.

Deyr 2023 (the El Nino floods) is the season replayed, from one of two archives:

- operational: the 50-member operational ensemble as it was issued each day, the
  product the live pipeline reads. Fetched from EWDS by the pipeline's own request.
- reforecast: the GloFAS v4 reforecast the trigger's backtests ran on (11 members,
  two issues a week), already in data/processed/.

From repo root (set SOM_DATA_REPO when running from a worktree):

    .venv/bin/python scripts/build_monitoring_replay_page.py fetch
        EWDS -> data/glofas/raw/forecast_operational_box/glofas_forecast_<date>.grib,
        one day per request and one request at a time: EWDS runs one job at a time
        per account, and the daily monitoring run queues on the same service.
        About five minutes a day. Resumable.
    .venv/bin/python scripts/build_monitoring_replay_page.py process [--upload]
        GRIB -> data/processed/monitoring_replay_deyr2023.parquet, by the pipeline's
        own etl.process_glofas, in the monitoring store's schema. --upload puts it
        on blob (dev) so scripts/restore_from_blob.py can bring it back.
    .venv/bin/python scripts/build_monitoring_replay_page.py build [--source reforecast]
        -> pages/monitoring-replay/ (index.html, replay.json, charts/). The default
        source is the operational parquet.

Nothing here reads or writes the monitoring store (monitoring/ on blob) or the
monitoring-status branch.
"""

import io
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "page_additions"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")  # ocha_stratus reads the blob credentials at import time

import pandas as pd  # noqa: E402

from src.constants import (ALL_TRIGGER_STATIONS, BENCHMARK_RP, BLOB_STAGE, STATIONS,  # noqa: E402
                           TRIGGER_STATIONS, TRIGGER_YEARS)
from src.monitoring import config as cfg  # noqa: E402

DATA = Path(os.environ.get("SOM_DATA_REPO", REPO)) / "data"
RAW = DATA / "glofas" / "raw" / "forecast_operational_box"
PROCESSED = DATA / "processed" / "monitoring_replay_deyr2023.parquet"
PROCESSED_BLOB = f"{cfg.PROJECT_PREFIX}/processed/{PROCESSED.name}"
OUT = REPO / "pages" / "monitoring-replay"
LIVE_PAGE = REPO / "pages" / "monitoring" / "index.html"

SEASON, YEAR = "deyr", 2023
DAYS = [d.date() for d in pd.date_range("2023-10-01", "2023-11-30")]  # the whole of the 2023 floods
# The archived forecasts are GloFAS version 4, so the replay only holds while the
# thresholds are fitted on the version 4 record (build() refuses otherwise).
ARCHIVE_VERSION = "glofas_v4"
VERSION_WORD = ARCHIVE_VERSION.rsplit("_v", 1)[1]
# The Somalia Humanitarian Fund's allocation for the El Nino floods, the date readers
# ask about. Announced on the seasonal outlook, before this trigger existed.
SHF_DATE = date(2023, 10, 12)
SHF_URL = ("https://reliefweb.int/report/somalia/somalia-humanitarian-fund-allocates-15-million-"
           "support-communities-highest-risk-el-nino-induced-flooding")
CHART_DPI = 130


def day_month(d):
    return f"{d.day} {d:%b}"


# ------------------------------------------------------------------- fetch
def raw_path(issue):
    return RAW / f"glofas_forecast_{issue}.grib"


def fetch():
    from src.monitoring import etl

    RAW.mkdir(parents=True, exist_ok=True)
    todo = [d for d in DAYS if not raw_path(d).exists()]
    print(f"[replay] {len(DAYS) - len(todo)} of {len(DAYS)} issues on disk, fetching {len(todo)}", flush=True)
    for n, issue in enumerate(todo, 1):
        part = raw_path(issue).with_suffix(".part")
        etl.download_glofas(issue, part)  # the pipeline's own request; blocks until done
        part.rename(raw_path(issue))
        print(f"[replay] {issue} [{n}/{len(todo)}]", flush=True)


# ----------------------------------------------------------------- process
def process(upload=False):
    from src.monitoring import etl

    missing = [str(d) for d in DAYS if not raw_path(d).exists()]
    if missing:
        raise SystemExit(f"[replay] {len(missing)} issue dates have no GRIB (run fetch): {', '.join(missing)}")
    frames = []
    for issue in DAYS:
        ids = etl.grib_process_ids(raw_path(issue))
        df = etl.process_glofas(raw_path(issue), issue)  # the pipeline's own function, unchanged
        if pd.Timestamp(df["issued_time"].iloc[0]).date() != issue:
            raise SystemExit(f"[replay] {raw_path(issue).name} holds the {df['issued_time'].iloc[0]} issue")
        df["model_version"] = (f"{ARCHIVE_VERSION} operational, as issued "
                               f"(gpi{ids['generatingProcessIdentifier']}/bp{ids['backgroundProcess']})")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)[etl.COLUMNS]
    # the same normalisation etl.save_rows applies before the store's parquet is written
    out["monitoring_date"] = pd.to_datetime(out["monitoring_date"])
    out["valid_date"] = pd.to_datetime(out["valid_date"])
    out["issued_time"] = pd.to_datetime(out["issued_time"], utc=True)
    rows = out.groupby("monitoring_date").size()
    expected = len(ALL_TRIGGER_STATIONS) * len(cfg.GLOFAS_LEADS)
    if not (rows == expected).all():
        raise SystemExit(f"[replay] issues without {expected} rows:\n{rows[rows != expected]}")
    runs = sorted(out.model_version.unique())
    if len(runs) != 1:
        # two fingerprints mean the system changed inside the season: the page says it did not
        raise SystemExit(f"[replay] the archive is not one GloFAS system: {runs}")
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED, index=False)
    print(f"[replay] wrote {PROCESSED} ({len(out)} rows, {len(DAYS)} issues, run: {runs[0]})")
    if upload:
        import ocha_stratus as stratus

        stratus.upload_blob_data(PROCESSED.read_bytes(), PROCESSED_BLOB, stage=BLOB_STAGE,
                                 content_type="application/octet-stream")
        print(f"[replay] uploaded {PROCESSED_BLOB} ({BLOB_STAGE})")


# ---------------------------------------------------------- forecast frames
def by_issue(df):
    """Monitoring-store rows -> {issue date: one day's frame, as etl.load_day returns it}."""
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["monitoring_date"] = pd.to_datetime(df["monitoring_date"])
    df["issued_time"] = pd.to_datetime(df["issued_time"], utc=True)
    return {d.date(): g.sort_values(["source", "station", "valid_date"]).reset_index(drop=True)
            for d, g in df.groupby("monitoring_date")}


def operational_frames():
    if not PROCESSED.exists():
        raise SystemExit(f"{PROCESSED} is missing: restore it (scripts/restore_from_blob.py) or run fetch + process")
    return by_issue(pd.read_parquet(PROCESSED))


def reforecast_frames():
    """The season's reforecast issues in the monitoring store's schema: per station and
    valid day the member median and spread, exactly what etl.process_glofas keeps."""
    from src.monitoring import etl

    parts = []
    for name in ("reforecast_glofas_v4", "reforecast_glofas_v4_lead8_12"):  # leads 1-7 and 8-12
        d = pd.read_parquet(DATA / "processed" / f"{name}.parquet")
        d = d[d.station.isin(ALL_TRIGGER_STATIONS)
              & d.issued_time.between(pd.Timestamp(DAYS[0]), pd.Timestamp(DAYS[-1]))]
        parts.append(d)
    mem = pd.concat(parts, ignore_index=True)
    g = mem.groupby(["issued_time", "station", "valid_time", "leadtime_days"])["discharge"]
    out = g.agg(value="median", value_min="min", value_max="max", n_members="count",
                value_p25=lambda x: x.quantile(0.25), value_p75=lambda x: x.quantile(0.75)).reset_index()
    out = out.rename(columns={"valid_time": "valid_date"})
    out["source"] = "glofas"
    out["river"] = out["station"].map(lambda s: STATIONS[s].river)
    out["monitoring_date"] = out["issued_time"].dt.normalize()
    out["model_version"] = f"{ARCHIVE_VERSION} reforecast ({int(out.n_members.min())} members)"
    return by_issue(out[etl.COLUMNS])


SOURCES = {
    "operational": {
        "frames": operational_frames,
        "hero": "the GloFAS forecast issued on each day",
        "banner": "operational, as issued",
        "about": ("The GloFAS operational ensemble as it was issued each day ({n} members, days 1 to 12), "
                  "retrieved from the Copernicus archive. This is the product the live pipeline reads."),
    },
    "reforecast": {
        "frames": reforecast_frames,
        "hero": "each archived GloFAS reforecast issue",
        "banner": "reforecast archive",
        "about": ("The GloFAS reforecast archive the trigger was tested on: {n} ensemble members, two issues a "
                  "week. The live pipeline reads the full operational ensemble every day, so the first "
                  "day on which a trigger is reached is only known to within the gap between two issues."),
    },
}


# ------------------------------------------------------------------- build
def save_chart(fig, path):
    """PNG at the page's size, reduced to a 256-colour palette (a flat chart loses nothing)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    plt.close(fig)
    img = Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    img.save(path, format="PNG", optimize=True)


def evaluate_days(frames, out_dir):
    """Issue date -> the status.json the live export would have written that day."""
    from pipelines.export_monitoring_status import levels_for_page, series_for_page
    from src.monitoring import evaluate, plot
    from src.monitoring import thresholds as thr

    levels_df = thr.load()
    (out_dir / "charts").mkdir()
    generated = datetime.now(timezone.utc).isoformat()
    days = {}
    for d in sorted(frames):
        df = frames[d]
        result = evaluate.evaluate(df, d, levels_df)
        save_chart(plot.monitoring_chart(df, result, levels_df), out_dir / "charts" / f"{d}.png")
        # the same fields, in the same order, as pipelines/export_monitoring_status.main
        status = {
            **result,
            "generated_at": generated,
            "chart": f"charts/{d}.png", "chart_stale": False,
            "glofas_operational": cfg.GLOFAS_OPERATIONAL,
            "lead_bands": {"action": list(cfg.ACTION_LEADS), "readiness": list(cfg.READINESS_LEADS)},
            "levels": levels_for_page(levels_df),
            "series": series_for_page(df),
            "stations": {r: list(s) for r, s in TRIGGER_STATIONS.items()},
            "titles": {"station": cfg.STATION_TITLE, "river": cfg.RIVER_TITLE, "source": cfg.SOURCE_TITLE,
                       "season": cfg.SEASON_TITLE},
            "swalim_note": ("The framework also allows the readiness trigger to be reached on a SWALIM moderate "
                            "flood risk alert for either river. Bulletins are not read by this pipeline."),
        }
        days[d] = json.loads(json.dumps(status, default=str))
    return days


def top_station(status, river, leg):
    """Highest forecast peak on a leg, as % of the station's own threshold (None if no forecast day)."""
    stations = status["windows"][f"{river}_{SEASON}"][leg]["stations"]
    return max((v["pct_of_threshold"] for v in stations.values() if v["pct_of_threshold"] is not None),
               default=None)


def first_reached(days, key):
    return next((d for d in sorted(days) if days[d][key]), None)


def rivers_of(status, key):
    return " and ".join(cfg.RIVER_TITLE[status["windows"][k]["river"]] for k in status[f"{key}_windows"])


def gauge_onsets():
    """River -> the day its second gauge reached its own benchmark level that season
    (the two-gauge benchmark of the Timing of activations page)."""
    import somlib as L

    out = {}
    for river in TRIGGER_STATIONS:
        hit = L.gauge_crossings(river, SEASON, BENCHMARK_RP).get(YEAR)
        if hit is not None:
            out[river] = hit.date()
    return out


def issue_notes(days, onsets):
    """One or two sentences per issue for the replay bar, all read off the evaluation."""
    dates = sorted(days)
    daily = all((b - a).days == 1 for a, b in zip(dates, dates[1:]))
    first = {"readiness": first_reached(days, "readiness"), "action": first_reached(days, "action")}
    before = max((d for d in dates if d < SHF_DATE), default=None)
    notes = {}
    for i, d in enumerate(dates):
        s = days[d]
        parts = []
        if d == SHF_DATE:
            parts.append("The day the Somalia Humanitarian Fund allocation was announced.")
        elif d == before and SHF_DATE not in days:
            parts.append("The last archived issue before the Somalia Humanitarian Fund allocation was announced "
                         f"on {day_month(SHF_DATE)}.")
        for key, name in (("readiness", "readiness"), ("action", "activation")):
            if d == first[key]:
                votes = ", ".join(f"{s['windows'][k][key]['max_votes']} of {s['windows'][k][key]['n_of']} stations"
                                  for k in s[f"{key}_windows"])
                parts.append(f"First issue on which the {name} trigger is reached: {rivers_of(s, key)}, {votes}.")
        if d in first.values() and not daily and i:
            parts.append(f"The issue before it is {day_month(dates[i - 1])}, so on daily forecasts the first day "
                         f"would fall from {day_month(dates[i - 1] + pd.Timedelta(days=1))} to {day_month(d)}.")
        tops = [top_station(s, r, "action") for r in ("juba", "shabelle")]
        if not s["action"] and not s["readiness"] and None not in tops:
            parts.append(f"Highest forecast in the activation band: Juba {tops[0]:.0f}% of its threshold, "
                         f"Shabelle {tops[1]:.0f}%.")
        for river, onset in onsets.items():
            if (dates[i - 1] if i else d - pd.Timedelta(days=1)) < onset <= d:
                when = "on this day" if d == onset else f"on {day_month(onset)}"
                parts.append(f"Observed {when}: the second {cfg.RIVER_TITLE[river]} gauge reached its own "
                             f"1-in-{BENCHMARK_RP} level.")
        notes[str(d)] = " ".join(parts)
    return notes


def about_html(days, onsets, source, n_members):
    dates = sorted(days)

    def first_txt(key, name):
        first = first_reached(days, key)
        if first is None:
            return f"The {name} trigger is not reached on any issue replayed."
        return f"The {name} trigger is first reached on the {day_month(first)} issue ({rivers_of(days[first], key)})."

    quiet = [r for r in TRIGGER_STATIONS
             if not any(days[d]["windows"][f"{r}_{SEASON}"][leg]["activated"] for d in dates
                        for leg in ("readiness", "action"))]
    items = [
        "<strong>What is replayed.</strong> Everything below the bar is the monitoring page's own code, fed with "
        "the pipeline's own evaluation and chart of the forecast issue selected.",
        f"<strong>Forecasts.</strong> {SOURCES[source]['about'].format(n=n_members)} GloFAS was already on version "
        f"{VERSION_WORD} in October {YEAR}, the version today's thresholds are fitted on.",
        f"<strong>Result.</strong> {first_txt('readiness', 'readiness')} {first_txt('action', 'activation')}",
        f"<strong>The Somalia Humanitarian Fund allocation.</strong> The Fund announced a $15 million allocation "
        f"for early action and response on {day_month(SHF_DATE)} {SHF_DATE.year}, on the seasonal El Ni&ntilde;o "
        f'outlook (<a href="{SHF_URL}">OCHA, {day_month(SHF_DATE)} {SHF_DATE.year}</a>). This trigger did not '
        "exist then and was not the basis for it.",
        f"<strong>Thresholds are today's.</strong> They are fitted on the GloFAS version {VERSION_WORD} record for "
        f"{TRIGGER_YEARS[0]} to {TRIGGER_YEARS[1]}, so the {YEAR} season is part of the record it is judged against.",
    ]
    for river in quiet:
        hi = [x for x in (top_station(days[d], river, "action") for d in dates) if x is not None]
        seen = (f"Two {cfg.RIVER_TITLE[river]} gauges had reached their own 1-in-{BENCHMARK_RP} level by "
                f"{day_month(onsets[river])}. " if river in onsets else "")
        items.append(
            f"<strong>The {cfg.RIVER_TITLE[river]}.</strong> Neither trigger is reached on the {cfg.RIVER_TITLE[river]} "
            f"on any issue replayed. {seen}The highest forecast in the activation band stays between {min(hi):.0f}% "
            f"and {max(hi):.0f}% of its threshold. The <a href=\"../glofas-version/\">GloFAS version switch</a> page "
            f"describes this gap in version {VERSION_WORD}.")
    items.append('<strong>The whole season on one chart</strong> (model, gauges and flood exposure): '
                 f'<a href="../activation-timing/#{SEASON}{YEAR}">Timing of activations, {SEASON.title()} {YEAR}</a>.')
    lis = "\n".join(f"        <li>{x}</li>" for x in items)
    return f"""
    <div class="replay-about">
      <h2>About this replay</h2>
      <ul>
{lis}
      </ul>
    </div>"""


REPLAY_CSS = """
/* replay bar and note; everything else on this page is the monitoring page's own CSS */
#replaybar { position:sticky; top:0; z-index:20; background:#1f2324; color:#fff; padding:9px 44px 8px; box-shadow:0 2px 8px rgba(0,0,0,.18); }
#replaybar .rb-top { display:flex; flex-wrap:wrap; align-items:center; gap:6px 12px; font-size:13.5px; line-height:1.4; }
#replaybar .rb-tag { background:#f2c94c; color:#1f2324; font-weight:700; font-size:11px; letter-spacing:.08em; text-transform:uppercase; padding:2px 8px; border-radius:3px; }
#replaybar .rb-days { display:flex; flex-wrap:wrap; align-items:center; gap:3px; margin-top:7px; }
#replaybar button { font:inherit; cursor:pointer; }
#replaybar .day { width:23px; height:22px; padding:0; border-radius:4px; font-size:11.5px; background:#565f61; border:1px solid #565f61; color:#fff; }
#replaybar .day.ready { background:#D48F2A; border-color:#D48F2A; color:#1f2324; }
#replaybar .day.act { background:#c9483c; border-color:#c9483c; }
#replaybar .day.cur { outline:2px solid #fff; outline-offset:1px; font-weight:700; }
#replaybar .nav { background:none; border:1px solid #6b7577; color:#fff; border-radius:4px; height:22px; padding:0 9px; font-size:11.5px; }
#replaybar .nav:disabled { opacity:.35; cursor:default; }
#replaybar .mon { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#9aa4a6; width:34px; }
#replaybar .brk { flex-basis:100%; height:0; }
#replaybar .mark { font-size:11.5px; color:#f2c94c; border-left:2px solid #f2c94c; padding:1px 3px 1px 5px; white-space:nowrap; }
#replaybar .mark.obs { color:#9cc7e6; border-left-color:#9cc7e6; }
#replaynote { background:#2a3133; color:#d6dddd; padding:8px 44px 9px; font-size:12.5px; line-height:1.5; }
#replaynote .rb-key { font-size:11.5px; color:#aab3b5; margin-top:4px; line-height:1.6; }
#replaynote .rb-key .sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin:0 4px 0 10px; }
#replaynote .rb-key .sw:first-child { margin-left:0; }
.replay-about { margin:20px 0 0; padding:14px 18px; border-radius:6px; background:#fffbea; border:1px solid #f0e2a8; border-left:5px solid #f2c94c; }
.replay-about h2 { margin:0 0 6px; font-size:16px; }
.replay-about ul { margin:6px 0 0; padding-left:20px; }
.replay-about li { font-size:13.5px; margin:3px 0; }
/* the day buttons wrap into a tall bar on a narrow screen: let it scroll away there */
@media (max-width:900px) { #replaybar { position:static; padding:9px 22px 8px; } #replaynote { padding:8px 22px 9px; } }
"""

# The controller answers the page's own fetch("status.json") with the issue selected,
# then calls the page's two scripts again. They overwrite what they wrote last time.
REPLAY_JS = r"""
(function () {
  const realFetch = window.fetch.bind(window);
  const shared = {};                       // districts.json and rivers.json, read once from the live page
  let R = null, cur = null;
  const answer = body => new Response(JSON.stringify(body), {status: 200, headers: {"Content-Type": "application/json"}});
  window.fetch = function (url, opts) {
    const key = String(url).split("?")[0];
    if (R && key === "status.json") return Promise.resolve(answer(R.status[cur]));
    if (key === "districts.json" || key === "rivers.json") {
      if (!shared[key]) {
        shared[key] = realFetch("../monitoring/" + key).then(r => { if (!r.ok) throw new Error(key + " " + r.status); return r.json(); });
        shared[key].catch(() => { delete shared[key]; });   // a failed read is tried again on the next day picked
      }
      return shared[key].then(answer);
    }
    return realFetch(url, opts);
  };
  const bar = document.getElementById("replaybar");
  const dm = s => new Date(s + "T00:00:00Z").toLocaleDateString("en-GB", {day: "numeric", month: "short", timeZone: "UTC"});
  const cls = s => s.action ? "act" : s.readiness ? "ready" : "";
  function drawDays() {
    let h = "", month = "";
    const marks = R.marks.slice();
    const flag = m => `<span class="mark ${m.cls}" title="${m.label}">${m.short}</span>`;
    R.dates.forEach(d => {
      if (d.slice(0, 7) !== month) { h += (month ? '<span class="brk"></span>' : "") + `<span class="mon">${dm(d).split(" ")[1]}</span>`; month = d.slice(0, 7); }
      while (marks.length && marks[0].date <= d) h += flag(marks.shift());
      const s = R.status[d], label = `${dm(d)}: ${s.status.toLowerCase()}`;
      h += `<button class="day ${cls(s)}" data-d="${d}" title="${label}" aria-label="${label}">${+d.slice(8)}</button>`;
    });
    marks.forEach(m => { h += flag(m); });
    document.getElementById("rb-days").innerHTML = h;
  }
  function show(d) {
    cur = d;
    try { history.replaceState(null, "", "#" + d); } catch (e) {}
    bar.querySelectorAll(".day").forEach(b => {
      b.classList.toggle("cur", b.dataset.d === d);
      if (b.dataset.d === d) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
    });
    const i = R.dates.indexOf(d);
    document.getElementById("rb-prev").disabled = i === 0;
    document.getElementById("rb-next").disabled = i === R.dates.length - 1;
    document.getElementById("rb-cur").textContent = `forecast issued ${dm(d)} ${d.slice(0, 4)}`;
    document.getElementById("rb-note").innerHTML = `<b>Forecast issued ${dm(d)} ${d.slice(0, 4)}.</b> ` + (R.notes[d] || "");
    window.__renderStatus(); window.__renderMap();
  }
  const step = k => { const i = R.dates.indexOf(cur) + k; if (i >= 0 && i < R.dates.length) show(R.dates[i]); };
  const fromHash = () => R.dates.includes(location.hash.slice(1)) ? location.hash.slice(1) : null;
  realFetch("replay.json").then(r => { if (!r.ok) throw new Error(r.status + " " + r.statusText); return r.json(); }).then(data => {
    R = data;
    drawDays();
    bar.addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      if (b.id === "rb-prev") step(-1); else if (b.id === "rb-next") step(1); else if (b.dataset.d) show(b.dataset.d);
    });
    document.addEventListener("keydown", e => {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      const a = document.activeElement;   // leave the arrows alone inside anything else that takes focus
      if (a && a !== document.body && !bar.contains(a)) return;
      if (e.key === "ArrowLeft") step(-1); else if (e.key === "ArrowRight") step(1);
    });
    window.addEventListener("hashchange", () => { const d = fromHash(); if (d && d !== cur) show(d); });
    show(fromHash() || R.start);
  }).catch(err => {
    document.getElementById("banner").innerHTML = `<span class="status">No replay data</span><span class="meta">Could not load replay.json (${err.message}).</span>`;
  });
})();
"""


def build_index(days, onsets, source, n_members):
    """pages/monitoring/index.html as a replay: its wording adapted, its two scripts made callable.
    Every edit must match the live page exactly once, so a reworded live page stops the build."""
    html = LIVE_PAGE.read_text(encoding="utf-8")

    def sub(pattern, repl, flags=0):
        nonlocal html
        html, n = re.subn(pattern, lambda m: repl, html, count=1, flags=flags)
        if n != 1:
            raise SystemExit(f"pages/monitoring/index.html has changed: no match for {pattern!r}")

    def swap(old, new, count=1):
        nonlocal html
        if html.count(old) != count:
            raise SystemExit(f"pages/monitoring/index.html has changed: expected {count} of {old!r}, "
                             f"found {html.count(old)}")
        html = html.replace(old, new)

    first, last = min(days), max(days)
    span = f"from {day_month(first)} to {day_month(last)} {YEAR}"
    sub(r"<title>.*?</title>",
        f"<title>Monitoring replay: {SEASON.title()} {YEAR} &middot; Somalia Riverine Flood Trigger</title>")
    sub(r'<meta name="description" content="[^"]*">',
        f'<meta name="description" content="The Somalia riverine flood monitoring page re-run on GloFAS '
        f'forecasts issued {span}, with today\'s trigger rules and thresholds.">')
    swap("/ live monitoring</p>", "/ monitoring replay</p>")
    swap("<h1>Live monitoring</h1>", f"<h1>Monitoring replay: {SEASON.title()} {YEAR}</h1>")
    sub(r"<p>Daily status of the anticipatory action trigger.*?</p>",
        f"<p>The <a href=\"../monitoring/\" style=\"color:#fff\">live monitoring page</a>, re-run on "
        f"{SOURCES[source]['hero']} {span}, with today's trigger rules and thresholds. This is a "
        "reconstruction of a past season, not a status.</p>", flags=re.S)
    # wording that only fits a live status
    swap("<h2>Latest forecasts against the trigger thresholds</h2>",
         "<h2>The forecast against the trigger thresholds</h2>")
    swap(" In July and August the pipeline still runs and this page still updates, but no email is sent.", "")
    swap("· currently monitoring <b>${seasonNow}</b>", "· monitoring <b>${seasonNow}</b>")
    swap("(operational), ${s.n_glofas_points} stations · checked ${fmtDate(s.monitoring_date)}",
         f"({SOURCES[source]['banner']}), ${{s.n_glofas_points}} stations · "
         "replayed for ${fmtDate(s.monitoring_date)}", count=2)
    gauges = "; ".join(f"{cfg.RIVER_TITLE[r]} {day_month(d)}" for r, d in onsets.items())
    bar = f"""
  <div id="replaybar">
    <div class="rb-top"><span class="rb-tag">Replay &middot; not live</span><span><b>{SEASON.title()} {YEAR}</b> &middot; <span id="rb-cur"></span></span><button class="nav" id="rb-prev" aria-label="Previous issue">&#9664;</button><button class="nav" id="rb-next" aria-label="Next issue">&#9654;</button><span style="color:#aab3b5;font-size:12px">pick a day, or use the arrow keys</span></div>
    <div class="rb-days" id="rb-days"></div>
  </div>
  <div id="replaynote">
    <div id="rb-note"></div>
    <div class="rb-key"><span class="sw" style="background:#565f61"></span>trigger not reached<span class="sw" style="background:#D48F2A"></span>readiness trigger reached<span class="sw" style="background:#c9483c"></span>activation trigger reached &nbsp;&middot;&nbsp; <span style="color:#f2c94c">SHF</span>: Somalia Humanitarian Fund allocation announced, {day_month(SHF_DATE)} &nbsp;&middot;&nbsp; <span style="color:#9cc7e6">gauges</span>: second gauge on that river over its own 1-in-{BENCHMARK_RP} level ({gauges})</div>
  </div>
"""
    sub(r"</header>\s*<article>", "</header>\n" + bar + "\n  <article>" + about_html(days, onsets, source, n_members))
    sub(r"</style>\s*</head>", REPLAY_CSS + "</style>\n</head>")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    if not (len(scripts) == 2 and 'fetch("status.json' in scripts[0] and '"districts.json"' in scripts[1]):
        raise SystemExit("pages/monitoring/index.html has changed: expected the status script, then the map script")
    for name, body in zip(("__renderStatus", "__renderMap"), scripts):
        html = html.replace(f"<script>{body}</script>", f"<script>window.{name} = function () {{{body}}};</script>")
    sub(r"</body>", f"<script>{REPLAY_JS}</script>\n</body>")
    return html


def build(source="operational"):
    if cfg.GLOFAS_OPERATIONAL != ARCHIVE_VERSION:
        raise SystemExit(
            f"The replay reads {ARCHIVE_VERSION} forecasts, but the thresholds now come from the "
            f"{cfg.GLOFAS_OPERATIONAL} record (src/monitoring/config.py), which is not the system that made "
            "them. Decide how the page should read (pin the old levels, or retire it) before rebuilding.")
    frames = SOURCES[source]["frames"]()
    n_members = int(min(df["n_members"].min() for df in frames.values()))
    onsets = gauge_onsets()
    # built aside and swapped in whole, so a failed build leaves the published page as it was
    tmp = Path(tempfile.mkdtemp(prefix=".monitoring-replay-", dir=OUT.parent))
    try:
        days = evaluate_days(frames, tmp)
        dates = sorted(days)
        marks = [{"date": str(SHF_DATE), "cls": "", "short": "SHF",
                  "label": f"Somalia Humanitarian Fund allocation announced, {day_month(SHF_DATE)}"}]
        marks += [{"date": str(d), "cls": "obs", "short": f"{cfg.RIVER_TITLE[r]} gauges",
                   "label": f"Second {cfg.RIVER_TITLE[r]} gauge over its own 1-in-{BENCHMARK_RP} level, {day_month(d)}"}
                  for r, d in onsets.items()]
        replay = {
            "season": SEASON, "year": YEAR, "source": source,
            "dates": [str(d) for d in dates],
            # open on the day readers ask about, or the last issue before it
            "start": str(max((d for d in dates if d <= SHF_DATE), default=dates[0])),
            "marks": sorted(marks, key=lambda m: m["date"]),
            "notes": issue_notes(days, onsets),
            "status": {str(d): days[d] for d in dates},
        }
        (tmp / "replay.json").write_text(json.dumps(replay, separators=(",", ":")) + "\n", encoding="utf-8")
        (tmp / "index.html").write_text(build_index(days, onsets, source, n_members), encoding="utf-8")
        tmp.chmod(0o755)
        if OUT.exists():
            shutil.rmtree(OUT)
        tmp.rename(OUT)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"[replay] wrote {OUT} from the {source} archive: {len(dates)} issues, "
          f"{day_month(dates[0])} to {day_month(dates[-1])} {YEAR}, {n_members} members")
    for key, name in (("readiness", "readiness"), ("action", "activation")):
        d = first_reached(days, key)
        print(f"[replay]   {name} trigger first reached: {d if d else 'never'}"
              + (f" ({', '.join(days[d][f'{key}_windows'])})" if d else ""))


if __name__ == "__main__":
    args = sys.argv[1:]
    step = args[0] if args else ""
    if step == "fetch":
        fetch()
    elif step == "process":
        process(upload="--upload" in args)
    elif step == "build":
        source = args[args.index("--source") + 1] if "--source" in args else "operational"
        if source not in SOURCES:
            raise SystemExit(f"--source must be one of {', '.join(SOURCES)}")
        build(source)
    else:
        raise SystemExit(__doc__)
    # cfgrib/eccodes can segfault during interpreter teardown on Linux; all work is
    # done and flushed by here (as in pipelines/check_forecasts.py).
    sys.stdout.flush()
    os._exit(0)
