# EF5 modeling for Somalia floods — handoff notes (July 2026)

Working notes from an exploratory EF5 modeling session (Claude Code,
Jul 17–21 2026; skill test completed Jul 22). All data produced lives on
Azure Blob — nothing local.

## Where everything is

Container `projects`, stage `dev`, prefix **`ds-aa-som-floods/raw/ef5/`**
(5,589 blobs, ~1.2 GB). Start with its `README.md`, which documents the
full layout. Load with the usual pattern:

```python
import ocha_stratus as stratus
df = stratus.load_csv_from_blob(
    "ds-aa-som-floods/raw/ef5/output_hist/2023/ts.beletweyne.crest.csv",
    stage="dev",
)
```

Highlights:

- `output_hist/<1996–2025>/ts.<station>.crest.csv` — 30-year OND hindcast,
  13 stations (7 Shabelle, 6 Jubba), daily discharge. Uncalibrated params.
- `analysis/beletweyne_real_discharge.json` (and `buloburte_…`) — real
  SWALIM daily discharge (m³/s) back to 1963, scraped from the
  snrfa.faoswalim.org station pages (the level CSVs on the portal don't
  include discharge; the station pages embed it in the Highcharts JS).
- `analysis/*.html` — interactive charts of the main results.
- `software/ef5_software_backup.zip` — EF5 v1.2.3 Windows binary + source.
- `precip/`, `pet/`, `static/`, `forecast/` — all model inputs, EF5-ready.

## Model setup

EF5/CREST + kinematic wave routing, HydroSHEDS 30″ terrain (clipped to
38–49°E, 2°S–11°N), CHIRPS v2 daily rain, flat monthly PET climatology.
Gauges snapped to flow-accumulation channels; contributing areas verified
against published basin areas. Each season run independently (IWU=30%,
no warm-up). On Windows, EF5's Linear Reservoir routing never writes
discharge — use `ROUTING=KW` (this cost half a day to find).

## Key findings

1. **Timing good, magnitude bad (uncalibrated).** EF5 independently ranks
   1997, 2006, 2019, 2023 as the top OND flood years and gets rise onset
   within ~1 week for about half of 113 comparable station-years — but
   overestimates peak discharge ~20× vs SWALIM observed discharge.
2. **Calibration (Belet Weyne, 10 OND seasons vs real discharge):** best
   single uniform parameter set is WM≈1300–2800, B≈0.24–0.28, KE=1.0
   (rest default), log-RMSE ≈1.60 vs 2.02 uncalibrated. The error surface
   is extremely flat, and error *direction* flips year to year (2× over
   some years, 4–5× under in others) — a single fixed parameter set
   cannot capture inter-annual variability. Validation on held-out years
   confirmed modest overfitting (1.65 vs 1.33 on a 3-year subset).
3. **The rainfall forecast is the binding constraint for AA, not the
   hydro model.** Real CHIRPS-GEFS forecasts issued Nov 3 and Nov 13 2023
   both missed the Nov 18 Belet Weyne peak because they underestimated
   mid-Nov rainfall ~5× even at 5-day lead. See
   `analysis/ef5_forecast_test.html`.
4. **SWALIM level data caps at Bank Full** (e.g. flat 8.3 m at Belet Weyne
   for two weeks, Nov 2023) — plateaus mean "≥", not an exact reading.
   Bardheere's gauge was destroyed 8 Nov 2023; Bualle's broke 14 Mar 2024.

## Forecast skill test (COMPLETED Jul 22 2026)

All 60 CHIRPS-GEFS issue dates (6/season × 10 OND seasons 2015–2025,
excl. 2020) fetched and run through EF5. Root cause of the earlier failed
run: both `control_skill_test*.txt` files were missing the
`[PETForcing PET]` section — fixed version is
`control/control_skill_test_fixed.txt`. Outputs in `output_skill_test/`,
metrics + per-window table in `analysis/skill_test_results.json`.

Setup: trigger = max EF5 discharge at Belet Weyne over the 6-day window
(issue..issue+5) ≥ Q*; observed event = SWALIM level ≥ 6.5 m (Moderate
Flood Risk) any day in the window. 60 windows, 11 with observed events.

Results at Q\*=2100 m³/s: **forecast POD 0.55 / FAR 0.00 / CSI 0.55**;
observed-rain hindcast over the same windows: POD 0.55 / FAR 0.14. Key
readings:

1. **Forecast ≈ hindcast at 6-day windows.** Most detection skill comes
   from the observed-rain model state at issue time, not the GEFS rain —
   most event windows were already flooding at issue (only 2 of 11 were
   true onsets: 2019-10-08 hit by both, 2024-11-05 missed by both).
2. **2024 is an initial-condition failure, not a forecast failure —
   RESOLVED (Jul 22).** Deyr-2024 rain was only moderate (137 mm upper-
   Shabelle OND total vs 220–286 mm in 2023/2019; every sub-box below
   those years), but Belet Weyne entered October 2024 at **6.61 m —
   already above the 6.5 m moderate threshold** and the highest season-
   start level on record (next highest ~5.4 m), after the wettest Jun–Sep
   (mean 5.48 m; sustained Ethiopian Kiremt flow). The seasonal EF5 runs
   start Oct 1 from a fixed dry state (IWU=30 %, zero channel flow), so
   they structurally cannot reproduce carryover-driven floods. AA
   implication: either run EF5 continuously (or warm up with Jun–Sep
   rain), or add the current observed river level as a trigger predictor
   alongside the model. Data: `analysis/deyr2024_comparison.json`.
   Side-finding: SWALIM's published discharge is a fixed deterministic
   rating transform of level (identical level→Q pairs pre/post 2024), so
   it contains no information beyond the level series.
3. GEFS underestimates the biggest peaks (at Q*≥6000 the hindcast keeps
   POD 0.36 vs forecast 0.18–0.27), consistent with the Nov-2023 pilot
   finding that CHIRPS-GEFS misses extreme rain even at short lead.
4. The 2019-11-19 event window missed at 2002 vs Q*=2100 — marginal; and
   the single "false alarm" (2023-10-22, fc 1730 at lower Q*) peaked at
   obs 6.28 m, just below the 6.5 m threshold — arguably not wrong.

Caveat: with only 2 onset windows the headline POD mostly measures
"is the river already high at issue date" — for AA lead-time value,
extend to more issue dates around onset periods.

**Level-aware trigger test (Jul 22,
`analysis/combined_trigger_results.json`):** over the same 60 windows, a
trigger on the *observed level alone* (≥6.0 m at issue date) scores
POD 1.0 / FAR 0.15 — strictly better than model+forecast (POD 0.55 /
FAR 0.0); "level ≥6.5 OR model Q≥2100" gives POD 0.91 / FAR 0.0. At
≤6-day horizons persistence dominates and the hydro model adds nothing
at Belet Weyne. The model's value proposition is therefore longer leads
and onset timing — which is exactly where CHIRPS-GEFS rainfall is the
binding constraint. Any operational trigger should include the current
SWALIM level as a predictor.

## Calibrated re-runs (Jul 22, `analysis/calibrated_results.json`)

Re-ran the 30-yr OND hindcast (both basins, `output_hist_cal2/`) and the
60-window skill test (`output_skill_cal/`) with the per-station cascade
parameters:

- **Magnitude fixed:** calibrated seasonal peaks at Belet Weyne are a
  median 1.7× the SWALIM rating discharge (uncalibrated was ~20×). The
  observed rating caps at bank-full (~470–530 m³/s), so part of the
  residual gap is the cap, not model error.
- **Calibrated trigger threshold:** Q* = 610 m³/s reproduces the observed
  moderate-flood season frequency (11 of 24 seasons with data); 7 of the
  model's top-11 years match observed flood seasons.
- **Detection unchanged:** calibrated model at Q*=610 scores the same
  POD 0.55 / FAR 0.0 as the uncalibrated run — calibration rescales
  magnitude but does not change which events CHIRPS-driven runs can see.
  The observed-level trigger still dominates at 6-day windows.
- Summary chart: `analysis/ef5_trigger_evidence.html` (blob) — stat
  tiles, trigger comparison, 30-yr hindcast vs observed floods,
  magnitude scatter.

## Overnight runs Jul 22→23 (accuracy push)

All forcings are now observation-based and complete on blob:
- `precip/` covers **every day Jan 1996 – Jan 2026** (10,959 clips).
- `pet_real/` is **Hobbins RefET** (CHC's CHIRPS-companion reference ET),
  monthly mm/day 1996–2026, replacing the flat climatology.

**Continuous 30-year run (`output_continuous/`, calibrated params + real
PET, one unbroken 1996–2026 simulation per basin):**
1. **It catches Deyr 2024.** Oct 1 2024 carries 6,137 m³/s of antecedent
   flow (the seasonal run starts dry); the OND-2024 peak is well above
   any trigger. Confirms the initial-condition diagnosis.
2. **Best year-ranking yet:** frequency-matched threshold (≈9,900 m³/s)
   puts 8 of 11 observed moderate-flood seasons in its top-11 (seasonal
   calibrated: 7/11) — and 2024 is one of them.
3. **CORRECTION (Jul 23): that run had zero PET.** EF5's built-in TIFF
   reader silently failed on the rasterio-written `pet_real/` grids
   (missing forcings are treated as 0), so the run above was a
   no-evapotranspiration model — which is the real cause of the
   saturation/wet bias, not just parameter mismatch. Diagnostic: EF5's
   output PET column read 0.00; files rewritten via `gdal_translate`
   (EF5 reads those) now show ~0.28 mm/h correctly. All 361 `pet_real/`
   grids fixed and re-uploaded. The 2024-catch and 8/11-ranking results
   still stand as reported (they came from that run as it was), but the
   recalibrated continuous run with WORKING real PET
   (`output_continuous_cal/`, `analysis/continuous_calibration.json`,
   `analysis/continuous_cal_verdict.json`) supersedes it.
   Lesson for the archive: EF5 only reads GDAL-written uncompressed
   untiled GeoTIFFs — always pass rasterio-produced rasters through
   `gdal_translate -co TILED=NO -co COMPRESS=NONE`, and verify by
   checking the PET/Precip columns in `ts.*.crest.csv`, since EF5 never
   errors on unreadable forcings.

**Warm-started skill test (Jun 1 start, IWU=30, `output_skill_warm/`,
`analysis/warm_skill_results.json`):** a June warm-up alone does NOT fix
2024-type windows (its 2024 window maxima stay ≤380 m³/s) — the carryover
signal needs the true multi-year state, not just Kiremt-season spin-up.

**Onset-densified skill test (21 extra weekly issue dates through OND
2019/2023/2024, warm-started, `output_skill_onset/`,
`analysis/onset_skill_windows.json`):** two genuine pre-flood event
windows found (level just under 6.5 m at issue): 2023-10-29 (fc max
868 m³/s — detectable) and 2024-10-15 (fc max 407 — marginal). Combined
with 2019-10-08 from the cold set, the onset sample is now 3: the
forecast chain detects ~2 of 3 onsets a few days ahead.

**Multi-station triggers (`analysis/multistation_thresholds.json`):**
frequency-matched calibrated Q* — Bulo Burte ≈810 (4/8 top-year matches),
Luuq ≈1,170 (4/7), Bardheere ≈1,560 (6/11; treat obs with caution).

## Open threads
- **30-year MAM/Gu hindcast — DONE (Jul 22).** All Mar–Jun CHIRPS days
  1996–2025 downloaded (blob `precip/` now has them; MAM PET months were
  copied from the flat climatology). Outputs on blob `output_hist_mam/`
  (30 years × 13 stations). Skill at Belet Weyne mirrors OND: the model's
  top-6 Gu years (2021, 2016, 2018, 2020, 2024, 2023) are exactly the six
  years whose observed Gu peak hits the SWALIM rating cap (~470 m³/s —
  the obs can't rank *within* that group); Spearman r = 0.815 over 23
  years with obs. Same caveat as OND: uncalibrated magnitudes ~10–20×
  observed-rating discharge.
- **Per-station calibration — DONE (Jul 22).** Real discharge scraped for
  all SNRFA stations that publish it (only 4 do: Belet Weyne, Bulo Burte,
  Luuq back to 1951, Bardheere back to 1963 — the other 7 pages embed
  all-null `flow_daily`; JSONs on blob `analysis/`). Cascade calibration
  (upstream fixed, then downstream local sub-basin swept, log1p-RMSE on
  10 OND seasons 2002–2021) in `analysis/cascade_state_{shabelle,jubba}.json`:
  - Shabelle: BeletWeyne WM=2800 B=0.28 (prior); BuloBurte local best
    WM=4000 B=0.45 (1.703, surface flat 1.70–1.74 → upstream dominates);
    downstream stations inherit.
  - Jubba: Luuq (Dollow floats with it) WM=4000 B=0.45 (1.476 vs 1.807
    uncalibrated); Bardheere local best WM=60 B=0.45 (2.103 — shallow
    optimum, notably worse fit than Luuq; treat Bardheere obs with
    caution, gauge history is messy). Saakow/Bualle/Jilib inherit.
  - Pattern everywhere: big gain leaving default WM=60, then a very flat
    valley for WM≈1300–4000; B≈0.45 slightly preferred at local stages.
  - Uniform-grid transfer test at Bulo Burte
    (`analysis/buloburte_calibration.json`): the Belet Weyne optimum
    scores 1.713 vs 1.673 for the site-best — calibration transfers.
- **Gu 2024 hindcast** got flood onset right (rise on exactly Apr 26, the
  real 6.5 m crossing date) but peaked ~2 weeks early (May 5 vs May 20) —
  unresolved whether upstream rain or routing is the cause.
- Rating curves in `analysis/rating_curves.json` are rough Manning's
  approximations superseded by the scraped real discharge — prefer the
  latter.
