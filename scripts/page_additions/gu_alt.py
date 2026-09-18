"""What the envelope looks like if the Gu windows run on GloFAS instead of Google.

Answers the working-group objection to Google Flood Hub in Gu. Prints, per configuration,
the envelope activation years, severe-year coverage, activations with no 1-in-3 flood
behind them, and the per-window activation years. Return periods and vote counts are held
at the adopted settings in every configuration (TRIGGER_CONFIG, 2026-09-18: Shabelle Deyr
1-in-5), so the comparison isolates the model.
"""
import somlib as L

CONFIGS = {
    "adopted (Google in Gu, v5 in Deyr)": {
        ("juba", "gu"): ("google_grrr", 5, 3), ("juba", "deyr"): ("glofas_v5", 4, 3),
        ("shabelle", "gu"): ("google_grrr", 6, 2), ("shabelle", "deyr"): ("glofas_v5", 5, 2)},
    "no Google: GEOGloWS on Juba Gu, v5 elsewhere": {
        ("juba", "gu"): ("geoglows", 5, 3), ("juba", "deyr"): ("glofas_v5", 4, 3),
        ("shabelle", "gu"): ("glofas_v5", 6, 2), ("shabelle", "deyr"): ("glofas_v5", 5, 2)},
    "GloFAS v5 in Gu too": {
        ("juba", "gu"): ("glofas_v5", 5, 3), ("juba", "deyr"): ("glofas_v5", 4, 3),
        ("shabelle", "gu"): ("glofas_v5", 6, 2), ("shabelle", "deyr"): ("glofas_v5", 5, 2)},
    "GloFAS v4 in Gu (what runs live)": {
        ("juba", "gu"): ("glofas_v4", 5, 3), ("juba", "deyr"): ("glofas_v5", 4, 3),
        ("shabelle", "gu"): ("glofas_v4", 6, 2), ("shabelle", "deyr"): ("glofas_v5", 5, 2)},
    "GloFAS v4 everywhere": {
        ("juba", "gu"): ("glofas_v4", 5, 3), ("juba", "deyr"): ("glofas_v4", 4, 3),
        ("shabelle", "gu"): ("glofas_v4", 6, 2), ("shabelle", "deyr"): ("glofas_v4", 5, 2)},
}

sev_all, flood_all = set(), set()
for river in ("juba", "shabelle"):
    for season in ("deyr", "gu"):
        fl, sv, _ = L.benchmark(river, season)
        sev_all |= sv; flood_all |= fl

for label, cfg in CONFIGS.items():
    env, per = set(), {}
    for (river, season), (model, rp, n_req) in cfg.items():
        yrs = set(L.activation_years(model, river, season, rp, n_req))
        per[f"{season} {river}"] = sorted(yrs)
        env |= yrs
    caught = sorted(sev_all & env)
    print(f"--- {label}")
    print(f"    activations   {len(env)} in 25: {sorted(env)}")
    print(f"    severe caught {len(caught)} of {len(sev_all)}: {caught}")
    print(f"    severe missed {sorted(sev_all - env)}")
    print(f"    outside the flood benchmark {sorted(env - flood_all)}")
    for k, v in per.items():
        print(f"      {k:16} {v}")
