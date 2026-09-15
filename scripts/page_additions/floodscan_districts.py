"""FloodScan flood exposure (FloodScan x WorldPop, the team's flood-exposure pipeline, DB table
app.floodscan_exposure) for the 14 anticipatory-action districts, daily 1998-2023, summed per river.
Caches the pull as floodscan_exposure_14.parquet and writes inundation_series.parquet
(date, river, value = people exposed) for the timing scripts. Then checks whether the two-gauge
benchmark flood years rank high in this series."""
import json
from pathlib import Path
import pandas as pd
import ocha_stratus as stratus
import somlib as L

S = Path(__file__).parent
DISTRICTS = {"juba": {"SO2605": "Doolow", "SO2606": "Luuq", "SO2602": "Baardheere", "SO2703": "Saakow", "SO2701": "Bu'aale", "SO2702": "Jilib"},
             "shabelle": {"SO2001": "Beledweyne", "SO2002": "Bulo Burto", "SO2003": "Jalalaqsi", "SO2101": "Jowhar", "SO2103": "Balcad", "SO2302": "Afgooye", "SO2305": "Qoryooley", "SO2301": "Marka"}}
cache = S / "floodscan_exposure_14.parquet"
if not cache.exists():
    codes = [c for d in DISTRICTS.values() for c in d]
    eng = stratus.get_engine(stage="prod")
    with eng.connect() as con:
        look = pd.read_sql("select adm2_pcode, adm2_name from app.admin_lookup where adm2_pcode in %(c)s", con, params={"c": tuple(codes)})
        print(look.sort_values("adm2_pcode").to_string(index=False))
        df = pd.read_sql("select valid_date, pcode, sum from app.floodscan_exposure where iso3='SOM' and adm_level='2' and pcode in %(c)s and valid_date <= '2023-12-31'", con, params={"c": tuple(codes)})
    df["valid_date"] = pd.to_datetime(df["valid_date"]); df.to_parquet(cache)
df = pd.read_parquet(cache)
river_of = {c: r for r, d in DISTRICTS.items() for c in d}
df["river"] = df.pcode.map(river_of)
ser = df.groupby(["valid_date", "river"])["sum"].sum().reset_index().rename(columns={"valid_date": "date", "sum": "value"})
ser.to_parquet(S / "inundation_series.parquet")
print(f"{len(df):,} district-days, {df.pcode.nunique()} districts, {df.valid_date.min().date()} to {df.valid_date.max().date()}")

m = json.load(open(S / "metrics.json")); win = {w["window"]: w for w in m["windows"]}
for season in ("deyr", "gu"):
    for river in ("juba", "shabelle"):
        w = win[f"{season.title()} {river.title()}"]; fl, sv = set(w["flood_years"]), set(w["severe_years"])
        s = ser[ser.river == river].set_index("date")["value"]; s = s[s.index.month.isin(L.SEASONS[season])]
        am = s.groupby(s.index.year).max(); am = am[(am.index >= 1998) & (am.index <= 2023)]
        rk = am.rank(ascending=False); rp = (len(am) + 1) / rk
        top = am.sort_values(ascending=False)
        print(f"\n{season.title()} {river.title()}: seasonal max people exposed in the river's AA districts, 1998-2023. F = benchmark flood, S = severe")
        print("  top 8:", ", ".join(f"{y}{' S' if y in sv else ' F' if y in fl else ''} ({v/1000:.0f}k, 1-in-{rp[y]:.0f})" for y, v in top.head(8).items()))
        print("  benchmark years:", ", ".join(f"{y}{' S' if y in sv else ''}: rank {int(rk[y])} ({am[y]/1000:.0f}k)" for y in sorted(fl) if y in rk.index))
        print(f"  benchmark years in top 8: {sum(1 for y in fl if y in rk.index and rk[y] <= 8)} of {len(fl)}; severe in top 8: {sum(1 for y in sv if y in rk.index and rk[y] <= 8)} of {len(sv)}")
