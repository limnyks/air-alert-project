"""Step 8a: find each oblast capital's names in both sources (D23/D24).

Prints only. Local data only (official file from Step 5, alerts.in.ua snapshot from Step 7a).
alerts.in.ua output is aggregated (counts, dates); no raw records are printed (terms 9.2).

The capital names are NOT assumed: the script searches the data with a few name stems per
capital and prints every candidate. We then write the mapping table from what we see.

Parts:
  1. Per oblast: official raion/hromada candidates (+ oblast-wide rows) and alerts.in.ua
     location candidates, with counts, dates, alert types and long alerts.
  2. alerts.in.ua alerts that started before the snapshot window: do they exist in the
     official file? (the only time overlap between train and test sources)

Run from the project root:
    python scripts/step08a_find_capitals.py | tee logs/step08a_output.txt
"""
import numpy as np
import pandas as pd

from step05_inspect_data import OFFICIAL, download, load, section
from step07b_compare_reference import load_snapshot

# region (volunteer/manifest naming) -> (capital, lowercase name stems to search for)
CAPITALS = {
    "Vinnytska oblast": ("Vinnytsia", ["vinnyts"]),
    "Volynska oblast": ("Lutsk", ["lutsk"]),
    "Dnipropetrovska oblast": ("Dnipro", ["dnipro city", "dniprovsk", "dnipro "]),
    "Donetska oblast": ("Kramatorsk", ["kramatorsk"]),
    "Zhytomyrska oblast": ("Zhytomyr", ["zhytomyr"]),
    "Zakarpatska oblast": ("Uzhhorod", ["uzhhorod"]),
    "Zaporizka oblast": ("Zaporizhzhia", ["zaporiz"]),
    "Ivano-Frankivska oblast": ("Ivano-Frankivsk", ["frankivsk"]),
    "Kyivska oblast": ("Bila Tserkva", ["bilotserk", "bila tserk"]),
    "Kirovohradska oblast": ("Kropyvnytskyi", ["kropyvnyts"]),
    "Lvivska oblast": ("Lviv", ["lviv"]),
    "Mykolaivska oblast": ("Mykolaiv", ["mykolai"]),
    "Odeska oblast": ("Odesa", ["odes"]),
    "Poltavska oblast": ("Poltava", ["poltav"]),
    "Rivnenska oblast": ("Rivne", ["rivn"]),
    "Sumska oblast": ("Sumy", ["sumy", "sumsk"]),
    "Ternopilska oblast": ("Ternopil", ["ternop"]),
    "Kharkivska oblast": ("Kharkiv", ["kharkiv"]),
    "Khersonska oblast": ("Kherson", ["kherson"]),
    "Khmelnytska oblast": ("Khmelnytskyi", ["khmelnyts"]),
    "Cherkaska oblast": ("Cherkasy", ["cherkas"]),
    "Chernivetska oblast": ("Chernivtsi", ["chernivts", "chernivet"]),
    "Chernihivska oblast": ("Chernihiv", ["chernihiv"]),
    "Kyiv City": ("Kyiv City", []),  # oblast-level unit in both sources, no search needed
}


def matches(value, stems):
    text = str(value).lower()
    return any(stem in text for stem in stems)


def hours(df):
    return (df["finished_at"] - df["started_at"]).dt.total_seconds() / 3600


def official_block(off, region, stems):
    o = off[off["oblast"] == region]
    obl = o[o["level"] == "oblast"]
    print(f"  official oblast-wide rows: {len(obl)}, last start {obl['started_at'].max()}")
    if not stems:
        return
    cand = o[o["raion"].map(lambda v: matches(v, stems)) | o["hromada"].map(lambda v: matches(v, stems))]
    if cand.empty:
        print("  official candidates: NONE FOUND")
        return
    g = cand.assign(h=hours(cand)).groupby(["level", "raion", "hromada"], dropna=False).agg(
        rows=("started_at", "size"), first=("started_at", "min"), last=("started_at", "max"),
        over_24h=("h", lambda x: int((x > 24).sum())))
    g["first"] = g["first"].dt.date
    g["last"] = g["last"].dt.date
    print("  official candidates (deduplicated rows):")
    print("    " + g.to_string().replace("\n", "\n    "))


def aic_block(aic, region, stems):
    a = aic[aic["name"] == region]
    print(f"  alerts.in.ua records for this unit: {len(a)}; location types: {a['location_type'].value_counts().to_dict()}")
    cand = a if not stems else a[a["location_title_en"].map(lambda v: matches(v, stems))]
    if cand.empty:
        print("  alerts.in.ua candidates: NONE FOUND")
        print(f"  alerts.in.ua raion titles here: {sorted(a.loc[a['location_type'] == 'raion', 'location_title_en'].unique())}")
        return
    g = cand.groupby(["location_type", "location_title_en"]).agg(
        records=("id", "size"),
        air_raid=("alert_type", lambda x: int((x == "air_raid").sum())),
        other_types=("alert_type", lambda x: int((x != "air_raid").sum())),
        red_share=("level", lambda x: round((x == "red").mean(), 3)),
        first=("started_at", "min"), last=("started_at", "max"))
    g["first"] = g["first"].dt.date
    g["last"] = g["last"].dt.date
    print("  alerts.in.ua candidates:")
    print("    " + g.to_string().replace("\n", "\n    "))


def capitals(off, aic):
    section("PART 1 - capital candidates per oblast (official file and alerts.in.ua)")
    for region, (capital, stems) in CAPITALS.items():
        print(f"\n### {region} -> {capital}  (stems: {stems})")
        official_block(off, region, stems)
        aic_block(aic, region, stems)


def old_alerts_overlap(off, aic, w0):
    section("PART 2 - alerts.in.ua alerts that started before the window: present in the official file?")
    old = aic[aic["started_at"] < w0]
    off_end = off["started_at"].max()
    print(f"alerts.in.ua alerts started before {w0}: {len(old)}; official file last start: {off_end}")
    rows = []
    for _, r in old.iterrows():
        o = off[off["oblast"] == r["name"]]
        diff = (o["started_at"] - r["started_at"]).abs().dt.total_seconds() / 60
        close = o[diff <= 5]
        rows.append({"alert_type": r["alert_type"], "location_type": r["location_type"],
                     "started_before_official_end": r["started_at"] <= off_end,
                     "official_match_5min": len(close) > 0,
                     "official_end_after_cutoff": bool(len(close)) and bool((close["finished_at"] >= off_end).any()),
                     "aic_still_running": pd.isna(r["finished_at"])})
    d = pd.DataFrame(rows)
    print("\ncounts (aggregated, no raw records):")
    print(d.groupby(["started_before_official_end", "alert_type", "official_match_5min"]).size().to_string())
    m = d[d["official_match_5min"]]
    print(f"\nmatched in official: {len(m)} of {int(d['started_before_official_end'].sum())} that started before the official file ended")
    print(f"  of matched: official record ends at/after the official cutoff: {int(m['official_end_after_cutoff'].sum())}; "
          f"alerts.in.ua still running: {int(m['aic_still_running'].sum())}")


def main():
    off = load(download(OFFICIAL)).drop_duplicates()
    aic, snap, folder = load_snapshot()
    w0 = snap - pd.DateOffset(months=1)
    section("SETUP")
    print(f"official rows (deduplicated): {len(off)}; alerts.in.ua snapshot {folder}: {len(aic)} records")
    capitals(off, aic)
    old_alerts_overlap(off, aic, w0)


if __name__ == "__main__":
    main()
