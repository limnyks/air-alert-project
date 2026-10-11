"""Step 10a: list every raion and match its names across sources (D28).

Matching is done on normalized Ukrainian names (not English transliterations).
Downloads (once, into data/raw/, local only):
  - official_data_uk.csv  (GitHub, same pinned commit as Step 5, ~41 MB)
  - alerts.in.ua location ID list (public CSV export of the sheet linked from their API docs)
  - HDX ukr_admin_boundaries.xlsx (~6.3 MB; needs the openpyxl package)
Writes config/raions.csv and prints every check and every unmatched name.

Checks:
  1. English and Ukrainian official files are row-aligned (same rows, same times, same levels).
  2. Every official raion gets exactly one alerts.in.ua ID; the snapshot's oblast for that ID agrees.
  3. Every official raion gets exactly one HDX raion code (if the HDX file is readable).
  4. Raions seen in the snapshot but not in the mapping; mapped raions with no snapshot records.

Run from the project root (after `pip install openpyxl`):
    python scripts/step10a_raion_mapping.py | tee logs/step10a_output.txt
"""
import csv
import io
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

from step05_inspect_data import OFFICIAL, download, load, section
from step07b_compare_reference import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_UK = "official_data_uk.csv"
UID_URL = "https://docs.google.com/spreadsheets/d/1XnTOzcPHd1LZUrarR1Fk43FUyl8Ae6a6M7pcwDRjNdA/export?format=csv"
UID_FILE = ROOT / "data" / "raw" / "alerts_in_ua" / "uid_list.csv"
HDX_URL = ("https://data.humdata.org/dataset/d23f529f-31e4-4021-a65b-13987e5cfb42/resource/"
           "c0f3df90-f167-43d1-93c4-b2e3738dbec3/download/ukr_admin_boundaries.xlsx")
HDX_FILE = ROOT / "data" / "raw" / "hdx" / "ukr_admin_boundaries.xlsx"
OUT = ROOT / "config" / "raions.csv"
ALIASES = ROOT / "config" / "raion_aliases.csv"  # renamed raions (old official name -> current name)
EXCLUDED_OBLASTS = {"Luhanska oblast"}  # D13 (Crimea is not in the official file)


def norm(name):
    """Lowercase, unify apostrophes, drop 'район'/'область'/'м.' and extra spaces."""
    s = str(name).lower().replace("’", "'").replace("ʼ", "'").replace("`", "'")
    s = re.sub(r"\b(район|область|м\.)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def has_cyrillic(series):
    return series.dropna().astype(str).str.contains(r"[а-яіїєґ]", case=False, regex=True).mean() > 0.5


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path.name} ...")
        urllib.request.urlretrieve(url, path)
    print(f"{path.relative_to(ROOT)}: {path.stat().st_size:,} bytes")
    return path


# ---------------------------------------------------------------- 1. official raions (uk + en)

def official_raions():
    section("CHECK 1 - official English vs Ukrainian files")
    en = load(download(OFFICIAL))
    uk = load(download(OFFICIAL_UK))
    print(f"rows: en {len(en)}, uk {len(uk)}")
    aligned = len(en) == len(uk) and (en["started_at"].values == uk["started_at"].values).all() \
        and (en["level"].values == uk["level"].values).all()
    print("row-aligned (same count, same start times, same levels):", aligned)
    if not aligned:
        raise SystemExit("STOP: files are not row-aligned - name pairs cannot be taken row by row")
    both = pd.DataFrame({"oblast_en": en["oblast"], "oblast_uk": uk["oblast"],
                         "raion_en": en["raion"], "raion_uk": uk["raion"], "level": en["level"],
                         "started_at": en["started_at"]}).dropna(subset=["raion_uk"])
    pairs = both.groupby(["oblast_en", "oblast_uk", "raion_en", "raion_uk"]).agg(
        official_rows=("level", "size"),
        raion_level_rows=("level", lambda x: int((x == "raion").sum())),
        first_raion_alert=("started_at", "min"), last_raion_alert=("started_at", "max")).reset_index()
    pairs = pairs[~pairs["oblast_en"].isin(EXCLUDED_OBLASTS)]
    dup_uk = pairs.groupby(["oblast_uk", "raion_uk"]).size()
    dup_en = pairs.groupby(["oblast_en", "raion_en"]).size()
    print(f"distinct raions (excluding {sorted(EXCLUDED_OBLASTS)}): {len(pairs)}; "
          f"uk names with >1 en name: {int((dup_uk > 1).sum())}; en names with >1 uk name: {int((dup_en > 1).sum())}")
    print("raions per oblast:", pairs.groupby("oblast_en").size().to_dict())
    return pairs


# ---------------------------------------------------------------- 2. alerts.in.ua IDs

def uid_raions():
    fetch(UID_URL, UID_FILE)
    rows = list(csv.reader(io.StringIO(UID_FILE.read_text(encoding="utf-8"))))
    hdr = next(i for i, r in enumerate(rows) if r[:3] == ["UID", "Назва", "Тип"])
    data = [r for r in rows[hdr + 1:] if len(r) > 2 and r[0].strip().isdigit()]
    print(f"ID list rows: {len(data)}; types: {dict(Counter(r[2] for r in data))}")
    return pd.DataFrame([{"aic_uid": int(r[0]), "aic_name_uk": r[1], "key": norm(r[1])}
                         for r in data if r[2] == "Район"])


def match_uids(pairs, uids, aic):
    section("CHECK 2 - official raions -> alerts.in.ua IDs (by normalized Ukrainian name)")
    alias = {norm(r["official_name_uk"]): norm(r["current_name_uk"]) for _, r in pd.read_csv(ALIASES).iterrows()}
    pairs = pairs.assign(key=pairs["raion_uk"].map(norm).map(lambda k: alias.get(k, k)))
    print(f"aliases applied (renamed raions): {sum(pairs['raion_uk'].map(norm).isin(alias))}")
    ambiguous = uids["key"].value_counts()
    ambiguous = set(ambiguous[ambiguous > 1].index)
    print("raion names that occur more than once in the ID list:", sorted(ambiguous) or "none")
    m = pairs.merge(uids, on="key", how="left")
    counts = m.groupby(["oblast_uk", "raion_uk"])["aic_uid"].nunique()
    print(f"official raions with exactly one ID: {int((counts == 1).sum())} of {len(counts)}; "
          f"with none: {int((counts == 0).sum())}; with several: {int((counts > 1).sum())}")
    print("official raions without an ID:", m.loc[m["aic_uid"].isna(), ["oblast_en", "raion_uk"]].values.tolist() or "none")

    # snapshot cross-check: the oblast of each ID in the snapshot must equal the official oblast
    snap = aic[aic["location_type"] == "raion"].groupby("location_uid").agg(
        snap_records=("id", "size"), snap_oblast_uk=("location_oblast", "first"),
        snap_title_en=("location_title_en", "first")).reset_index()
    snap["location_uid"] = pd.to_numeric(snap["location_uid"], errors="coerce").astype(float)
    m["aic_uid"] = pd.to_numeric(m["aic_uid"], errors="coerce").astype(float)
    m = m.merge(snap, left_on="aic_uid", right_on="location_uid", how="left")
    compared = m[m["snap_oblast_uk"].notna()]
    wrong = compared[compared["snap_oblast_uk"].map(norm) != compared["oblast_uk"].map(norm)]
    print(f"oblast cross-check (snapshot location_oblast vs official oblast): compared {len(compared)}, different: {len(wrong)}",
          wrong[["oblast_uk", "raion_uk", "aic_uid", "snap_oblast_uk"]].values.tolist() if len(wrong) else "")
    seen = set(snap["location_uid"].astype(int))
    mapped = set(m["aic_uid"].dropna().astype(int))
    extra = snap[~snap["location_uid"].astype(int).isin(mapped)]
    print(f"\nsnapshot raion IDs not in the mapping: {len(extra)}",
          extra[["location_uid", "snap_title_en", "snap_records"]].values.tolist() if len(extra) else "")
    quiet = m[m["aic_uid"].notna() & ~m["aic_uid"].astype("Int64").isin(seen)]
    print(f"mapped raions with no snapshot records (no alerts in the month, or not covered): {len(quiet)}")
    if len(quiet):
        print(quiet[["oblast_en", "raion_en", "aic_uid", "last_raion_alert"]].to_string(index=False))
    return m.drop(columns=["location_uid", "snap_oblast_uk"])


# ---------------------------------------------------------------- 3. HDX codes

def match_hdx(m):
    section("CHECK 3 - official raions -> HDX raion codes")
    try:
        fetch(HDX_URL, HDX_FILE)
        sheets = pd.read_excel(HDX_FILE, sheet_name=None)
    except Exception as error:  # missing openpyxl, network, format - report and continue without codes
        print(f"HDX not available ({type(error).__name__}: {error}) - continuing without HDX codes")
        return m.assign(hdx_pcode=pd.NA, hdx_name_en=pd.NA)
    for name, df in sheets.items():
        print(f"sheet '{name}': {len(df)} rows, columns {list(df.columns)}")
    for name, df in sheets.items():
        cols = {c.lower(): c for c in df.columns}
        pcode = next((c for k, c in cols.items() if "adm2" in k and "pcode" in k), None)
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
        skip = ("pcode", "ref", "date", "valid", "area", "len", "shape")
        adm2_names = [c for c in text_cols if "adm2" in c.lower() and not any(x in c.lower() for x in skip)]
        adm1_names = [c for c in text_cols if "adm1" in c.lower() and not any(x in c.lower() for x in skip)]
        if not (pcode and adm2_names and adm1_names):
            continue
        adm2_uk = next((c for c in adm2_names if has_cyrillic(df[c])), None)
        latin = [c for c in adm2_names if not has_cyrillic(df[c])]
        adm2_en = next((c for c in latin if "en" in c.lower()), latin[0] if latin else None)
        adm1_uk = next((c for c in adm1_names if has_cyrillic(df[c])), None)
        if not (adm2_uk and adm1_uk):
            continue
        print(f"\nusing sheet '{name}': pcode={pcode}, raion uk={adm2_uk}, raion en={adm2_en}, oblast uk={adm1_uk}")
        h = df[[pcode, adm2_uk, adm1_uk] + ([adm2_en] if adm2_en else [])].drop_duplicates(subset=[pcode])
        h = h.assign(key=h[adm2_uk].map(norm), okey=h[adm1_uk].map(norm)).rename(
            columns={pcode: "hdx_pcode", adm2_en: "hdx_name_en"} if adm2_en else {pcode: "hdx_pcode"})
        print(f"HDX raions: {len(h)}")
        out = m.assign(okey=m["oblast_uk"].map(norm)).merge(h[["hdx_pcode", "key", "okey"] + (["hdx_name_en"] if adm2_en else [])],
                                                              on=["key", "okey"], how="left")
        n = out.groupby(["oblast_uk", "raion_uk"])["hdx_pcode"].nunique()
        print(f"official raions with exactly one HDX code: {int((n == 1).sum())} of {len(n)}")
        print("official raions without an HDX code:", out.loc[out["hdx_pcode"].isna(), ["oblast_uk", "raion_uk"]].values.tolist() or "none")
        unused = h[~h["hdx_pcode"].isin(out["hdx_pcode"].dropna())]
        print(f"HDX raions not matched to any official raion (e.g. occupied, no data): {len(unused)}")
        print("   ", unused[[adm1_uk, adm2_uk]].values.tolist())
        return out.drop(columns=["okey"])
    print("no sheet with ADM1/ADM2 name and pcode columns found - continuing without HDX codes")
    return m.assign(hdx_pcode=pd.NA, hdx_name_en=pd.NA)


def main():
    pairs = official_raions()
    section("alerts.in.ua ID list")
    uids = uid_raions()
    aic, _, _ = load_snapshot()
    m = match_uids(pairs, uids, aic)
    m = match_hdx(m)

    kyiv = pd.DataFrame([{"oblast_en": "Kyiv City", "raion_en": "", "raion_uk": "", "aic_uid": 31,
                          "note": "city unit (oblast level in both sources)"}])
    cols = ["oblast_en", "oblast_uk", "raion_en", "raion_uk", "aic_uid", "aic_name_uk", "snap_title_en",
            "hdx_pcode", "hdx_name_en", "official_rows", "raion_level_rows", "first_raion_alert",
            "last_raion_alert", "snap_records"]
    out = pd.concat([m.reindex(columns=cols), kyiv], ignore_index=True)
    out["aic_uid"] = out["aic_uid"].astype("Int64")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    section("RESULT")
    print(f"saved {len(out)} rows -> {OUT.relative_to(ROOT)}")
    print(f"rows with alerts.in.ua ID: {int(out['aic_uid'].notna().sum())}; with HDX code: {int(out['hdx_pcode'].notna().sum())}")


if __name__ == "__main__":
    main()
