"""Step 5: download the two alert datasets at a pinned commit and print facts about them.

Prints only: no cleaning, no modeling, nothing is changed in the data.

Parts:
  1. Download (once) and fingerprint the files.
  2. Basic facts per file: rows, columns, nulls, duplicates, date range, durations.
  3. Official file: alert levels over time, oblast list.
  4. Volunteer file: naive (invented end) share over time, region list.
  5. Region names in both files; Luhansk / Crimea check.
  6. Before vs after the alert reform (2026-09-06).
  7. Agreement between official and volunteer on their overlap period.

Run from the project root:
    python scripts/step05_inspect_data.py
"""
import hashlib
import platform
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

COMMIT = "7ccd4367b7057a4541ce109ca09cf1f0220f6a75"  # Vadimkin dataset, 2026-10-10 05:01 UTC
BASE_URL = f"https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/{COMMIT}/datasets"
OFFICIAL = "official_data_en.csv"
VOLUNTEER = "volunteer_data_en.csv"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / COMMIT[:7]

REFORM_DAY = pd.Timestamp("2026-09-06", tz="UTC")  # yellow/red levels (res. 1092); Kyiv start date
WINDOW_DAYS = 28  # length of the before/after comparison windows
MATCH_WINDOWS_MIN = [2, 5, 10, 30]  # "same alert" if starts differ by at most this many minutes
MAIN_WINDOW_MIN = 5

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")


def download(name):
    path = RAW_DIR / name
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{name}: {path.stat().st_size:,} bytes, sha256 {digest}")
    return path


def load(path):
    df = pd.read_csv(path)
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def duration_min(df):
    return (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60


def to_ns(series):
    """UTC timestamps -> int64 nanoseconds (independent of pandas' default time resolution)."""
    return series.dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


def quarter(series):
    return series.dt.tz_localize(None).dt.to_period("Q")


def month(series):
    return series.dt.tz_localize(None).dt.to_period("M")


def describe_durations(minutes, label):
    print(f"\ndurations, {label} (minutes), n = {len(minutes)}")
    print(f"  <= 0 min: {(minutes <= 0).sum()}   (0, 1) min: {((minutes > 0) & (minutes < 1)).sum()}   "
          f"> 24 h: {(minutes > 24 * 60).sum()}")
    q = minutes.quantile([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]).round(1)
    print("  quantiles:", dict(zip(q.index, q.values)))


# ---------------------------------------------------------------- part 2

def basic_facts(df, name):
    section(f"PART 2 - {name}: basic facts")
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print("null values per column:")
    print(df.isna().sum().to_string())
    print("exact duplicate rows:", df.duplicated().sum())
    print("sorted by started_at:", df["started_at"].is_monotonic_increasing)
    print("started_at  min / max:", df["started_at"].min(), "/", df["started_at"].max())
    print("finished_at min / max:", df["finished_at"].min(), "/", df["finished_at"].max())
    print("last 3 rows by started_at:")
    print(df.sort_values("started_at").tail(3).to_string(index=False))


# ---------------------------------------------------------------- part 3

def official_facts(off):
    section("PART 3 - official: alert levels and oblasts")
    print("level counts:")
    print(off["level"].value_counts(dropna=False).to_string())
    print("\nrows per quarter by level (by started_at, UTC):")
    print(pd.crosstab(quarter(off["started_at"]), off["level"]).to_string())
    describe_durations(duration_min(off), "official, all rows")
    print("\nrows per oblast:")
    print(off["oblast"].value_counts(dropna=False).to_string())


# ---------------------------------------------------------------- part 4

def volunteer_facts(vol):
    section("PART 4 - volunteer: naive (invented end) rows and regions")
    print("naive value counts:")
    print(vol["naive"].value_counts(dropna=False).to_string())
    naive = vol["naive"] == True  # noqa: E712  (column may be bool or object)
    d = duration_min(vol)
    print(f"naive rows with duration exactly 30 min: {(d[naive].round(6) == 30).mean():.4f} (share)")
    describe_durations(d[~naive], "volunteer, non-naive rows only")
    print("\nper month (last 15 months): rows, naive share")
    per_month = vol.groupby(month(vol["started_at"]))["naive"].agg(rows="size", naive_share=lambda s: (s == True).mean())  # noqa: E712
    print(per_month.tail(15).round(3).to_string())
    print("\nrows per region:")
    print(vol["region"].value_counts(dropna=False).to_string())


# ---------------------------------------------------------------- part 5

def region_names(off, vol):
    section("PART 5 - region names in both files")
    off_names = set(off["oblast"].dropna())
    vol_names = set(vol["region"].dropna())
    print(f"official oblasts: {len(off_names)}, volunteer regions: {len(vol_names)}, common: {len(off_names & vol_names)}")
    print("only in official :", sorted(off_names - vol_names))
    print("only in volunteer:", sorted(vol_names - off_names))
    keys = ("luhan", "crimea", "sevastop")
    hits = sorted(n for n in off_names | vol_names if any(k in n.lower() for k in keys))
    print("names containing Luhansk / Crimea / Sevastopol:", hits)


# ---------------------------------------------------------------- part 6

def reform_window(df, name, naive=None):
    start_before = REFORM_DAY - pd.Timedelta(days=WINDOW_DAYS)
    end_after = REFORM_DAY + pd.Timedelta(days=WINDOW_DAYS)
    before = (df["started_at"] >= start_before) & (df["started_at"] < REFORM_DAY)
    after = (df["started_at"] >= REFORM_DAY) & (df["started_at"] < end_after)
    d = duration_min(df)
    real = ~naive if naive is not None else pd.Series(True, index=df.index)
    print(f"\n{name}: before = [{start_before.date()}, {REFORM_DAY.date()}), after = [{REFORM_DAY.date()}, {end_after.date()})")
    for label, mask in (("before", before), ("after", after)):
        n = int(mask.sum())
        line = f"  {label:6s}: rows {n:6d}, rows/day {n / WINDOW_DAYS:7.1f}, median duration (real ends) {d[mask & real].median():6.1f} min"
        if naive is not None:
            line += f", naive share {naive[mask].mean() if n else float('nan'):.3f}"
        print(line)


def reform_facts(off, vol):
    section("PART 6 - before vs after the alert reform (2026-09-06)")
    reform_window(off, "official")
    reform_window(vol, "volunteer", naive=(vol["naive"] == True))  # noqa: E712
    print("\nvolunteer rows per week (weeks end on Sunday; the last week may be partial):")
    recent = vol[vol["started_at"] >= pd.Timestamp("2026-07-27", tz="UTC")]
    weekly = recent.groupby(pd.Grouper(key="started_at", freq="W-SUN")).agg(
        rows=("region", "size"), naive_share=("naive", lambda s: (s == True).mean()))  # noqa: E712
    print(weekly.round(3).to_string())


# ---------------------------------------------------------------- part 7

def merge_intervals(starts, ends):
    """Union of overlapping intervals (int64 ns). Returns merged starts and ends, sorted."""
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], ends[order]
    running_end = np.maximum.accumulate(e)
    new_block = np.ones(len(s), dtype=bool)
    new_block[1:] = s[1:] > running_end[:-1]
    first = np.flatnonzero(new_block)
    return s[first], np.maximum.reduceat(e, first)


def nearest(a, b):
    """For each value in a: index of the nearest value in sorted, non-empty b, and the distance in minutes."""
    i = np.searchsorted(b, a)
    lo = np.clip(i - 1, 0, len(b) - 1)
    hi = np.clip(i, 0, len(b) - 1)
    d_lo, d_hi = np.abs(a - b[lo]), np.abs(b[hi] - a)
    j = np.where(d_lo <= d_hi, lo, hi)
    return j, np.minimum(d_lo, d_hi) / 60e9


def agreement(off, vol):
    section("PART 7 - agreement between official and volunteer (overlap period)")
    t0, t1 = off["started_at"].min(), off["started_at"].max()
    print(f"overlap period (official start range): {t0} -> {t1}")
    print("official is rolled up to oblast level: union of all its oblast/raion/hromada alerts per oblast")
    print("(= 'some part of the oblast is under alert'). Volunteer rows are used as they are.")

    o = off[(off["started_at"] >= t0) & (off["started_at"] <= t1)]
    v = vol[(vol["started_at"] >= t0) & (vol["started_at"] <= t1)]
    common = sorted(set(o["oblast"].dropna()) & set(v["region"].dropna()))

    per_region, vol_rows, off_rows = [], [], []
    for region in common:
        orr = o[o["oblast"] == region]
        os_, oe_ = merge_intervals(to_ns(orr["started_at"]), to_ns(orr["finished_at"]))
        vr = v[v["region"] == region].sort_values("started_at")
        vs, ve = to_ns(vr["started_at"]), to_ns(vr["finished_at"])
        vnaive = (vr["naive"] == True).to_numpy()  # noqa: E712
        if len(os_) == 0 or len(vs) == 0:
            continue
        jv, dv = nearest(vs, os_)  # each volunteer alert -> nearest official episode
        _, do = nearest(os_, vs)   # each official episode -> nearest volunteer alert
        matched = dv <= MAIN_WINDOW_MIN
        real_end = matched & ~vnaive
        end_diff = (ve[real_end] - oe_[jv[real_end]]) / 60e9  # volunteer minus official, minutes
        per_region.append({
            "region": region,
            "vol_alerts": len(vs),
            "off_episodes": len(os_),
            f"vol_found_in_off@{MAIN_WINDOW_MIN}m": matched.mean(),
            f"off_found_in_vol@{MAIN_WINDOW_MIN}m": (do <= MAIN_WINDOW_MIN).mean(),
            "median_|start_diff|_min": np.median(dv[matched]) if matched.any() else np.nan,
            "median_end_diff_min(v-o)": np.median(end_diff) if len(end_diff) else np.nan,
            "naive_repairable_share": matched[vnaive].mean() if vnaive.any() else np.nan,
        })
        vol_rows.append(pd.DataFrame({"q": quarter(vr["started_at"]).to_numpy(), "dist": dv, "naive": vnaive}))
        off_q = pd.Series(pd.to_datetime(os_, utc=True)).dt.tz_localize(None).dt.to_period("Q")
        off_rows.append(pd.DataFrame({"q": off_q.to_numpy(), "dist": do}))

    vol_all = pd.concat(vol_rows, ignore_index=True)
    off_all = pd.concat(off_rows, ignore_index=True)

    print(f"\ncommon regions compared: {len(per_region)}")
    print(f"total volunteer alerts: {len(vol_all)}, total official oblast-episodes: {len(off_all)}")
    print("\nall regions pooled - share found in the other source, by matching window:")
    for w in MATCH_WINDOWS_MIN:
        print(f"  +-{w:2d} min: volunteer found in official {(vol_all['dist'] <= w).mean():.3f}   "
              f"official found in volunteer {(off_all['dist'] <= w).mean():.3f}")
    print(f"\nnaive volunteer alerts with an official match within {MAIN_WINDOW_MIN} min "
          f"(their end could be repaired): {(vol_all.loc[vol_all['naive'], 'dist'] <= MAIN_WINDOW_MIN).mean():.3f}")

    print(f"\nper quarter, window +-{MAIN_WINDOW_MIN} min:")
    by_q = pd.DataFrame({
        "vol_alerts": vol_all.groupby("q").size(),
        "vol_found_in_off": vol_all.groupby("q")["dist"].apply(lambda d: (d <= MAIN_WINDOW_MIN).mean()),
        "off_episodes": off_all.groupby("q").size(),
        "off_found_in_vol": off_all.groupby("q")["dist"].apply(lambda d: (d <= MAIN_WINDOW_MIN).mean()),
    })
    print(by_q.round(3).to_string())

    print(f"\nper region, window +-{MAIN_WINDOW_MIN} min:")
    print(pd.DataFrame(per_region).round(3).to_string(index=False))


def main():
    section("PART 1 - environment and files")
    print("python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__)
    print("dataset commit:", COMMIT)
    off = load(download(OFFICIAL))
    vol = load(download(VOLUNTEER))

    basic_facts(off, "official")
    basic_facts(vol, "volunteer")
    official_facts(off)
    volunteer_facts(vol)
    region_names(off, vol)
    reform_facts(off, vol)
    agreement(off, vol)


if __name__ == "__main__":
    main()
