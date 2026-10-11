"""Step 6: diagnose why the official and volunteer files disagree.

Prints only: no cleaning, no modeling. Uses the files downloaded in Step 5 (same pinned commit).

Parts:
  1. Official duplicate rows: where are they?
  2. Volunteer self-overlap: does a region get a new alert while its previous one is still running?
  3. Where do volunteer alert starts fall relative to official alerts (near a start / inside / outside)?
  4. Time-based agreement: minutes under alert in each source and their overlap (Jaccard).
  5. Very long official alerts (> 24 h).

Two official definitions of "oblast under alert" are compared:
  U = union of all official rows of the oblast (oblast, raion and hromada level)
  O = union of the oblast-level rows only

Run from the project root:
    python scripts/step06_diagnose_sources.py
"""
import numpy as np
import pandas as pd

from step05_inspect_data import OFFICIAL, VOLUNTEER, download, load, merge_intervals, quarter, section, to_ns

NEAR_MIN = 5          # a volunteer start within +-5 min of an official start counts as "near_start"
LONG_HOURS = 24       # "very long" official alert
MIN_NS = 60 * 10**9   # one minute in nanoseconds


# ---------------------------------------------------------------- part 1

def official_duplicates(off):
    section("PART 1 - official duplicate rows")
    dup = off.duplicated(keep="first")
    print(f"duplicate rows (copies beyond the first): {dup.sum()} of {len(off)}")
    print("\nby level:")
    print(pd.DataFrame({"rows": off.groupby("level").size(),
                        "duplicates": off[dup].groupby("level").size()}).fillna(0).astype(int).to_string())
    print("\nby quarter:")
    print(pd.DataFrame({"rows": off.groupby(quarter(off["started_at"])).size(),
                        "duplicates": off[dup].groupby(quarter(off.loc[dup, "started_at"])).size()})
          .fillna(0).astype(int).to_string())
    copies = off.groupby(list(off.columns), dropna=False).size()
    print("\nhow many times a duplicated row appears {copies: number of distinct rows}:",
          copies[copies > 1].value_counts().sort_index().to_dict())


# ---------------------------------------------------------------- part 2

def volunteer_self_overlap(vol):
    section("PART 2 - volunteer: new alert while an earlier alert in the same region is still running")
    parts = []
    for region, g in vol.groupby("region"):
        g = g.sort_values("started_at")
        s, e = to_ns(g["started_at"]), to_ns(g["finished_at"])
        naive = (g["naive"] == True).to_numpy()  # noqa: E712
        overlap = np.zeros(len(s), dtype=bool)
        overlap[1:] = s[1:] < np.maximum.accumulate(e)[:-1]
        e_real = np.where(naive, np.iinfo(np.int64).min, e)  # ignore invented ends of naive rows
        overlap_real = np.zeros(len(s), dtype=bool)
        overlap_real[1:] = s[1:] < np.maximum.accumulate(e_real)[:-1]
        parts.append(pd.DataFrame({"region": region, "q": quarter(g["started_at"]).to_numpy(),
                                   "overlap": overlap, "overlap_real_ends": overlap_real}))
    d = pd.concat(parts, ignore_index=True)
    print("share of volunteer alerts that start while an earlier alert in the same region is still running")
    print("(overlap_real_ends ignores the invented 30-min ends of naive rows)")
    print("\nby quarter:")
    print(d.groupby("q")[["overlap", "overlap_real_ends"]].mean().round(3).to_string())
    print("\nby region:")
    print(d.groupby("region")[["overlap", "overlap_real_ends"]].mean().round(3).sort_values("overlap").to_string())


# ---------------------------------------------------------------- part 3

def official_episodes(off_region):
    """Merged official intervals (int64 ns) for one oblast, under both definitions."""
    all_rows = merge_intervals(to_ns(off_region["started_at"]), to_ns(off_region["finished_at"]))
    obl = off_region[off_region["level"] == "oblast"]
    oblast_rows = merge_intervals(to_ns(obl["started_at"]), to_ns(obl["finished_at"]))
    return {"U": all_rows, "O": oblast_rows}


def classify(v, s, e):
    """For each volunteer start v: 'near_start', 'inside' an official alert, or 'outside'.
    Also returns the index of the official episode containing v (or -1)."""
    label = np.full(len(v), "outside", dtype=object)
    k_inside = np.full(len(v), -1)
    if len(s) == 0:
        return label, k_inside
    i = np.searchsorted(s, v)
    lo, hi = np.clip(i - 1, 0, len(s) - 1), np.clip(i, 0, len(s) - 1)
    near = np.minimum(np.abs(v - s[lo]), np.abs(s[hi] - v)) <= NEAR_MIN * MIN_NS
    k = np.searchsorted(s, v, side="right") - 1
    inside = (k >= 0) & (v < e[np.clip(k, 0, len(e) - 1)])
    label[inside] = "inside"
    label[near] = "near_start"
    k_inside[inside] = k[inside]
    return label, k_inside


def start_positions(v, episodes):
    section("PART 3 - where volunteer alert starts fall relative to official alerts")
    print(f"near_start = within +-{NEAR_MIN} min of an official start; inside = during an official alert;")
    print("outside = official says no alert. inside_long = inside an official episode longer than "
          f"{LONG_HOURS} h (definition U).")
    parts = []
    for region, g in v.groupby("region"):
        if region not in episodes:
            continue
        g = g.sort_values("started_at")
        vs = to_ns(g["started_at"])
        row = {"region": region, "q": quarter(g["started_at"]).to_numpy()}
        for name, (s, e) in episodes[region].items():
            label, k = classify(vs, s, e)
            row[name] = label
            if name == "U":
                if len(s):
                    length_h = (e - s) / (3600 * 10**9)
                    row["U_inside_long"] = (k >= 0) & (length_h[np.clip(k, 0, len(s) - 1)] > LONG_HOURS)
                else:
                    row["U_inside_long"] = np.zeros(len(vs), dtype=bool)
        parts.append(pd.DataFrame(row))
    d = pd.concat(parts, ignore_index=True)
    for name in ("U", "O"):
        print(f"\ndefinition {name}, by quarter (shares of volunteer starts):")
        t = pd.crosstab(d["q"], d[name], normalize="index")
        t["n"] = d.groupby("q").size()
        if name == "U":
            t["inside_long"] = d.groupby("q")["U_inside_long"].mean()
        print(t.round(3).to_string())
    print("\ndefinition U, by region (shares of volunteer starts, whole overlap period):")
    t = pd.crosstab(d["region"], d["U"], normalize="index")
    t["inside_long"] = d.groupby("region")["U_inside_long"].mean()
    t["n"] = d.groupby("region").size()
    print(t.round(3).sort_values("near_start").to_string())


# ---------------------------------------------------------------- part 4

def coverage(starts, ends, t0_ns, n):
    """Boolean array of n minutes from t0: True if any interval covers part of that minute."""
    diff = np.zeros(n + 1, dtype=np.int32)
    a = np.clip((starts - t0_ns) // MIN_NS, 0, n)
    b = np.clip(-((t0_ns - ends) // MIN_NS), 0, n)  # ceiling division
    keep = b > a
    np.add.at(diff, a[keep], 1)
    np.add.at(diff, b[keep], -1)
    return np.cumsum(diff[:-1]) > 0


def time_agreement(vol, episodes, t0, t1):
    section("PART 4 - time-based agreement (minute grid over the overlap period)")
    print("Volunteer intervals are used as they are (naive rows keep their invented 30-min ends, ~5% of rows).")
    t0_naive, t1_naive = t0.tz_localize(None), t1.tz_localize(None)  # UTC wall time, same as to_ns()
    t0_ns = t0_naive.as_unit("ns").value
    n = int(np.ceil((t1_naive.as_unit("ns").value - t0_ns) / MIN_NS))
    bounds = pd.date_range(t0_naive.to_period("Q").start_time, t1_naive, freq="QS")
    b_idx = (bounds.as_unit("ns").asi8 - t0_ns) // MIN_NS
    qid = np.searchsorted(b_idx, np.arange(n), side="right") - 1
    labels = bounds.to_period("Q")
    nq = len(labels)

    keys = ["vol", "U", "O", "inter_VU", "union_VU", "inter_VO", "union_VO"]
    per_q = {k: np.zeros(nq) for k in keys}
    per_region = []
    for region in sorted(episodes):
        g = vol[vol["region"] == region]
        vc = coverage(to_ns(g["started_at"]), to_ns(g["finished_at"]), t0_ns, n)
        uc = coverage(*episodes[region]["U"], t0_ns, n)
        oc = coverage(*episodes[region]["O"], t0_ns, n)
        arrays = {"vol": vc, "U": uc, "O": oc, "inter_VU": vc & uc, "union_VU": vc | uc,
                  "inter_VO": vc & oc, "union_VO": vc | oc}
        for k, arr in arrays.items():
            per_q[k] += np.bincount(qid, weights=arr.astype(np.float64), minlength=nq)
        tot = {k: arr.sum() for k, arr in arrays.items()}
        per_region.append({"region": region,
                           "share_time_vol": tot["vol"] / n, "share_time_U": tot["U"] / n, "share_time_O": tot["O"] / n,
                           "jaccard_VU": tot["inter_VU"] / tot["union_VU"] if tot["union_VU"] else np.nan,
                           "jaccard_VO": tot["inter_VO"] / tot["union_VO"] if tot["union_VO"] else np.nan,
                           "P(U|vol)": tot["inter_VU"] / tot["vol"] if tot["vol"] else np.nan,
                           "P(vol|U)": tot["inter_VU"] / tot["U"] if tot["U"] else np.nan})

    minutes_in_q = np.bincount(qid, minlength=nq) * len(episodes)
    q = pd.DataFrame({
        "share_time_vol": per_q["vol"] / minutes_in_q,
        "share_time_U": per_q["U"] / minutes_in_q,
        "share_time_O": per_q["O"] / minutes_in_q,
        "jaccard_VU": per_q["inter_VU"] / per_q["union_VU"],
        "jaccard_VO": per_q["inter_VO"] / per_q["union_VO"],
        "P(U|vol)": per_q["inter_VU"] / per_q["vol"],
        "P(vol|U)": per_q["inter_VU"] / per_q["U"],
    }, index=labels)
    print("\nshare_time_* = share of region-minutes under alert; jaccard = overlap / union of alert time;")
    print("P(U|vol) = share of volunteer alert time that official U also marks as alert; P(vol|U) = the reverse.")
    print("\nby quarter (all regions pooled):")
    print(q.round(3).to_string())
    print("\nby region (whole overlap period):")
    print(pd.DataFrame(per_region).round(3).sort_values("jaccard_VU").to_string(index=False))


# ---------------------------------------------------------------- part 5

def long_alerts(off):
    section(f"PART 5 - official alerts longer than {LONG_HOURS} h (duplicates removed)")
    hours = (off["finished_at"] - off["started_at"]).dt.total_seconds() / 3600
    long = off[hours > LONG_HOURS].assign(hours=hours[hours > LONG_HOURS])
    print("rows:", len(long))
    print("\nby oblast and level:")
    print(pd.crosstab(long["oblast"], long["level"], margins=True).to_string())
    print("\nby year of start:")
    print(long.groupby(long["started_at"].dt.year).size().to_string())
    print("\n10 longest:")
    cols = ["oblast", "raion", "hromada", "level", "started_at", "hours"]
    print(long.sort_values("hours", ascending=False)[cols].head(10).round(1).to_string(index=False))


def main():
    off = load(download(OFFICIAL))
    vol = load(download(VOLUNTEER))

    official_duplicates(off)
    volunteer_self_overlap(vol)

    off_u = off.drop_duplicates()
    t0, t1 = off_u["started_at"].min(), off_u["started_at"].max()
    print(f"\noverlap period used in parts 3-4: {t0} -> {t1}")
    episodes = {region: official_episodes(g) for region, g in off_u.groupby("oblast")}
    v = vol[(vol["started_at"] >= t0) & (vol["started_at"] <= t1)]

    start_positions(v, episodes)
    time_agreement(vol, episodes, t0, t1)
    long_alerts(off_u)


if __name__ == "__main__":
    main()
