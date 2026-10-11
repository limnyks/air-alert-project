"""Step 7b: compare the volunteer file with the alerts.in.ua snapshot in the test period.

Prints only. Local data only (no network): the Step 5 volunteer file and the latest
alerts.in.ua snapshot from Step 7a. Raw alerts.in.ua records are not printed (terms 9.2),
except the few very old alerts in part 5, which are needed to verify one finding.

Parts:
  1. alert_level values (yellow/red?): counts, by oblast, by week, durations by level.
  2. Kyiv City 1:1 (both sources are oblast-level there): start matching and minute agreement.
  3. All oblasts: alerts.in.ua air_raid alerts rolled up per oblast ("any part" union)
     vs volunteer: time agreement, start positions, last alert per source.
  4. Volunteer naive (invented-end) alerts: how long did they really last per alerts.in.ua?
  5. alerts.in.ua alerts that started before the window (very old starts).

Run from the project root:
    python scripts/step07b_compare_reference.py | tee logs/step07b_output.txt
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step05_inspect_data import VOLUNTEER, download, load, merge_intervals, nearest, section, to_ns
from step06_diagnose_sources import MIN_NS, NEAR_MIN, classify, coverage

ROOT = Path(__file__).resolve().parents[1]
SNAP_ROOT = ROOT / "data" / "raw" / "alerts_in_ua"
MATCH_WINDOWS_MIN = [2, 5, 10]


def load_snapshot():
    folder = sorted(p for p in SNAP_ROOT.iterdir() if p.is_dir())[-1]
    manifest = json.loads((folder / "manifest.json").read_text())
    frames = []
    for entry in manifest:
        if entry["http_status"] != 200:
            continue
        alerts = json.loads((folder / entry["file"]).read_text())["alerts"]
        frames.append(pd.DataFrame(alerts).assign(name=entry["name"], unit_uid=entry["uid"]))
    aic = pd.concat(frames, ignore_index=True)
    for col in ("started_at", "finished_at"):
        aic[col] = pd.to_datetime(aic[col], utc=True, format="ISO8601")
    aic["level"] = aic["alert_level"].fillna("None").astype(str)
    snap = pd.to_datetime(folder.name, format="%Y%m%dT%H%M%SZ", utc=True)
    return aic, snap, folder.name


def minutes(start, end):
    return (end - start).dt.total_seconds() / 60


def pct(x, p):
    return np.percentile(x, p) if len(x) else np.nan


def week(series):
    return series.dt.tz_localize(None).dt.to_period("W-SUN")


# ---------------------------------------------------------------- part 1

def alert_levels(aic):
    section("PART 1 - alert_level values (all alerts.in.ua records)")
    print(aic["level"].value_counts().to_string())
    print("\nalert_type x alert_level:")
    print(pd.crosstab(aic["alert_type"], aic["level"]).to_string())

    air = aic[aic["alert_type"] == "air_raid"].copy()
    print("\nair_raid only - share of each level by oblast:")
    t = pd.crosstab(air["name"], air["level"], normalize="index").round(3)
    t["n"] = air.groupby("name").size()
    print(t.to_string())
    print("\nair_raid only - share of each level by week (week ending Sunday):")
    t = pd.crosstab(week(air["started_at"]), air["level"], normalize="index").round(3)
    t["n"] = air.groupby(week(air["started_at"])).size()
    print(t.to_string())
    done = air[air["finished_at"].notna()].copy()
    done["minutes"] = minutes(done["started_at"], done["finished_at"])
    print("\nair_raid, finished only - duration (minutes) by level:")
    print(done.groupby("level")["minutes"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string())


# ---------------------------------------------------------------- shared comparison

def episodes_of(df, snap):
    """Merged intervals (int64 ns); alerts still running at the snapshot end at the snapshot time."""
    ends = df["finished_at"].fillna(snap)
    return merge_intervals(to_ns(df["started_at"]), to_ns(ends))


def compare(vol_r, aic_r, snap, w0_ns, n_min):
    """Volunteer rows vs alerts.in.ua rows for one region. Returns a dict of metrics."""
    vol_r = vol_r.sort_values("started_at")
    vs, ve = to_ns(vol_r["started_at"]), to_ns(vol_r["finished_at"])
    vnaive = (vol_r["naive"] == True).to_numpy()  # noqa: E712
    s, e = episodes_of(aic_r, snap)
    out = {"vol_alerts": len(vs), "aic_episodes": len(s)}
    if len(vs) and len(s):
        _, dv = nearest(vs, s)
        _, da = nearest(s, vs)
        for w in MATCH_WINDOWS_MIN:
            out[f"vol_in_aic@{w}m"] = (dv <= w).mean()
            out[f"aic_in_vol@{w}m"] = (da <= w).mean()
        label, _ = classify(vs, s, e)
        for cls in ("near_start", "inside", "outside"):
            out[f"start_{cls}"] = (label == cls).mean()
    vc = coverage(vs, ve, w0_ns, n_min)
    ac = coverage(s, e, w0_ns, n_min)
    inter, union = (vc & ac).sum(), (vc | ac).sum()
    out.update({"time_vol": vc.mean(), "time_aic": ac.mean(),
                "jaccard": inter / union if union else np.nan,
                "P(aic|vol)": inter / vc.sum() if vc.sum() else np.nan,
                "P(vol|aic)": inter / ac.sum() if ac.sum() else np.nan})
    return out


# ---------------------------------------------------------------- part 2

def kyiv_city(vol_w, air, snap, w0_ns, n_min):
    section("PART 2 - Kyiv City 1:1 (both sources oblast-level)")
    v = vol_w[vol_w["region"] == "Kyiv City"].sort_values("started_at")
    a = air[air["name"] == "Kyiv City"]
    print("alerts.in.ua location_type for Kyiv City:", a["location_type"].value_counts().to_dict())
    m = compare(v, a, snap, w0_ns, n_min)
    for k, val in m.items():
        print(f"  {k:22s} {val:.3f}" if isinstance(val, float) else f"  {k:22s} {val}")

    s, e = episodes_of(a, snap)
    vs, ve = to_ns(v["started_at"]), to_ns(v["finished_at"])
    j, dv = nearest(vs, s)
    matched = dv <= NEAR_MIN
    real = matched & ~(v["naive"] == True).to_numpy()  # noqa: E712
    start_diff = (vs[matched] - s[j[matched]]) / MIN_NS
    end_diff = (ve[real] - e[j[real]]) / MIN_NS
    print(f"\n  matched within {NEAR_MIN} min: {matched.sum()} of {len(vs)} volunteer alerts")
    print(f"  start difference volunteer - aic (min): median {pct(start_diff, 50):.2f}, "
          f"10%/90% {pct(start_diff, 10):.2f} / {pct(start_diff, 90):.2f}")
    print(f"  end difference volunteer - aic, real ends (min): median {pct(end_diff, 50):.2f}, "
          f"10%/90% {pct(end_diff, 10):.2f} / {pct(end_diff, 90):.2f}")
    unmatched = v[~matched]
    print(f"\n  volunteer alerts without an aic start within {NEAR_MIN} min: {len(unmatched)} "
          f"(naive among them: {(unmatched['naive'] == True).sum()})")  # noqa: E712
    print(unmatched[["started_at", "finished_at", "naive"]].head(10).to_string(index=False))


# ---------------------------------------------------------------- part 3

def all_oblasts(vol_w, air, snap, w0_ns, n_min):
    section("PART 3 - all oblasts: volunteer vs alerts.in.ua air_raid rolled up per oblast ('any part')")
    rows = []
    for name, a in air.groupby("name"):
        v = vol_w[vol_w["region"] == name]
        m = compare(v, a, snap, w0_ns, n_min)
        m.update({"region": name,
                  "last_vol_start": v["started_at"].max(), "last_aic_start": a["started_at"].max()})
        rows.append(m)
    d = pd.DataFrame(rows).set_index("region").sort_values("jaccard")
    cols1 = ["vol_alerts", "aic_episodes", "time_vol", "time_aic", "jaccard", "P(aic|vol)", "P(vol|aic)"]
    cols2 = [f"vol_in_aic@{NEAR_MIN}m", f"aic_in_vol@{NEAR_MIN}m", "start_near_start", "start_inside", "start_outside"]
    print("time-based agreement (shares of window minutes):")
    print(d[cols1].round(3).to_string())
    print(f"\nstart matching (+-{NEAR_MIN} min) and where volunteer starts fall:")
    print(d[cols2].round(3).to_string())
    print("\nlast alert start per source:")
    print(d[["last_vol_start", "last_aic_start"]].to_string())
    tot = d[["time_vol", "time_aic"]].mean()
    print(f"\nmean over regions: time_vol {tot['time_vol']:.3f}, time_aic {tot['time_aic']:.3f}, "
          f"jaccard {d['jaccard'].mean():.3f}, P(aic|vol) {d['P(aic|vol)'].mean():.3f}, "
          f"P(vol|aic) {d['P(vol|aic)'].mean():.3f}")


# ---------------------------------------------------------------- part 4

def naive_alerts(vol_w, air, snap):
    section("PART 4 - volunteer naive (invented-end) alerts vs alerts.in.ua")
    print("ref_minutes = time from the volunteer start to the end of the alerts.in.ua episode")
    print(f"that contains it or starts within +-{NEAR_MIN} min of it (same measure for naive and normal alerts).")
    parts = []
    for name, a in air.groupby("name"):
        v = vol_w[vol_w["region"] == name].sort_values("started_at")
        if v.empty:
            continue
        s, e = episodes_of(a, snap)
        if len(s) == 0:
            continue
        vs = to_ns(v["started_at"])
        label, k_inside = classify(vs, s, e)
        j_near, _ = nearest(vs, s)
        idx = np.where(label == "inside", k_inside, np.where(label == "near_start", j_near, -1))
        ref = np.where(idx >= 0, (e[np.clip(idx, 0, len(e) - 1)] - vs) / MIN_NS, np.nan)
        parts.append(pd.DataFrame({"region": name, "naive": (v["naive"] == True).to_numpy(),  # noqa: E712
                                   "found": idx >= 0, "ref_minutes": ref,
                                   "vol_minutes": minutes(v["started_at"], v["finished_at"]).to_numpy()}))
    d = pd.concat(parts, ignore_index=True)
    print(f"\nvolunteer alerts in window: {len(d)}, naive: {d['naive'].sum()} ({d['naive'].mean():.3f})")
    print("share with an alerts.in.ua episode found:", d.groupby("naive")["found"].mean().round(3).to_dict())
    print("\nref_minutes by naive flag (found only):")
    print(d[d["found"]].groupby("naive")["ref_minutes"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string())
    normal = d[d["found"] & ~d["naive"]]
    print(f"\nnormal alerts: volunteer duration vs ref_minutes, median {normal['vol_minutes'].median():.1f} "
          f"vs {normal['ref_minutes'].median():.1f}; share where |difference| <= 2 min: "
          f"{((normal['vol_minutes'] - normal['ref_minutes']).abs() <= 2).mean():.3f}")
    print("\nnaive share by region (window):")
    print(d.groupby("region")["naive"].agg(["size", "mean"]).round(3).sort_values("mean").to_string())


# ---------------------------------------------------------------- part 5

def old_starts(aic, w0):
    section("PART 5 - alerts.in.ua alerts that started before the window")
    old = aic[aic["started_at"] < w0].copy()
    print(f"window start: {w0}; alerts that started earlier: {len(old)}")
    if old.empty:
        return
    print(old.groupby(["name", "location_type", "alert_type"]).size().to_string())
    old["status"] = np.where(old["finished_at"].isna(), "still running", "finished")
    old["hours_so_far"] = (old["finished_at"].fillna(pd.Timestamp.now(tz="UTC")) - old["started_at"]).dt.total_seconds() / 3600
    print("\noldest 10:")
    cols = ["name", "location_title_en", "location_type", "alert_type", "started_at", "status", "hours_so_far"]
    print(old.sort_values("started_at")[cols].head(10).round(1).to_string(index=False))


def main():
    vol = load(download(VOLUNTEER))
    aic, snap, folder = load_snapshot()
    w0 = snap - pd.DateOffset(months=1)
    w1 = vol["started_at"].max()
    section("SETUP")
    print(f"alerts.in.ua snapshot {folder}: {len(aic)} records, {aic['name'].nunique()} units")
    print(f"comparison window: {w0} -> {w1} (volunteer data ends at w1)")

    w0_ns = w0.tz_localize(None).as_unit("ns").value
    n_min = int(np.ceil((w1.tz_localize(None).as_unit("ns").value - w0_ns) / MIN_NS))
    vol_w = vol[(vol["finished_at"] >= w0) & (vol["started_at"] <= w1)]  # alerts overlapping the window
    air = aic[aic["alert_type"] == "air_raid"]
    print(f"volunteer alerts in window: {len(vol_w)}; alerts.in.ua air_raid records: {len(air)} "
          f"(other types excluded: {len(aic) - len(air)})")

    alert_levels(aic)
    kyiv_city(vol_w, air, snap, w0_ns, n_min)
    all_oblasts(vol_w, air, snap, w0_ns, n_min)
    naive_alerts(vol_w, air, snap)
    old_starts(aic, w0)


if __name__ == "__main__":
    main()
