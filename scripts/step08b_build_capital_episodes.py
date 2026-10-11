"""Step 8b: build the capital episodes table (D20, D23, D24, D26) and run the planned checks.

Capital under alert = oblast-wide alert in its oblast OR its raion OR its own city/hromada
(names from config/capitals.csv). Overlapping alerts are merged into on/off episodes.
Training source: official file (to its end, 2026-09-07). Test source: alerts.in.ua snapshot,
air_raid only. Alerts still running at the snapshot are marked "censored".

Output (local only, gitignored - contains alerts.in.ua-derived rows):
    data/processed/capital_episodes.csv
Also writes logs/step08b_monthly_share_official.csv (official-derived monthly shares only).

Prints: build checks, gap sizes, and the checks for solutions A-D of D26.

Run from the project root:
    python scripts/step08b_build_capital_episodes.py | tee logs/step08b_output.txt
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from step05_inspect_data import OFFICIAL, VOLUNTEER, download, load, merge_intervals, section, to_ns
from step06_diagnose_sources import MIN_NS, coverage
from step07b_compare_reference import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "capitals.csv"
OUT = ROOT / "data" / "processed" / "capital_episodes.csv"
MONTHLY_OUT = ROOT / "logs" / "step08b_monthly_share_official.csv"

ADAPT_DAYS = 7               # adaptation week (D26-C)
SAT_THRESHOLD = 0.80         # D26-D
RATIO_RULE = 1.5             # D26-B
BEFORE_SWITCH_DAYS = 365     # D26-B: compare with the 12 months before the switch
LAST_TRAIN_DAYS = 92         # D26-D: last 3 months of training
CONTINUITY_DAYS = 28         # before/after statistics
BRIDGE_DAYS = 90             # volunteer bridge window before the official end
HEALTHY = ["Kharkivska oblast", "Kyivska oblast", "Dnipropetrovska oblast", "Mykolaivska oblast"]
DAY_NS = 24 * 60 * MIN_NS


def ns(ts):
    return ts.tz_localize(None).as_unit("ns").value


def load_capitals():
    with open(CONFIG, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["aic_titles"] = [t for t in r["aic_titles"].split("|") if t]
    return rows


def official_rows(off, cap, with_hromada=True):
    o = off[off["oblast"] == cap["region"]]
    if cap["region"] == "Kyiv City":
        return o
    m = (o["level"] == "oblast") | ((o["level"] == "raion") & (o["raion"] == cap["official_raion"]))
    if with_hromada and cap["official_hromada"]:
        m |= (o["level"] == "hromada") & (o["hromada"] == cap["official_hromada"])
    return o[m]


def aic_rows(air, cap, with_city=True):
    a = air[air["name"] == cap["region"]]
    titles = cap["aic_titles"] if with_city else [t for t in cap["aic_titles"] if t.endswith("raion") or t == "Kyiv city"]
    return a[(a["location_type"] == "oblast") | a["location_title_en"].isin(titles)]


def merged(df, censor_end=None):
    ends = df["finished_at"] if censor_end is None else df["finished_at"].fillna(censor_end)
    return merge_intervals(to_ns(df["started_at"]), to_ns(ends))


def share(cov, a, b):
    seg = cov[max(a, 0):max(b, 0)]
    return seg.mean() if len(seg) else np.nan


def main():
    caps = load_capitals()
    off = load(download(OFFICIAL)).drop_duplicates()
    vol = load(download(VOLUNTEER))
    aic, snap, folder = load_snapshot()
    air = aic[aic["alert_type"] == "air_raid"]

    t0, t_train_end = off["started_at"].min(), off["started_at"].max()
    w0 = snap - pd.DateOffset(months=1)
    adapt_end = w0 + pd.Timedelta(days=ADAPT_DAYS)
    vol_end = vol["started_at"].max()
    t0_ns, train_end_ns, snap_ns = ns(t0), ns(t_train_end), ns(snap)
    n_off = int(np.ceil((train_end_ns - t0_ns) / MIN_NS))
    w0_ns = ns(w0)
    n_aic = int(np.ceil((snap_ns - w0_ns) / MIN_NS))

    section("SETUP")
    print(f"capitals: {len(caps)}; official {t0} -> {t_train_end}; alerts.in.ua snapshot {folder}")
    print(f"adaptation week: {w0} -> {adapt_end}; final test: {adapt_end} -> {vol_end.floor('min')} (volunteer end) / {snap} (snapshot)")

    # ------------------------------------------------------------ build
    section("BUILD - capital episodes (merged on/off periods)")
    rows, gaps, checks = [], {"official": [], "alerts_in_ua": []}, []
    cov_off, cov_aic = {}, {}
    for cap in caps:
        o = official_rows(off, cap)
        a = aic_rows(air, cap)
        so, eo = merged(o)
        sa, ea = merged(a, censor_end=snap)
        for src, s, e in (("official", so, eo), ("alerts_in_ua", sa, ea)):
            assert np.all(s[1:] > e[:-1]), f"overlap after merge: {cap['region']} {src}"
            gaps[src].append((s[1:] - e[:-1]) / MIN_NS)
            for start, end in zip(s, e):
                rows.append({"region": cap["region"], "capital": cap["capital"], "source": src,
                             "start_utc": pd.Timestamp(start, tz="UTC"), "end_utc": pd.Timestamp(end, tz="UTC"),
                             "end_type": "censored" if (src == "alerts_in_ua" and end == snap_ns) else "known"})
        cov_off[cap["region"]] = coverage(so, eo, t0_ns, n_off)
        cov_aic[cap["region"]] = coverage(sa, ea, w0_ns, n_aic)
        checks.append({"capital": cap["capital"], "official_rows": len(o), "official_episodes": len(so),
                       "aic_records": len(a), "aic_episodes": len(sa),
                       "aic_censored": int((ea == snap_ns).sum()),
                       "official_last": pd.Timestamp(so.max(), tz="UTC").date() if len(so) else None})
    c = pd.DataFrame(checks).set_index("capital")
    print(c.to_string())
    missing = c[(c["official_episodes"] == 0) | (c["aic_episodes"] == 0)]
    if len(missing):
        raise SystemExit(f"STOP: capitals without episodes in a source: {list(missing.index)}")
    print("check: every capital has episodes in both sources - OK; no overlaps after merge - OK")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {len(rows)} episodes -> {OUT.relative_to(ROOT)} (local only)")

    print("\ngaps between consecutive episodes (minutes, pooled):")
    for src, parts in gaps.items():
        g = np.concatenate(parts)
        print(f"  {src:13s} n={len(g)}  <=1 min {np.mean(g <= 1):.3f}  <=2 min {np.mean(g <= 2):.3f}  "
              f"<=5 min {np.mean(g <= 5):.3f}  median {np.median(g):.1f}")

    # ------------------------------------------------------------ A
    section("CHECK A - how much alert time comes only from city/hromada records")
    a_rows = []
    for cap in caps:
        o_all, o_nohrom = official_rows(off, cap), official_rows(off, cap, with_hromada=False)
        ca = coverage(*merged(o_nohrom), t0_ns, n_off)
        extra_off = (cov_off[cap["region"]] & ~ca).sum() / max(cov_off[cap["region"]].sum(), 1)
        a_nocity = aic_rows(air, cap, with_city=False)
        cb = coverage(*merged(a_nocity, censor_end=snap), w0_ns, n_aic)
        extra_aic = (cov_aic[cap["region"]] & ~cb).sum() / max(cov_aic[cap["region"]].sum(), 1)
        hrom = o_all[o_all["level"] == "hromada"]
        long_hrom = int(((hrom["finished_at"] - hrom["started_at"]).dt.total_seconds() > 24 * 3600).sum())
        a_rows.append({"capital": cap["capital"], "official_extra_share": extra_off, "aic_extra_share": extra_aic,
                       "official_hromada_rows_over_24h": long_hrom})
    print(pd.DataFrame(a_rows).set_index("capital").round(4).to_string())

    # ------------------------------------------------------------ B
    section(f"CHECK B - label structure switch (official): share before vs after, rule x{RATIO_RULE}")
    month_bounds = pd.date_range(t0.tz_localize(None).to_period("M").start_time, t_train_end.tz_localize(None), freq="MS")
    m_idx = (month_bounds.as_unit("ns").asi8 - t0_ns) // MIN_NS
    mid = np.searchsorted(m_idx, np.arange(n_off), side="right") - 1
    monthly, b_rows = {}, []
    for cap in caps:
        cov = cov_off[cap["region"]]
        monthly[cap["capital"]] = np.bincount(mid, weights=cov.astype(np.float64), minlength=len(m_idx)) / np.bincount(mid, minlength=len(m_idx))
        if cap["region"] == "Kyiv City":
            b_rows.append({"capital": cap["capital"], "switch": None, "before": np.nan, "after": np.nan,
                           "ratio": np.nan, "train_from": t0.date(), "note": "oblast-level unit, no switch"})
            continue
        obl = off[(off["oblast"] == cap["region"]) & (off["level"] == "oblast")]
        sw = obl["started_at"].max()
        k = int((ns(sw) - t0_ns) // MIN_NS)
        before = share(cov, k - BEFORE_SWITCH_DAYS * 24 * 60, k)
        after = share(cov, k, n_off)
        ratio = after / before if before else np.nan
        shift = ratio > RATIO_RULE or ratio < 1 / RATIO_RULE
        b_rows.append({"capital": cap["capital"], "switch": sw.date(), "before": before, "after": after,
                       "ratio": ratio, "train_from": sw.date() if shift else t0.date(),
                       "note": "after switch only" if shift else "full history"})
    print(pd.DataFrame(b_rows).set_index("capital").round(3).to_string())
    m = pd.DataFrame(monthly, index=month_bounds.to_period("M")).round(4)
    m.to_csv(MONTHLY_OUT)
    print(f"\nmonthly shares (official, all capitals) saved -> {MONTHLY_OUT.relative_to(ROOT)}")
    print("last 12 months, selected capitals:")
    print(m[["Kyiv", "Kharkiv", "Dnipro", "Odesa", "Lviv", "Kramatorsk"]].tail(12).to_string())

    # ------------------------------------------------------------ C
    section("CHECK C - consistency official -> alerts.in.ua")
    print(f"(1) before/after statistics: official last {CONTINUITY_DAYS} days vs alerts.in.ua whole snapshot window")
    ep = pd.DataFrame(rows)
    ep["minutes"] = (ep["end_utc"] - ep["start_utc"]).dt.total_seconds() / 60
    pre_from = t_train_end - pd.Timedelta(days=CONTINUITY_DAYS)
    c_rows = []
    for cap in caps:
        e_off = ep[(ep["capital"] == cap["capital"]) & (ep["source"] == "official") & (ep["start_utc"] >= pre_from)]
        e_aic = ep[(ep["capital"] == cap["capital"]) & (ep["source"] == "alerts_in_ua") & (ep["start_utc"] >= w0)]
        c_rows.append({"capital": cap["capital"],
                       "share_pre": share(cov_off[cap["region"]], n_off - CONTINUITY_DAYS * 24 * 60, n_off),
                       "share_post": cov_aic[cap["region"]].mean(),
                       "episodes_per_day_pre": len(e_off) / CONTINUITY_DAYS,
                       "episodes_per_day_post": len(e_aic) / (n_aic / (24 * 60)),
                       "median_min_pre": e_off["minutes"].median(),
                       "median_min_post": e_aic.loc[e_aic["end_type"] == "known", "minutes"].median()})
    print(pd.DataFrame(c_rows).set_index("capital").round(3).to_string())

    print(f"\n(2) volunteer bridge (healthy regions): Jaccard of volunteer oblast alert time with the capital series")
    bridge = []
    for region in HEALTHY:
        cap = next(cp for cp in caps if cp["region"] == region)
        v = vol[vol["region"] == region]
        vs, ve = to_ns(v["started_at"]), to_ns(v["finished_at"])
        vc_off = coverage(vs, ve, t0_ns, n_off)[n_off - BRIDGE_DAYS * 24 * 60:]
        co = cov_off[region][n_off - BRIDGE_DAYS * 24 * 60:]
        n_post = int((ns(vol_end) - w0_ns) // MIN_NS)
        vc_aic = coverage(vs, ve, w0_ns, n_aic)[:n_post]
        ca = cov_aic[region][:n_post]
        j = lambda x, y: (x & y).sum() / max((x | y).sum(), 1)
        bridge.append({"region": region, "capital": cap["capital"],
                       f"jaccard_vs_official_last_{BRIDGE_DAYS}d": j(vc_off, co),
                       "jaccard_vs_aic_window": j(vc_aic, ca)})
    print(pd.DataFrame(bridge).set_index("region").round(3).to_string())

    # ------------------------------------------------------------ D
    section(f"CHECK D - saturation (exclude if > {SAT_THRESHOLD:.0%} of time under alert)")
    d_rows = []
    adapt_n = ADAPT_DAYS * 24 * 60
    for cap in caps:
        last3 = share(cov_off[cap["region"]], n_off - LAST_TRAIN_DAYS * 24 * 60, n_off)
        adapt = cov_aic[cap["region"]][:adapt_n].mean()
        d_rows.append({"capital": cap["capital"], f"share_last_{LAST_TRAIN_DAYS}d_train": last3,
                       "share_adaptation_week": adapt,
                       "exclude": bool(last3 > SAT_THRESHOLD or adapt > SAT_THRESHOLD)})
    d = pd.DataFrame(d_rows).set_index("capital")
    print(d.round(3).to_string())
    print("excluded by rule:", list(d.index[d["exclude"]]) or "none")


if __name__ == "__main__":
    main()
