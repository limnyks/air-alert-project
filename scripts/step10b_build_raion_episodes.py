"""Step 10b: build raion on/off episodes (D28) and run the per-raion checks (rules of D26/D27).

Raion under alert = oblast-wide alert in its oblast OR an alert for that raion.
  official file: level "oblast" rows of the oblast + level "raion" rows with the raion's (official) name
  alerts.in.ua:  air_raid records of type "oblast" for the oblast + type "raion" with the raion's ID
Kyiv City is one unit (all its rows in both sources). Gaps <= 2 min are joined (D27-1).
Alerts still running at the snapshot are "censored".

Rules (fixed before seeing results):
  switch rule  (D26-B): share after the oblast's switch vs the 12 months before; outside x1.5 -> train after switch
  saturation   (D26-D): share > 80% in the last 92 training days or in the adaptation week -> excluded

Outputs:
  data/processed/raion_episodes.csv   (local only: contains alerts.in.ua-derived rows)
  logs/step10b_raion_checks.csv       (one row per unit, aggregated numbers only)

Processes one unit at a time to keep memory small.

Run from the project root:
    python scripts/step10b_build_raion_episodes.py | tee logs/step10b_output.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

from step05_inspect_data import OFFICIAL, download, load, merge_intervals, section, to_ns
from step06_diagnose_sources import MIN_NS, coverage
from step07b_compare_reference import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
RAIONS = ROOT / "config" / "raions.csv"
CAPITALS = ROOT / "config" / "capitals.csv"
OUT = ROOT / "data" / "processed" / "raion_episodes.csv"
CHECKS_OUT = ROOT / "logs" / "step10b_raion_checks.csv"

GAP_MIN = 2
ADAPT_DAYS, SAT_THRESHOLD, RATIO_RULE = 7, 0.80, 1.5
BEFORE_SWITCH_DAYS, LAST_TRAIN_DAYS, CONTINUITY_DAYS = 365, 92, 28
DAY_MIN = 24 * 60


def ns(ts):
    return ts.tz_localize(None).as_unit("ns").value


def join_gaps(s, e, tol_min=GAP_MIN):
    """Merge sorted, non-overlapping episodes whose gap is <= tol_min minutes."""
    if len(s) == 0:
        return s, e
    keep = np.ones(len(s), dtype=bool)
    keep[1:] = (s[1:] - e[:-1]) > tol_min * MIN_NS
    first = np.flatnonzero(keep)
    return s[first], np.maximum.reduceat(e, first)


def episodes(df, censor_end=None):
    ends = df["finished_at"] if censor_end is None else df["finished_at"].fillna(censor_end)
    s, e = merge_intervals(to_ns(df["started_at"]), to_ns(ends))
    return join_gaps(s, e)


def share(cov, a, b):
    seg = cov[max(a, 0):max(b, 0)]
    return float(seg.mean()) if len(seg) else np.nan


def main():
    units = pd.read_csv(RAIONS)
    capital_raions = set(pd.read_csv(CAPITALS)["official_raion"].dropna())
    off = load(download(OFFICIAL)).drop_duplicates()
    aic, snap, folder = load_snapshot()
    air = aic[aic["alert_type"] == "air_raid"].copy()
    air["uid"] = pd.to_numeric(air["location_uid"], errors="coerce")

    t0, t_end = off["started_at"].min(), off["started_at"].max()
    w0 = snap - pd.DateOffset(months=1)
    t0_ns, end_ns, snap_ns, w0_ns = ns(t0), ns(t_end), ns(snap), ns(w0)
    n_off = int(np.ceil((end_ns - t0_ns) / MIN_NS))
    n_aic = int(np.ceil((snap_ns - w0_ns) / MIN_NS))
    switch = off[off["level"] == "oblast"].groupby("oblast")["started_at"].max()

    section("SETUP")
    print(f"units: {len(units)} (raions + Kyiv City); official {t0} -> {t_end}; snapshot {folder}; "
          f"adaptation {w0} -> {w0 + pd.Timedelta(days=ADAPT_DAYS)}; gap rule <= {GAP_MIN} min")

    rows, checks, gaps = [], [], {"official": [], "alerts_in_ua": []}
    for _, u in units.iterrows():
        oblast, raion_en, uid = u["oblast_en"], u["raion_en"], int(u["aic_uid"])
        is_city = oblast == "Kyiv City"
        o = off[off["oblast"] == oblast]
        if not is_city:
            o = o[(o["level"] == "oblast") | ((o["level"] == "raion") & (o["raion"] == raion_en))]
        a = air[air["name"] == oblast]
        if not is_city:
            a = a[(a["location_type"] == "oblast") | ((a["location_type"] == "raion") & (a["uid"] == uid))]
        so, eo = episodes(o)
        sa, ea = episodes(a, censor_end=snap)
        unit = "Kyiv City" if is_city else f"{oblast} / {raion_en}"
        for src, s, e in (("official", so, eo), ("alerts_in_ua", sa, ea)):
            assert np.all(s[1:] > e[:-1]), f"overlap after merge: {unit} {src}"
            gaps[src].append((s[1:] - e[:-1]) / MIN_NS)
            rows.append(pd.DataFrame({"unit": unit, "oblast": oblast, "raion": raion_en if not is_city else "",
                                      "aic_uid": uid, "source": src,
                                      "start_utc": pd.to_datetime(s, utc=True), "end_utc": pd.to_datetime(e, utc=True),
                                      "end_type": np.where((src == "alerts_in_ua") & (e == snap_ns), "censored", "known")}))

        cov_o = coverage(so, eo, t0_ns, n_off)
        cov_a = coverage(sa, ea, w0_ns, n_aic)
        if is_city:
            sw, before, after = None, np.nan, np.nan
        else:
            sw = switch[oblast]
            k = int((ns(sw) - t0_ns) // MIN_NS)
            before, after = share(cov_o, k - BEFORE_SWITCH_DAYS * DAY_MIN, k), share(cov_o, k, n_off)
        ratio = after / before if before else np.nan
        shift = bool(ratio > RATIO_RULE or ratio < 1 / RATIO_RULE) if not np.isnan(ratio) else False
        last3 = share(cov_o, n_off - LAST_TRAIN_DAYS * DAY_MIN, n_off)
        adapt = share(cov_a, 0, ADAPT_DAYS * DAY_MIN)
        pre_from_ns = end_ns - CONTINUITY_DAYS * DAY_MIN * MIN_NS
        dur_pre = (eo[so >= pre_from_ns] - so[so >= pre_from_ns]) / MIN_NS
        known = ea != snap_ns
        dur_post = (ea[known & (sa >= w0_ns)] - sa[known & (sa >= w0_ns)]) / MIN_NS
        checks.append({
            "unit": unit, "oblast": oblast, "is_capital_raion": raion_en in capital_raions,
            "official_episodes": len(so), "aic_episodes": len(sa), "aic_censored": int((~known).sum()),
            "switch": sw.date() if sw is not None else None, "share_before": before, "share_after": after,
            "ratio": ratio, "train_from": (sw.date() if shift else t0.date()) if sw is not None else t0.date(),
            "share_last92_train": last3, "share_adaptation": adapt,
            "excluded": bool(last3 > SAT_THRESHOLD or adapt > SAT_THRESHOLD),
            "share_pre28": share(cov_o, n_off - CONTINUITY_DAYS * DAY_MIN, n_off), "share_post": float(cov_a.mean()),
            "eps_per_day_pre28": int((so >= pre_from_ns).sum()) / CONTINUITY_DAYS,
            "eps_per_day_post": int((sa >= w0_ns).sum()) / (n_aic / DAY_MIN),
            "median_min_pre28": float(np.median(dur_pre)) if len(dur_pre) else np.nan,
            "median_min_post": float(np.median(dur_post)) if len(dur_post) else np.nan})
        del cov_o, cov_a

    c = pd.DataFrame(checks)
    ep = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ep.to_csv(OUT, index=False)
    c.to_csv(CHECKS_OUT, index=False)

    section("BUILD")
    print(f"episodes: {len(ep)} (official {int((ep['source'] == 'official').sum())}, "
          f"alerts.in.ua {int((ep['source'] == 'alerts_in_ua').sum())}, censored {int((ep['end_type'] == 'censored').sum())})")
    empty = c[(c["official_episodes"] == 0) | (c["aic_episodes"] == 0)]
    print("units with no episodes in a source:", empty[["unit", "official_episodes", "aic_episodes"]].values.tolist() or "none")
    print(f"saved -> {OUT.relative_to(ROOT)} (local only); per-unit checks -> {CHECKS_OUT.relative_to(ROOT)}")
    print(f"\ngaps between episodes after the <= {GAP_MIN} min rule (minutes):")
    for src, parts in gaps.items():
        g = np.concatenate(parts)
        print(f"  {src:13s} n={len(g)}  <=5 min {np.mean(g <= 5):.3f}  <=10 min {np.mean(g <= 10):.3f}  median {np.median(g):.1f}")

    section("SATURATION (excluded units) and SWITCH RULE")
    print(f"excluded by the {SAT_THRESHOLD:.0%} rule: {int(c['excluded'].sum())} of {len(c)}")
    print(c.loc[c["excluded"], ["unit", "share_last92_train", "share_adaptation"]].round(3).to_string(index=False))
    after_only = c[c["train_from"] != t0.date()]
    print(f"\ntrain after switch only: {len(after_only)} units; full history: {len(c) - len(after_only)}")
    per_oblast = c.groupby("oblast").agg(units=("unit", "size"), excluded=("excluded", "sum"),
                                         after_switch=("train_from", lambda x: int((x != t0.date()).sum())))
    print(per_oblast.to_string())

    section("BEFORE (official, last 28 days) vs AFTER (alerts.in.ua window) - modeled units only")
    m = c[~c["excluded"]]
    print(m[["share_pre28", "share_post", "eps_per_day_pre28", "eps_per_day_post",
             "median_min_pre28", "median_min_post"]].describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_string())
    print("\ncapital raions (compare with the Step 8b capital table):")
    print(c.loc[c["is_capital_raion"], ["unit", "share_last92_train", "share_adaptation", "excluded", "train_from"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
