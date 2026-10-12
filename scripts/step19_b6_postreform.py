"""Step 19: B6 = B4 + hour pattern on the logit scale (D46), evaluated on post-reform days outside the spent test window.

Data: live alerts.in.ua snapshot 20261012T120011Z only (step17 rules), coverage = 30 days before its cutoff.
Origins: every full hour (1-day warm-up), (a) before the test window with +24 h labels before 2026-09-18 08:22 UTC,
(b) after it (from 2026-10-11 08:22 UTC) with labels before the data end. The guard blocks test-window labels.
For each evaluated Kyiv day the hour profile is learned only from post-reform data before that day (expanding).
Criteria and the evidence threshold are fixed in D46.
Outputs: printed tables, logs/step19_b6_scores.csv, config/b6_logit.json.

Run from the project root:
    python scripts/step19_b6_postreform.py | tee logs/step19_output.txt
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aaf import baselines as bl, config, leakage, metrics as m  # noqa: E402
from aaf.data import ns  # noqa: E402
from aaf.history import UNKNOWN_END  # noqa: E402
from aaf.targets import fold_origins, map_target  # noqa: E402
from step14_live_forecast import LIVE, to_ns  # noqa: E402
from step17_live_horizons import merged_history  # noqa: E402

SNAP = "20261012T120011Z"
HL = json.loads((config.ROOT / "config" / "b4_halflife.json").read_text())["half_life_days"]
K_SHRINK = 7.0
MIN_DAYS = 14
H, DAY = config.HOUR_NS, 24 * config.HOUR_NS
OUT_FILE = config.ROOT / "config" / "b6_logit.json"
SCORES_FILE = config.ROOT / "logs" / "step19_b6_scores.csv"


def section(t):
    print(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


def live_histories(folder, T):
    ok = [e for e in json.loads((folder / "manifest.json").read_text()) if e["http_status"] == 200]
    a = pd.concat([pd.DataFrame(json.loads((folder / e["file"]).read_text())["alerts"]).assign(unit_name=e["name"]) for e in ok],
                  ignore_index=True)
    a = a[a["alert_type"] == "air_raid"].copy()
    a["s"] = to_ns(a["started_at"])
    fin = a["finished_at"].notna()
    a["e"] = np.int64(UNKNOWN_END)
    a.loc[fin, "e"] = to_ns(a.loc[fin, "finished_at"])
    a = a[a["s"] < T]
    a["uid"] = pd.to_numeric(a["location_uid"], errors="coerce")
    units = pd.read_csv(config.UNITS_CLEAN)
    rai = pd.read_csv(config.ROOT / "config" / "raions.csv")
    rai["unit"] = np.where(rai["oblast_en"] == "Kyiv City", "Kyiv City", rai["oblast_en"] + " / " + rai["raion_en"].fillna(""))
    u = units[~units["excluded"]].merge(rai[["unit", "oblast_en", "aic_uid"]], on="unit", how="left")
    out = {}
    for _, x in u.iterrows():
        g = a[a["unit_name"] == x["oblast_en"]]
        if x["oblast_en"] != "Kyiv City":
            g = g[(g["location_type"] == "oblast") | ((g["location_type"] == "raion") & (g["uid"] == x["aic_uid"]))]
        out[x["unit"]] = merged_history(g["s"].to_numpy(np.int64), g["e"].to_numpy(np.int64), T)
    return out


def kyiv_day_start(t):
    return ns(pd.Timestamp(int(t), tz="UTC").tz_convert(config.TZ).normalize())


def main():
    T = int(pd.to_datetime(SNAP, format="%Y%m%dT%H%M%SZ").as_unit("ns").value)
    since = T - 30 * DAY
    covered = lambda t: (np.asarray(t) >= since) & (np.asarray(t) < T)  # noqa: E731
    hist = live_histories(LIVE / SNAP, T)
    units = list(hist)
    ts = lambda v: pd.Timestamp(int(v), tz="UTC")  # noqa: E731
    before = fold_origins(ts(since + DAY), config.ADAPT[1])
    after = fold_origins(config.SNAPSHOT, ts(T))
    origins = np.concatenate([before, after])
    period = np.array(["before"] * len(before) + ["after"] * len(after))

    section("EVALUATION SET (post-reform, outside the spent test window)")
    days = pd.DatetimeIndex(origins.astype("datetime64[ns]")).tz_localize("UTC").tz_convert(config.TZ).date
    day_codes, day = np.unique(days, return_inverse=True)
    n_eval_days = len(origins) / 24
    print(f"origins: before {len(before)} ({ts(before[0]).tz_convert(config.TZ):%d %b %H:%M} -> {ts(before[-1]).tz_convert(config.TZ):%d %b %H:%M} Kyiv), "
          f"after {len(after)} ({ts(after[0]).tz_convert(config.TZ):%d %b %H:%M} -> {ts(after[-1]).tz_convert(config.TZ):%d %b %H:%M} Kyiv)")
    print(f"= {len(origins)} origins = {n_eval_days:.1f} days of origins, on {len(day_codes)} calendar days; units {len(units)}")
    exploratory = n_eval_days < MIN_DAYS
    print(f"evidence threshold {MIN_DAYS} days -> {'EXPLORATORY (too few days for a clear answer)' if exploratory else 'enough days'}")

    # expanding-window hour profile: one per evaluated Kyiv day, from post-reform data before that day starts
    d0 = {d: kyiv_day_start(origins[day == i][0]) for i, d in enumerate(day_codes)}
    prof = {d: bl.logit_hour_profile(hist.values(), since, d0[d], K_SHRINK, covered) for d in day_codes}

    y = np.stack([map_target(hist[u], origins) for u in units])
    p4 = np.stack([bl.ewshare(hist[u], origins, HL, covered=covered, since=since) for u in units])
    p6 = np.zeros_like(p4)
    for i, d in enumerate(day_codes):
        sel = day == i
        for ui, u in enumerate(units):
            p6[ui, sel] = bl.ewshare_logit(hist[u], origins[sel], HL, prof[d], covered=covered, since=since)

    section("RESULTS (macro Brier, lower is better)")
    nd = len(day_codes)
    agg = {n: m.daily_sums(m.brier_items(p, y), day, nd) for n, p in (("B4", p4), ("B6", p6))}
    rows = []
    for n, p in (("B4", p4), ("B6", p6)):
        r = {"model": n, "macro_brier": m.macro(*agg[n])[0], "macro_logloss": m.macro(*m.daily_sums(m.logloss_items(p, y), day, nd))[0],
             "ece": m.reliability(p, y)[1], "max_forecast": float(p.max())}
        for k in (1, 6, 12, 24):
            r[f"brier_+{k}h"] = m.macro(*m.daily_sums(m.brier_items(p[:, :, k - 1:k], y[:, :, k - 1:k]), day, nd))[0]
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model")
    t.to_csv(SCORES_FILE)
    print(t.round(4).to_string())
    d1 = m.bootstrap(m.skill, [agg["B6"], agg["B4"]], n_rep=1000)
    print(f"\nB6 vs B4 skill {d1[0]:+.4f}, joint day-block 95 % interval [{d1[1]:+.4f}, {d1[2]:+.4f}] ({nd} day blocks)")

    section("CRITERIA (D46)")
    c1 = d1[1] > 0
    c2 = t.loc["B6", "ece"] <= t.loc["B4", "ece"]
    n_over = int(((p6 > 0.95) & (p4 <= 0.95)).sum())
    c3 = n_over == 0
    passed = bool(c1 and c2 and c3)
    print(f"(1) interval above 0: {c1}  (2) ECE not higher ({t.loc['B6', 'ece']:.4f} vs {t.loc['B4', 'ece']:.4f}): {c2}  "
          f"(3) no B6 > 0.95 where B4 <= 0.95 ({n_over} cases): {c3}")
    verdict = ("exploratory - " if exploratory else "") + ("criteria passed" if passed else "criteria not passed")
    print(f"-> {verdict}; map model for +6/+12/+24 h: {'B6' if passed and not exploratory else 'B4 (unchanged)'}")

    section("LEAKAGE CHECK - B6 (12 units x 10 times)")
    sample = {u: hist[u] for u in units[:: max(1, len(units) // 12)]}
    times = [int(origins[i]) for i in np.linspace(0, len(origins) - 1, 10).astype(int)]
    def f(hh, tt):
        return bl.ewshare_logit(hh, [tt], HL, prof[day_codes[day[np.searchsorted(origins, tt)]]], covered=covered, since=since)[0]
    problems = leakage.check(f, sample, times)
    print(f"problems: {len(problems)}", problems[:5] if problems else "")
    if problems:
        raise SystemExit("STOP: leakage detected")

    section("EXAMPLE - live forecast at the data cutoff, +12 h vs +24 h (profile from all post-reform data before the cutoff)")
    s_live = bl.logit_hour_profile(hist.values(), since, T, K_SHRINK, covered)
    print("hour profile s(k) (logit shift; 0 = average hour): " + " ".join(f"{k:02d}:{v:+.2f}" for k, v in enumerate(s_live)))
    ex = []
    for u in ("Kyiv City", "Odeska oblast / Odeskyi raion", "Kharkivska oblast / Kharkivskyi raion", "Lvivska oblast / Lvivskyi raion"):
        if u not in hist:
            continue
        b4 = bl.ewshare(hist[u], [T], HL, covered=covered, since=since)[0]
        b6 = bl.ewshare_logit(hist[u], [T], HL, s_live, covered=covered, since=since)[0]
        ex.append({"raion": u, "B4 +12h": b4[11], "B4 +24h": b4[23], "B6 +12h": b6[11], "B6 +24h": b6[23]})
    lt = {k: ts(T + k * H).tz_convert(config.TZ).strftime("%d %b %H:%M") for k in (12, 24)}
    print(f"+12 h = {lt[12]} Kyiv, +24 h = {lt[24]} Kyiv")
    print(pd.DataFrame(ex).round(3).to_string(index=False))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    OUT_FILE.write_text(json.dumps({
        "model": "B6 = sigmoid(logit(B4) + s(hour)) (D46)", "half_life_days": HL, "k_shrink_days": K_SHRINK,
        "eval_origins": int(len(origins)), "eval_days": round(n_eval_days, 2), "exploratory": bool(exploratory),
        "criteria": {"interval_above_0": bool(c1), "ece_not_higher": bool(c2), "over_095_cases": n_over},
        "passed": passed, "use_on_map": bool(passed and not exploratory), "skill_vs_B4": [round(float(x), 4) for x in d1],
        "live_profile_logit": [round(float(v), 4) for v in s_live],
        "chosen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_commit": commit}, indent=2))
    print(f"\nsaved -> {OUT_FILE.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
