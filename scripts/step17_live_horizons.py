"""Step 17: live forecast for +1 h, +6 h and +24 h from the freshest alerts.in.ua snapshot (D44).

Per horizon, the model that was best on the validation folds (step16): +1 h -> B2 (calibrated persistence),
+6 h and +24 h -> B4 (exponentially weighted share, half-life from config/b4_halflife.json). B3 is kept as the
tested reference column. Target = P(raion under an announced air-raid alert at the moment cutoff + h).

Data rules as in step14 (oblast-wide OR raion alert, air raid only, merged, gaps <= 2 min joined), plus:
- data cutoff T = snapshot start (folder name); later starts ignored; an end after T (or missing) is unknown at T;
- coverage = the 30 days before T (the month_ago window); older records are not coverage (M12);
- B2 transition table from the covered hours of the 8 weeks before T (here: only the 30 covered days).
Outputs (commit right away, before cutoff + 1 h): forecasts/horizons_<snapshot>.csv/.json

Run from the project root (after the live snapshot finished):
    python scripts/step17_live_horizons.py | tee logs/step17_output.txt
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aaf import baselines as bl, config  # noqa: E402
from aaf.history import UNKNOWN_END, History  # noqa: E402
from step14_live_forecast import LIVE, OUT, merged_share, to_ns  # noqa: E402

H, MIN = config.HOUR_NS, config.MIN_NS
DAY = 24 * H
HORIZONS = (1, 6, 24)
B4 = json.loads((config.ROOT / "config" / "b4_halflife.json").read_text())


def merged_history(s, e, T):
    """Merge overlapping / near-touching alerts into one leakage-safe History at T."""
    if len(s) == 0:
        return History(np.zeros(0, np.int64), np.zeros(0, np.int64))
    o = np.argsort(s, kind="stable")
    s, e = s[o], e[o]
    run = np.maximum.accumulate(e)
    new = np.ones(len(s), bool)
    new[1:] = s[1:] > run[:-1] + 2 * MIN
    first = np.flatnonzero(new)
    ms, me = s[first], np.maximum.reduceat(e, first)
    return History(ms.astype(np.int64), np.where(me > T, UNKNOWN_END, me).astype(np.int64))


def main():
    named = [x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--folder=")]   # dry runs on an older snapshot
    folder = LIVE / named[0] if named else sorted(p for p in LIVE.iterdir() if p.is_dir())[-1]
    manifest = json.loads((folder / "manifest.json").read_text())
    ok = [e for e in manifest if e["http_status"] == 200]
    print(f"live snapshot {folder.name}: {len(ok)}/{len(manifest)} units OK")
    if len(ok) < 24:
        raise SystemExit("STOP: incomplete snapshot -> option A (existing data, labeled)")
    cutoff = pd.to_datetime(folder.name, format="%Y%m%dT%H%M%SZ", utc=True)
    T = int(cutoff.tz_localize(None).as_unit("ns").value)
    since = T - 30 * DAY
    covered = lambda t: (np.asarray(t) >= since) & (np.asarray(t) < T)  # noqa: E731

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
    rai.loc[rai["unit"] == "Kyiv City", "hdx_pcode"] = "UA8000"
    u = units.merge(rai[["unit", "oblast_en", "aic_uid", "hdx_pcode", "raion_uk"]], on="unit", how="left")

    rows = []
    for _, x in u.iterrows():
        g = a[a["unit_name"] == x["oblast_en"]]
        if x["oblast_en"] != "Kyiv City":
            g = g[(g["location_type"] == "oblast") | ((g["location_type"] == "raion") & (g["uid"] == x["aic_uid"]))]
        s, e = g["s"].to_numpy(np.int64), g["e"].to_numpy(np.int64)
        h = merged_history(s, e, T)
        b2 = bl.predict("B2", bl.fit(h, T, covered), h, [T])[0]
        b4 = bl.ewshare(h, [T], B4["half_life_days"], covered=covered, since=since)[0]
        rows.append({"unit": x["unit"], "raion_uk": x["raion_uk"], "hdx_pcode": x["hdx_pcode"], "excluded": bool(x["excluded"]),
                     "on_at_cutoff": int(h.state_at([T])[0]),
                     "p_1h": float(b2[0]), "p_6h": float(b4[5]), "p_24h": float(b4[23]),
                     "b3_ref": merged_share(s, np.minimum(e, T), T - 7 * DAY, T)})
    f = pd.DataFrame(rows)

    # observed yellow/red mix in the last 7 days per oblast (descriptive, D33; as in step14)
    w = a[a["s"] >= T - 7 * DAY].assign(e=lambda d: np.minimum(d["e"], T))
    mix = w.groupby("unit_name").apply(lambda g: pd.Series({
        "records_7d": len(g), "red_share": float((g["alert_level"] == "red").mean()),
        "median_min_red": float(((g["e"] - g["s"]) / MIN)[g["alert_level"] == "red"].median()),
        "median_min_yellow": float(((g["e"] - g["s"]) / MIN)[g["alert_level"] == "yellow"].median())}), include_groups=False)

    now = pd.Timestamp.now(tz="UTC")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    targets = {f"+{k}h": (cutoff + pd.Timedelta(hours=k)) for k in HORIZONS}
    meta = {"models": {"+1h": "B2 calibrated persistence (best at +1 h on validation)",
                       "+6h": f"B4 exponentially weighted share, half-life {B4['half_life_days']} d (chosen on validation, not tested on held-out data)",
                       "+24h": f"B4 exponentially weighted share, half-life {B4['half_life_days']} d (chosen on validation, not tested on held-out data)",
                       "b3_ref": "B3 7-day rolling share (best on the post-reform test, D39)"},
            "data_up_to_utc": cutoff.isoformat(), "data_up_to_kyiv": cutoff.tz_convert(config.TZ).isoformat(),
            "forecast_made_utc": now.isoformat(timespec="seconds"),
            "target_times_utc": {k: v.isoformat() for k, v in targets.items()},
            "target_times_kyiv": {k: v.tz_convert(config.TZ).isoformat() for k, v in targets.items()},
            "code_commit_before_forecast": commit, "units": int(len(f)), "modeled_units": int((~f["excluded"]).sum()),
            "note": "probability that the raion is under an announced air-raid alert at the target moment. "
                    "Not a warning system - follow official alerts.",
            "observed_last_7_days_by_oblast": mix.round(3).reset_index().to_dict(orient="records")}
    stem = f"horizons_{folder.name}"
    OUT.mkdir(exist_ok=True)
    f.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    mod = f[~f["excluded"]]
    print(f"data up to: {meta['data_up_to_kyiv']} (Kyiv); targets: " + ", ".join(f"{k} {v.tz_convert(config.TZ):%H:%M %d %b}" for k, v in targets.items()))
    print(f"modeled units: {len(mod)}; under alert at cutoff: {int(mod['on_at_cutoff'].sum())}")
    print(mod[["p_1h", "p_6h", "p_24h", "b3_ref"]].describe().loc[["mean", "50%", "min", "max"]].round(3).to_string())
    print(f"saved -> forecasts/{stem}.csv and .json  -> COMMIT NOW, before {targets['+1h'].isoformat()}")


if __name__ == "__main__":
    main()
