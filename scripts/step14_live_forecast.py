"""Step 14: live B3 forecast for the next 24 h from the freshest alerts.in.ua snapshot (D42).

Same rules as the evaluation: raion under alert = oblast-wide OR raion alert (air raid only), overlapping
alerts merged, gaps <= 2 min joined; B3 = share of time under alert in the 7 days before the data cutoff.
Data cutoff ("data up to") = start time of the live snapshot (folder name); later starts are ignored.
Forecast window = from the next full hour after this script runs, for 24 h (B3 is flat across hours),
so a commit made right after this script is provably earlier than the window start.
Also writes the observed yellow/red mix and median durations of the last 7 days per oblast (D33).

Outputs (committed): forecasts/forecast_<snapshot>.csv, forecasts/forecast_<snapshot>.json
Run from the project root (after the live snapshot finished):
    python scripts/step14_live_forecast.py | tee logs/step14_live_forecast_output.txt
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import config  # noqa: E402

LIVE = config.ROOT / "data" / "raw" / "alerts_in_ua_live"
OUT = config.ROOT / "forecasts"
H = config.HOUR_NS
MIN = config.MIN_NS


def to_ns(series):
    return pd.to_datetime(series, utc=True, format="ISO8601").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


def merged_share(starts, ends, a, b):
    """Merge overlapping intervals, join gaps <= 2 min, return share of [a, b) under alert."""
    if len(starts) == 0:
        return 0.0
    o = np.argsort(starts)
    s, e = starts[o], ends[o]
    run = np.maximum.accumulate(e)
    new = np.ones(len(s), bool)
    new[1:] = s[1:] > run[:-1] + 2 * MIN
    first = np.flatnonzero(new)
    ms, me = s[first], np.maximum.reduceat(e, first)
    return float(np.clip(np.minimum(me, b) - np.maximum(ms, a), 0, None).sum() / (b - a))


def main():
    folder = sorted(p for p in LIVE.iterdir() if p.is_dir())[-1]
    manifest = json.loads((folder / "manifest.json").read_text())
    ok = [e for e in manifest if e["http_status"] == 200]
    print(f"live snapshot {folder.name}: {len(ok)}/{len(manifest)} units OK")
    if len(ok) < 24:
        raise SystemExit("STOP: incomplete snapshot -> option B (existing data, labeled)")
    cutoff = pd.to_datetime(folder.name, format="%Y%m%dT%H%M%SZ", utc=True)
    T = int(cutoff.tz_localize(None).as_unit("ns").value)
    a7 = T - 7 * 24 * H

    rec = []
    for e in ok:
        d = pd.DataFrame(json.loads((folder / e["file"]).read_text())["alerts"]).assign(unit_name=e["name"])
        rec.append(d)
    a = pd.concat(rec, ignore_index=True)
    a = a[a["alert_type"] == "air_raid"].copy()
    a["s"] = to_ns(a["started_at"])
    a["e"] = to_ns(a["finished_at"].fillna(cutoff.isoformat()))
    a = a[a["s"] < T]                                              # data window ends at the cutoff
    a["e"] = np.minimum(a["e"], T)                                 # running at the cutoff: counted up to it
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
        rows.append({"unit": x["unit"], "raion_uk": x["raion_uk"], "hdx_pcode": x["hdx_pcode"], "excluded": bool(x["excluded"]),
                     "p_alert_b3": merged_share(g["s"].to_numpy(), g["e"].to_numpy(), a7, T)})
    f = pd.DataFrame(rows)

    # observed yellow/red mix in the last 7 days per oblast (descriptive, D33)
    w = a[a["s"] >= a7]
    mix = w.groupby("unit_name").apply(lambda g: pd.Series({
        "records_7d": len(g), "red_share": float((g["alert_level"] == "red").mean()),
        "median_min_red": float(((g["e"] - g["s"]) / MIN)[g["alert_level"] == "red"].median()),
        "median_min_yellow": float(((g["e"] - g["s"]) / MIN)[g["alert_level"] == "yellow"].median())}), include_groups=False)

    now = pd.Timestamp.now(tz="UTC")
    win0 = (now + pd.Timedelta(minutes=1)).ceil("h")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    meta = {"model": "B3 (7-day rolling share; best on the post-reform test, D39)",
            "data_up_to_utc": cutoff.isoformat(), "data_up_to_kyiv": cutoff.tz_convert(config.TZ).isoformat(),
            "forecast_made_utc": now.isoformat(timespec="seconds"),
            "window_start_utc": win0.isoformat(), "window_end_utc": (win0 + pd.Timedelta(hours=24)).isoformat(),
            "window_start_kyiv": win0.tz_convert(config.TZ).isoformat(),
            "code_commit_before_forecast": commit, "units": int(len(f)), "modeled_units": int((~f["excluded"]).sum()),
            "note": "probability that the raion is under an announced air-raid alert at any given moment in the window; "
                    "same value for every hour (B3 is flat). Not a warning system - follow official alerts.",
            "observed_last_7_days_by_oblast": mix.round(3).reset_index().to_dict(orient="records")}
    OUT.mkdir(exist_ok=True)
    stem = f"forecast_{folder.name}"
    f.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"data up to: {meta['data_up_to_kyiv']} (Kyiv); forecast window: {meta['window_start_kyiv']} + 24 h")
    print(f"units: {len(f)} (modeled {meta['modeled_units']}); B3 probability: median {f.loc[~f['excluded'], 'p_alert_b3'].median():.3f}, "
          f"min {f.loc[~f['excluded'], 'p_alert_b3'].min():.3f}, max {f.loc[~f['excluded'], 'p_alert_b3'].max():.3f}")
    print(f"saved -> forecasts/{stem}.csv and .json  -> COMMIT NOW, before {meta['window_start_utc']}")


if __name__ == "__main__":
    main()
