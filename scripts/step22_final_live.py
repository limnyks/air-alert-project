"""Step 22: live forecast of the FINAL model (D47), +1...+24 h, from the freshest alerts.in.ua snapshot.

Same model as step21b: level half-life 7 d (smoothed), hour profile and w_h refit now on all labels before
the forecast origin (observation weight half-life 30 d, rows every 3 h, last 120 days).
Origin = the last full hour before the data cutoff; targets = origin + 1...24 h (Kyiv time labels).
alert_now = state at the data cutoff (a running alert counts as ON); the level uses complete hours only.
Output (commit right away, before the first target): forecasts/final_<snapshot>.csv/.json

Run from the project root:
    python scripts/step22_final_live.py | tee logs/step22_output.txt
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aaf import config, final  # noqa: E402
from step14_live_forecast import LIVE, OUT  # noqa: E402

HL, HP, STRIDE = 7, 30, 3
H = config.HOUR_NS


def main():
    named = [x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--folder=")]   # dry runs on an older snapshot
    folder = LIVE / named[0] if named else sorted(p for p in LIVE.iterdir() if p.is_dir())[-1]
    manifest = json.loads((folder / "manifest.json").read_text())
    ok = sum(e["http_status"] == 200 for e in manifest)
    print(f"live snapshot {folder.name}: {ok}/{len(manifest)} units OK")
    if ok < 24:
        raise SystemExit("STOP: incomplete snapshot - keep the previous forecast on the map and say so")
    P = final.build_panel(folder.name)
    G, COV, F, S, hod = P["G"], P["COV"], P["F"], P["S"], P["hod"]
    U, N = S.shape
    T = P["T"]
    origin = int(G[-1] + H)                                   # last full hour <= cutoff (bins end at origin)

    # level at the origin: all complete covered hours before it
    d = 0.5 ** (1.0 / (24.0 * HL))
    x = np.where(COV[None, :], F, 0.0)
    num = lfilter([1.0], [1.0, -d], x, axis=1)[:, -1:]
    den = lfilter([1.0], [1.0, -d], np.broadcast_to(COV.astype(float), x.shape), axis=1)[:, -1:]
    lvl = final.smoothed_level(num, den)[:, 0]

    # refit s and w_h on all labels before the origin (same as the daily refit in step21b)
    valid = np.zeros(N, bool)
    valid[1:] = COV[:-1]
    j0 = N - 120 * 24 - 24
    tr_o = np.array([j for j in range(max(j0, 1), N - 1, STRIDE) if valid[j]])
    ru, ro, rh = (a.ravel() for a in np.meshgrid(np.arange(U), tr_o, np.arange(1, 25), indexing="ij"))
    lab = ro + rh
    keep = (lab < N) & COV[np.minimum(lab, N - 1)]
    ru, ro, rh, lab = ru[keep], ro[keep], rh[keep], lab[keep]
    num_all, den_all = final.level_parts(P, HL)
    LL = final.logit(final.smoothed_level(num_all, den_all))
    wt = 0.5 ** ((N - lab) / (24.0 * HP))
    beta = final.fit_logistic(LL[ru, ro], hod[lab].astype(np.int64), (rh - 1).astype(np.int64),
                              S[ru, ro].astype(float), S[ru, lab].astype(float), wt)
    assert lab.max() < N

    targets = origin + np.arange(1, 25, dtype=np.int64) * H
    t_kyiv = pd.DatetimeIndex(targets.astype("datetime64[ns]")).tz_localize("UTC").tz_convert(config.TZ)
    alert_now = np.array([P["hist"][u].state_at([T])[0] for u in P["units"]], dtype=float)
    p = final.predict(final.logit(lvl), t_kyiv.hour.to_numpy()[None, :].repeat(U, 0), alert_now, beta)

    units = pd.read_csv(config.UNITS_CLEAN)
    rai = pd.read_csv(config.ROOT / "config" / "raions.csv")
    rai["unit"] = np.where(rai["oblast_en"] == "Kyiv City", "Kyiv City", rai["oblast_en"] + " / " + rai["raion_en"].fillna(""))
    rai.loc[rai["unit"] == "Kyiv City", "hdx_pcode"] = "UA8000"
    f = units[["unit", "excluded"]].merge(rai[["unit", "hdx_pcode", "raion_uk"]], on="unit", how="left")
    pr = pd.DataFrame(p, columns=[f"p_{h}h" for h in range(1, 25)]).assign(unit=P["units"], alert_now=alert_now.astype(int),
                                                                           level=lvl)
    f = f.merge(pr, on="unit", how="left")

    cutoff = pd.Timestamp(T, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    meta = {"model": "final hour-by-hour model (D47): logit p = logit(level, half-life 7 d) + hour shift + w_h x alert_now",
            "data_up_to_utc": cutoff.isoformat(), "origin_utc": pd.Timestamp(origin, tz="UTC").isoformat(),
            "forecast_made_utc": now.isoformat(timespec="seconds"),
            "target_times_utc": [pd.Timestamp(t, tz="UTC").isoformat() for t in targets],
            "hour_profile": [round(float(v), 4) for v in beta[:24]], "w_h": [round(float(v), 4) for v in beta[24:]],
            "code_commit_before_forecast": commit, "modeled_units": int(f["p_1h"].notna().sum()),
            "raions_under_alert_at_cutoff": int(alert_now.sum()),
            "note": "probability that the raion is under an announced air-raid alert at each target hour. "
                    "Not a warning system - follow official alerts."}
    stem = f"final_{folder.name}"
    f.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"data up to {cutoff.tz_convert(config.TZ):%d %b %H:%M} Kyiv; first target {t_kyiv[0]:%d %b %H:%M}, last {t_kyiv[-1]:%d %b %H:%M} Kyiv")
    print(f"raions modeled {meta['modeled_units']}, under alert at cutoff {meta['raions_under_alert_at_cutoff']}")
    m = f[f["p_1h"].notna()]
    print("median probability by horizon: " + " ".join(f"+{h}:{m[f'p_{h}h'].median():.2f}" for h in (1, 3, 6, 12, 18, 24)))
    print(f"saved -> forecasts/{stem}.csv/.json -> COMMIT NOW, before {t_kyiv[0]:%H:%M} Kyiv")


if __name__ == "__main__":
    main()
