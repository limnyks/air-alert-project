"""Step 23: past-days data for the map - the FINAL model's forecasts (D47) vs what really happened.

For Kyiv days from 2026-09-13 to the data end, forecast origins at 06:00, 12:00 and 18:00 Kyiv time.
Each origin is forecast with a model refit ONLY on labels before that origin (same code path as step21b:
level half-life 7 d, hour shift + w_h refit with observation weight half-life 30 d, rows every 3 h, last 120 days;
asserted in every refit). Actual = the alert state of every raion at every hour (same definition as the labels).
Output (aggregated numbers and alert flags only): forecasts/past_days.json
Prints the headline numbers for the shown origins.

Run from the project root:
    python scripts/step23_past_days.py | tee logs/step23_output.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aaf import config, final  # noqa: E402
from aaf.baselines import hour_of_day  # noqa: E402
from step14_live_forecast import LIVE  # noqa: E402

HL, HP, STRIDE, CAP_DAYS = 7, 30, 3, 120
FIRST_DAY = pd.Timestamp("2026-09-13", tz=config.TZ)
MADE_AT = (6, 12, 18)
H = config.HOUR_NS


def main():
    folder = sorted(p for p in LIVE.iterdir() if p.is_dir())[-1]
    P = final.build_panel(folder.name)
    G, COV, S, hod = P["G"], P["COV"], P["S"], P["hod"]
    units = P["units"]
    U, N = S.shape
    num, den = final.level_parts(P, HL)
    LL = final.logit(final.smoothed_level(num, den))

    # origins: 06/12/18 Kyiv on each day from FIRST_DAY while at least +1 h of the future is known
    origins = []
    for d in pd.date_range(FIRST_DAY, pd.Timestamp(G[-1], tz="UTC").tz_convert(config.TZ).normalize(), freq="D"):
        for hr in MADE_AT:
            t = d + pd.Timedelta(hours=hr)
            j = int(np.searchsorted(G, int(t.tz_convert("UTC").tz_localize(None).as_unit("ns").value)))
            if j < N - 1 and G[j] == int(t.tz_convert("UTC").tz_localize(None).as_unit("ns").value):
                origins.append(j)
    origins = np.array(origins)
    print(f"snapshot {folder.name}; origins {len(origins)} ({pd.Timestamp(G[origins[0]], tz='UTC').tz_convert(config.TZ):%d %b %H:%M} -> "
          f"{pd.Timestamp(G[origins[-1]], tz='UTC').tz_convert(config.TZ):%d %b %H:%M} Kyiv), raions {U}")

    # training pool: origins every STRIDE h over the last CAP_DAYS + eval span, sorted by label time
    valid = np.zeros(N, bool)
    valid[1:] = COV[:-1]
    j0 = max(int(origins[0]) - (CAP_DAYS + 5) * 24, 1)
    tr_o = np.array([j for j in range(j0, N - 1, STRIDE) if valid[j]])
    ru, ro, rh = (a.ravel() for a in np.meshgrid(np.arange(U), tr_o, np.arange(1, 25), indexing="ij"))
    lab = ro + rh
    keep = (lab < N) & COV[np.minimum(lab, N - 1)]
    ru, ro, rh, lab = ru[keep], ro[keep], rh[keep], lab[keep]
    order = np.argsort(lab, kind="stable")
    ru, ro, rh, lab = ru[order], ro[order], rh[order], lab[order]
    ry = S[ru, lab].astype(float)
    ra = S[ru, ro].astype(float)
    rk = hod[lab].astype(np.int64)
    print(f"training pool {len(lab):,} rows")

    probs = np.full((len(origins), U, 24), np.nan)
    beta = None
    for oi, jD in enumerate(origins):
        lo, hi = np.searchsorted(lab, jD - CAP_DAYS * 24), np.searchsorted(lab, jD, side="left")
        assert hi == 0 or lab[hi - 1] < jD, "training label at or after the forecast origin"
        w = 0.5 ** ((jD - lab[lo:hi]) / (24.0 * HP))
        beta = final.fit_logistic(LL[ru[lo:hi], ro[lo:hi]], rk[lo:hi], (rh[lo:hi] - 1).astype(np.int64), ra[lo:hi], ry[lo:hi], w, beta)
        probs[oi, :, :] = final.predict(LL[:, jD], hour_of_day(G[jD] + np.arange(1, 25) * H)[None, :].repeat(U, 0), S[:, jD].astype(float), beta)
    print(f"{len(origins)} refits, each on labels strictly before its origin")

    # actual state per raion per hour, from the first origin to the data end ('?' = not known yet)
    base = int(origins[0])
    actual = {u: "".join("1" if v else "0" for v in S[i, base:N]) for i, u in enumerate(units)}
    pct = np.rint(probs * 100).astype(int)
    out = {"snapshot": folder.name, "units": units, "model": "final hour-by-hour model (D47), refit before every origin",
           "base_utc": pd.Timestamp(G[base], tz="UTC").isoformat(), "n_hours": int(N - base),
           "origins": [int(j - base) for j in origins],
           "p": pct.transpose(0, 1, 2).tolist(), "actual": actual}
    (config.ROOT / "forecasts" / "past_days.json").write_text(json.dumps(out, separators=(",", ":")))
    size = (config.ROOT / "forecasts" / "past_days.json").stat().st_size / 1e6
    print(f"saved -> forecasts/past_days.json ({size:.2f} MB)")

    # headline numbers over every shown (origin, horizon) with a known outcome
    yy, pp, cnt = [], [], 0
    for oi, jD in enumerate(origins):
        for h in range(1, 25):
            if jD + h <= N - 1 and COV[jD + h]:
                yy.append(S[:, jD + h].astype(float)); pp.append(probs[oi, :, h - 1]); cnt += 1
    yy, pp = np.concatenate(yy), np.concatenate(pp)
    print(f"known outcomes shown: {cnt} (origin, hour) maps x {U} raions")
    print(f"Brier {np.mean((pp - yy) ** 2):.4f}; mean forecast where an alert happened {pp[yy == 1].mean():.3f}, where not {pp[yy == 0].mean():.3f}; "
          f"alert rate {yy.mean():.3f}")
    bins = np.minimum((pp * 10).astype(int), 9)
    print("reliability (forecast bin: mean forecast / observed / n): " + "; ".join(
        f"{b/10:.1f}: {pp[bins == b].mean():.2f}/{yy[bins == b].mean():.2f}/{(bins == b).sum()}" for b in range(10) if (bins == b).any()))


if __name__ == "__main__":
    main()
