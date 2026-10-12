"""Step 20: experimental B6 live values (D46, exploratory) for the map's optional layer.

Same snapshot and rules as step17/step19; hour profile from all post-reform data before the cutoff.
+1 h is not produced here (the map uses B2 for +1 h in both layers).
Output (commit right away): forecasts/horizons_<snapshot>_b6.csv

Run from the project root:
    python scripts/step20_b6_live.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aaf import baselines as bl, config  # noqa: E402
from step14_live_forecast import LIVE, OUT  # noqa: E402
from step19_b6_postreform import HL, K_SHRINK, live_histories  # noqa: E402

DAY = 24 * config.HOUR_NS


def main():
    folder = sorted(p for p in LIVE.iterdir() if p.is_dir())[-1]
    cutoff = pd.to_datetime(folder.name, format="%Y%m%dT%H%M%SZ", utc=True)
    T = int(cutoff.tz_localize(None).as_unit("ns").value)
    since = T - 30 * DAY
    covered = lambda t: (t >= since) & (t < T)  # noqa: E731
    hist = live_histories(folder, T)
    s = bl.logit_hour_profile(hist.values(), since, T, K_SHRINK, covered)
    rows = []
    for u, h in hist.items():
        p = bl.ewshare_logit(h, [T], HL, s, covered=covered, since=since)[0]
        rows.append({"unit": u, "b6_6h": float(p[5]), "b6_12h": float(p[11]), "b6_24h": float(p[23])})
    f = pd.DataFrame(rows)
    out = OUT / f"horizons_{folder.name}_b6.csv"
    f.to_csv(out, index=False)
    print(f"data up to {cutoff.tz_convert(config.TZ)}; made {pd.Timestamp.now(tz=config.TZ):%H:%M} Kyiv; raions {len(f)}")
    print(f[["b6_6h", "b6_12h", "b6_24h"]].describe().loc[["mean", "50%", "max"]].round(3).to_string())
    print(f"saved -> {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
