"""Step 21a: data checks for the final model - DST, ongoing alerts, source reconciliation, regime shift timing.

Run from the project root:
    python scripts/step21a_final_data.py | tee logs/step21a_output.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import config, final  # noqa: E402
from aaf.data import ns  # noqa: E402
from aaf.history import UNKNOWN_END  # noqa: E402


def section(t):
    print(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


def main():
    P = final.build_panel()
    G, F, COV, kyiv = P["G"], P["F"], P["COV"], P["kyiv"]

    section("TIME: Kyiv hours from the tz database (DST)")
    for ts in ("2025-10-25 22:00", "2025-10-26 02:00", "2026-03-28 22:00", "2026-03-29 02:00", "2026-10-24 22:00", "2026-10-25 02:00"):
        k = pd.Timestamp(ts, tz="UTC").tz_convert(config.TZ)
        print(f"  {ts} UTC -> {k:%Y-%m-%d %H:%M} Kyiv (UTC{k:%z})")

    section("ONGOING ALERTS (end unknown)")
    n_mid = sum(int((h.ends[:-1] == UNKNOWN_END).sum()) for h in P["hist"].values())
    n_last = sum(int(len(h.ends) > 0 and h.ends[-1] == UNKNOWN_END) for h in P["hist"].values())
    print(f"  raions with an alert running at the data end (counted as ON): {n_last}; unknown ends before the last episode: {n_mid}")
    print("  code (aaf/history.py, state_at): ((k > 0) & (last_end > times))  # unknown end = int64 max > any time -> ON")

    section("COVERAGE")
    print(f"  grid {kyiv[0]:%Y-%m-%d} -> {kyiv[-1]:%Y-%m-%d %H:%M} Kyiv; hours {len(G)}, covered {COV.mean():.3f}")
    gap = ~COV
    edges = np.flatnonzero(np.diff(gap.astype(int)))
    for i in range(0, len(edges) - 1, 2):
        a, b = G[edges[i] + 1], G[edges[i + 1]] + config.HOUR_NS
        print(f"  not covered: {pd.Timestamp(a, tz='UTC').tz_convert(config.TZ):%Y-%m-%d %H:%M} -> {pd.Timestamp(b, tz='UTC').tz_convert(config.TZ):%Y-%m-%d %H:%M} Kyiv")

    def window(a, b):
        m = (G >= ns(pd.Timestamp(a, tz="UTC"))) & (G < ns(pd.Timestamp(b, tz="UTC"))) & COV
        return F[:, m].mean(axis=1), int(m.sum())

    section("SOURCE RECONCILIATION (no overlap exists: official ends 7 Sep, live month starts 12 Sep) - adjacent weeks")
    o1, n1 = window("2026-08-22", "2026-08-29")          # last full clean official week before the outage
    o2, n2 = window("2026-09-03", "2026-09-07 05:00")    # official after the outage
    l1, n3 = window("2026-09-12 12:00", "2026-09-19 12:00")
    l2, n4 = window("2026-09-19 12:00", "2026-09-26 12:00")
    rows = [("official 22-29 Aug", o1, n1), ("official 3-7 Sep", o2, n2), ("alerts.in.ua 12-19 Sep", l1, n3), ("alerts.in.ua 19-26 Sep", l2, n4)]
    for name, v, n in rows:
        print(f"  {name:24s} hours {n:4d}  national mean share {v.mean():.3f}  median raion {np.median(v):.3f}")
    off_v, live_v = (o1 * n1 + o2 * n2) / (n1 + n2), l1
    print(f"  gap (alerts.in.ua first week - official last 2 clean windows): national {live_v.mean() - off_v.mean():+.3f} "
          f"(ratio {live_v.mean() / off_v.mean():.2f}x)")
    d = live_v - off_v
    print(f"  per raion: median difference {np.median(d):+.3f}, IQR [{np.percentile(d, 25):+.3f}, {np.percentile(d, 75):+.3f}], "
          f"raions higher in alerts.in.ua: {(d > 0).sum()}/{len(d)}, correlation of raion levels {np.corrcoef(off_v, live_v)[0, 1]:+.2f}")
    print("  note: adjacent weeks cannot separate a source difference from a real change between the weeks")

    section("REGIME SHIFT TIMING - monthly national share under alert and peak Kyiv hours")
    nat = pd.Series(np.where(COV, F.mean(axis=0), np.nan), index=kyiv)
    nat = nat[nat.index >= pd.Timestamp("2025-01-01", tz=config.TZ)]
    for mth, g in nat.groupby(nat.index.strftime("%Y-%m")):
        g = g.dropna()
        if len(g) < 24 * 5:
            continue
        hp = g.groupby(g.index.hour).mean()
        top = ", ".join(f"{h:02d}" for h in hp.sort_values(ascending=False).index[:3])
        night = hp.loc[[22, 23, 0, 1, 2, 3, 4]].mean() / hp.mean()
        morning = hp.loc[[8, 9, 10, 11]].mean() / hp.mean()
        print(f"  {mth}  share {g.mean():.3f}  top hours {top}  night/avg {night:.2f}  morning(08-11)/avg {morning:.2f}  days {len(g) / 24:.0f}")


if __name__ == "__main__":
    main()
