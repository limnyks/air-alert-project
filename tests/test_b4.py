"""B4 (D43): exponentially weighted share - hand-checkable cases."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import baselines as bl, config, leakage  # noqa: E402
from aaf.history import History  # noqa: E402

H = config.HOUR_NS
D = 24 * H


def daily(shares):
    """Each hour of day d is under alert for shares[d] of the hour (so no time-of-day pattern)."""
    s, e = [], []
    for d, x in enumerate(shares):
        if x > 0:
            st = d * D + np.arange(24, dtype=np.int64) * H
            s.append(st)
            e.append(st + int(x * H))
    return History(np.concatenate(s), np.concatenate(e))


class TestB4(unittest.TestCase):
    def test_constant_share(self):
        h = daily([0.25] * 40)
        for hl in (3, 14, 120):
            self.assertAlmostEqual(bl.ewshare(h, [40 * D], hl, since=0)[0, 0], 0.25, places=6)

    def test_between_b3_and_long_mean(self):
        h = daily([0.2] * 60 + [0.04] * 7)
        p = bl.ewshare(h, [67 * D], 14, since=0)[0, 0]
        self.assertTrue(0.04 < p < 0.2)

    def test_uncovered_hours_are_skipped(self):
        h = daily([0.5] * 10 + [0.0] * 5)            # last 5 days: no alerts, but marked as "no data"
        cov = lambda t: np.asarray(t) < 10 * D
        self.assertAlmostEqual(bl.ewshare(h, [15 * D], 7, covered=cov, since=0)[0, 0], 0.5, places=6)

    def test_flat_and_no_leakage(self):
        h = daily(np.random.default_rng(1).uniform(0, 0.6, 30))
        p = bl.ewshare(h, [20 * D], 7, since=0)
        self.assertTrue(np.allclose(p, p[0, 0]))
        f = lambda hist, t: bl.ewshare(hist, [t], 7, since=0)[0]
        self.assertEqual(leakage.check(f, {"u": h}, [10 * D + 5 * H, 20 * D]), [])

    def test_profile_flat_and_b5_equals_b4(self):
        h = daily([0.3] * 70)                                    # same share every hour -> flat profile
        r = bl.day_profile([h], 70 * D)
        self.assertTrue(np.allclose(r, 1.0))
        self.assertTrue(np.allclose(bl.ewshare_daily(h, [70 * D], 7, r, since=0), bl.ewshare(h, [70 * D], 7, since=0)))

    def test_b5_no_leakage(self):
        h = daily(np.random.default_rng(2).uniform(0, 0.6, 70))
        r = bl.day_profile([h], 60 * D)
        f = lambda hist, t: bl.ewshare_daily(hist, [t], 7, r, since=0)[0]
        self.assertEqual(leakage.check(f, {"u": h}, [61 * D + 5 * H, 65 * D]), [])


if __name__ == "__main__":
    unittest.main()
