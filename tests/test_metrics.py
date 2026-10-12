"""Metric tests against hand-computed values. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import metrics as m  # noqa: E402


class TestBinary(unittest.TestCase):
    def test_brier_and_logloss(self):
        y = np.array([1, 0, 1, 0])
        self.assertEqual(m.brier_items(y, y).mean(), 0.0)                       # perfect forecast
        self.assertAlmostEqual(m.brier_items(np.full(4, 0.5), y).mean(), 0.25)   # (0.5)^2
        self.assertAlmostEqual(m.logloss_items(np.full(4, 0.5), y).mean(), np.log(2))

    def test_reliability_perfectly_calibrated(self):
        p = np.full(10, 0.3)
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])                             # 30 % happened
        rows, ece = m.reliability(p, y)
        self.assertEqual([(b, n) for b, n, _, _ in rows], [(3, 10)])
        self.assertAlmostEqual(rows[0][2], 0.3)
        self.assertAlmostEqual(rows[0][3], 0.3)
        self.assertAlmostEqual(ece, 0.0)

    def test_confident_wrong(self):
        # confident: 0.9, 0.9, 0.1 -> wrong: the 0.9 with y = 0 -> 1/3
        self.assertAlmostEqual(m.confident_wrong([0.9, 0.9, 0.1, 0.5], [1, 0, 0, 1]), 1 / 3)

    def test_average_precision(self):
        # ranks: 0.9 (pos) precision 1/1; 0.7 (pos) precision 2/3 -> AP = (1 + 2/3) / 2
        self.assertAlmostEqual(m.average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]), (1 + 2 / 3) / 2)


class TestNextStart(unittest.TestCase):
    def test_binned(self):
        probs = np.array([[0.5, 0.25, 0.25]])
        self.assertAlmostEqual(m.binned_logscore_items(probs, np.array([2]))[0], np.log(4))   # -ln 0.25
        self.assertTrue(np.isnan(m.binned_logscore_items(probs, np.array([-1]))[0]))         # unit on: not scored
        self.assertAlmostEqual(m.within_hours_brier_items(probs, np.array([1]), 1)[0], 0.25)  # (0.5 - 1)^2


class TestDurations(unittest.TestCase):
    def setUp(self):
        self.grid = np.array([10.0, 20.0, 30.0])
        self.cdf = np.array([[0.2, 0.7, 1.0]])

    def test_known_duration(self):
        s = m.duration_scores(self.cdf, self.grid, np.array([15.0]))
        self.assertAlmostEqual(s["log"][0], np.log(2))              # mass of (10, 20] = 0.7 - 0.2 = 0.5
        self.assertAlmostEqual(s["crps"][0], 0.2**2 * 10 + 0.3**2 * 10)   # 0.4 + 0.9 = 1.3
        self.assertAlmostEqual(s["pinball_0.5"][0], 2.5)            # median 20 -> 0.5 * |15 - 20|
        self.assertEqual(s["covered_80"][0], 1.0)                   # [10, 30] contains 15

    def test_censored_duration(self):
        s = m.duration_scores(self.cdf, self.grid, np.array([15.0]), censored=np.array([True]))
        self.assertAlmostEqual(s["log"][0], -np.log(0.8))           # P(D > 10) = 1 - 0.2
        self.assertTrue(np.isnan(s["crps"][0]))


class TestAggregation(unittest.TestCase):
    def setUp(self):
        # 2 units x 4 origins (days 0, 0, 1, 1) x 1 horizon; unit A loss 1, unit B loss 3
        self.items = np.array([[[1.0]] * 4, [[3.0]] * 4])
        self.day = np.array([0, 0, 1, 1])

    def test_macro_micro(self):
        s, c = m.daily_sums(self.items, self.day, 2)
        self.assertAlmostEqual(m.macro(s, c)[0], 2.0)
        x = self.items.copy()
        x[1, 1:] = np.nan                                           # unit B: only 1 scored item
        s, c = m.daily_sums(x, self.day, 2)
        self.assertAlmostEqual(m.macro(s, c)[0], 2.0)               # each unit counts equally
        self.assertAlmostEqual(m.micro(s, c), (4 * 1 + 3) / 5)      # every item counts equally

    def test_skill(self):
        model = m.daily_sums(self.items, self.day, 2)
        ref = m.daily_sums(self.items * 2, self.day, 2)            # reference twice as bad
        self.assertAlmostEqual(m.skill(model, ref), 0.5)

    def test_bootstrap_and_claim(self):
        model = m.daily_sums(self.items, self.day, 2)
        ref = m.daily_sums(self.items * 2, self.day, 2)
        est, lo, hi = m.bootstrap(m.skill, [model, ref], n_rep=200)
        self.assertAlmostEqual(est, 0.5)
        self.assertAlmostEqual(lo, 0.5)                             # constant data: no uncertainty
        self.assertAlmostEqual(hi, 0.5)
        self.assertEqual(m.claim(lo, hi), "improves")
        self.assertEqual(m.claim(-0.1, 0.2), "no claim (interval includes 0)")


class TestD38(unittest.TestCase):
    def test_near_zero_baseline_does_not_explode(self):
        # unit A: baseline loss 1e-6, model 1e-4 (per-unit ratio would be 1 - 100 = -99)
        # unit B: baseline 0.2, model 0.1
        model = (np.array([[1e-4], [0.1]]), np.ones((2, 1)))
        ref = (np.array([[1e-6], [0.2]]), np.ones((2, 1)))
        primary = m.skill(model, ref)
        self.assertAlmostEqual(primary, 1 - (1e-4 + 0.1) / (1e-6 + 0.2), places=9)   # ~0.4997
        self.assertGreater(primary, 0.49)
        self.assertLess(np.nanmean(m.per_unit_skill(model, ref)), -40)   # the old formula: about -49

    def test_micro_skill(self):
        model = (np.array([[1.0], [3.0]]), np.array([[1.0], [3.0]]))     # unit means 1 and 1
        ref = (np.array([[2.0], [3.0]]), np.array([[1.0], [3.0]]))       # unit means 2 and 1
        self.assertAlmostEqual(m.micro_skill(model, ref), 1 - 4 / 5)
        self.assertAlmostEqual(m.skill(model, ref), 1 - 1.0 / 1.5)

    def test_joint_resampling_keeps_units_together(self):
        # two units with IDENTICAL losses that vary by day; the statistic is their difference.
        # Joint resampling (same days for all units) gives 0 in every replicate;
        # independent per-unit resampling would not.
        rng = np.random.default_rng(3)
        daily = rng.uniform(0, 1, 30)
        sums = np.vstack([daily, daily])
        cnts = np.ones_like(sums)
        diff = lambda p: float(p[0][0][0].sum() - p[0][0][1].sum())    # unit 1 minus unit 2 (sums)
        est, lo, hi = m.bootstrap(lambda *pairs: diff(pairs), [(sums, cnts)], n_rep=300)
        self.assertEqual((est, lo, hi), (0.0, 0.0, 0.0))
        est, lo, hi = m.bootstrap(lambda *pairs: diff(pairs), [(sums, cnts)], n_rep=300, block_days=7)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_per_horizon(self):
        # 1 unit, 2 origins, 2 horizons: model halves the loss at h=1, equals baseline at h=2
        model = np.array([[[0.1, 0.2], [0.1, 0.2]]])
        ref = np.array([[[0.2, 0.2], [0.2, 0.2]]])
        res = m.per_horizon_skill(model, ref, np.array([0, 1]), 2)
        self.assertAlmostEqual(res[0][1], 0.5)
        self.assertAlmostEqual(res[1][1], 0.0)


if __name__ == "__main__":
    unittest.main()
