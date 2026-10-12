"""Tests for the evaluation harness (aaf). Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import config, leakage  # noqa: E402
from aaf.data import ns  # noqa: E402
from aaf.history import UNKNOWN_END, History  # noqa: E402
from aaf.selection import share  # noqa: E402
from aaf.targets import (ProtectedWindowError, any_in_hour_target, fold_origins, guard,  # noqa: E402
                         map_target, next_start_target)

H = config.HOUR_NS
M = config.MIN_NS
T0 = ns(pd.Timestamp("2026-05-01", tz="UTC"))   # a day well before validation


def hist(*pairs):
    """History from (start_hour, end_hour) pairs relative to T0; end None = unknown."""
    s = np.array([T0 + int(a * H) for a, _ in pairs], dtype=np.int64)
    e = np.array([UNKNOWN_END if b is None else T0 + int(b * H) for _, b in pairs], dtype=np.int64)
    return History(s, e)


class TestHistory(unittest.TestCase):
    def test_observable_hides_future_and_running_ends(self):
        h = hist((1, 2), (5, 7), (10, 11))
        o = h.observable(T0 + 6 * H)                       # at hour 6: 2nd alert running
        self.assertEqual(len(o.starts), 2)                 # 3rd alert (hour 10) not visible
        self.assertEqual(o.ends[0], T0 + 2 * H)            # finished alert: end known
        self.assertEqual(o.ends[1], UNKNOWN_END)           # running alert: end unknown

    def test_state_at(self):
        h = hist((1, 2), (5, 7))
        t = T0 + np.array([0, 1, 1.5, 2, 6, 7, 8]) * H
        np.testing.assert_array_equal(h.state_at(t.astype(np.int64)), [0, 1, 1, 0, 1, 0, 0])

    def test_rejects_overlaps(self):
        with self.assertRaises(AssertionError):
            hist((1, 3), (2, 4))


class TestTargets(unittest.TestCase):
    def test_map_any_next(self):
        h = hist((2.5, 2.6), (5, 8))
        o = np.array([T0], dtype=np.int64)
        np.testing.assert_array_equal(map_target(h, o, (1, 2, 3, 5, 8))[0], [0, 0, 0, 1, 0])
        # hour (2,3] contains the short alert 2.5-2.6 although the state at +3 h is off
        np.testing.assert_array_equal(any_in_hour_target(h, o, (1, 2, 3, 4))[0], [0, 0, 1, 0])
        self.assertEqual(next_start_target(h, o)[0], 3)        # next start at +2.5 h -> bin 3
        self.assertEqual(next_start_target(h, o + 6 * H)[0], -1)  # on at hour 6

    def test_guard_blocks_test_windows(self):
        with self.assertRaises(ProtectedWindowError):
            guard([ns(pd.Timestamp("2026-08-01", tz="UTC"))])       # pre-reform test
        with self.assertRaises(ProtectedWindowError):
            guard([ns(pd.Timestamp("2026-09-25", tz="UTC"))])       # post-reform test
        guard([ns(pd.Timestamp("2026-08-01", tz="UTC"))], final_test=True)   # allowed in final mode
        guard([ns(pd.Timestamp("2026-06-01", tz="UTC"))])                   # validation: allowed

    def test_last_fold_labels_stay_out_of_test(self):
        a, b = config.FOLDS[-1]
        o = fold_origins(a, b)
        h = hist((0, 1))
        map_target(h, o)                                   # would raise if any label touched the test
        self.assertLess(o.max() + 24 * H, ns(b) + 1)


class TestShare(unittest.TestCase):
    def test_exact_share(self):
        h = hist((1, 3), (5, None))
        self.assertAlmostEqual(share(h, T0, T0 + 10 * H), (2 + 5) / 10)   # unknown end counts to b


class TestLeakage(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        starts = np.cumsum(rng.integers(30, 400, 200)) * M + T0
        ends = starts + rng.integers(5, 25, 200) * M
        self.h = {"u": History(starts.astype(np.int64), ends.astype(np.int64))}
        self.times = [int(starts[i] + 3 * M) for i in (20, 60, 120)]       # some inside running alerts
        self.times += [int(ends[i] + 2 * M) for i in (30, 90)]             # some between alerts

    def test_honest_forecaster_passes(self):
        def honest(h, t):   # 7-day share and current state, using only the past
            o = h.observable(t)
            return np.array([share(o, t - 7 * 24 * H, t), o.state_at([t])[0], len(o.starts_in(t - 24 * H, t))])
        self.assertEqual(leakage.check(honest, self.h, self.times), [])

    def test_peeking_at_future_state_is_caught(self):
        def leaky(h, t):
            return h.state_at([t + H])                     # looks one hour ahead
        self.assertNotEqual(leakage.check(leaky, self.h, self.times), [])

    def test_using_running_alert_end_is_caught(self):
        def leaky(h, t):   # remaining duration of the running alert: needs its end, unknown at t
            k = np.searchsorted(h.starts, t, side="right") - 1
            return np.array([min(h.ends[k] - t, 10**15) if k >= 0 else 0])
        self.assertNotEqual(leakage.check(leaky, self.h, self.times), [])


if __name__ == "__main__":
    unittest.main()
