"""Alert history of one unit, and the leakage-safe view of it at a forecast time t.

Episodes are non-overlapping [start, end) intervals in int64 UTC nanoseconds. An end is "known at t"
only if end <= t; an alert still running at t has an unknown end at t.

Models and baselines must use only:
  - History.observable(t)  -> a new History with what was known at time t
  - History.state_at(times) for times <= the forecast time (safe: compares ends with the asked time)
Targets (aaf.targets) use the full history and are protected by a guard.
"""
from dataclasses import dataclass

import numpy as np

UNKNOWN_END = np.iinfo(np.int64).max   # end not known at the time of observation


@dataclass(frozen=True)
class History:
    starts: np.ndarray   # int64 ns, sorted
    ends: np.ndarray     # int64 ns; UNKNOWN_END if not known

    def __post_init__(self):
        assert self.starts.dtype == np.int64 and self.ends.dtype == np.int64
        assert np.all(np.diff(self.starts) > 0), "starts must be strictly increasing"
        assert np.all(self.ends > self.starts), "every episode must end after it starts"
        assert np.all(self.starts[1:] >= self.ends[:-1]), "episodes must not overlap"

    def observable(self, t):
        """What was known at time t: episodes that started before t; ends after t become unknown."""
        k = int(np.searchsorted(self.starts, t, side="left"))
        ends = self.ends[:k].copy()
        ends[ends > t] = UNKNOWN_END
        return History(self.starts[:k].copy(), ends)

    def state_at(self, times):
        """1 if under alert at each time (a running alert with unknown end counts as on)."""
        times = np.asarray(times, dtype=np.int64)
        k = np.searchsorted(self.starts, times, side="right")       # starts <= time
        last_end = np.where(k > 0, self.ends[np.clip(k - 1, 0, None)], np.iinfo(np.int64).min)
        return ((k > 0) & (last_end > times)).astype(np.int8)

    def starts_in(self, a, b):
        """Starts in [a, b)."""
        return self.starts[np.searchsorted(self.starts, a, "left"):np.searchsorted(self.starts, b, "left")]
