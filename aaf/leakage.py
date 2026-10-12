"""Leakage test (protocol 12.7): a forecaster's output at time t must not change when
(a) the history is cut at t, or (b) fake alerts are added after t.

A forecaster is any function f(history, t_ns) -> numpy array (probabilities or inputs).
"""
import numpy as np

from . import config
from .history import History


def with_fake_future(hist, t, rng, n=20):
    """Keep what is observable at t, then append n fake alerts after t (and after any running alert)."""
    obs = hist.observable(t)
    start = int(max(t, obs.ends[-1] if len(obs.ends) and obs.ends[-1] != np.iinfo(np.int64).max else t)) + config.HOUR_NS
    if len(obs.ends) and obs.ends[-1] == np.iinfo(np.int64).max:
        ends = obs.ends.copy()
        ends[-1] = start - config.MIN_NS            # the running alert "ends" in the fake future
    else:
        ends = obs.ends
    gaps = rng.integers(10, 300, size=n) * config.MIN_NS
    durs = rng.integers(5, 120, size=n) * config.MIN_NS
    s = start + np.cumsum(gaps + np.concatenate([[0], durs[:-1]]))
    return History(np.concatenate([obs.starts, s]).astype(np.int64), np.concatenate([ends, s + durs]).astype(np.int64))


def check(forecaster, histories, times, seed=0, atol=1e-12):
    """Returns a list of (unit, t, problem) for every leak found; empty list = no leakage detected."""
    rng = np.random.default_rng(seed)
    problems = []
    for unit, hist in histories.items():
        for t in times:
            full = np.asarray(forecaster(hist, t), dtype=float)
            cut = np.asarray(forecaster(hist.observable(t), t), dtype=float)
            fake = np.asarray(forecaster(with_fake_future(hist, t, rng), t), dtype=float)
            if not np.allclose(full, cut, atol=atol, equal_nan=True):
                problems.append((unit, int(t), "differs between full data and data cut at t"))
            if not np.allclose(full, fake, atol=atol, equal_nan=True):
                problems.append((unit, int(t), "changes when fake alerts are added after t"))
    return problems
