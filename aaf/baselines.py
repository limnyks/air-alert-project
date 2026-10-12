"""Baselines (protocol 12.4) for the map target P(on at t+h), and an empirical duration baseline.

All use only: on-time accumulated before t (does not depend on unknown ends), the state at t, and
statistics from data before the fold cut. Verified by aaf.leakage in step12c.
"""
import numpy as np
import pandas as pd

from . import config
from .history import UNKNOWN_END

H = config.HOUR_NS
DAY = 24 * H
WEEKS8 = 56 * DAY


def ontime_upto(hist, x):
    """Total time under alert in (-inf, x) for each x (int64 ns). Uses ends only as min(end, x)."""
    x = np.asarray(x, dtype=np.int64)
    dur = np.where(hist.ends == UNKNOWN_END, 0, hist.ends - hist.starts).astype(float)
    cum = np.concatenate([[0.0], np.cumsum(dur)])                 # completed time of episodes < k
    k = np.searchsorted(hist.starts, x, side="left")              # episodes starting before x
    last = np.clip(k - 1, 0, max(len(hist.starts) - 1, 0))
    if len(hist.starts) == 0:
        return np.zeros(len(x))
    part = np.minimum(hist.ends[last], x) - hist.starts[last]     # the last started episode, cut at x
    return np.where(k > 0, cum[np.maximum(k - 1, 0)] + part, 0.0)


def hour_of_week(times_ns):
    t = pd.DatetimeIndex(np.asarray(times_ns, dtype="datetime64[ns]")).tz_localize("UTC").tz_convert(config.TZ)
    return (t.dayofweek * 24 + t.hour).to_numpy()


def fit(hist, cut, covered=None):
    """Statistics from the 8 weeks before `cut` (ns): B1 climatology table and B2 transition table.
    covered(times) -> bool array: times with data (outage / source gaps excluded); None = all covered."""
    cov = covered or (lambda t: np.ones(len(t), bool))
    g = np.arange(cut - WEEKS8, cut, H, dtype=np.int64)
    g = g[cov(g) & cov(g + H - 1)]
    frac = (ontime_upto(hist, g + H) - ontime_upto(hist, g)) / H
    how = hour_of_week(g)
    clim = np.full(168, frac.mean() if len(frac) else 0.0)
    for w in range(168):
        if (how == w).any():
            clim[w] = frac[how == w].mean()
    g2 = g[g + 24 * H < cut]
    s0 = hist.state_at(g2)
    trans = np.zeros((2, 24))
    for j, h in enumerate(config.HORIZONS):
        ok = cov(g2 + h * H)
        sh = hist.state_at(g2 + h * H)
        for s in (0, 1):
            m = (s0 == s) & ok
            trans[s, j] = (sh[m].sum() + 1) / (m.sum() + 2)        # Laplace smoothing
    return {"clim": clim, "trans": trans}


def predict(name, params, hist, origins):
    """(origins, 24) probabilities for baseline `name` in {B1, B2, B3}."""
    o = np.asarray(origins, dtype=np.int64)
    lt = o[:, None] + np.asarray(config.HORIZONS, dtype=np.int64)[None, :] * H
    if name == "B1":
        return params["clim"][hour_of_week(lt.ravel())].reshape(lt.shape)
    if name == "B2":
        return params["trans"][hist.state_at(o)]
    if name == "B3":
        share7 = (ontime_upto(hist, o) - ontime_upto(hist, o - 7 * DAY)) / (7 * DAY)
        return np.repeat(share7[:, None], len(config.HORIZONS), axis=1)
    raise ValueError(name)


DUR_GRID = np.arange(5.0, 24 * 60 + 5, 5.0)                       # minutes; last point = 24 h


def duration_cdf(hist, cut, since):
    """Empirical CDF of durations of alerts that started and ended in [since, cut)."""
    m = (hist.starts >= since) & (hist.ends <= cut)
    d = (hist.ends[m] - hist.starts[m]) / config.MIN_NS
    if len(d) == 0:
        return np.linspace(0, 1, len(DUR_GRID))
    cdf = (d[None, :] <= DUR_GRID[:, None]).mean(axis=1)
    cdf[-1] = 1.0                                                  # everything beyond 24 h in the last bin
    return np.maximum.accumulate(np.clip(cdf, 1e-4, 1.0))
