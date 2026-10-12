"""Baselines (protocol 12.4) for the map target P(on at t+h), and an empirical duration baseline.

All use only: on-time accumulated before t (does not depend on unknown ends), the state at t, and
statistics from data before the fold cut. Verified by aaf.leakage in step12c.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

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


def ewshare(hist, origins, half_life_days, covered=None, since=None):
    """B4 (D43): exponentially weighted share of time under alert before each origin, (origins, 24), flat.
    p(t) = sum(w * on) / sum(w * covered) over past hours, w = 0.5 ** (age / half-life); hours without data
    (covered False) are left out of both sums. Origins must lie on one hourly grid (e.g. all full hours).
    since: earliest time with data (ns); default = first alert start."""
    o = np.asarray(origins, dtype=np.int64)
    end = int(o.max())
    if (end - o).min() < 0 or ((end - o) % H).any():
        raise ValueError("origins must lie on one hourly grid")
    first = since if since is not None else (int(hist.starts[0]) if len(hist.starts) else end - H)
    n = max(int(np.ceil((end - first) / H)), 1)
    g = end - H * np.arange(n, 0, -1, dtype=np.int64)              # hourly bins [g, g + H); the last ends at `end`
    frac = (ontime_upto(hist, g + H) - ontime_upto(hist, g)) / H
    cov = (covered(g) & covered(g + H - 1)) if covered else np.ones(n, bool)
    d = 0.5 ** (1.0 / (24.0 * half_life_days))
    num = lfilter([1.0], [1.0, -d], np.where(cov, frac, 0.0))
    den = lfilter([1.0], [1.0, -d], cov.astype(float))
    k = n - 1 - (end - o) // H                                      # bin that ends exactly at each origin
    p = np.full(len(o), np.nan)
    ok = (k >= 0) & (den[np.clip(k, 0, None)] > 0)
    p[ok] = num[k[ok]] / den[k[ok]]
    return np.repeat(p[:, None], len(config.HORIZONS), axis=1)


def hour_of_day(times_ns):
    t = pd.DatetimeIndex(np.asarray(times_ns, dtype="datetime64[ns]")).tz_localize("UTC").tz_convert(config.TZ)
    return t.hour.to_numpy()


def day_profile(hists, cut, covered=None):
    """D45: pooled hour-of-day profile r[24] from the covered hours of the 8 weeks before `cut`:
    r[k] = sum over units of the share of time under alert in Kyiv hour k / sum over units of the overall share.
    hists: iterable of Histories (pass .observable(cut) or full histories; only time before cut is read)."""
    cov = covered or (lambda t: np.ones(len(t), bool))
    g = np.arange(cut - WEEKS8, cut, H, dtype=np.int64)
    g = g[cov(g) & cov(g + H - 1)]
    hod = hour_of_day(g)
    num, den = np.zeros(24), 0.0
    for h in hists:
        frac = (ontime_upto(h, g + H) - ontime_upto(h, g)) / H
        num += np.array([frac[hod == k].mean() if (hod == k).any() else np.nan for k in range(24)])
        den += frac.mean() if len(frac) else 0.0
    r = num / den if den > 0 else np.ones(24)
    r = np.where(np.isfinite(r), r, 1.0)
    return r / r.mean()


def ewshare_daily(hist, origins, half_life_days, profile, covered=None, since=None):
    """B5 (D45): B4 level times the daily profile at each label time, capped at 1. (origins, 24)."""
    base = ewshare(hist, origins, half_life_days, covered, since)[:, 0]
    lt = np.asarray(origins, dtype=np.int64)[:, None] + np.asarray(config.HORIZONS, dtype=np.int64)[None, :] * H
    return np.minimum(1.0, base[:, None] * profile[hour_of_day(lt.ravel()).reshape(lt.shape)])


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logit_hour_profile(hists, a, b, k_shrink=7.0, covered=None):
    """D46: s[24] on the logit scale from the hours in [a, b) (ns), pooled over `hists`:
    s[k] = w * (logit(pooled share in Kyiv hour k) - logit(pooled overall share)), w = n / (n + k_shrink),
    n = days of data used. Only time before b is read. Returns zeros if there is no data."""
    cov = covered or (lambda t: np.ones(len(t), bool))
    g = np.arange(a, b - H + 1, H, dtype=np.int64)
    g = g[cov(g) & cov(g + H - 1)]
    if len(g) == 0:
        return np.zeros(24)
    frac = np.mean([(ontime_upto(h, g + H) - ontime_upto(h, g)) / H for h in hists], axis=0)
    hod = hour_of_day(g)
    if frac.mean() <= 0 or any(not (hod == k).any() for k in range(24)):
        return np.zeros(24)
    raw = np.array([_logit(frac[hod == k].mean()) for k in range(24)]) - _logit(frac.mean())
    n = len(g) / 24.0
    return raw * n / (n + k_shrink)


def ewshare_logit(hist, origins, half_life_days, profile, covered=None, since=None):
    """B6 (D46): sigmoid(logit(B4 level) + profile[Kyiv hour of the label time]). (origins, 24)."""
    base = ewshare(hist, origins, half_life_days, covered, since)[:, 0]
    lt = np.asarray(origins, dtype=np.int64)[:, None] + np.asarray(config.HORIZONS, dtype=np.int64)[None, :] * H
    z = _logit(base)[:, None] + np.asarray(profile)[hour_of_day(lt.ravel()).reshape(lt.shape)]
    return 1.0 / (1.0 + np.exp(-z))


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
