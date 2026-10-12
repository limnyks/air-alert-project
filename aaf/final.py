"""Final model (hour-by-hour, +1...+24 h): data panel and model pieces.

logit p(u, t+h) = logit(smoothed level of u at t) + s[Kyiv hour of t+h] + w[h] * alert_now(u, t)

Panel: one hourly grid in UTC from the official start to the live data end. Per unit and hour bin:
share of the bin under alert (F), alert state at the start of the bin (S); per bin: covered or not (COV:
official record minus the outage, then the live alerts.in.ua month; the gap between them is not covered).
Hours are converted to Kyiv time with the tz database (DST handled by pandas/zoneinfo).
"""
import json

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config
from . import evaluate as ev
from .baselines import ontime_upto
from .data import load_histories, load_units, ns
from .history import UNKNOWN_END, History

H = config.HOUR_NS
DAY = 24 * H
SNAP = "20261012T120011Z"
PRIOR_HOURS = 24.0           # level smoothing: pseudo-hours at the national level (fixed, not tuned)


def combined(off, live, since):
    """Official episodes (clipped at the official end) + live episodes from `since` on."""
    e_off = ns(config.OFFICIAL_END)
    k = off.starts < e_off
    s1, e1 = off.starts[k], np.minimum(off.ends[k], e_off)
    keep = live.ends > since
    s2, e2 = np.maximum(live.starts[keep], since), live.ends[keep]
    return History(np.concatenate([s1, s2]).astype(np.int64), np.concatenate([e1, e2]).astype(np.int64))


def build_panel():
    import sys
    sys.path.insert(0, str(config.ROOT / "scripts"))
    from step19_b6_postreform import live_histories          # step17 rules for the live snapshot
    from step14_live_forecast import LIVE

    T = int(pd.to_datetime(SNAP, format="%Y%m%dT%H%M%SZ").as_unit("ns").value)
    since = T - 30 * DAY
    units = load_units()["unit"].tolist()
    off = load_histories("official", units)
    live = live_histories(LIVE / SNAP, T)
    hist = {u: combined(off[u], live[u], since) for u in units}

    g0 = (ns(config.OFFICIAL_START) // H + 1) * H
    G = np.arange(g0, (T // H) * H, H, dtype=np.int64)          # bins [G, G + H), all ending <= T
    off_cov = ev.covered_fn(ev.official_segments())
    cov = lambda t: off_cov(t) | ((np.asarray(t) >= since) & (np.asarray(t) < T))  # noqa: E731
    COV = cov(G) & cov(G + H - 1)
    F = np.stack([(ontime_upto(hist[u], G + H) - ontime_upto(hist[u], G)) / H for u in units])
    S = np.stack([hist[u].state_at(G) for u in units]).astype(np.int8)
    kyiv = pd.DatetimeIndex(G.astype("datetime64[ns]")).tz_localize("UTC").tz_convert(config.TZ)
    return {"T": T, "since": since, "units": units, "hist": hist, "G": G, "COV": COV, "F": F, "S": S,
            "hod": kyiv.hour.to_numpy(), "kyiv": kyiv}


def level_parts(P, half_life_days):
    """num, den (units, bins): exponentially weighted on-time and covered time over bins < j, at origin G[j]."""
    d = 0.5 ** (1.0 / (24.0 * half_life_days))
    x = np.where(P["COV"][None, :], P["F"], 0.0)
    num = lfilter([1.0], [1.0, -d], x, axis=1)
    den = lfilter([1.0], [1.0, -d], np.broadcast_to(P["COV"].astype(float), x.shape), axis=1)
    pad = np.zeros((x.shape[0], 1))
    return np.hstack([pad, num[:, :-1]]), np.hstack([pad, den[:, :-1]])   # shift: origin j sees bins < j


def smoothed_level(num, den, prior=PRIOR_HOURS):
    """(num + a*m) / (den + a), m = national mean of the raw levels at the same origin (data before it only)."""
    raw = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    m = np.nanmean(raw, axis=0)
    m = np.where(np.isfinite(m), m, 0.2)
    return (num + prior * m[None, :]) / (den + prior)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _loglik(off, k, h, a, y, wt, beta, ridge):
    z = np.clip(off + beta[k] + a * beta[24 + h], -35.0, 35.0)
    return float(np.sum(wt * (y * z - np.logaddexp(0.0, z)))) - 0.5 * ridge * float(beta @ beta)


def fit_logistic(off, k, h, a, y, wt, beta=None, iters=50, ridge=1e-3):
    """Weighted logistic regression with a fixed offset and 48 coefficients: s[k] (24 Kyiv hours of the
    target time) and w[h] (24 horizons, times alert_now). Damped Newton steps (a step is halved until the
    penalised log-likelihood does not decrease); the Hessian is built with bincounts (rows have one hour
    dummy and at most one horizon dummy). Returns beta (48,)."""
    beta = np.zeros(48) if beta is None else beta.copy()
    ll = _loglik(off, k, h, a, y, wt, beta, ridge)
    for _ in range(iters):
        p = sigmoid(off + beta[k] + a * beta[24 + h])
        r = wt * (y - p)
        v = wt * p * (1 - p)
        g = np.concatenate([np.bincount(k, r, 24), np.bincount(h, r * a, 24)]) - ridge * beta
        Hm = np.zeros((48, 48))
        Hm[np.arange(24), np.arange(24)] = np.bincount(k, v, 24)
        Hm[24 + np.arange(24), 24 + np.arange(24)] = np.bincount(h, v * a, 24)
        cross = np.bincount(k * 24 + h, v * a, 576).reshape(24, 24)
        Hm[:24, 24:] = cross
        Hm[24:, :24] = cross.T
        Hm += ridge * np.eye(48)
        step = np.linalg.solve(Hm, g)
        t, new = 1.0, ll
        while t > 1e-4:
            new = _loglik(off, k, h, a, y, wt, beta + t * step, ridge)
            if new >= ll - 1e-9:
                break
            t *= 0.5
        beta += t * step
        done = abs(new - ll) < 1e-10 * max(1.0, abs(ll)) or np.abs(t * step).max() < 1e-6
        ll = new
        if done:
            break
    return beta


def predict(lvl_logit, hods_target, alert_now, beta):
    """lvl_logit (n,), hods_target (n, 24), alert_now (n,) -> (n, 24) probabilities."""
    return sigmoid(lvl_logit[:, None] + beta[hods_target] + alert_now[:, None] * beta[24:][None, :])


def load_beta(path):
    return np.array(json.loads(path.read_text())["beta"])
