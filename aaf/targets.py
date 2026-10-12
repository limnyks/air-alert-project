"""Forecast origins, targets (protocol 12.2) and the guard that protects the test windows (12.1)."""
import numpy as np
import pandas as pd

from . import config
from .data import ns


class ProtectedWindowError(RuntimeError):
    pass


def guard(times_ns, final_test=False):
    """Refuse to read labels inside the test windows unless in final-test mode."""
    if final_test:
        return
    t = np.asarray(times_ns, dtype=np.int64)
    for a, b in config.PROTECTED:
        if np.any((t >= ns(a)) & (t < ns(b))):
            raise ProtectedWindowError(f"label times inside protected window {a} -> {b}; final-test mode required")


def fold_origins(fold_start, fold_end, horizon_hours=24):
    """Full UTC hours in a fold, stopping early enough that every +24 h label stays inside the fold."""
    last = pd.Timestamp(fold_end) - pd.Timedelta(hours=horizon_hours)   # labels end before fold_end
    times = pd.date_range(pd.Timestamp(fold_start).ceil("h"), last, freq="h", inclusive="left")
    return times.tz_convert("UTC").tz_localize(None).as_unit("ns").asi8


def label_times(origins_ns, horizons=config.HORIZONS):
    return np.asarray(origins_ns, dtype=np.int64)[:, None] + np.asarray(horizons, dtype=np.int64)[None, :] * config.HOUR_NS


def map_target(hist, origins_ns, horizons=config.HORIZONS, final_test=False):
    """Primary target: alert state at t+h. Shape (origins, horizons)."""
    lt = label_times(origins_ns, horizons)
    guard(lt, final_test)
    return hist.state_at(lt.ravel()).reshape(lt.shape)


def any_in_hour_target(hist, origins_ns, horizons=config.HORIZONS, final_test=False):
    """Secondary target: any alert during (t+h-1h, t+h]."""
    lt = label_times(origins_ns, horizons)
    guard(lt - config.HOUR_NS, final_test)
    guard(lt, final_test)
    on_before = hist.state_at((lt - config.HOUR_NS).ravel()).reshape(lt.shape)
    k1 = np.searchsorted(hist.starts, (lt - config.HOUR_NS).ravel(), side="right")
    k2 = np.searchsorted(hist.starts, lt.ravel(), side="right")
    return ((on_before == 1) | ((k2 - k1).reshape(lt.shape) > 0)).astype(np.int8)


def next_start_target(hist, origins_ns, max_hours=24, final_test=False):
    """Hours bin (1..24) of the next start after t, 25 = later; -1 where the unit is on at t."""
    t = np.asarray(origins_ns, dtype=np.int64)
    guard(t + max_hours * config.HOUR_NS, final_test)
    on = hist.state_at(t)
    k = np.searchsorted(hist.starts, t, side="right")
    nxt = np.where(k < len(hist.starts), hist.starts[np.clip(k, 0, len(hist.starts) - 1)], np.iinfo(np.int64).max)
    wait_h = np.ceil((nxt - t) / config.HOUR_NS)
    bins = np.where(wait_h <= max_hours, wait_h, max_hours + 1).astype(int)
    return np.where(on == 1, -1, bins)
