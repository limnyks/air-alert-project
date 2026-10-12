"""Load modeled units and their episodes from the Step 10b outputs."""
import numpy as np
import pandas as pd

from . import config
from .history import UNKNOWN_END, History


def load_units(clean=True):
    """Modeled units: the D36-clean list if it exists (written by step12a), else the Step 10b list."""
    if clean and config.UNITS_CLEAN.exists():
        u = pd.read_csv(config.UNITS_CLEAN)
    else:
        u = pd.read_csv(config.UNIT_CHECKS)
    return u[~u["excluded"]].reset_index(drop=True)


def load_histories(source, units=None):
    """{unit: History} for source 'official' or 'alerts_in_ua'. Censored ends -> UNKNOWN_END."""
    ep = pd.read_csv(config.EPISODES)
    ep = ep[ep["source"] == source]
    if units is not None:
        ep = ep[ep["unit"].isin(set(units))]
    start = pd.to_datetime(ep["start_utc"], utc=True, format="ISO8601")
    end = pd.to_datetime(ep["end_utc"], utc=True, format="ISO8601")
    # data window ends at the snapshot time for all units (requests ran ~13 min longer, M10)
    after_cutoff = (start >= config.SNAPSHOT).to_numpy()
    # zero-length records (start == end) cannot be "under alert" at any moment
    zero = ((end <= start) & (ep["end_type"] == "known")).to_numpy()
    if after_cutoff.any() or zero.any():
        print(f"[load_histories {source}] dropped {int(after_cutoff.sum())} episodes starting after the snapshot cutoff, "
              f"{int(zero.sum())} zero-length episodes")
    ep = ep[~after_cutoff & ~zero]
    out = {}
    for unit, g in ep.groupby("unit", sort=False):
        s = pd.to_datetime(g["start_utc"], utc=True, format="ISO8601").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")
        e = pd.to_datetime(g["end_utc"], utc=True, format="ISO8601").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")
        e = np.where(g["end_type"].to_numpy() == "censored", UNKNOWN_END, e)
        order = np.argsort(s)
        out[unit] = History(s[order], e[order])
    return out


def ns(ts):
    return pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).as_unit("ns").value
