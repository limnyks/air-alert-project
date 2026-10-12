"""D36 fix 1: unit exclusion (80 % rule, D26-D) and training starts (x1.5 switch rule, D26-B)
recomputed with official data before the validation start only."""
import numpy as np
import pandas as pd

from . import config
from .data import ns
from .history import UNKNOWN_END

SAT_THRESHOLD = 0.80
RATIO_RULE = 1.5
LAST_DAYS = 92
BEFORE_SWITCH_DAYS = 365
DAY_NS = 24 * config.HOUR_NS


def share(hist, a, b):
    """Exact share of time under alert in [a, b) from interval overlaps (unknown ends count up to b)."""
    if b <= a:
        return np.nan
    ends = np.where(hist.ends == UNKNOWN_END, b, hist.ends)
    overlap = np.clip(np.minimum(ends, b) - np.maximum(hist.starts, a), 0, None)
    return overlap.sum() / (b - a)


def recompute(histories, checks):
    """checks: Step 10b per-unit table (unit, oblast, switch, excluded, train_from, ...)."""
    v = ns(config.VAL_START)
    t0 = ns(config.OFFICIAL_START)
    rows = []
    for _, c in checks.iterrows():
        h = histories[c["unit"]].observable(v)            # nothing after the validation start
        last = share(h, v - LAST_DAYS * DAY_NS, v)
        if pd.isna(c["switch"]):
            before = after = ratio = np.nan
            train_from = config.OFFICIAL_START.date()
        else:
            sw = ns(pd.Timestamp(c["switch"], tz="UTC"))
            before = share(h, max(sw - BEFORE_SWITCH_DAYS * DAY_NS, t0), sw)
            after = share(h, sw, v)
            ratio = after / before if before else np.nan
            shift = (not np.isnan(ratio)) and (ratio > RATIO_RULE or ratio < 1 / RATIO_RULE)
            train_from = pd.Timestamp(c["switch"]).date() if shift else config.OFFICIAL_START.date()
        rows.append({"unit": c["unit"], "oblast": c["oblast"], "switch": c["switch"],
                     "share_last92_preval": last, "share_before": before, "share_after_preval": after,
                     "ratio_preval": ratio, "train_from": str(train_from),
                     "excluded": bool(last > SAT_THRESHOLD),
                     "excluded_step10b": bool(c["excluded"]), "train_from_step10b": c["train_from"]})
    return pd.DataFrame(rows)
