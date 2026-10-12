"""M0-stack (D40): logistic regression combining the B1/B2/B3 forecasts with hour of day and horizon.
Fitted on validation-fold predictions only; coefficients frozen in config/m0_stack.json."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config

FILE = config.ROOT / "config" / "m0_stack.json"
COLUMNS = ["const", "logit_B1", "logit_B2", "logit_B3", "logit_B2_x_h", "sin_hour", "cos_hour", "h"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(probs, origins):
    """Rows for every (unit, origin, horizon): shape (U*O*24, len(COLUMNS))."""
    U, O, K = probs["B1"].shape
    hh = np.broadcast_to((np.arange(1, K + 1) / 24.0)[None, None, :], (U, O, K))
    lt = np.asarray(origins, dtype=np.int64)[:, None] + np.arange(1, K + 1)[None, :] * config.HOUR_NS
    hour = pd.DatetimeIndex(lt.ravel().astype("datetime64[ns]")).tz_localize("UTC").tz_convert(config.TZ).hour.to_numpy()
    ang = np.broadcast_to((2 * np.pi * hour.reshape(O, K) / 24)[None, :, :], (U, O, K))
    l1, l2, l3 = _logit(probs["B1"]), _logit(probs["B2"]), _logit(probs["B3"])
    cols = [np.ones((U, O, K)), l1, l2, l3, l2 * hh, np.sin(ang), np.cos(ang), hh]
    return np.stack([c.ravel() for c in cols], axis=1)


def fit(probs, y, origins, stride=3):
    X = design(probs, origins)
    yy = np.asarray(y, float).ravel()
    keep = np.arange(len(yy)) % stride == 0                       # subsample rows for speed
    res = sm.GLM(yy[keep], X[keep], family=sm.families.Binomial()).fit()
    return dict(zip(COLUMNS, res.params.tolist()))


def save(coef, meta):
    FILE.write_text(json.dumps({"coefficients": coef, **meta}, indent=2))


def load():
    return json.loads(FILE.read_text())["coefficients"] if FILE.exists() else None


def predict(coef, probs, origins):
    U, O, K = probs["B1"].shape
    beta = np.array([coef[c] for c in COLUMNS])
    z = design(probs, origins) @ beta
    return (1 / (1 + np.exp(-z))).reshape(U, O, K)
