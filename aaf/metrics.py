"""Metrics, aggregation and day-block bootstrap (protocol 12.3).

Convention: per-item losses are arrays of shape (units, origins, ...) with NaN for items that are
not scored (masked origins, unit on/off conditions). Lower loss = better, except skill scores.
"""
import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------- binary forecasts (map targets)

def brier_items(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return (p - y) ** 2


def logloss_items(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def reliability(p, y, bins=10):
    """Rows: bin, n, mean forecast, observed frequency; plus ECE (report only)."""
    p, y = np.asarray(p, float).ravel(), np.asarray(y, float).ravel()
    ok = ~(np.isnan(p) | np.isnan(y))
    p, y = p[ok], y[ok]
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append((b, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    ece = sum(n * abs(f - o) for _, n, f, o in rows) / max(len(p), 1)
    return rows, ece


def confident_wrong(p, y, lo=0.2, hi=0.8):
    """Among forecasts <= lo or >= hi: share that were wrong (report only)."""
    p, y = np.asarray(p, float).ravel(), np.asarray(y, float).ravel()
    ok = ~(np.isnan(p) | np.isnan(y))
    p, y = p[ok], y[ok]
    conf = (p <= lo) | (p >= hi)
    wrong = ((p >= hi) & (y == 0)) | ((p <= lo) & (y == 1))
    return float(wrong[conf].mean()) if conf.any() else np.nan


def average_precision(p, y):
    """PR-AUC as average precision (report only)."""
    p, y = np.asarray(p, float).ravel(), np.asarray(y, float).ravel()
    ok = ~(np.isnan(p) | np.isnan(y))
    p, y = p[ok], y[ok]
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-p, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


# ---------------------------------------------------------------- next start (binned distribution)

def binned_logscore_items(probs, k):
    """probs: (..., K) distribution over bins 1..K; k: observed bin (1..K), -1 = not scored."""
    probs, k = np.asarray(probs, float), np.asarray(k)
    p = np.take_along_axis(probs, np.clip(k - 1, 0, probs.shape[-1] - 1)[..., None], axis=-1)[..., 0]
    return np.where(k >= 1, -np.log(np.clip(p, EPS, 1)), np.nan)


def within_hours_brier_items(probs, k, hours):
    """Brier of P(next start within `hours`) = sum of bins 1..hours."""
    probs, k = np.asarray(probs, float), np.asarray(k)
    p = probs[..., :hours].sum(axis=-1)
    y = ((k >= 1) & (k <= hours)).astype(float)
    return np.where(k >= 1, (p - y) ** 2, np.nan)


# ---------------------------------------------------------------- durations (distribution on a minute grid)

def duration_scores(cdf, grid, d, censored=None, quantiles=(0.1, 0.5, 0.9)):
    """cdf: (n, G) predicted CDF at grid minutes (increasing, ending at 1 or the tail mass);
    d: observed (or censored-at) durations in minutes. Returns dict of per-item arrays."""
    cdf, grid, d = np.asarray(cdf, float), np.asarray(grid, float), np.asarray(d, float)
    censored = np.zeros(len(d), bool) if censored is None else np.asarray(censored, bool)
    j = np.clip(np.searchsorted(grid, d, side="left"), 0, len(grid) - 1)       # first grid point >= d
    prev = np.where(j > 0, cdf[np.arange(len(d)), np.maximum(j - 1, 0)], 0.0)
    mass = cdf[np.arange(len(d)), j] - prev                                     # P(d in (grid[j-1], grid[j]])
    surv = 1.0 - prev                                                           # P(D > grid[j-1]) for censored
    log = -np.log(np.clip(np.where(censored, surv, mass), EPS, 1))
    step = np.diff(np.concatenate([[0.0], grid]))
    ind = (grid[None, :] >= d[:, None]).astype(float)
    crps = (((cdf - ind) ** 2) * step[None, :]).sum(axis=1)
    out = {"log": log, "crps": np.where(censored, np.nan, crps)}
    for q in quantiles:
        qv = grid[np.clip((cdf < q).sum(axis=1), 0, len(grid) - 1)]
        out[f"pinball_{q}"] = np.where(censored, np.nan, np.maximum(q * (d - qv), (q - 1) * (d - qv)))
    lo = grid[np.clip((cdf < 0.1).sum(axis=1), 0, len(grid) - 1)]
    hi = grid[np.clip((cdf < 0.9).sum(axis=1), 0, len(grid) - 1)]
    out["covered_80"] = np.where(censored, np.nan, ((d >= lo) & (d <= hi)).astype(float))
    return out


# ---------------------------------------------------------------- aggregation and bootstrap

def daily_sums(items, day_index, n_days):
    """items: (units, origins, ...) losses; day_index: (origins,) day number. -> sums, counts (units, days)."""
    x = np.asarray(items, float).reshape(items.shape[0], items.shape[1], -1)
    ok = ~np.isnan(x)
    s = np.where(ok, x, 0).sum(axis=2)
    c = ok.sum(axis=2)
    sums = np.zeros((x.shape[0], n_days))
    cnts = np.zeros((x.shape[0], n_days))
    for u in range(x.shape[0]):
        sums[u] = np.bincount(day_index, weights=s[u], minlength=n_days)
        cnts[u] = np.bincount(day_index, weights=c[u], minlength=n_days)
    return sums, cnts


def macro(sums, cnts):
    """Mean over units of each unit's mean loss (units without scored items are skipped)."""
    per_unit = sums.sum(axis=1) / np.where(cnts.sum(axis=1) > 0, cnts.sum(axis=1), np.nan)
    return float(np.nanmean(per_unit)), per_unit


def micro(sums, cnts):
    return float(sums.sum() / cnts.sum())


def skill(model, ref):
    """PRIMARY skill (D38): 1 - macroLoss(model) / macroLoss(ref).
    Losses are first averaged per unit, then over units (each unit counts equally), then ONE ratio.
    No per-unit ratios: a unit with a near-zero baseline loss cannot blow up the result."""
    return float(1 - macro(*model)[0] / macro(*ref)[0])


def micro_skill(model, ref):
    """Skill from losses pooled over all scored items (reported next to the primary)."""
    return float(1 - micro(*model) / micro(*ref))


def per_unit_skill(model, ref):
    """Descriptive only (never averaged into a claim): 1 - loss_u(model) / loss_u(ref) per unit."""
    _, m = macro(*model)
    _, r = macro(*ref)
    return 1 - m / np.where(r > 0, r, np.nan)


def per_horizon_skill(model_items, ref_items, day_index, n_days):
    """Primary and micro skill for each horizon separately. Items: (units, origins, horizons)."""
    out = []
    for h in range(model_items.shape[2]):
        mo = daily_sums(model_items[:, :, h:h + 1], day_index, n_days)
        rf = daily_sums(ref_items[:, :, h:h + 1], day_index, n_days)
        out.append((h + 1, skill(mo, rf), micro_skill(mo, rf)))
    return out


def bootstrap(stat, pairs, n_rep=2000, seed=0, block_days=1):
    """Block bootstrap over days. Each replicate draws ONE set of day indices and applies it to every
    unit (all rows) and every input pair (model and baseline), so days of mass attacks stay intact
    across units and the model-baseline pairing is kept. block_days=1: day blocks (primary);
    block_days=7: week blocks (sensitivity). Returns (estimate, low95, high95)."""
    rng = np.random.default_rng(seed)
    n_days = pairs[0][0].shape[1]
    est = stat(*pairs)
    reps = np.empty(n_rep)
    for i in range(n_rep):
        if block_days == 1:
            d = rng.integers(0, n_days, n_days)
        else:
            n_blocks = -(-n_days // block_days)
            starts = rng.integers(0, max(n_days - block_days, 0) + 1, n_blocks)
            d = (starts[:, None] + np.arange(block_days)[None, :]).ravel()[:n_days]
        reps[i] = stat(*[(s[:, d], c[:, d]) for s, c in pairs])
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return est, float(lo), float(hi)


def claim(low, high):
    """Claim rule (12.3): an improvement only if the 95 % interval excludes 0 (positive = better)."""
    return "improves" if low > 0 else ("worse" if high < 0 else "no claim (interval includes 0)")
