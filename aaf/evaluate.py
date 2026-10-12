"""Shared evaluation pieces for validation and the final test: histories per window, baseline and
M0 predictions, labels, masks, and the results tables (protocol 12, D38 wording)."""
import numpy as np
import pandas as pd

from . import baselines as bl, config, metrics as m
from .data import ns
from .history import UNKNOWN_END, History
from .targets import map_target

H = config.HOUR_NS
DAY = 24 * H
NAMES = ("B1", "B2", "B3")


def outage_periods():
    o = pd.read_csv(config.OUTAGE_PERIODS)
    return [(ns(pd.Timestamp(a)), ns(pd.Timestamp(b))) for a, b in zip(o["start_utc"], o["end_utc"])]


def covered_fn(segments):
    """segments: list of (a, b) ns with data. Returns covered(times) -> bool array."""
    def covered(t):
        t = np.asarray(t, dtype=np.int64)
        ok = np.zeros(len(t), bool)
        for a, b in segments:
            ok |= (t >= a) & (t < b)
        return ok
    return covered


def official_segments():
    """Official coverage minus outage periods."""
    segs, a = [], ns(config.OFFICIAL_START)
    for oa, ob in sorted(outage_periods()):
        segs.append((a, oa))
        a = ob
    segs.append((a, ns(config.OFFICIAL_END)))
    return segs


def combined_history(off, aic):
    """Official (clipped at its end) + alerts.in.ua (from the adaptation start) as one History.
    The gap between them has no data (handled by `covered`)."""
    e_off = ns(config.OFFICIAL_END)
    a0 = ns(config.ADAPT[0])
    k = off.starts < e_off
    s1, e1 = off.starts[k], np.minimum(off.ends[k], e_off)
    keep = (aic.ends > a0)
    s2 = np.maximum(aic.starts[keep], a0)
    e2 = aic.ends[keep]
    s, e = np.concatenate([s1, s2]), np.concatenate([e1, e2])
    order = np.argsort(s)
    s, e = s[order], e[order]
    first = np.ones(len(s), bool)
    first[1:] = s[1:] > s[:-1]                                    # drop exact duplicate starts if any
    return History(s[first].astype(np.int64), e[first].astype(np.int64))


def baseline_probs(hists, units, origins, cut, covered=None):
    """dict name -> (units, origins, 24); statistics from the 8 weeks before `cut` (one cut)."""
    out = {n: np.zeros((len(units), len(origins), 24)) for n in NAMES}
    for i, u in enumerate(units):
        h = hists[u]
        params = bl.fit(h.observable(cut), cut, covered)
        for n in NAMES:
            out[n][i] = bl.predict(n, params, h, origins)
    return out


def labels(hists, units, origins, final_test):
    return np.stack([map_target(hists[u], origins, final_test=final_test) for u in units])


def origin_mask(origins, lookback_days=7):
    """False for origins whose [t - 7 d, t + 24 h] overlaps an outage period (features or labels affected)."""
    o = np.asarray(origins, dtype=np.int64)
    ok = np.ones(len(o), bool)
    for a, b in outage_periods():
        ok &= ~((o - lookback_days * DAY < b) & (o + DAY > a))
    return ok


def report(title, probs, y, origins, best="B3", n_rep=1000):
    """Results tables for one window. probs: dict name -> (U, O, 24); y: (U, O, 24)."""
    day = ((origins - origins.min()) // DAY).astype(int)
    n_days = int(day.max()) + 1
    agg = {n: m.daily_sums(m.brier_items(p, y), day, n_days) for n, p in probs.items()}
    rows = []
    for n, p in probs.items():
        ll = m.daily_sums(m.logloss_items(p, y), day, n_days)
        rows.append({"model": n, "macro_brier": m.macro(*agg[n])[0], "micro_brier": m.micro(*agg[n]),
                     "macro_logloss": m.macro(*ll)[0], "ece": m.reliability(p, y)[1],
                     "avg_precision": m.average_precision(p, y), "confident_wrong": m.confident_wrong(p, y)})
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
    print(f"forecast origins: {len(origins)} over {len(np.unique(day))} days; units: {y.shape[0]}; base rate {np.nanmean(y):.4f}")
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    sk = []
    print(f"\nskill vs best baseline {best} (primary = 1 - macroBrier/macroBrier(best)); 95 % intervals:")
    for n in probs:
        if n == best:
            continue
        d1 = m.bootstrap(m.skill, [agg[n], agg[best]], n_rep=n_rep)
        d7 = m.bootstrap(m.skill, [agg[n], agg[best]], n_rep=n_rep, block_days=7)
        verdict = m.claim(d1[1], d1[2])
        if verdict == "improves" and d7[1] <= 0:
            verdict = "no claim (week-block interval includes 0)"      # D38: the more cautious one
        print(f"  {n}: {d1[0]:+.4f}  day-block [{d1[1]:+.4f}, {d1[2]:+.4f}]  week-block [{d7[1]:+.4f}, {d7[2]:+.4f}]"
              f"  micro {m.micro_skill(agg[n], agg[best]):+.4f}  -> {verdict}")
        sk.append({"model": n, "skill": d1[0], "day_lo": d1[1], "day_hi": d1[2], "week_lo": d7[1], "week_hi": d7[2],
                   "micro_skill": m.micro_skill(agg[n], agg[best]), "verdict": verdict})
    hs = [1, 3, 6, 12, 24]
    ph = {n: [m.macro(*m.daily_sums(m.brier_items(p[:, :, h - 1:h], y[:, :, h - 1:h]), day, n_days))[0] for h in hs]
          for n, p in probs.items()}
    print("\nmacro Brier by horizon:")
    print(pd.DataFrame(ph, index=[f"+{h}h" for h in hs]).round(4).to_string())
    return t, pd.DataFrame(sk), pd.DataFrame(ph, index=[f"+{h}h" for h in hs])
