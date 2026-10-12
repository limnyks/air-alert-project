"""Step 13: THE FINAL TEST (protocol 12.1, D31, D38-D40). Run ONCE, after M0 is frozen in a commit.

Windows:
  pre-reform  : origins 2026-07-13 00:00 -> 2026-09-06 05:00 UTC, official labels; B1/B2 statistics from the
                8 weeks before 2026-07-12 00:00 (24 h purge); origins whose [t-7d, t+24h] touches the outage
                period are masked (D37).
  post-reform : origins 2026-09-18 09:00 -> 2026-10-10 08:00 UTC, alerts.in.ua labels (air raid); B1/B2
                statistics from the 8 weeks before the test start (official + gap + adaptation week, gaps and
                outage excluded); B3 and the state at t from alerts.in.ua only.
Models: B1, B2, B3 (best baseline, D39) and M0-stack if config/m0_stack.json exists.

Run from the project root (once):
    python scripts/step13_final_test.py | tee logs/step13_final_test_output.txt
"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import baselines as bl, config, evaluate as ev, m0, metrics as mt  # noqa: E402
from aaf.data import load_histories, load_units, ns  # noqa: E402

H = config.HOUR_NS


def hourly(a, b):
    """Full UTC hours in [a, b) as ns."""
    t = pd.date_range(pd.Timestamp(a).ceil("h"), pd.Timestamp(b), freq="h", inclusive="left")
    return t.tz_convert("UTC").tz_localize(None).as_unit("ns").asi8


def durations(hists, units, a, b, cdf_hists, since, cut, mask_outage):
    """Duration baseline scores for alerts starting in [a, b): log score (censoring-aware), median abs error, 80 % coverage."""
    logs, err, cov = [], [], []
    for u in units:
        h = hists[u]
        sel = (h.starts >= a) & (h.starts < b)
        if mask_outage:
            for oa, ob in ev.outage_periods():
                sel &= ~((h.starts < ob) & (h.ends > oa))
        if not sel.any():
            continue
        cens = h.ends[sel] == np.iinfo(np.int64).max
        d = np.where(cens, (ns(config.SNAPSHOT) - h.starts[sel]), h.ends[sel] - h.starts[sel]) / config.MIN_NS
        cdf = bl.duration_cdf(cdf_hists[u].observable(cut), cut, since)
        sc = mt.duration_scores(np.repeat(cdf[None, :], len(d), 0), bl.DUR_GRID, d, censored=cens)
        logs.append(np.nanmean(sc["log"]))
        cov.append(np.nanmean(sc["covered_80"]))
        med = bl.DUR_GRID[np.searchsorted(cdf, 0.5)]
        err.append(np.median(np.abs(d[~cens] - med)) if (~cens).any() else np.nan)
    return {"units": len(logs), "log_score": float(np.nanmean(logs)), "median_abs_error_min": float(np.nanmedian(err)),
            "coverage_80": float(np.nanmean(cov))}


def run_window(name, hists, units, origins, cut, covered, final_label_hists, coef):
    probs = ev.baseline_probs(hists, units, origins, cut, covered)
    if coef is not None:
        probs["M0"] = m0.predict(coef, probs, origins)
    y = ev.labels(final_label_hists, units, origins, final_test=True)
    return ev.report(name, probs, y, origins)


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "aaf", "scripts", "config"], capture_output=True, text=True,
                           cwd=config.ROOT).stdout.strip()
    print(f"FINAL TEST run at {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC, code commit {commit}"
          f"{' (WARNING: uncommitted changes in aaf/scripts/config)' if dirty else ''}")
    units = load_units()["unit"].tolist()
    off = load_histories("official", units)
    aic = load_histories("alerts_in_ua", units)
    coef = m0.load()
    print(f"units: {len(units)}; models: B1, B2, B3{', M0-stack (frozen)' if coef else ' (no M0 file: baselines only)'}")

    # ---- pre-reform test (secondary)
    o_pre = hourly(config.PRE_TEST[0], config.OFFICIAL_END - pd.Timedelta(hours=24))
    o_pre = o_pre[ev.origin_mask(o_pre)]
    cut_pre = ns(config.PRE_TEST[0] - config.PURGE)
    cov_off = ev.covered_fn(ev.official_segments())
    t1, s1, p1 = run_window("PRE-REFORM TEST (secondary): official labels", off, units, o_pre, cut_pre, cov_off, off, coef)

    # ---- post-reform test (primary)
    comb = {u: ev.combined_history(off[u], aic[u]) for u in units}
    o_post = hourly(config.POST_TEST[0], config.SNAPSHOT - pd.Timedelta(hours=24))
    cut_post = ns(config.POST_TEST[0])
    cov_post = ev.covered_fn(ev.official_segments() + [(ns(config.ADAPT[0]), ns(config.SNAPSHOT))])
    t2, s2, p2 = run_window("POST-REFORM TEST (PRIMARY, latest period): alerts.in.ua labels", comb, units, o_post, cut_post,
                            cov_post, comb, coef)

    # ---- durations (empirical baseline from the official year before each window)
    print(f"\n{'=' * 90}\nDURATION BASELINE on the test windows\n{'=' * 90}")
    d1 = durations(off, units, ns(config.PRE_TEST[0]), ns(config.OFFICIAL_END), off, cut_pre - 365 * 24 * H, cut_pre, True)
    d2 = durations(comb, units, ns(config.POST_TEST[0]), ns(config.SNAPSHOT), off, ns(config.OFFICIAL_END) - 365 * 24 * H,
                   ns(config.OFFICIAL_END), False)
    print("pre-reform :", {k: round(v, 3) for k, v in d1.items()})
    print("post-reform:", {k: round(v, 3) for k, v in d2.items()})

    out = config.ROOT / "logs"
    for tag, (t, s, p) in {"pre": (t1, s1, p1), "post": (t2, s2, p2)}.items():
        t.to_csv(out / f"step13_{tag}_scores.csv", index=False)
        s.to_csv(out / f"step13_{tag}_skill.csv", index=False)
        p.to_csv(out / f"step13_{tag}_by_horizon.csv")
    pd.DataFrame([{"window": "pre", **d1}, {"window": "post", **d2}]).to_csv(out / "step13_durations.csv", index=False)
    print("\nresults saved -> logs/step13_*.csv")


if __name__ == "__main__":
    main()
