"""Step 12c (stage 6c): baselines B1-B3 on the 4 validation folds; choose and save the best baseline.

Protocol 12.4/12.6, D38. Validation folds only (the guard blocks test-window labels).
For each fold: statistics from the 8 weeks before (fold start - 24 h purge); origins every full hour.
Outputs: printed tables, logs/step12c_validation_scores.csv, config/best_baseline.json.

Run from the project root:
    python scripts/step12c_baselines_validation.py | tee logs/step12c_output.txt
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import baselines as bl, config, leakage, metrics as m  # noqa: E402
from aaf.data import load_histories, load_units, ns  # noqa: E402
from aaf.targets import fold_origins, map_target  # noqa: E402

NAMES = ("B1", "B2", "B3")
LABELS = {"B1": "climatology (hour of week, 8 weeks)", "B2": "calibrated persistence", "B3": "7-day rolling share"}
BEST_FILE = config.ROOT / "config" / "best_baseline.json"
SCORES_FILE = config.ROOT / "logs" / "step12c_validation_scores.csv"


def section(t):
    print(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


def main():
    units = load_units()["unit"].tolist()
    hist = load_histories("official", units)
    v0 = ns(config.VAL_START)
    origins, cuts = [], []
    for a, b in config.FOLDS:
        o = fold_origins(a, b)
        origins.append(o)
        cuts.append(np.full(len(o), ns(a - config.PURGE)))
    origins, cuts = np.concatenate(origins), np.concatenate(cuts)
    day = ((origins - v0) // (24 * config.HOUR_NS)).astype(int)
    n_days = int(day.max()) + 1
    section("SETUP")
    print(f"units: {len(units)}; validation origins: {len(origins)} over {len(np.unique(day))} days; folds: {len(config.FOLDS)}")

    probs = {n: np.full((len(units), len(origins), 24), np.nan) for n in NAMES}
    y = np.zeros((len(units), len(origins), 24))
    dur_items = {"log": [], "crps": [], "pinball_0.5": [], "covered_80": []}
    dur_median_err = []
    for ui, u in enumerate(units):
        h = hist[u]
        y[ui] = map_target(h, origins)
        for cut in np.unique(cuts):
            sel = cuts == cut
            params = bl.fit(h.observable(int(cut)), int(cut))          # statistics strictly before the cut
            for n in NAMES:
                probs[n][ui, sel] = bl.predict(n, params, h, origins[sel])
            # duration baseline: empirical CDF from the year before the cut, scored on alerts in the fold
            cdf = bl.duration_cdf(h.observable(int(cut)), int(cut), int(cut) - 365 * 24 * config.HOUR_NS)
            fold_end = int(cut) + config.PURGE.value + 14 * 24 * config.HOUR_NS
            mfold = (h.starts >= int(cut) + config.PURGE.value) & (h.ends <= fold_end)
            d = (h.ends[mfold] - h.starts[mfold]) / config.MIN_NS
            if len(d):
                sc = m.duration_scores(np.repeat(cdf[None, :], len(d), 0), bl.DUR_GRID, d)
                for k in dur_items:
                    dur_items[k].append(np.nanmean(sc[k]))
                med = bl.DUR_GRID[np.searchsorted(cdf, 0.5)]
                dur_median_err.append(np.median(np.abs(d - med)))

    section("LEAKAGE CHECK - baselines on real histories (12 units x 10 times)")
    sample = {u: hist[u] for u in units[:: max(1, len(units) // 12)]}
    times = [int(origins[i]) for i in np.linspace(0, len(origins) - 1, 10).astype(int)]
    problems = []
    for n in NAMES:
        def f(hh, t, n=n):
            cut = int(cuts[np.searchsorted(origins, t)])
            return bl.predict(n, bl.fit(hh.observable(cut), cut), hh, [t])[0]
        problems += [(n,) + p for p in leakage.check(f, sample, times)]
    print(f"problems: {len(problems)}", problems[:5] if problems else "")
    if problems:
        raise SystemExit("STOP: leakage detected")

    section("VALIDATION SCORES (map target, Brier; lower is better)")
    agg = {n: m.daily_sums(m.brier_items(probs[n], y), day, n_days) for n in NAMES}
    rows = []
    for n in NAMES:
        ll = m.daily_sums(m.logloss_items(probs[n], y), day, n_days)
        rows.append({"baseline": n, "what": LABELS[n], "macro_brier": m.macro(*agg[n])[0], "micro_brier": m.micro(*agg[n]),
                     "macro_logloss": m.macro(*ll)[0], "ece": m.reliability(probs[n], y)[1],
                     "avg_precision": m.average_precision(probs[n], y)})
    t = pd.DataFrame(rows)
    t.to_csv(SCORES_FILE, index=False)
    print(t.round(4).to_string(index=False))
    best = t.loc[t["macro_brier"].idxmin(), "baseline"]
    print(f"\nbest baseline by validation macro Brier: {best} ({LABELS[best]})")

    section(f"OTHERS vs BEST ({best}): primary skill with day-block and week-block 95 % intervals")
    for n in NAMES:
        if n == best:
            continue
        d1 = m.bootstrap(m.skill, [agg[n], agg[best]], n_rep=1000)
        d7 = m.bootstrap(m.skill, [agg[n], agg[best]], n_rep=1000, block_days=7)
        print(f"{n} vs {best}: skill {d1[0]:+.4f}  day-block [{d1[1]:+.4f}, {d1[2]:+.4f}]  week-block [{d7[1]:+.4f}, {d7[2]:+.4f}]"
              f"  micro {m.micro_skill(agg[n], agg[best]):+.4f}")

    section("SANITY CHECK - Brier by horizon (persistence should be best at +1 h and worse at +24 h; climatology flat)")
    hs = [1, 3, 6, 12, 24]
    tab = {n: [m.macro(*m.daily_sums(m.brier_items(probs[n][:, :, h - 1:h], y[:, :, h - 1:h]), day, n_days))[0] for h in hs] for n in NAMES}
    print(pd.DataFrame(tab, index=[f"+{h}h" for h in hs]).round(4).to_string())

    section("DURATION BASELINE (empirical distribution from the year before each fold), validation alerts")
    print({k: round(float(np.mean(v)), 3) for k, v in dur_items.items()},
          f"| median absolute error of the predicted median: {np.median(dur_median_err):.1f} min")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    BEST_FILE.write_text(json.dumps({
        "best_baseline": best, "description": LABELS[best], "chosen_on": "validation folds 2026-05-18 -> 2026-07-13 (macro Brier)",
        "validation_macro_brier": {r["baseline"]: round(r["macro_brier"], 6) for r in rows},
        "chosen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_commit": commit,
        "note": "chosen before any model is fitted and before any test-window evaluation (D38)"}, indent=2))
    print(f"\nsaved -> {BEST_FILE.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
