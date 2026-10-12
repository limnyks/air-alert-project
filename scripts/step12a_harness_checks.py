"""Step 12a (stage 6a): apply the D36 fixes on real data with the new harness.

Fix 1: recompute unit exclusion (80 % rule) and training starts (x1.5 switch rule) with official data
       before 2026-05-18 only; compare with the Step 10b versions; write config/units_clean.csv.
Fix 2: outage days (no official alert anywhere while the volunteer file has alerts);
       write config/outage_days.csv.
Also: the harness leakage check on real histories with a simple honest forecaster.

Run from the project root:
    python -m unittest discover -s tests -v
    python scripts/step12a_harness_checks.py | tee logs/step12a_output.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import config, leakage, outages, selection  # noqa: E402
from aaf.data import load_histories, ns  # noqa: E402

H = config.HOUR_NS


def section(title):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")


def main():
    checks = pd.read_csv(config.UNIT_CHECKS)
    hist = load_histories("official")

    section("FIX 1 - exclusion and training starts recomputed before 2026-05-18")
    r = selection.recompute(hist, checks)
    r.to_csv(config.UNITS_CLEAN, index=False)
    changed_excl = r[r["excluded"] != r["excluded_step10b"]]
    changed_start = r[r["train_from"] != r["train_from_step10b"]]
    print(f"units: {len(r)}; excluded now: {int(r['excluded'].sum())} (Step 10b: {int(r['excluded_step10b'].sum())}); "
          f"modeled now: {int((~r['excluded']).sum())}")
    print(f"exclusion changed for {len(changed_excl)} units:")
    print(changed_excl[["unit", "share_last92_preval", "excluded", "excluded_step10b"]].round(3).to_string(index=False)
          if len(changed_excl) else "  none")
    print(f"\ntraining start changed for {len(changed_start)} units:")
    print(changed_start[["unit", "ratio_preval", "train_from", "train_from_step10b"]].round(3).to_string(index=False)
          if len(changed_start) else "  none")
    print(f"\nnow excluded: {r.loc[r['excluded'], 'unit'].tolist()}")
    print(f"saved -> {config.UNITS_CLEAN.relative_to(config.ROOT)}")

    section("FIX 2 - possible outages of the official feed")
    o = outages.detect()
    o.to_csv(config.OUTAGE_DAYS, index=False)
    print(f"days checked: {len(o)}; days with no official alert at all: {int((o['official_alerts'] == 0).sum())}; "
          f"of these with volunteer alerts (flagged as outage): {int(o['outage'].sum())}")
    if o["outage"].any():
        print(o[o["outage"]].to_string(index=False))
    print(f"saved -> {config.OUTAGE_DAYS.relative_to(config.ROOT)}")

    section("LEAKAGE CHECK on real histories (honest forecaster)")
    units = r.loc[~r["excluded"], "unit"].tolist()
    sample = {u: hist[u] for u in units[:: max(1, len(units) // 10)]}
    times = [ns(config.FOLDS[0][0]) + k * 37 * H for k in range(20)]   # inside validation
    def honest(h, t):
        o = h.observable(t)
        return np.array([selection.share(o, t - 7 * 24 * H, t), o.state_at([t])[0]])
    problems = leakage.check(honest, sample, times)
    print(f"units checked: {len(sample)}, times: {len(times)}, problems: {len(problems)}")


if __name__ == "__main__":
    main()
