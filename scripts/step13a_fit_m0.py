"""Step 13a: fit M0-stack on validation-fold predictions only (D40) and freeze it in config/m0_stack.json.

Run from the project root:
    python scripts/step13a_fit_m0.py | tee logs/step13a_output.txt
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import config, evaluate as ev, leakage, m0  # noqa: E402
from aaf import baselines as bl  # noqa: E402
from aaf.data import load_histories, load_units, ns  # noqa: E402
from aaf.targets import fold_origins  # noqa: E402


def main():
    units = load_units()["unit"].tolist()
    off = load_histories("official", units)
    probs = {n: [] for n in ev.NAMES}
    ys, origins = [], []
    for a, b in config.FOLDS:
        o = fold_origins(a, b)
        p = ev.baseline_probs(off, units, o, ns(a - config.PURGE))
        for n in ev.NAMES:
            probs[n].append(p[n])
        ys.append(ev.labels(off, units, o, final_test=False))
        origins.append(o)
    probs = {n: np.concatenate(v, axis=1) for n, v in probs.items()}
    y = np.concatenate(ys, axis=1)
    origins = np.concatenate(origins)

    coef = m0.fit(probs, y, origins)
    print("M0-stack coefficients (fitted on validation folds only):")
    for k, v in coef.items():
        print(f"  {k:14s} {v:+.4f}")
    probs["M0"] = m0.predict(coef, probs, origins)
    ev.report("VALIDATION (IN-SAMPLE for M0 - not for claims)", probs, y, origins, n_rep=200)

    # leakage: M0 output at t uses only baseline forecasts at t + calendar
    sample = {u: off[u] for u in units[:: max(1, len(units) // 10)]}
    times = [int(origins[i]) for i in np.linspace(0, len(origins) - 1, 8).astype(int)]
    cuts = {int(o): ns(a - config.PURGE) for a, b in config.FOLDS for o in fold_origins(a, b)}
    def f(h, t):
        cut = cuts[t]
        prm = bl.fit(h.observable(cut), cut)
        pr = {n: bl.predict(n, prm, h, [t])[None, :, :] for n in ev.NAMES}
        return m0.predict(coef, pr, np.array([t]))[0, 0]
    problems = leakage.check(f, sample, times)
    print(f"\nleakage check M0 (10 units x 8 times): problems {len(problems)}")
    if problems:
        raise SystemExit("STOP: leakage")
    m0.save(coef, {"fitted_on": "validation folds 2026-05-18 -> 2026-07-13 (official)", "rows_stride": 3,
                   "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "note": "D40: frozen before the final test; inputs on the post-reform test come from a mixed 8-week window"})
    print(f"saved -> {m0.FILE.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
