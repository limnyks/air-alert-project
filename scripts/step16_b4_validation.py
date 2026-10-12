"""Step 16: B4 = exponentially weighted share (D43); choose the half-life on the 4 validation folds.

Same units, origins and labels as step12c (official data, validation folds only; the guard blocks test labels).
Selection rule fixed in D43 before running: lowest pooled macro Brier; B4 replaces B3 on the map only if the
winner beats B3 in at least 3 of 4 folds. Also prints Brier by horizon with B2 (for the +1/+6/+24 h question).
Outputs: printed tables, logs/step16_b4_scores.csv, config/b4_halflife.json.

Run from the project root:
    python scripts/step16_b4_validation.py | tee logs/step16_output.txt
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import baselines as bl, config, evaluate as ev, leakage, metrics as m  # noqa: E402
from aaf.data import load_histories, load_units, ns  # noqa: E402
from aaf.targets import fold_origins, map_target  # noqa: E402

HALF_LIVES = (3, 7, 14, 30, 60, 120)
OUT_FILE = config.ROOT / "config" / "b4_halflife.json"
SCORES_FILE = config.ROOT / "logs" / "step16_b4_scores.csv"
DAY = 24 * config.HOUR_NS


def section(t):
    print(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


def main():
    units = load_units()["unit"].tolist()
    hist = load_histories("official", units)
    covered = ev.covered_fn(ev.official_segments())
    since = ns(config.OFFICIAL_START)
    v0 = ns(config.VAL_START)
    origins, cuts, fold = [], [], []
    for i, (a, b) in enumerate(config.FOLDS):
        o = fold_origins(a, b)
        origins.append(o)
        cuts.append(np.full(len(o), ns(a - config.PURGE)))
        fold.append(np.full(len(o), i))
    origins, cuts, fold = np.concatenate(origins), np.concatenate(cuts), np.concatenate(fold)
    day = ((origins - v0) // DAY).astype(int)
    n_days = int(day.max()) + 1

    names = ["B2", "B3"] + [f"B4_h{h}" for h in HALF_LIVES]
    probs = {n: np.zeros((len(units), len(origins), 24)) for n in names}
    y = np.zeros((len(units), len(origins), 24))
    for ui, u in enumerate(units):
        h = hist[u]
        y[ui] = map_target(h, origins)
        for cut in np.unique(cuts):
            sel = cuts == cut
            params = bl.fit(h.observable(int(cut)), int(cut))
            probs["B2"][ui, sel] = bl.predict("B2", params, h, origins[sel])
            probs["B3"][ui, sel] = bl.predict("B3", params, h, origins[sel])
        for hl in HALF_LIVES:
            probs[f"B4_h{hl}"][ui] = bl.ewshare(h, origins, hl, covered=covered, since=since)
    section("SETUP")
    print(f"units: {len(units)}; origins: {len(origins)} over {len(np.unique(day))} days; folds: {len(config.FOLDS)}; "
          f"NaN forecasts: {sum(int(np.isnan(p).sum()) for p in probs.values())}")

    section("POOLED VALIDATION SCORES (map target; lower is better)")
    agg = {n: m.daily_sums(m.brier_items(p, y), day, n_days) for n, p in probs.items()}
    rows = []
    for n, p in probs.items():
        ll = m.daily_sums(m.logloss_items(p, y), day, n_days)
        r = {"model": n, "macro_brier": m.macro(*agg[n])[0], "micro_brier": m.micro(*agg[n]),
             "macro_logloss": m.macro(*ll)[0], "ece": m.reliability(p, y)[1]}
        for i in range(len(config.FOLDS)):
            f = fold == i
            r[f"fold{i + 1}_brier"] = m.macro(*m.daily_sums(m.brier_items(p[:, f], y[:, f]), day[f], n_days))[0]
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(SCORES_FILE, index=False)
    print(t.round(4).to_string(index=False))

    b4 = t[t["model"].str.startswith("B4")]
    win = b4.loc[b4["macro_brier"].idxmin(), "model"]
    hl = int(win.split("_h")[1])
    folds_better = int(sum(t.loc[t["model"] == win, f"fold{i + 1}_brier"].item() < t.loc[t["model"] == "B3", f"fold{i + 1}_brier"].item()
                           for i in range(len(config.FOLDS))))
    use_b4 = folds_better >= 3
    print(f"\nwinner by pooled macro Brier: {win} (half-life {hl} days); beats B3 in {folds_better}/4 folds "
          f"-> {'B4 replaces B3 on the map' if use_b4 else 'B3 stays the map model'} (rule D43)")

    section(f"SKILL vs B3 (1 - macroBrier/macroBrier(B3)), 95 % intervals")
    sk = {}
    for n in names:
        if n == "B3":
            continue
        d1 = m.bootstrap(m.skill, [agg[n], agg["B3"]], n_rep=1000)
        d7 = m.bootstrap(m.skill, [agg[n], agg["B3"]], n_rep=1000, block_days=7)
        sk[n] = d1
        print(f"  {n:8s} {d1[0]:+.4f}  day-block [{d1[1]:+.4f}, {d1[2]:+.4f}]  week-block [{d7[1]:+.4f}, {d7[2]:+.4f}]"
              f"  micro {m.micro_skill(agg[n], agg['B3']):+.4f}")

    section("MACRO BRIER BY HORIZON (for the +1 / +6 / +24 h question)")
    hs = [1, 3, 6, 12, 24]
    ph = {n: [m.macro(*m.daily_sums(m.brier_items(probs[n][:, :, k - 1:k], y[:, :, k - 1:k]), day, n_days))[0] for k in hs]
          for n in ("B2", "B3", win)}
    print(pd.DataFrame(ph, index=[f"+{k}h" for k in hs]).round(4).to_string())

    section(f"LEAKAGE CHECK - {win} on real histories (12 units x 10 times)")
    sample = {u: hist[u] for u in units[:: max(1, len(units) // 12)]}
    times = [int(origins[i]) for i in np.linspace(0, len(origins) - 1, 10).astype(int)]
    f = lambda hh, tt: bl.ewshare(hh, [tt], hl, covered=covered, since=since)[0]  # noqa: E731
    problems = leakage.check(f, sample, times)
    print(f"problems: {len(problems)}", problems[:5] if problems else "")
    if problems:
        raise SystemExit("STOP: leakage detected")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    OUT_FILE.write_text(json.dumps({
        "model": "B4 exponentially weighted share (D43)", "half_life_days": hl, "use_on_map": bool(use_b4),
        "folds_better_than_B3": folds_better, "chosen_on": "validation folds 2026-05-18 -> 2026-07-13 (pooled macro Brier)",
        "validation_macro_brier": {r["model"]: round(r["macro_brier"], 6) for r in rows},
        "skill_vs_B3": [round(float(x), 4) for x in sk[win]],
        "chosen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_commit": commit,
        "note": "not evaluated on any test window (final test spent, D31); describe as chosen on validation only"}, indent=2))
    print(f"\nsaved -> {OUT_FILE.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
