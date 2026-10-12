"""Step 18: B5 = B4 x daily pattern (D45) vs B4 on the 4 validation folds; apply the pre-registered rule.

Same units, origins and labels as step16. The hour-of-day profile for each fold comes from the 8 weeks
before the fold cut (fold start - 24 h purge), pooled over all raions.
Rule (D45): B5 replaces B4 for +6/+12/+24 h only if (1) lower pooled macro Brier, (2) better in >= 3 of 4
folds, (3) ECE not higher than B4's.
Outputs: printed tables, logs/step18_b5_scores.csv, config/b5_daily.json.

Run from the project root:
    python scripts/step18_b5_validation.py | tee logs/step18_output.txt
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

HL = json.loads((config.ROOT / "config" / "b4_halflife.json").read_text())["half_life_days"]
OUT_FILE = config.ROOT / "config" / "b5_daily.json"
SCORES_FILE = config.ROOT / "logs" / "step18_b5_scores.csv"
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

    section("DAILY PROFILES (8 weeks before each fold cut, pooled over raions; 1.0 = average hour)")
    profiles = {}
    for cut in np.unique(cuts):
        profiles[int(cut)] = bl.day_profile([hist[u].observable(int(cut)) for u in units], int(cut), covered)
    print(pd.DataFrame({pd.Timestamp(c).strftime("cut %m-%d"): p for c, p in profiles.items()},
                       index=[f"{k:02d}h" for k in range(24)]).round(2).T.to_string())

    names = ("B2", "B4", "B5")
    probs = {n: np.zeros((len(units), len(origins), 24)) for n in names}
    y = np.zeros((len(units), len(origins), 24))
    for ui, u in enumerate(units):
        h = hist[u]
        y[ui] = map_target(h, origins)
        probs["B4"][ui] = bl.ewshare(h, origins, HL, covered=covered, since=since)
        for cut in np.unique(cuts):
            sel = cuts == cut
            probs["B2"][ui, sel] = bl.predict("B2", bl.fit(h.observable(int(cut)), int(cut)), h, origins[sel])
            probs["B5"][ui, sel] = bl.ewshare_daily(h, origins[sel], HL, profiles[int(cut)], covered=covered, since=since)

    section("POOLED VALIDATION SCORES (map target; lower is better)")
    agg = {n: m.daily_sums(m.brier_items(p, y), day, n_days) for n, p in probs.items()}
    rows = []
    for n, p in probs.items():
        ll = m.daily_sums(m.logloss_items(p, y), day, n_days)
        r = {"model": n, "macro_brier": m.macro(*agg[n])[0], "macro_logloss": m.macro(*ll)[0], "ece": m.reliability(p, y)[1]}
        for i in range(len(config.FOLDS)):
            f = fold == i
            r[f"fold{i + 1}_brier"] = m.macro(*m.daily_sums(m.brier_items(p[:, f], y[:, f]), day[f], n_days))[0]
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model")
    t.to_csv(SCORES_FILE)
    print(t.round(4).to_string())

    d1 = m.bootstrap(m.skill, [agg["B5"], agg["B4"]], n_rep=1000)
    d7 = m.bootstrap(m.skill, [agg["B5"], agg["B4"]], n_rep=1000, block_days=7)
    print(f"\nB5 vs B4 skill {d1[0]:+.4f}  day-block [{d1[1]:+.4f}, {d1[2]:+.4f}]  week-block [{d7[1]:+.4f}, {d7[2]:+.4f}]")

    section("MACRO BRIER BY HORIZON")
    hs = [1, 3, 6, 12, 24]
    ph = {n: [m.macro(*m.daily_sums(m.brier_items(probs[n][:, :, k - 1:k], y[:, :, k - 1:k]), day, n_days))[0] for k in hs] for n in names}
    print(pd.DataFrame(ph, index=[f"+{k}h" for k in hs]).round(4).to_string())

    section("RULE D45")
    c1 = t.loc["B5", "macro_brier"] < t.loc["B4", "macro_brier"]
    folds = int(sum(t.loc["B5", f"fold{i + 1}_brier"] < t.loc["B4", f"fold{i + 1}_brier"] for i in range(4)))
    c2 = folds >= 3
    c3 = t.loc["B5", "ece"] <= t.loc["B4", "ece"]
    adopt = bool(c1 and c2 and c3)
    print(f"(1) lower pooled macro Brier: {c1}  (2) better in {folds}/4 folds: {c2}  (3) ECE not higher: {c3}"
          f"\n-> {'B5 replaces B4 for +6/+12/+24 h' if adopt else 'B4 stays for +6/+12/+24 h'}")

    section("LEAKAGE CHECK - B5 on real histories (12 units x 10 times)")
    sample = {u: hist[u] for u in units[:: max(1, len(units) // 12)]}
    times = [int(origins[i]) for i in np.linspace(0, len(origins) - 1, 10).astype(int)]
    f = lambda hh, tt: bl.ewshare_daily(hh, [tt], HL, profiles[int(cuts[np.searchsorted(origins, tt)])],  # noqa: E731
                                        covered=covered, since=since)[0]
    problems = leakage.check(f, sample, times)
    print(f"problems: {len(problems)}", problems[:5] if problems else "")
    if problems:
        raise SystemExit("STOP: leakage detected")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    OUT_FILE.write_text(json.dumps({
        "model": "B5 = B4 x pooled hour-of-day profile (D45)", "adopt": adopt, "half_life_days": HL,
        "criteria": {"lower_pooled_macro_brier": bool(c1), "folds_better": folds, "ece_not_higher": bool(c3)},
        "validation": {n: {"macro_brier": round(float(t.loc[n, "macro_brier"]), 6), "ece": round(float(t.loc[n, "ece"]), 6)} for n in names},
        "skill_vs_B4": [round(float(x), 4) for x in d1],
        "chosen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_commit": commit,
        "note": "validation only; not tested on held-out data"}, indent=2))
    print(f"\nsaved -> {OUT_FILE.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
