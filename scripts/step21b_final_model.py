"""Step 21b: final model - choose recency on 2025, confirm on 2026, day-by-day evaluation vs flat B4.

logit p(u, t+h) = logit(smoothed level, half-life hl) + s[Kyiv hour of t+h] + w[h] * alert_now(u, t)
s and w: one weighted logistic regression (level as offset), observation weight 0.5^(label age / hp),
rows = (raion, origin every 3 h, horizon) with label time before the refit day (no look-ahead).

Stage 1: level half-life hl in {7, 14, 30, 60, 90} d (flat smoothed level), chosen on 2025.
Stage 2: hour/alert half-life hp in {7, 14, 30, 60, 90} d (full model, weekly refit), chosen on 2025.
Stage 3: chosen (hl, hp) with a DAILY refit: primary = mean macro Brier over +1...+24 h vs flat B4 (h = 7 d)
with a joint day-block bootstrap; per-horizon table; ECE and top-bin calibration; ablations;
breakdown by period (2025 / 2026 Jan-Aug / the alerts.in.ua weeks); w_h curve and hour profile; leakage test.
Note: the evaluation period includes the two spent test windows (re-used on the user's instruction; they
are reported in the period breakdown) and the 2025-2026 exploration data where H1 was found.

Run from the project root:
    python scripts/step21b_final_model.py | tee logs/step21b_output.txt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import config, final, leakage, metrics as m  # noqa: E402
from aaf.data import ns  # noqa: E402

GRID = (7, 14, 30, 60, 90)
STRIDE = 3                                   # training origins every 3 h
EVAL_START = pd.Timestamp("2025-01-01", tz=config.TZ)
CONFIRM_START = pd.Timestamp("2026-01-01", tz=config.TZ)
OUT = config.ROOT / "logs"
DAYH = 24


def section(t):
    print(f"\n{'=' * 90}\n{t}\n{'=' * 90}", flush=True)


class Acc:
    """Accumulates Brier sums per (unit, day) overall and per horizon, plus reliability bins."""
    def __init__(self, U, n_days):
        self.s = np.zeros((U, n_days)); self.c = np.zeros((U, n_days))
        self.sh = np.zeros((U, n_days, 24)); self.ch = np.zeros((U, n_days, 24))
        self.bn = np.zeros((n_days, 10)); self.bp = np.zeros((n_days, 10)); self.by = np.zeros((n_days, 10))

    def add(self, p, y, ok, day):
        """p, y, ok: (U, O, 24); day: (O,) day index."""
        b = np.where(ok, (p - y) ** 2, 0.0)
        for o in range(p.shape[1]):
            self.sh[:, day[o], :] += b[:, o, :]
            self.ch[:, day[o], :] += ok[:, o, :]
        dd = np.broadcast_to(day[None, :, None], p.shape)[ok]
        pp, yy = p[ok], y[ok]
        idx = dd * 10 + np.minimum((pp * 10).astype(int), 9)
        nb = self.bn.size
        self.bn += np.bincount(idx, minlength=nb).reshape(self.bn.shape)
        self.bp += np.bincount(idx, pp, nb).reshape(self.bn.shape)
        self.by += np.bincount(idx, yy, nb).reshape(self.bn.shape)

    def part(self, days):
        return self.sh[:, days].sum(axis=2), self.ch[:, days].sum(axis=2)

    def brier(self, days):
        return m.macro(*self.part(days))[0]

    def ece(self, days=None):
        days = slice(None) if days is None else days
        bn, bp, by = self.bn[days].sum(0), self.bp[days].sum(0), self.by[days].sum(0)
        return float(np.sum(np.abs(bp - by)) / bn.sum())

    def top(self, days=None, lo_bin=8):
        days = slice(None) if days is None else days
        bn, bp, by = self.bn[days].sum(0), self.bp[days].sum(0), self.by[days].sum(0)
        n = bn[lo_bin:].sum()
        return int(n), float(bp[lo_bin:].sum() / n), float(by[lo_bin:].sum() / n)

    def table(self, days):
        bn, bp, by = self.bn[days].sum(0), self.bp[days].sum(0), self.by[days].sum(0)
        return [(f"{b / 10:.1f}-{(b + 1) / 10:.1f}", int(bn[b]), bp[b] / bn[b], by[b] / bn[b]) for b in range(10) if bn[b] > 0]


def main():
    t0 = time.time()
    P = final.build_panel()
    G, COV, S, hod = P["G"], P["COV"], P["S"], P["hod"]
    U, N = S.shape
    jE = int(np.searchsorted(G, ns(EVAL_START)))
    jC = int(np.searchsorted(G, ns(CONFIRM_START)))
    jT = int(np.searchsorted(G, ns(pd.Timestamp("2024-01-01", tz=config.TZ))))
    valid_origin = np.zeros(N, bool)
    valid_origin[1:] = COV[:-1]
    ev_orig = np.array([j for j in range(jE, N - 24) if valid_origin[j]])
    day_of = (G[ev_orig] - G[jE]) // final.DAY
    n_days = int(day_of.max()) + 1
    is_confirm = np.zeros(n_days, bool); is_confirm[day_of[ev_orig >= jC]] = True
    live_days = np.unique(day_of[G[ev_orig] >= P["since"]])
    jan_aug = np.unique(day_of[(ev_orig >= jC) & (G[ev_orig] < ns(pd.Timestamp("2026-09-07", tz="UTC")))])
    print(f"evaluation origins {len(ev_orig)} on {n_days} days ({EVAL_START:%Y-%m-%d} -> "
          f"{pd.Timestamp(G[ev_orig[-1]], tz='UTC').tz_convert(config.TZ):%Y-%m-%d %H:%M} Kyiv); choose on 2025 "
          f"({(~is_confirm).sum()} days), confirm on 2026 ({is_confirm.sum()} days, of which alerts.in.ua {len(live_days)})")
    D_choose, D_conf = np.flatnonzero(~is_confirm), np.flatnonzero(is_confirm)

    # labels and masks for evaluation origins: (U, O, 24)
    H_idx = ev_orig[:, None] + np.arange(1, 25)[None, :]
    Y = S[:, H_idx].astype(float)
    OK = np.broadcast_to(COV[H_idx][None, :, :], Y.shape)
    HOD_T = hod[H_idx]
    A_NOW = S[:, ev_orig].astype(float)

    # training rows (unit, origin, horizon), origins every STRIDE hours from 2024, sorted by label index
    tr_o = np.array([j for j in range(max(jT, 1), N - 1, STRIDE) if valid_origin[j]])
    rr_u, rr_o, rr_h = np.meshgrid(np.arange(U, dtype=np.int16), tr_o, np.arange(1, 25, dtype=np.int8), indexing="ij")
    rr_u, rr_o, rr_h = rr_u.ravel(), rr_o.ravel(), rr_h.ravel()
    lab = rr_o + rr_h
    keep = (lab < N) & COV[np.minimum(lab, N - 1)]
    rr_u, rr_o, rr_h, lab = rr_u[keep], rr_o[keep], rr_h[keep], lab[keep]
    order = np.argsort(lab, kind="stable")
    rr_u, rr_o, rr_h, lab = rr_u[order], rr_o[order], rr_h[order], lab[order]
    rr_o, lab = rr_o.astype(np.int32), lab.astype(np.int32)
    rr_y = S[rr_u, lab]
    rr_a = S[rr_u, rr_o]
    rr_k = hod[lab].astype(np.int8)
    print(f"training row pool: {len(lab):,} rows; {time.time() - t0:.0f} s", flush=True)

    levels = {}
    def level_logit(hl):
        if hl not in levels:
            num, den = final.level_parts(P, hl)
            levels[hl] = (final.logit(final.smoothed_level(num, den)), np.divide(num, den, out=np.zeros_like(num), where=den > 0))
        return levels[hl]

    def run(hl, hp, refit_every, ablation=None, keep_betas=False):
        LL, _ = level_logit(hl)
        acc = Acc(U, n_days)
        beta, betas = None, {}
        cap = min(4 * hp, 365) * DAYH
        starts = list(range(0, n_days, refit_every))
        for d0 in starts:
            sel = (day_of >= d0) & (day_of < d0 + refit_every)
            if not sel.any():
                continue
            jD = int(ev_orig[sel][0])                                   # refit at the first origin of the block
            lo, hi = np.searchsorted(lab, jD - cap), np.searchsorted(lab, jD, side="left")   # labels strictly before the block
            assert hi == 0 or lab[hi - 1] < jD
            w = 0.5 ** ((jD - lab[lo:hi]) / (24.0 * hp))
            k = rr_k[lo:hi].astype(np.int64) if ablation != "no_hour" else np.zeros(hi - lo, np.int64)
            a = rr_a[lo:hi].astype(float) if ablation != "no_alert" else np.zeros(hi - lo)
            beta = final.fit_logistic(LL[rr_u[lo:hi], rr_o[lo:hi]], k, rr_h[lo:hi].astype(np.int64) - 1, a,
                                      rr_y[lo:hi].astype(float), w, beta)
            if keep_betas:
                betas[d0] = beta.copy()
            o = ev_orig[sel]
            hk = HOD_T[sel] if ablation != "no_hour" else np.zeros_like(HOD_T[sel])
            an = A_NOW[:, sel] if ablation != "no_alert" else np.zeros_like(A_NOW[:, sel])
            z = LL[:, o][:, :, None] + beta[hk][None, :, :] + an[:, :, None] * beta[24:][None, None, :]
            acc.add(final.sigmoid(z), Y[:, sel], OK[:, sel], day_of[sel])
        return acc, beta, betas

    def flat(level):
        acc = Acc(U, n_days)
        for d0 in range(0, n_days, 30):
            sel = (day_of >= d0) & (day_of < d0 + 30)
            p = np.repeat(level[:, ev_orig[sel]][:, :, None], 24, axis=2)
            acc.add(p, Y[:, sel], OK[:, sel], day_of[sel])
        return acc

    def mean_brier(acc, days):
        return acc.brier(days)

    section("STAGE 1 - level half-life (flat smoothed level); choose on 2025, confirm on 2026")
    b4 = flat(level_logit(7)[1])
    print(f"  reference flat B4 (h = 7 d, unsmoothed): 2025 {b4.brier(D_choose):.4f} | 2026 {b4.brier(D_conf):.4f}")
    s1 = []
    for hl in GRID:
        a = flat(final.sigmoid(level_logit(hl)[0]))
        s1.append((hl, a.brier(D_choose), a.brier(D_conf)))
        print(f"  hl {hl:3d} d: 2025 {s1[-1][1]:.4f} | 2026 {s1[-1][2]:.4f}", flush=True)
    hl = min(s1, key=lambda r: r[1])[0]
    print(f"  -> chosen level half-life: {hl} d (lowest 2025 Brier)")

    section(f"STAGE 2 - hour/alert half-life (full model, level hl = {hl} d, weekly refit); choose on 2025")
    s2, weekly = [], {}
    for hp in GRID:
        a, _, _ = run(hl, hp, 7)
        weekly[hp] = a
        s2.append((hp, a.brier(D_choose), a.brier(D_conf)))
        print(f"  hp {hp:3d} d: 2025 {s2[-1][1]:.4f} | 2026 {s2[-1][2]:.4f}   ({time.time() - t0:.0f} s)", flush=True)
    hp = min(s2, key=lambda r: r[1])[0]
    print(f"  -> chosen hour/alert half-life: {hp} d (lowest 2025 Brier)")

    section(f"STAGE 3 - final model (hl {hl} d, hp {hp} d), DAILY refit, vs flat B4")
    fin, beta_last, betas = run(hl, hp, 1, keep_betas=True)
    no_hour, _, _ = run(hl, hp, 7, "no_hour")
    no_alert, _, _ = run(hl, hp, 7, "no_alert")
    models = {"B4 flat (h7)": b4, "final": fin, "final (weekly refit)": weekly[hp],
              "final without hour shift (weekly)": no_hour, "final without alert_now (weekly)": no_alert}
    periods = {"2025 (choose)": D_choose, "2026 all (confirm)": D_conf, "2026 Jan-Aug (official)": jan_aug,
               "alerts.in.ua weeks (12 Sep-11 Oct)": live_days}
    rows = []
    for name, acc in models.items():
        r = {"model": name}
        for pn, dd in periods.items():
            r[pn] = acc.brier(dd)
        for pn, dd in periods.items():
            r[f"ECE {pn.split(' (')[0]}"] = acc.ece(dd)
        n, f, o = acc.top(D_conf)
        r["2026 top bin (>=0.8): n"], r["mean forecast"], r["observed"] = n, f, o
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model")
    print(t.round(4).to_string())
    t.to_csv(OUT / "step21b_final_scores.csv")

    print("\nPRIMARY: skill vs flat B4 = 1 - macroBrier(final)/macroBrier(B4), mean over +1...+24 h; joint day-block 95 % interval")
    for pn, dd in periods.items():
        d1 = m.bootstrap(m.skill, [fin.part(dd), b4.part(dd)], n_rep=1000)
        d7 = m.bootstrap(m.skill, [fin.part(dd), b4.part(dd)], n_rep=1000, block_days=7)
        print(f"  {pn:36s} {d1[0]:+.4f}  day-block [{d1[1]:+.4f}, {d1[2]:+.4f}]  week-block [{d7[1]:+.4f}, {d7[2]:+.4f}]  "
              f"({len(dd)} days)")
    print("  ablations (what each part adds), 2026, weekly refits on both sides:")
    for name in ("final without hour shift (weekly)", "final without alert_now (weekly)"):
        d1 = m.bootstrap(m.skill, [weekly[hp].part(D_conf), models[name].part(D_conf)], n_rep=1000)
        print(f"    full vs {name:36s}: {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}]")

    print("\nCALIBRATION: ECE difference final - B4 with a day-block 95 % interval; reliability table (2026)")
    rng = np.random.default_rng(0)
    for pn, dd in periods.items():
        reps = []
        for _ in range(500):
            dr = rng.choice(dd, len(dd))
            reps.append(fin.ece(dr) - b4.ece(dr))
        print(f"  {pn:36s} ECE final {fin.ece(dd):.4f} vs B4 {b4.ece(dd):.4f}: diff {fin.ece(dd) - b4.ece(dd):+.4f} "
              f"[{np.percentile(reps, 2.5):+.4f}, {np.percentile(reps, 97.5):+.4f}]")
    print("  bin        n_final   forecast  observed |  n_B4      forecast  observed")
    for (bn_, n1, f1, o1), (_, n2, f2, o2) in zip(fin.table(D_conf), b4.table(D_conf)):
        print(f"  {bn_}  {n1:9d}   {f1:.3f}    {o1:.3f}   | {n2:9d}   {f2:.3f}    {o2:.3f}")

    print("\nPER HORIZON (2026, macro Brier)")
    ph = {}
    for name in ("B4 flat (h7)", "final"):
        acc = models[name]
        ph[name] = [m.macro(acc.sh[:, D_conf, h], acc.ch[:, D_conf, h])[0] for h in range(24)]
    tab = pd.DataFrame(ph, index=[f"+{h + 1}h" for h in range(24)])
    tab["skill"] = 1 - tab["final"] / tab["B4 flat (h7)"]
    print(tab.round(4).T.to_string())
    tab.to_csv(OUT / "step21b_per_horizon.csv")

    print(f"\nw_h curve (last daily fit; effect of an alert running now, log-odds):")
    print("  " + " ".join(f"+{h + 1}:{v:.2f}" for h, v in enumerate(beta_last[24:])))
    print("hour profile s (last fit, log-odds relative to the level; Kyiv hour of the target time):")
    print("  " + " ".join(f"{k:02d}:{v:+.2f}" for k, v in enumerate(beta_last[:24])))
    print(f"  (mean of s = {beta_last[:24].mean():+.3f}: overall shift of the level)")

    section("LEAKAGE CHECK - final model on real histories (12 raions x 10 times)")
    units = P["units"]
    since = P["since"]
    from aaf import baselines as bl
    days_keys = sorted(betas)
    num_all, den_all = final.level_parts(P, hl)
    raw = np.divide(num_all, den_all, out=np.full(num_all.shape, np.nan), where=den_all > 0)
    nat_mean = np.nanmean(raw, axis=0)                                  # from all raions, bins before each origin
    from scipy.signal import lfilter
    from aaf.baselines import ontime_upto
    dcy = 0.5 ** (1.0 / (24.0 * hl))
    def forecaster(h, t):
        """Final model for one raion at origin t, recomputed from this raion's History only."""
        j = int(np.searchsorted(G, t))
        g = G[:j]
        x = np.where(COV[:j], (ontime_upto(h, g + final.H) - ontime_upto(h, g)) / final.H, 0.0)
        num = lfilter([1.0], [1.0, -dcy], x)[-1]
        den = lfilter([1.0], [1.0, -dcy], COV[:j].astype(float))[-1]
        lvl = (num + final.PRIOR_HOURS * nat_mean[j]) / (den + final.PRIOR_HOURS)
        b = betas[max(k for k in days_keys if k <= int((t - G[jE]) // final.DAY))]
        return final.predict(final.logit(np.array([lvl])), hod[j + np.arange(1, 25)][None, :],
                             np.array([float(h.state_at([t])[0])]), b)[0]
    # the forecaster must reproduce the evaluated predictions exactly (same model), then pass the leakage check
    sample = {u: P["hist"][u] for u in units[:: max(1, len(units) // 12)]}
    times = [int(G[ev_orig[i]]) for i in np.linspace(0, len(ev_orig) - 1, 10).astype(int)]
    problems = leakage.check(forecaster, sample, times)
    print(f"problems: {len(problems)}", problems[:3] if problems else "")
    LLp = final.smoothed_level(num_all, den_all)
    diffs = [abs(float(forecaster(P["hist"][u], tt)[0]) - float(final.predict(final.logit(LLp[[units.index(u)], int(np.searchsorted(G, tt))]),
             hod[int(np.searchsorted(G, tt)) + np.arange(1, 25)][None, :], S[[units.index(u)], int(np.searchsorted(G, tt))].astype(float),
             betas[max(k for k in days_keys if k <= int((tt - G[jE]) // final.DAY))])[0, 0])) for u in sample for tt in times]
    print(f"forecaster used in the check reproduces the evaluated model: max |difference| {max(diffs):.2e}")
    print(f"training labels strictly before each refit: asserted in every refit ({len(betas)} refits)")

    json.dump({"level_half_life_days": hl, "hour_half_life_days": hp, "prior_hours": final.PRIOR_HOURS,
               "beta": [float(x) for x in beta_last], "stage1": s1, "stage2": s2,
               "fitted_on_labels_before_utc": pd.Timestamp(G[ev_orig[-1]], tz="UTC").isoformat()},
              open(OUT / "step21b_final_params.json", "w"), indent=2)
    print(f"\ndone in {time.time() - t0:.0f} s; params -> logs/step21b_final_params.json (not committed)")


if __name__ == "__main__":
    main()
