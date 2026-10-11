"""Step 11: exploratory hypothesis tests (H1, H4, H3, H2) on data that cannot bias validation or tests.

Data rule (protocol section 12): only the official source and only alert starts before
2026-05-18 00:00 UTC (start of the validation folds). Each unit from its `train_from`.
Units: the 104 modeled units (logs/step10b_raion_checks.csv, excluded == False).

H1  daily / weekly pattern: Poisson GLM of hourly start counts with exposure = off-time in the hour
    (a start can only happen while off). Model A constant vs Model B + Fourier terms
    (3 daily + 2 weekly harmonics, Kyiv local time): likelihood-ratio test, df = 10.
H4  clustering beyond the daily pattern: Pearson dispersion of Model B and of Model C
    (B + month levels, so slow rate changes are not mistaken for clustering - IDS-paper warning);
    time-rescaling under Model C: rescaled gaps between starts should be Exp(1) (KS test, CV = 1).
H3  neighbours: for adjacent raions (HDX boundary lines), only after both oblasts' switch to raion
    alerts: how often a start in B comes within 1 min of a start in A (co-announcement) or 1-60 min
    after one (lead), vs a null where A's starts are shifted by whole days (keeps time of day).
H2  two kinds of alerts: per unit, log-normal vs Weibull vs 2-component log-normal mixture for
    durations (official ends are all known); AIC.

Outputs: printed summary + logs/step11_unit_results.csv (one row per unit, aggregated numbers).

Run from the project root (after `pip install scipy statsmodels`):
    python scripts/step11_explore_hypotheses.py | tee logs/step11_output.txt
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from step05_inspect_data import section
from step06_diagnose_sources import MIN_NS, coverage

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "data" / "processed" / "raion_episodes.csv"
CHECKS = ROOT / "logs" / "step10b_raion_checks.csv"
RAIONS = ROOT / "config" / "raions.csv"
HDX = ROOT / "data" / "raw" / "hdx" / "ukr_admin_boundaries.xlsx"
UNIT_OUT = ROOT / "logs" / "step11_unit_results.csv"

EDA_END = pd.Timestamp("2026-05-18", tz="UTC")
TZ = "Europe/Kyiv"
HOUR_NS = 60 * MIN_NS
DAY_NS = 24 * HOUR_NS
MIN_STARTS = 30          # fewer starts in the window -> unit skipped for H1/H4
MIN_EPISODES_DUR = 200   # fewer episodes -> unit skipped for H2
MIN_PAIR_DAYS = 30       # shorter common raion-era window -> pair skipped for H3
NULL_SHIFTS_DAYS = range(1, 8)


def ns(ts):
    return ts.tz_convert("UTC").tz_localize(None).as_unit("ns").value


# ---------------------------------------------------------------- H1 / H4

def fourier(local_times):
    hod = local_times.hour.to_numpy() + 0.5
    how = local_times.dayofweek.to_numpy() * 24 + hod
    cols = {}
    for k in (1, 2, 3):
        cols[f"d_sin{k}"], cols[f"d_cos{k}"] = np.sin(2 * np.pi * k * hod / 24), np.cos(2 * np.pi * k * hod / 24)
    for k in (1, 2):
        cols[f"w_sin{k}"], cols[f"w_cos{k}"] = np.sin(2 * np.pi * k * how / 168), np.cos(2 * np.pi * k * how / 168)
    return pd.DataFrame(cols)


def hourly(eps, t_from):
    """Hourly start counts and off-exposure (fraction of the hour without alert) in [t_from, EDA_END)."""
    h0 = t_from.floor("h")
    n_h = int((EDA_END - h0) / pd.Timedelta(hours=1))
    s = eps["start_ns"].to_numpy()
    e = eps["end_ns"].to_numpy()
    cov = coverage(s, e, ns(h0), n_h * 60)
    off = 1 - cov.reshape(n_h, 60).mean(axis=1)
    st = s[(s >= ns(t_from)) & (s < ns(EDA_END))]
    idx = (st - ns(h0)) // HOUR_NS
    y = np.bincount(idx, minlength=n_h).astype(float)
    times = pd.date_range(h0, periods=n_h, freq="h").tz_convert(TZ)
    return y, off, times, st, idx, ns(h0)


def h1_h4(y, off, times, st, idx, h0_ns):
    m = off > 0
    F = fourier(times)
    X_a = np.ones((m.sum(), 1))
    X_b = sm.add_constant(F.to_numpy()[m], has_constant="add")
    months = pd.get_dummies(times.tz_convert("UTC").tz_localize(None).to_period("M").astype(str), drop_first=True).to_numpy(float)
    X_c = np.hstack([X_b, months[m]])
    offset = np.log(off[m])
    fit = lambda X: sm.GLM(y[m], X, family=sm.families.Poisson(), offset=offset).fit()
    fa, fb, fc = fit(X_a), fit(X_b), fit(X_c)
    lr = 2 * (fb.llf - fa.llf)
    # time rescaling under Model C (intensity per off-hour, assumed uniform inside the hour)
    lam = np.zeros(len(y))
    lam[m] = fc.fittedvalues / off[m]
    contrib = lam * off
    c_prev = np.concatenate([[0.0], np.cumsum(contrib)[:-1]])
    frac = ((st - h0_ns) % HOUR_NS) / HOUR_NS
    big_l = c_prev[idx] + contrib[idx] * frac
    gaps = np.diff(big_l)
    gaps = gaps[gaps > 0]
    ks = stats.kstest(gaps, "expon")
    return {"starts": int(y.sum()), "hours": int(m.sum()),
            "h1_lr": lr, "h1_p": stats.chi2.sf(lr, X_b.shape[1] - 1),
            "fano_raw": y[m].var() / y[m].mean() if y[m].mean() > 0 else np.nan,
            "disp_B": fb.pearson_chi2 / fb.df_resid, "disp_C": fc.pearson_chi2 / fc.df_resid,
            "ks_C": ks.statistic, "ks_p": ks.pvalue, "cv_rescaled": gaps.std() / gaps.mean(),
            "short_gap_share": float(np.mean(gaps < 0.1))}


# ---------------------------------------------------------------- H2

def h2(durations_min):
    d = durations_min[durations_min > 0]
    x = np.log(d)
    jac = -x.sum()  # log-density of durations = density of log(d) - log(d)
    mu, sd = x.mean(), x.std()
    ll_ln = stats.norm.logpdf(x, mu, sd).sum() + jac
    c, _, scale = stats.weibull_min.fit(d, floc=0)
    ll_wb = stats.weibull_min.logpdf(d, c, 0, scale).sum()
    # 2-component Gaussian mixture on log-durations (EM)
    w, m1, m2 = 0.5, np.quantile(x, 0.25), np.quantile(x, 0.75)
    s1 = s2 = sd
    for _ in range(300):
        p1 = w * stats.norm.pdf(x, m1, s1)
        p2 = (1 - w) * stats.norm.pdf(x, m2, s2)
        r = p1 / np.maximum(p1 + p2, 1e-300)
        w = r.mean()
        m1, m2 = (r * x).sum() / r.sum(), ((1 - r) * x).sum() / (1 - r).sum()
        s1 = max(np.sqrt((r * (x - m1) ** 2).sum() / r.sum()), 1e-3)
        s2 = max(np.sqrt(((1 - r) * (x - m2) ** 2).sum() / (1 - r).sum()), 1e-3)
    ll_mix = np.log(w * stats.norm.pdf(x, m1, s1) + (1 - w) * stats.norm.pdf(x, m2, s2) + 1e-300).sum() + jac
    aic = {"lognormal": 4 - 2 * ll_ln, "weibull": 4 - 2 * ll_wb, "mixture": 10 - 2 * ll_mix}
    best_single = min(aic["lognormal"], aic["weibull"])
    lo, hi = sorted([(np.exp(m1), w), (np.exp(m2), 1 - w)])
    return {"dur_n": len(d), "dur_median": float(np.median(d)), "best_dur_model": min(aic, key=aic.get),
            "mixture_gain_aic": best_single - aic["mixture"],
            "mix_short_median": lo[0], "mix_short_weight": lo[1], "mix_long_median": hi[0]}


# ---------------------------------------------------------------- H3

def h3(units, starts_by_unit):
    lines = pd.read_excel(HDX, sheet_name="ukr_adminlines")
    pc = units.dropna(subset=["hdx_pcode"]).set_index("hdx_pcode")["unit"].to_dict()
    pairs = {tuple(sorted((pc[a], pc[b]))) for a, b in zip(lines["left_pcod"], lines["right_pcod"])
             if a in pc and b in pc and a != b}
    print(f"adjacent pairs of modeled raions (from HDX boundary lines): {len(pairs)}")
    sw = units.set_index("unit")["switch"].to_dict()
    obl = units.set_index("unit")["oblast"].to_dict()
    rows = []
    for a, b in pairs:
        if pd.isna(sw.get(a)) or pd.isna(sw.get(b)):
            continue
        w0 = max(pd.Timestamp(sw[a], tz="UTC"), pd.Timestamp(sw[b], tz="UTC")) + pd.Timedelta(days=1)
        if (EDA_END - w0).days < MIN_PAIR_DAYS:
            continue
        lo, length = ns(w0), ns(EDA_END) - ns(w0)
        for src, dst in ((a, b), (b, a)):
            sa = starts_by_unit[src]
            sb = starts_by_unit[dst]
            sa, sb = sa[(sa >= lo) & (sa < lo + length)], sb[(sb >= lo) & (sb < lo + length)]
            if len(sa) < 10 or len(sb) < 10:
                continue

            def rates(src_starts):
                src_starts = np.sort(src_starts)
                i = np.searchsorted(src_starts, sb, side="right")
                before = np.where(i > 0, sb - src_starts[np.clip(i - 1, 0, len(src_starts) - 1)], np.inf)
                after = np.where(i < len(src_starts), src_starts[np.clip(i, 0, len(src_starts) - 1)] - sb, np.inf)
                co = (np.minimum(before, after) <= MIN_NS)
                lead = ~co & (before > MIN_NS) & (before <= 60 * MIN_NS)
                return co.mean(), lead.mean()

            co, lead = rates(sa)
            null = np.array([rates(lo + (sa - lo + k * DAY_NS) % length) for k in NULL_SHIFTS_DAYS])
            rows.append({"from": src, "to": dst, "same_oblast": obl[src] == obl[dst], "days": length / DAY_NS,
                         "co_share": co, "co_null": null[:, 0].mean(), "lead_share": lead, "lead_null": null[:, 1].mean()})
    return pd.DataFrame(rows)


def main():
    units = pd.read_csv(CHECKS)
    units = units[~units["excluded"]].copy()
    rai = pd.read_csv(RAIONS)
    rai["unit"] = np.where(rai["oblast_en"] == "Kyiv City", "Kyiv City", rai["oblast_en"] + " / " + rai["raion_en"].fillna(""))
    units = units.merge(rai[["unit", "hdx_pcode"]], on="unit", how="left")
    ep = pd.read_csv(EPISODES)
    ep = ep[(ep["source"] == "official") & ep["unit"].isin(units["unit"])].copy()
    for col in ("start_utc", "end_utc"):
        ep[col] = pd.to_datetime(ep[col], utc=True, format="ISO8601")
    ep["start_ns"] = ep["start_utc"].dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")
    ep["end_ns"] = ep["end_utc"].dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")

    section("SETUP")
    print(f"modeled units: {len(units)}; exploration window: each unit's train_from -> {EDA_END} (official only)")

    results, starts_by_unit, pooled = [], {}, []
    for _, u in units.iterrows():
        t_from = pd.Timestamp(u["train_from"], tz="UTC")
        e = ep[ep["unit"] == u["unit"]]
        y, off, times, st, idx, h0_ns = hourly(e, t_from)
        assert st.max() < ns(EDA_END) if len(st) else True, "data after the exploration window used"
        starts_by_unit[u["unit"]] = e["start_ns"].to_numpy()
        row = {"unit": u["unit"], "oblast": u["oblast"], "train_from": u["train_from"]}
        pooled.append(pd.DataFrame({"hod": times.hour, "dow": times.dayofweek, "y": y, "off": off}))
        if y.sum() >= MIN_STARTS:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    row.update(h1_h4(y, off, times, st, idx, h0_ns))
                row["fit_warnings"] = len(caught)
            except Exception as error:  # e.g. singular design for a unit always under alert in a month
                row["fit_error"] = f"{type(error).__name__}: {error}"[:120]
        dur = e[(e["start_utc"] >= t_from) & (e["start_utc"] < EDA_END)]
        dmin = ((dur["end_ns"] - dur["start_ns"]) / MIN_NS).to_numpy()
        if len(dmin) >= MIN_EPISODES_DUR:
            row.update(h2(dmin))
        results.append(row)
    r = pd.DataFrame(results)
    r.to_csv(UNIT_OUT, index=False)
    t = r.dropna(subset=["h1_lr"])

    section("H1 - daily / weekly pattern of alert starts")
    errors = r["fit_error"].dropna() if "fit_error" in r else pd.Series(dtype=str)
    warned = int((r.get("fit_warnings", pd.Series(dtype=float)).fillna(0) > 0).sum())
    print(f"units tested: {len(t)}; skipped (< {MIN_STARTS} starts or fit error): {len(r) - len(t)}; "
          f"fit errors: {len(errors)}; units with fitting warnings (e.g. months without starts): {warned}")
    if len(errors):
        print("fit errors:", r.loc[errors.index, ["unit", "fit_error"]].values.tolist())
    print(f"share of units with LR-test p < 0.001: {(t['h1_p'] < 0.001).mean():.3f}; median LR statistic: {t['h1_lr'].median():.1f} (df = 10)")
    p = pd.concat(pooled)
    by_h = p.groupby("hod")[["y", "off"]].sum()
    by_d = p.groupby("dow")[["y", "off"]].sum()
    print("\npooled start rate per off-hour by Kyiv hour of day:")
    print((by_h["y"] / by_h["off"]).round(3).to_string())
    print("\npooled start rate per off-hour by weekday (0 = Monday):")
    print((by_d["y"] / by_d["off"]).round(3).to_string())

    section("H4 - clustering beyond the daily pattern and slow changes")
    print(f"median dispersion: raw Fano {t['fano_raw'].median():.2f}; Model B (pattern) {t['disp_B'].median():.2f}; "
          f"Model C (pattern + month levels) {t['disp_C'].median():.2f}   (1 = Poisson, > 1 = clustering)")
    print(f"share of units with Model C dispersion > 1.5: {(t['disp_C'] > 1.5).mean():.3f}")
    print(f"time-rescaling under Model C: share of units with KS p < 0.001: {(t['ks_p'] < 0.001).mean():.3f}; "
          f"median KS statistic {t['ks_C'].median():.3f}; median CV of rescaled gaps {t['cv_rescaled'].median():.2f} (Poisson: 1); "
          f"median share of rescaled gaps < 0.1: {t['short_gap_share'].median():.3f} (Poisson: 0.095)")

    section("H3 - neighbouring raions (raion era only, after both oblasts' switch)")
    n = h3(units, starts_by_unit)
    if n.empty:
        print("no pairs with enough data")
    else:
        n["co_ratio"] = n["co_share"] / n["co_null"].replace(0, np.nan)
        n["lead_ratio"] = n["lead_share"] / n["lead_null"].replace(0, np.nan)
        print(f"directed pairs analysed: {len(n)}; median common window {n['days'].median():.0f} days")
        print(n.groupby("same_oblast")[["co_share", "co_null", "co_ratio", "lead_share", "lead_null", "lead_ratio"]]
              .median().round(3).to_string())
        print(f"share of directed pairs with lead ratio > 1.5: {(n['lead_ratio'] > 1.5).mean():.3f}")

    section("H2 - alert durations: one distribution or two kinds?")
    d = r.dropna(subset=["best_dur_model"])
    print(f"units analysed: {len(d)} (>= {MIN_EPISODES_DUR} episodes); best model by AIC: {d['best_dur_model'].value_counts().to_dict()}")
    print(f"share of units where the mixture beats the best single distribution by > 10 AIC: {(d['mixture_gain_aic'] > 10).mean():.3f}")
    print(f"mixture components (medians over units): short {d['mix_short_median'].median():.1f} min "
          f"(weight {d['mix_short_weight'].median():.2f}), long {d['mix_long_median'].median():.1f} min; "
          f"overall median duration {d['dur_median'].median():.1f} min")
    print(f"\nper-unit results -> {UNIT_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
