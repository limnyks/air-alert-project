"""Step 11b: raion neighbours from HDX shapes (fixes M9) and an exact, calibrated clustering test (fixes M8).

Same data rule as Step 11: official source, alert starts before 2026-05-18 only.

Part A  neighbours: download HDX ukr_admin_boundaries.geojson.zip (~84 MB, once, local only), read the
        raion (adm2) layer, call two raions neighbours if their borders share points (coordinates rounded
        to ~10 cm; if that finds almost nothing, ~10 m). Writes config/raion_adjacency.csv.
Part B  H4 exact time-rescaling: Model C (daily/weekly Fourier + month levels) fitted as in Step 11;
        the cumulative intensity is accumulated minute by minute over off-time only; rescaled gaps
        between starts must be Exp(1) for a non-clustered process (KS test, CV = 1, 9.5 % gaps < 0.1).
Part C  calibration: for 10 units, simulate a NON-clustered process from that unit's fitted intensity
        (durations resampled from the unit's own alerts), run the identical pipeline, and check that the
        test does not reject. If it rejects on simulated data, the test is not trustworthy.
Part D  H3 rerun with the neighbour pairs.

Run from the project root:
    python scripts/step11b_neighbours_and_clustering.py | tee logs/step11b_output.txt
"""
import io
import json
import urllib.request
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from step05_inspect_data import section
from step06_diagnose_sources import MIN_NS, coverage
from step11_explore_hypotheses import EDA_END, HOUR_NS, DAY_NS, MIN_PAIR_DAYS, NULL_SHIFTS_DAYS, fourier, hourly, ns

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "data" / "processed" / "raion_episodes.csv"
CHECKS = ROOT / "logs" / "step10b_raion_checks.csv"
RAIONS = ROOT / "config" / "raions.csv"
GEO_URL = ("https://data.humdata.org/dataset/d23f529f-31e4-4021-a65b-13987e5cfb42/resource/"
           "681beb86-391b-4a08-8140-ca52e80fcdce/download/ukr_admin_boundaries.geojson.zip")
GEO_ZIP = ROOT / "data" / "raw" / "hdx" / "ukr_admin_boundaries.geojson.zip"
ADJ_OUT = ROOT / "config" / "raion_adjacency.csv"
UNIT_OUT = ROOT / "logs" / "step11b_unit_results.csv"
N_CALIBRATION_UNITS = 10
SEED = 20261011


# ---------------------------------------------------------------- A. neighbours

def load_adm2():
    if not GEO_ZIP.exists():
        GEO_ZIP.parent.mkdir(parents=True, exist_ok=True)
        print("downloading ukr_admin_boundaries.geojson.zip (~84 MB) ...")
        urllib.request.urlretrieve(GEO_URL, GEO_ZIP)
    print(f"{GEO_ZIP.relative_to(ROOT)}: {GEO_ZIP.stat().st_size:,} bytes")
    with zipfile.ZipFile(GEO_ZIP) as z:
        names = z.namelist()
        print("files in zip:", names)
        layer = next(n for n in names if "adm2" in n.lower() or "admin2" in n.lower())
        print("raion layer:", layer)
        return json.load(io.TextIOWrapper(z.open(layer), encoding="utf-8"))


def rings(geometry):
    if geometry["type"] == "Polygon":
        return geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        return [ring for polygon in geometry["coordinates"] for ring in polygon]
    return []


def neighbours(geo, pcode_to_unit, decimals):
    owners = defaultdict(set)
    for f in geo["features"]:
        props = f["properties"]
        key = next(k for k in props if "adm2" in k.lower() and "pcode" in k.lower())
        unit = pcode_to_unit.get(props[key])
        if unit is None:
            continue
        for ring in rings(f["geometry"]):
            for lon, lat, *_ in ring:
                owners[(round(lon, decimals), round(lat, decimals))].add(unit)
    shared = defaultdict(int)
    for units in owners.values():
        if len(units) > 1:
            u = sorted(units)
            for i in range(len(u)):
                for j in range(i + 1, len(u)):
                    shared[(u[i], u[j])] += 1
    return {pair: n for pair, n in shared.items() if n >= 2}


def part_a(units):
    section("PART A - raion neighbours from HDX shapes")
    geo = load_adm2()
    props0 = geo["features"][0]["properties"]
    print(f"features: {len(geo['features'])}; property keys: {sorted(props0)}")
    rai = pd.read_csv(RAIONS)
    rai["unit"] = np.where(rai["oblast_en"] == "Kyiv City", "Kyiv City", rai["oblast_en"] + " / " + rai["raion_en"].fillna(""))
    pc = {r["hdx_pcode"]: r["unit"] for _, r in rai.dropna(subset=["hdx_pcode"]).iterrows() if r["unit"] in set(units["unit"])}
    name_key = next((k for k in props0 if "adm2" in k.lower() and k.lower().endswith("name1")), None)
    pcode_key = next(k for k in props0 if "adm2" in k.lower() and "pcode" in k.lower())
    kyiv = [f["properties"][pcode_key] for f in geo["features"] if name_key and f["properties"].get(name_key) == "Київ"]
    print(f"Kyiv City HDX code from the 'Київ' feature: {kyiv}")
    if len(kyiv) == 1 and "Kyiv City" in set(units["unit"]):
        pc[kyiv[0]] = "Kyiv City"
    print(f"modeled units with an HDX shape: {len(pc)} of {len(units)}")
    for decimals in (6, 4):
        pairs = neighbours(geo, pc, decimals)
        deg = pd.Series(dict(Counter_from_pairs(pairs)))
        print(f"coordinates rounded to {decimals} decimals: {len(pairs)} neighbour pairs; neighbours per unit "
              f"min {deg.min() if len(deg) else 0}, median {deg.median() if len(deg) else 0}, max {deg.max() if len(deg) else 0}; "
              f"units without neighbours: {len(pc) - len(deg)}")
        if len(pairs) >= len(pc):  # plausible: on average at least 2 neighbours per unit (each pair counted once)
            break
    adj = pd.DataFrame([{"unit_a": a, "unit_b": b, "shared_points": n} for (a, b), n in sorted(pairs.items())])
    adj.to_csv(ADJ_OUT, index=False)
    print(f"saved {len(adj)} pairs -> {ADJ_OUT.relative_to(ROOT)} (derived from HDX COD-AB, CC BY-IGO)")
    return adj


def Counter_from_pairs(pairs):
    c = defaultdict(int)
    for a, b in pairs:
        c[a] += 1
        c[b] += 1
    return c


# ---------------------------------------------------------------- B/C. exact time-rescaling

def model_c_rates(y, off, times):
    """Fit Model C (Fourier + month levels); return the start intensity per off-hour for every hour."""
    m = off > 0
    months = pd.get_dummies(times.tz_convert("UTC").tz_localize(None).to_period("M").astype(str), drop_first=True).to_numpy(float)
    X_all = np.hstack([sm.add_constant(fourier(times).to_numpy(), has_constant="add"), months])
    fit = sm.GLM(y[m], X_all[m], family=sm.families.Poisson(), offset=np.log(off[m])).fit()
    # intensity from the model formula for EVERY hour (also hours that were fully under alert), so a
    # simulation from it does not copy the real alert timing (which would fake clustering)
    return np.exp(X_all @ fit.params)


def rescaled_gaps(starts_ns, off_minute, lam_hour, h0_ns):
    """Cumulative intensity minute by minute over off-time; value at each start; gaps between starts."""
    rate_min = np.repeat(lam_hour / 60.0, 60)
    cum = np.concatenate([[0.0], np.cumsum(rate_min * off_minute)])
    m_idx = (starts_ns - h0_ns) // MIN_NS        # intensity accumulated up to the start's minute
    m_idx = m_idx[(m_idx >= 0) & (m_idx < len(cum))]  # guard: only starts inside the grid
    gaps = np.diff(cum[m_idx])
    return gaps[gaps > 0]


def test_unit(eps, t_from):
    y, off, times, st, idx, h0_ns = hourly(eps, t_from)
    n_min = len(y) * 60
    off_minute = 1.0 - coverage(eps["start_ns"].to_numpy(), eps["end_ns"].to_numpy(), h0_ns, n_min)
    lam = model_c_rates(y, off, times)
    gaps = rescaled_gaps(st, off_minute, lam, h0_ns)
    ks = stats.kstest(gaps, "expon")
    return {"n_gaps": len(gaps), "ks_stat": ks.statistic, "ks_p": ks.pvalue, "cv": gaps.std() / gaps.mean(),
            "short_gap_share": float(np.mean(gaps < 0.1))}, (lam, h0_ns, len(y))


def simulate(lam_hour, h0_ns, n_hours, durations_min, rng):
    """Non-clustered process: starts from the fitted intensity while off; durations resampled."""
    rate_min = np.repeat(lam_hour / 60.0, 60)
    cum = np.concatenate([[0.0], np.cumsum(rate_min)])
    t, starts, ends = 0, [], []
    while True:
        target = cum[t] + rng.exponential()
        m = int(np.searchsorted(cum, target))      # first minute boundary where the intensity is exceeded
        if m >= len(cum):
            break
        d = max(1, int(round(rng.choice(durations_min))))
        starts.append(m - 1)
        ends.append(m - 1 + d)
        t = m - 1 + d
        if t >= len(cum) - 1:
            break
    s = h0_ns + np.array(starts, dtype=np.int64) * MIN_NS
    e = h0_ns + np.array(ends, dtype=np.int64) * MIN_NS
    return pd.DataFrame({"start_ns": s, "end_ns": e})


# ---------------------------------------------------------------- D. H3

def h3(adj, units, starts_by_unit):
    sw = units.set_index("unit")["switch"].to_dict()
    obl = units.set_index("unit")["oblast"].to_dict()
    rows = []
    for a, b in zip(adj["unit_a"], adj["unit_b"]):
        if pd.isna(sw.get(a)) or pd.isna(sw.get(b)):
            continue
        w0 = max(pd.Timestamp(sw[a], tz="UTC"), pd.Timestamp(sw[b], tz="UTC")) + pd.Timedelta(days=1)
        if (EDA_END - w0).days < MIN_PAIR_DAYS:
            continue
        lo, length = ns(w0), ns(EDA_END) - ns(w0)
        for src, dst in ((a, b), (b, a)):
            sa, sb = starts_by_unit[src], starts_by_unit[dst]
            sa, sb = sa[(sa >= lo) & (sa < lo + length)], sb[(sb >= lo) & (sb < lo + length)]
            if len(sa) < 10 or len(sb) < 10:
                continue

            def rates(src_starts):
                src_starts = np.sort(src_starts)
                i = np.searchsorted(src_starts, sb, side="right")
                before = np.where(i > 0, sb - src_starts[np.clip(i - 1, 0, len(src_starts) - 1)], np.inf)
                after = np.where(i < len(src_starts), src_starts[np.clip(i, 0, len(src_starts) - 1)] - sb, np.inf)
                co = np.minimum(before, after) <= MIN_NS
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
    ep = pd.read_csv(EPISODES)
    ep = ep[(ep["source"] == "official") & ep["unit"].isin(units["unit"])].copy()
    for col in ("start_utc", "end_utc"):
        t = pd.to_datetime(ep[col], utc=True, format="ISO8601")
        ep[col.replace("_utc", "_ns")] = t.dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")

    section("SETUP")
    print(f"modeled units: {len(units)}; official data before {EDA_END}")
    adj = part_a(units)

    section("PART B - H4 exact time-rescaling under Model C (real data)")
    rows, fitted, starts_by_unit = [], {}, {}
    for _, u in units.iterrows():
        t_from = pd.Timestamp(u["train_from"], tz="UTC")
        e = ep[ep["unit"] == u["unit"]]
        starts_by_unit[u["unit"]] = e["start_ns"].to_numpy()
        w = e[(e["start_ns"] >= ns(t_from)) & (e["start_ns"] < ns(EDA_END))]
        if len(w) < 30:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, fit_info = test_unit(e, t_from)
        dur = ((w["end_ns"] - w["start_ns"]) / MIN_NS).to_numpy()
        fitted[u["unit"]] = (fit_info, dur, len(w))
        rows.append({"unit": u["unit"], **res})
    r = pd.DataFrame(rows)
    r.to_csv(UNIT_OUT, index=False)
    print(f"units tested: {len(r)}")
    print(f"share of units with KS p < 0.001: {(r['ks_p'] < 0.001).mean():.3f}; median KS {r['ks_stat'].median():.3f}; "
          f"median CV {r['cv'].median():.2f} (Poisson 1); median share of gaps < 0.1: {r['short_gap_share'].median():.3f} (Poisson 0.095)")

    section("PART C - calibration: same pipeline on simulated NON-clustered data")
    rng = np.random.default_rng(SEED)
    by_starts = sorted(fitted, key=lambda k: fitted[k][2])
    picks = [by_starts[int(i)] for i in np.linspace(0, len(by_starts) - 1, N_CALIBRATION_UNITS)]
    cal = []
    for unit in picks:
        (lam, h0_ns, n_hours), dur, n_real = fitted[unit]
        sim = simulate(lam, h0_ns, n_hours, dur, rng)
        t_from = pd.Timestamp(h0_ns, tz="UTC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, _ = test_unit(sim, t_from)
        real = r.set_index("unit").loc[unit]
        cal.append({"unit": unit, "real_starts": n_real, "sim_starts": len(sim),
                    "sim_ks_p": res["ks_p"], "sim_cv": res["cv"], "sim_short": res["short_gap_share"],
                    "real_ks_p": real["ks_p"], "real_cv": real["cv"], "real_short": real["short_gap_share"]})
    c = pd.DataFrame(cal)
    print(c.round(4).to_string(index=False))
    print(f"\nsimulated units rejected at p < 0.001: {(c['sim_ks_p'] < 0.001).sum()} of {len(c)} "
          f"(a valid test should reject ~0); real units rejected: {(c['real_ks_p'] < 0.001).sum()} of {len(c)}")

    section("PART D - H3 neighbouring raions (raion era only)")
    n = h3(adj, units, starts_by_unit)
    if n.empty:
        print("no pairs with enough data")
        return
    n["co_ratio"] = n["co_share"] / n["co_null"].replace(0, np.nan)
    n["lead_ratio"] = n["lead_share"] / n["lead_null"].replace(0, np.nan)
    print(f"directed pairs analysed: {len(n)}; median common window {n['days'].median():.0f} days")
    print(n.groupby("same_oblast")[["co_share", "co_null", "co_ratio", "lead_share", "lead_null", "lead_ratio"]]
          .median().round(3).to_string())
    print(f"share of directed pairs with lead ratio > 1.5: {(n['lead_ratio'] > 1.5).mean():.3f}")


if __name__ == "__main__":
    main()
