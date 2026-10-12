"""Step 15: build the raion map page (docs/map.html) from the committed live forecast and the final-test replay.

Main view: live forecast for +1 h / +6 h / +24 h (forecasts/horizons_<latest>.csv/.json, step17, D44). Replay: for
each of the 22 post-reform test days, the B3 forecast made at 06:00 Kyiv time (the tested model) and the alerts
that actually happened in the next 24 h.
Shapes: HDX COD-AB adm2 (raions) and adm1 (oblast borders), projected and simplified here; plain SVG, no
external map service. Page follows the Artifact contract (no doctype/html/head/body tags).

Run from the project root:
    python scripts/step15_build_map.py
"""
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import config  # noqa: E402
from aaf.baselines import ontime_upto  # noqa: E402
from aaf.data import load_histories, ns  # noqa: E402
from aaf.evaluate import combined_history  # noqa: E402

GEO = config.ROOT / "data" / "raw" / "hdx" / "ukr_admin_boundaries.geojson.zip"
OUT = config.ROOT / "docs" / "map.html"
W = 1000.0
LAT0 = 48.5


def project_all(features):
    xs, ys = [], []
    for f in features:
        for ring in rings(f["geometry"]):
            for lon, lat, *_ in ring:
                xs.append(lon * math.cos(math.radians(LAT0)))
                ys.append(-lat)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    k = W / (x1 - x0)
    return lambda lon, lat: ((lon * math.cos(math.radians(LAT0)) - x0) * k, (-lat - y0) * k), (y1 - y0) * k


def rings(g):
    if g["type"] == "Polygon":
        return g["coordinates"]
    if g["type"] == "MultiPolygon":
        return [r for p in g["coordinates"] for r in p]
    return []


def path(geom, proj, tol=0.6):
    parts = []
    for ring in rings(geom):
        pts, last = [], None
        for lon, lat, *_ in ring:
            x, y = proj(lon, lat)
            if last is None or abs(x - last[0]) + abs(y - last[1]) >= tol:
                pts.append((x, y))
                last = (x, y)
        if len(pts) >= 3:
            parts.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z")
    return "".join(parts)


def main():
    fc_csv = sorted((config.ROOT / "forecasts").glob("horizons_*.csv"))[-1]
    meta = json.loads(fc_csv.with_suffix(".json").read_text())
    fc = pd.read_csv(fc_csv)
    with zipfile.ZipFile(GEO) as z:
        adm2 = json.load(io.TextIOWrapper(z.open("ukr_admin2.geojson"), encoding="utf-8"))["features"]
        adm1 = json.load(io.TextIOWrapper(z.open("ukr_admin1.geojson"), encoding="utf-8"))["features"]
    proj, height = project_all(adm1)

    # replay: 22 test days, forecast at 06:00 Kyiv (03:00 UTC), outcome = next 24 h
    units = fc["unit"].tolist()
    off, aic = load_histories("official", units), load_histories("alerts_in_ua", units)
    comb = {u: combined_history(off[u], aic[u]) for u in units}
    days = pd.date_range("2026-09-19 03:00", "2026-10-10 03:00", freq="D", tz="UTC")
    H, D = config.HOUR_NS, 24 * config.HOUR_NS
    replay = {"days": [d.tz_convert(config.TZ).strftime("%d %b") for d in days], "units": {}}
    for _, r in fc.iterrows():
        h = comb[r["unit"]]
        t = np.array([ns(d) for d in days], dtype=np.int64)
        p = (ontime_upto(h, t) - ontime_upto(h, t - 7 * D)) / (7 * D)
        actual = (ontime_upto(h, t + D) - ontime_upto(h, t)) / D
        replay["units"][r["hdx_pcode"]] = {"p": np.round(p, 3).tolist(), "a": np.round(actual, 3).tolist()}
    adm2_obl = {f["properties"]["adm2_pcode"]: f["properties"]["adm1_pcode"] for f in adm2}

    info = {r["hdx_pcode"]: {"name": r["raion_uk"] if isinstance(r["raion_uk"], str) else "Київ", "p": [round(float(r[c]), 3) for c in ("p_1h", "p_6h", "p_12h", "p_24h")],
                             "ex": bool(r["excluded"]), "obl": r["unit"].split(" / ")[0], "op": adm2_obl[r["hdx_pcode"]]}
            for _, r in fc.iterrows()}
    obl = {}
    for f in adm1:
        op = f["properties"]["adm1_pcode"]
        mods = [pc for pc, i in info.items() if i["op"] == op and not i["ex"]]
        n_all = sum(1 for i in info.values() if i["op"] == op)
        if not n_all:
            continue
        o = {"name": f["properties"]["adm1_name1"], "n": len(mods), "n_all": n_all}
        if mods:
            o["p"] = np.round(np.mean([info[pc]["p"] for pc in mods], axis=0), 3).tolist()
            o["rp"] = np.round(np.mean([replay["units"][pc]["p"] for pc in mods], axis=0), 3).tolist()
            o["ra"] = np.round(np.mean([replay["units"][pc]["a"] for pc in mods], axis=0), 3).tolist()
            o["rhit"] = [int(sum(replay["units"][pc]["a"][d] > 0 for pc in mods)) for d in range(len(days))]
        obl[op] = o
    def center(f):
        pr = f["properties"]
        if pr.get("center_lon") is not None and pr.get("center_lat") is not None:
            return proj(pr["center_lon"], pr["center_lat"])
        pts = [proj(lon, lat) for ring in rings(f["geometry"]) for lon, lat, *_ in ring]   # fallback: vertex mean
        return (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))
    labels = "".join('<text class="ol" x="{:.0f}" y="{:.0f}">{}</text>'.format(*center(f), f["properties"]["adm1_name1"])
                     for f in adm1 if f["properties"]["adm1_pcode"] in obl)
    shapes = []
    for f in adm2:
        pc, pr = f["properties"]["adm2_pcode"], f["properties"]
        kind = "ex" if pc in info and info[pc]["ex"] else ("m" if pc in info else "nd")
        label = (info[pc]["name"] if pc in info else pr["adm2_name1"]) + ", " + pr["adm1_name1"]
        shapes.append(f'<path class="r {kind}" data-pc="{pc}" data-op="{pr["adm1_pcode"]}" data-label="{label}" d="{path(f["geometry"], proj)}"/>')
    borders = "".join(f'<path class="ob" d="{path(f["geometry"], proj, 0.9)}"/>' for f in adm1)
    mix = {m["unit_name"]: m for m in meta["observed_last_7_days_by_oblast"]}
    page = TEMPLATE.replace("__H__", f"{height:.0f}").replace("__SHAPES__", "".join(shapes)).replace("__BORDERS__", borders).replace("__LABELS__", labels)
    page = page.replace("__DATA__", json.dumps({"info": info, "obl": obl, "replay": replay, "meta": meta, "mix": mix}, ensure_ascii=False))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"saved -> {OUT.relative_to(config.ROOT)} ({OUT.stat().st_size/1e6:.2f} MB); raion shapes {len(shapes)}; forecast file {fc_csv.name}")


TEMPLATE = r"""<title>Raion Alert Outlook</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
/* Layout: one wide map with a slim control column; replay controls under the map */
:root{--bg:#f3f4f6;--panel:#ffffff;--ink:#1d2330;--muted:#5f6878;--line:#c9ced8;--nodata:#dfe2e8;
--p0:#ffffff;--p1:#f8d3c8;--p2:#e3221b;--p3:#e5523a;--p4:#6b0f1a;--hit:#111827;--border:#1b2230;
--display:"Bricolage Grotesque",Georgia,serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14171d;--panel:#1c2028;--ink:#e6e8ec;--muted:#9aa3b2;--line:#3a404c;--nodata:#2a2f39;
--p0:#ffffff;--p1:#4f2a27;--p2:#e3221b;--p3:#b8261b;--p4:#6b0f1a;--border:#f2f4f7;--hit:#f5f6f8;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14171d;--panel:#1c2028;--ink:#e6e8ec;--muted:#9aa3b2;--line:#3a404c;--nodata:#2a2f39;
--p0:#ffffff;--p1:#4f2a27;--p2:#e3221b;--p3:#b8261b;--p4:#6b0f1a;--border:#f2f4f7;--hit:#f5f6f8;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body)}
.wrap{max-width:1180px;margin:0 auto;padding-inline:16px;padding-block:20px 32px;display:grid;gap:16px}
h1{font:600 clamp(24px,3.2vw,34px)/1.1 var(--display);margin:0;text-wrap:balance}
.sub{color:var(--muted);margin:4px 0 0;max-width:70ch}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
@media (max-width:860px){.grid{grid-template-columns:minmax(0,1fr)}}
.map{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px;min-width:0}
svg{width:100%;height:auto;display:block}
.r{stroke:var(--panel);stroke-width:.4;cursor:pointer}
.lvl-oblast .r{stroke-width:0}
.r.nd{fill:url(#hatch-nd)}.r.ex{fill:url(#hatch-ex)}
.r:hover,.r:focus{stroke:var(--ink);stroke-width:1.4}
.r.hit{stroke:var(--hit);stroke-width:1.6}
.ob{fill:none;stroke:var(--border);stroke-width:2.2;stroke-linejoin:round;pointer-events:none}
.ol{font:600 11px var(--body);fill:var(--ink);paint-order:stroke;stroke:var(--panel);stroke-width:3px;text-anchor:middle;pointer-events:none;display:none}
.lvl-oblast .ol{display:block}
.side{display:grid;gap:12px;align-content:start;min-width:0}
.box{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.k{font:500 11px/1.3 var(--body);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.v{font:400 14px/1.4 var(--mono);font-variant-numeric:tabular-nums}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tabs button{font:500 14px var(--body);padding:6px 12px;border-radius:999px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
.tabs button[aria-pressed="true"]{background:var(--ink);color:var(--panel);border-color:var(--ink)}
.legend{display:grid;grid-template-columns:repeat(5,1fr);gap:2px;margin-top:6px}
.legend span{height:12px}.ticks{display:flex;justify-content:space-between;font:12px var(--mono);color:var(--muted)}
.sw{display:inline-block;width:14px;height:12px;vertical-align:-2px;margin-right:6px;border:1px solid var(--line)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--panel);padding:6px 9px;border-radius:6px;font:13px/1.35 var(--body);max-width:260px}
.replay{display:grid;gap:6px}
.future{display:grid;gap:6px;max-width:520px}
.hzt{font:500 13px var(--body);color:var(--ink)}
input[type=range]{width:100%}
.note{color:var(--muted);font-size:13px;margin:0}
.warn{border-left:3px solid var(--p3);padding-left:10px}
:focus-visible{outline:2px solid var(--p3);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="wrap">
  <div><h1>Raion Alert Outlook</h1>
  <p class="sub" id="sub"></p></div>
  <div class="tabs" role="group" aria-label="View">
    <button id="tab-live" aria-pressed="true">Forecast</button>
    <button id="tab-replay" aria-pressed="false">Replay: test days</button>
  </div>
  <div class="box future" id="hz-tabs">
    <label class="k" for="hz">Forecast: chance of an alert in</label>
    <input type="range" id="hz" min="0" max="3" step="1" value="0" aria-valuetext="1 hour">
    <div class="ticks hzt"><span>1 hour</span><span>6 hours</span><span>12 hours</span><span>24 hours</span></div>
    <div class="v" id="hz-label"></div>
  </div>
  <div class="tabs" role="group" aria-label="Level">
    <button id="lvl-raion" aria-pressed="true">Raions</button>
    <button id="lvl-oblast" aria-pressed="false">Oblasts</button>
  </div>
  <div class="grid">
    <div class="map"><svg viewBox="0 0 1000 __H__" role="img" aria-label="Map of Ukrainian raions coloured by alert probability">
      <defs>
        <pattern id="hatch-nd" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--nodata)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--line)" stroke-width="2"/></pattern>
        <pattern id="hatch-ex" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--p3)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--p4)" stroke-width="3"/></pattern>
      </defs>
      <g id="raions">__SHAPES__</g><g>__BORDERS__</g></svg></div>
    <div class="side">
      <div class="box"><div class="k" id="legend-title">Chance of being under alert (redder = higher)</div>
        <div class="legend" style="display:block;height:12px;background:linear-gradient(to right,var(--p0),var(--p2),var(--p4))"></div>
        <div class="ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
        <p class="note" style="margin-top:8px"><span class="sw" style="background:var(--p3)"></span>Hatched red: near-permanent alert (&gt;80% of time), not forecast<br><span class="sw" style="background:var(--nodata)"></span>Hatched grey: no data (Crimea, Sevastopol, Luhanska, Chornobyl zone)</p>
        <p class="note" id="hit-note" hidden><span class="sw" style="background:transparent;border:2px solid var(--hit)"></span>Dark outline: an alert actually happened in the next 24 h</p></div>
      <div class="box replay" id="replay-box" hidden>
        <label class="k" for="day">Forecast made at 06:00 Kyiv, <span id="day-label"></span></label>
        <input type="range" id="day" min="0" max="21" value="0">
        <div class="v" id="day-stats"></div></div>
      <div class="box"><div class="k">Data and model</div><div class="v" id="meta"></div></div>
      <div class="box" id="mix-box"><div class="k">Observed last 7 days (alerts.in.ua)</div><div class="v" id="mix">Hover an oblast's raion</div></div>
      <div class="box warn"><p class="note">Research project, not a warning system. Always follow official air-raid alerts. Probabilities refer to announced alerts, not to attacks.</p></div>
    </div>
  </div>
  <p class="note">Forecast = chance that the raion is under an announced alert at that exact moment. +1 hour: calibrated persistence (B2) - mostly "is an alert running now, and how long do they last here". +6, +12 and +24 hours: exponentially weighted share of time under alert (B4, half-life 7 days: a day a week ago counts half). Each was the best for its horizon on the validation folds (May-July 2026); B4 is <b>not yet tested on held-out data</b>. The hour of day is not used: a version with the night/day pattern (more time under alert 22:00-05:00) had a slightly lower average error on validation but less reliable probabilities, so by our pre-set rule it was not adopted. The replay shows B3 (7-day share), the model that was tested once on the 22 post-reform days (best of four, macro Brier 0.126). Shapes: HDX COD-AB (Kartographia, CC BY-IGO). Alerts: official dataset (Vadimkin, from the official Telegram channel) and alerts.in.ua.</p>
</div>
<div id="tip" hidden></div>
<script>
const DATA = __DATA__;
// continuous, linear 0-100 %: grey -> mid red -> deep red (theme-aware via CSS colour mixing)
const col = p => { p = Math.max(0, Math.min(1, p));
  return p < 0.5 ? `color-mix(in oklab, var(--p2) ${(p * 200).toFixed(1)}%, var(--p0))`
                 : `color-mix(in oklab, var(--p4) ${((p - 0.5) * 200).toFixed(1)}%, var(--p2))`; };
const kyiv = s => new Date(s).toLocaleString("en-GB", {timeZone: "Europe/Kyiv", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"});
const pct = p => (100 * p).toFixed(0) + "%";
const paths = [...document.querySelectorAll(".r.m")];
const m = DATA.meta;
const sub = () => document.getElementById("sub").textContent = mode === "live"
  ? "Data up to " + kyiv(m.data_up_to_utc) + " Kyiv time. Chance of an alert at " + kyiv(m.target_times_utc[HZ[hz]]) + " (" + HZ[hz] + ")."
  : "Replay of the 22 post-reform test days: B3 forecast at 06:00 Kyiv vs what happened in the next 24 h.";
document.getElementById("meta").innerHTML = "data up to: " + kyiv(m.data_up_to_utc) + "<br>forecast made: " + kyiv(m.forecast_made_utc) + "<br>raions forecast: " + m.modeled_units + "<br>code commit: " + m.code_commit_before_forecast;
let mode = "live", day = 0, level = "raion", hz = 0;
const HZ = ["+1h", "+6h", "+12h", "+24h"];
const svg = document.querySelector(".map svg");
function draw() {
  let hits = 0, sumP = 0, sumA = 0, n = 0;
  for (const el of paths) {
    const pc = el.dataset.pc, i = DATA.info[pc];
    let p = i.p[hz], hit = false;
    if (mode === "replay") { const r = DATA.replay.units[pc]; p = r.p[day]; hit = r.a[day] > 0; sumP += p; sumA += r.a[day]; n++; if (hit) hits++; }
    if (level === "oblast") { const o = DATA.obl[i.op]; p = mode === "replay" ? o.rp[day] : o.p[hz]; hit = false; }
    el.style.fill = col(p); el.classList.toggle("hit", hit);
  }
  sub();
  document.getElementById("hz-label").textContent = "at " + kyiv(m.target_times_utc[HZ[hz]]) + " Kyiv time · model: " + (hz === 0 ? "B2 (persistence)" : "B4 (weighted share, half-life 7 days)");
  document.getElementById("hz-tabs").hidden = mode !== "live";
  document.getElementById("hit-note").hidden = mode !== "replay";
  document.getElementById("replay-box").hidden = mode !== "replay";
  if (mode === "replay") {
    document.getElementById("day-label").textContent = DATA.replay.days[day];
    document.getElementById("day-stats").innerHTML = "raions with an alert in the next 24 h: " + hits + " of " + n + "<br>mean forecast " + pct(sumP / n) + " vs actual share of time " + pct(sumA / n);
  }
}
for (const [id, lv] of [["lvl-raion", "raion"], ["lvl-oblast", "oblast"]]) {
  document.getElementById(id).addEventListener("click", () => { level = lv; svg.classList.toggle("lvl-oblast", lv === "oblast");
    for (const b of document.querySelectorAll('[aria-label="Level"] button')) b.setAttribute("aria-pressed", String(b.id === id)); draw(); });
}
for (const [id, md] of [["tab-live", "live"], ["tab-replay", "replay"]]) {
  document.getElementById(id).addEventListener("click", () => { mode = md; for (const b of document.querySelectorAll('[aria-label="View"] button')) b.setAttribute("aria-pressed", String(b.id === id)); draw(); });
}
const HZTXT = ["1 hour", "6 hours", "12 hours", "24 hours"];
document.getElementById("hz").addEventListener("input", e => { hz = +e.target.value; e.target.setAttribute("aria-valuetext", HZTXT[hz]); draw(); });
document.getElementById("day").addEventListener("input", e => { day = +e.target.value; draw(); });
const tip = document.getElementById("tip");
document.getElementById("raions").addEventListener("mousemove", e => {
  const el = e.target.closest(".r"); if (!el) { tip.hidden = true; return; }
  const pc = el.dataset.pc, i = DATA.info[pc];
  let txt = el.dataset.label;
  const o = DATA.obl[el.dataset.op];
  if (level === "oblast" && o) {
    txt = "<b>" + o.name + "</b>";
    if (!o.n) txt += "<br>all raions near-permanent alert";
    else if (mode === "live") txt += "<br>average chance of alert at " + HZ[hz] + ": <b>" + pct(o.p[hz]) + "</b><br>(= expected share of its " + o.n + " forecast raions)";
    else txt += "<br>forecast " + pct(o.rp[day]) + " · actual " + pct(o.ra[day]) + "<br>raions with an alert: " + o.rhit[day] + " of " + o.n;
    tip.innerHTML = txt; tip.hidden = false; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px"; return;
  }
  if (el.classList.contains("nd")) txt += "<br>no data";
  else if (i && i.ex) txt += "<br>near-permanent alert, not forecast";
  else if (mode === "live") txt += "<br>chance of alert at " + HZ[hz] + ": <b>" + pct(i.p[hz]) + "</b>";
  else { const r = DATA.replay.units[pc]; txt += "<br>forecast " + pct(r.p[day]) + " · actual " + pct(r.a[day]) + " of the next 24 h"; }
  tip.innerHTML = txt; tip.hidden = false; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px";
  const mx = i && DATA.mix[i.obl];
  document.getElementById("mix").innerHTML = mx ? i.obl + "<br>alert records: " + mx.records_7d + "<br>red level: " + pct(mx.red_share) +
    "<br>median length: red " + Math.round(mx.median_min_red) + " min · yellow " + Math.round(mx.median_min_yellow) + " min" : "no data";
});
document.getElementById("raions").addEventListener("mouseleave", () => tip.hidden = true);
draw();
</script>
"""

if __name__ == "__main__":
    main()
