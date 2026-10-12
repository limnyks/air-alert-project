"""Step 15: build the raion map page (docs/map.html) from the committed live forecast and the final-test replay.

Main view: live forecast of the final hour-by-hour model, +1...+24 h (forecasts/final_<latest>.csv/.json, step22,
D47). Also writes a self-contained copy for GitHub Pages: map/index.html. Replay: for
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
import re
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
OUT_PAGES = config.ROOT / "map" / "index.html"
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
    fc_csv = sorted((config.ROOT / "forecasts").glob("final_*.csv"))[-1]
    meta = json.loads(fc_csv.with_suffix(".json").read_text())
    fc = pd.read_csv(fc_csv)
    with zipfile.ZipFile(GEO) as z:
        adm2 = json.load(io.TextIOWrapper(z.open("ukr_admin2.geojson"), encoding="utf-8"))["features"]
        adm1 = json.load(io.TextIOWrapper(z.open("ukr_admin1.geojson"), encoding="utf-8"))["features"]
    proj, height = project_all(adm1)

    # past days (step23): the final model's forecasts vs what really happened
    units = fc["unit"].tolist()
    pj = json.loads((config.ROOT / "forecasts" / "past_days.json").read_text())
    pc_of = dict(zip(fc["unit"], fc["hdx_pcode"]))
    past = {"pcs": [pc_of[u] for u in pj["units"]], "base_utc": pj["base_utc"], "origins": pj["origins"], "p": pj["p"],
            "actual": {pc_of[u]: v for u, v in pj["actual"].items()}, "n_hours": pj["n_hours"]}
    adm2_obl = {f["properties"]["adm2_pcode"]: f["properties"]["adm1_pcode"] for f in adm2}
    mcd = pd.read_csv(config.ROOT / "config" / "mcdonalds_cities.csv")           # step24: cities with an open McDonald's
    mcd_info = {"cities": [{"city": r["city"], "n": int(r["n_open"]), "pc": r["pcode"], "unit": r["unit"].replace(" / ", ", ")} for _, r in mcd.iterrows()],
                "fetched": (config.ROOT / "config" / "mcdonalds_cities_fetched.txt").read_text().strip()}

    info = {r["hdx_pcode"]: {"name": r["raion_uk"] if isinstance(r["raion_uk"], str) else "Київ", "p": [round(float(r[f"p_{h}h"]), 3) if pd.notna(r[f"p_{h}h"]) else None for h in range(1, 25)],
                             "now": int(r["alert_now"]) if pd.notna(r["alert_now"]) else 0, "en": r["unit"].replace(" / ", ", "),
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
    bd = [path(f["geometry"], proj, 0.9) for f in adm1]
    # light halo under a dark line: visible on white (low) raions, on burgundy (high) raions and on the background
    borders = "".join(f'<path class="obh" d="{d}"/>' for d in bd) + "".join(f'<path class="ob" d="{d}"/>' for d in bd)
    mix = {m["unit_name"]: m for m in meta.get("observed_last_7_days_by_oblast", [])}
    page = TEMPLATE.replace("__H__", f"{height:.0f}").replace("__SHAPES__", "".join(shapes)).replace("__BORDERS__", borders).replace("__LABELS__", labels)
    page = page.replace("__DATA__", json.dumps({"info": info, "obl": obl, "past": past, "mcd": mcd_info, "meta": meta, "mix": mix}, ensure_ascii=False))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    # GitHub Pages copy: a full document, system fonts only (no external requests)
    OUT_PAGES.parent.mkdir(exist_ok=True)
    head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    body = page.split("<style>", 1)
    title = re.sub(r'<link rel="stylesheet" href="https://fonts\.googleapis\.com[^>]*>', "", body[0])
    css, rest = body[1].split("</style>", 1)
    OUT_PAGES.write_text(head + title.strip() + "\n<style>\nbody{margin:0}[hidden]{display:none!important}\n" + css + "</style>\n</head>\n<body>\n" + rest + "\n</body>\n</html>\n", encoding="utf-8")
    print(f"saved -> {OUT_PAGES.relative_to(config.ROOT)} ({OUT_PAGES.stat().st_size/1e6:.2f} MB)")
    print(f"saved -> {OUT.relative_to(config.ROOT)} ({OUT.stat().st_size/1e6:.2f} MB); raion shapes {len(shapes)}; forecast file {fc_csv.name}")


TEMPLATE = r"""<title>Raion Alert Outlook</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Koulen&family=Instrument+Serif:ital@0;1&display=swap">
<style>
/* Editorial report style: warm "mist" page, white paper cards, hairline rules, condensed display type,
   serif reading text, pill-outline controls, calm reveal motion. Probability scale: one sequential
   magma-like scale (white -> red -> burgundy, as in the earlier versions), lightness strictly decreasing,
   no red/green pair; the map card stays light in both themes so the scale reads the same. */
:root{--bg:#dbdcd7;--card:#ffffff;--paper:#ffffff;--ink:#010101;--muted:#5d5d58;--line:#010101;--soft:#c7c7c1;
--rs:#c9c8c0;--nodata:#d9d9d3;--hatch:#9b9b94;--border:#010101;--halo:#ffffff;
--c0:#ffffff;--c1:#f4a08f;--c2:#e3221b;--c3:#a5161b;--c4:#6b0f1a;
--display:"Koulen","Bebas Neue",Impact,"Haettenschweiler","Avenir Next Condensed","Arial Narrow",sans-serif-condensed,sans-serif;
--serif:"Instrument Serif","Iowan Old Style",Charter,Georgia,serif;
--sans:"Helvetica Neue",Helvetica,Arial,"Segoe UI",system-ui,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#101010;--card:#1b1b1b;--paper:#f4f3ee;--ink:#ecebe6;--muted:#a7a7a0;--line:#ecebe6;--soft:#3a3a37;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#101010;--card:#1b1b1b;--paper:#f4f3ee;--ink:#ecebe6;--muted:#a7a7a0;--line:#ecebe6;--soft:#3a3a37;color-scheme:dark}
[hidden]{display:none!important}
body{background:var(--bg);color:var(--ink);font:400 15px/1.5 var(--sans);letter-spacing:-.01em}
.wrap{max-width:1320px;margin:0 auto;padding-inline:clamp(16px,2.4vw,32px);padding-block:0 40px;display:grid;gap:20px}
/* top bar: pill-outline view switch + small two-line note, section dots on the right */
.bar{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;display:flex;align-items:center;justify-content:space-between;gap:16px;
  padding:18px 0 14px;background:var(--bg)}
.lefts{display:flex;align-items:center;gap:16px;flex-wrap:wrap;min-width:0}
.pills{display:flex;gap:6px;flex-wrap:wrap}
.pills button,.tabs button,select{font:600 14px/1 var(--sans);letter-spacing:-.01em;padding:10px 16px;border:2px solid var(--ink);border-radius:999px;
  background:transparent;color:var(--ink);cursor:pointer;transition:color .15s cubic-bezier(.4,0,.2,1),background-color .15s cubic-bezier(.4,0,.2,1)}
.pills button:hover,.tabs button:hover{background:var(--ink);color:var(--bg)}
.pills button[aria-pressed="true"],.tabs button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tabs button:disabled{opacity:.3;cursor:default;background:transparent;color:var(--ink)}
.hint{font:600 12px/1.35 var(--sans);margin:0;color:var(--ink)}
.hint span{display:block;font-variant-numeric:tabular-nums}
.dots{display:flex;align-items:center;gap:7px;font:600 13px var(--sans);white-space:nowrap}
.dots span{margin-right:6px}
.dot{width:8px;height:8px;border-radius:50%;border:1.5px solid var(--ink);background:transparent;padding:0;cursor:pointer}
.dot[aria-current="true"]{background:var(--ink)}
/* hero: condensed display headline, then a rule and the serif lede */
.hero{padding-top:clamp(8px,2vw,24px)}
h1{font:400 clamp(60px,10.6vw,176px)/.88 var(--display);font-synthesis:none;text-transform:uppercase;margin:0;letter-spacing:-.005em;text-wrap:balance;will-change:transform}
.lede{display:grid;grid-template-columns:minmax(0,220px) minmax(0,1fr);gap:24px;border-top:2px solid var(--line);margin-top:clamp(18px,2.6vw,32px);padding-top:18px}
.tag{display:inline-flex;align-items:center;gap:8px;font:400 22px/1 var(--display);text-transform:uppercase;align-self:start}
.tag i{font:600 12px/1 var(--sans);font-style:normal;border:1.5px solid var(--ink);border-radius:999px;padding:4px 9px}
.sub{font:400 clamp(19px,1.9vw,24px)/1.35 var(--serif);margin:0;max-width:62ch;letter-spacing:-.01em}
/* rows */
.box{background:var(--card);border-radius:3px;padding:16px 18px}
.future{display:grid;grid-template-columns:minmax(0,260px) minmax(0,1fr);gap:8px 24px;align-items:center;background:transparent;border-top:2px solid var(--line);border-radius:0;padding:18px 0 4px}
.future > .k{grid-row:span 3;font:600 clamp(20px,2vw,28px)/1.15 var(--sans);letter-spacing:-.025em;text-transform:none;color:var(--ink);align-self:start}
.future > *:not(.k){grid-column:2}
.k{font:600 12px/1.3 var(--sans);letter-spacing:.01em;text-transform:uppercase;color:var(--muted)}
.v{font:400 14px/1.55 var(--sans);font-variant-numeric:tabular-nums}
.note{color:var(--muted);font:400 16px/1.4 var(--serif);margin:0}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:20px}
@media (max-width:900px){.grid{grid-template-columns:minmax(0,1fr)}}
.grid.pair{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
.grid.pair .side{grid-column:1/-1;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
@media (max-width:900px){.grid.pair{grid-template-columns:minmax(0,1fr)}}
.map{background:var(--paper);border-radius:3px;padding:10px;min-width:0;color:#010101}
.map svg{width:100%;height:auto;display:block}
.mt{padding:4px 6px 8px;color:#5d5d58}
.r{stroke:var(--rs);stroke-width:.35;cursor:pointer}
.r.m{transition:fill .45s cubic-bezier(.4,0,.2,1)}
.lvl-oblast .r{stroke-width:0}
.r.nd{fill:url(#hatch-nd)}.r.ex{fill:url(#hatch-ex)}
.r:hover,.r:focus{stroke:#010101;stroke-width:1.4}
.ob,.obh{fill:none;stroke-linejoin:round;pointer-events:none}
.obh{stroke:var(--halo);stroke-width:4}
.ob{stroke:var(--border);stroke-width:1.6}
.ol{display:none}
.side{display:grid;gap:12px;align-content:start;min-width:0}
.legend{display:block;height:14px;margin-top:8px;border-radius:2px}
.ticks{display:flex;justify-content:space-between;font:600 12px var(--sans);color:var(--muted);margin-top:5px;font-variant-numeric:tabular-nums}
.hzt{color:var(--ink)}
.sw{display:inline-block;width:14px;height:12px;vertical-align:-1px;margin-right:6px;border:1px solid var(--soft);border-radius:2px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:9px 13px;font:600 13px/1.4 var(--sans);max-width:270px;border-radius:12px}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:32px;background:transparent;margin:0;cursor:pointer}
input[type=range]::-webkit-slider-runnable-track{height:2px;background:var(--ink)}
input[type=range]::-moz-range-track{height:2px;background:var(--ink)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:24px;height:24px;margin-top:-11px;background:var(--bg);border:2px solid var(--ink);border-radius:50%;transition:background-color .15s}
input[type=range]::-moz-range-thumb{width:20px;height:20px;background:var(--bg);border:2px solid var(--ink);border-radius:50%}
input[type=range]:hover::-webkit-slider-thumb{background:var(--ink)}
.warn{background:transparent;border:2px solid var(--ink)}.warn .note{color:var(--ink)}
.prow{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
select{padding-right:34px;appearance:auto;background:var(--card)}
.trip{display:grid;gap:14px;max-width:880px}
.disc{border:2px solid var(--ink);border-radius:3px;padding:18px 20px;background:var(--card);font:400 18px/1.4 var(--serif)}
.disc b{display:block;font:600 20px/1.2 var(--sans);letter-spacing:-.025em;margin-bottom:6px}
.pick{display:flex;gap:8px;flex-wrap:wrap}.pick select{flex:1 1 220px;min-width:0}
.bigs{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.big{font:400 clamp(64px,11vw,128px)/.85 var(--display);font-synthesis:none;font-variant-numeric:tabular-nums;margin-top:4px}
.verdict{border-left:4px solid var(--soft);padding-left:12px;font:400 20px/1.35 var(--serif)}
.tbars{display:flex;align-items:flex-end;gap:3px;height:76px}
.tbars button{flex:1 1 0;min-width:0;border:0;border-radius:2px 2px 0 0;cursor:pointer;padding:0}
.tbars button[aria-pressed="true"]{outline:2px solid var(--ink);outline-offset:1px}
.tbt{display:flex;justify-content:space-between;font:600 12px var(--sans);color:var(--muted)}
.foot{border-top:2px solid var(--line);padding-top:clamp(20px,3vw,36px);display:grid;gap:16px;overflow:hidden}
.foot h2{font:400 clamp(56px,11.5vw,184px)/.88 var(--display);font-synthesis:none;text-transform:uppercase;margin:0}
.foot p{font:400 20px/1.4 var(--serif);margin:0;max-width:60ch}
.reveal{opacity:0;transform:translateY(18px);transition:opacity .6s cubic-bezier(.4,0,.2,1),transform .6s cubic-bezier(.4,0,.2,1)}
.reveal.in{opacity:1;transform:none}
:focus-visible{outline:2px solid var(--ink);outline-offset:3px}
@media (max-width:760px){.lede,.future{grid-template-columns:minmax(0,1fr)}.future > .k{grid-row:auto}.future > *:not(.k){grid-column:1}
  .hint{display:none}.dots span{display:none}.bar{padding:12px 0 10px}.pills button{padding:9px 12px;font-size:13px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}.reveal{opacity:1;transform:none}h1{transform:none!important}}
</style>
<div class="wrap">
  <header class="bar">
    <div class="lefts">
      <div class="pills" role="group" aria-label="View">
        <button id="tab-live" aria-pressed="true">Forecast</button>
        <button id="tab-past" aria-pressed="false">Past days</button>
        <button id="tab-trip" aria-pressed="false">Can I go?</button>
      </div>
      <p class="hint"><span id="clock">Kyiv --:--:--</span><span id="dataline"></span></p>
    </div>
    <nav class="dots" aria-label="Views"><span id="dotname">Forecast</span>
      <button class="dot" data-for="tab-live" aria-label="Forecast" aria-current="true"></button><button class="dot" data-for="tab-past" aria-label="Past days: forecast vs real"></button><button class="dot" data-for="tab-trip" aria-label="Can I go?"></button></nav>
  </header>
  <section class="hero">
    <h1>Raion Alert Outlook</h1>
    <div class="lede"><div class="tag">24 h <i>110 raions</i></div><p class="sub" id="sub"></p></div>
  </section>
  <div class="box future" id="past-box" hidden>
    <label class="k" for="pday">Forecast made on</label>
    <div class="prow"><select id="pday"></select>
      <div class="tabs" id="pmade" role="group" aria-label="Made at"><button id="pm-0">06:00</button><button id="pm-1">12:00</button><button id="pm-2">18:00</button></div></div>
    <p class="note">Kyiv time.</p>
  </div>
  <div class="box future" id="hz-tabs">
    <label class="k" for="hz" id="hz-title">Forecast: chance of an alert in the next hours</label>
    <input type="range" id="hz" min="0" max="23" step="1" value="0" aria-valuetext="+1 hour">
    <div class="ticks hzt"><span>+1 h</span><span>+6 h</span><span>+12 h</span><span>+18 h</span><span>+24 h</span></div>
    <div class="v" id="hz-label"></div>
  </div>
  <div class="tabs" role="group" aria-label="Level" id="lvl-box">
    <button id="lvl-raion" aria-pressed="true">Raions</button>
    <button id="lvl-oblast" aria-pressed="false">Oblasts</button>
  </div>
  <div class="trip" id="trip" hidden>
    <div class="disc" role="alert"><b>Probabilities only - not a basis for any decision about your safety or life.</b>
      This is a research model. A low chance of an alert does not mean it is safe: an alert can start at any moment, and beyond the first few hours the model is only slightly better than a plain average. Always follow the official air-raid alerts.</div>
    <div class="box" style="display:grid;gap:10px">
      <div class="k">A one-hour trip within the city (assumed: no travel while an announced alert is active there)</div>
      <div class="pick"><select id="tplace" aria-label="Destination"><option value="dnipro">cross the Dnipro in Kyiv (Left Bank - Right Bank)</option><option>McDonald's</option><option>a shopping mall</option></select>
        <select id="tcity" aria-label="City or raion"></select>
        <select id="thour" aria-label="Hour of the trip"></select></div>
      <div class="v" id="tline"></div>
      <div class="bigs"><div><div class="k">Chance of an alert</div><div class="big" id="tp-alert"></div></div>
        <div><div class="k">Chance you can go</div><div class="big" id="tp-go"></div></div></div>
      <div class="verdict" id="tverdict"></div>
      <p class="note" id="tnow" hidden></p>
      <p class="note" id="tdn" hidden>Assumed: no crossing while an announced alert is active in Kyiv. Possible closures of bridges or the metro are not in our data and were not checked.</p>
      <div><div class="k" style="margin-bottom:6px">Chance of an alert, hour by hour (tap a bar to choose the hour)</div>
        <div class="tbars" id="tbars"></div><div class="tbt"><span>+1 h</span><span>+12 h</span><span>+24 h</span></div></div>
      <p class="note" id="tmcd"></p>
      <p class="note">Kyiv is one unit in our data, so the Left and Right Banks get the same number. A city is shown as the raion it belongs to (several towns can share one raion). The number is the model's chance of an alert at the start of the chosen hour.</p>
    </div>
  </div>
  <div class="grid" id="grid">
    <div class="map" id="map1"><div class="k mt" id="mt1" hidden>Our forecast</div><svg viewBox="0 0 1000 __H__" role="img" aria-label="Map of Ukrainian raions coloured by alert probability">
      <defs>
        <pattern id="hatch-nd" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--nodata)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--hatch)" stroke-width="2"/></pattern>
        <pattern id="hatch-ex" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--c3)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--c4)" stroke-width="3"/></pattern>
      </defs>
      <g id="raions">__SHAPES__</g><g>__BORDERS__</g></svg></div>
    <div class="map" id="map2" hidden><div class="k mt">What really happened</div></div>
    <div class="side">
      <div class="box"><div class="k" id="legend-title">Chance of being under alert (darker = higher)</div>
        <div class="legend" style="background:linear-gradient(to right,var(--c0),var(--c1),var(--c2),var(--c3),var(--c4))"></div>
        <div class="ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
        <p class="note" style="margin-top:8px"><span class="sw" style="background:var(--c3)"></span>Hatched dark red: near-permanent alert (&gt;80% of time), not forecast<br><span class="sw" style="background:var(--nodata)"></span>Hatched grey: no data (Crimea, Sevastopol, Luhanska, Chornobyl zone)</p>
        <p class="note" id="past-legend" style="margin-top:8px" hidden>Right map: <span class="sw" style="background:var(--c3)"></span>under alert &nbsp;<span class="sw" style="background:var(--c0)"></span>no alert &nbsp;<span class="sw" style="background:var(--nodata)"></span>not known yet</p></div>
      <div class="box" id="past-stats" hidden><div class="k">This hour</div><div class="v" id="pstats"></div>
        <p class="note" style="margin-top:8px">The model is refit before every forecast using only earlier data. Its settings were chosen on 2025 data, so these days are a later check, not an untouched test.</p></div>
      <div class="box"><div class="k">Data and model</div><div class="v" id="meta"></div></div>
      <div class="box warn"><p class="note">Research project, not a warning system. Always follow official air-raid alerts. Probabilities refer to announced alerts, not to attacks.</p></div>
    </div>
  </div>
  <section class="foot" id="foot">
    <h2>Follow official alerts.</h2>
    <p>This is a research project, not a warning system. It shows probabilities of announced air-raid alerts, not of attacks, and a low number never means safe.</p>
  </section>
</div>
<div id="tip" hidden></div>
<script>
const DATA = __DATA__;
const STOPS = ["--c0", "--c1", "--c2", "--c3", "--c4"].map(v => "var(" + v + ")");   // one sequential scale, 0 % -> 100 %
const col = p => { p = Math.max(0, Math.min(1, p)); const x = p * 4, k = Math.min(3, Math.floor(x));
  return `color-mix(in oklab, ${STOPS[k + 1]} ${((x - k) * 100).toFixed(1)}%, ${STOPS[k]})`; };
const kyiv = s => new Date(s).toLocaleString("en-GB", {timeZone: "Europe/Kyiv", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"});
const pct = p => (100 * p).toFixed(0) + "%";
const HZ = Array.from({length: 24}, (_, k) => "+" + (k + 1) + "h");
const HZTXT = HZ.map(h => h.replace("h", " hours").replace("+1 hours", "+1 hour"));
const m = DATA.meta, PD = DATA.past;
const clock = () => document.getElementById("clock").textContent = "Kyiv " + new Date().toLocaleTimeString("en-GB", {timeZone: "Europe/Kyiv", hour: "2-digit", minute: "2-digit", second: "2-digit"});
clock(); setInterval(clock, 1000);
document.getElementById("dataline").textContent = "Data up to " + kyiv(m.data_up_to_utc) + " Kyiv";
const svg = document.querySelector("#map1 svg");
const svg2 = svg.cloneNode(true);                       // second map: what really happened
svg2.querySelector("defs").remove();                    // hatch patterns are defined once, in the first svg
svg2.querySelector("#raions").id = "raions2";
svg2.setAttribute("aria-label", "Map of raions that were really under alert");
document.getElementById("map2").appendChild(svg2);
const paths = [...svg.querySelectorAll(".r.m")], paths2 = [...svg2.querySelectorAll(".r.m")];
let mode = "live", level = "raion", hz = 0, pdi = 0, pmi = 0;
const val = o => o.p[hz];

// past days: forecast origins (06:00 / 12:00 / 18:00 Kyiv), forecasts in %, real alert flags per hour
const MADE = [6, 12, 18], baseMs = Date.parse(PD.base_utc);
const fmt = new Intl.DateTimeFormat("en-GB", {timeZone: "Europe/Kyiv", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", hourCycle: "h23"});
const parts = ms => { const f = fmt.formatToParts(new Date(ms)), g = t => f.find(x => x.type === t).value; return {date: g("year") + "-" + g("month") + "-" + g("day"), hour: +g("hour")}; };
const org = PD.origins.map((off, k) => Object.assign({k, off, ms: baseMs + off * 3600e3}, parts(baseMs + off * 3600e3)));
const days = [...new Set(org.map(o => o.date))];
const originFor = (di, mi) => org.find(o => o.date === days[di] && o.hour === MADE[mi]);
const pidx = Object.fromEntries(PD.pcs.map((pc, i) => [pc, i]));
const pday = document.getElementById("pday");
days.forEach((d, i) => pday.add(new Option(new Date(d + "T12:00:00Z").toLocaleDateString("en-GB", {weekday: "short", day: "2-digit", month: "short", timeZone: "UTC"}), i)));
{ const last = org[org.length - 1]; pdi = days.indexOf(last.date); pmi = MADE.indexOf(last.hour); pday.value = pdi; }
function pastVals(pc) {
  const o = originFor(pdi, pmi), ui = pidx[pc];
  if (!o || ui === undefined) return null;
  const ch = PD.actual[pc][o.off + hz + 1];
  return {p: PD.p[o.k][ui][hz] / 100, a: ch === "1" ? 1 : ch === "0" ? 0 : null};
}
function draw() {
  const past = mode === "past", trip = mode === "trip", lv = past ? "raion" : level;
  document.getElementById("trip").hidden = !trip;
  document.getElementById("grid").hidden = trip;
  document.getElementById("hz-tabs").hidden = trip;
  if (trip) { document.getElementById("past-box").hidden = true; document.getElementById("lvl-box").hidden = true; drawTrip(); return; }
  svg.classList.toggle("lvl-oblast", lv === "oblast");
  document.getElementById("grid").classList.toggle("pair", past);
  for (const id of ["map2", "mt1", "past-box", "past-stats", "past-legend"]) document.getElementById(id).hidden = !past;
  document.getElementById("lvl-box").hidden = past;
  document.getElementById("hz-title").textContent = past ? "Hours after the forecast was made" : "Forecast: chance of an alert in the next hours";
  if (!past) {
    for (const el of paths) { const i = DATA.info[el.dataset.pc]; el.style.fill = col(lv === "oblast" ? val(DATA.obl[i.op]) : val(i)); }
    document.getElementById("sub").textContent = "Data up to " + kyiv(m.data_up_to_utc) + " Kyiv time. Chance of an alert at " + kyiv(m.target_times_utc[hz]) + " (" + HZ[hz] + ").";
    document.getElementById("hz-label").textContent = HZ[hz] + " = " + kyiv(m.target_times_utc[hz]) + " Kyiv time";
    return;
  }
  const o = originFor(pdi, pmi);
  for (let b = 0; b < 3; b++) { const el = document.getElementById("pm-" + b); el.disabled = !originFor(pdi, b); el.setAttribute("aria-pressed", String(b === pmi)); }
  const target = o.ms + (hz + 1) * 3600e3;
  let nA = 0, nN = 0, sA = 0, sN = 0;
  paths.forEach((el, q) => {
    const v = pastVals(el.dataset.pc);
    if (!v) { el.style.fill = paths2[q].style.fill = "var(--nodata)"; return; }
    el.style.fill = col(v.p);
    paths2[q].style.fill = v.a === null ? "var(--nodata)" : v.a ? "var(--c3)" : "var(--c0)";
    if (v.a === 1) { nA++; sA += v.p; } else if (v.a === 0) { nN++; sN += v.p; }
  });
  document.getElementById("sub").textContent = "Forecast made " + kyiv(o.ms) + " Kyiv time. Left: our forecast for " + kyiv(target) + " (" + HZ[hz] + "). Right: what really happened at that hour.";
  document.getElementById("hz-label").textContent = HZ[hz] + " after the forecast = " + kyiv(target) + " Kyiv time";
  document.getElementById("pstats").innerHTML = (nA + nN) === 0
    ? "Not known yet (the data ends " + kyiv(m.data_up_to_utc) + ")."
    : "raions under alert: " + nA + " of " + (nA + nN) + "<br>average forecast for them: " + (nA ? pct(sA / nA) : "-") + "<br>average forecast for the others: " + (nN ? pct(sN / nN) : "-");
}
document.getElementById("meta").innerHTML = "data up to: " + kyiv(m.data_up_to_utc) + "<br>forecast made: " + kyiv(m.forecast_made_utc) + "<br>raions forecast: " + m.modeled_units + "<br>code commit: " + m.code_commit_before_forecast;
for (const [id, lv] of [["lvl-raion", "raion"], ["lvl-oblast", "oblast"]]) {
  document.getElementById(id).addEventListener("click", () => { level = lv;
    for (const b of document.querySelectorAll('[aria-label="Level"] button')) b.setAttribute("aria-pressed", String(b.id === id)); draw(); });
}
for (const [id, md] of [["tab-live", "live"], ["tab-past", "past"], ["tab-trip", "trip"]]) {
  document.getElementById(id).addEventListener("click", () => { mode = md; for (const b of document.querySelectorAll('[aria-label="View"] button')) b.setAttribute("aria-pressed", String(b.id === id)); draw(); });
}
document.getElementById("hz").addEventListener("input", e => { hz = +e.target.value; e.target.setAttribute("aria-valuetext", HZTXT[hz]); draw(); });
pday.addEventListener("change", () => { pdi = +pday.value; if (!originFor(pdi, pmi)) pmi = [0, 1, 2].find(b => originFor(pdi, b)); draw(); });
for (let b = 0; b < 3; b++) document.getElementById("pm-" + b).addEventListener("click", () => { pmi = b; draw(); });
// "Can I go?" tab: one-hour trip, chance of an alert at the chosen hour for the chosen raion (D48)
const tcity = document.getElementById("tcity"), thour = document.getElementById("thour"), tplace = document.getElementById("tplace");
let tk = 0;
const CITIES = DATA.mcd.cities.slice().sort((x, y) => x.city.localeCompare(y.city, "uk"));   // option value = index (towns can share a raion)
{ CITIES.forEach((c, k) => tcity.add(new Option(c.city + " - " + c.unit + " (" + c.n + (c.n === 1 ? " restaurant)" : " restaurants)"), k)));
  tcity.value = CITIES.findIndex(c => DATA.info[c.pc].obl === "Kyiv City");
  var KYIV = tcity.value;
  m.target_times_utc.forEach((t, k) => thour.add(new Option(kyiv(t) + "  (" + HZ[k] + ")", k)));
}
function drawTrip() {
  const dn = tplace.value === "dnipro";
  document.getElementById("tmcd").textContent = "Only cities with an open McDonald's are listed (" + DATA.mcd.cities.length + " cities; the McDonald's restaurant locator, " + DATA.mcd.fetched.slice(0, 16).replace("T", " ") + " UTC). A restaurant can still be closed today - this tab does not know that.";
  if (dn) tcity.value = KYIV;
  tcity.disabled = dn;
  document.getElementById("tdn").hidden = !dn;
  const c = CITIES[+tcity.value], i = DATA.info[c.pc], p = i.p[tk], t = kyiv(m.target_times_utc[tk]);
  document.getElementById("sub").textContent = "Data up to " + kyiv(m.data_up_to_utc) + " Kyiv time. Chance that an announced alert is active during a one-hour trip.";
  document.getElementById("tline").textContent = dn ? "Crossing the Dnipro in Kyiv (Left Bank - Right Bank), " + t + " Kyiv time"
    : "Trip to " + tplace.value + " in " + c.city + ", " + t + " Kyiv time";
  document.getElementById("tp-alert").textContent = pct(p);
  document.getElementById("tp-go").textContent = pct(1 - p);
  const v = document.getElementById("tverdict");
  v.textContent = p < 0.2 ? "Alert unlikely at this hour - the trip is probably possible. Not a guarantee."
    : p < 0.5 ? "Alert possible - the trip is uncertain." : "Alert likely - the trip is probably not possible.";
  v.style.borderLeftColor = col(p);
  const now = document.getElementById("tnow");
  now.hidden = !(i.now && tk < 3);
  now.textContent = "An alert was active here at the data time (" + kyiv(m.data_up_to_utc) + "); the first hours are affected most.";
  const bars = document.getElementById("tbars"); bars.textContent = "";
  i.p.forEach((q, k) => { const b = document.createElement("button");
    b.style.height = Math.max(3, Math.round(q * 76)) + "px"; b.style.background = col(q);
    b.setAttribute("aria-pressed", String(k === tk)); b.setAttribute("aria-label", kyiv(m.target_times_utc[k]) + ": " + pct(q) + " chance of an alert");
    b.title = kyiv(m.target_times_utc[k]) + ": " + pct(q); b.addEventListener("click", () => { tk = k; thour.value = k; drawTrip(); });
    bars.appendChild(b); });
}
tcity.addEventListener("change", drawTrip); tplace.addEventListener("change", drawTrip);
thour.addEventListener("change", () => { tk = +thour.value; drawTrip(); });
const tip = document.getElementById("tip");
function hover(e) {
  const el = e.target.closest(".r"); if (!el) { tip.hidden = true; return; }
  const pc = el.dataset.pc, i = DATA.info[pc], past = mode === "past";
  let txt = el.dataset.label;
  const o = DATA.obl[el.dataset.op];
  if (!past && level === "oblast" && o) {
    txt = "<b>" + o.name + "</b>";
    if (!o.n) txt += "<br>all raions near-permanent alert";
    else txt += "<br>average chance of alert at " + HZ[hz] + ": <b>" + pct(val(o)) + "</b><br>(= expected share of its " + o.n + " forecast raions)";
  } else if (el.classList.contains("nd")) txt += "<br>no data";
  else if (i && i.ex) txt += "<br>near-permanent alert, not forecast";
  else if (!past) txt += "<br>chance of alert at " + kyiv(m.target_times_utc[hz]) + ": <b>" + pct(val(i)) + "</b>" + (i.now ? "<br>alert running at the data time" : "");
  else { const v = pastVals(pc); if (v) txt += "<br>forecast: <b>" + pct(v.p) + "</b><br>real: <b>" + (v.a === null ? "not known yet" : v.a ? "under alert" : "no alert") + "</b>"; }
  tip.innerHTML = txt; tip.hidden = false; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px";
}

// motion: calm reveals, a slowly drifting headline, view dots (all off with "reduce motion")
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
const revIO = new IntersectionObserver(es => { for (const e of es) if (e.isIntersecting) { e.target.classList.add("in"); revIO.unobserve(e.target); } }, {rootMargin: "0px 0px -6% 0px"});
for (const el of document.querySelectorAll(".box, .map, .foot, .lede")) {
  const onScreen = el.getBoundingClientRect().top < innerHeight;          // visible at load: shown at once, no fade
  el.classList.add("reveal"); (still || onScreen) ? el.classList.add("in") : revIO.observe(el); }
const h1 = document.querySelector("h1");
let ticking = false;
addEventListener("scroll", () => { if (ticking || still) return; ticking = true; requestAnimationFrame(() => {
  h1.style.transform = "translateY(" + Math.min(scrollY * 0.18, 120) + "px)"; ticking = false; }); }, {passive: true});
const NAMES = {live: "Forecast", past: "Past days", trip: "Can I go?"};
const setDots = () => { document.getElementById("dotname").textContent = NAMES[mode];
  for (const d of document.querySelectorAll(".dot")) d.setAttribute("aria-current", String(d.dataset.for === "tab-" + mode)); };
for (const d of document.querySelectorAll(".dot")) d.addEventListener("click", () => document.getElementById(d.dataset.for).click());
for (const id of ["tab-live", "tab-past", "tab-trip"]) document.getElementById(id).addEventListener("click", setDots);
for (const id of ["raions", "raions2"]) { const g = document.getElementById(id); g.addEventListener("mousemove", hover); g.addEventListener("mouseleave", () => tip.hidden = true); }
draw();
</script>
"""

if __name__ == "__main__":
    main()
