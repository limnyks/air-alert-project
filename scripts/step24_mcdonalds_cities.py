"""Step 24: cities that have an OPEN McDonald's restaurant (user's request), mapped to our forecast raions.

Source: the public restaurant locator https://restaurants.mcdonalds.ua/ (its page embeds the restaurant list:
city, coordinates, status open/closed). Each restaurant is assigned to the raion polygon (HDX COD-AB adm2)
that contains its coordinates; a city takes the raion of most of its open restaurants.
Only derived facts are saved: config/mcdonalds_cities.csv (city, number of open restaurants, raion pcode, unit).
The list changes over time; the fetch time is written to config/mcdonalds_cities_fetched.txt.

Run from the project root:
    python scripts/step24_mcdonalds_cities.py | tee logs/step24_output.txt
"""
import collections
import html
import io
import json
import re
import sys
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaf import config  # noqa: E402

URL = "https://restaurants.mcdonalds.ua/"
GEO = config.ROOT / "data" / "raw" / "hdx" / "ukr_admin_boundaries.geojson.zip"
PAT = re.compile(r'\{\\"id\\":(\d+),\\"mcd_id\\":\\"(\d+)\\",\\"name\\":\\"(.*?)\\",\\"address\\":(?:\\"(.*?)\\"|null),\\"city\\":(?:\\"(.*?)\\"|null),'
                 r'\\"zip\\".*?\\"latitude\\":([\d.\-]+),\\"longitude\\":([\d.\-]+).*?\\"status\\":\\"(\w+)\\"')


def inside(x, y, ring):
    r = np.array(ring)[:, :2]
    xs, ys, xs2, ys2 = r[:, 0], r[:, 1], np.roll(r[:, 0], -1), np.roll(r[:, 1], -1)
    return bool((((ys > y) != (ys2 > y)) & (x < (xs2 - xs) * (y - ys) / (ys2 - ys + 1e-18) + xs)).sum() % 2)


def main():
    # system curl (certificate checks stay ON; Python's bundled CA store cannot verify this site's chain)
    raw = subprocess.run(["curl", "-sSL", "-m", "60", "-A", "air-alerts-forecast (research project)", URL],
                         capture_output=True, check=True).stdout
    page = html.unescape(raw.decode("utf-8"))
    n_marker = page.count('\\"mcd_id\\"')
    rec = PAT.findall(page)
    print(f"restaurant records on the page: {n_marker}; parsed: {len(rec)}")
    if not rec or len(rec) != n_marker:
        raise SystemExit("STOP: the page format changed or the parse is incomplete")
    with zipfile.ZipFile(GEO) as z:
        adm2 = json.load(io.TextIOWrapper(z.open("ukr_admin2.geojson"), encoding="utf-8"))["features"]

    def find(lon, lat):
        for f in adm2:
            g = f["geometry"]
            for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                if inside(lon, lat, poly[0]) and not any(inside(lon, lat, h) for h in poly[1:]):
                    return f["properties"]["adm2_pcode"]
        return None

    fc = pd.read_csv(sorted((config.ROOT / "forecasts").glob("final_*.csv"))[-1])
    pc2unit = dict(zip(fc["hdx_pcode"], fc["unit"]))
    excl = dict(zip(fc["hdx_pcode"], fc["excluded"]))
    d = pd.DataFrame([{"city": r[4], "lat": float(r[5]), "lon": float(r[6]), "status": r[7]} for r in rec])
    d["pc"] = [find(lo, la) for lo, la in zip(d["lon"], d["lat"])]
    print(f"status: {d['status'].value_counts().to_dict()}; outside every raion polygon: {int(d['pc'].isna().sum())}")
    rows = []
    for city, g in d[d["status"] == "open"].groupby("city"):
        pc = collections.Counter(g["pc"]).most_common(1)[0][0]
        if pc in pc2unit and not excl[pc]:
            rows.append({"city": city, "n_open": len(g), "pcode": pc, "unit": pc2unit[pc]})
        else:
            print(f"  dropped (raion not forecast): {city} -> {pc}")
    out = pd.DataFrame(rows).sort_values("n_open", ascending=False)
    out.to_csv(config.ROOT / "config" / "mcdonalds_cities.csv", index=False)
    (config.ROOT / "config" / "mcdonalds_cities_fetched.txt").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n")
    only_closed = sorted(set(d.loc[d["status"] == "closed", "city"]) - set(d.loc[d["status"] == "open", "city"]))
    print(f"cities with an open restaurant (kept): {len(out)}; cities with only closed restaurants (dropped): {len(only_closed)}: {', '.join(only_closed)}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
