"""Work out which areas of New Zealand to build, without typing a single bbox.

Queenstown's 24 regions are hand-written bboxes in config.REGIONS. That does
not survive contact with a country: New Zealand has roughly 2.1 million
buildings in the LINZ outlines, and nobody is hand-drawing boxes around them.
Worse, a hand-written list fails SILENTLY -- config once held 23 entries while
data/regions held 24, the missing one was the town centre, and two district
builds skipped it without erroring.

So the region list becomes a derived thing: lay a grid over the country, ask
LINZ how many buildings fall in each cell, and keep the cells that have any.
A cell with no buildings is not a region; a cell with 40,000 is split until
each piece is the size the pipeline is known to handle.

WHAT "KNOWN TO HANDLE" MEANS, measured rather than guessed. Queenstown's 24
regions average 640 buildings and its largest is 2,712. The per-region build
is a single process holding one region's rasters, and memory is the binding
constraint (see build_layout_geojson's worker sizing). So the target here is
the measured middle of what already works, not an aspiration.

THIS ONLY PLANS. It writes a JSON list of candidate regions with their
building counts, and prints what the run would cost. It fetches nothing
heavy, starts nothing, and spends no compute -- deciding to build is a
separate, funded act.

Counting uses the LINZ WFS `resultType=hits` form, which returns a count
without returning geometry, so a national survey is a few thousand small
requests against an API that is free with a key.

Usage:
    python tools/plan_national_regions.py --zoom 10            # dry plan
    python tools/plan_national_regions.py --zoom 10 --count    # ask LINZ
    python tools/plan_national_regions.py --bbox 174.6 -41.4 175.1 -41.0
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "national_regions.json"

# NZ mainland plus Stewart Island, in WGS84. Chathams are deliberately out:
# they are a separate survey and a separate decision.
NZ_BBOX = [166.3, -47.4, 178.6, -34.3]

# Measured on the district that works: 24 regions, mean 640 buildings, max
# 2,712. A candidate region over the ceiling is split; under the floor it is
# not worth its own build and is merged into a neighbour later.
TARGET_BUILDINGS = 1500
MAX_BUILDINGS = 3000
MIN_BUILDINGS = 25

LINZ_WFS = "https://data.linz.govt.nz/services;key={key}/wfs"


def tiles_covering(bbox, z):
    """Web-mercator tiles at zoom z covering a WGS84 bbox."""
    def xy(lon, lat):
        n = 2 ** z
        x = int((lon + 180.0) / 360.0 * n)
        r = math.radians(max(-85.05, min(85.05, lat)))
        y = int((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * n)
        return max(0, min(n - 1, x)), max(0, min(n - 1, y))

    x0, y0 = xy(bbox[0], bbox[3])
    x1, y1 = xy(bbox[2], bbox[1])
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def tile_bbox(x, y, z):
    n = 2.0 ** z
    def lon(v):
        return v / n * 360.0 - 180.0
    def lat(v):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * v / n))))
    return [lon(x), lat(y + 1), lon(x + 1), lat(y)]


def count_buildings(bbox, key, layer, retries=3):
    """How many building outlines fall in this bbox, without fetching them."""
    q = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": f"layer-{layer}", "resultType": "hits",
        "srsName": "EPSG:4326",
        # LON,LAT. The WFS 2.0 spec says EPSG:4326 is lat,lon and this
        # endpoint does not agree: lat-first returns numberMatched="0" with a
        # perfectly valid 200. Measured on a bbox holding 5,588 Queenstown
        # buildings -- lat,lon gave 0, lon,lat gave 5,588. A wrong axis order
        # here does not error, it reports an empty country.
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326",
    }
    url = LINZ_WFS.format(key=key) + "?" + urllib.parse.urlencode(q)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
            marker = 'numberMatched="'
            i = body.find(marker)
            if i < 0:
                return None
            return int(body[i + len(marker):].split('"')[0])
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=10,
                    help="grid zoom; 10 is about 25 km a cell at NZ latitudes")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    help="limit the plan to this WGS84 bbox (for a trial)")
    ap.add_argument("--count", action="store_true",
                    help="actually ask LINZ for building counts")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import config
    bbox = a.bbox or NZ_BBOX
    cells = tiles_covering(bbox, a.zoom)
    if a.limit:
        cells = cells[:a.limit]
    print(f"grid: zoom {a.zoom}, {len(cells)} cells over "
          f"{'the given bbox' if a.bbox else 'New Zealand'}")

    if not a.count:
        print("\ndry plan only -- pass --count to ask LINZ how many buildings "
              "are in each cell.\nThat is a few thousand small requests "
              "against a free API, and starts no build.")
        return 0

    key = os.environ.get("LINZ_API_KEY") or getattr(config, "LINZ_API_KEY", None)
    if not key:
        print("set LINZ_API_KEY to count")
        return 1
    layer = config.LINZ_BUILDING_OUTLINES_LAYER

    # PROVE THE COUNTER WORKS BEFORE TRUSTING IT ON A COUNTRY. A wrong axis
    # order, an expired key or a renamed layer all come back as a valid
    # response saying zero, and a survey built on that reports an empty New
    # Zealand and nobody sees an error. So: count a bbox known to hold
    # thousands of buildings first, and refuse to go on if it reads empty.
    probe = count_buildings(list(config.PILOT_BBOX), key, layer)
    if not probe or probe < 100:
        print(f"SELF-CHECK FAILED: the pilot bbox {list(config.PILOT_BBOX)} "
              f"reports {probe} buildings, and it holds thousands.")
        print("Something is wrong with the key, the layer id or the request "
              "-- not with New Zealand. Not surveying.")
        return 1
    print(f"self-check: pilot bbox reads {probe:,} buildings, counter works")

    regions, empty, failed, total = [], 0, 0, 0
    for i, (x, y) in enumerate(cells, 1):
        bb = tile_bbox(x, y, a.zoom)
        n = count_buildings(bb, key, layer)
        if n is None:
            failed += 1
            continue
        if n < MIN_BUILDINGS:
            empty += 1
            continue
        total += n
        regions.append({"name": f"z{a.zoom}_{x}_{y}", "bbox": bb,
                        "buildings": n,
                        "split_into": max(1, math.ceil(n / MAX_BUILDINGS))})
        if i % 50 == 0:
            print(f"  {i}/{len(cells)} cells, {len(regions)} populated, "
                  f"{total:,} buildings so far", flush=True)

    builds = sum(r["split_into"] for r in regions)
    OUT.write_text(json.dumps(
        {"zoom": a.zoom, "target_buildings": TARGET_BUILDINGS,
         "max_buildings": MAX_BUILDINGS, "regions": regions}, indent=1))
    print(f"\n{len(regions)} populated cells, {empty} empty, {failed} failed")
    print(f"{total:,} buildings -> {builds} build regions "
          f"(splitting anything over {MAX_BUILDINGS:,})")
    print(f"written to {OUT}")

    # what it would cost, from what the district actually took
    per_h_predict = 7.0 / 15353
    per_h_build = 4.3 / 15353
    print(f"\nat the rates Queenstown actually took on a 16-core VM:")
    print(f"  face precompute   {total * per_h_predict:,.0f} VM-hours")
    print(f"  district build    {total * per_h_build:,.0f} VM-hours")
    print(f"  total             {total * (per_h_predict + per_h_build):,.0f} "
          f"VM-hours  ({total * (per_h_predict+per_h_build)/24:,.0f} days on one, "
          f"{total * (per_h_predict+per_h_build)/24/20:,.1f} days on twenty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
