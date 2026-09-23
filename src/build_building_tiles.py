"""Buildings as vector tiles, plus an aggregate grid for the zooms above them.

WHY. The solar potential data must not download whole; the map needs an
approach that will be fast for all NZ, then increase in accuracy as we get
closer?"

That is exactly the shape. Panel layouts have been vector tiles for weeks and
scale fine -- a client fetches the tiles it is looking at and nothing else.
Buildings were the one layer still downloaded whole: 21.5 MB for Queenstown's
15,353 after the detail split, which is about 2.9 GB for New Zealand's 2.1
million. The map would never open.

TWO LAYERS, BECAUSE ZOOMING OUT IS NOT THE SAME PROBLEM AS ZOOMING IN.

    buildings.pmtiles      z13-16, one feature per building, full properties.
    building_cells.pmtiles z0-12, one feature per grid cell, with the SUMS.

The cells are not decoration and they are not a simplification of the
buildings -- they exist because the dashboard adds up every building in view,
and a tiled source cannot be added up. Tippecanoe drops features to fit a
zoom's byte budget, so summing what the map happens to have rendered at z10
silently undercounts, and undercounts differently as you pan. The cell carries
the total for everything inside it, dropped features included, so the figure
is exact at every zoom. That is the difference between coarser and wrong.

Each cell carries the count and, for every coverage step the dropdown offers,
the summed kWp and kWh -- computed per building with that building's own
cov_poa_N, so the sunniest-part-first rule survives the aggregation.

WHAT IS NOT HERE. The cells hold no economics: cost, payback and savings are
per building by design (a district-wide dollar figure invites being read as
a forecast for the town), and they are computed client-side
from the estimate anyway.

Usage: python src/build_building_tiles.py
"""

import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "solar_potential.geojson"

# A PYRAMID, NOT ONE GRID. First attempt used a single z12 grid for every zoom
# below the buildings, and the seam gave it away: z12.9 reported 7,839
# buildings and z13.1 reported 2,267 for the same place. Nothing was wrong with
# either sum -- a 7 km cell that clips the edge of the viewport is counted
# whole, so at the handover zoom, where the viewport is barely wider than one
# cell, most of what was counted was off screen.
#
# The fix is for the cell to stay small RELATIVE TO THE VIEW: each band uses a
# grid three zooms finer than the zoom you are at, so the viewport always spans
# roughly eight cells and the edge error is bounded at about a quarter instead
# of several times over. Which is also the requirement: fast for all of NZ,
# then more accurate as the map zooms in.
#
# All three resolutions ship in one layer tagged `lod`; the style draws one
# band at a time and the dashboard sums the matching one. They are a few
# thousand boxes nationally, so carrying all three costs nothing.
# grid zoom = the band's TOP zoom + 3, which is what makes "three zooms finer
# than the view" true at the tightest zoom in the band rather than only at the
# loosest. Getting this wrong the first time left a 1.9x step at the handover:
# the band shown up to z13 used a z13 grid, so at z12.9 one cell was the whole
# screen and its off-screen half was counted.
CELL_BANDS = [
    # (grid zoom, shown from, shown up to)
    (9,  0,  6),
    (12, 6,  9),
    (16, 9,  13),
]
BUILDING_MIN_Z = 13  # buildings are served from here
BUILDING_MAX_Z = 16  # and over-zoomed beyond, as panel layouts already are

COVERAGE_STEPS = [5, 10, 15, 25, 50, 75, 100]

# Everything the map draws or the panel reads. Deliberately a list rather than
# "whatever is in the file": an unlisted property is one nobody has decided to
# ship, and a tile that carries all 56 of them is most of the problem this
# module exists to solve.
KEEP = [
    "building_id", "address", "address_count",
    "kwp", "panel_count", "ac_kwh_year", "ac_kwh_day_avg",
    "facet_area_m2", "facet_count", "obstruction_count",
    "avg_poa_kwh_m2", "roof_confidence",
    "no_estimate_reason", "no_estimate_text",
] + [f"cov_poa_{p}" for p in COVERAGE_STEPS] \
  + [f"fill_panels_{d}" for d in range(10, 101, 10)] \
  + [f"fill_kwh_{d}" for d in range(10, 101, 10)] \
  + ["fill_panels_arrays", "fill_kwh_arrays"] \
  + [f"sys_kwh_{n}" for n in (7, 10, 14, 17, 20, 27, 34, 45, 68)]


def tile_of(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    r = math.radians(max(-85.05, min(85.05, lat)))
    y = int((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_bounds(x, y, z):
    n = 2.0 ** z
    def lon(xx):
        return xx / n * 360.0 - 180.0
    def lat(yy):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
    return lon(x), lat(y + 1), lon(x + 1), lat(y)


def _centroid(geom):
    try:
        ring = geom["coordinates"][0]
        if ring and isinstance(ring[0][0], (list, tuple)):
            ring = ring[0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    except Exception:
        return None


def _best_poa(p, pct):
    """Mean irradiance over the sunniest pct% of this roof.

    The same ladder preview.html interpolates, so a cell's total is the sum of
    exactly the per-building numbers the map shows when you zoom into it."""
    steps = [c for c in COVERAGE_STEPS if p.get(f"cov_poa_{c}") is not None]
    if not steps:
        return p.get("avg_poa_kwh_m2") or 0.0
    q = max(steps[0], min(steps[-1], pct))
    for i, c in enumerate(steps):
        if q == c:
            return p[f"cov_poa_{c}"]
        if q < c:
            lo, hi = steps[i - 1], c
            vlo, vhi = p[f"cov_poa_{lo}"], p[f"cov_poa_{hi}"]
            return vlo + (vhi - vlo) * (q - lo) / (hi - lo)
    return p[f"cov_poa_{steps[-1]}"]


def main():
    if not SRC.exists():
        print(f"no {SRC}")
        return 1
    doc = json.loads(SRC.read_text())
    feats = doc["features"]
    print(f"{len(feats)} buildings")

    pv = config.PV_ASSUMPTIONS
    panel_kw = pv["panel_rated_power_w"] / 1000.0
    density_kw = panel_kw / pv["panel_area_m2"]  # kWp per m2 of roof
    derate = (pv["inverter_efficiency_pct"] / 100.0) \
        * (1 - pv["system_derate_pct"] / 100.0)

    slim = []
    grids = {z: defaultdict(lambda: {"n": 0, "n_est": 0, "panel_count": 0,
                                     "fitted_kwp": 0.0, "fitted_kwh": 0.0,
                                     "area": 0.0})
             for z, _, _ in CELL_BANDS}
    for f in feats:
        p = f["properties"]
        slim.append({"type": "Feature", "geometry": f["geometry"],
                     "properties": {k: p[k] for k in KEEP if k in p}})
        c = _centroid(f["geometry"])
        if c is None:
            continue
        area = p.get("facet_area_m2") or 0.0
        per_pct = None
        if area > 0:
            per_pct = []
            for pct in COVERAGE_STEPS:
                kwp = area * (pct / 100.0) * density_kw
                per_pct.append((pct, kwp, kwp * _best_poa(p, pct) * derate))
        for z in grids:
            cell = grids[z][tile_of(c[0], c[1], z)]
            cell["n"] += 1
            cell["area"] += area
            cell["panel_count"] += p.get("panel_count") or 0
            cell["fitted_kwp"] += (p.get("kwp") or 0.0)
            cell["fitted_kwh"] += (p.get("ac_kwh_year") or 0.0)
            if per_pct:
                cell["n_est"] += 1
                for pct, kwp, kwh in per_pct:
                    cell[f"kwp_{pct}"] = cell.get(f"kwp_{pct}", 0.0) + kwp
                    cell[f"kwh_{pct}"] = cell.get(f"kwh_{pct}", 0.0) + kwh

    tmp_b = DATA / "_buildings_slim.geojson"
    tmp_b.write_text(json.dumps(
        {"type": "FeatureCollection", "features": slim},
        separators=(",", ":")))

    cell_feats = []
    for cz, cells in sorted(grids.items()):
      for (x, y), v in sorted(cells.items()):
        w, s, e, n = tile_bounds(x, y, cz)
        props = {"cell": f"{cz}/{x}/{y}", "lod": cz,
                 "n": v["n"], "n_est": v["n_est"],
                 "panel_count": v["panel_count"],
                 "fitted_kwp": round(v["fitted_kwp"], 1),
                 "fitted_kwh": round(v["fitted_kwh"], 0),
                 "facet_area_m2": round(v["area"], 0),
                 # ground area, so the choropleth can shade by density rather
                 # than by total -- a coarse cell holds more land, not more sun
                 "km2": round(abs(e - w) * abs(n - s) * 111.32 * 111.32
                              * math.cos(math.radians((n + s) / 2)), 3)}
        for pct in COVERAGE_STEPS:
            props[f"kwp_{pct}"] = round(v.get(f"kwp_{pct}", 0.0), 1)
            props[f"kwh_{pct}"] = round(v.get(f"kwh_{pct}", 0.0), 0)
        cell_feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [w, s], [e, s], [e, n], [w, n], [w, s]]]},
            "properties": props})
    tmp_c = DATA / "_building_cells.geojson"
    tmp_c.write_text(json.dumps(
        {"type": "FeatureCollection", "features": cell_feats},
        separators=(",", ":")))
    print("cells: " + ", ".join(f"{len(g)} at z{z}" for z, g in sorted(grids.items())))

    def tip(out, layer, src, zmin, zmax, extra=()):
        cmd = ["tippecanoe", "-o", str(out), "--force", "-l", layer,
               "-Z", str(zmin), "-z", str(zmax),
               "--no-tile-compression" if False else "--simplification=2",
               *extra, str(src)]
        subprocess.run(cmd, check=True, cwd=ROOT)

    # Buildings: never drop one. A missing building is someone's house absent
    # from the map, and the byte budget is met by the zoom floor instead --
    # below BUILDING_MIN_Z there are no building tiles at all, there are cells.
    tip(DATA / "buildings.pmtiles", "buildings", tmp_b,
        BUILDING_MIN_Z, BUILDING_MAX_Z,
        ["--no-feature-limit", "--no-tile-size-limit"])
    # Cells: a few thousand boxes nationally, so they always all fit.
    tip(DATA / "building_cells.pmtiles", "cells", tmp_c,
        0, max(hi for _, _, hi in CELL_BANDS),
        ["--no-feature-limit", "--no-tile-size-limit"])

    tmp_b.unlink(missing_ok=True)
    tmp_c.unlink(missing_ok=True)

    # Two small files the page used to read out of the big one.
    #
    # assumptions: a single object, 1 kB, and the page cannot draw a number
    # without it.
    (DATA / "assumptions.json").write_text(json.dumps(doc.get("assumptions", {})))
    # addresses: the search is client-side autocomplete over what the map
    # knows, and with buildings tiled it can no longer see them all. A flat
    # index is right at district scale (0.5 MB) and is NOT right nationally --
    # 2.1 million addresses is roughly 34 MB, so national search needs this
    # sharded by prefix or replaced with a geocoder. Recorded in
    # docs/scaling-and-iteration.md rather than pretended away.
    addrs = []
    for f in feats:
        p2 = f["properties"]
        if not p2.get("address"):
            continue
        c2 = _centroid(f["geometry"])
        if c2 is None:
            continue
        addrs.append([p2["address"], round(c2[0], 6), round(c2[1], 6),
                      p2["building_id"]])
    addrs.sort()
    (DATA / "addresses.json").write_text(json.dumps(addrs, separators=(",", ":")))
    print(f"  addresses.json: {len(addrs)} addresses, "
          f"{(DATA / 'addresses.json').stat().st_size / 1e6:.1f} MB")
    for name in ("buildings.pmtiles", "building_cells.pmtiles"):
        print(f"  {name}: {(DATA / name).stat().st_size / 1e6:.1f} MB")
    print(f"  (solar_potential.geojson was "
          f"{SRC.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
