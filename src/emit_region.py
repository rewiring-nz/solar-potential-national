"""One region -> its own tiles, detail, cells, addresses and summary.

WHY THIS STAGE EXISTS. Everything the map serves used to be cut from two
merged files, and every stage after the merge read the whole district into
memory. That is 400 MB at Queenstown and 65 GB at New Zealand
(docs/scale-architecture.md). The browser has read tiles since 20 September,
and tiles have the property the merge never had: they are built from a small
area and combined mechanically. So the region finishes by emitting its own,
and combine_regions joins them.

Reads:   data/regions/<r>/{solar_potential,panel_layouts}.geojson, the
         heat-map raster and its sidecar, the outlines (for LINZ `use`).
Writes:  data/out/<r>/
           buildings.pmtiles        z13-16, the map's building layer
           panel_layouts.pmtiles    z13-16, panels/facets/obstructions
           cells.json               PARTIAL sums per grid cell and type,
                                    to be added across regions before tiling
           detail/13/<x>/<y>.json   per-building horizons and shade masks
           heatmap/<z>/<x>/<y>.png  raster tiles, composited at seams later
           addresses.json           [address, lon, lat, id]
           solar_potential.geojson  the region's buildings, fully enriched
           summary.json             totals, provenance, and a per-building
                                    ladder the deploy gate compares

NEVER WRITES INTO data/regions/<r>/. The region files are what the gate,
rerank and patch tools read; this stage reads them and writes only under
data/out/<r>/, so running it twice is the same as running it once and a
half-finished run leaves nothing corrupted behind it.

Usage: python src/emit_region.py <region> [--out data/out]
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.region_build import (DATA_DIR, area_paths, area_centroid_wgs84,
                              write_json_atomic)
from src.bake_density_deciles import bake
from src.build_terrain_masks import apply_masks
from src.building_horizon import load_far_dem
from src.shrink_panels_for_tiles import shrink
from src.building_types import classify, BTYPES
from src.build_building_tiles import (tile_of, tile_bounds, _centroid, _best_poa,
                                      CELL_BANDS, COVERAGE_STEPS, KEEP,
                                      BUILDING_MIN_Z, BUILDING_MAX_Z)
from src.split_building_detail import DETAIL_KEYS, DETAIL_Z

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = DATA_DIR / "out"
DEM_WIDE = DATA_DIR / "dem_wide_mosaic.tif"
# z9, not 13: the live site has served heat-map tiles from z9 since the
# tile switch (2,205 tiles, 31 MB), and the district view at z10-12 is where
# most first looks happen. Emitting from 13 quietly deleted those zooms on
# the first combine.
HEATMAP_ZMIN, HEATMAP_ZMAX = 9, 17

# What the map's building layer carries, plus the type fields the toggle reads.
SLIM_KEEP = KEEP + ["btype", "use", "name", "img_dx", "img_dy"]

# The panel layer's properties -- the same list run_district_build.sh passed
# to tippecanoe, kept here because this is now the only place it is cut.
LAYOUT_PROPS = ["kind", "building_id", "fill_rank", "fill_order", "array_id",
                "array_size", "ac_kwh_year", "slope_deg", "aspect_deg",
                "roof_confidence", "poa_kwh_m2_yr", "panel_count", "btype"]


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _tippecanoe(out, layer, src, zmin, zmax, extra):
    cmd = ["tippecanoe", "-q", "-o", str(out), "--force", "-l", layer,
           "-Z", str(zmin), "-z", str(zmax), *extra, str(src)]
    subprocess.run(cmd, check=True, cwd=ROOT)


def _use_by_id(paths):
    """LINZ `use` and `name` per building, from the outlines this region was
    built from. Read with json rather than geopandas: two string columns do
    not need a GeoDataFrame."""
    out = {}
    for cand in (paths["dir"] / "building_outlines.geojson", paths["outlines"]):
        if cand.exists():
            try:
                for f in json.loads(cand.read_text())["features"]:
                    p = f["properties"]
                    out[int(p["building_id"])] = (p.get("use"), p.get("name"))
                break
            except Exception:
                continue
    return out


def emit(region, out_root=OUT_ROOT):
    t0 = time.time()
    paths = area_paths(region)
    sp_path, lay_path = paths["solar_potential"], paths["panel_layouts"]
    if not sp_path.exists() or not lay_path.exists():
        raise SystemExit(f"[{region}] missing solar_potential or panel_layouts -- "
                         f"run the region stages first")
    out = Path(out_root) / region
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    sp = json.loads(sp_path.read_text())
    layouts = json.loads(lay_path.read_text())
    feats = sp["features"]
    print(f"[{region}] {len(feats)} buildings, {len(layouts['features'])} layout features")

    # 1. the ladders (deciles, coverage POA, system steps) -- per building
    matched = bake(sp, layouts)

    # 2. terrain masks, at THIS region's sun, from the DEM window it reaches
    # area_centroid_wgs84 returns (lat, lon) -- SolarModel's order, not
    # GeoJSON's. Unpacking it the other way round put every region's sun at
    # longitude 168 south and was caught only because pvlib refuses a latitude
    # of 168.
    c = area_centroid_wgs84(region)
    lat, lon = c if c else (-45.03, 168.66)
    if DEM_WIDE.exists():
        import rasterio
        from src.region_build import area_bbox_nztm
        band, tr, nd = load_far_dem(DEM_WIDE, area_bbox_nztm(region))
        if band is not None:
            apply_masks(sp, band, tr, nd, lat, lon)
        else:
            print(f"[{region}] WARNING: wide DEM does not reach this region -- no tshade")
    else:
        print(f"[{region}] WARNING: no wide DEM -- no tshade")

    # 3. building type, from LINZ use/name or size
    use = _use_by_id(paths)
    by_type = defaultdict(lambda: {"n": 0, "n_est": 0, "panel_count": 0,
                                   "kwp": 0.0, "kwh": 0.0})
    for f in feats:
        p = f["properties"]
        u, nm = use.get(int(p["building_id"]), (None, None))
        p["btype"] = classify(u, nm, p.get("facet_area_m2"), p.get("kwp"))
        if u and u != "Unknown":
            p["use"] = u
        if nm:
            p["name"] = nm
        t = by_type[p["btype"]]
        t["n"] += 1
        if (p.get("panel_count") or 0) > 0:
            t["n_est"] += 1
        t["panel_count"] += p.get("panel_count") or 0
        t["kwp"] += p.get("kwp") or 0.0
        t["kwh"] += p.get("ac_kwh_year") or 0.0

    # THE DRAWING FOLLOWS THE PHOTO. register_imagery measured, per building,
    # how far the orthophoto sits from the LiDAR (relief displacement: 28% of
    # pilot roofs 2 m or more, per building, no regional constant). Every
    # number was computed where the LiDAR is and stays there; the geometry
    # the map DRAWS -- outline, facets, panels, obstructions -- moves by the
    # shift so it lands on the roof people see in the image.
    # OFF BY DEFAULT UNTIL IT IS VALIDATED. The stage's shifts were right on
    # 42 Suburb Street and wrong at scale: on the first district run,
    # neighbouring roofs (<40 m apart, same photo) agreed on direction only
    # 38-49% of the time against 33% for pure chance, medians of 3.6-4.1 m
    # with 8-14% at the search bound. That is edge-matching locking onto
    # trees and shadows, not relief displacement. Set SOLAR_IMAGE_SHIFT=1 to
    # apply what register_imagery measured; the measurement is kept on disk
    # either way so a better gate can be tested against it.
    shifts = {}
    sf = paths["dir"] / "image_shift.json"
    if sf.exists() and os.environ.get("SOLAR_IMAGE_SHIFT", "0") == "1":
        try:
            shifts = json.loads(sf.read_text())
        except Exception:
            shifts = {}
    def _shift_geom(geom, bid):
        s = shifts.get(str(bid))
        if not s:
            return geom
        dx, dy = s[0], s[1]
        c = _centroid(geom)
        if c is None:
            return geom
        lat = c[1]
        dlon = dx / (111320.0 * max(0.2, __import__("math").cos(__import__("math").radians(lat))))
        dlat = dy / 110540.0
        def mv(node):
            if isinstance(node[0], (int, float)):
                return [node[0] + dlon, node[1] + dlat]
            return [mv(n) for n in node]
        return {"type": geom["type"], "coordinates": mv(geom["coordinates"])}
    n_shifted = 0
    for f in feats:
        p = f["properties"]
        s = shifts.get(str(p["building_id"]))
        if s:
            p["img_dx"], p["img_dy"] = s[0], s[1]
            f["geometry"] = _shift_geom(f["geometry"], p["building_id"])
            n_shifted += 1
    if shifts:
        print(f"[{region}] drawing shifted onto the photo for {n_shifted} buildings")

    # The enriched region document, for tools and for the bucket. Written
    # BEFORE the detail is split out, so it is the complete record.
    write_json_atomic(tmp / "solar_potential.geojson", sp)

    # 4. detail tiles, and the slim buildings the map reads
    pv = config.PV_ASSUMPTIONS
    panel_kw = pv["panel_rated_power_w"] / 1000.0
    density_kw = panel_kw / pv["panel_area_m2"]
    derate = (pv["inverter_efficiency_pct"] / 100.0) * (1 - pv["system_derate_pct"] / 100.0)

    detail = defaultdict(dict)
    slim = []
    cells = defaultdict(lambda: {"n": 0, "n_est": 0, "panel_count": 0,
                                 "fitted_kwp": 0.0, "fitted_kwh": 0.0, "area": 0.0})
    addrs = []
    ladder = {}
    for f in feats:
        p = f["properties"]
        cen = _centroid(f["geometry"])
        d = {k: p[k] for k in DETAIL_KEYS if k in p}
        if d and cen is not None:
            x, y = tile_of(cen[0], cen[1], DETAIL_Z)
            detail[(x, y)][str(p["building_id"])] = d
        slim.append({"type": "Feature", "geometry": f["geometry"],
                     "properties": {k: p[k] for k in SLIM_KEEP if k in p}})
        ladder[str(p["building_id"])] = [p.get("panel_count") or 0,
                                         int(round(p.get("ac_kwh_year") or 0))]
        if cen is None:
            continue
        if p.get("address"):
            addrs.append([p["address"], round(cen[0], 6), round(cen[1], 6), p["building_id"]])
        area = p.get("facet_area_m2") or 0.0
        per_pct = None
        if area > 0:
            per_pct = []
            for pct in COVERAGE_STEPS:
                kwp = area * (pct / 100.0) * density_kw
                per_pct.append((pct, kwp, kwp * _best_poa(p, pct) * derate))
        # ONE CELL PER (grid, cell, type): the sums stay exact per type at
        # every zoom, and summing every type gives the old all-buildings total,
        # so a frontend that does not know about types reads the same numbers.
        for z, _, _ in CELL_BANDS:
            key = (z, *tile_of(cen[0], cen[1], z), p["btype"])
            cell = cells[key]
            cell["n"] += 1
            cell["area"] += area
            cell["panel_count"] += p.get("panel_count") or 0
            cell["fitted_kwp"] += p.get("kwp") or 0.0
            cell["fitted_kwh"] += p.get("ac_kwh_year") or 0.0
            if per_pct:
                cell["n_est"] += 1
                for pct, kwp, kwh in per_pct:
                    cell[f"kwp_{pct}"] = cell.get(f"kwp_{pct}", 0.0) + kwp
                    cell[f"kwh_{pct}"] = cell.get(f"kwh_{pct}", 0.0) + kwh

    for (x, y), payload in detail.items():
        p = tmp / "detail" / str(DETAIL_Z) / str(x) / f"{y}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, separators=(",", ":")))

    slim_path = tmp / "_buildings_slim.geojson"
    slim_path.write_text(json.dumps({"type": "FeatureCollection", "features": slim},
                                    separators=(",", ":")))
    _tippecanoe(tmp / "buildings.pmtiles", "buildings", slim_path,
                BUILDING_MIN_Z, BUILDING_MAX_Z,
                ["--simplification=2", "--no-feature-limit", "--no-tile-size-limit"])
    slim_path.unlink()

    (tmp / "cells.json").write_text(json.dumps(
        {"/".join(map(str, k)): v for k, v in cells.items()}, separators=(",", ":")))
    addrs.sort()
    (tmp / "addresses.json").write_text(json.dumps(addrs, separators=(",", ":")))

    # 5. the panel layer, from a SHRUNK COPY of the layouts. Every layout
    # feature also gets its building's type, so the map can hide a type's
    # panels with a plain attribute filter instead of a per-building lookup.
    btype_of = {int(f["properties"]["building_id"]): f["properties"]["btype"] for f in feats}
    lay_copy = copy.deepcopy(layouts)
    for f in lay_copy["features"]:
        b = f["properties"].get("building_id")
        f["properties"]["btype"] = btype_of.get(int(b), "home") if b is not None else "home"
        if b is not None and shifts.get(str(b)):
            f["geometry"] = _shift_geom(f["geometry"], b)
    shrink(lay_copy)
    lay_tmp = tmp / "_layouts.geojson"
    lay_tmp.write_text(json.dumps(lay_copy, separators=(",", ":")))
    # --maximum-tile-bytes is a PER-REGION budget on purpose. The district-wide
    # tippecanoe used to thin z13/z14 against one 500 kB tile budget; cut per
    # region and joined with -pk, each region thinned against its own smaller
    # share and the joined z13 tiles came out 3x heavier (1.1 MB max, 191 kB
    # median against 368/66). A 200 kB per-region cap leaves a seam tile
    # touched by three regions at about the old ceiling.
    _tippecanoe(tmp / "panel_layouts.pmtiles", "layout", lay_tmp, 13, 16,
                ["--drop-densest-as-needed", "--detect-shared-borders",
                 "--maximum-tile-bytes=200000",
                 *[a for k in LAYOUT_PROPS for a in ("-y", k)]])
    lay_tmp.unlink()
    from_selected = sum(1 for f in layouts["features"]
                        if f["properties"].get("kind") == "facet"
                        and f["properties"].get("from_selected"))

    # 6. heat-map tiles
    n_heat = 0
    if paths["heatmap_png"].exists() and paths["heatmap_json"].exists():
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("hmt", ROOT / "tools" / "build_heatmap_tiles.py")
        hmt = module_from_spec(spec)
        spec.loader.exec_module(hmt)
        corners = json.loads(paths["heatmap_json"].read_text())["coordinates"]
        n_heat = hmt.tiles_for_region(paths["heatmap_png"], corners, tmp / "heatmap",
                                      HEATMAP_ZMIN, HEATMAP_ZMAX, label=region)
        n_heat = max(n_heat, 0)
    else:
        print(f"[{region}] WARNING: no heat-map raster -- no heat-map tiles")

    # 7. the summary: what the gate compares and what status reports
    summary = {
        "region": region, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_sha(), "centroid": [round(lon, 5), round(lat, 5)],
        "n": len(feats), "n_est": sum(1 for f in feats if (f["properties"].get("panel_count") or 0) > 0),
        "n_matched_layout": matched,
        "panel_count": sum(f["properties"].get("panel_count") or 0 for f in feats),
        "kwp": round(sum(f["properties"].get("kwp") or 0.0 for f in feats), 1),
        "kwh": round(sum(f["properties"].get("ac_kwh_year") or 0.0 for f in feats)),
        "by_type": {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv)
                        for kk, vv in v.items()} for k, v in by_type.items()},
        "from_selected_facets": from_selected,
        "detail_tiles": len(detail), "heatmap_tiles": n_heat, "addresses": len(addrs),
        "assumptions": sp.get("assumptions", {}),
        "seconds": round(time.time() - t0, 1),
        "ladder": ladder,
    }
    (tmp / "summary.json").write_text(json.dumps(summary, separators=(",", ":")))

    # Atomic swap: the region's output either is the previous complete set or
    # this one, never a mixture.
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    sizes = {p.name: p.stat().st_size / 1e6 for p in out.glob("*.pmtiles")}
    print(f"[{region}] emitted in {summary['seconds']}s: "
          + ", ".join(f"{k} {v:.1f} MB" for k, v in sizes.items())
          + f", {len(detail)} detail tiles, {n_heat} heat-map tiles, "
          f"{summary['panel_count']:,} panels, {summary['kwh']/1e3:,.0f} MWh")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args()
    from src.preflight import preflight
    preflight("emit_region", a.region)
    emit(a.region, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
