"""
Shared plumbing for per-region builds.

An "area" is either the original pilot ("pilot", inputs at data/ root) or a
config.REGIONS name (inputs at data/regions/<name>/). Every build script
resolves its inputs/outputs through area_paths() so the pilot keeps working
exactly as before while regions build into their own directories.

Region bboxes overlap each other (and cover the pilot), so the same
building_id can be fetched by several regions. dedupe_outlines() assigns
each building to exactly one owning region -- the one whose bbox holds the
building's centroid deepest inside (max distance to the nearest bbox edge,
i.e. the region where the building is least likely to have clipped DSM or
imagery at the boundary) -- and writes building_outlines_dedup.geojson per
region. Builders read the deduped file when it exists, so solar/layout/
raster outputs are disjoint across regions by construction and a final
merge is a plain concatenation.
"""

import json
import os
import sys
from pathlib import Path

import pyproj

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REGIONS_DIR = DATA_DIR / "regions"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)


def write_json_atomic(path, obj):
    """Write JSON via a temp file + rename, never straight over the target.

    Several pipeline stages REWRITE their own input in place (gate_panels,
    rerank_layouts, shrink_panels_for_tiles, bake_density_deciles,
    build_terrain_masks, add_addresses). A plain write_text truncates the
    file first, so an interrupt -- a crash, a killed background run, a full
    disk -- during a multi-hundred-MB dump leaves a truncated, unparseable
    artifact and the stage's input is gone. os.replace is atomic on the same
    filesystem, so the target is either the old file or the new one.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def all_areas():
    # "pilot" is the original town-centre build area -- the 10 REGIONS bboxes
    # do NOT cover the centre (confirmed: central-town buildings exist only in
    # the pilot outlines), so every full-town dedupe/build/merge must include it.
    return ["pilot"] + list(config.REGIONS)


def area_paths(name):
    # pilot builds into data/regions/pilot/ like any region (inputs symlinked
    # to the data-root mosaics); data-root output files are the MERGE TARGETS
    # and must never double as an area's own outputs.
    d = REGIONS_DIR / name
    outlines = d / "building_outlines_dedup.geojson"
    if not outlines.exists():
        outlines = d / "building_outlines.geojson"
    return {
        "dir": d,
        "outlines": outlines,
        "dsm": d / "dsm_mosaic.tif",
        "imagery": d / "imagery_mosaic.tif",
        "solar_potential": d / "solar_potential.geojson",
        "panel_layouts": d / "panel_layouts.geojson",
        "heatmap_png": d / "heatmap_raster.png",
        "heatmap_json": d / "heatmap_raster.json",
    }


def area_bbox_nztm(name):
    if name == "pilot":
        return list(config.PILOT_BBOX_NZTM2000)
    w, s, e, n = config.REGIONS[name]
    minx, miny = TO_NZTM.transform(w, s)
    maxx, maxy = TO_NZTM.transform(e, n)
    return [minx, miny, maxx, maxy]


def area_centroid_wgs84(name):
    """(lat, lon) -- SolarModel argument order."""
    if name == "pilot":
        return None  # SolarModel() defaults to the pilot location
    w, s, e, n = config.REGIONS[name]
    return ((s + n) / 2, (w + e) / 2)


def areas_from_argv(argv):
    """CLI convention shared by the builders: no args = pilot (unchanged
    original behaviour); 'all' = every region; else the named regions."""
    args = argv[1:]
    if not args:
        return ["pilot"]
    if args == ["all"]:
        return all_areas()
    for a in args:
        if a != "pilot" and a not in config.REGIONS:
            raise SystemExit(f"unknown region {a!r} (known: pilot, {', '.join(config.REGIONS)})")
    return args


def _edge_margin(x, y, bbox):
    minx, miny, maxx, maxy = bbox
    return min(x - minx, maxx - x, y - miny, maxy - y)


def dedupe_outlines(region_names=None):
    """Assign every building to its deepest-inside region and write
    building_outlines_dedup.geojson for each. Run after fetching outlines,
    before any build."""
    import geopandas as gpd

    region_names = region_names or all_areas()
    frames = {}
    for name in region_names:
        path = REGIONS_DIR / name / "building_outlines.geojson"
        gdf = gpd.read_file(path)
        frames[name] = gdf

    owner = {}  # building_id -> (margin, region)
    for name, gdf in frames.items():
        bbox = area_bbox_nztm(name)
        for row in gdf.itertuples():
            c = row.geometry.centroid
            margin = _edge_margin(c.x, c.y, bbox)
            bid = row.building_id
            if bid not in owner or margin > owner[bid][0]:
                owner[bid] = (margin, name)

    total = 0
    demolished = getattr(config, "DEMOLISHED_BUILDING_IDS", set())
    for name, gdf in frames.items():
        keep = gdf[gdf["building_id"].map(lambda b: owner[b][1] == name and b not in demolished)]
        out = REGIONS_DIR / name / "building_outlines_dedup.geojson"
        keep.to_file(out, driver="GeoJSON")
        print(f"{name}: {len(keep)}/{len(gdf)} buildings owned")
        total += len(keep)
    print(f"{total} unique buildings across {len(frames)} regions")


if __name__ == "__main__":
    dedupe_outlines(sys.argv[1:] or None)
