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
    """Every build area: what config lists, UNION what is on disk.

    "pilot" is the original town-centre build area -- the REGIONS bboxes do
    NOT cover the centre (confirmed: central-town buildings exist only in the
    pilot outlines), so every full-town dedupe/build/merge must include it.

    THE UNION IS THE POINT, and it was learned the hard way. config.REGIONS
    held 23 entries while data/regions held 24, and the missing one was
    `pilot` -- the town centre, where most of the flagged roofs are and
    where reviewers look first. Two district rebuilds skipped it in silence and it
    got the same wrong roof back twice. Iterating the config alone can MISS A
    WHOLE REGION without ever erroring.

    Five patch drivers each carried their own private copy of this fix
    (tools/redo_*.py, tools/patch_labelled.py, tools/refill_partial_roofs.py)
    while all_areas -- the function every STAGE calls -- still did not have
    it. One definition, here, where both kinds of caller already look.
    """
    on_disk = set()
    if REGIONS_DIR.exists():
        on_disk = {p.name for p in REGIONS_DIR.iterdir() if p.is_dir()}
    return sorted(on_disk | {"pilot"} | set(config.REGIONS))


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


def area_bbox_wgs84(name):
    """The region's bbox, from config if it is listed there, else from its own
    outlines on disk.

    A REGION MUST NOT NEED TWO SOURCES OF TRUTH TO BE BUILDABLE. config.REGIONS
    is hand-maintained, data/regions/ is what a fetch actually produced, and
    they drift: `pilot` was on disk and not in the config (two district builds
    skipped it in silence), and `sunshine_bay_west` is the same today. With the
    bbox derivable from the outlines, a region that has data is buildable, full
    stop -- which is also the only way a national rollout works, since nobody
    is going to hand-write a bbox for every suburb in New Zealand.
    """
    # A queued task's bbox first: a worker built from the queue may hold no
    # config entry for the region at all (tools/enqueue_regions.py).
    task = REGIONS_DIR / name / "task.json"
    if task.exists():
        try:
            bbox = json.loads(task.read_text()).get("bbox")
            if bbox and len(bbox) == 4:
                return [float(v) for v in bbox]
        except Exception:
            pass
    if name in config.REGIONS:
        return list(config.REGIONS[name])
    if name == "pilot":
        return list(config.PILOT_BBOX)
    path = REGIONS_DIR / name / "building_outlines.geojson"
    if not path.exists():
        path = REGIONS_DIR / name / "building_outlines_dedup.geojson"
    if not path.exists():
        raise KeyError(f"region {name!r}: no bbox in config and no outlines "
                       f"at {path.parent}")
    import geopandas as gpd
    g = gpd.read_file(path).to_crs("EPSG:4326")
    w, s2, e, n = g.total_bounds
    return [float(w), float(s2), float(e), float(n)]


def area_bbox_nztm(name):
    if name == "pilot":
        return list(config.PILOT_BBOX_NZTM2000)
    w, s, e, n = area_bbox_wgs84(name)
    minx, miny = TO_NZTM.transform(w, s)
    maxx, maxy = TO_NZTM.transform(e, n)
    return [minx, miny, maxx, maxy]


def area_centroid_wgs84(name):
    """(lat, lon) -- SolarModel argument order."""
    if name == "pilot":
        return None  # SolarModel() defaults to the pilot location
    w, s, e, n = area_bbox_wgs84(name)
    return ((s + n) / 2, (w + e) / 2)


def areas_from_argv(argv):
    """CLI convention shared by the builders: no args = pilot (unchanged
    original behaviour); 'all' = every region; else the named regions."""
    args = argv[1:]
    if not args:
        return ["pilot"]
    if args == ["all"]:
        return all_areas()
    known = set(all_areas())
    for a in args:
        if a not in known:
            raise SystemExit(f"unknown region {a!r} (known: {', '.join(sorted(known))})")
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
