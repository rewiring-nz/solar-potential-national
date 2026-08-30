"""
Placement quality gate, applied as a post-filter over built panel layouts.

Drops any placed panel whose underlying LiDAR contradicts "flat roof
surface here" -- the failure classes from the 23-building field-report set
(docs/bugdoc-2026-08-22.md):

- too few building-class returns beneath it  -> carpark / air / demolished
  (19 Industrial Pl, 10/16 Kent St, 61 Ballarat St)
- surface not meaningfully above bare earth  -> ground-level slab/yard
- points disagree with the local panel plane -> covers vents/plant/level
  changes the obstruction pass missed (17 Marine Pde; audit's 3,459 lumpy)

Runs per region on panel_layouts.geojson IN PLACE (before rerank/deciles/
shrink/tile). The same checks move into fit time with the Wave-1 rebuild;
this post-filter exists so the worst placements leave the live map a
rebuild-cycle earlier.

Usage: python src/gate_panels.py [region ...]   (default: all areas)
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import shapely.vectorized
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pointcloud_source import PointCloudSource
from src.region_build import all_areas, area_paths, write_json_atomic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform

MIN_HEIGHT_ABOVE_GROUND_M = 2.0  # a panel's surface must stand this far above the LOCAL ground,
# measured from LiDAR ground-class returns nearby (not the 8m DEM, and not the
# building-class flag). Ground classification is the most reliable product in a
# LAS file; building classification is not -- 6 Shotover St is a real commercial
# roof where only 24% of returns under each panel carried the building flag, and
# a classification-fraction rule deleted 155 of its 163 panels. Geometry decides.
GROUND_SEARCH_RADIUS_M = 20.0
MIN_EVIDENCE_PTS = 8        # fewer total returns than this under a panel = survey too thin
# to judge here at all -> keep. (7 Cedar Dr: a real roof at 2.0 pts/m2 total
# had 63 of 69 fitted panels executed by absolute-count thresholds.)
MAX_LOCAL_RMS = 0.28        # points under one 2m panel should fit their own plane this well
BELOW_PLANE_TOLERANCE_M = 0.35  # returns further below the panel's own roof level than this are
# not the roof -- see the lumpy test for the measurement that motivated it


def panel_ok(poly, pc, dem, dem_transform_inv):
    minx, miny, maxx, maxy = poly.bounds
    # A veto requires EVIDENCE AGAINST a roof, never mere absence of data:
    # a LiDAR coverage gap (zero returns of ANY class) means "unknown" and
    # the panel stays. Cost of this lesson: a cropped-tile coverage hole
    # made the first gate run execute 96 healthy houses.
    pts_all = pc.points_in_bbox(minx - 0.3, miny - 0.3, maxx + 0.3, maxy + 0.3, building_only=False)
    if len(pts_all) == 0:
        return True, "no_coverage_kept"
    pts = pc.points_in_bbox(minx - 0.3, miny - 0.3, maxx + 0.3, maxy + 0.3, building_only=True)
    inside_all = shapely.vectorized.contains(poly, pts_all[:, 0], pts_all[:, 1])
    n_all = int(inside_all.sum())
    if n_all < MIN_EVIDENCE_PTS:
        return True, "thin_coverage_kept"  # not enough returns of ANY class to judge
    all_in = pts_all[inside_all]
    # Surface height under the panel: the upper cluster of returns, so a few
    # ground returns seen through a gap can't drag it down.
    roof_z = float(np.percentile(all_in[:, 2], 75))

    # Local ground from ground-class returns in a neighbourhood; if the tile
    # has none nearby, fall back to the lowest returns around the panel.
    c = poly.centroid
    r = GROUND_SEARCH_RADIUS_M
    around = pc.points_in_bbox(c.x - r, c.y - r, c.x + r, c.y + r, building_only=False)
    ground_cls = pc.ground_points_in_bbox(c.x - r, c.y - r, c.x + r, c.y + r)
    if len(ground_cls) >= 20:
        local_ground = float(np.percentile(ground_cls[:, 2], 50))
    elif len(around) >= 20:
        local_ground = float(np.percentile(around[:, 2], 5))
    else:
        return True, "no_ground_reference_kept"
    if (roof_z - local_ground) < MIN_HEIGHT_ABOVE_GROUND_M:
        # sits at ground level: carpark, yard, slab, or air over a gap where
        # the only returns are the ground below
        return False, "sparse"
    inside = shapely.vectorized.contains(poly, pts[:, 0], pts[:, 1]) if len(pts) else np.zeros(0, bool)
    pp = pts[inside] if len(pts) and inside.any() else all_in
    # NO height-above-DEM test. The wide DEM is 8m-resolution smoothed bare
    # earth: on sloping ground its cell averages uphill terrain, so a real
    # single-storey roof can sit <1m above it (4 Abbottswood Ln: roof 392.9,
    # DEM 391.4 -> every panel wrongly read as ground-level, including the
    # north face that carries REAL installed panels in the photo). Height is
    # already implied by LAS building classification, which is per-return and
    # far more reliable here; a rooftop parking deck is an exclusion-list case,
    # not a height-rule case.
    # Local planarity: the points under one panel must fit their own plane --
    # but measured only against the ROOF SURFACE, not against everything the
    # scanner saw through the gap at a roof edge.
    #
    # This test exists to catch a panel sitting on unmodelled structure: a
    # chimney, a vent, plant. Structure is ABOVE the roof. Points BELOW it are
    # the wall, the eave soffit, or the ground seen past the edge, and a panel
    # near a roof edge picks them up routinely. Measured over three commercial
    # roofs, of the panels this test rejected:
    #     82% had their outliers mostly BELOW the roof plane  (edge artefact)
    #     10% had them mostly ABOVE                            (real structure)
    # and separately, 87% of all gate drops were edge panels. It was cutting
    # 45 Camp St from 63 fitted panels to 31 -- a roof Josh reported as
    # "sparsely populated even though plenty of extra space".
    #
    # Same physics guard obstruction_detection already applies to its own
    # candidates: deviation on both sides is roof form, deviation above is an
    # object. Drop the below-plane returns before fitting the local plane.
    if len(pp) >= 6:
        roof_ref = float(np.percentile(pp[:, 2], 75))
        on_roof = pp[pp[:, 2] >= roof_ref - BELOW_PLANE_TOLERANCE_M]
        if len(on_roof) >= 6:
            pp = on_roof
    if len(pp) >= 6:
        x0, y0 = pp[:, 0].mean(), pp[:, 1].mean()
        A = np.column_stack([pp[:, 0] - x0, pp[:, 1] - y0, np.ones(len(pp))])
        try:
            coeffs, *_ = np.linalg.lstsq(A, pp[:, 2], rcond=None)
            rms = float(np.sqrt(np.mean((A @ coeffs - pp[:, 2]) ** 2)))
            if rms > MAX_LOCAL_RMS:
                return False, "lumpy"
        except np.linalg.LinAlgError:
            pass
    return True, "ok"


def gate_area(name, pc, dem, dem_inv, only_ids=None):
    import config
    non_roof = getattr(config, "NON_ROOF_BUILDING_IDS", set())
    path = area_paths(name)["panel_layouts"]
    if not path.exists():
        print(f"{name}: no layouts, skipping")
        return
    d = json.loads(path.read_text())
    # Counter, not a fixed dict: adding a new veto reason to panel_ok() should
    # show up in the summary, never KeyError halfway through an area.
    kept, dropped = [], Counter()
    errors = 0  # gate crashes keep the panel, but MUST be visible: a systematic
    # failure here would otherwise look like a clean run that changed nothing.
    for f in d["features"]:
        if f["properties"].get("kind") != "panel" or f["geometry"]["type"] != "Polygon":
            kept.append(f)
            continue
        if only_ids is not None and f["properties"].get("building_id") not in only_ids:
            kept.append(f)          # patch mode: gate only the rebuilt buildings
            continue
        if f["properties"].get("building_id") in non_roof:
            dropped["sparse"] += 1
            continue
        try:
            poly = shp_transform(TO_NZTM, shape(f["geometry"]))
            ok, why = panel_ok(poly, pc, dem, dem_inv)
        except Exception:
            ok, why = True, "error-kept"  # never drop a panel on a gate crash
            errors += 1
        if ok:
            kept.append(f)
        else:
            dropped[why] += 1
    n_dropped = sum(dropped.values())
    d["features"] = kept
    write_json_atomic(path, d)
    reasons = ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())) or "none"
    msg = f"{name}: dropped {n_dropped} panels ({reasons})"
    if errors:
        msg += f"  [WARNING: {errors} panels kept because the gate errored]"
    print(msg)


_W = {}


def _init_gate_worker():
    import os
    os.environ["SOLAR_LAZ_SINGLE"] = "1"   # see pointcloud_source: one decode thread each
    """Per-process context: the point-cloud reader and the wide DEM. Loaded once
    per worker, not per panel."""
    # Workers x cached tiles x decoded-tile size is the real memory bill. Eight
    # workers each caching eight tiles crashed a 64 GB machine on Wellington's
    # dense survey; two tiles per worker is plenty here because a panel query
    # touches exactly the tile(s) under one building.
    _W["pc"] = PointCloudSource(max_cached_tiles=3)
    with rasterio.open(DATA_DIR / "dem_wide_mosaic.tif") as ds:
        _W["dem"] = ds.read(1)
        _W["dem_inv"] = ~ds.transform


def _gate_one(feature_json):
    f = json.loads(feature_json)
    try:
        poly = shp_transform(TO_NZTM, shape(f["geometry"]))
        minx, miny, maxx, maxy = poly.bounds
        if (maxx - minx) > 50 or (maxy - miny) > 50:
            # A panel is ~2 m. A bounds span past 50 m is corrupt geometry, and
            # its bbox query scans every tile -- one such panel wedged the
            # island_bay gate at 5,000/121,273 with ordered reporting hiding
            # everything queued behind it. Corrupt input never earns a panel.
            return feature_json, False, "corrupt-geometry"
        ok, why = panel_ok(poly, _W["pc"], _W["dem"], _W["dem_inv"])
    except Exception:
        return feature_json, True, "error-kept"
    return feature_json, ok, why



def _gate_batch(feature_jsons):
    return [_gate_one(fj) for fj in feature_jsons]


def gate_area_parallel(name, jobs=None):
    """gate_area, fanned across processes. This stage was the wall-clock floor
    of every build: single-threaded at 9-11 minutes per area while everything
    around it scaled with cores. Panels are independent, so it fans trivially;
    measured behaviour is identical because panel_ok is pure per panel."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    import config
    non_roof = getattr(config, "NON_ROOF_BUILDING_IDS", set())
    path = area_paths(name)["panel_layouts"]
    if not path.exists():
        print(f"{name}: no layouts, skipping")
        return
    d = json.loads(path.read_text())
    kept, dropped, errors = [], Counter(), 0
    todo = []
    for f in d["features"]:
        if f["properties"].get("kind") != "panel" or f["geometry"]["type"] != "Polygon":
            kept.append(f)
        elif f["properties"].get("building_id") in non_roof:
            dropped["sparse"] += 1
        else:
            todo.append(json.dumps(f))
    jobs = jobs or max(1, min(4, (os.cpu_count() or 2) - 1))
    # Spawn, never fork. On Linux the default is fork, and forked children
    # inherit the parent's initialised GDAL/rasterio state -- workers segfault
    # and the pool dies with BrokenProcessPool (seen on the VM's first run).
    # macOS spawns by default, which is why local testing never showed it.
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_gate_worker,
                             mp_context=ctx) as ex:
        # Explicit futures, completion order, hard timeout. ex.map wedged
        # deterministically on island_bay at result 5,000 with every worker
        # asleep and the data proven clean single-process -- whatever the
        # stdlib queue pathology was, ordered iteration hid all progress
        # behind it. as_completed reports truth in real time, and a future
        # that never finishes gets NAMED and counted, never waited on forever.
        from concurrent.futures import as_completed
        BATCH = 64
        done_n = 0
        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        futs = {ex.submit(_gate_batch, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futs, timeout=None):
            try:
                results = fut.result(timeout=600)
            except Exception as exc:
                print(f"  {name}: batch {futs[fut]} failed ({exc!r}); "
                      f"{BATCH} panels kept ungated", flush=True)
                results = [(fj, True, "error-kept") for fj in batches[futs[fut]]]
            for fj, ok, why in results:
                done_n += 1
                if done_n % 5000 == 0:
                    print(f"  {name}: {done_n}/{len(todo)} panels gated", flush=True)
                if why == "error-kept":
                    errors += 1
                if ok:
                    kept.append(json.loads(fj))
                else:
                    dropped[why] += 1
    n_dropped = sum(dropped.values())
    d["features"] = kept
    write_json_atomic(path, d)
    reasons = ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())) or "none"
    msg = f"{name}: dropped {n_dropped} panels ({reasons}) [{jobs} workers]"
    if errors:
        msg += f"  [WARNING: {errors} panels kept because the gate errored]"
    print(msg, flush=True)


def main():
    for name in (sys.argv[1:] or all_areas()):
        gate_area_parallel(name)


if __name__ == "__main__":
    main()
