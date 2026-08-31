"""
Run the full pipeline over every building and write per-facet and
per-panel geometries (not just building-level aggregates) to
data/panel_layouts.geojson, so the frontend can show the actual proposed
layout -- not just a kWp number -- when a building is clicked.

One FeatureCollection, features tagged with a "kind" property ("facet" or
"panel") and "building_id" so the frontend can filter to just the
clicked building's layout. Facets carry slope/aspect/irradiance;
panels carry which facet they belong to.

Runs the per-building work across processes. Buildings are independent --
segmentation, obstruction detection and packing all read shared rasters and
write nothing shared -- and the parallel wrapper around this only fans out
across AREAS, so rebuilding one area (the loop Josh actually iterates on) used
one core out of twelve and took hours. The same per-building work parallelises
to ~25 minutes for the pilot in scan_defects.py.

Usage: python src/build_layout_geojson.py [area ...] [--jobs N]
"""

import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

import pyproj
import rasterio
from shapely.ops import transform as shapely_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import geopandas as gpd

from src.roof_segmentation import segment_building_best, _area_weighted_inlier
from src.pointcloud_source import PointCloudSource
from src.panel_fitting import fit_panels_on_facet, drop_minor_arrays, assign_fill_ranks
from src.obstruction_detection import detect_obstructions_combined
from src.solar_model import SolarModel
from src.building_shading import building_shading_factor
from src.building_horizon import (far_profile as _hz_far_profile,
                                  facet_horizon_factor as _hz_facet_factor,
                                  eave_height as _hz_eave_height)
from src.region_build import area_paths, area_centroid_wgs84, areas_from_argv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEEP_SHADE_FACTOR = 0.45  # a panel keeping less than this share of the year's direct beam is
# under a tree/neighbour and should not be proposed at all

# Below this share of a roof's points lying within 30 cm of the planes we are
# about to place panels on, do not propose panels at all. Josh said it twice
# about 10 Stanley St, once unprompted and once on the comparison sheet: "very
# complicated roof, this one should probably just have no panels on it at all".
# Showing a confident-looking layout on a roof we have not understood is worse
# than showing nothing -- the number carries authority it has not earned.
# 10 Stanley measures 35% on-plane; the next worst building in the sampled
# pilot set is 52%, so this separates the genuinely unmodelled roofs rather
# than trimming a continuum.
MIN_ROOF_CONFIDENCE = 0.45
BIG_ROOF_M2 = 1000.0
BIG_ROOF_FACET_MIN_FIT = 0.60
BIG_ROOF_MIN_PANELS = 8


def _facet_fit(f, pc_source):
    """This one facet's own on-plane fraction -- how believable it is alone."""
    try:
        import shapely.vectorized
        g = f["geometry"]
        minx, miny, maxx, maxy = g.bounds
        pts = pc_source.points_in_bbox(minx, miny, maxx, maxy, building_only=True)
        if len(pts) < 12:
            return 1.0            # too few points to judge: do not punish
        inside = shapely.vectorized.contains(g, pts[:, 0], pts[:, 1])
        pts = pts[inside]
        if len(pts) < 12:
            return 1.0
        import numpy as np
        r = pts[:, 2] - (f["plane_a"] * pts[:, 0] + f["plane_b"] * pts[:, 1] + f["plane_c"])
        r = r - np.median(r)
        return float((np.abs(r) < 0.15).mean())
    except Exception:
        return 1.0

# Raised from 6. The 6 was set while chasing OOM crashes, and the actual cause
# turned out to be PointCloudSource caching all 441 LiDAR tiles unbounded --
# fixed with an LRU there -- but the cap was never put back. Measured startup is
# 0.4s per worker and steady RSS is well under a gigabyte each.
DEFAULT_MAX_JOBS = 10

# How long the whole pool may go without a single building completing before it
# says so, and before it gives up. Normal per-building time is around a second
# and the slowest roof measured on frankton_flats is 16s, so 120s of total
# silence across every worker already means something is wrong.
STALL_REPORT_S = 120
STALL_ABORT_S = 1800

# No single building may cost the district. Building 4722059 on frankton_flats
# (16,010 m2, 226 m across, 73,226 points) does not finish: partition_roof alone
# handles it in 41s, but segment_building_best compares several strategies and
# then post-processes 51 facets, and the whole call was still running after 14
# minutes. It stalled two rebuilds before the watchdog above could even name it.
# One airport-scale roof missing from the map is a far smaller loss than the
# other 14,000 buildings not shipping, so a building over budget is dropped and
# reported by id -- never silently.
BUILDING_TIME_BUDGET_S = 600  # the partition alone may now spend 240s on a
# big commercial roof (see roof_partition.CUT_TIME_BUDGET_MAX_S), so the
# whole-building alarm has to leave room for obstructions and panel fitting
# on top of that. Still bounded: a stall is cut off, just later.


class _BuildingTimeout(Exception):
    pass


def _on_timeout(signum, frame):
    raise _BuildingTimeout()

_CTX = {}


def _init_worker(area, model):
    """One heavy context per process. The SolarModel is built ONCE in the parent
    and shipped in (0.5 MB, picklable): constructing it per worker would fire a
    NASA POWER request per process."""
    paths = area_paths(area)
    dsm_ds = rasterio.open(paths["dsm"])
    dem_wide_path = DATA_DIR / "dem_wide_mosaic.tif"
    if dem_wide_path.exists():
        _dw = rasterio.open(dem_wide_path)
        _CTX.update({"dem_wide_band": _dw.read(1), "dem_wide_transform": _dw.transform,
                     "dem_wide_nodata": _dw.nodata})
    else:
        # no wide DEM shipped for this deployment -- per-building far-horizon
        # correction degrades to a no-op rather than failing the build
        _CTX.update({"dem_wide_band": None, "dem_wide_transform": None,
                     "dem_wide_nodata": None})
    _CTX.update({
        "gdf": gpd.read_file(paths["outlines"]).set_index("building_id", drop=False),
        "dsm_ds": dsm_ds,
        "dsm_band": dsm_ds.read(1),
        "imagery_ds": rasterio.open(paths["imagery"]) if paths["imagery"].exists() else None,
        "pc_source": PointCloudSource(),
        "model": model,
        "to_wgs84": pyproj.Transformer.from_crs("EPSG:2193", "EPSG:4326",
                                                always_xy=True).transform,
    })


def _build_one(building_id):
    """Everything for one building. Returns its GeoJSON features."""
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(BUILDING_TIME_BUDGET_S)
    try:
        return _build_one_inner(building_id)
    except _BuildingTimeout:
        print(f"  building {building_id} DROPPED: over {BUILDING_TIME_BUDGET_S}s budget",
              file=sys.stderr, flush=True)
        return []
    except Exception as exc:
        print(f"  building {building_id} FAILED: {exc!r}", flush=True)
        return []
    finally:
        signal.alarm(0)


def _build_one_inner(building_id):
    c = _CTX
    model, dsm_ds, dsm_band = c["model"], c["dsm_ds"], c["dsm_band"]
    to_wgs84, pc_source, imagery_ds = c["to_wgs84"], c["pc_source"], c["imagery_ds"]
    row_geom = c["gdf"].loc[building_id].geometry

    features = []
    # Imagery is passed in so the partition can cut on roof creases the LiDAR
    # cannot resolve -- see roof_partition.partition_roof. Rural areas have no
    # imagery and fall back to LiDAR-only cuts.
    facets = segment_building_best(dsm_ds, pc_source, row_geom, building_id,
                                   imagery_ds=imagery_ds)

    # Do not propose panels on a roof we have not understood -- see
    # MIN_ROOF_CONFIDENCE. Facets are still emitted so the roof draws on the
    # map; only the layout is withheld.
    confidence = _area_weighted_inlier(facets, pc_source) if facets else 0.0
    modelled = confidence >= MIN_ROOF_CONFIDENCE

    # Josh, on large commercial roofs: "it might be best to try find clear areas
    # of flat space that are very large, to place panels on. Rather than trying
    # to squeeze in every possible face", and: ignore any clean area that would
    # take fewer than 8 panels. 32 Frankton Road is the case: 4,032 m2 shipped
    # as two sheets fitting 15.3% and 16.3% carrying 1,138 panels. On big roofs
    # each facet must EARN panels: it has to be a believable plane on its own,
    # and it has to take at least 8 panels.
    big_roof = row_geom.area >= BIG_ROOF_M2

    # Per-building FAR terrain horizon (Josh, 30 Aug: every number must take
    # the building's own horizon into account). The POA lookup carries the
    # AREA's shared profile; _hz_facet_factor converts each facet to this
    # building's own terrain, aspect-aware, so a valley-floor roof loses its
    # east sun where a hilltop roof three streets over does not. Near-field
    # neighbours/trees stay with building_shading_factor below -- the far/near
    # split is what stops the same blocker being counted twice.
    _bld_far = None
    if _CTX.get("dem_wide_band") is not None:
        _eave = _hz_eave_height(dsm_band, dsm_ds.transform, dsm_ds.nodata, row_geom)
        _bld_far = _hz_far_profile(_CTX["dem_wide_band"], _CTX["dem_wide_transform"],
                                   _CTX["dem_wide_nodata"], row_geom, _eave)

    per_facet = []
    for f in facets:
        facet_centroid = f["geometry"].centroid
        shading_factor = building_shading_factor(
            dsm_band, dsm_ds.transform, dsm_ds.nodata,
            facet_centroid.x, facet_centroid.y, model.hourly,
            own_geom=f["geometry"], terrain_horizon_profile=model.horizon_profile)
        if _bld_far is not None:
            shading_factor *= _hz_facet_factor(_bld_far, model.horizon_profile,
                                               f["slope_deg"], f["aspect_deg"], model.hourly)
        facet_poa = model.annual_poa_kwh_per_m2(f["slope_deg"], f["aspect_deg"])
        if not modelled:
            per_facet.append({"facet": f, "panels": [], "obstructions": [],
                              "poa": facet_poa * shading_factor,
                              "shading_factor": shading_factor})
            continue
        plane = (f["plane_a"], f["plane_b"], f["plane_c"])
        obstructions = detect_obstructions_combined(imagery_ds, pc_source, f["geometry"], plane,
                                                    roof_geom=f.get("building_geometry"))
        siblings = [other for other in facets if other is not f]
        if big_roof and _facet_fit(f, pc_source) < BIG_ROOF_FACET_MIN_FIT:
            per_facet.append({"facet": f, "panels": [], "obstructions": obstructions,
                              "poa": facet_poa * shading_factor,
                              "shading_factor": shading_factor})
            continue
        panels = fit_panels_on_facet(f, obstructions=obstructions, sibling_facets=siblings)
        if big_roof and len(panels) < BIG_ROOF_MIN_PANELS:
            panels = []
        kept_panels = []
        for pnl in panels:
            cpt = pnl["geometry"].centroid
            psf = building_shading_factor(dsm_band, dsm_ds.transform, dsm_ds.nodata,
                                          cpt.x, cpt.y, model.hourly, own_geom=f["geometry"],
                                          terrain_horizon_profile=model.horizon_profile)
            # deep-shade veto stays a NEAR-FIELD test on purpose: the far
            # terrain correction lowers yield but must not delete panels a
            # valley still deserves
            if psf < DEEP_SHADE_FACTOR:
                continue
            if _bld_far is not None:
                psf *= _hz_facet_factor(_bld_far, model.horizon_profile,
                                        f["slope_deg"], f["aspect_deg"], model.hourly)
            pnl["poa_kwh_m2_yr"] = facet_poa * psf
            pnl["shading_factor"] = psf
            kept_panels.append(pnl)
        per_facet.append({"facet": f, "panels": kept_panels, "obstructions": obstructions,
                          "poa": facet_poa * shading_factor, "shading_factor": shading_factor})

    kept_panel_lists = drop_minor_arrays([pf["panels"] for pf in per_facet])
    all_kept = [pnl for panels in kept_panel_lists for pnl in panels]
    if all_kept:
        assign_fill_ranks(all_kept)

    for pf, panels in zip(per_facet, kept_panel_lists):
        f = pf["facet"]
        features.append({
            "type": "Feature",
            "geometry": shapely_transform(to_wgs84, f["geometry"]).__geo_interface__,
            "properties": {
                "kind": "facet",
                "building_id": int(building_id),
                "slope_deg": round(f["slope_deg"], 1),
                "aspect_deg": round(f["aspect_deg"], 1),
                "poa_kwh_m2_yr": round(pf["poa"], 0),
                "panel_count": len(panels),
                "roof_confidence": round(confidence, 2),
            },
        })
        for o in pf["obstructions"]:
            features.append({
                "type": "Feature",
                "geometry": shapely_transform(to_wgs84, o).__geo_interface__,
                "properties": {"kind": "obstruction", "building_id": int(building_id)},
            })
        for pnl in panels:
            y = model.facet_yield(f, 1, shading_factor=pnl.get("shading_factor", pf["shading_factor"]))
            features.append({
                "type": "Feature",
                "geometry": shapely_transform(to_wgs84, pnl["geometry"]).__geo_interface__,
                "properties": {
                    "kind": "panel",
                    "building_id": int(building_id),
                    "ac_kwh_year": round(y["ac_kwh_year"], 0),
                    "fill_rank": pnl["fill_rank"],
                    "fill_order": pnl["fill_order"],
                    "array_id": pnl["array_id"],
                    "array_size": pnl["array_size"],
                },
            })
    return features


def main(area="pilot", jobs=None, limit=0, dry_run=False):
    paths = area_paths(area)
    gdf = gpd.read_file(paths["outlines"])
    ids = [int(b) for b in gdf["building_id"].tolist()]
    if limit:
        ids = ids[:limit]

    # A missing imagery mosaic is ACCEPTED (rural regions genuinely have none --
    # LINZ aerial is urban-only) but it must never pass unremarked: obstruction
    # detection loses the colour half of its evidence and roof_partition loses
    # its image lines, so the build is quietly worse than one with imagery. This
    # bit us on Island Bay, where a rebuild was queued against a mosaic that had
    # been cleaned up for disk and would have shipped a degraded layout with
    # nothing in the log to say why.
    if not paths["imagery"].exists():
        print(f"[{area}] WARNING: no imagery mosaic -- building LiDAR-ONLY. "
              f"Obstruction detection and roof partitioning are degraded. "
              f"Fetch with: python src/fetch_regions.py {area}", flush=True)
    # Same rule for the wide DEM: its absence turns the per-building far-horizon
    # correction into a no-op, so yields quietly revert to the area-wide terrain
    # profile. Written as a silent fallback when the horizon work landed -- which
    # is precisely the failure mode that cost us the terrain masks and nearly
    # cost us an Island Bay rebuild.
    if not (DATA_DIR / "dem_wide_mosaic.tif").exists():
        print(f"[{area}] WARNING: no data/dem_wide_mosaic.tif -- per-building "
              f"horizons are OFF and yields fall back to the area terrain "
              f"profile.", flush=True)
    # And the point cloud, which is the PRIMARY input: without tiles over this
    # region every building silently drops to the 1 m DSM, which is roughly a
    # sixteenth of the Wellington survey's sampling and cannot resolve a hip.
    try:
        _probe = PointCloudSource()
        _b = gdf.total_bounds
        if len(_probe.points_in_bbox(_b[0], _b[1], _b[2], _b[3], building_only=True)) == 0:
            print(f"[{area}] WARNING: NO LiDAR point-cloud tiles cover this "
                  f"region -- every building falls back to the 1 m DSM. "
                  f"Fetch with: python src/fetch_pointcloud_regions.py {area}",
                  flush=True)
    except Exception as _exc:
        print(f"[{area}] WARNING: could not probe the point cloud ({_exc!r})", flush=True)

    print(f"[{area}] Building solar yield lookup table (pvlib + NASA POWER)...")
    centroid = area_centroid_wgs84(area)
    model = SolarModel() if centroid is None else SolarModel(*centroid)

    # Bounded deliberately, and not by core count. PointCloudSource caches every
    # decoded LiDAR tile for the life of its process -- the module docstring for
    # run_full_build.sh warns the full set is ~10GB decoded -- so each worker
    # carries its own copy of whatever tiles its buildings touch. Defaulting to
    # cpu_count-1 gave 11 workers on this machine and the run was killed before
    # it produced a single feature. Six is what has actually been measured
    # working, and it already gives most of the speedup (280s -> 114s on 100
    # buildings).
    jobs = jobs or min(DEFAULT_MAX_JOBS, max(1, (os.cpu_count() or 2) - 1))
    print(f"[{area}] {len(ids)} buildings on {jobs} workers", flush=True)

    features, done, t0 = [], 0, time.time()
    if jobs == 1:
        _init_worker(area, model)
        for bid in ids:
            features.extend(_build_one(bid))
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(ids)} elapsed={time.time() - t0:.1f}s", flush=True)
    else:
        # Submitted one building at a time rather than ex.map(chunksize=8), and
        # watched. ex.map yields strictly in order, so a single roof that never
        # finishes stops the counter dead while the other workers quietly drain
        # and go idle -- the build then sits at one core pegged at 100% with no
        # error, no log line, and no way to tell which building is responsible.
        # That happened twice on frankton_flats on 28 Aug and cost the evening;
        # the second time it burned an hour before anyone looked at ps. Per-task
        # dispatch overhead is microseconds against ~1s of work per building, so
        # the chunking was buying nothing worth this.
        results = {}
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(area, model)) as ex:
            futs = {ex.submit(_build_one, b): b for b in ids}
            pending, last_progress = set(futs), time.time()
            while pending:
                finished, pending = wait(pending, timeout=STALL_REPORT_S)
                for fut in finished:
                    results[futs[fut]] = fut.result()
                    done += 1
                    if done % 200 == 0:
                        print(f"  {done}/{len(ids)} elapsed={time.time() - t0:.1f}s",
                              flush=True)
                if finished:
                    last_progress = time.time()
                    continue
                stalled = time.time() - last_progress
                outstanding = sorted(futs[f] for f in pending)
                print(f"  [STALL] no building has completed in {stalled:.0f}s. "
                      f"{len(outstanding)} outstanding: {outstanding[:10]}"
                      f"{' ...' if len(outstanding) > 10 else ''}",
                      file=sys.stderr, flush=True)
                if stalled > STALL_ABORT_S:
                    # Abort loudly rather than pretend. A build that silently
                    # drops buildings would publish a map with holes in it.
                    for f in pending:
                        f.cancel()
                    raise RuntimeError(
                        f"[{area}] build stalled {stalled:.0f}s with "
                        f"{len(outstanding)} buildings outstanding: {outstanding[:20]}")
        # emitted in the input order so two builds of the same area diff cleanly
        for b in ids:
            features.extend(results.get(b, []))

    if dry_run:
        print(f"[{area}] dry run: {len(features)} features in {time.time() - t0:.0f}s "
              f"on {jobs} workers -- nothing written")
        return
    geojson = {"type": "FeatureCollection", "features": features}
    out_path = paths["panel_layouts"]
    out_path.write_text(json.dumps(geojson))

    n_facets = sum(1 for f in features if f["properties"]["kind"] == "facet")
    n_panels = sum(1 for f in features if f["properties"]["kind"] == "panel")
    n_obs = sum(1 for f in features if f["properties"]["kind"] == "obstruction")
    gated = len({f["properties"]["building_id"] for f in features
                 if f["properties"]["kind"] == "facet"
                 and f["properties"].get("roof_confidence", 1.0) < MIN_ROOF_CONFIDENCE})
    print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f}MB) in {time.time() - t0:.0f}s")
    print(f"{n_facets} facets, {n_panels} panels, {n_obs} obstructions across {len(ids)} buildings")
    print(f"{gated} buildings withheld as not confidently modelled (<{MIN_ROOF_CONFIDENCE:.0%} on-plane)")


if __name__ == "__main__":
    _jobs, _limit = None, 0
    _argv = sys.argv[:]
    for _flag in ("--jobs", "--limit"):
        if _flag in _argv:
            _i = _argv.index(_flag)
            _val = int(_argv[_i + 1])
            _argv = _argv[:_i] + _argv[_i + 2:]
            if _flag == "--jobs":
                _jobs = _val
            else:
                _limit = _val
    _dry = "--dry-run" in _argv
    _argv = [a for a in _argv if a != "--dry-run"]
    for _area in areas_from_argv(_argv):
        main(_area, jobs=_jobs, limit=_limit, dry_run=_dry)
