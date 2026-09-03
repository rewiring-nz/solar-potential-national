"""
Assert a stage's inputs exist BEFORE it runs, and stop if they don't.

Three separate failures this month shared one shape: an input was missing, no
stage raised, and the pipeline produced plausible output that was quietly
worse. That is the most expensive failure mode this project has, because the
result looks fine and nobody goes looking.

  1. `data/dem_wide_mosaic.tif` was never shipped to the VM. The gate worker's
     initializer opened it, every worker died, the pool raised
     BrokenProcessPool -- and panels shipped UNGATED to the public site.
  2. `build_terrain_masks` ran before `merge_regions`. Masks write only into
     the merged file; merge regenerates that file from the region files. The
     masks were silently wiped and terrain shading became a no-op.
  3. The just-in-time imagery cleanup deleted mosaics that a later stage
     needed, and the scorecard crashed on them.

Each was found by accident, days later. This module makes the first two
impossible and the third loud.

Two kinds of check, because the incidents were two different kinds of bug:

  EXISTENCE -- the named inputs are present and non-empty. A zero-byte raster
  is worse than a missing one: rasterio opens it and the failure surfaces
  somewhere far away.

  FRESHNESS -- an ordering constraint expressed as mtimes. `build_terrain_masks`
  writes only into the merged file, so it is only correct if the merged file is
  at least as new as every region file it was merged from. That single rule
  catches incident 2 without anyone having to remember the stage order.

Usage as a library (preferred -- then it cannot be forgotten):

    from src.preflight import preflight
    preflight("gate_panels", area)

Usage from a shell script, before a long chain:

    python src/preflight.py gate_panels queenstown_hill
    python src/preflight.py --all-stages pilot

Exits non-zero and prints what is missing and how to get it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# A district merge needs most regions actually built. Not 100%: a region can
# legitimately have no layouts (no imagery, no buildings), and demanding
# perfection would make the guard something people route around.
MERGE_MIN_BUILT_FRACTION = 0.8

# How much of a region's building extent the DSM must overlap before the build
# is allowed to proceed. Deliberately loose: real regions have irregular LiDAR
# edges and a bounding-box overlap is a crude proxy. This is set to catch the
# catastrophic case (arrowtown_hills: 0% overlap, whole region built as zeros),
# not to police normal survey boundaries.
DSM_COVERAGE_MIN = 0.20


class PreflightError(SystemExit):
    """SystemExit so an unguarded stage still dies with a readable message
    rather than a traceback the shell script will bury in a log."""


# What each stage reads. Keys of `region` are area_paths() keys; `root` entries
# are filenames under data/. `optional` inputs degrade the result but do not
# invalidate it -- they warn instead of stopping.
#
# Adding a stage here is cheap and worth doing when you add one: the cost of a
# missing entry is another silent-degradation incident.
REQUIRED = {
    "build_layout_geojson": {
        "region": ["outlines", "dsm"],
        "root": ["dem_wide_mosaic.tif"],
        "optional_region": ["imagery"],
    },
    "gate_panels": {
        # The incident. gate_area opens the wide DEM in the worker initializer,
        # so a missing file kills every worker at once and the pool error says
        # nothing about DEMs.
        "region": ["panel_layouts"],
        "root": ["dem_wide_mosaic.tif"],
    },
    "rerank_layouts":        {"region": ["panel_layouts"]},
    "derive_solar_potential": {"region": ["panel_layouts", "outlines"]},
    "patch_roof_confidence":  {"region": ["panel_layouts", "solar_potential"]},
    "bake_building_horizons": {
        "region": ["solar_potential", "outlines", "dsm"],
        "root": ["dem_wide_mosaic.tif"],
    },
    "add_addresses":          {"region": ["solar_potential"]},
    "build_heatmap_raster":   {"region": ["solar_potential", "outlines", "dsm"]},
    # merge_regions REGENERATES the district files from the region files, so a
    # full merge run while most regions are missing their outputs replaces a
    # complete district with a partial one -- destructively, and with only a
    # per-region WARNING line in a log nobody reads. Same shape as the
    # terrain-mask incident. Checked below rather than here, because it is a
    # proportion rather than a list of names.
    "merge_regions":          {"most_regions_built": True},
    "bake_density_deciles":   {"root": ["panel_layouts.geojson", "solar_potential.geojson"]},
    "shrink_panels_for_tiles": {"root": ["panel_layouts.geojson"]},
    "build_terrain_masks": {
        "root": ["solar_potential.geojson", "dem_wide_mosaic.tif"],
        # Incident 2, as a checkable invariant rather than a comment.
        "fresher_than_regions": "solar_potential.geojson",
    },
    "build_seasonal_curves":  {"root": ["solar_potential.geojson"]},
}

# What to tell someone whose input is missing. Worth keeping accurate: the
# whole point is that the message arrives before hours of compute, not after.
HOW_TO_GET = {
    "dem_wide_mosaic.tif":
        "NO script in this repo builds it -- copy it from a machine that has "
        "one (it is gitignored, so it does NOT travel with a clone). This is "
        "exactly how the ungated-panel incident happened: a fresh VM had every "
        "other input and silently lacked this one.",
    "outlines":  "python src/fetch_regions.py <region>",
    "dsm":       "python src/fetch_regions.py <region>  (pass 1: LiDAR/DSM)",
    "imagery":   "python src/fetch_regions.py <region>  (pass 2: aerial imagery)",
    "panel_layouts":   "python src/build_layout_geojson.py <region>",
    "solar_potential": "python src/derive_solar_potential.py <region>",
    "panel_layouts.geojson":   "python src/merge_regions.py",
    "solar_potential.geojson": "python src/merge_regions.py",
}


def _describe(path, key):
    hint = HOW_TO_GET.get(key) or HOW_TO_GET.get(Path(key).name)
    return f"    {path}\n      get it with: {hint}" if hint else f"    {path}"


def _check_file(path, key, problems, what="missing"):
    if not path.exists():
        problems.append((f"{what}: {key}", _describe(path, key)))
        return False
    if path.stat().st_size == 0:
        # A zero-byte raster opens without error and fails far from here.
        problems.append((f"empty (0 bytes): {key}", _describe(path, key)))
        return False
    return True


def _region_files_newer_than(root_name):
    """Region files newer than the merged file mean the merge is stale --
    anything that writes ONLY into the merged file is about to be wiped by the
    next merge, or is about to overwrite fresher per-region work."""
    from src.region_build import all_areas, area_paths
    merged = DATA_DIR / root_name
    if not merged.exists():
        return []
    m = merged.stat().st_mtime
    stale = []
    for name in all_areas():
        p = area_paths(name).get(Path(root_name).stem.replace(".geojson", ""))
        if p is None:
            p = area_paths(name)["dir"] / root_name
        if p.exists() and p.stat().st_mtime > m + 1.0:
            stale.append(name)
    return stale


def preflight(stage, region=None, fatal=True):
    """Check `stage`'s inputs. Raises PreflightError unless fatal=False, in
    which case it returns the list of problems for the caller to decide on."""
    spec = REQUIRED.get(stage)
    if spec is None:
        return []                      # unknown stage: never block on ignorance

    problems, warnings = [], []

    if spec.get("region"):
        if region is None:
            raise PreflightError(f"preflight: stage {stage} needs a region name")
        from src.region_build import area_paths
        paths = area_paths(region)
        for key in spec["region"]:
            _check_file(paths[key], key, problems)

    for key in spec.get("optional_region", []):
        if region is not None:
            from src.region_build import area_paths
            p = area_paths(region)[key]
            if not p.exists() or p.stat().st_size == 0:
                warnings.append((f"optional input absent: {key}", _describe(p, key)))

    # DOES THE DSM ACTUALLY COVER THE BUILDINGS? A present, valid, openable DSM
    # can still describe the wrong ground -- arrowtown_hills' covers ground
    # 340 m west of all 50 of its buildings.
    #
    # BUT A GAP IS NOT AUTOMATICALLY A BUG, and the first version of this check
    # got that wrong badly enough to have blocked a correct build. Where the
    # LiDAR survey genuinely stops, the pipeline already does the right thing:
    # _diagnose_no_facets returns "no_lidar" and every affected building carries
    # "Not enough laser survey data over this roof". arrowtown_hills' 50
    # buildings are all correctly marked that way, and arrowtown_east's 132.
    # That is the designed behaviour, not a failure.
    #
    # So the extent gap is only worth reporting when the POINT CLOUD disagrees
    # with it -- points present where the DSM says nothing means a raster that
    # came back short while the data exists. Points absent too means the survey
    # simply ends there, and the build should proceed and label it.
    if spec.get("region") and region is not None and "dsm" in spec.get("region", []):
        try:
            from src.region_build import area_paths
            p = area_paths(region)
            if p["dsm"].exists() and p["outlines"].exists():
                import rasterio
                import geopandas as gpd
                with rasterio.open(p["dsm"]) as ds:
                    dx0, dy0, dx1, dy1 = ds.bounds
                dd = p["dir"] / "building_outlines_dedup.geojson"
                gdf = gpd.read_file(dd if dd.exists() else p["outlines"])
                bx0, by0, bx1, by1 = gdf.total_bounds
                ox = max(0.0, min(dx1, bx1) - max(dx0, bx0))
                oy = max(0.0, min(dy1, by1) - max(dy0, by0))
                bw = max(bx1 - bx0, 1e-9)
                bh = max(by1 - by0, 1e-9)
                frac = (ox / bw) * (oy / bh)
                if frac < DSM_COVERAGE_MIN:
                    # Is there point-cloud data where the DSM has none? If not,
                    # the survey ends here and the build will correctly mark
                    # those buildings no_lidar -- let it run.
                    has_points = False
                    try:
                        from src.pointcloud_source import PointCloudSource
                        pcs = PointCloudSource(max_cached_tiles=1)
                        for geom in list(gdf.geometry)[:20]:
                            pts = pcs.points_in_bbox(*geom.buffer(2).bounds)
                            if pts is not None and len(pts) > 50:
                                has_points = True
                                break
                    except Exception:
                        has_points = True     # cannot tell: do not block
                    extents = (
                        f"    dsm bounds      "
                        f"{[round(v) for v in (dx0, dy0, dx1, dy1)]}"
                        f"\n      building bounds "
                        f"{[round(v) for v in (bx0, by0, bx1, by1)]}")
                    if not has_points:
                        warnings.append((
                            f"DSM covers {100 * frac:.0f}% of this region's "
                            f"buildings, and there is no point cloud either",
                            extents
                            + "\n      The LiDAR survey does not reach this"
                              " region. Building anyway:"
                              "\n      every affected roof is marked 'Not enough"
                              " laser survey data'."))
                    else:
                        problems.append((
                            f"DSM covers only {100 * frac:.0f}% of this "
                            f"region's buildings, but the point cloud HAS data "
                            f"there",
                            extents
                            + "\n      The export came back smaller than"
                              " requested while the survey data exists."
                              "\n      Re-fetch it:  rm data/regions/"
                            + f"{region}/dsm_mosaic.tif && "
                              f"python src/fetch_regions.py {region}"))
        except Exception:
            pass          # a diagnostic must never be the thing that breaks a build

    for name in spec.get("root", []):
        _check_file(DATA_DIR / name, name, problems)

    # A DISTRICT-wide merge (no region named) must not run against a mostly
    # empty tree. Merging a named subset is a normal, deliberate operation --
    # the Wellington deploy does exactly that -- so the check applies only when
    # no region was given.
    if spec.get("most_regions_built") and region is None:
        from src.region_build import all_areas, area_paths
        areas = list(all_areas())
        built = [a for a in areas if area_paths(a)["panel_layouts"].exists()]
        if areas and len(built) < MERGE_MIN_BUILT_FRACTION * len(areas):
            missing = sorted(set(areas) - set(built))
            problems.append((
                f"district merge with only {len(built)} of {len(areas)} regions built",
                "    missing layouts: " + ", ".join(missing[:8])
                + ("..." if len(missing) > 8 else "")
                + "\n      merge_regions REGENERATES the district files from the"
                  "\n      region files, so running it now would REPLACE the full"
                  "\n      district with a partial one."
                  "\n      If that is genuinely what you want, merge the regions"
                  "\n      you mean by name: python src/merge_regions.py <region>..."))

    fresh_target = spec.get("fresher_than_regions")
    if fresh_target and not problems:
        stale = _region_files_newer_than(fresh_target)
        if stale:
            problems.append((
                f"stale merge: data/{fresh_target} is older than region files",
                "    regions rebuilt since the last merge: "
                + ", ".join(sorted(stale)[:8])
                + ("..." if len(stale) > 8 else "")
                + "\n      This stage writes ONLY into the merged file, and"
                  "\n      merge_regions regenerates that file from the region"
                  "\n      files -- so its work would be silently wiped."
                  "\n      Run: python src/merge_regions.py   first."))

    label = f"{stage}" + (f" [{region}]" if region else "")
    for head, detail in warnings:
        print(f"[{label}] WARNING {head}\n{detail}", flush=True)

    if problems and fatal:
        lines = [f"preflight FAILED for {label} -- not starting, because the "
                 f"output would be wrong rather than absent:", ""]
        for head, detail in problems:
            lines.append(f"  {head}")
            lines.append(detail)
        lines.append("")
        raise PreflightError("\n".join(lines))

    return problems


def main():
    args = [a for a in sys.argv[1:]]
    if "--all-stages" in args:
        args.remove("--all-stages")
        region = args[0] if args else None
        bad = 0
        for stage in REQUIRED:
            try:
                preflight(stage, region)
                print(f"  ok    {stage}")
            except SystemExit as e:
                bad += 1
                print(f"  FAIL  {stage}\n{e}")
        return 1 if bad else 0

    if not args:
        print(__doc__.strip().splitlines()[0])
        print("usage: python src/preflight.py <stage> [region]")
        print("       python src/preflight.py --all-stages [region]")
        print("stages: " + ", ".join(sorted(REQUIRED)))
        return 2

    stage = args[0]
    region = args[1] if len(args) > 1 else None
    preflight(stage, region)
    print(f"preflight ok: {stage}" + (f" [{region}]" if region else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
