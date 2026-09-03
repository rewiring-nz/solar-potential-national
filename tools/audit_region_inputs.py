"""
Check every region's inputs for being present-but-wrong.

Written because two failures in the 3 Sep district run both passed every check
the build had, and both produced output indistinguishable from a real result:

  arrowtown_hills   a DSM that was present, valid, openable and correctly
                    projected, and described ground 340 m west of every
                    building in the region. 50 buildings built as zeros.
  14 regions        imagery_mosaic.tif absent, so obstruction detection ran
                    without its input and found ~31% fewer obstructions. The
                    build logged a warning and carried on.

Every check in preflight was a PRESENCE check, and presence was never the
problem. So this asks the questions that actually distinguish good input from
bad, per region:

  COVERAGE   does the raster's extent overlap the buildings it is supposed to
             describe? This is the arrowtown_hills check.
  DATA       within the buildings, is there real data, or is the overlap
             nodata? A DSM can cover a region on paper and be empty over it.
  CRS        do outlines, DSM and imagery agree on what the coordinates mean? A
             silent CRS mismatch produces exactly the arrowtown_hills symptom
             without the extent looking wrong.
  BLANK      is the imagery actually imagery, or black tiles? A failed export
             can write a structurally perfect all-zero raster.

Sampling, not exhaustive: a fixed sample of buildings per region, enough to
tell "this region is broken" from "this region is fine", which is the only
distinction being made here. Cheap enough to run before every district build.

Usage:
    python tools/audit_region_inputs.py
    python tools/audit_region_inputs.py --region arrowtown_hills --verbose
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_BUILDINGS = 40      # enough to separate broken from fine
MIN_COVERAGE = 0.20        # bounding-box overlap, loose: LiDAR edges are ragged
# A second, much stricter tier that warns rather than fails. The fail threshold
# has to stay loose or it condemns regions with ragged survey edges, but loose
# thresholds hide the half-broken case: arrowtown_east passed at 45% DSM
# coverage while every healthy region sits at 97-100%, and it was
# simultaneously the worst-performing region in the district (1.89 facets and
# 17.6 panels per building against ~4.9 and ~46). "Passes the floor" and
# "looks like its neighbours" are different questions and both are worth asking.
WARN_COVERAGE = 0.90
# A raster touched this recently is very likely still being written. Merges take
# minutes and produce a valid-but-empty file the whole time.
MID_WRITE_S = 180
MIN_DATA_FRAC = 0.30       # of sampled buildings, this many must have real data
BLANK_STD = 1.0            # imagery with less variation than this is not imagery


def _bbox_overlap_frac(rb, gb):
    """Fraction of the building extent covered by the raster extent."""
    ox = max(0.0, min(rb[2], gb[2]) - max(rb[0], gb[0]))
    oy = max(0.0, min(rb[3], gb[3]) - max(rb[1], gb[1]))
    return (ox / max(gb[2] - gb[0], 1e-9)) * (oy / max(gb[3] - gb[1], 1e-9))


def audit_region(name, verbose=False):
    import numpy as np
    import geopandas as gpd
    import rasterio
    import rasterio.windows
    from src.region_build import area_paths

    p = area_paths(name)
    out = {"region": name, "problems": [], "notes": []}

    dd = p["dir"] / "building_outlines_dedup.geojson"
    src = dd if dd.exists() else p["outlines"]
    if not src.exists():
        out["problems"].append("no building outlines")
        return out
    gdf = gpd.read_file(src)
    if not len(gdf):
        out["problems"].append("building outlines file is empty")
        return out
    out["buildings"] = len(gdf)
    gb = tuple(gdf.total_bounds)

    # A CRS mismatch produces the arrowtown_hills symptom without the extents
    # themselves looking suspicious, so it is checked before anything spatial.
    if gdf.crs is not None and gdf.crs.to_epsg() not in (2193, None):
        out["problems"].append(f"outlines are EPSG:{gdf.crs.to_epsg()}, expected 2193")

    geoms = list(gdf.geometry)
    step = max(1, len(geoms) // SAMPLE_BUILDINGS)
    sample = geoms[::step][:SAMPLE_BUILDINGS]

    for key, label in (("dsm", "DSM"), ("imagery", "imagery")):
        path = p[key]
        if not path.exists():
            # imagery is optional to the build; the DSM is not
            (out["problems"] if key == "dsm" else out["notes"]).append(
                f"{label} missing")
            continue
        # A file still being written reads as structurally valid and mostly
        # empty, which is indistinguishable from a blank export. Caught this
        # for real: auditing during tools/repair_imagery.sh reported
        # frankton_arm's imagery as "nearly featureless, likely a blank export"
        # while rasterio was three minutes into merging it. Reporting a false
        # failure is worse than reporting nothing, because it sends someone to
        # re-fetch several GB that were fine.
        import time
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            age = 1e9
        if age < MID_WRITE_S:
            out["notes"].append(
                f"{label} was modified {age:.0f}s ago -- probably still being "
                f"written; re-run the audit once it settles")
            continue

        try:
            ds = rasterio.open(path)
        except Exception as e:
            out["problems"].append(f"{label} will not open: {type(e).__name__}")
            continue

        if ds.crs is not None and ds.crs.to_epsg() != 2193:
            out["problems"].append(
                f"{label} is EPSG:{ds.crs.to_epsg()}, expected 2193")

        cov = _bbox_overlap_frac(tuple(ds.bounds), gb)
        out[f"{key}_coverage"] = cov
        if cov < MIN_COVERAGE:
            # A GAP IS NOT AUTOMATICALLY A FAULT, and this check cried wolf on
            # every build until it learned the difference. arrowtown_hills has
            # 0% DSM coverage and is handled correctly: the survey does not
            # reach it, so every roof there is marked no_lidar -- "Not enough
            # laser survey data over this roof" -- which is the designed
            # behaviour, not a failure. Flagging it as a PROBLEM on every run
            # trains everyone to skim past the section.
            #
            # The gap only matters when the POINT CLOUD disagrees with it:
            # points where the raster has none means an export came back short
            # while the data exists. Same rule preflight uses.
            has_points = False
            try:
                from src.pointcloud_source import PointCloudSource
                pcs = PointCloudSource(max_cached_tiles=1)
                for g in sample[:20]:
                    pts = pcs.points_in_bbox(*g.buffer(2).bounds)
                    if pts is not None and len(pts) > 50:
                        has_points = True
                        break
            except Exception:
                has_points = True          # cannot tell: report it
            if has_points:
                out["problems"].append(
                    f"{label} covers {100 * cov:.0f}% of the buildings but the "
                    f"point cloud HAS data there "
                    f"(raster {[round(v) for v in ds.bounds]} vs "
                    f"buildings {[round(v) for v in gb]})")
            else:
                out["notes"].append(
                    f"{label} covers {100 * cov:.0f}% and there is no point "
                    f"cloud either -- the survey does not reach this region, "
                    f"every roof will be marked no_lidar")
            ds.close()
            continue
        if cov < WARN_COVERAGE:
            out["notes"].append(
                f"{label} covers only {100 * cov:.0f}% of the buildings -- "
                f"healthy regions sit at 97-100%, so part of this region is "
                f"being built on nothing")

        # Coverage on paper is not data on the ground.
        nod = ds.nodata
        withdata = 0
        stds = []
        for g in sample:
            try:
                w = rasterio.windows.from_bounds(*g.bounds, ds.transform)
                a = ds.read(1, window=w, boundless=True,
                            fill_value=nod if nod is not None else 0)
            except Exception:
                continue
            if a.size == 0:
                continue
            fin = np.isfinite(a)
            if nod is not None:
                fin &= (a != nod)
            if fin.mean() > 0.3:
                withdata += 1
                if key == "imagery":
                    stds.append(float(a[fin].std()) if fin.any() else 0.0)
        frac = withdata / max(len(sample), 1)
        out[f"{key}_data_frac"] = frac
        if frac < MIN_DATA_FRAC:
            out["problems"].append(
                f"{label} has real data over only {100 * frac:.0f}% of sampled "
                f"buildings -- covered on paper, empty in practice")
        if key == "imagery" and stds:
            med = sorted(stds)[len(stds) // 2]
            out["imagery_std"] = med
            if med < BLANK_STD:
                out["problems"].append(
                    f"imagery is nearly featureless over buildings "
                    f"(median std {med:.2f}) -- likely a blank export")
        ds.close()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    from src.region_build import all_areas

    regions = [a.region] if a.region else list(all_areas())
    bad, warn = [], []
    print(f"auditing {len(regions)} regions "
          f"({SAMPLE_BUILDINGS} sampled buildings each)\n")
    # Two columns per raster, because they catch different failures. COVERAGE is
    # bounding-box overlap and catches a wholly displaced export; ON-DATA is the
    # share of sampled buildings actually sitting on real pixels, and is the only
    # one that sees a hole in the middle of an extent that looks complete.
    print(f"  {'region':22s} {'bldgs':>6s} {'dsm cov':>8s} {'dsm data':>9s} "
          f"{'img cov':>8s} {'img data':>9s}  status")
    for r in regions:
        try:
            o = audit_region(r, a.verbose)
        except Exception as e:
            print(f"  {r:24s} audit ERROR {type(e).__name__}: {e}")
            bad.append(r)
            continue
        dsm = o.get("dsm_coverage")
        img = o.get("imagery_coverage")
        status = "ok" if not o["problems"] else "PROBLEM"
        if o["problems"]:
            bad.append(r)
        elif o["notes"]:
            warn.append(r)
            status = "ok (" + "; ".join(o["notes"]) + ")"
        def pc(v):
            return ("%.0f%%" % (100 * v)) if v is not None else "-"
        print(f"  {r:22s} {o.get('buildings', 0):>6d} "
              f"{pc(dsm):>8s} {pc(o.get('dsm_data_frac')):>9s} "
              f"{pc(img):>8s} {pc(o.get('imagery_data_frac')):>9s}  {status}")
        for pr in o["problems"]:
            print(f"      -> {pr}")

    print(f"\n{len(bad)} regions with problems, {len(warn)} with notes, "
          f"{len(regions) - len(bad) - len(warn)} clean")
    if bad:
        print("\nThese would build to completion and produce plausible-looking")
        print("output. That is what makes them worth catching here.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
