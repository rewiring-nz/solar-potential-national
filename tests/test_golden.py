"""
Golden tests: pin what the segmentation currently produces for real buildings.

These are REGRESSION tests, not correctness tests. score_all_marked.py already
measures us against Josh's ground truth and answers "is this roof right?".
This file answers a different and, for refactoring, more urgent question:
"did anything change that I did not mean to change?"

That is the missing half of the safety net. roof_segmentation.py is 2,979
lines mixing plane fitting, facet merging, repair and geometry attachment, and
the code review's recommendation to split it is not safe to act on while the
only way to notice a behaviour change is to look at renders by eye. With these
in place, a refactor that quietly moves a facet boundary fails in seconds.

The recorded values are whatever the code produced when they were recorded --
they carry no claim of being right. When a deliberate improvement moves them,
re-record and SAY SO in the commit, with the reason. An unexplained diff here
is a bug; an explained one is a decision.

Requires the region's DSM, point cloud and imagery, which are gitignored. On a
machine without them each affected building SKIPS rather than fails, so this
is useful on a build machine and harmless anywhere else.

Run:      .venv/bin/python tests/test_golden.py
Re-record: .venv/bin/python tests/test_golden.py --record
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN = Path(__file__).resolve().parent / "golden_buildings.json"

# Tolerances. Facet COUNT is exact -- a changed count is always worth a look.
# Areas get a little room for floating-point and library-version noise, but not
# enough to hide a moved boundary: 1% of a 60 m2 facet is 0.6 m2, well under
# the size of any real geometry change.
AREA_REL_TOL = 0.01


def _open_area(area):
    """Rasters and point cloud for one area, or None if this machine lacks them."""
    import rasterio
    from src.region_build import area_paths
    from src.pointcloud_source import PointCloudSource
    p = area_paths(area)
    if not p["outlines"].exists() or not p["dsm"].exists():
        return None
    import geopandas as gpd
    return {
        "gdf": gpd.read_file(p["outlines"]).set_index("building_id", drop=False),
        "dsm": rasterio.open(p["dsm"]),
        "imagery": rasterio.open(p["imagery"]) if p["imagery"].exists() else None,
        "pc": PointCloudSource(),
    }


def _measure(ctx, building_id):
    """The shape of one building's segmentation, as comparable numbers."""
    from src.roof_segmentation import segment_building_best
    if building_id not in ctx["gdf"].index:
        return None
    geom = ctx["gdf"].loc[building_id, "geometry"]
    facets = segment_building_best(ctx["dsm"], ctx["pc"], geom, building_id,
                                   imagery_ds=ctx["imagery"]) or []
    return {
        "facets": len(facets),
        "areas": sorted(round(f["geometry"].area, 2) for f in facets),
        "slopes": sorted(round(float(f["slope_deg"]), 1) for f in facets),
        "total_area": round(sum(f["geometry"].area for f in facets), 2),
    }


def _compare(name, got, want):
    """Returns a list of human-readable differences."""
    out = []
    if got["facets"] != want["facets"]:
        out.append(f"facet count {want['facets']} -> {got['facets']}")
    if abs(got["total_area"] - want["total_area"]) > AREA_REL_TOL * max(want["total_area"], 1):
        out.append(f"total area {want['total_area']} -> {got['total_area']} m2")
    if got["facets"] == want["facets"]:
        for a, b in zip(want["areas"], got["areas"]):
            if abs(a - b) > AREA_REL_TOL * max(a, 1):
                out.append(f"facet area {a} -> {b} m2")
        for a, b in zip(want["slopes"], got["slopes"]):
            if abs(a - b) > 0.5:
                out.append(f"facet slope {a} -> {b} deg")
    return out


def record():
    """Re-measure every listed building and overwrite the golden file."""
    spec = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {"buildings": {}}
    by_area = {}
    for bid, rec in spec["buildings"].items():
        by_area.setdefault(rec["area"], []).append(bid)

    for area, bids in sorted(by_area.items()):
        ctx = _open_area(area)
        if ctx is None:
            print(f"  skip {area}: data not on this machine")
            continue
        for bid in bids:
            m = _measure(ctx, int(bid))
            if m is None:
                print(f"  skip {bid}: not in {area} outlines")
                continue
            spec["buildings"][bid].update(m)
            print(f"  recorded {bid} ({area}): {m['facets']} facets, "
                  f"{m['total_area']} m2")
    spec["_note"] = ("Recorded output of segment_building_best, NOT ground truth. "
                     "Re-record deliberately and explain the change in the commit.")
    GOLDEN.write_text(json.dumps(spec, indent=1, sort_keys=True))
    print(f"\nwrote {GOLDEN}")
    return 0


def main():
    if "--record" in sys.argv:
        return record()
    if not GOLDEN.exists():
        print(f"no golden file at {GOLDEN} -- run with --record first")
        return 2

    spec = json.loads(GOLDEN.read_text())
    by_area = {}
    for bid, rec in spec["buildings"].items():
        if "facets" in rec:                      # only recorded ones are testable
            by_area.setdefault(rec["area"], []).append(bid)

    checked = skipped = 0
    failures = []
    for area, bids in sorted(by_area.items()):
        ctx = _open_area(area)
        if ctx is None:
            skipped += len(bids)
            print(f"  skip  {area} ({len(bids)} buildings): data not on this machine")
            continue
        for bid in bids:
            want = spec["buildings"][bid]
            got = _measure(ctx, int(bid))
            if got is None:
                skipped += 1
                print(f"  skip  {bid}: not in {area} outlines")
                continue
            diffs = _compare(bid, got, want)
            checked += 1
            if diffs:
                failures.append((bid, want.get("address", ""), diffs))
                print(f"  FAIL  {bid} {want.get('address','')}")
                for d in diffs[:6]:
                    print(f"          {d}")
            else:
                print(f"  pass  {bid} {want.get('address','')} "
                      f"({got['facets']} facets)")

    print(f"\n{checked - len(failures)}/{checked} matched"
          + (f", {skipped} skipped (no data)" if skipped else ""))
    if failures:
        print("\nIf these changes are DELIBERATE, re-record with:")
        print("  .venv/bin/python tests/test_golden.py --record")
        print("and explain in the commit what moved and why.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
