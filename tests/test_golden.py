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
    """Prepare one area, or None if this machine lacks its data.

    Deliberately sets up the REAL build worker rather than calling segmentation
    directly. build_layout_geojson._build_one is the exact per-building path a
    district rebuild runs, so what these tests pin is what actually ships:
    facets, obstructions, panels and annual kWh -- not a reimplementation that
    could drift away from the pipeline while still passing.
    """
    from src.region_build import area_paths, area_centroid_wgs84
    p = area_paths(area)
    if not p["outlines"].exists() or not p["dsm"].exists():
        return None
    import geopandas as gpd
    from src.solar_model import SolarModel
    import src.build_layout_geojson as B
    try:
        c = area_centroid_wgs84(area)
        model = SolarModel() if c is None else SolarModel(*c)
        B._init_worker(area, model)
    except Exception as exc:
        print(f"  (cannot prepare {area}: {type(exc).__name__}: {exc})")
        return None
    return {
        "build_one": B._build_one,
        "ids": set(gpd.read_file(p["outlines"])["building_id"].tolist()),
    }


def _measure(ctx, building_id):
    """One building through the real pipeline, as comparable numbers."""
    if building_id not in ctx["ids"]:
        return None
    feats = ctx["build_one"](building_id) or []
    facets = [f for f in feats if f["properties"]["kind"] == "facet"]
    panels = [f for f in feats if f["properties"]["kind"] == "panel"]
    obstr = [f for f in feats if f["properties"]["kind"] == "obstruction"]
    kwh = sum(f["properties"].get("ac_kwh_year") or 0.0 for f in panels)
    return {
        "facets": len(facets),
        "panels": len(panels),
        "obstructions": len(obstr),
        "ac_kwh_year": round(kwh, 1),
        # NOT areas: _build_one emits WGS84 geometry, where .area is degrees
        # squared and means nothing. What the pipeline does carry per facet is
        # its orientation, its irradiance and how many panels it took -- which
        # are closer to the published numbers anyway.
        "slopes": sorted(round(float(f["properties"].get("slope_deg") or 0.0), 1)
                         for f in facets),
        "aspects": sorted(round(float(f["properties"].get("aspect_deg") or 0.0), 1)
                          for f in facets),
        "facet_panels": sorted(int(f["properties"].get("panel_count") or 0)
                               for f in facets),
        "poa": sorted(round(float(f["properties"].get("poa_kwh_m2_yr") or 0.0), 1)
                      for f in facets),
    }


def _compare(name, got, want):
    """Returns a list of human-readable differences."""
    out = []
    for key, label in (("facets", "facet count"),
                       ("panels", "panel count"),
                       ("obstructions", "obstruction count")):
        if key in want and got.get(key) != want[key]:
            out.append(f"{label} {want[key]} -> {got.get(key)}")
    # The published number. 0.5% of a 13,000 kWh roof is 65 kWh -- tight enough
    # that a real modelling change shows up, loose enough to survive rounding.
    if "ac_kwh_year" in want and want["ac_kwh_year"]:
        if abs(got.get("ac_kwh_year", 0) - want["ac_kwh_year"]) > 0.005 * want["ac_kwh_year"]:
            out.append(f"annual yield {want['ac_kwh_year']} -> "
                       f"{got.get('ac_kwh_year')} kWh")
    if got["facets"] == want["facets"]:
        for a, b in zip(want.get("slopes", []), got["slopes"]):
            if abs(a - b) > 0.5:
                out.append(f"facet slope {a} -> {b} deg")
        for a, b in zip(want.get("aspects", []), got["aspects"]):
            if abs(a - b) > 0.5:
                out.append(f"facet aspect {a} -> {b} deg")
        if want.get("facet_panels") != got["facet_panels"]:
            out.append(f"panels per facet {want.get('facet_panels')} -> "
                       f"{got['facet_panels']}")
        for a, b in zip(want.get("poa", []), got["poa"]):
            if abs(a - b) > AREA_REL_TOL * max(a, 1):
                out.append(f"facet irradiance {a} -> {b} kWh/m2/yr")
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
                  f"{m['panels']} panels, {m['ac_kwh_year']:.0f} kWh/yr")
    spec["_note"] = ("Recorded output of build_layout_geojson._build_one -- the "
                     "real per-building pipeline path -- NOT ground truth. "
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
