"""
Bake per-building density-decile stats into data/solar_potential.geojson:
for each building, cumulative placed panel count and annual kWh at fill
densities 10..100 in steps of 10, as properties fill_panels_10..100 and
fill_kwh_10..100 (kWh rounded to int).

Why: with panel geometry served as vector tiles, the frontend can no longer
sum every panel in the viewport for the left dashboard's "buildings in map
view" estimate (tiles outside the view / below the zoom simply aren't
loaded). These ten numbers per building make that estimate a cheap sum over
the (small, always fully loaded) buildings source at ANY zoom, and they
also power the per-building panel-placement box without needing the
building's tiles. Runs on the MERGED files after merge_regions.

Usage: python src/bake_density_deciles.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preflight import preflight
from src.region_build import write_json_atomic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DECILES = list(range(10, 101, 10))
MIN_CLEAN_ARRAY = 4       # matches panel_fitting.MINOR_ARRAY_MIN_PANELS
# Panel counts a real quote lands on, at 440W: roughly 3, 4.5, 6, 7.5, 9, 12,
# 15, 20 and 30 kW. Households sit in the first half of that ladder.
SYSTEM_PANEL_STEPS = [7, 10, 14, 17, 20, 27, 34, 45, 68]



COVERAGE_STEPS = [5, 10, 25, 50, 100]


def _ring_area(geom):
    """Plan area of a polygon in raw degrees-squared.

    Only RELATIVE area within one building matters here -- these values are
    used as weights and as a fraction of the building's own total -- and the
    degree-to-metre scale is constant across a single roof, so it cancels.
    Not a metre area, and deliberately not reprojected for ~90k facets.
    """
    rings = geom["coordinates"] if geom["type"] == "Polygon" else \
        [r for poly in geom["coordinates"] for r in poly]
    total = 0.0
    for ring in rings[:1] if geom["type"] == "Polygon" else rings:
        a = 0.0
        for i in range(len(ring) - 1):
            a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
        total += abs(a) / 2
    return total


def _coverage_poa(facets, pct):
    """Area-weighted mean POA over the SUNNIEST `pct` of a building's roof.

    Roof coverage is not linear in output and treating it as such overstates a
    partial system badly. Someone covering 10% of their roof puts the panels on
    the best 10%, which yields well above the roof average; someone covering
    100% is also taking the south face. Josh: "if just 10% is covered, it would
    be the sunniest 10% ... 100% would not be 10 times higher than the 10%".

    kWp stays linear in area -- half the roof is half the panels -- so only the
    POA term changes, which keeps this a drop-in for the existing estimate.
    """
    if not facets:
        return 0.0
    ordered = sorted(facets, key=lambda t: -t[0])
    total_area = sum(a for _, a in ordered)
    if total_area <= 0:
        return 0.0
    target = total_area * pct / 100.0
    acc_area = acc_weighted = 0.0
    for poa, area in ordered:
        take = min(area, target - acc_area)
        if take <= 0:
            break
        acc_area += take
        acc_weighted += take * poa
    return acc_weighted / acc_area if acc_area > 0 else 0.0


def main():
    preflight("bake_density_deciles")
    layouts = json.loads((DATA_DIR / "panel_layouts.geojson").read_text())
    per_building = {}
    # Facet (area, POA) pairs per building, for the coverage curve below.
    facets_by_building = {}
    for f in layouts["features"]:
        p = f["properties"]
        if p["kind"] == "facet":
            poa = p.get("poa_kwh_m2_yr")
            if poa:
                facets_by_building.setdefault(p["building_id"], []).append(
                    (float(poa), _ring_area(f["geometry"])))
            continue
        if p["kind"] != "panel":
            continue
        per_building.setdefault(p["building_id"], []).append(
            (p.get("fill_rank", 100), p.get("ac_kwh_year", 0),
             p.get("fill_order", 0), p.get("array_size", 1)))

    sp_path = DATA_DIR / "solar_potential.geojson"
    sp = json.loads(sp_path.read_text())
    matched = 0
    for feat in sp["features"]:
        b = feat["properties"]["building_id"]
        panels = per_building.get(b, [])
        for d in DECILES:
            kept = [t for t in panels if t[0] <= d]
            feat["properties"][f"fill_panels_{d}"] = len(kept)
            feat["properties"][f"fill_kwh_{d}"] = int(round(sum(t[1] for t in kept)))

        # "Clean arrays only": panels sitting in a contiguous block of at least
        # MIN_CLEAN_ARRAY. Baked per building so the map-view total for that
        # mode is a real sum and not an approximation -- 92% of panels are in
        # blocks of 30+, but the buildings where that is NOT true are exactly
        # the complex roofs this mode is meant to treat differently.
        clean = [t for t in panels if t[3] >= MIN_CLEAN_ARRAY]
        feat["properties"]["fill_panels_arrays"] = len(clean)
        feat["properties"]["fill_kwh_arrays"] = int(round(sum(t[1] for t in clean)))

        # Coverage curve: the mean irradiance of the sunniest X% of THIS roof,
        # so the "what if 10% of roofs were covered" figures reflect that the
        # good roof gets used first. See _coverage_poa.
        fac = facets_by_building.get(b, [])
        for pct in COVERAGE_STEPS:
            feat["properties"][f"cov_poa_{pct}"] = int(round(_coverage_poa(fac, pct)))

        # System-size targeting: cumulative kWh by fill_order, so the frontend
        # can ask "the best N panels" (a 6kW system) and get the right energy
        # without loading the panel tiles. Stored at the sizes a real quote
        # uses; households are 3-12kW (Josh), so the ladder is dense there.
        by_order = sorted(t for t in panels if t[2])
        for n in SYSTEM_PANEL_STEPS:
            sel = by_order[:n]
            feat["properties"][f"sys_kwh_{n}"] = int(round(sum(t[1] for t in sel)))
        if panels:
            matched += 1
    write_json_atomic(sp_path, sp)
    print(f"Baked deciles for {matched}/{len(sp['features'])} buildings "
          f"({sp_path.stat().st_size / 1e6:.1f}MB)")
    # The join is by building_id across two independently written files. If it
    # ever breaks (a type change, a stale merge), every decile bakes as 0 and
    # the whole in-view estimate silently reads zero on a map that still looks
    # correct. Say so here instead of shipping it.
    if matched < 0.5 * len(sp["features"]):
        print(f"  WARNING: only {matched} of {len(sp['features'])} buildings matched a "
              f"layout by building_id -- expected most of them. Check that "
              f"panel_layouts.geojson is the merged file for this build.")


if __name__ == "__main__":
    main()
