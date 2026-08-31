"""
Copy each building's roof_confidence from its layout onto its building record.

The frontend reads building properties from solar_potential.geojson, not from
the layout tiles -- the "buildings" map source is that file. roof_confidence is
computed during the layout build and written onto FACET features, so without
this it never reaches the panel, and a roof whose layout was withheld reads
0 kW with nothing explaining why.

It cannot simply be written by build_heatmap.py, which is what produces
solar_potential.geojson, because that stage runs BEFORE layouts and is not run
at all by the layouts-only rebuild (run_layouts_regate_par.sh) -- the loop
actually used for iteration. So this patches in place afterwards, the same
shape as add_addresses.py.

Idempotent: re-running overwrites the same field.

Usage: python src/patch_roof_confidence.py [region ...]   (default: all areas)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preflight import preflight
from src.region_build import all_areas, area_paths, areas_from_argv, write_json_atomic


def patch_area(name):
    paths = area_paths(name)
    layouts, potential = paths["panel_layouts"], paths["solar_potential"]
    if not layouts.exists() or not potential.exists():
        print(f"{name}: missing layouts or solar_potential, skipping")
        return

    # Area-weighted, matching how the value is computed at build time: one
    # small bad facet should not condemn a roof that is otherwise understood.
    num, den = {}, {}
    for f in json.loads(layouts.read_text())["features"]:
        pr = f["properties"]
        if pr.get("kind") != "facet" or pr.get("roof_confidence") is None:
            continue
        bid = pr["building_id"]
        # facet area is not carried in the layout; weight by panel_count + 1 so
        # a facet that took no panels still counts, but a big array counts more
        w = float(pr.get("panel_count") or 0) + 1.0
        num[bid] = num.get(bid, 0.0) + float(pr["roof_confidence"]) * w
        den[bid] = den.get(bid, 0.0) + w

    if not num:
        print(f"{name}: no roof_confidence in layouts (built before the field existed)")
        return

    data = json.loads(potential.read_text())
    hit = 0
    for f in data["features"]:
        bid = f["properties"].get("building_id")
        if bid in num:
            f["properties"]["roof_confidence"] = round(num[bid] / den[bid], 2)
            hit += 1
    write_json_atomic(potential, data)
    print(f"{name}: roof_confidence set on {hit} of {len(data['features'])} buildings")


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def patch_merged():
    """Same patch on the merged pair the site actually serves.

    merge_regions runs BEFORE this can, so patching only the per-region files
    leaves the served solar_potential.geojson without the field -- and that is
    the one the map reads."""
    layouts, potential = DATA_DIR / "panel_layouts.geojson", DATA_DIR / "solar_potential.geojson"
    if not layouts.exists() or not potential.exists():
        print("merged: missing files, skipping")
        return
    num, den = {}, {}
    for f in json.loads(layouts.read_text())["features"]:
        pr = f["properties"]
        if pr.get("kind") != "facet" or pr.get("roof_confidence") is None:
            continue
        bid = pr["building_id"]
        w = float(pr.get("panel_count") or 0) + 1.0
        num[bid] = num.get(bid, 0.0) + float(pr["roof_confidence"]) * w
        den[bid] = den.get(bid, 0.0) + w
    if not num:
        print("merged: no roof_confidence in layouts")
        return
    data = json.loads(potential.read_text())
    hit = 0
    for f in data["features"]:
        bid = f["properties"].get("building_id")
        if bid in num:
            f["properties"]["roof_confidence"] = round(num[bid] / den[bid], 2)
            hit += 1
    write_json_atomic(potential, data)
    print(f"merged: roof_confidence set on {hit} of {len(data['features'])} buildings")


def main():
    argv = [a for a in sys.argv if a != "--merged-only"]
    if "--merged-only" not in sys.argv:
        for area in areas_from_argv(argv):
            preflight("patch_roof_confidence", area)
            patch_area(area)
    patch_merged()


if __name__ == "__main__":
    main()
