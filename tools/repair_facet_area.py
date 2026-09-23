"""Recompute facet_area_m2 / avg_poa_kwh_m2 from the merged layouts.

Repairs the damage from patch_buildings reading p["area_m2"] off a facet
feature, which the layout emitter does not write. Every building that driver
touched came out with facet_area_m2 = 0, and Heat Map mode -- whose whole
estimate is kWp = area x coverage x density -- showed it as 0.0 kW, while
Panel Layout mode two clicks away showed the same roof's real output.
Measured before the fix: 2,496 of the district's 14,507 roofs with panels.

The driver is fixed; this repairs the standing file so the district does not
need a four-hour rebuild to stop lying about a sixth of its roofs.

Computed the same way src/derive_solar_potential.py does it, through the same
helper, so this cannot drift from the stage whose output it is patching.

ONLY THE ZEROS, AND FROM THE REGION LAYOUTS.

The first version rewrote every building from data/panel_layouts.geojson, on
the reasoning that the merged file is what tippecanoe tiles and therefore what
the map draws. That was wrong twice over. Rendered against imagery, the merged
file's reading for #5373416 is 9 facets cutting across the roof while the
region file's 5 match both the imagery and what the current code produces --
the merged copy is the OLDER one, not the newer. And the divergence is a
laptop artefact in the first place: this machine's region layouts are from
16 September and the build VM's are from the 19th, so a local merge ships
three-day-old geometry.

So this repairs only what is provably broken -- a building with facets whose
area is exactly zero, which no roof has -- and takes the value from the region
file, which is derive_solar_potential's own input. Everything else is left
alone for the district build to regenerate properly.

Usage: python tools/repair_facet_area.py [--write]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.derive_solar_potential import _facet_area_m2
from src.region_build import write_json_atomic

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    from src.region_build import all_areas, area_paths
    sp_path = DATA / "solar_potential.geojson"
    if not sp_path.exists():
        print("need data/solar_potential.geojson")
        return 1

    agg = defaultdict(lambda: {"area": 0.0, "poa_w": 0.0, "n": 0})
    for region in all_areas():
        lp = area_paths(region)["panel_layouts"]
        if not lp.exists():
            continue
        for f in json.loads(lp.read_text())["features"]:
            p = f["properties"]
            if p.get("kind") != "facet":
                continue
            b = agg[p["building_id"]]
            area = _facet_area_m2(f)
            b["area"] += area
            b["poa_w"] += area * (p.get("poa_kwh_m2_yr") or 0.0)
            b["n"] += 1

    sp = json.loads(sp_path.read_text())
    fixed = moved = 0
    for f in sp["features"]:
        p = f["properties"]
        b = agg.get(p.get("building_id"))
        if not b or b["n"] == 0:
            continue
        was = p.get("facet_area_m2") or 0
        if was:
            continue                 # only the provable zeros
        area = round(b["area"], 1)
        if area <= 0:
            continue
        p["facet_area_m2"] = area
        p["avg_poa_kwh_m2"] = round(b["poa_w"] / b["area"], 0)
        moved += 1
        fixed += 1

    print(f"{fixed} buildings repaired (roof area was zero, now from the "
          f"region layouts)")
    if not a.write:
        print("dry run -- pass --write to apply")
        return 0
    write_json_atomic(sp_path, sp)
    print("written. Now re-run:")
    print("  python src/bake_density_deciles.py")
    print("  python src/split_building_detail.py")
    print("  python src/build_building_tiles.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
