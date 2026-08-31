"""Shrink panel polygons by GAP_M before tiling so adjacent panels show a
thin sliver of real roof between them (user-requested look; an outline
can't fake a gap). Run on the merged panel_layouts.geojson AFTER merge,
BEFORE tippecanoe. Facets/obstructions untouched."""
import json, sys, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preflight import preflight
from src.region_build import write_json_atomic
GAP_M = 0.04  # was 0.07 -- Josh: gaps a touch smaller
# The buffer is applied in DEGREES using the longitude scale at the panel's
# latitude, so the north-south shrink is ~1/cos(lat) larger than east-west
# (~5.7cm vs 4cm at -45). Cosmetically invisible at panel size and this is the
# look that was signed off; noted so it isn't rediscovered as a mystery.
DATA = Path(__file__).resolve().parent.parent / "data"
def main():
    preflight("shrink_panels_for_tiles")
    path = DATA / "panel_layouts.geojson"
    d = json.loads(path.read_text())
    # NOT idempotent: a second pass shrinks the already-shrunk polygons again
    # (0.04m -> 0.08m gaps, and small panels vanish entirely). This runs inside
    # a chain that is routinely re-run from a middle step, so mark the file.
    if d.get("panels_shrunk_m"):
        print(f"already shrunk by {d['panels_shrunk_m']}m -- skipping "
              f"(delete the key to force a re-shrink)")
        return
    from shapely.geometry import shape, mapping
    n = 0
    for f in d["features"]:
        if f["properties"].get("kind") != "panel" or f["geometry"]["type"] != "Polygon":
            continue
        lat = f["geometry"]["coordinates"][0][0][1]
        deg = GAP_M / (111320.0 * math.cos(math.radians(lat)))
        g = shape(f["geometry"]).buffer(-deg, join_style=2)
        if not g.is_empty and g.geom_type == "Polygon":
            f["geometry"] = mapping(g)
            n += 1
    d["panels_shrunk_m"] = GAP_M
    write_json_atomic(path, d)
    print(f"shrunk {n} panels by {GAP_M}m for tile gaps")
if __name__ == "__main__":
    main()
