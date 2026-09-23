"""Rebuild every roof whose facets do not cover it.

Residual fill was missing from the vision path until 18 Sep, so buildings
on that chain shipped with bare uncovered roof -- no facet, therefore no
panels. This finds them by GEOMETRY alone (no re-prediction needed): a
selected reading whose faces leave more than the fill threshold uncovered
is exactly a building the fix will change.

    python tools/refill_partial_roofs.py            # count them
    python tools/refill_partial_roofs.py --patch
"""

import argparse, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SEL = Path("data/selected_faces")
PY_ = sys.executable


from src.region_build import all_areas as _all_regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--regions", nargs="*", default=None)
    a = ap.parse_args()
    import geopandas as gpd
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from src.region_build import area_paths

    grand = 0
    for region in (a.regions or _all_regions()):
        op = area_paths(region)["outlines"]
        if not Path(op).exists():
            continue
        gdf = gpd.read_file(op)
        todo = []
        for _, row in gdf.iterrows():
            bid = int(row["building_id"])
            fp = SEL / f"{bid}.json"
            if not fp.exists():
                continue
            try:
                faces = json.loads(fp.read_text())["faces"]
                polys = [Polygon(r) for r in faces]
                polys = [p for p in polys if p.is_valid and p.area > 0.5]
                if not polys:
                    continue
                cov = unary_union(polys).intersection(row.geometry).area
            except Exception:
                continue
            fpa = row.geometry.area
            residual = fpa - cov
            if residual > max(12.0, 0.12 * fpa):
                todo.append(bid)
        print(f"{region}: {len(todo)} part-covered roofs", flush=True)
        grand += len(todo)
        if a.patch and todo:
            for i in range(0, len(todo), 60):
                chunk = todo[i:i + 60]
                rc = subprocess.run(
                    [PY_, "src/patch_buildings.py", *map(str, chunk),
                     "--area", region, "--skip-tiles"],
                    env={**os.environ, "SOLAR_SELECTED_FACES": "1"}).returncode
                print(f"{region}: chunk {i//60} rc={rc}", flush=True)
    print(f"TOTAL {grand} part-covered roofs", flush=True)


if __name__ == "__main__":
    main()
