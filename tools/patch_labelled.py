"""Rebuild every marked roof, region by region.

Run after a change that alters how markup is consumed (e.g. the drawn-line
keepouts, 17 Sep), so the marked roofs pick it up without a district rebuild.
Patches with --skip-tiles per region; rebuild tiles once afterwards.

    .venv/bin/python tools/patch_labelled.py            # dry: list counts
    .venv/bin/python tools/patch_labelled.py --patch
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PY = sys.executable



from src.region_build import all_areas as _all_regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--regions", nargs="*", default=None)
    a = ap.parse_args()
    import config
    import geopandas as gpd
    from src.region_build import area_paths
    from src.roof_line_source import _labels, VOID_FLAGS

    labelled = {int(k) for k, v in _labels().items()
                if v.get("problem") not in VOID_FLAGS}
    print(f"{len(labelled)} labelled buildings", flush=True)
    seen = set()
    for region in (a.regions or _all_regions()):
        outlines = area_paths(region)["outlines"]
        if not Path(outlines).exists():
            continue
        gdf = gpd.read_file(outlines)
        ids = sorted(set(int(x) for x in gdf["building_id"]) & labelled - seen)
        if not ids:
            continue
        seen |= set(ids)
        print(f"{region}: {len(ids)} labelled roofs", flush=True)
        if a.patch:
            r = subprocess.run(
                [PY, "src/patch_buildings.py", *map(str, ids),
                 "--area", region, "--skip-tiles"],
                env={**os.environ, "SOLAR_SELECTED_FACES": "1"})
            print(f"{region}: patch rc={r.returncode}", flush=True)
    missing = labelled - seen
    if missing:
        print(f"not in any region outlines: {sorted(missing)}", flush=True)


if __name__ == "__main__":
    main()
