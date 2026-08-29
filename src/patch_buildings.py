"""Rebuild NAMED buildings with the current code and push just them live.

The iterate-by-full-rebuild loop takes hours; this takes minutes. It runs the
same per-building path as build_layout_geojson (so what you see is what a full
rebuild would produce for those buildings), swaps their features into the
region file, the merged district file and solar_potential, re-runs the panel
shrink + tippecanoe over the district, and optionally commits.

Usage:
  python src/patch_buildings.py 5371108 4734850 ... [--area pilot] [--push]

The area flag is only needed for buildings outside the pilot region; ids from
several areas need one invocation per area. Never run while a district rebuild
is writing the same files.
"""
import argparse, json, subprocess, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", type=int)
    ap.add_argument("--area", default="pilot")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--skip-tiles", action="store_true",
                    help="patch the geojson only (for chained invocations; run tiles once at the end)")
    a = ap.parse_args()

    from src.region_build import area_paths, area_centroid_wgs84
    from src.solar_model import SolarModel
    import src.build_layout_geojson as blg

    t0 = time.time()
    c = area_centroid_wgs84(a.area)
    blg._init_worker(a.area, SolarModel(*c) if c else SolarModel())
    new_feats = {}
    for bid in a.ids:
        feats = blg._build_one(bid)
        new_feats[bid] = feats
        n = sum(1 for f in feats if f["properties"]["kind"] == "panel")
        print(f"  #{bid}: {len(feats)} features, {n} panels  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # run the same post-stages those buildings would get in a full build
    ids = set(a.ids)

    def patch(path):
        d = json.load(open(path))
        before = len(d["features"])
        d["features"] = [f for f in d["features"]
                         if f["properties"].get("building_id") not in ids]
        for bid in a.ids:
            d["features"].extend(new_feats[bid])
        json.dump(d, open(path, "w"))
        print(f"  patched {path.name}: {before} -> {len(d['features'])} features", flush=True)

    region = area_paths(a.area)["panel_layouts"]
    patch(region)
    # gate just this area's new panels (in place, cheap for a handful of ids)
    import rasterio
    from src.gate_panels import gate_area
    from src.pointcloud_source import PointCloudSource
    with rasterio.open(DATA / "dem_wide_mosaic.tif") as ds:
        dem = ds.read(1)
        dem_inv = ~ds.transform
    gate_area(a.area, PointCloudSource(), dem, dem_inv, only_ids=ids)
    # re-copy region layouts into the merged district file
    patch(DATA / "panel_layouts.geojson")

    if not a.skip_tiles:
        subprocess.run([sys.executable, "src/shrink_panels_for_tiles.py"], check=True, cwd=ROOT)
        subprocess.run(
            ["tippecanoe", "-o", "data/panel_layouts.pmtiles", "--force", "-l", "layout",
             "-Z13", "-z16", "--drop-densest-as-needed", "--detect-shared-borders",
             "-y", "kind", "-y", "building_id", "-y", "fill_rank", "-y", "fill_order",
             "-y", "array_id", "-y", "array_size", "-y", "ac_kwh_year", "-y", "slope_deg",
             "-y", "aspect_deg", "-y", "roof_confidence", "-y", "poa_kwh_m2_yr",
             "-y", "panel_count", "data/panel_layouts.geojson"],
            check=True, cwd=ROOT)
        print(f"  tiles rebuilt ({time.time()-t0:.0f}s total)", flush=True)

    if a.push:
        subprocess.run(["git", "add", "data/panel_layouts.pmtiles"], cwd=ROOT, check=True)
        subprocess.run(["git", "-c", "user.name=Josh", "-c", "user.email=josh@ideatious.com",
                        "commit", "-q", "-m",
                        f"Patch buildings {' '.join(map(str, a.ids))} with current code"],
                       cwd=ROOT, check=True)
        subprocess.run(["git", "push", "-q", "origin", "main"], cwd=ROOT, check=True)
        print("  pushed live", flush=True)

if __name__ == "__main__":
    main()
