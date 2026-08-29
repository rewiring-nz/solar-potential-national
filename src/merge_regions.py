"""
Merge per-region build outputs into the site-level data files.

Because dedupe_outlines() already assigned every building to exactly one
region before the builds ran, this is a plain concatenation -- no overlap
resolution here.

Outputs:
- data/solar_potential.geojson    (merged; assumptions from the first region)
- data/panel_layouts.geojson      (merged -- NOTE: at full Queenstown scale
  this lands in the hundreds of MB, fine as pipeline output on disk but NOT
  servable to browsers as one fetch; the deploy path needs the planned
  PMTiles conversion before this goes live)
- data/heatmaps/<region>.png + .json, plus data/heatmaps/manifest.json
  listing every region raster for the frontend to load as one source each.

Usage: python src/merge_regions.py [region ...]   (default: all regions)
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import DATA_DIR, all_areas, area_paths, write_json_atomic

HEATMAPS_DIR = DATA_DIR / "heatmaps"


def _owned_ids(name):
    """Current ownership per the area's dedup file -- regions rebuilt before
    a later dedupe generation can still CONTAIN buildings that now belong to
    a newer neighbour (frankton_arm, 23 Aug: 1,072 duplicates); the merge is
    where stale extras get dropped, so ownership always wins without forcing
    neighbour rebuilds."""
    import geopandas as gpd
    f = area_paths(name)["dir"] / "building_outlines_dedup.geojson"
    if not f.exists():
        return None
    return set(gpd.read_file(f)["building_id"].astype(int))


# 7 decimal places is ~1.1cm at this latitude -- finer than the LINZ outlines
# themselves and far finer than anything the map draws. Straight off the WFS
# the coordinates carry 17 significant digits, which on solar_potential is
# 6.7MB of the 19.2MB the browser must download and JSON.parse before the map
# can show anything. Rounding is worth roughly half of that.
COORD_PRECISION = 7


def _round_coords(node):
    """Recursively round a GeoJSON coordinates array in place-ish."""
    if isinstance(node, (int, float)):
        return round(node, COORD_PRECISION)
    return [_round_coords(v) for v in node]


def merge_geojson(regions, key, out_path, round_coords=False):
    merged = None
    for name in regions:
        path = area_paths(name)[key]
        if not path.exists():
            print(f"  WARNING: {name} has no {path.name}, skipping")
            continue
        data = json.loads(path.read_text())
        owned = _owned_ids(name)
        if owned is not None:
            data["features"] = [f for f in data["features"]
                                if int(f["properties"].get("building_id", -1)) in owned]
        if merged is None:
            merged = data
        else:
            merged["features"].extend(data["features"])
    if merged is None:
        raise SystemExit(f"no region produced {key}")
    if round_coords:
        for f in merged["features"]:
            f["geometry"]["coordinates"] = _round_coords(f["geometry"]["coordinates"])
    write_json_atomic(out_path, merged)
    print(f"{out_path.name}: {len(merged['features'])} features, "
          f"{out_path.stat().st_size / 1e6:.1f}MB")


def collect_heatmaps(regions):
    HEATMAPS_DIR.mkdir(exist_ok=True)
    manifest = []
    for name in regions:
        paths = area_paths(name)
        if not paths["heatmap_png"].exists():
            print(f"  WARNING: {name} has no heatmap raster, skipping")
            continue
        shutil.copy2(paths["heatmap_png"], HEATMAPS_DIR / f"{name}.png")
        meta = json.loads(paths["heatmap_json"].read_text())
        manifest.append({"name": name, "png": f"data/heatmaps/{name}.png",
                          "coordinates": meta["coordinates"]})
    write_json_atomic(HEATMAPS_DIR / "manifest.json", manifest)
    print(f"heatmaps/manifest.json: {len(manifest)} region rasters")
    # The manifest is rebuilt from scratch above, so the overview-raster
    # entries (png_lod/size/size_lod) have to be regenerated with it -- without
    # this the frontend silently loses its LOD path and goes back to uploading
    # the full-resolution rasters at every zoom. Cheap when nothing changed:
    # the builder skips any LOD newer than its source.
    from src.build_heatmap_lod import main as build_lod
    build_lod()


def main():
    regions = sys.argv[1:] or all_areas()
    # Only the buildings file: panel_layouts goes through tippecanoe, which
    # quantizes to the tile grid anyway, and it is never fetched by a browser.
    merge_geojson(regions, "solar_potential", DATA_DIR / "solar_potential.geojson",
                  round_coords=True)
    merge_geojson(regions, "panel_layouts", DATA_DIR / "panel_layouts.geojson")
    collect_heatmaps(regions)


if __name__ == "__main__":
    main()
