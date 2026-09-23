"""
Download the raw LiDAR point-cloud tiles covering every config.REGIONS bbox
into data/pointcloud/, so the segmentation pipeline has the same >8 pts/m2
source for the new regions that the pilot area already has.

Two-step per region: the LINZ tile index (layer 105905, "Otago - Queenstown
LiDAR Tile Index 2021") maps an NZTM bbox to tilenames like "CC11_1000_0712";
each tilename becomes CL2_<sheet>_2021_<tile>.laz in OpenTopography's public
bulk store for the same survey (dataset NZ21_Otago -- LINZ hosts the derived
DSM/DEM rasters but not the raw point cloud). laspy reads these plain .laz
directly; the pilot's original tiles are the same data saved as .copc.laz,
so a tile is skipped if either variant is already on disk.

Resumable by design: existing tiles are skipped, downloads go to a .part
file first and rename on completion, so a killed run never leaves a
truncated .laz behind for laspy to choke on later.

Usage: python src/fetch_pointcloud_regions.py [region ...]  (default: all)
"""

import os
import sys
from pathlib import Path

import pyproj
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.surveys import survey_for
from src.fetch_data import fetch_building_outlines

POINTCLOUD_DIR = Path(__file__).resolve().parent.parent / "data" / "pointcloud"
# Survey-specific values live in config, never here: hard-coding them meant the
# Wellington repo asked the OTAGO bulk store for 2021-named tiles and every
# download 404'd, leaving regions to fall back silently to the 1 m DSM.
# Per SURVEY, not per repo. These remain as the defaults for a deployment with
# no registry; tilename_to_filename and the fetch take the survey's own values
# where one covers the region. See src/surveys.py.
BULK_URL = config.POINTCLOUD_BULK_URL
TILE_YEAR = config.POINTCLOUD_TILE_YEAR
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)


def tilename_to_filename(tilename, year=None):
    sheet, tile = tilename.split("_", 1)  # "CC11_1000_0712" -> ("CC11", "1000_0712")
    return f"CL2_{sheet}_{year or TILE_YEAR}_{tile}.laz"


def area_bbox_wgs84(name):
    # Through region_build, which derives a bbox from the region's own
    # outlines when config does not list it -- a region that has data is
    # buildable, full stop.
    from src.region_build import area_bbox_wgs84 as _bbox
    return _bbox(name)


def tiles_for_bbox_wgs84(bbox, api_key):
    minx, miny = TO_NZTM.transform(bbox[0], bbox[1])
    maxx, maxy = TO_NZTM.transform(bbox[2], bbox[3])
    data = fetch_building_outlines(
        [minx, miny, maxx, maxy], api_key,
        layer_id=survey_for(bbox)["lidar_tile_index_layer"])
    return sorted({f["properties"]["tilename"] for f in data["features"]})


def download_tile(filename, store=None, retries=4):
    dest = POINTCLOUD_DIR / filename
    copc_variant = POINTCLOUD_DIR / filename.replace(".laz", ".copc.laz")
    if dest.exists() or copc_variant.exists():
        return "exists"
    part = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            resp = requests.get(f"{store or BULK_URL}/{filename}", stream=True, timeout=120)
            if resp.status_code == 404:
                return "missing-upstream"
            resp.raise_for_status()
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
            part.rename(dest)
            return f"{dest.stat().st_size / 1e6:.0f}MB"
        except (requests.exceptions.RequestException, OSError) as e:
            # transient S3 timeouts/read errors are routine over a multi-GB run;
            # back off and retry rather than killing the whole batch
            if attempt == retries - 1:
                raise
            import time
            print(f"    retry {attempt + 1} after {type(e).__name__}")
            time.sleep(10 * (attempt + 1))


def main(region_names=None):
    """region_names lets fetch_regions call this directly, so setting up a new
    region pulls its point cloud automatically instead of relying on someone
    remembering a second script."""
    load_dotenv()
    api_key = os.environ["LINZ_API_KEY"]
    POINTCLOUD_DIR.mkdir(parents=True, exist_ok=True)
    # "pilot" is a first-class area with its own bbox -- its exclusive CBD
    # tiles were silently never fetched by the regions-only default, which
    # left holes over the town centre (Turner St, 23 Aug).
    region_names = region_names or sys.argv[1:] or (["pilot"] + list(config.REGIONS))

    # filename -> (region, the bulk store that region's survey uses). BOTH the
    # store and the YEAR are per survey, and until 22 Sep neither reached
    # here: the year came from the module-level TILE_YEAR, so asking for
    # Wanaka's 2022 tiles produced CL2_CA12_2021_*.laz and every one of 165
    # tiles 404'd. The run reported a "coverage gap to investigate" and
    # carried on to build Wanaka off the 1 m DSM.
    all_tiles = {}
    for name in region_names:
        bbox = area_bbox_wgs84(name)
        sv = survey_for(bbox, name)
        store = sv.get("pointcloud_bulk_url")
        if not store:
            print(f"{name}: survey {sv.get('name')} publishes no point cloud "
                  f"-- this region will build from its 1 m DSM")
            continue
        year = sv.get("pointcloud_tile_year") or TILE_YEAR
        tiles = tiles_for_bbox_wgs84(bbox, api_key)
        print(f"{name}: {len(tiles)} tiles from {store.rsplit('/', 1)[-1]} ({year})")
        names = [tilename_to_filename(t, year) for t in tiles]
        for fn in names:
            all_tiles.setdefault(fn, (name, store))
        # RECORD WHICH TILES THIS REGION USES. Tiles are shared at region
        # borders, and publish_region deletes a tile only when no other region
        # still on the disk lists it -- which it can only know from this file.
        try:
            from src.region_build import area_paths
            d = area_paths(name)["dir"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "pointcloud_tiles.txt").write_text("\n".join(sorted(names)) + "\n")
        except Exception as exc:
            print(f"  (could not record {name}'s tile list: {exc})")

    print(f"\n{len(all_tiles)} unique tiles across {len(region_names)} regions")
    missing_upstream = []
    for i, filename in enumerate(sorted(all_tiles)):
        _region, store = all_tiles[filename]
        result = download_tile(filename, store)
        if result == "missing-upstream":
            missing_upstream.append(filename)
        print(f"  [{i + 1}/{len(all_tiles)}] {filename}: {result}")

    if missing_upstream:
        print(f"\nWARNING: {len(missing_upstream)} tiles not in the bulk store "
              f"(coverage gap to investigate): {missing_upstream[:10]}")
    return missing_upstream


if __name__ == "__main__":
    main()
