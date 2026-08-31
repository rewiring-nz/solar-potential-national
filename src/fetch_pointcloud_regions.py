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
from src.fetch_data import fetch_building_outlines

POINTCLOUD_DIR = Path(__file__).resolve().parent.parent / "data" / "pointcloud"
# Survey-specific values live in config, never here: hard-coding them meant the
# Wellington repo asked the OTAGO bulk store for 2021-named tiles and every
# download 404'd, leaving regions to fall back silently to the 1 m DSM.
BULK_URL = config.POINTCLOUD_BULK_URL
TILE_YEAR = config.POINTCLOUD_TILE_YEAR
TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)


def tilename_to_filename(tilename):
    sheet, tile = tilename.split("_", 1)  # "CC11_1000_0712" -> ("CC11", "1000_0712")
    return f"CL2_{sheet}_{TILE_YEAR}_{tile}.laz"


def area_bbox_wgs84(name):
    if name == "pilot":
        return config.PILOT_BBOX
    return config.REGIONS[name]


def tiles_for_bbox_wgs84(bbox, api_key):
    minx, miny = TO_NZTM.transform(bbox[0], bbox[1])
    maxx, maxy = TO_NZTM.transform(bbox[2], bbox[3])
    data = fetch_building_outlines([minx, miny, maxx, maxy], api_key,
                                    layer_id=config.LINZ_LIDAR_TILE_INDEX_LAYER)
    return sorted({f["properties"]["tilename"] for f in data["features"]})


def download_tile(filename, retries=4):
    dest = POINTCLOUD_DIR / filename
    copc_variant = POINTCLOUD_DIR / filename.replace(".laz", ".copc.laz")
    if dest.exists() or copc_variant.exists():
        return "exists"
    part = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BULK_URL}/{filename}", stream=True, timeout=120)
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
    remembering a second script (Josh, 31 Aug)."""
    load_dotenv()
    api_key = os.environ["LINZ_API_KEY"]
    POINTCLOUD_DIR.mkdir(parents=True, exist_ok=True)
    # "pilot" is a first-class area with its own bbox -- its exclusive CBD
    # tiles were silently never fetched by the regions-only default, which
    # left holes over the town centre (Turner St, 23 Aug).
    region_names = region_names or sys.argv[1:] or (["pilot"] + list(config.REGIONS))

    all_tiles = {}  # filename -> first region needing it (tiles can span regions)
    for name in region_names:
        tiles = tiles_for_bbox_wgs84(area_bbox_wgs84(name), api_key)
        print(f"{name}: {len(tiles)} tiles")
        for t in tiles:
            all_tiles.setdefault(tilename_to_filename(t), name)

    print(f"\n{len(all_tiles)} unique tiles across {len(region_names)} regions")
    missing_upstream = []
    for i, filename in enumerate(sorted(all_tiles)):
        result = download_tile(filename)
        if result == "missing-upstream":
            missing_upstream.append(filename)
        print(f"  [{i + 1}/{len(all_tiles)}] {filename}: {result}")

    if missing_upstream:
        print(f"\nWARNING: {len(missing_upstream)} tiles not in the bulk store "
              f"(coverage gap to investigate): {missing_upstream[:10]}")
    return missing_upstream


if __name__ == "__main__":
    main()
