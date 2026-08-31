"""
Fetch building outlines + DSM + imagery for every config.REGIONS entry into
data/regions/<name>/, reusing fetch_data.py's LINZ WFS/Exports machinery.

Resumable by design: each output file is skipped if it already exists, so the
script can be re-run after any network failure or interrupted export and only
does the remaining work. Imagery exports are the slow part (LINZ generates
the cropped export server-side before download -- minutes per region) and by
far the largest (~0.35GB/km2 at 0.1m), so DSMs for every region are fetched
first (small, quick -- the pipeline can start on a region as soon as its
imagery lands later).

Large regions are split into <= MAX_EXPORT_KM2 chunks per export job and
mosaicked back together -- one giant imagery export both risks the Exports
API's job-size ceiling and buffers multi-GB zips in memory.

Usage: python src/fetch_regions.py [region ...]   (default: all regions)
"""

import os
import sys
from pathlib import Path

import pyproj
import rasterio
from dotenv import load_dotenv
from rasterio.merge import merge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.fetch_data import fetch_building_outlines, fetch_raster

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REGIONS_DIR = DATA_DIR / "regions"
MAX_EXPORT_KM2 = 8.0  # imagery chunk ceiling -- ~2.8GB per chunk at 0.1m, comfortably inside
# the Exports API's limits and this machine's memory for the download buffer

TO_NZTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)


def bbox_nztm(bbox_wgs84):
    x0, y0 = TO_NZTM.transform(bbox_wgs84[0], bbox_wgs84[1])
    x1, y1 = TO_NZTM.transform(bbox_wgs84[2], bbox_wgs84[3])
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def bbox_area_km2(bbox_wgs84):
    b = bbox_nztm(bbox_wgs84)
    return abs(b[2] - b[0]) * abs(b[3] - b[1]) / 1e6


def split_bbox(bbox_wgs84, max_km2):
    """Recursively halve along the longer axis until every part <= max_km2."""
    if bbox_area_km2(bbox_wgs84) <= max_km2:
        return [bbox_wgs84]
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    # compare metric extents, not degree extents (degrees of longitude are shorter here)
    b = bbox_nztm(bbox_wgs84)
    if (b[2] - b[0]) >= (b[3] - b[1]):
        mid = (min_lon + max_lon) / 2
        parts = [[min_lon, min_lat, mid, max_lat], [mid, min_lat, max_lon, max_lat]]
    else:
        mid = (min_lat + max_lat) / 2
        parts = [[min_lon, min_lat, max_lon, mid], [min_lon, mid, max_lon, max_lat]]
    out = []
    for p in parts:
        out.extend(split_bbox(p, max_km2))
    return out


def fetch_raster_chunked(bbox_wgs84, api_key, layer_id, name, out_dir, format_key):
    """fetch_raster, but split into MAX_EXPORT_KM2 chunks and re-mosaicked."""
    mosaic_path = out_dir / f"{name}_mosaic.tif"
    if mosaic_path.exists():
        print(f"  {mosaic_path.name} exists, skipping")
        return mosaic_path
    parts = split_bbox(bbox_wgs84, MAX_EXPORT_KM2)
    part_paths = []
    for i, part in enumerate(parts):
        part_name = name if len(parts) == 1 else f"{name}_part{i}"
        part_path = out_dir / f"{part_name}_mosaic.tif"
        if not part_path.exists():
            print(f"  exporting {part_name} ({bbox_area_km2(part):.1f} km2)...")
            fetch_raster(part, api_key, layer_id, part_name, out_dir=out_dir, format_key=format_key)
        part_paths.append(part_path)
    if len(part_paths) == 1:
        if part_paths[0] != mosaic_path:
            part_paths[0].rename(mosaic_path)
        return mosaic_path
    srcs = [rasterio.open(p) for p in part_paths]
    mosaic, transform = merge(srcs)
    profile = srcs[0].profile
    # BIGTIFF: a merged 0.1m imagery mosaic for a large region exceeds classic
    # TIFF's 4GB ceiling (arrowtown_millbrook was the first to hit it).
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform,
                   BIGTIFF="IF_SAFER")
    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    print(f"  merged {len(part_paths)} parts -> {mosaic_path.name}")
    return mosaic_path


def main():
    load_dotenv()
    api_key = os.environ.get("LINZ_API_KEY")
    if not api_key:
        raise SystemExit("LINZ_API_KEY not set")

    wanted = sys.argv[1:] or list(config.REGIONS)
    for name in wanted:
        if name not in config.REGIONS:
            raise SystemExit(f"unknown region {name!r} -- known: {list(config.REGIONS)}")

    # Pass 1: outlines + DSM for every region (small and fast).
    for name in wanted:
        bbox = config.REGIONS[name]
        out_dir = REGIONS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        outlines_path = out_dir / "building_outlines.geojson"
        if not outlines_path.exists():
            print(f"[{name}] fetching building outlines...")
            data = fetch_building_outlines(bbox_nztm(bbox), api_key)
            import json
            outlines_path.write_text(json.dumps(data))
            print(f"  {len(data['features'])} outlines")
        else:
            print(f"[{name}] outlines exist, skipping")

        print(f"[{name}] DSM...")
        fetch_raster_chunked(bbox, api_key, config.LINZ_DSM_LAYER, "dsm", out_dir, "grid")

    # Pass 2: imagery (the long pole), region by region.
    for name in wanted:
        bbox = config.REGIONS[name]
        out_dir = REGIONS_DIR / name
        print(f"[{name}] imagery ({bbox_area_km2(bbox):.1f} km2)...")
        try:
            fetch_raster_chunked(bbox, api_key, config.LINZ_IMAGERY_LAYER, "imagery", out_dir, "raster")
        except Exception as e:
            # LINZ's 0.1m aerial layer is URBAN-only: rural regions 400 here.
            # Never let that abort the run -- DSM+outlines are what the build
            # actually requires, and builds degrade gracefully without imagery.
            print(f"  WARNING: imagery unavailable for {name} ({type(e).__name__}) -- LiDAR-only build")

    # Pass 3: the raw LiDAR point cloud. This is the pipeline's PRIMARY input --
    # segmentation, obstruction height evidence, panel gating and shading all
    # read it, and the Wellington survey carries 16 pts/m2 against the 1 m DSM's
    # single sample. Without it every building falls back to the DSM SILENTLY.
    # It used to be a second script you had to remember to run. Josh, 31 Aug:
    # "that should be part of the automatic process for all future regions."
    print(f"\n[point cloud] fetching LiDAR tiles for {len(wanted)} region(s)...")
    try:
        from src.fetch_pointcloud_regions import main as fetch_pointcloud
        missing = fetch_pointcloud(wanted)
        if missing:
            print(f"  WARNING: {len(missing)} tiles missing from the bulk store -- "
                  f"those parts of the region will build from the 1 m DSM only.")
    except Exception as e:
        print(f"  WARNING: point-cloud fetch FAILED ({type(e).__name__}: {e}).\n"
              f"  These regions would build from the 1 m DSM only, which is far "
              f"coarser. Re-run: python src/fetch_pointcloud_regions.py "
              + " ".join(wanted))

    print("\nAll requested regions fetched.")


if __name__ == "__main__":
    main()
