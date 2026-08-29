"""
Tile our own 0.1m LINZ imagery (data/imagery_mosaic.tif) into a standard
Web Mercator XYZ pyramid, served locally alongside Esri World Imagery.

Why: Esri's free World Imagery has a hard zoom cutoff that varies by
region -- Queenstown runs out of resolution partway through a normal
"zoom in on a roof" gesture and starts showing "Map data not yet
available". We already own 0.1m imagery for the whole pilot bbox (the
same LINZ layer the building outlines were extracted from) -- finer
than Esri ever offered here -- so tiling it ourselves and layering it
on top removes the cutoff entirely for the pilot area, and Esri still
shows through unchanged everywhere else.

Zoom range z14-19: z19 at Queenstown's latitude is ~0.21m/pixel, already
sharper than most uses need; z20 (~0.1m/px, our native resolution) would
be ~5x the tile count for a marginal further gain and wasn't worth the
extra generation time for a pilot.

Usage: python src/build_tiles.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TILES_DIR = Path(__file__).resolve().parent.parent / "tiles"
MIN_ZOOM, MAX_ZOOM = 14, 19
TILE_SIZE = 256

_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform


def tile_bounds_3857(x, y, z):
    """Returns (left, bottom, right, top) of an XYZ tile in EPSG:3857 metres."""
    n = 2 ** z
    lon_left, lon_right = x / n * 360 - 180, (x + 1) / n * 360 - 180
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    left, bottom = _to_3857(lon_left, lat_bottom)
    right, top = _to_3857(lon_right, lat_top)
    return left, bottom, right, top


def main():
    src_path = DATA_DIR / "imagery_mosaic.tif"
    with rasterio.open(src_path) as src:
        data_bounds_3857 = transform_bounds(src.crs, "EPSG:3857", *src.bounds)

    left3857, bottom3857, right3857, top3857 = data_bounds_3857
    print(f"Source data extent in EPSG:3857: {data_bounds_3857}")

    total_written = total_skipped = 0
    t0 = time.time()

    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        n = 2 ** z
        world_size = 2 * math.pi * 6378137  # circumference of EPSG:3857's sphere, metres
        origin = -world_size / 2

        x_min = max(0, int((left3857 - origin) / world_size * n))
        x_max = min(n - 1, int((right3857 - origin) / world_size * n))
        # Tile y increases southward (y=0 is the northernmost row), opposite to
        # how northing increases -- so the *top* (largest Y) maps to the
        # *smaller* tile y, and vice versa. Got this backwards on the first
        # pass (computed a "y increases northward" index) and it silently
        # produced y_min > y_max, an empty range, for every zoom level.
        y_min = max(0, int((1 - (top3857 - origin) / world_size) * n))
        y_max = min(n - 1, int((1 - (bottom3857 - origin) / world_size) * n))

        z_written = 0
        with rasterio.open(src_path) as src:
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    tb = tile_bounds_3857(x, y, z)
                    res = (tb[2] - tb[0]) / TILE_SIZE
                    dst_transform = Affine(res, 0, tb[0], 0, -res, tb[3])
                    dst = np.zeros((4, TILE_SIZE, TILE_SIZE), dtype=np.uint8)

                    reproject(
                        source=rasterio.band(src, [1, 2, 3, 4]),
                        destination=dst,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=dst_transform, dst_crs="EPSG:3857",
                        resampling=Resampling.bilinear,
                    )

                    if dst[3].max() == 0:  # fully transparent -- outside our data, skip
                        total_skipped += 1
                        continue

                    out_dir = TILES_DIR / str(z) / str(x)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    img = np.moveaxis(dst, 0, -1)
                    Image.fromarray(img, mode="RGBA").save(out_dir / f"{y}.png")
                    z_written += 1
                    total_written += 1

        print(f"z{z}: {z_written} tiles written, elapsed={time.time() - t0:.0f}s")

    print(f"\nDone: {total_written} tiles written, {total_skipped} skipped (outside data extent), "
          f"{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
