"""The heat map as raster tiles instead of 24 positioned images.

WHY. After buildings became vector tiles, the heat map is the largest thing the
map downloads: measured on the live site, 9.6 MB for one street view. It is
served as one PNG per region -- 24 of them, 135 MB on disk -- attached to the
map as image sources, so looking at one street pulls the whole region's image.
The LOD machinery in preview.html (three sizes per region, attach/detach by
viewport, a megapixel budget) exists entirely to make that bearable.

Tiles make it a non-problem: a view fetches the dozen 256x256 tiles under it
and nothing else, the browser caches them, and the cost stops depending on how
big the region is. That is the same reason panel layouts and now buildings are
tiles, and it is the last layer that was not.

SOURCE IS data/heatmaps/, NOT data/regions/. The committed, deployed PNGs are
what the live map shows today, and they are not byte-identical to the region
rasters they were exported from -- so tiling those instead would silently
change the picture while claiming only to change how it is delivered. It also
means this runs correctly on a laptop whose region data is behind the VM's.

HOW. Each heat map is a north-up EPSG:2193 grid -- the four
corners in its sidecar JSON are an exact NZTM rectangle, checked, not assumed
-- so it can be given a real affine transform and reprojected into web mercator
with rasterio. Regions overlap at their edges, so a tile touched by more than
one is alpha-composited rather than overwritten, which is what the image-source
version did implicitly by drawing them on top of each other.

Fully transparent tiles are not written. Most of a bounding box over Queenstown
is lake and mountain, and an empty tile is 100 bytes of nothing that still costs
a request.

Usage: python tools/build_heatmap_tiles.py [--zmin 13] [--zmax 17] [--regions a b]
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "heatmap_tiles"
TILE = 256


def tile_xy(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    r = math.radians(max(-85.05, min(85.05, lat)))
    y = int((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds_3857(x, y, z):
    """Tile bounds in EPSG:3857 metres."""
    c = 20037508.342789244
    s = 2 * c / (2 ** z)
    return (-c + x * s, c - (y + 1) * s, -c + (x + 1) * s, c - y * s)


def tiles_for_region(png, corners, out_dir, zmin=13, zmax=17, label=""):
    """Reproject one region's heat-map PNG into web-mercator tiles under
    out_dir/z/x/y.png. Returns the number written, or -1 if the sidecar's
    corners are not an NZTM rectangle.

    A FUNCTION OF ONE REGION, so emit_region can write a region's tiles into
    that region's own output folder and combine_regions can composite the seam
    tiles later. main() below is the whole-district form and calls this per
    region into one folder, which is what it always did.
    """
    import numpy as np
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import pyproj
    to_nztm = pyproj.Transformer.from_crs(4326, 2193, always_xy=True).transform
    out_dir = Path(out_dir)
    nz = [to_nztm(x, y) for x, y in corners]
    xs = [p[0] for p in nz]
    ys = [p[1] for p in nz]
    # The sidecar promises a rectangle. Check it, because everything below
    # assumes a north-up grid and a silently skewed one would put the heat
    # map metres away from the roofs it describes.
    if (abs(nz[0][0] - nz[3][0]) > 0.5 or abs(nz[1][0] - nz[2][0]) > 0.5
            or abs(nz[0][1] - nz[1][1]) > 0.5
            or abs(nz[3][1] - nz[2][1]) > 0.5):
        print(f"  {label}: corners are not an NZTM rectangle -- skipped")
        return -1
    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)

    img = np.array(Image.open(png).convert("RGBA"))
    h, w = img.shape[:2]
    src_t = from_bounds(west, south, east, north, w, h)
    src = np.moveaxis(img, 2, 0)

    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    n_written = 0
    for z in range(zmin, zmax + 1):
        x0, y0 = tile_xy(min(lons), max(lats), z)
        x1, y1 = tile_xy(max(lons), min(lats), z)
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                b = tile_bounds_3857(tx, ty, z)
                dst = np.zeros((4, TILE, TILE), dtype=np.uint8)
                reproject(
                    source=src, destination=dst,
                    src_transform=src_t, src_crs="EPSG:2193",
                    dst_transform=from_bounds(*b, TILE, TILE),
                    dst_crs="EPSG:3857",
                    resampling=Resampling.bilinear,
                    src_nodata=None, dst_nodata=None)
                if not dst[3].any():
                    continue          # nothing of this region lands here
                p = out_dir / str(z) / str(tx) / f"{ty}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                new = Image.fromarray(np.moveaxis(dst, 0, 2), "RGBA")
                if p.exists():
                    # a tile on a region seam: keep both, do not overwrite
                    new = Image.alpha_composite(
                        Image.open(p).convert("RGBA"), new)
                new.save(p, optimize=True)
                n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmin", type=int, default=13)
    ap.add_argument("--zmax", type=int, default=17)
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    from PIL import Image
    # These rasters are legitimately enormous -- speargrass_hayes is 208
    # megapixels -- and PIL refuses anything over ~179 Mpx as a possible
    # decompression bomb. They are our own build output, so the guard is
    # answering a question nobody asked and it killed the run two regions
    # from the end.
    Image.MAX_IMAGE_PIXELS = None
    import pyproj
    from src.region_build import all_areas

    to_nztm = pyproj.Transformer.from_crs(4326, 2193, always_xy=True).transform
    if a.clean and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((DATA / "heatmaps" / "manifest.json").read_text())
    by_name = {m["name"]: m for m in manifest}
    regions = a.regions or [m["name"] for m in manifest]
    written = skipped = 0
    for region in regions:
        entry = by_name.get(region)
        if not entry:
            continue
        png = ROOT / entry["png"]
        if not png.exists():
            continue
        n_written = tiles_for_region(png, entry["coordinates"], OUT,
                                     a.zmin, a.zmax, label=region)
        if n_written < 0:
            continue
        written += n_written
        print(f"  {region}: {n_written} tiles", flush=True)

    total = sum(p.stat().st_size for p in OUT.rglob("*.png"))
    n = len(list(OUT.rglob("*.png")))
    print(f"{n} tiles, {total/1e6:.1f} MB "
          f"({total/max(n,1)/1e3:.1f} kB each), z{a.zmin}-{a.zmax}")
    (OUT / "meta.json").write_text(json.dumps(
        {"minzoom": a.zmin, "maxzoom": a.zmax, "tileSize": TILE}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
