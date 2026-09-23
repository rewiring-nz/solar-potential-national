"""Terrain-RGB tiles from the DSM, so the map can be viewed in 3D.

A 3D option for the map: terrain, trees and the building.

All three at once, from one surface. The DSM is the top of everything the
LiDAR hit -- ground where there is ground, tree canopy where there are
trees, roof where there is a roof -- so encoding IT rather than the bare
earth gives terrain, trees and buildings in a single layer. It is also
exactly the surface the shading model reads, so the 3D view shows what the
estimates were computed from rather than a decorative extrusion beside it.

Output is the Mapbox terrain-RGB encoding MapLibre expects:
    height = -10000 + (R*65536 + G*256 + B) * 0.1
written as an XYZ pyramid of PNGs under data/terrain/.

No GDAL command line and no rio-rgbify on this machine, so the reprojection
is done through rasterio's WarpedVRT, which reads the NZTM raster directly
into web-mercator tiles.

Usage:
    python tools/build_terrain_tiles.py                 # every region
    python tools/build_terrain_tiles.py pilot --max-zoom 17
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, Window
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "terrain"
TILE = 256
R_EARTH = 6378137.0
ORIGIN = math.pi * R_EARTH          # 20037508.342789244


def lonlat_to_merc(lon, lat):
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def tile_bounds(z, x, y):
    n = 2 ** z
    span = 2 * ORIGIN / n
    return (-ORIGIN + x * span, ORIGIN - (y + 1) * span,
            -ORIGIN + (x + 1) * span, ORIGIN - y * span)


def merc_to_tile(z, mx, my):
    n = 2 ** z
    span = 2 * ORIGIN / n
    return int((mx + ORIGIN) / span), int((ORIGIN - my) / span)


def despike(h, m_per_px, rise_m=8.0, window_m=9.0):
    """Remove thin spikes that are not real structures.

    The DSM carries returns from things that are not surfaces -- a crane,
    a mast, a bird, a stray multipath return -- a town-centre building came
    out way too high and spiky in 3D.
    Measured there: cells sitting 45-47 m above the median of everything
    within 15 m of them.

    A median filter over a WINDOW IN METRES is the discriminator, because a
    real object is wider than its height error is tall. Inside a building or
    a tree crown the local median is the building or the crown, so the cell
    barely differs from it and nothing is touched; a pole or a noise return
    is one or two cells wide, so the median ignores it and the difference is
    large. Only cells exceeding the local median by `rise_m` are pulled back
    to it, which leaves every genuine roof and tree standing.

    Skipped where a pixel is already coarser than the window: at district
    zooms the filter would be smoothing hills, not spikes.
    """
    if m_per_px <= 0 or window_m / m_per_px < 3:
        return h
    from scipy.ndimage import median_filter
    w = int(max(3, min(15, round(window_m / m_per_px))))
    if w % 2 == 0:
        w += 1
    med = median_filter(h, size=w, mode="nearest")
    return np.where(h - med > rise_m, med, h)


def _overlaps(win, vrt):
    return not (win.col_off + win.width <= 0 or win.row_off + win.height <= 0
                or win.col_off >= vrt.width or win.row_off >= vrt.height)


def encode(h):
    """Mapbox terrain-RGB.

    NODATA MUST NOT BE A CLIFF. The first version encoded NaN as the
    encoding's zero, which decodes to -10000 m, and asserted in this
    docstring that MapLibre would never show it. It shows it: every hole and
    every tile edge became a 10 km chasm and the live map rendered two
    thirds of the view as vertical smears. Holes are filled from the nearest
    measured height instead, so a gap is flat ground at about the right
    level rather than a hole through the planet.
    """
    if not np.isfinite(h).all():
        finite = np.isfinite(h)
        if not finite.any():
            h = np.zeros_like(h)
        else:
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(~finite, return_distances=False,
                                         return_indices=True)
            h = h[tuple(idx)]
    v = (h + 10000.0) * 10.0
    v = np.clip(np.rint(v), 0, 256 ** 3 - 1).astype(np.uint32)
    rgb = np.empty(v.shape + (3,), np.uint8)
    rgb[..., 0] = (v >> 16) & 255
    rgb[..., 1] = (v >> 8) & 255
    rgb[..., 2] = v & 255
    return rgb


def build_all(regions, min_z, max_z, area_paths):
    """Tile by TILE, not region by region.

    Writing one region at a time meant a tile covered by two regions got
    whichever wrote last, and a tile whose neighbour came from a different
    region filled its gaps from different data -- so the two disagreed along
    their shared edge and the 3D view showed stepped cliffs at every seam.
    Here each tile gathers from EVERY region that covers it before anything
    is filled, so a gap is only ever filled when no survey has the answer.
    """
    srcs = []
    for r in regions:
        try:
            p = area_paths(r)
        except Exception:
            continue
        if not Path(p["dsm"]).exists():
            continue
        src = rasterio.open(p["dsm"])
        vrt = WarpedVRT(src, crs="EPSG:3857", resampling=Resampling.bilinear,
                        src_nodata=src.nodata, nodata=np.nan, dtype="float32")
        srcs.append((r, vrt))
    if not srcs:
        print("no DSMs found")
        return 0
    # A COARSE FLOOR UNDER THE DETAIL. The 1 m DSM exists only in patches
    # over the built-up areas, so between and around them a tile had holes,
    # and MapLibre draws the boundary between terrain and no-terrain as a
    # vertical curtain -- the walls standing over Lake Wakatipu in the first
    # 3D builds. The district DEM is 8 m bare earth over 29 x 38 km, which
    # is everywhere the DSM is not. It fills what the DSM cannot answer:
    # detail where we surveyed it, honest coarse ground elsewhere, and no
    # cliff where one ends and the other begins.
    dem = None
    dem_path = ROOT / "data" / "dem_wide_mosaic.tif"
    if dem_path.exists():
        dsrc = rasterio.open(dem_path)
        dem = WarpedVRT(dsrc, crs="EPSG:3857", resampling=Resampling.bilinear,
                        src_nodata=dsrc.nodata, nodata=np.nan, dtype="float32")
    print(f"{len(srcs)} region surfaces"
          f"{', plus the wide DEM as a floor' if dem is not None else ''}",
          flush=True)

    written = 0
    for z in range(min_z, max_z + 1):
        want = {}
        for _, vrt in srcs:
            l, b, r2, t = vrt.bounds
            x0, y0 = merc_to_tile(z, l, t)
            x1, y1 = merc_to_tile(z, r2, b)
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    want.setdefault((x, y), []).append(vrt)
        span_m = 2 * ORIGIN / (2 ** z) * math.cos(math.radians(-45.03))
        m_per_px = span_m / TILE
        for (x, y), vrts in want.items():
            b = tile_bounds(z, x, y)
            acc = np.full((TILE, TILE), np.nan, "float32")
            for vrt in vrts:
                win = from_bounds(*b, transform=vrt.transform)
                if not _overlaps(win, vrt):
                    continue
                clipped = win.intersection(Window(0, 0, vrt.width, vrt.height))
                if clipped.width < 1 or clipped.height < 1:
                    continue
                sx, sy = TILE / win.width, TILE / win.height
                ow = max(1, int(round(clipped.width * sx)))
                oh = max(1, int(round(clipped.height * sy)))
                # A ROOF EDGE IS A CLIFF, NOT A RAMP. Bilinear resampling
                # smears the 1 m step at a building's edge across a couple
                # of metres, and anything draped on the terrain -- the
                # panels especially -- then runs down that ramp and appears
                # to hang off the side of the building: unrealistic
                # panels drooped over the sides of buildings on walls
                # rather than rooftops." At detail zooms the tile grid is
                # already near the 1 m source, so nearest keeps the step
                # sharp; at overview zooms bilinear still avoids aliasing.
                rs = (Resampling.nearest if m_per_px <= 1.5
                      else Resampling.bilinear)
                try:
                    part = vrt.read(1, window=clipped, out_shape=(oh, ow),
                                    resampling=rs)
                except Exception:
                    continue
                ox = max(0, min(TILE - 1, int(round((clipped.col_off - win.col_off) * sx))))
                oy = max(0, min(TILE - 1, int(round((clipped.row_off - win.row_off) * sy))))
                ow = min(ow, TILE - ox)
                oh = min(oh, TILE - oy)
                if ow < 1 or oh < 1:
                    continue
                tgt = acc[oy:oy + oh, ox:ox + ow]
                acc[oy:oy + oh, ox:ox + ow] = np.where(
                    np.isfinite(tgt), tgt, part[:oh, :ow])
            if dem is not None and not np.isfinite(acc).all():
                win = from_bounds(*b, transform=dem.transform)
                if _overlaps(win, dem):
                    clipped = win.intersection(Window(0, 0, dem.width, dem.height))
                    if clipped.width >= 1 and clipped.height >= 1:
                        sx, sy = TILE / win.width, TILE / win.height
                        ow = max(1, int(round(clipped.width * sx)))
                        oh = max(1, int(round(clipped.height * sy)))
                        try:
                            part = dem.read(1, window=clipped, out_shape=(oh, ow),
                                            resampling=Resampling.bilinear)
                            ox = max(0, min(TILE - 1, int(round((clipped.col_off - win.col_off) * sx))))
                            oy = max(0, min(TILE - 1, int(round((clipped.row_off - win.row_off) * sy))))
                            ow = min(ow, TILE - ox); oh = min(oh, TILE - oy)
                            if ow >= 1 and oh >= 1:
                                tgt = acc[oy:oy + oh, ox:ox + ow]
                                acc[oy:oy + oh, ox:ox + ow] = np.where(
                                    np.isfinite(tgt), tgt, part[:oh, :ow])
                        except Exception:
                            pass
            if not np.isfinite(acc).any():
                continue
            acc = despike(acc, span_m / TILE)
            d = OUT / str(z) / str(x)
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(encode(acc)).save(d / f"{y}.png", optimize=True)
            written += 1
        print(f"  z{z}: {written} tiles so far", flush=True)
    for _, vrt in srcs:
        vrt.close()
    if dem is not None:
        dem.close()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="*")
    ap.add_argument("--min-zoom", type=int, default=11)
    # 1 m data is fully resolved by about z17 at this latitude (0.84 m/px);
    # asking for more just interpolates and multiplies the file count by four.
    ap.add_argument("--max-zoom", type=int, default=17)
    a = ap.parse_args()
    from src.region_build import area_paths
    d = ROOT / "data/regions"
    regions = a.regions or sorted(p.name for p in d.iterdir() if p.is_dir())
    OUT.mkdir(parents=True, exist_ok=True)
    total = build_all(regions, a.min_zoom, a.max_zoom, area_paths)
    (OUT / "meta.json").write_text(json.dumps(
        {"encoding": "mapbox", "tileSize": TILE,
         "minzoom": a.min_zoom, "maxzoom": a.max_zoom,
         "source": "LINZ 1 m DSM (surface: ground, trees and roofs)"}, indent=1))
    print(f"TOTAL {total} terrain tiles -> {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
