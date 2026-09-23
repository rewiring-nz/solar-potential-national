"""Where each roof actually is in the picture, relative to where the LiDAR
puts it.

On 42 Suburb Street and 13 Douglas Avenue the outline does not align well
with the actual roof (a common problem): the
panels seem to follow the outline not the image, it should be the reverse,
matched to the actual image of the roof because that's what people are
actually seeing when they look."

WHAT IS ACTUALLY MISALIGNED, measured before anything was changed. Against the
LiDAR's own building-classified returns, those outlines are NOT misplaced --
the best shift for each is 0.5-1 m and moves single-digit percent of returns.
The outline and the LiDAR agree. It is the ORTHOPHOTO that sits off both:
42 Suburb Street's image is 5 m from its LiDAR, and across 61 pilot buildings
28% are 2 m or more off and 16% are 3 m or more, with a per-region median of
zero. That is relief displacement -- a roof stands above the ground the photo
was rectified to, so it leans away from the camera by an amount that depends
on its height and where it sat in the frame. Local, per building, and not a
constant anyone could subtract.

So the panels are physically in the right place and the picture is not. The
right response is still to follow the picture: people judge the map against it.
This stage measures each building's image offset -- the XY shift that best
aligns the photo's edges with the LiDAR's height edges, within a few metres --
and emit_region moves the DRAWN geometry (facets, panels, obstructions, the
outline) by it. Every number stays where the LiDAR computed it; only the
drawing follows the photo.

Gated, because a wrong shift is worse than none: applied only when the
aligned edges agree materially better than the unaligned ones, and never
beyond MAX_SHIFT_M. Everything else records zero.

Writes data/regions/<r>/image_shift.json: {building_id: [dx_east_m, dy_north_m, agreement]}

Usage: python src/register_imagery.py <region>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_build import area_paths, write_json_atomic

SEARCH_M = 4.0        # the search window each way
MAX_SHIFT_M = 6.0     # never move a drawing further than this (the search reaches 5.7)
MIN_GAIN = 1.04       # aligned edge agreement must beat unaligned by this factor
MIN_AREA_M2 = 40.0    # smaller than this and the edges are the noise


def _edges(a):
    from scipy import ndimage
    a = a.astype(float)
    a = (a - a.mean()) / (a.std() + 1e-9)
    e = np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))
    return e / (e.max() + 1e-9)


def shift_for(geom_m, dsm_ds, img_ds):
    """(dx_east, dy_north, gain) for one building, or None."""
    import rasterio
    from rasterio.windows import from_bounds, transform as wtransform
    from rasterio.warp import reproject, Resampling
    pad = SEARCH_M + 2
    try:
        w = from_bounds(*geom_m.buffer(pad).bounds, dsm_ds.transform).round_offsets().round_lengths()
        w = w.intersection(rasterio.windows.Window(0, 0, dsm_ds.width, dsm_ds.height))
        dsm = dsm_ds.read(1, window=w).astype(float)
        tr = wtransform(w, dsm_ds.transform)
    except Exception:
        return None
    if dsm.size < 100 or not np.isfinite(dsm).all():
        return None
    img = np.zeros(dsm.shape, dtype=np.float32)
    try:
        src_w = from_bounds(*geom_m.buffer(pad + 6).bounds, img_ds.transform)
        src = img_ds.read(1, window=src_w).astype(np.float32)
        reproject(src, img, src_transform=wtransform(src_w, img_ds.transform),
                  src_crs=img_ds.crs, dst_transform=tr, dst_crs=dsm_ds.crs,
                  resampling=Resampling.average)
    except Exception:
        return None
    if img.std() < 1:
        return None
    ed, ei = _edges(dsm), _edges(img)
    res = float(abs(tr.a))
    s = int(round(SEARCH_M / res))
    base = float((ed * ei).sum())
    best = (base, 0, 0)
    for dr in range(-s, s + 1):
        for dc in range(-s, s + 1):
            score = float((ed * np.roll(np.roll(ei, dr, 0), dc, 1)).sum())
            if score > best[0]:
                best = (score, dr, dc)
    # SIGN SETTLED IN THE BROWSER, not on paper. np.roll(ei, dc, 1) with
    # dc > 0 carries image content EAST; if that is what aligns the photo's
    # edges with the LiDAR's, the photo's roof was WEST of the LiDAR's, so the
    # drawing moves west: dx = -dc. Written the other way round first, it
    # pushed 42 Suburb Street's outline a further 5 m off the roof; the
    # correct sense was confirmed by looking, then written down here.
    dx = best[2] * res             # +east  (drawing moves with the photo)
    dy = -best[1] * res            # +north (rows grow southward)
    gain = best[0] / max(base, 1e-9)
    return dx, dy, gain


def register(region):
    import rasterio
    import geopandas as gpd
    paths = area_paths(region)
    if not paths["imagery"].exists():
        print(f"[{region}] no imagery -- no image shifts (all zero)")
        write_json_atomic(paths["dir"] / "image_shift.json", {})
        return {}
    t0 = time.time()
    gdf = gpd.read_file(paths["outlines"]).to_crs("EPSG:2193")
    dsm_ds, img_ds = rasterio.open(paths["dsm"]), rasterio.open(paths["imagery"])
    out, applied, mags = {}, 0, []
    for row in gdf.itertuples():
        g = row.geometry
        if g is None or g.is_empty or g.area < MIN_AREA_M2:
            continue
        r = shift_for(g, dsm_ds, img_ds)
        if not r:
            continue
        dx, dy, gain = r
        if gain >= MIN_GAIN and np.hypot(dx, dy) <= MAX_SHIFT_M and (dx or dy):
            out[str(row.building_id)] = [round(dx, 1), round(dy, 1), round(gain, 3)]
            applied += 1
            mags.append(float(np.hypot(dx, dy)))
    write_json_atomic(paths["dir"] / "image_shift.json", out)
    print(f"[{region}] image registration: {applied}/{len(gdf)} buildings shifted "
          f"(median {np.median(mags) if mags else 0:.1f} m, max {max(mags) if mags else 0:.1f} m) "
          f"in {time.time() - t0:.0f}s")
    return out


def main():
    region = sys.argv[1]
    from src.preflight import preflight
    preflight("register_imagery", region)
    register(region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
