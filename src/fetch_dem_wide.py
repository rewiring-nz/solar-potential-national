"""Fetch the wide, 8m DEM used for distant terrain horizons.

The output is the root-level asset every horizon stage reads:
data/dem_wide_mosaic.tif.

WHY THE EXTENT IS DERIVED AND THE FILE IS CHECKED. Both used to be taken on
trust -- the extent from a constant in config.py, and the file from the fact
that it existed. On 22 September both were wrong at once:

  * config.DEM_WIDE_BBOX was written when the district was Queenstown. Adding
    Kingston and Wanaka to config.REGIONS did not widen it, so the towns being
    built sat outside the extent their own horizons were supposed to come from.
  * The mosaic actually on disk -- on the laptop AND on the build VM -- covered
    168.47 to 168.87 lon, smaller again than even that stale constant, and
    `ensure_dem_wide` skipped the fetch because the file was present.

Neither failure is visible. building_horizon.far_profile marches out to
FAR_MAX_KM and "stops at the DEM edge anyway", so a DEM that ends early does
not error: the horizon simply comes back lower, the roof looks sunnier than it
is, and the number ships. Kingston is 20 km south of that mosaic's bottom edge
and Wanaka is off it entirely, so every building in both towns would have been
modelled with open sky in the directions that are in fact mountains.

So: the extent is computed from the regions that exist right now, and the file
is trusted only if its own bounds contain that extent.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.fetch_data import fetch_raster

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEM_WIDE_LAYER = config.LINZ_WIDE_DEM_LAYER

# The horizon marcher walks 20 km (building_horizon.FAR_MAX_KM). 30 km is that
# plus margin, so a region can grow a little without silently losing terrain.
WIDE_DEM_BUFFER_M = 30_000


def wide_dem_bbox_wgs84():
    """The WGS84 extent the wide DEM must cover: every region, buffered.

    Buffered in METRES, in NZTM, because a degree of longitude is 0.7 km
    narrower at Kingston than at Hawea and a degree-based buffer would be
    short exactly where the district is longest.

    config.DEM_WIDE_BBOX, where a deployment sets one, is unioned in rather
    than replaced: it may cover terrain that matters for reasons this function
    cannot know, and widening is always safe.
    """
    import pyproj

    fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    inv = pyproj.Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    boxes = [config.PILOT_BBOX, *config.REGIONS.values()]
    pts = [fwd.transform(lon, lat)
           for b in boxes
           for lon, lat in ((b[0], b[1]), (b[2], b[3]))]
    b = WIDE_DEM_BUFFER_M
    w, s = inv.transform(min(p[0] for p in pts) - b, min(p[1] for p in pts) - b)
    e, n = inv.transform(max(p[0] for p in pts) + b, max(p[1] for p in pts) + b)

    fixed = getattr(config, "DEM_WIDE_BBOX", None)
    if fixed:
        w, s = min(w, fixed[0]), min(s, fixed[1])
        e, n = max(e, fixed[2]), max(n, fixed[3])
    return [w, s, e, n]


def mosaic_covers(mosaic_path, bbox):
    """Do the mosaic's own bounds contain `bbox`? None if it cannot be read.

    Densified reprojection, not four transformed corners: an NZTM rectangle is
    a curved quadrilateral in WGS84, and comparing corners alone would call a
    mosaic sufficient when its middle edge falls inside the requirement.
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:
        return None
    try:
        with rasterio.open(mosaic_path) as r:
            have = transform_bounds(r.crs, "EPSG:4326", *r.bounds, densify_pts=21)
    except Exception:
        return None
    return (have[0] <= bbox[0] and have[1] <= bbox[1]
            and have[2] >= bbox[2] and have[3] >= bbox[3])


def ensure_dem_wide(api_key, out_dir=DATA_DIR):
    """Fetch the wide DEM unless the one on disk already covers the district."""
    out_dir = Path(out_dir)
    mosaic_path = out_dir / "dem_wide_mosaic.tif"
    bbox = wide_dem_bbox_wgs84()
    if mosaic_path.exists() and mosaic_path.stat().st_size > 0:
        covers = mosaic_covers(mosaic_path, bbox)
        if covers is not False:
            # None means rasterio is not installed here, which is the case in
            # the pure-function tests; presence is the only check available.
            print(f"  {mosaic_path} exists, skipping")
            return mosaic_path
        print(f"  {mosaic_path} does not cover {bbox} -- refetching. Horizons "
              f"baked against the old one are now stale.")

    print(f"Fetching 8m DEM layer {DEM_WIDE_LAYER} for bbox {bbox}...")
    return fetch_raster(bbox, api_key, DEM_WIDE_LAYER, "dem_wide", out_dir=out_dir)


def main():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.environ.get("LINZ_API_KEY")
    if not api_key:
        raise SystemExit("LINZ_API_KEY not set")
    DATA_DIR.mkdir(exist_ok=True)
    ensure_dem_wide(api_key)


if __name__ == "__main__":
    main()
