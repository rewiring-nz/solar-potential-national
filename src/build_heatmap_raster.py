"""
Per-pixel generation-potential heatmap across the whole pilot area, at 0.4m
resolution, computed directly from the LiDAR point cloud.

This replaced a facet-based version (uniform colour per segmented facet at 1m,
plus a heavily-smoothed 1m-DSM gradient fallback for unresolved area) after
direct user comparison against the vector-rendered facets exposed both of that
approach's structural weaknesses at once: wherever segmentation was coarse or
collapsed on a building, the heat map inherited the error verbatim (one real
multi-pitch roof rendering a single uniform colour), and the 1m raster itself
reads blurry at building zoom because the browser bilinear-upscales it ~10x.
The requested reference look -- Google's Solar API flux layer -- is per-pixel
at fine resolution: real gradients within a roof face, soft (not razor) edges,
granularity that shows dormers and curvature.

The point cloud carries ~5-10 pts/m2 here, comfortably supporting a 0.4m
surface: per building, grid the points (inverse-distance-weighted z from the
nearest neighbours), lightly smooth, take per-pixel slope/aspect from the
gradient, look up POA, scale by the building's near-field shading factor, and
composite into one pilot-wide RGBA at 0.4m with a 1px feather. Independent of
roof segmentation entirely -- a building whose facets collapsed still renders
its true per-pixel structure, because the pixels come straight from the
measured surface.

Usage: python src/build_heatmap_raster.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from rasterio.warp import transform as warp_transform
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preflight import preflight
from src.pointcloud_source import PointCloudSource
from src.solar_model import SolarModel
from src.building_shading import building_shading_factor
from src.building_horizon import (far_profile as _hz_far_profile,
                                  far_beam_ratio as _hz_far_ratio,
                                  eave_height as _hz_eave_height)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VMIN, VMAX = 700, 1650  # kWh/m2/yr -- same fixed scale as preview.html's legend and demo_figure.py

# "Iron" palette -- same one Google's own Solar API reference implementation
# uses for its annual flux layer (googlemaps-samples/js-solar-potential,
# colors.ts: ironPalette = ['00000A','91009C','E64616','FEB400','FFFFF6']),
# the recognisable Project Sunroof look. Must stay in sync with preview.html's
# legend gradient.
HEAT_COLORS = ["#00000a", "#91009c", "#e64616", "#feb400", "#fffff6"]
HEAT_CMAP = LinearSegmentedColormap.from_list("solar_heat", HEAT_COLORS, N=256)

HR_RES_M = 0.4  # output resolution -- fine enough for real within-roof structure (dormers, hips,
# curvature) from ~5-10 pts/m2 LiDAR without inventing detail the points can't support
IDW_K = 6  # neighbours per grid-node z estimate
IDW_MAX_DIST_M = 1.2  # a node farther than this from any point has no real surface evidence
Z_SMOOTH_SIGMA_PX = 1.0  # light smoothing before the gradient -- suppresses per-point noise at
# 0.4m without smearing a ridge line more than ~one pixel either side (the "thin gradient
# between changing angles" look explicitly asked for, vs the old 3.5px smear at 1m resolution)
EDGE_FEATHER_SIGMA_PX = 0.8  # soft alpha edge at the footprint boundary -- deliberately not
# razor-sharp, per the same request
SLOPE_BIN_DEG, ASPECT_BIN_DEG, MAX_SLOPE_DEG = 5, 10, 45
ALPHA = 235  # slight base transparency; the frontend's raster-opacity does the rest


def build_lookup_array(model):
    # Sized from the model's own keys, not MAX_SLOPE_DEG: the model may carry
    # bins past the render cap (Wellington's does), and the renderer clamps
    # slope reads to MAX_SLOPE_DEG anyway.
    max_slope_bin = max(sb for sb, _ in model.lookup)
    lookup = np.full((max_slope_bin // SLOPE_BIN_DEG + 1, 360 // ASPECT_BIN_DEG), np.nan)
    for (slope_bin, aspect_bin), poa in model.lookup.items():
        lookup[slope_bin // SLOPE_BIN_DEG, aspect_bin // ASPECT_BIN_DEG] = poa
    return lookup


SHADE_GRID_STEP_M = 1.0  # matches the DSM's own resolution -- finer invents
# detail the 1 m surface cannot support, and costs time for nothing.
SHADE_PAD_M = 1.0        # lattice nodes this far outside the footprint are still
# computed, so bilinear interpolation at the roof edge has real neighbours


def shading_grid(dsm_band, dsm_transform, dsm_nodata, geom, hourly,
                 terrain_horizon_profile=None, step_m=SHADE_GRID_STEP_M):
    """Per-POSITION direct-beam shading over one building, on a `step_m`
    lattice: {fraction of the year's DNI that reaches THIS point}.

    The raster used to scale a whole building by ONE factor taken at its
    footprint centroid, so its within-roof variation came only from
    orientation -- a roof half-buried under a neighbour's macrocarpa rendered
    as brightly as an open one, and Josh (31 Aug) asked why the heat map does
    not show shadows. It now varies per position, which is the whole point of
    a per-pixel layer.

    Returns (grid, xs, ys): grid[j, i] is the factor at (xs[i], ys[j]), ys
    DESCENDING to match raster row order. Nodes with no evidence fall back to
    the building-centroid factor rather than to 1.0, so a failure reads as
    "as shaded as the rest of this roof", never as "fully open".

    Also the exact data the manual-panel-placement idea needs baked into the
    tiles -- keep it separable.
    """
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx - SHADE_PAD_M, maxx + SHADE_PAD_M + step_m, step_m)
    ys = np.arange(maxy + SHADE_PAD_M, miny - SHADE_PAD_M - step_m, -step_m)
    gx, gy = np.meshgrid(xs, ys)

    centre = geom.centroid
    base = building_shading_factor(dsm_band, dsm_transform, dsm_nodata,
                                   centre.x, centre.y, hourly, own_geom=geom,
                                   terrain_horizon_profile=terrain_horizon_profile)
    grid = np.full(gx.shape, base, dtype=np.float32)

    # Only near the roof: everything else is discarded downstream anyway, and
    # a sprawling L-shaped footprint would otherwise pay for its whole bbox.
    near = shapely.contains_xy(geom.buffer(SHADE_PAD_M), gx, gy)
    rows, cols = np.nonzero(near)
    for r, c in zip(rows, cols):
        grid[r, c] = building_shading_factor(
            dsm_band, dsm_transform, dsm_nodata, float(gx[r, c]), float(gy[r, c]),
            hourly, own_geom=geom, terrain_horizon_profile=terrain_horizon_profile)
    return grid, xs, ys


def _sample_grid(grid, xs, ys, gx, gy):
    """Bilinear sample of a shading grid at arbitrary points."""
    if len(xs) < 2 or len(ys) < 2:
        return np.full(gx.shape, float(grid.flat[0]), dtype=np.float32)
    fi = np.clip((gx - xs[0]) / (xs[1] - xs[0]), 0, len(xs) - 1.001)
    fj = np.clip((ys[0] - gy) / (ys[0] - ys[1]), 0, len(ys) - 1.001)
    i0, j0 = fi.astype(int), fj.astype(int)
    ti, tj = fi - i0, fj - j0
    g00 = grid[j0, i0];       g10 = grid[j0, i0 + 1]
    g01 = grid[j0 + 1, i0];   g11 = grid[j0 + 1, i0 + 1]
    return ((g00 * (1 - ti) + g10 * ti) * (1 - tj)
            + (g01 * (1 - ti) + g11 * ti) * tj).astype(np.float32)


def render_building(points, geom, lookup, x_origin, y_origin, shading_factor):
    """Returns (poa_grid, row0, col0) for this building's bbox in the pilot-wide
    grid, or None when there's no usable point coverage. poa_grid is NaN
    outside the footprint / away from real points."""
    if len(points) < 12:
        return None
    minx, miny, maxx, maxy = geom.bounds
    col0 = int(np.floor((minx - x_origin) / HR_RES_M))
    row0 = int(np.floor((y_origin - maxy) / HR_RES_M))
    cols = int(np.ceil((maxx - minx) / HR_RES_M)) + 1
    rows = int(np.ceil((maxy - miny) / HR_RES_M)) + 1
    if cols <= 0 or rows <= 0:
        return None

    xs = x_origin + (col0 + np.arange(cols) + 0.5) * HR_RES_M
    ys = y_origin - (row0 + np.arange(rows) + 0.5) * HR_RES_M
    gx, gy = np.meshgrid(xs, ys)

    tree = cKDTree(points[:, :2])
    k = min(IDW_K, len(points))
    dist, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.maximum(dist, 0.05)
    z = (points[idx, 2] * w).sum(axis=1) / w.sum(axis=1)
    z = z.reshape(rows, cols)
    no_evidence = dist[:, 0].reshape(rows, cols) > IDW_MAX_DIST_M

    z_s = gaussian_filter(z, sigma=Z_SMOOTH_SIGMA_PX)
    dz_dy, dz_dx = np.gradient(z_s, HR_RES_M)
    dz_dy = -dz_dy  # rows increase southward; gradient wants northward-positive
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect_deg = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360

    slope_idx = np.clip(np.round(slope_deg / SLOPE_BIN_DEG).astype(int), 0, MAX_SLOPE_DEG // SLOPE_BIN_DEG)
    aspect_idx = np.round(aspect_deg / ASPECT_BIN_DEG).astype(int) % (360 // ASPECT_BIN_DEG)
    # shading_factor is either one scalar (legacy) or (grid, xs, ys) sampled
    # per pixel -- see shading_grid.
    if isinstance(shading_factor, tuple):
        shade = _sample_grid(shading_factor[0], shading_factor[1], shading_factor[2], gx, gy)
    else:
        shade = shading_factor
    poa = lookup[slope_idx, aspect_idx] * shade

    inside = shapely.contains_xy(geom, gx, gy)
    poa[~inside | no_evidence] = np.nan
    return poa.astype(np.float32), row0, col0


def main(area="pilot"):
    preflight("build_heatmap_raster", area)
    from src.region_build import area_paths, area_bbox_nztm, area_centroid_wgs84
    paths = area_paths(area)
    gdf = gpd.read_file(paths["outlines"])
    pc_source = PointCloudSource()
    dsm_ds = rasterio.open(paths["dsm"])
    dsm_band = dsm_ds.read(1)

    print(f"[{area}] Building solar yield lookup table...")
    centroid = area_centroid_wgs84(area)
    model = SolarModel() if centroid is None else SolarModel(*centroid)
    lookup = build_lookup_array(model)

    minx, miny, maxx, maxy = area_bbox_nztm(area)
    x_origin, y_origin = minx, maxy  # top-left
    width = int(np.ceil((maxx - minx) / HR_RES_M))
    height = int(np.ceil((maxy - miny) / HR_RES_M))
    print(f"Pilot grid {width}x{height} at {HR_RES_M}m")
    poa_full = np.full((height, width), np.nan, dtype=np.float32)

    t0 = time.time()
    rendered = 0
    dem_wide_path = DATA_DIR / "dem_wide_mosaic.tif"
    if dem_wide_path.exists():
        _dw = rasterio.open(dem_wide_path)
        dem_wide_band, dem_wide_transform, dem_wide_nodata = _dw.read(1), _dw.transform, _dw.nodata
    else:
        dem_wide_band = dem_wide_transform = dem_wide_nodata = None
        print(f"[{area}] WARNING: no data/dem_wide_mosaic.tif -- far-terrain "
              f"correction is OFF for this raster.", flush=True)

    for i, row in enumerate(gdf.itertuples()):
        bminx, bminy, bmaxx, bmaxy = row.geometry.bounds
        points = pc_source.points_in_bbox(bminx - 1, bminy - 1, bmaxx + 1, bmaxy + 1, building_only=True)
        centroid = row.geometry.centroid
        shade_grid, shade_xs, shade_ys = shading_grid(
            dsm_band, dsm_ds.transform, dsm_ds.nodata, row.geometry, model.hourly,
            terrain_horizon_profile=model.horizon_profile)
        # Per-building FAR terrain correction (see build_layout_geojson):
        # the lookup carries the AREA horizon; this building's own terrain may
        # differ. Scalar here (per-pixel aspects share the correction) -- the
        # panel/economics path gets the aspect-aware version.
        if dem_wide_band is not None:
            eave = _hz_eave_height(dsm_band, dsm_ds.transform, dsm_ds.nodata, row.geometry)
            far = _hz_far_profile(dem_wide_band, dem_wide_transform, dem_wide_nodata,
                                  row.geometry, eave)
            # Far terrain is kilometres away: one ratio per building is right
            # for it, unlike the near field this grid now resolves.
            shade_grid = shade_grid * _hz_far_ratio(far, model.horizon_profile, model.hourly)
        result = render_building(points, row.geometry, lookup, x_origin, y_origin,
                                 (shade_grid, shade_xs, shade_ys))
        if result is None:
            continue
        poa, r0, c0 = result
        rr0, cc0 = max(r0, 0), max(c0, 0)
        rr1 = min(r0 + poa.shape[0], height)
        cc1 = min(c0 + poa.shape[1], width)
        if rr1 <= rr0 or cc1 <= cc0:
            continue
        sub = poa[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
        target = poa_full[rr0:rr1, cc0:cc1]
        take = ~np.isnan(sub)
        target[take] = sub[take]
        rendered += 1
        if i % 200 == 0:
            print(f"  {i}/{len(gdf)} elapsed={time.time() - t0:.0f}s")

    covered = ~np.isnan(poa_full)
    print(f"{rendered}/{len(gdf)} buildings rendered, {covered.sum()} covered pixels "
          f"({covered.sum() * HR_RES_M ** 2:.0f} m2 of roof)")

    norm = Normalize(vmin=VMIN, vmax=VMAX)
    rgba = (HEAT_CMAP(norm(np.nan_to_num(poa_full, nan=VMIN))) * 255).astype(np.uint8)
    alpha = np.where(covered, float(ALPHA), 0.0)
    # Feather: soften the alpha edge AND the colour a touch so facet-scale
    # transitions read as thin gradients, not aliased staircases.
    alpha = gaussian_filter(alpha, sigma=EDGE_FEATHER_SIGMA_PX)
    alpha = np.minimum(alpha, np.where(gaussian_filter(covered.astype(float), 1.5) > 0.05, 255, 0))
    for ch in range(3):
        band = rgba[..., ch].astype(float)
        band_s = gaussian_filter(np.where(covered, band, 0.0), sigma=EDGE_FEATHER_SIGMA_PX)
        weight = gaussian_filter(covered.astype(float), sigma=EDGE_FEATHER_SIGMA_PX)
        rgba[..., ch] = np.where(weight > 0.02, band_s / np.maximum(weight, 0.02), band).astype(np.uint8)
    rgba[..., 3] = alpha.astype(np.uint8)

    out_png = paths["heatmap_png"]
    Image.fromarray(rgba, mode="RGBA").save(out_png, optimize=True)

    lons, lats = warp_transform("EPSG:2193", "EPSG:4326",
                                 [minx, maxx, maxx, minx], [maxy, maxy, miny, miny])
    coordinates = list(zip(lons, lats))
    meta_path = paths["heatmap_json"]
    meta_path.write_text(json.dumps({"coordinates": coordinates}))
    print(f"\nSaved {out_png} ({out_png.stat().st_size / 1e6:.1f}MB) and {meta_path}")

    dsm_ds.close()


if __name__ == "__main__":
    from src.region_build import areas_from_argv
    for _area in areas_from_argv(sys.argv):
        main(_area)
