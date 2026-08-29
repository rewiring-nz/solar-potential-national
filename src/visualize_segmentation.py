"""
Dev tool: run roof_segmentation on a handful of real buildings and render
an overlay PNG (DSM hillshade + footprint outline + facet polygons
coloured by aspect) so segmentation quality can be checked by eye before
running it over all 1270 buildings.

Usage: python src/visualize_segmentation.py [n_buildings]
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.roof_segmentation import segment_building

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def hillshade(arr, nodata):
    mask = arr == nodata
    gy, gx = np.gradient(np.where(mask, np.nan, arr))
    gy, gx = np.nan_to_num(gy), np.nan_to_num(gx)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.deg2rad(315), np.deg2rad(45)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    shade[mask] = np.nan
    return shade


def main(n_buildings=12):
    gdf = gpd.read_file(DATA_DIR / "building_outlines.geojson")
    # Sort by area descending, take a spread: some big, some median-sized,
    # so segmentation gets tested against both easy and hard (few-pixel) cases.
    gdf["_area"] = gdf.geometry.area
    gdf = gdf.sort_values("_area", ascending=False).reset_index(drop=True)
    n = len(gdf)
    sample_idx = sorted(set(list(range(0, min(n, n_buildings // 2))) +
                             list(np.linspace(n_buildings, n - 1, n_buildings - n_buildings // 2, dtype=int))))
    sample = gdf.iloc[sample_idx]

    dsm_ds = rasterio.open(DATA_DIR / "dsm_mosaic.tif")

    all_facets = []
    for _, row in sample.iterrows():
        facets = segment_building(dsm_ds, row.geometry, row["building_id"])
        all_facets.extend(facets)
        print(f"building {row['building_id']}: area={row.geometry.area:.1f}m2 -> {len(facets)} facets "
              + ", ".join(f"[{f['slope_deg']:.0f}deg/{f['aspect_deg']:.0f}deg, {f['area_m2']:.1f}m2]" for f in facets))

    minx, miny, maxx, maxy = sample.total_bounds
    pad = 20
    window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, dsm_ds.transform)
    arr = dsm_ds.read(1, window=window)
    win_transform = dsm_ds.window_transform(window)
    shade = hillshade(arr, dsm_ds.nodata)

    fig, ax = plt.subplots(figsize=(14, 14))
    extent = (win_transform.c, win_transform.c + arr.shape[1] * win_transform.a,
              win_transform.f + arr.shape[0] * win_transform.e, win_transform.f)
    ax.imshow(shade, cmap="gray", extent=extent, origin="upper")

    sample.boundary.plot(ax=ax, color="cyan", linewidth=0.8)

    cmap = plt.cm.hsv
    for f in all_facets:
        poly = f["geometry"]
        color = cmap(f["aspect_deg"] / 360)
        xs, ys = poly.exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor=color, edgecolor="black",
                                 alpha=0.6, linewidth=0.5))

    ax.set_title(f"Roof segmentation: {len(sample)} buildings, {len(all_facets)} facets "
                 f"(colour = aspect, cyan = building footprint)")
    ax.set_aspect("equal")
    out_path = DATA_DIR / "segmentation_preview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")
    dsm_ds.close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(n)
