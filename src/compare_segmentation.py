"""
Side-by-side comparison: old RANSAC-only segment_building() vs new
segment_building_image_guided(), for one building. Temporary diagnostic
for validating the image-guided roofline integration before it replaces
the old function pilot-wide.

Usage: python src/compare_segmentation.py <building_id>
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.roof_segmentation import segment_building, segment_building_image_guided
from src.obstruction_detection import detect_obstructions
from src.panel_fitting import fit_panels_on_facet

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def render(ax, facets, imagery_ds, row, title):
    all_panels, all_obs = [], []
    for f in facets:
        obs = detect_obstructions(imagery_ds, f["geometry"])
        all_obs.extend(obs)
        all_panels.extend(fit_panels_on_facet(f, obstructions=obs))

    minx, miny, maxx, maxy = row.geometry.bounds
    pad = 3
    window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, imagery_ds.transform)
    img = imagery_ds.read([1, 2, 3], window=window)
    img = np.moveaxis(img, 0, -1)
    wt = imagery_ds.window_transform(window)
    extent = (wt.c, wt.c + img.shape[1] * wt.a, wt.f + img.shape[0] * wt.e, wt.f)
    ax.imshow(img, extent=extent, origin="upper")

    for f in facets:
        xs, ys = f["geometry"].exterior.xy
        ax.plot(xs, ys, color="white", linewidth=1.2, linestyle=(0, (2, 2)))
    for o in all_obs:
        xs, ys = o.exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor="#a855f7", edgecolor="#a855f7", alpha=0.85))
    for p in all_panels:
        xs, ys = p["geometry"].exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor="#000000", edgecolor="#7fd4ff",
                                 linewidth=0.7, alpha=0.45))
    ax.set_title(f"{title}: {len(all_panels)} panels, {len(facets)} facets")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main(building_id):
    gdf = gpd.read_file(DATA_DIR / "building_outlines.geojson")
    row = gdf[gdf["building_id"] == building_id].iloc[0]
    dsm_ds = rasterio.open(DATA_DIR / "dsm_mosaic.tif")
    imagery_ds = rasterio.open(DATA_DIR / "imagery_mosaic.tif")

    facets_old = segment_building(dsm_ds, row.geometry, building_id)
    facets_new = segment_building_image_guided(dsm_ds, imagery_ds, row.geometry, building_id)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    render(axes[0], facets_old, imagery_ds, row, "OLD (RANSAC-only)")
    render(axes[1], facets_new, imagery_ds, row, "NEW (image-guided)")
    fig.tight_layout()
    out_path = DATA_DIR / f"compare_segmentation_{building_id}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
