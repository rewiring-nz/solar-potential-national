"""
Reproduce the live map's exact colour scheme (dotted facet outline only
-- no fill, semi-transparent black panels, purple obstructions) in
matplotlib for one building, to check whether a "solid purple" look is a
MapLibre-specific rendering issue or shows up in any renderer given the
same data.

Usage: python src/debug_render_check.py <building_id>
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
from src.roof_segmentation import segment_building
from src.obstruction_detection import detect_obstructions
from src.panel_fitting import fit_panels_on_facet

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main(building_id):
    gdf = gpd.read_file(DATA_DIR / "building_outlines.geojson")
    row = gdf[gdf["building_id"] == building_id].iloc[0]
    dsm_ds = rasterio.open(DATA_DIR / "dsm_mosaic.tif")
    imagery_ds = rasterio.open(DATA_DIR / "imagery_mosaic.tif")

    facets = segment_building(dsm_ds, row.geometry, building_id)
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

    fig, ax = plt.subplots(figsize=(12, 12))
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

    ax.set_title(f"Building #{building_id}: {len(all_panels)} panels, {len(all_obs)} obstructions "
                 f"-- same colours as the live map")
    ax.set_aspect("equal")
    out_path = DATA_DIR / f"debug_render_{building_id}.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"Saved {out_path}")
    dsm_ds.close()
    imagery_ds.close()


if __name__ == "__main__":
    main(int(sys.argv[1]))
