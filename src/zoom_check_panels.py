"""Dev tool: visualize fitted panels on top of segmented facets for a few buildings."""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.roof_segmentation import segment_building
from src.panel_fitting import fit_panels_on_facet
from src.visualize_segmentation import hillshade

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main(building_ids):
    gdf = gpd.read_file(DATA_DIR / "building_outlines.geojson")
    dsm_ds = rasterio.open(DATA_DIR / "dsm_mosaic.tif")

    fig, axes = plt.subplots(1, len(building_ids), figsize=(7 * len(building_ids), 7))
    if len(building_ids) == 1:
        axes = [axes]

    for ax, bid in zip(axes, building_ids):
        row = gdf[gdf["building_id"] == bid].iloc[0]
        facets = segment_building(dsm_ds, row.geometry, bid)
        all_panels = []
        for f in facets:
            panels = fit_panels_on_facet(f)
            all_panels.extend(panels)
        n_panels = len(all_panels)
        kwp = n_panels * config_panel_power()
        print(f"building {bid}: {len(facets)} facets, {n_panels} panels ({sum(p['area_m2'] for p in all_panels):.1f}m2 of panel)")

        minx, miny, maxx, maxy = row.geometry.bounds
        pad = 4
        window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, dsm_ds.transform)
        arr = dsm_ds.read(1, window=window)
        wt = dsm_ds.window_transform(window)
        shade = hillshade(arr, dsm_ds.nodata)
        extent = (wt.c, wt.c + arr.shape[1] * wt.a, wt.f + arr.shape[0] * wt.e, wt.f)
        ax.imshow(shade, cmap="gray", extent=extent, origin="upper")

        xs, ys = row.geometry.exterior.xy
        ax.plot(xs, ys, color="cyan", linewidth=1.5)

        cmap = plt.cm.hsv
        for f in facets:
            poly = f["geometry"]
            color = cmap(f["aspect_deg"] / 360)
            xs, ys = poly.exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor=color, edgecolor="none", alpha=0.3))

        for p in all_panels:
            xs, ys = p["geometry"].exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor="none", edgecolor="red", linewidth=1.2))

        ax.set_title(f"building {bid}: {n_panels} panels")
        ax.set_aspect("equal")

    out_path = DATA_DIR / "zoom_check_panels.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    dsm_ds.close()


def config_panel_power():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
    return config.PV_ASSUMPTIONS["panel_rated_power_w"] / 1000


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]]
    main(ids)
