"""Dev tool: tight zoom on one building to closely check facet quality."""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.roof_segmentation import segment_building
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
        print(f"building {bid}: {len(facets)} facets")
        for f in facets:
            print(f"  slope={f['slope_deg']:.1f} aspect={f['aspect_deg']:.0f} area={f['area_m2']:.1f} pts={f['point_count']}")

        minx, miny, maxx, maxy = row.geometry.bounds
        pad = 5
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
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor=color, edgecolor="black",
                                     alpha=0.65, linewidth=0.8))
            cx, cy = poly.centroid.x, poly.centroid.y
            ax.annotate(f"{f['slope_deg']:.0f}/{f['aspect_deg']:.0f}", (cx, cy), fontsize=8, ha="center",
                        color="white", weight="bold")
        ax.set_title(f"building {bid} ({row.geometry.area:.0f}m2)")
        ax.set_aspect("equal")

    out_path = DATA_DIR / "zoom_check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    dsm_ds.close()


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]]
    main(ids)
