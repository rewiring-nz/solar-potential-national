"""Dev tool: visualize detected obstructions and their effect on panel fitting."""
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
from src.obstruction_detection import detect_obstructions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main(building_ids):
    gdf = gpd.read_file(DATA_DIR / "building_outlines.geojson")
    dsm_ds = rasterio.open(DATA_DIR / "dsm_mosaic.tif")
    imagery_ds = rasterio.open(DATA_DIR / "imagery_mosaic.tif")

    fig, axes = plt.subplots(2, len(building_ids), figsize=(7 * len(building_ids), 14))
    if len(building_ids) == 1:
        axes = axes.reshape(2, 1)

    for col, bid in enumerate(building_ids):
        row = gdf[gdf["building_id"] == bid].iloc[0]
        facets = segment_building(dsm_ds, row.geometry, bid)

        minx, miny, maxx, maxy = row.geometry.bounds
        pad = 3
        window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, imagery_ds.transform)
        img = imagery_ds.read([1, 2, 3], window=window)
        img = np.moveaxis(img, 0, -1)
        wt = imagery_ds.window_transform(window)
        extent = (wt.c, wt.c + img.shape[1] * wt.a, wt.f + img.shape[0] * wt.e, wt.f)

        for r in range(2):
            axes[r, col].imshow(img, extent=extent, origin="upper")
            xs, ys = row.geometry.exterior.xy
            axes[r, col].plot(xs, ys, color="cyan", linewidth=1.5)
            axes[r, col].set_aspect("equal")

        all_obstructions = []
        n_panels_before = n_panels_after = 0
        for f in facets:
            obs = detect_obstructions(imagery_ds, f["geometry"])
            all_obstructions.extend(obs)
            n_panels_before += len(fit_panels_on_facet(f))
            panels_after = fit_panels_on_facet(f, obstructions=obs)
            n_panels_after += len(panels_after)
            for p in panels_after:
                pxs, pys = p["geometry"].exterior.xy
                axes[1, col].add_patch(MplPolygon(list(zip(pxs, pys)), closed=True, facecolor="none",
                                                    edgecolor="lime", linewidth=1.2))

        for o in all_obstructions:
            oxs, oys = o.exterior.xy
            axes[0, col].add_patch(MplPolygon(list(zip(oxs, oys)), closed=True, facecolor="red",
                                                edgecolor="red", alpha=0.5))
            axes[1, col].add_patch(MplPolygon(list(zip(oxs, oys)), closed=True, facecolor="red",
                                                edgecolor="red", alpha=0.5))

        print(f"building {bid}: {len(facets)} facets, {len(all_obstructions)} obstructions flagged, "
              f"panels {n_panels_before} -> {n_panels_after}")
        axes[0, col].set_title(f"building {bid}: flagged obstructions (red)")
        axes[1, col].set_title(f"panels avoiding obstructions: {n_panels_after}")

    out_path = DATA_DIR / "zoom_check_obstructions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    dsm_ds.close()
    imagery_ds.close()


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]]
    main(ids)
