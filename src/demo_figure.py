"""
Polished demonstration figure: real aerial imagery background, roof facets
coloured by annual solar irradiance (the "how much sunshine hits this part
of the roof" heatmap), and fitted panels drawn on top -- both halves of
the pipeline in one picture, on one real building.

Usage: python src/demo_figure.py <building_id>
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.roof_segmentation import segment_building
from src.obstruction_detection import detect_obstructions
from src.panel_fitting import fit_panels_on_facet
from src.solar_model import SolarModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main(building_id):
    gdf = gpd.read_file(DATA_DIR / "building_outlines.geojson")
    row = gdf[gdf["building_id"] == building_id].iloc[0]

    dsm_ds = rasterio.open(DATA_DIR / "dsm_mosaic.tif")
    imagery_ds = rasterio.open(DATA_DIR / "imagery_mosaic.tif")

    print("Loading solar model (pvlib + NASA POWER)...")
    model = SolarModel()

    facets = segment_building(dsm_ds, row.geometry, building_id)
    all_panels = []
    facet_irradiance = []
    for f in facets:
        obstructions = detect_obstructions(imagery_ds, f["geometry"])
        panels = fit_panels_on_facet(f, obstructions=obstructions)
        all_panels.extend(panels)
        poa = model.annual_poa_kwh_per_m2(f["slope_deg"], f["aspect_deg"])
        facet_irradiance.append(poa)
        print(f"  facet: slope={f['slope_deg']:.0f} deg, aspect={f['aspect_deg']:.0f} deg, "
              f"{poa:.0f} kWh/m2/yr, {len(panels)} panels ({len(obstructions)} obstructions avoided)")

    import config
    kwp = len(all_panels) * config.PV_ASSUMPTIONS["panel_rated_power_w"] / 1000

    minx, miny, maxx, maxy = row.geometry.bounds
    pad = 6
    window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, imagery_ds.transform)
    img = imagery_ds.read([1, 2, 3], window=window)
    img = np.moveaxis(img, 0, -1)
    wt = imagery_ds.window_transform(window)
    extent = (wt.c, wt.c + img.shape[1] * wt.a, wt.f + img.shape[0] * wt.e, wt.f)

    # Same fixed 700-1650 kWh/m2/yr scale and diverging blue(south)-yellow-red(north)
    # palette as preview.html's live legend, so this figure and the map read consistently.
    norm = Normalize(vmin=700, vmax=1650)
    cmap = plt.cm.RdYlBu_r

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, extent=extent, origin="upper")

    for f, poa in zip(facets, facet_irradiance):
        poly = f["geometry"]
        color = cmap(norm(poa))
        xs, ys = poly.exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor=color, edgecolor="white",
                                 linewidth=0.8, alpha=0.72))

    for p in all_panels:
        xs, ys = p["geometry"].exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor="#0a1f44", edgecolor="#7fd4ff",
                                 linewidth=0.6, alpha=0.88))

    xs, ys = row.geometry.exterior.xy
    ax.plot(xs, ys, color="cyan", linewidth=1.8)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Annual solar irradiance on roof facet (kWh/m²/yr)")

    ax.set_title(f"Building #{building_id}: {len(all_panels)} panels, {kwp:.1f} kWp\n"
                 f"Facet colour = sunshine received, navy = fitted panels", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    out_path = DATA_DIR / f"demo_figure_{building_id}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved {out_path}")
    dsm_ds.close()
    imagery_ds.close()


if __name__ == "__main__":
    main(int(sys.argv[1]))
