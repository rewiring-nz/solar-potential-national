"""
Show one rooftop's panel layout at multiple zoom levels, plus a
"straightened" view rotated so the main facet's edge is horizontal --
the closest equivalent to multiple viewing angles achievable from
nadir-only (straight-down) aerial imagery. LINZ doesn't provide oblique
imagery, so this is context/detail crops of the same top-down photo, not
true 3D perspectives -- said plainly here rather than implied.

Usage: python src/demo_multi_angle.py <building_id>
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import rotate as ndi_rotate
from shapely.affinity import rotate as shapely_rotate, translate as shapely_translate

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.roof_segmentation import segment_building
from src.obstruction_detection import detect_obstructions
from src.panel_fitting import fit_panels_on_facet, _edge_aligned_axes
from src.solar_model import SolarModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def draw_facets_and_panels(ax, facets, all_panels, obstructions_by_facet):
    cmap = plt.cm.YlOrRd
    irr = [f.get("_poa", 0) for f in facets]
    vmin, vmax = (min(irr), max(irr)) if irr and min(irr) != max(irr) else (0, 1)
    for f in facets:
        poly = f["geometry"]
        color = cmap((f.get("_poa", vmin) - vmin) / (vmax - vmin + 1e-9))
        xs, ys = poly.exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor=color, edgecolor="white",
                                 linewidth=0.6, alpha=0.55))
    for obs_list in obstructions_by_facet:
        for o in obs_list:
            oxs, oys = o.exterior.xy
            ax.add_patch(MplPolygon(list(zip(oxs, oys)), closed=True, facecolor="red", edgecolor="red", alpha=0.6))
    for p in all_panels:
        xs, ys = p["geometry"].exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor="#0a1f44", edgecolor="#7fd4ff", linewidth=0.7))


def main(building_id):
    gdf = gpd.read_file(DATA_DIR / "building_outlines.geojson")
    row = gdf[gdf["building_id"] == building_id].iloc[0]
    dsm_ds = rasterio.open(DATA_DIR / "dsm_mosaic.tif")
    imagery_ds = rasterio.open(DATA_DIR / "imagery_mosaic.tif")

    model = SolarModel()
    facets = segment_building(dsm_ds, row.geometry, building_id)
    all_panels, obstructions_by_facet = [], []
    for f in facets:
        f["_poa"] = model.annual_poa_kwh_per_m2(f["slope_deg"], f["aspect_deg"])
        obs = detect_obstructions(imagery_ds, f["geometry"])
        obstructions_by_facet.append(obs)
        all_panels.extend(fit_panels_on_facet(f, obstructions=obs))

    def imshow_crop(ax, pad):
        minx, miny, maxx, maxy = row.geometry.bounds
        window = rasterio.windows.from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, imagery_ds.transform)
        img = imagery_ds.read([1, 2, 3], window=window)
        img = np.moveaxis(img, 0, -1)
        wt = imagery_ds.window_transform(window)
        extent = (wt.c, wt.c + img.shape[1] * wt.a, wt.f + img.shape[0] * wt.e, wt.f)
        ax.imshow(img, extent=extent, origin="upper")
        return extent

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    imshow_crop(axes[0, 0], pad=15)
    draw_facets_and_panels(axes[0, 0], facets, all_panels, obstructions_by_facet)
    xs, ys = row.geometry.exterior.xy
    axes[0, 0].plot(xs, ys, color="cyan", linewidth=1.5)
    axes[0, 0].set_title("Wide context (street + neighbours)")

    imshow_crop(axes[0, 1], pad=2)
    draw_facets_and_panels(axes[0, 1], facets, all_panels, obstructions_by_facet)
    axes[0, 1].plot(xs, ys, color="cyan", linewidth=1.5)
    axes[0, 1].set_title("Building-only crop")

    biggest_facet = max(facets, key=lambda f: f["area_m2"])
    fminx, fminy, fmaxx, fmaxy = biggest_facet["geometry"].bounds
    window = rasterio.windows.from_bounds(fminx - 1, fminy - 1, fmaxx + 1, fmaxy + 1, imagery_ds.transform)
    img = imagery_ds.read([1, 2, 3], window=window)
    img = np.moveaxis(img, 0, -1)
    wt = imagery_ds.window_transform(window)
    extent = (wt.c, wt.c + img.shape[1] * wt.a, wt.f + img.shape[0] * wt.e, wt.f)
    axes[1, 0].imshow(img, extent=extent, origin="upper")
    draw_facets_and_panels(axes[1, 0], [biggest_facet],
                            [p for p in all_panels if p["facet_aspect_deg"] == biggest_facet["aspect_deg"]],
                            [obstructions_by_facet[facets.index(biggest_facet)]])
    axes[1, 0].set_xlim(fminx - 1, fmaxx + 1)
    axes[1, 0].set_ylim(fminy - 1, fmaxy + 1)
    axes[1, 0].set_title(f"Close-up: largest facet ({biggest_facet['area_m2']:.0f}m2) -- setback/obstruction detail")

    u_hat, _ = _edge_aligned_axes(biggest_facet["geometry"], biggest_facet["aspect_deg"])
    edge_angle_deg = np.degrees(np.arctan2(u_hat[1], u_hat[0]))
    rot_img = ndi_rotate(img, angle=edge_angle_deg, reshape=True, order=1)
    cx, cy = biggest_facet["geometry"].centroid.x, biggest_facet["geometry"].centroid.y
    axes[1, 1].imshow(rot_img, origin="upper")
    rot_ax_transform = lambda geom: shapely_rotate(shapely_translate(geom, -cx, -cy), edge_angle_deg, origin=(0, 0))
    h, w = rot_img.shape[:2]
    px_per_m = 1 / abs(wt.a)
    for p in all_panels:
        if p["facet_aspect_deg"] != biggest_facet["aspect_deg"]:
            continue
        g = rot_ax_transform(p["geometry"])
        xs2, ys2 = g.exterior.xy
        px = [w / 2 + x * px_per_m for x in xs2]
        py = [h / 2 - y * px_per_m for y in ys2]
        axes[1, 1].add_patch(MplPolygon(list(zip(px, py)), closed=True, facecolor="#0a1f44", edgecolor="#7fd4ff", linewidth=0.8))
    axes[1, 1].set_title("Straightened: rotated so this facet's edge is horizontal\n(closest equivalent to a square-on view -- imagery is nadir-only, no true oblique angle exists)")
    axes[1, 1].set_xlim(w / 2 - 15 * px_per_m, w / 2 + 15 * px_per_m)
    axes[1, 1].set_ylim(h / 2 + 15 * px_per_m, h / 2 - 15 * px_per_m)

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Building #{building_id}: {len(all_panels)} panels -- multiple views of the same rooftop", fontsize=14)
    out_path = DATA_DIR / f"demo_multi_angle_{building_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    dsm_ds.close()
    imagery_ds.close()


if __name__ == "__main__":
    main(int(sys.argv[1]))
