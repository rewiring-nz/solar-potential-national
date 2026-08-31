"""
Render one building's segmentation + panels over imagery, for diagnosing
facet-boundary quality. Companion to refit_one.py — same pipeline call.

Usage:
    python src/render_building_debug.py 5119630 [--out /tmp/x.png] [--strategy best]

--strategy lets you render a single segmentation strategy instead of the
segment_building_best winner: best | regiongrow | global | native
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, MultiPolygon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import shapely
from src.refit_one import _area_of, _load
from src.roof_segmentation import (
    segment_building_best,
    segment_building_from_pointcloud_regiongrow,
    segment_building_from_pointcloud_global,
    segment_building_from_pointcloud_native,
)
from src.obstruction_detection import detect_obstructions_combined
from src.panel_fitting import fit_panels_on_facet

def _skeleton(ctx, geom, bid):
    from src.roof_partition import top_surface
    from src.roof_skeleton import skeleton_roof
    minx, miny, maxx, maxy = geom.bounds
    pts = ctx["pc"].points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
    pts = pts[shapely.contains_xy(geom, pts[:, 0], pts[:, 1])]
    return skeleton_roof(bid, geom.buffer(0), top_surface(pts))


def _arrangement(ctx, geom, bid):
    from src.roof_segmentation import _arrangement_facets
    minx, miny, maxx, maxy = geom.bounds
    pts = ctx["pc"].points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
    pts = pts[shapely.contains_xy(geom, pts[:, 0], pts[:, 1])]
    return _arrangement_facets(pts, geom, bid)


def _byplanes(ctx, geom, bid):
    from src.roof_partition import partition_by_planes
    minx, miny, maxx, maxy = geom.bounds
    pts = ctx["pc"].points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1, building_only=True)
    pts = pts[shapely.contains_xy(geom, pts[:, 0], pts[:, 1])]
    return partition_by_planes(bid, geom.buffer(0), pts)


STRATEGIES = {
    "skeleton": _skeleton,
    "arrangement": _arrangement,
    "byplanes": _byplanes,
    "best": lambda ctx, geom, bid: segment_building_best(
        ctx["dsm"], ctx["pc"], geom, bid, imagery_ds=ctx["img"]),
    "regiongrow": lambda ctx, geom, bid: segment_building_from_pointcloud_regiongrow(
        ctx["pc"], geom, bid),
    "global": lambda ctx, geom, bid: segment_building_from_pointcloud_global(
        ctx["pc"], geom, bid),
    "native": lambda ctx, geom, bid: segment_building_from_pointcloud_native(
        ctx["dsm"], geom, bid),
}


def _draw_geom(ax, geom, **kw):
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if not isinstance(p, Polygon) or p.is_empty:
            continue
        ax.add_patch(MplPoly(np.asarray(p.exterior.coords), closed=True, **kw))
        for ring in p.interiors:
            ax.add_patch(MplPoly(np.asarray(ring.coords), closed=True,
                                 facecolor="none", edgecolor=kw.get("edgecolor"),
                                 linewidth=kw.get("linewidth", 1)))


def render(building_id, out, strategy="best", with_panels=True):
    area, _ = _area_of(building_id)
    if area is None:
        raise SystemExit(f"#{building_id} not found")
    ctx = _load(area)
    row = ctx["gdf"].loc[building_id]
    geom = row.geometry

    facets = STRATEGIES[strategy](ctx, geom, building_id)

    pad = 4.0
    minx, miny, maxx, maxy = geom.bounds
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad

    fig, ax = plt.subplots(figsize=(14, 14))
    if ctx["img"] is not None:
        w = from_bounds(minx, miny, maxx, maxy, ctx["img"].transform)
        img = ctx["img"].read([1, 2, 3], window=w)
        ax.imshow(np.transpose(img, (1, 2, 0)), extent=(minx, maxx, miny, maxy))

    cmap = plt.get_cmap("tab20")
    for i, f in enumerate(facets):
        c = cmap(i % 20)
        _draw_geom(ax, f["geometry"], facecolor=(*c[:3], 0.25), edgecolor=c,
                   linewidth=2.5)
        cx, cy = f["geometry"].centroid.x, f["geometry"].centroid.y
        ax.annotate(f"{i}: {f['slope_deg']:.0f}°/{f['aspect_deg']:.0f}°"
                    f"\n{f['geometry'].area:.0f}m²",
                    (cx, cy), color="white", fontsize=9, ha="center",
                    path_effects=None,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55))
        if with_panels:
            plane = (f["plane_a"], f["plane_b"], f["plane_c"])
            obst = detect_obstructions_combined(ctx["img"], ctx["pc"], f["geometry"], plane)
            for o in obst:
                _draw_geom(ax, o, facecolor=(1, 0, 0, 0.35), edgecolor="red", linewidth=1)
            sibs = [o for o in facets if o is not f]
            for p in fit_panels_on_facet(f, obstructions=obst, sibling_facets=sibs):
                _draw_geom(ax, p["geometry"], facecolor=(0.03, 0.12, 0.25, 0.75),
                           edgecolor="#7fd4ff", linewidth=0.8)

    _draw_geom(ax, geom, facecolor="none", edgecolor="yellow", linewidth=1.5)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"#{building_id}  {area}  strategy={strategy}  facets={len(facets)}")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}  ({len(facets)} facets)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    out = "/tmp/building_debug.png"
    strategy = "best"
    if "--out" in argv:
        i = argv.index("--out"); out = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if "--strategy" in argv:
        i = argv.index("--strategy"); strategy = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    render(int(argv[0]), out, strategy)
