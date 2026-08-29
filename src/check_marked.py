"""What the pipeline produces for one building, in the terms Josh marks in.

Run this the moment a marked roof comes back: it prints the faces with their
own fit, the obstructions, and how much roof falls outside the footprint, all
from the SAME calls build_layout_geojson makes -- so what is printed is what
would ship, not a tool's private idea of the pipeline.

Usage: python src/check_marked.py <building_id> [--area pilot]
"""
import argparse, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, geopandas as gpd, rasterio, shapely.vectorized
from shapely.geometry import Point
from shapely.ops import unary_union
from src.region_build import area_paths
from src.pointcloud_source import PointCloudSource
from src.roof_segmentation import segment_building_best
from src.obstruction_detection import detect_obstructions_combined
from src.roof_partition import _points_in, _inlier_fraction

ap = argparse.ArgumentParser()
ap.add_argument("building_id", type=int)
ap.add_argument("--area", default="pilot")
a = ap.parse_args()

p = area_paths(a.area)
gdf = gpd.read_file(p["outlines"]).set_index("building_id", drop=False)
g = gdf.loc[a.building_id].geometry
dsm = rasterio.open(p["dsm"]); img = rasterio.open(p["imagery"]); pc = PointCloudSource()
mnx, mny, mxx, mxy = g.bounds
print(f"#{a.building_id}  footprint {g.area:.1f} m2")
print(f"  grid labels  x {[int(v % 1000) for v in np.arange(np.ceil(mnx/5)*5, mxx, 5)]}"
      f"  y {[int(v % 1000) for v in np.arange(np.ceil(mny/5)*5, mxy, 5)]}")
print(f"  world origin x {int(mnx//1000)*1000}  y {int(mny//1000)*1000}")

facets = segment_building_best(dsm, pc, g, a.building_id, imagery_ds=img)
raw = pc.points_in_bbox(mnx-4, mny-4, mxx+4, mxy+4, building_only=True)
pts = raw[shapely.vectorized.contains(g, raw[:, 0], raw[:, 1])]
roof = unary_union([f["geometry"] for f in facets]) if facets else None
print(f"\nFACES: {len(facets)}")
for f in sorted(facets, key=lambda x: -x["geometry"].area):
    pl = np.array([f["plane_a"], f["plane_b"], f["plane_c"]])
    sub = _points_in(f["geometry"], pts)
    fit = _inlier_fraction(sub, pl) if len(sub) >= 5 else float("nan")
    c = f["geometry"].centroid
    print(f"  {f['geometry'].area:7.1f} m2  slope {f['slope_deg']:5.1f}  "
          f"aspect {f['aspect_deg']:6.1f}  fit {fit:6.1%}  "
          f"at ({c.x % 1000:.1f}, {c.y % 1000:.1f})")

obs = []
for f in facets:
    plane = (f["plane_a"], f["plane_b"], f["plane_c"])
    obs += detect_obstructions_combined(img, pc, f["geometry"], plane,
                                        roof_geom=f.get("building_geometry"))
if obs and roof is not None:
    u = unary_union([o.buffer(0) for o in obs]).intersection(roof)
    print(f"\nOBSTRUCTIONS: {len(obs)} covering {u.area:.1f} m2 "
          f"({u.area/roof.area:.1%} of roof)")
    for o in sorted(obs, key=lambda x: -x.area)[:8]:
        c = o.centroid
        print(f"  {o.area:6.1f} m2 at ({c.x % 1000:.1f}, {c.y % 1000:.1f})")
else:
    print("\nOBSTRUCTIONS: none")

zin = pts[:, 2]
lo, hi = np.percentile(zin, [5, 95])
near = raw[(raw[:, 2] > lo - 0.5) & (raw[:, 2] < hi + 0.5)]
out = near[~shapely.vectorized.contains(g, near[:, 0], near[:, 1])]
if len(out):
    d = np.array([g.distance(Point(x, y)) for x, y in out[:, :2]])
    k = d <= 3.0
    print(f"\nOVERHANG: {int(k.sum())} roof-height points outside the footprint "
          f"({k.sum()/(k.sum()+len(pts)):.1%} of this roof), median {np.median(d[k]):.2f} m out")
if roof is not None:
    print(f"modelled roof {roof.area:.1f} m2 vs footprint {g.area:.1f} m2")
