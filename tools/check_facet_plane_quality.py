"""
Do facet planes actually fit the roof they claim to describe?

A facet carries a plane, and everything downstream trusts it: panel tilt, the
height test that decides what is an obstruction, the yield calculation. If the
plane is wrong the panels are placed on a surface that is not there.

MEASURE IT ROBUSTLY OR NOT AT ALL. Taking the RMS over every LiDAR point inside
a facet says 48.7% of roof area sits on a plane off by more than 0.35 m, which
is alarming and wrong. Rooftop equipment stands proud of the plane BY
DEFINITION, so on any roof with plant the residuals are dominated by the very
objects the plane is supposed to help detect. Judge the plane on the central 80%
of residuals and the same roofs give 5.9%.

Result over Josh's first 22 labelled roofs (2 Sep 2026), robust:

    RMS < 0.10 m      5781 m2   74.7%
    RMS 0.10-0.20 m    306 m2    4.0%
    RMS 0.20-0.35 m   1191 m2   15.4%
    RMS > 0.35 m       459 m2    5.9%

So plane fitting is NOT a problem: three quarters of roof area sits within 10 cm
of its own plane, and what is genuinely poor is concentrated in small facets
rather than the large ones. This rules out bad planes as a cause of the
segmenter's disagreement with the drawn labels.

Usage:
    python tools/check_facet_plane_quality.py
    python tools/check_facet_plane_quality.py --n 30 --naive   # show the trap
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LABELS = ROOT / "data" / "roof_labels.json"

BANDS = [("<0.10", 0.10), ("0.10-0.20", 0.20), ("0.20-0.35", 0.35),
         ("0.35-0.60", 0.60), (">0.60", float("inf"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--naive", action="store_true",
                    help="use every point, which rooftop plant inflates")
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    import rasterio
    import shapely
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.pointcloud_source import PointCloudSource

    labels = json.loads(LABELS.read_text())["buildings"]
    ids = sorted(int(k) for k in labels)[:a.n]
    pc = PointCloudSource()
    ctxs, band, worst, tot = {}, {b[0]: 0.0 for b in BANDS}, [], 0.0

    for bid in ids:
        area = labels[str(bid)].get("area")
        if area not in ctxs:
            p = area_paths(area)
            ctxs[area] = None if not (p["outlines"].exists() and p["dsm"].exists()) else {
                "gdf": gpd.read_file(p["outlines"]).set_index("building_id", drop=False),
                "dsm": rasterio.open(p["dsm"]),
                "img": rasterio.open(p["imagery"]) if p["imagery"].exists() else None,
            }
        ctx = ctxs[area]
        if not ctx or bid not in ctx["gdf"].index:
            continue
        geom = ctx["gdf"].loc[bid].geometry
        try:
            facets = segment_building_best(ctx["dsm"], pc, geom, bid,
                                           imagery_ds=ctx["img"]) or []
        except Exception:
            continue
        pts = pc.points_in_bbox(*geom.bounds)
        if pts is None or not len(pts):
            continue
        for f in facets:
            if f.get("plane_a") is None:
                continue
            g = f["geometry"]
            ins = pts[shapely.contains_xy(g, pts[:, 0], pts[:, 1])]
            if len(ins) < 12:
                continue
            r = ins[:, 2] - (f["plane_a"] * ins[:, 0] + f["plane_b"] * ins[:, 1]
                             + f["plane_c"])
            if a.naive:
                rms = float(np.sqrt(np.mean(r ** 2)))
            else:
                lo, hi = np.percentile(r, [10, 90])
                core = r[(r >= lo) & (r <= hi)]
                rms = float(np.sqrt(np.mean(core ** 2)))
            inl = float(np.mean(np.abs(r) <= 0.15))
            tot += g.area
            for name, hi_ in BANDS:
                if rms < hi_:
                    band[name] += g.area
                    break
            if rms > 0.35:
                worst.append((rms, inl, g.area, bid))

    if tot <= 0:
        print("nothing measured")
        return 1
    mode = "NAIVE (every point -- plant inflates this)" if a.naive else "ROBUST (central 80%)"
    print(f"facet plane fit, {mode}, by roof area:\n")
    mx = max(band.values()) or 1
    for name, _ in BANDS:
        v = band[name]
        print(f"  RMS {name:<10}{v:>8.0f} m2  {100 * v / tot:>5.1f}%  "
              f"{'#' * int(44 * v / mx)}")
    bad = band["0.35-0.60"] + band[">0.60"]
    print(f"\n  total {tot:.0f} m2 over {len(ids)} roofs")
    print(f"  area off by more than 0.35 m: {bad:.0f} m2 ({100 * bad / tot:.1f}%)")
    worst.sort(reverse=True)
    if worst:
        print("\n  worst facets (share of points within 15 cm of the plane):")
        for rms, inl, ar, bid in worst[:6]:
            print(f"    #{bid}  {ar:>6.0f} m2  RMS {rms:.2f} m   {inl:.0%} on-plane")
    return 0


if __name__ == "__main__":
    sys.exit(main())
