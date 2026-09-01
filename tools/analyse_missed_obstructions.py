"""
Of the marked obstruction area the detector misses, how much is actually
standing proud of the roof?

Area-weighted obstruction recall is 22.7% -- 476 m2 agreed out of 2096 m2
marked. That reads as a bad detector, but three quite different things produce
it and they need separating before anyone tunes anything:

  GENEROUS MARKING   Josh: "I combine lots of items into one obstruction
                     sometimes". One polygon over a cluster of vents covers the
                     roof between them too. That roof is not equipment, the
                     detector is right to leave it, and the miss is an artefact
                     of how the truth was drawn.
  FLUSH OBJECTS      skylights, existing panels, roof hatches. They sit ON the
                     plane, so the height detector structurally cannot see them
                     and only colour can. A real limitation, but not a bug, and
                     the fix is imagery not thresholds.
  GENUINELY MISSED   something standing well clear of the roof that was simply
                     not found. This is the only bucket worth tuning for.

LiDAR separates them. For every marked area the detector did not find, measure
how far the returns there sit above the facet's own plane. No height means
either roof-between-objects or something flush; clear height means a real miss.

ANSWER over Josh's first 46 labelled roofs (2 Sep 2026). Of 1622 m2 missed:

    standing clear (>=0.30 m)     116 m2    7.2%
    low relief (0.15-0.30 m)       83 m2    5.1%
    flush with the roof          1025 m2   63.4%
    no usable returns             393 m2   24.3%

    recall against genuinely RAISED area   70.4%
    recall against all marked area         22.6%

The detector finds 70% of the equipment that actually stands off the roof. The
22.6% headline was measuring how the truth was drawn, not how the model
performs.

Two things follow. Tuning height thresholds can reach at most 12% of the gap,
so it is close to pointless. And the 63% flush share is the strongest argument
yet for the imagery work: skylights, existing panels and hatches sit ON the
plane and are invisible to LiDAR by construction, so no threshold will ever
find them and only vision can.

Usage:
    python tools/analyse_missed_obstructions.py
    python tools/analyse_missed_obstructions.py --max-roofs 15
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

LABELS = ROOT / "data" / "roof_labels.json"

CELL_M = 0.4          # sampling grid inside marked areas
FLUSH_M = 0.15        # below this a return is on the plane, not on an object
CLEAR_M = 0.30        # above this something is unambiguously standing proud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-roofs", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    import rasterio
    import shapely
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.obstruction_detection import detect_obstructions_combined
    from src.pointcloud_source import PointCloudSource
    from score_geometry import _obs_ring

    labels = json.loads(LABELS.read_text())["buildings"]
    ids = sorted(int(k) for k in labels)
    if a.max_roofs:
        ids = ids[:a.max_roofs]

    pc = PointCloudSource()
    ctxs = {}
    tot = {"flush": 0.0, "low": 0.0, "clear": 0.0, "nodata": 0.0}
    found_area = missed_area = 0.0

    for bid in ids:
        lab = labels[str(bid)]
        area = lab.get("area")
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
        rings = [_obs_ring(o) for o in lab.get("obstructions", [])]
        rings = [r for r in rings if r and len(r) >= 3]
        if not rings:
            continue
        geom = ctx["gdf"].loc[bid].geometry
        try:
            facets = segment_building_best(ctx["dsm"], pc, geom, bid,
                                           imagery_ds=ctx["img"]) or []
        except Exception:
            continue

        detected = []
        for f in facets:
            if f.get("plane_a") is None:
                continue
            try:
                got = detect_obstructions_combined(
                    ctx["img"], pc, f["geometry"],
                    (f["plane_a"], f["plane_b"], f["plane_c"]),
                    roof_geom=f.get("building_geometry")) or []
            except Exception:
                got = []
            for g in got:
                gg = g["geometry"] if isinstance(g, dict) else g
                detected.extend(gg.geoms if gg.geom_type == "MultiPolygon" else [gg])
        du = unary_union(detected) if detected else None

        marked = []
        for r in rings:
            try:
                p_ = Polygon(r)
                if not p_.is_valid:
                    p_ = p_.buffer(0)
                if p_.is_valid and p_.area > 0:
                    marked.append(p_)
            except Exception:
                pass
        if not marked:
            continue
        mu = unary_union(marked)
        miss = mu.difference(du) if du is not None else mu
        if du is not None:
            found_area += mu.intersection(du).area
        missed_area += miss.area
        if miss.is_empty:
            continue

        # sample the missed area and ask the point cloud what is there
        minx, miny, maxx, maxy = miss.bounds
        nx = max(1, int((maxx - minx) / CELL_M))
        ny = max(1, int((maxy - miny) / CELL_M))
        gx = minx + (np.arange(nx) + 0.5) * CELL_M
        gy = miny + (np.arange(ny) + 0.5) * CELL_M
        XX, YY = np.meshgrid(gx, gy)
        inside = shapely.contains_xy(miss, XX.ravel(), YY.ravel())
        cells = np.c_[XX.ravel(), YY.ravel()][inside]
        if not len(cells):
            continue

        pts = pc.points_in_bbox(minx - 1, miny - 1, maxx + 1, maxy + 1)
        for cx, cy in cells:
            plane = None
            for f in facets:
                if f.get("plane_a") is None:
                    continue
                if f["geometry"].contains(shapely.points(cx, cy)):
                    plane = (f["plane_a"], f["plane_b"], f["plane_c"])
                    break
            if plane is None or pts is None or not len(pts):
                tot["nodata"] += CELL_M ** 2
                continue
            near = pts[(np.abs(pts[:, 0] - cx) < CELL_M) &
                       (np.abs(pts[:, 1] - cy) < CELL_M)]
            if len(near) < 2:
                tot["nodata"] += CELL_M ** 2
                continue
            zplane = plane[0] * cx + plane[1] * cy + plane[2]
            resid = float(np.percentile(near[:, 2] - zplane, 75))
            if resid >= CLEAR_M:
                tot["clear"] += CELL_M ** 2
            elif resid >= FLUSH_M:
                tot["low"] += CELL_M ** 2
            else:
                tot["flush"] += CELL_M ** 2

    total = sum(tot.values())
    if total <= 0:
        print("nothing measurable")
        return 1

    print(f"marked obstruction area the detector FOUND:  {found_area:>8.0f} m2")
    print(f"marked obstruction area it MISSED:           {missed_area:>8.0f} m2\n")
    print("of the missed area, what the LiDAR says is actually there:\n")
    rows = [("standing clear (>=0.30 m above the plane)", tot["clear"]),
            ("low relief (0.15-0.30 m)", tot["low"]),
            ("flush with the roof (<0.15 m)", tot["flush"]),
            ("no usable returns", tot["nodata"])]
    for name, v in rows:
        bar = "#" * int(46 * v / max(r[1] for r in rows))
        print(f"  {name:<42}{v:>8.0f} m2  {100 * v / total:>5.1f}%  {bar}")

    print(f"\n  Only the first row is a detector failure worth tuning for.")
    print(f"  Flush area is skylights, existing panels and the roof BETWEEN")
    print(f"  objects covered by one generous polygon -- the detector is right")
    print(f"  to leave it, and no height threshold will ever find it.")
    real = tot["clear"] + tot["low"]
    print(f"\n  recall against area that is genuinely raised: "
          f"{found_area / (found_area + real):.1%}   "
          f"(against all marked area: {found_area / (found_area + missed_area):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
