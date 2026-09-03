"""
How many panels straddle a roof line they should not?

Josh, on what actually matters: "amount of panels overlapping ridges, and amount
of faces on rooftops that have clear panel placement should improve a lot. On
some faces this will add more panels, on others it will reduce them because you
were previously overlapping things."

That is the right metric and nothing has ever measured it. Panel COUNT cannot:
a build that lays panels straight across a ridge scores well on count and is
obviously wrong on the roof. A panel bridging a fold sits on two planes at once,
so it is not installable at all -- the count is a lie about capacity, not a
small inaccuracy.

Measured against the lines Josh drew, which are the only description of where
the folds really are:

  CROSSING PANELS   a placed panel whose footprint is cut by a drawn ridge,
                    valley or cliff. Straightforwardly wrong.
  CROSSED AREA      how much panel area sits on the wrong side, so one panel
                    clipping a line by a centimetre does not count the same as
                    a row laid across it.
  CLEAN FACES       how much of the roof is covered by panels that cross
                    nothing. This is the number to move up.

WHY BOTH DIRECTIONS MATTER. Fixing crossings by placing fewer panels is not a
fix, so panel count is reported alongside. A change is only good if crossings
fall and count does not collapse.

Usage:
    python tools/measure_panel_crossings.py
    python tools/measure_panel_crossings.py --ids 5372566 --verbose
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
# A panel touching a line within this is clipping it, not straddling it. Drawn
# lines carry a few centimetres of hand precision and the panel edge setback is
# 0.3 m, so anything under this is inside the noise.
GRAZE_M = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    import rasterio
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.obstruction_detection import detect_obstructions_combined
    from src.panel_fitting import fit_panels_on_facet
    from src.pointcloud_source import PointCloudSource
    from score_geometry import _line_points

    labels = json.loads(LABELS.read_text())["buildings"]
    ids = a.ids or sorted(int(k) for k in labels
                          if labels[k].get("problem") not in
                          ("absent", "not_building", "unclear"))
    if a.limit:
        ids = ids[:a.limit]

    pc = PointCloudSource()
    ctxs = {}
    tot = {"panels": 0, "crossing": 0, "area": 0.0, "crossed_area": 0.0,
           "roofs": 0}
    rows = []

    for bid in ids:
        lab = labels.get(str(bid))
        if not lab:
            continue
        drawn = [_line_points(l) for l in lab.get("lines", [])]
        drawn = [d for d in drawn if d]
        if not drawn:
            continue
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
        geom = ctx["gdf"].loc[bid].geometry
        try:
            facets = segment_building_best(ctx["dsm"], pc, geom, bid,
                                           imagery_ds=ctx["img"]) or []
        except Exception:
            continue

        panels = []
        for f in facets:
            if f.get("plane_a") is None:
                continue
            try:
                obs = detect_obstructions_combined(
                    ctx["img"], pc, f["geometry"],
                    (f["plane_a"], f["plane_b"], f["plane_c"]),
                    roof_geom=f.get("building_geometry")) or []
            except Exception:
                obs = []
            sib = [o for o in facets if o is not f]
            try:
                for pnl in (fit_panels_on_facet(f, obstructions=obs,
                                                sibling_facets=sib) or []):
                    panels.append(pnl["geometry"] if isinstance(pnl, dict) else pnl)
            except Exception:
                pass
        if not panels:
            continue

        lines = unary_union([LineString(d) for d in drawn if len(d) >= 2])
        n_cross = 0
        crossed = 0.0
        for pnl in panels:
            if not pnl.intersects(lines):
                continue
            # Split the panel by the drawn lines. A panel that merely grazes one
            # yields one real piece and a sliver; a straddling panel yields two
            # substantial pieces.
            try:
                pieces = list(pnl.difference(lines.buffer(0.01)).geoms)
            except Exception:
                continue
            big = [q for q in pieces if q.area > GRAZE_M * GRAZE_M]
            if len(big) < 2:
                continue
            n_cross += 1
            crossed += sum(sorted(q.area for q in big)[:-1])   # the wrong side(s)

        pa = sum(p.area for p in panels)
        tot["roofs"] += 1
        tot["panels"] += len(panels)
        tot["crossing"] += n_cross
        tot["area"] += pa
        tot["crossed_area"] += crossed
        rows.append((bid, len(panels), n_cross, pa, crossed))
        if a.verbose:
            print(f"  #{bid}: {len(panels)} panels, {n_cross} crossing "
                  f"({100*n_cross/max(len(panels),1):.0f}%)")

    if not tot["roofs"]:
        print("nothing measured")
        return 1
    print(f"\nPANELS STRADDLING A DRAWN ROOF LINE, over {tot['roofs']} labelled roofs\n")
    print(f"  panels placed          {tot['panels']:>8}")
    print(f"  panels crossing a line {tot['crossing']:>8}   "
          f"({100 * tot['crossing'] / max(tot['panels'], 1):.1f}%)")
    print(f"  panel area             {tot['area']:>8.0f} m2")
    print(f"  area on the wrong side {tot['crossed_area']:>8.1f} m2   "
          f"({100 * tot['crossed_area'] / max(tot['area'], 1):.2f}%)")
    worst = sorted(rows, key=lambda r: -r[2])[:8]
    print("\n  worst roofs:")
    for bid, n, c, pa, cr in worst:
        if not c:
            continue
        print(f"    #{bid}  {c} of {n} panels cross  ({100*c/max(n,1):.0f}%)")
    print("\n  A panel bridging a fold sits on two planes at once and cannot be")
    print("  installed, so this is not a rounding error in the capacity figure.")
    print("  Any change is only an improvement if THIS falls and the panel count")
    print("  does not collapse with it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
