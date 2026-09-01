"""
How far is the LINZ building outline from the roof the LiDAR actually sees?

Josh: "There are a lot of errors where the building outline is not aligned with
the rooftop... you might cut off a lot of edges if the building outline is the
limit."

He is describing two different things and they need separating, because only one
of them costs panels:

  IMAGERY LEAN is a display artefact. An aerial photo is taken at an angle, so a
  tall roof appears displaced from its true ground position. It makes the photo
  disagree with the outline on screen. It does NOT move the roof.

  OUTLINE ERROR is real. If the surveyed footprint is smaller than, or offset
  from, the actual roof, then every stage that clips to it -- facet geometry
  (roof_segmentation clips with polygon.intersection(building_geom)), the point
  filter (contains_xy(footprint, ...)), panel fitting -- silently discards real
  roof. That is lost capacity nobody sees.

LiDAR settles it. The point cloud is georeferenced 3D with no perspective
displacement, so where it says roof is, roof is. This measures the outline
against that, per building:

  ROOF OUTSIDE   roof-height area beyond the outline, which the pipeline drops
  OUTLINE EMPTY  outline area with no roof over it, which it wastes effort on
  BEST SHIFT     the (dx, dy) that maximises agreement -- a consistent non-zero
                 shift across many buildings would mean a systematic
                 georeferencing offset rather than per-building survey error

Usage:
    python tools/measure_outline_offset.py              # every labelled roof
    python tools/measure_outline_offset.py --ids 4719759
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
LABELS = DATA_DIR / "roof_labels.json"

PAD_M = 6.0          # look this far outside the outline for roof it is missing
CELL_M = 0.5         # sampling grid; finer than the 1 m DSM buys nothing
MIN_HEIGHT_M = 2.0   # same bar gate_panels uses for "this is a roof, not ground"
SHIFT_MAX_M = 2.0    # widest translation considered
SHIFT_STEP_M = 0.25


def roof_mask(pc, geom, ground_z, others=None):
    """A grid over the padded footprint, marking cells with roof-height returns.

    Deliberately built from the point cloud rather than the DSM: the DSM is a
    1 m raster already resampled, and the question here is about sub-metre
    boundary placement."""
    import numpy as np
    from shapely.geometry import Point

    minx, miny, maxx, maxy = geom.bounds
    x0, y0 = minx - PAD_M, miny - PAD_M
    x1, y1 = maxx + PAD_M, maxy + PAD_M
    nx = max(1, int((x1 - x0) / CELL_M))
    ny = max(1, int((y1 - y0) / CELL_M))

    pts = pc.points_in_bbox(x0, y0, x1, y1)
    if pts is None or len(pts) == 0:
        return None, None, (x0, y0, nx, ny)

    high = pts[pts[:, 2] >= ground_z + MIN_HEIGHT_M]
    roof = np.zeros((ny, nx), dtype=bool)
    if len(high):
        ix = ((high[:, 0] - x0) / CELL_M).astype(int)
        iy = ((high[:, 1] - y0) / CELL_M).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        roof[iy[ok], ix[ok]] = True

    # THE NEIGHBOURS. points_in_bbox already filters to LAS class 6, so trees
    # and ground are gone -- but class 6 is EVERY building, and at a 6 m pad the
    # house next door is squarely inside the window. Without this the measure
    # reports the neighbour's roof as roof this building is losing, which in a
    # dense street is most of the answer.
    
    gx = x0 + (np.arange(nx) + 0.5) * CELL_M
    gy = y0 + (np.arange(ny) + 0.5) * CELL_M
    XX, YY = np.meshgrid(gx, gy)
    import shapely
    inside = shapely.contains_xy(geom, XX.ravel(), YY.ravel()).reshape(ny, nx)
    if others is not None and not others.is_empty:
        on_neighbour = shapely.contains_xy(
            others, XX.ravel(), YY.ravel()).reshape(ny, nx)
        roof = np.logical_and(roof, ~on_neighbour)

    # CONNECTEDNESS. Excluding mapped neighbours is not enough: a shed, carport
    # or garage that LINZ never digitised has no outline to exclude it by, and
    # on a large building the search window reaches several of them. Measured
    # distances proved it -- 46% of the "missing roof" sat MORE than 6 m beyond
    # the outline, which is not an eave and not a survey error, it is a separate
    # structure. Only roof CONTIGUOUS with this building's own roof can be roof
    # this building is losing, so keep the connected components that touch it.
    seed = np.logical_and(roof, inside)
    if seed.any():
        roof = _connected_to(roof, seed)
    return roof, inside, (x0, y0, nx, ny)


def _connected_to(mask, seed):
    """Components of `mask` that touch `seed`, 8-connected."""
    import numpy as np
    try:
        from scipy import ndimage
        lab, k = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        keep = set(np.unique(lab[seed])) - {0}
        return np.isin(lab, list(keep))
    except ImportError:
        pass
    # no scipy: grow the seed outward until it stops changing
    out = seed.copy()
    while True:
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown[1:, 1:] |= out[:-1, :-1]
        grown[:-1, :-1] |= out[1:, 1:]
        grown[1:, :-1] |= out[:-1, 1:]
        grown[:-1, 1:] |= out[1:, :-1]
        grown &= mask
        if grown.sum() == out.sum():
            return grown
        out = grown


def best_shift(roof, inside):
    """Translation of the OUTLINE that best covers the roof cells."""
    import numpy as np
    steps = int(SHIFT_MAX_M / SHIFT_STEP_M)
    best = (0.0, 0.0, -1.0)
    for sy in range(-steps, steps + 1):
        for sx in range(-steps, steps + 1):
            shifted = np.roll(np.roll(inside, sy, axis=0), sx, axis=1)
            inter = np.logical_and(shifted, roof).sum()
            union = np.logical_or(shifted, roof).sum()
            iou = inter / union if union else 0.0
            if iou > best[2]:
                best = (sx * CELL_M, sy * CELL_M, iou)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    from src.region_build import area_paths
    from src.pointcloud_source import PointCloudSource

    if not LABELS.exists():
        print("no labels yet")
        return 2
    labels = json.loads(LABELS.read_text())["buildings"]
    ids = a.ids or sorted(int(k) for k in labels)

    pc = PointCloudSource()
    ctxs = {}
    cell_m2 = CELL_M * CELL_M
    rows = []

    print(f"outline vs the roof the LiDAR sees, {len(ids)} buildings\n")
    print(f"{'building':>10}{'roof m2':>9}{'outside':>9}{'%lost':>7}"
          f"{'empty':>8}{'IoU':>6}{'best shift':>13}{'nbrs':>6}")

    for bid in ids:
        lab = labels[str(bid)]
        area = lab.get("area")
        if area not in ctxs:
            p = area_paths(area)
            if not p["outlines"].exists():
                continue
            ctxs[area] = gpd.read_file(p["outlines"]).set_index("building_id", drop=False)
        gdf = ctxs[area]
        if bid not in gdf.index:
            continue
        geom = gdf.loc[bid].geometry

        c = geom.centroid
        g = pc.ground_points_in_bbox(c.x - 25, c.y - 25, c.x + 25, c.y + 25)
        if g is None or len(g) < 20:
            continue
        ground_z = float(np.percentile(g[:, 2], 50))

        # every OTHER outline overlapping this window, dissolved
        win = geom.buffer(PAD_M + 1)
        near = gdf[gdf.geometry.intersects(win)]
        near = near[near["building_id"] != bid]
        from shapely.ops import unary_union
        others = unary_union(list(near.geometry)) if len(near) else None

        roof, inside, _ = roof_mask(pc, geom, ground_z, others)
        if roof is None or not roof.any():
            continue

        roof_area = roof.sum() * cell_m2
        outside = np.logical_and(roof, ~inside).sum() * cell_m2
        empty = np.logical_and(inside, ~roof).sum() * cell_m2
        inter = np.logical_and(roof, inside).sum()
        union = np.logical_or(roof, inside).sum()
        iou = inter / union if union else 0.0
        dx, dy, best_iou = best_shift(roof, inside)

        rows.append({"id": bid, "neighbours": len(near), "roof": roof_area, "outside": outside,
                     "empty": empty, "iou": iou, "dx": dx, "dy": dy,
                     "gain": best_iou - iou})
        print(f"{bid:>10}{roof_area:>9.0f}{outside:>9.0f}"
              f"{100 * outside / roof_area:>6.0f}%{empty:>8.0f}"
              f"{iou:>6.2f}{f'{dx:+.2f},{dy:+.2f}':>13}{len(near):>6}")

    if not rows:
        print("nothing measurable")
        return 1

    tot_roof = sum(r["roof"] for r in rows)
    tot_out = sum(r["outside"] for r in rows)
    tot_empty = sum(r["empty"] for r in rows)
    mdx = float(np.median([r["dx"] for r in rows]))
    mdy = float(np.median([r["dy"] for r in rows]))
    mean_gain = float(np.mean([r["gain"] for r in rows]))
    shifted = sum(1 for r in rows if abs(r["dx"]) >= 0.5 or abs(r["dy"]) >= 0.5)

    print(f"\nover {len(rows)} buildings:")
    print(f"  roof seen by LiDAR      {tot_roof:>8.0f} m2")
    print(f"  outside the outline     {tot_out:>8.0f} m2  "
          f"({100 * tot_out / tot_roof:.1f}% -- clipped away today)")
    print(f"  outline with no roof    {tot_empty:>8.0f} m2")
    print(f"  median best shift        {mdx:+.2f}, {mdy:+.2f} m")
    print(f"  buildings wanting >=0.5 m {shifted} of {len(rows)}")
    print(f"  mean IoU gain from shifting {mean_gain:+.3f}")
    print("\nA median shift near zero with individual buildings wanting large,")
    print("DIFFERENT shifts means per-building survey error, not a systematic")
    print("georeferencing offset -- so a global correction would not help and")
    print("the fix has to be per building.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
