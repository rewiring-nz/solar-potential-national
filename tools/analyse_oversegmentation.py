"""
Where the segmenter draws a line the labeller did not, is anything actually
there?

Precision is now the weaker half of the baseline (53.1% against 64.9% recall),
which means roughly half of predicted line length is structure nobody drew. That
is either over-segmentation -- cutting a roof where the surface does not change
-- or the labeller not bothering with a real but minor feature. Those need
different responses and a percentage cannot separate them.

So measure the roof itself at each predicted line. For a segment, take LiDAR
points in a narrow band either side, fit a plane to each, and ask two questions
the partition itself is supposed to be asking:

  FOLD   the angle between the two surface normals. A real ridge, hip or valley
         has a fold; a false cut across one continuous plane has none.
  STEP   the height difference across the line. A real cliff has a step; a cut
         through flat roof has none.

If lines the labeller DREW have folds and steps, and lines they did NOT have
neither, then the partition is cutting where nothing happens and the fix is a
threshold. If both look alike, the geometry is genuinely ambiguous and no
threshold will separate them -- which is worth knowing before spending days
tuning one.

Usage:
    python tools/analyse_oversegmentation.py
    python tools/analyse_oversegmentation.py --band 0.8 --tol 0.75
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


def plane_of(pts):
    """Least-squares plane; returns (normal, mean z) or None."""
    import numpy as np
    if len(pts) < 6:
        return None
    A = np.c_[pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean(),
              np.ones(len(pts))]
    try:
        coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    except Exception:
        return None
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n) or 1.0
    return n, float(np.mean(pts[:, 2]))


def fold_and_step(pc, a, b, band):
    """Normal angle and height offset across one segment."""
    import numpy as np
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    dx, dy = bx - ax, by - ay
    L = float(np.hypot(dx, dy))
    if L < 0.5:
        return None
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux                       # unit normal to the segment

    pad = band + 0.5
    pts = pc.points_in_bbox(min(ax, bx) - pad, min(ay, by) - pad,
                            max(ax, bx) + pad, max(ay, by) + pad)
    if pts is None or len(pts) < 12:
        return None
    rel_x = pts[:, 0] - ax
    rel_y = pts[:, 1] - ay
    along = rel_x * ux + rel_y * uy
    across = rel_x * nx + rel_y * ny
    on = (along > 0.15 * L) & (along < 0.85 * L)   # ignore the ends
    left = pts[on & (across > 0.12) & (across < band)]
    right = pts[on & (across < -0.12) & (across > -band)]
    pl, pr = plane_of(left), plane_of(right)
    if pl is None or pr is None:
        return None
    cosang = float(np.clip(abs(np.dot(pl[0], pr[0])), -1, 1))
    fold = float(np.degrees(np.arccos(cosang)))
    step = abs(pl[1] - pr[1])
    return fold, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=1.0,
                    help="metres either side of a line to fit planes in")
    ap.add_argument("--tol", type=float, default=0.75)
    ap.add_argument("--max-roofs", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import geopandas as gpd
    import rasterio
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.obstruction_detection import detect_obstructions_combined
    from src.pointcloud_source import PointCloudSource
    from score_geometry import (line_scores, predicted_lines_from_facets,
                                predicted_obstruction_lines, _line_points)

    labels = json.loads(LABELS.read_text())["buildings"]
    ids = sorted(int(k) for k in labels)
    if a.max_roofs:
        ids = ids[:a.max_roofs]

    pc = PointCloudSource()
    ctxs = {}
    matched, unmatched = [], []
    n_lines = 0

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
        geom = ctx["gdf"].loc[bid].geometry
        true_lines = [t for t in (_line_points(l) for l in lab.get("lines", [])) if t]
        if not true_lines:
            continue
        try:
            facets = segment_building_best(ctx["dsm"], pc, geom, bid,
                                           imagery_ds=ctx["img"]) or []
        except Exception:
            continue

        obs_rings = []
        for f in facets:
            if f.get("plane_a") is None:
                continue
            try:
                found = detect_obstructions_combined(
                    ctx["img"], pc, f["geometry"],
                    (f["plane_a"], f["plane_b"], f["plane_c"]),
                    roof_geom=f.get("building_geometry")) or []
            except Exception:
                found = []
            for g in found:
                gg = g["geometry"] if isinstance(g, dict) else g
                for pgon in (gg.geoms if gg.geom_type == "MultiPolygon" else [gg]):
                    obs_rings.append(list(pgon.exterior.coords))

        pred = (predicted_lines_from_facets(facets, geom)
                + predicted_obstruction_lines(obs_rings, geom))
        for seg in pred:
            n_lines += 1
            s = line_scores([seg], true_lines, a.tol)
            if not s:
                continue
            fs = fold_and_step(pc, seg[0], seg[1], a.band)
            if fs is None:
                continue
            (matched if s["precision"] >= 0.5 else unmatched).append(fs)

    if not matched or not unmatched:
        print("not enough measurable lines")
        return 1

    def describe(name, rows):
        f = np.array([r[0] for r in rows])
        s = np.array([r[1] for r in rows])
        print(f"  {name:<34} n={len(rows)}")
        print(f"     fold angle  median {np.median(f):>5.1f} deg   "
              f"p25 {np.percentile(f, 25):>5.1f}   p75 {np.percentile(f, 75):>5.1f}")
        print(f"     height step median {np.median(s):>5.2f} m     "
              f"p25 {np.percentile(s, 25):>5.2f}   p75 {np.percentile(s, 75):>5.2f}")

    print(f"measured {n_lines} predicted segments across {len(ids)} roofs\n")
    describe("lines the labeller DID draw", matched)
    describe("lines the labeller did NOT draw", unmatched)

    mf = np.array([r[0] for r in matched]); uf = np.array([r[0] for r in unmatched])
    ms = np.array([r[1] for r in matched]); us = np.array([r[1] for r in unmatched])
    print(f"\n  separation:")
    print(f"    fold  {np.median(mf):.1f} deg vs {np.median(uf):.1f} deg")
    print(f"    step  {np.median(ms):.2f} m vs {np.median(us):.2f} m")

    # What a threshold could actually buy: how much of the unmatched set sits
    # below a bar that keeps most of the matched set.
    print(f"\n  if a cut needed a real fold or step to survive:")
    for fold_min, step_min in ((3.0, 0.15), (5.0, 0.25), (8.0, 0.40)):
        keep_m = float(np.mean((mf >= fold_min) | (ms >= step_min)))
        keep_u = float(np.mean((uf >= fold_min) | (us >= step_min)))
        print(f"    fold>={fold_min:>4.1f} deg or step>={step_min:.2f} m  ->  "
              f"keeps {keep_m:.0%} of drawn lines, {keep_u:.0%} of undrawn")
    print("\n  A threshold is only worth having where it drops far more undrawn")
    print("  lines than drawn ones. If both columns fall together, the geometry")
    print("  does not separate them and tuning will trade recall for precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
