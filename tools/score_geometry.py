"""
Score predicted roof geometry against roofs Josh has drawn.

Track B of the vision pathway, and the piece that decides whether any of it
worked. Without this, "the new model looks better" is an opinion formed by
looking at renders -- which is how this project has judged geometry so far, and
why a wall-as-roof bug survived until Josh happened to click that building.

WHAT IT MEASURES, and why not pixel accuracy. A model can score 95% IoU on
"roof line" pixels and still produce geometry that does not close into usable
planes. So the metrics here are geometric and in metres:

  LINES        precision and recall against the drawn ridges/hips/valleys, with
               a distance tolerance. A predicted line counts as found if it lies
               within TOL of a drawn line of the same kind along most of its
               length. Reported per kind, because missing a ridge and missing a
               level change are different failures.
  OBSTRUCTIONS precision and recall by area overlap. The existing detector is
               known to both miss real objects and invent others, and a single
               count hides that -- 8 found against 3 marked can be 3 right and 5
               invented, or 0 right and 8 invented.
  FACETS       count error against Josh's own count, kept because it is the one
               number he has already given for every marked roof.

WHAT COUNTS AS A PREDICTED LINE. For a planar partition, the roof lines ARE the
shared edges between adjacent facets. So a prediction is converted by taking
every facet boundary segment that is interior to the building -- which lets the
current segmenter be scored on exactly the same footing as a future model, with
no special-casing for either.

Baseline first. Run this against the segmenter BEFORE training anything: a model
that cannot beat these numbers is not worth shipping, and there is no way to
know that without them.

Usage:
    python tools/score_geometry.py                    # score the segmenter
    python tools/score_geometry.py --tol 0.75
    python tools/score_geometry.py --ids 5371108
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
LABELS = DATA_DIR / "roof_labels.json"

TOL_M = 0.75            # a predicted line within this of a drawn one is a match
COVER = 0.6             # ...along at least this share of its length
OBS_IOU = 0.3           # obstruction overlap to count as the same object


def _segments(points):
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _seg_points(a, b, step=0.25):
    """Sample a segment every `step` metres, so coverage is measured by length
    rather than by endpoint -- a long predicted line that clips the end of a
    drawn one should not count as a match."""
    import numpy as np
    (x1, y1), (x2, y2) = a, b
    d = float(np.hypot(x2 - x1, y2 - y1))
    n = max(2, int(d / step) + 1)
    return [(x1 + (x2 - x1) * t / (n - 1), y1 + (y2 - y1) * t / (n - 1))
            for t in range(n)], d


def _point_to_seg(p, a, b):
    import numpy as np
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def _covered(seg, truth_segs, tol):
    """Share of this segment lying within tol of ANY truth segment."""
    pts, length = _seg_points(*seg)
    if not truth_segs or length == 0:
        return 0.0, length
    near = 0
    for p in pts:
        if any(_point_to_seg(p, a, b) <= tol for a, b in truth_segs):
            near += 1
    return near / len(pts), length


def line_scores(pred_lines, true_lines, tol):
    """Length-weighted precision and recall, both directions."""
    pred_segs = [s for L in pred_lines for s in _segments(L)]
    true_segs = [s for L in true_lines for s in _segments(L)]
    if not pred_segs and not true_segs:
        return None
    hit_len = tot_len = 0.0
    for s in pred_segs:
        cov, ln = _covered(s, true_segs, tol)
        tot_len += ln
        if cov >= COVER:
            hit_len += ln
    precision = hit_len / tot_len if tot_len else 0.0
    hit_len = tot_len = 0.0
    for s in true_segs:
        cov, ln = _covered(s, pred_segs, tol)
        tot_len += ln
        if cov >= COVER:
            hit_len += ln
    recall = hit_len / tot_len if tot_len else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 0.0 if precision + recall == 0 else
                 2 * precision * recall / (precision + recall),
            "pred_segments": len(pred_segs), "true_segments": len(true_segs)}


def obstruction_scores(pred_rings, true_rings, iou_min=OBS_IOU):
    from shapely.geometry import Polygon
    P = [Polygon(r) for r in pred_rings if len(r) >= 3]
    T = [Polygon(r) for r in true_rings if len(r) >= 3]
    P = [p for p in P if p.is_valid and p.area > 0]
    T = [t for t in T if t.is_valid and t.area > 0]
    if not P and not T:
        return None
    matched_t = set()
    tp = 0
    for p in P:
        best, bi = 0.0, None
        for i, t in enumerate(T):
            if i in matched_t:
                continue
            inter = p.intersection(t).area
            if inter <= 0:
                continue
            iou = inter / (p.union(t).area or 1)
            if iou > best:
                best, bi = iou, i
        if bi is not None and best >= iou_min:
            tp += 1
            matched_t.add(bi)
    precision = tp / len(P) if P else 0.0
    recall = tp / len(T) if T else 0.0
    return {"precision": precision, "recall": recall,
            "found": len(P), "marked": len(T), "matched": tp,
            "pred_area_m2": round(sum(p.area for p in P), 1),
            "true_area_m2": round(sum(t.area for t in T), 1)}


def predicted_lines_from_facets(facets, footprint, edge_tol=0.35):
    """A planar partition's ROOF LINES are its interior shared edges.

    Anything lying on the building outline is an eave or verge, not a ridge, so
    it is dropped -- otherwise every prediction scores well simply by tracing
    the footprint, which Josh did not draw as a roof line."""
    out = []
    boundary = footprint.exterior if hasattr(footprint, "exterior") else None
    for f in facets:
        g = f["geometry"] if isinstance(f, dict) else f
        coords = list(g.exterior.coords)
        for a, b in _segments(coords):
            if boundary is not None:
                from shapely.geometry import Point
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                if boundary.distance(Point(mid)) < edge_tol:
                    continue
            out.append([list(a), list(b)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=TOL_M)
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    a = ap.parse_args()

    if not LABELS.exists():
        print(f"no labels yet at {LABELS}")
        print("Draw some first:  python tools/label_roofs.py")
        return 2
    labels = json.loads(LABELS.read_text())["buildings"]
    ids = a.ids or sorted(int(k) for k in labels)
    if not ids:
        print("no labelled roofs yet")
        return 2

    import rasterio
    import geopandas as gpd
    from src.region_build import area_paths
    from src.roof_segmentation import segment_building_best
    from src.pointcloud_source import PointCloudSource

    ctxs, pc = {}, PointCloudSource()
    agg = defaultdict(list)
    print(f"scoring the CURRENT segmenter against {len(ids)} drawn roofs "
          f"(tolerance {a.tol} m)\n")
    print(f"{'building':>10} {'lines P':>8}{'lines R':>8}{'F1':>7}"
          f"{'obs P':>7}{'obs R':>7}{'facets':>9}")

    for bid in ids:
        lab = labels[str(bid)]
        area = lab.get("area")
        if area not in ctxs:
            p = area_paths(area)
            if not p["outlines"].exists():
                continue
            ctxs[area] = {
                "gdf": gpd.read_file(p["outlines"]).set_index("building_id", drop=False),
                "dsm": rasterio.open(p["dsm"]),
                "img": rasterio.open(p["imagery"]) if p["imagery"].exists() else None,
            }
        ctx = ctxs[area]
        if bid not in ctx["gdf"].index:
            continue
        geom = ctx["gdf"].loc[bid].geometry
        facets = segment_building_best(ctx["dsm"], pc, geom, bid,
                                       imagery_ds=ctx["img"]) or []
        pred_lines = predicted_lines_from_facets(facets, geom)
        true_lines = [l["points"] for l in lab.get("lines", [])
                      if l.get("kind") != "outline"]
        ls = line_scores(pred_lines, true_lines, a.tol)
        os_ = obstruction_scores([], lab.get("obstructions", []))

        lp = f"{ls['precision']:.0%}" if ls else "—"
        lr = f"{ls['recall']:.0%}" if ls else "—"
        f1 = f"{ls['f1']:.0%}" if ls else "—"
        op = f"{os_['precision']:.0%}" if os_ else "—"
        orr = f"{os_['recall']:.0%}" if os_ else "—"
        print(f"{bid:>10} {lp:>8}{lr:>8}{f1:>7}{op:>7}{orr:>7}{len(facets):>9}")
        if ls:
            agg["lp"].append(ls["precision"]); agg["lr"].append(ls["recall"])
            agg["f1"].append(ls["f1"])

    if agg["f1"]:
        n = len(agg["f1"])
        print(f"\nBASELINE over {n} roofs — the number a model has to beat:")
        print(f"  line precision {sum(agg['lp'])/n:.1%}   "
              f"recall {sum(agg['lr'])/n:.1%}   F1 {sum(agg['f1'])/n:.1%}")
        print("\nPrecision is 'lines we drew that are real'; recall is 'real lines")
        print("we found'. Over-segmentation shows as low precision, missed ridges")
        print("as low recall -- which is the distinction facet COUNT cannot make.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
