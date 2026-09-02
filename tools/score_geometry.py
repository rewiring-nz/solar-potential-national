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

WHAT COUNTS AS A PREDICTED LINE. For a planar partition, the roof lines are the
shared edges between adjacent facets -- but that is not the whole story, and
scoring only those understated the segmenter badly.

The pipeline has TWO ways to represent a raised feature. It can partition it, in
which case its edges show up as facet boundaries; or it can carve it out as an
obstruction, in which case they do not. For a dormer, carving is arguably the
better choice -- you cannot usefully panel its sides and you must avoid it --
but a labeller draws its edges as cliff lines either way.

Measured on Josh's first 46 roofs: 61% of MISSED cliff lines, 55% of missed
valleys and 48% of missed ridges had a detected obstruction sitting within a
metre. Most "missed" lines were being found and simply represented the other
way. So obstruction boundaries count as predicted lines too. This costs
precision honestly -- claiming those edges as structure means being scored on
them -- and it stops the metric punishing a representation choice.

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
    """Overlap by AREA, deliberately not by count.

    Josh: "numbers are a bad way to measure obstructions because I combine
    lots of items into one obstruction sometimes." Exactly right, and it
    breaks one-to-one matching outright: draw a single polygon over a cluster
    of five vents, have the detector find five separate blobs, and a matcher
    pairs one of them and calls the other four false positives. The detector
    was correct and scores 20%.

    Area is cardinality-independent. Merging five marks into one, or splitting
    one into five, does not move these numbers at all -- only whether the same
    square metres of roof are covered.

      precision  of the area the detector flagged, how much is really equipment
      recall     of the area actually marked, how much the detector found

    Counts are still reported, but as context for reading the areas, never as
    the score.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    def clean(rings):
        out = []
        for r in rings:
            if not r or len(r) < 3:
                continue
            g = Polygon(r)
            if not g.is_valid:
                g = g.buffer(0)          # self-intersections drawn by hand
            if g.is_valid and g.area > 0:
                out.append(g)
        return out

    P, T = clean(pred_rings), clean(true_rings)
    if not P and not T:
        return None
    # dissolve first, so overlapping marks are not counted twice
    pu = unary_union(P) if P else None
    tu = unary_union(T) if T else None
    pa = pu.area if pu else 0.0
    ta = tu.area if tu else 0.0
    inter = pu.intersection(tu).area if (pu and tu) else 0.0
    union = pu.union(tu).area if (pu and tu) else (pa or ta)
    return {"precision": (inter / pa) if pa else 0.0,
            "recall": (inter / ta) if ta else 0.0,
            "iou": (inter / union) if union else 0.0,
            "found": len(P), "marked": len(T),
            "pred_area_m2": round(pa, 1), "true_area_m2": round(ta, 1),
            "overlap_m2": round(inter, 1)}


def predicted_obstruction_lines(obs_rings, footprint, edge_tol=0.35):
    """A carved obstruction's outline is a claim about roof structure, so its
    edges are predicted lines -- minus anything lying on the building outline,
    for the same reason facet boundaries drop those."""
    from shapely.geometry import Point
    boundary = footprint.exterior if hasattr(footprint, "exterior") else None
    out = []
    for ring in obs_rings or []:
        if not ring or len(ring) < 3:
            continue
        pts = list(ring)
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        for a, b in _segments(pts):
            if boundary is not None:
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                if boundary.distance(Point(mid)) < edge_tol:
                    continue
            out.append([list(a), list(b)])
    return out


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


def _line_points(l):
    """A drawn line as a polyline, from either shape the tool has written."""
    if l.get("points"):
        return l["points"]
    if l.get("a") and l.get("b"):
        return [l["a"], l["b"]]
    return None


def _polyline_ring(pts, w=0.4):
    """A drawn run has no area until it is given a width; expand the centreline
    to the band the tool displays and records."""
    import math
    h, left, right = w / 2.0, [], []
    for i, p in enumerate(pts):
        a = pts[max(0, i - 1)]
        b = pts[min(len(pts) - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1.0
        dx, dy = dx / L, dy / L
        left.append((p[0] - dy * h, p[1] + dx * h))
        right.append((p[0] + dy * h, p[1] - dx * h))
    return left + right[::-1]


def _obs_ring(o):
    """An obstruction as a closed ring, whatever it was drawn as."""
    if isinstance(o, list):          # already a ring
        return o
    if o.get("ring"):
        return o["ring"]
    if o.get("shape") == "polyline" and o.get("pts"):
        return _polyline_ring(o["pts"], o.get("width_m", 0.4))
    if o.get("shape") in ("triangle", "polygon") and o.get("pts"):
        return o["pts"]
    x, y, w, h = o.get("x"), o.get("y"), o.get("w"), o.get("h")
    if None in (x, y, w, h):
        return []
    if o.get("shape") == "ellipse":
        import math
        cx, cy = x + w / 2, y + h / 2
        return [(cx + math.cos(i / 28 * math.tau) * w / 2,
                 cy + math.sin(i / 28 * math.tau) * h / 2) for i in range(28)]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


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
    from src.obstruction_detection import detect_obstructions_combined
    from src.pointcloud_source import PointCloudSource

    ctxs, pc = {}, PointCloudSource()
    agg = defaultdict(list)
    total_roofs = len(ids)
    print(f"scoring the CURRENT segmenter against {len(ids)} drawn roofs "
          f"(tolerance {a.tol} m)\n")
    print(f"{'building':>10} {'lines P':>8}{'lines R':>8}{'F1':>7}"
          f"{'obs P':>7}{'obs R':>7}{'obs m2 f/m':>14}{'facets':>8}")

    for bid in ids:
        lab = labels[str(bid)]
        # A roof the labeller says is not a roof -- bare ground, a slab, a
        # wrong outline -- has no geometry worth scoring against. Including it
        # would measure the segmenter against something nobody claims exists.
        if lab.get("problem"):
            continue
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
        # The predicted side used to be hardcoded empty, so obstruction
        # precision and recall were always 0/0 -- every obstruction anyone drew
        # scored against nothing. Run the real detector, the same call
        # build_layout_geojson makes.
        pred_obs = []
        for f in facets:
            if f.get("plane_a") is None:
                continue
            try:
                found = detect_obstructions_combined(
                    ctx["img"], pc, f["geometry"],
                    (f["plane_a"], f["plane_b"], f["plane_c"]),
                    roof_geom=f.get("building_geometry"))
            except Exception as e:
                print(f"    obstruction detection failed on a facet: "
                      f"{type(e).__name__}: {e}")
                continue
            for o in (found or []):
                g = o["geometry"] if isinstance(o, dict) else o
                if getattr(g, "geom_type", "") == "Polygon":
                    pred_obs.append(list(g.exterior.coords))
                elif getattr(g, "geom_type", "") == "MultiPolygon":
                    pred_obs.extend(list(part.exterior.coords) for part in g.geoms)

        pred_lines = (predicted_lines_from_facets(facets, geom)
                      + predicted_obstruction_lines(pred_obs, geom))
        # The tool writes each line as {kind, a, b} and mirrors it into
        # "points" for consumers like this one; older files have only a/b.
        # Obstructions likewise carry a "ring" alongside their drawn form --
        # this used to read the raw dicts and would throw on the first real
        # batch of labels.
        true_lines = [_line_points(l) for l in lab.get("lines", [])
                      if l.get("kind") != "outline"]
        true_lines = [p for p in true_lines if p]
        # PARTIAL ANNOTATION. Marking is per-category: several roofs carry
        # obstructions and no lines, or the reverse. Scoring a category the
        # labeller did not touch is not a measurement -- with no drawn lines,
        # precision is 0 by construction because nothing can match, and folding
        # that into the mean drags the whole baseline down for roofs where
        # nobody claimed the segmenter was wrong. 26 Panorama Terrace and 5
        # Beach Street were both being reported as total failures on exactly
        # this basis. Score a category only where it was actually marked.
        ls = line_scores(pred_lines, true_lines, a.tol) if true_lines else None

        # RECALL BY KIND. Overall recall says half the drawn lines are missed
        # but not WHICH half, and the three kinds fail for different reasons: a
        # ridge is a fold the LiDAR sees plainly, a cliff is a height break that
        # should be even easier, and a valley on a shallow roof can be almost
        # invisible in a 1 m surface. Knowing which kind is being lost points at
        # a specific mechanism instead of "the segmenter is 54% right".
        for kind in ("ridge", "valley", "cliff"):
            kl = [_line_points(l) for l in lab.get("lines", [])
                  if l.get("kind") == kind]
            kl = [x for x in kl if x]
            if not kl:
                continue
            ks = line_scores(pred_lines, kl, a.tol)
            if ks:
                agg[f"rec_{kind}"].append(ks["recall"])
                agg[f"len_{kind}"].append(ks["true_segments"])
        marked_obs = [_obs_ring(o) for o in lab.get("obstructions", [])]
        marked_obs = [r for r in marked_obs if r and len(r) >= 3]
        os_ = obstruction_scores(pred_obs, marked_obs) if marked_obs else None

        lp = f"{ls['precision']:.0%}" if ls else "—"
        lr = f"{ls['recall']:.0%}" if ls else "—"
        f1 = f"{ls['f1']:.0%}" if ls else "—"
        op = f"{os_['precision']:.0%}" if os_ else "—"
        orr = f"{os_['recall']:.0%}" if os_ else "—"
        am = f"{os_['pred_area_m2']:.0f}/{os_['true_area_m2']:.0f}" if os_ else "—"
        print(f"{bid:>10} {lp:>8}{lr:>8}{f1:>7}{op:>7}{orr:>7}{am:>14}{len(facets):>8}")
        if ls:
            # PRECISION NEEDS A COMPLETE ROOF. Measured on Josh's first 42:
            # marking density correlates with precision at r = +0.394, and the
            # densely-marked half scores 63.0% against 43.2% for the sparse
            # half -- same segmenter, 20 points apart. A line the labeller did
            # not get to is indistinguishable from one the model invented, so
            # precision is only meaningful where they say they drew everything.
            # Recall is unaffected: it is measured against what WAS drawn.
            agg["lr"].append(ls["recall"])
            agg["f1"].append(ls["f1"])
            if lab.get("complete"):
                agg["lp"].append(ls["precision"])
        if os_:
            # weight by MARKED area: a roof with 200 m2 of plant should not
            # count the same as a shed with one vent
            agg["op"].append(os_["precision"]); agg["orr"].append(os_["recall"])
            agg["oa_pred"].append(os_["pred_area_m2"])
            agg["oa_true"].append(os_["true_area_m2"])
            agg["oa_over"].append(os_["overlap_m2"])

    if agg["f1"]:
        n = len(agg["f1"])
        n_complete = len(agg["lp"])
        skipped = total_roofs - n
        print(f"\nBASELINE over {n} roofs with drawn lines"
              + (f" ({skipped} more marked only obstructions)" if skipped else "")
              + " — the number a model has to beat:")
        print(f"  line RECALL {sum(agg['lr']) / n:.1%}   "
              f"(F1 {sum(agg['f1']) / n:.1%} against possibly-incomplete labels)")
        if n_complete:
            print(f"  line PRECISION {sum(agg['lp']) / n_complete:.1%}   "
                  f"over the {n_complete} roofs marked COMPLETE")
        else:
            print("  line precision: not reportable — no roof is marked complete.")
            print("    Precision counts model lines the labeller did not draw as")
            print("    errors, which is only fair where they drew everything.")
            print("    Tick 'I marked EVERY line' in the tool to enable it.")
        print("\nPrecision is 'lines we drew that are real'; recall is 'real lines")
        print("we found'. Over-segmentation shows as low precision, missed ridges")
        print("as low recall -- which is the distinction facet COUNT cannot make.")
    kinds = [k for k in ("ridge", "valley", "cliff") if agg[f"rec_{k}"]]
    if kinds:
        print("\n  RECALL BY LINE KIND — which drawn lines get missed:")
        for k in kinds:
            v = agg[f"rec_{k}"]
            print(f"    {k:<7} {sum(v) / len(v):>6.1%}   "
                  f"on {len(v)} roofs, {sum(agg[f'len_{k}'])} segments")

    if agg["op"]:
        m = len(agg["op"])
        pa, ta, ov = sum(agg["oa_pred"]), sum(agg["oa_true"]), sum(agg["oa_over"])
        print(f"\n  OBSTRUCTIONS over {m} roofs that HAVE marked obstructions, "
              f"measured by AREA "
              f"(counts would be misleading -- one drawn shape often covers "
              f"several objects):")
        print(f"    per-roof mean:  precision {sum(agg['op'])/m:.1%}   "
              f"recall {sum(agg['orr'])/m:.1%}")
        print(f"    area-weighted:  precision {ov/pa if pa else 0:.1%}   "
              f"recall {ov/ta if ta else 0:.1%}")
        print(f"    {pa:.0f} m2 flagged, {ta:.0f} m2 marked, {ov:.0f} m2 agreed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
