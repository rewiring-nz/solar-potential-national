"""
Sub-divide a roof face by how its surface TURNS, not by searching for cuts.

Why this exists
---------------
roof_partition cuts a roof one straight line at a time, keeping a cut when it
improves the plane fit enough to justify the panel area it costs. That works for
a ridge, which is one line that separates two large faces, and it structurally
cannot produce a hip.

A hip is several cuts forming a closed shape, and each one alone barely improves
anything. On 7 Anderson Heights the central hip sits inside a single 75 m2 face
with 11% of its points more than half a metre off plane; the first cut into it
gains 1.5% and costs 2.8 m2 of setback against 2.0 allowed, so it is refused and
the hip Josh drew stays paved over. Four different attempts to force it through
were measured against the four roofs he has marked up, and every one made the
whole worse:

    imagery-guided cuts   1 of 4 face counts exact (4 of 4 without)
    looser setback rule   3 of 4
    multi-cut lookahead   2 of 4
    merging coplanar      no effect -- blocked downstream, correctly

They fail for the same reason: they all still decide ONE LINE AT A TIME against
an economic test, and no sequence of individually-marginal cuts survives that.

So stop searching for cuts. Work out how many ways the surface turns inside a
face, from the points themselves, and then CONSTRUCT the boundaries: two planes
that meet do so along their exact intersection line. That is Josh's own
description of what a roof is -- "big flat planes... trim those planes by either
the edge of the building or another plane" -- applied inside a face rather than
across a whole building, which is where an earlier attempt at it went wrong.

Applied only where it is needed: a face whose plane already explains its points
is returned untouched, so this cannot disturb the roofs that are already right.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import split as shapely_split, unary_union

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Only faces that are actually failing get touched.
TRIGGER_INLIER = 0.90        # a face fitting better than this is left alone
TRIGGER_BAND_M = 0.15
TRIGGER_MIN_AREA_M2 = 20.0
TRIGGER_MIN_POINTS = 120

NORMAL_K = 14                # neighbours per local plane fit -- enough to average
# out LiDAR noise at ~0.42 m spacing without smoothing a hip crease away
NORMAL_MAX_RADIUS_M = 2.0
CLUSTER_ANGLE_DEG = 14.0     # normals closer than this are the same surface
MIN_CLUSTER_POINTS = 40
MIN_CLUSTER_SHARE = 0.10     # of the face's points; below this it is noise
MAX_SUBFACES = 5             # a hip is 3-4; more than this is not a feature
MIN_SUBFACE_M2 = 6.0


def _local_normals(pts, k=NORMAL_K):
    """Unit normal at each point, from a plane through its k nearest neighbours."""
    tree = cKDTree(pts[:, :2])
    d, idx = tree.query(pts[:, :2], k=min(k, len(pts)))
    normals = np.zeros((len(pts), 3))
    for i in range(len(pts)):
        nb = pts[idx[i][d[i] <= NORMAL_MAX_RADIUS_M]]
        if len(nb) < 4:
            normals[i] = (0.0, 0.0, 1.0)
            continue
        c = nb - nb.mean(axis=0)
        # smallest singular vector is the surface normal
        try:
            n = np.linalg.svd(c, full_matrices=False)[2][-1]
        except np.linalg.LinAlgError:
            n = np.array([0.0, 0.0, 1.0])
        if n[2] < 0:
            n = -n
        normals[i] = n / (np.linalg.norm(n) or 1.0)
    return normals


def _cluster_by_normal(normals, min_points):
    """Greedy grouping of normals by direction, biggest group first."""
    remaining = np.ones(len(normals), bool)
    groups = []
    cos_tol = np.cos(np.radians(CLUSTER_ANGLE_DEG))
    while remaining.sum() >= min_points and len(groups) < MAX_SUBFACES:
        idx = np.flatnonzero(remaining)
        # seed on the densest direction: the normal most agreed with
        sim = normals[idx] @ normals[idx].T
        seed = idx[int(np.argmax((sim > cos_tol).sum(axis=1)))]
        member = remaining & ((normals @ normals[seed]) > cos_tol)
        if member.sum() < min_points:
            break
        groups.append(member)
        remaining &= ~member
    return groups


def _fit(pts):
    x0, y0 = pts[:, 0].mean(), pts[:, 1].mean()
    A = np.column_stack([pts[:, 0] - x0, pts[:, 1] - y0, np.ones(len(pts))])
    c, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    a, b = float(c[0]), float(c[1])
    return a, b, float(c[2]) - a * x0 - b * y0


def _inlier(pts, plane, band=TRIGGER_BAND_M):
    r = pts[:, 2] - (plane[0] * pts[:, 0] + plane[1] * pts[:, 1] + plane[2])
    return float((np.abs(r - np.median(r)) < band).mean())


def _cut_on_planes(poly, planes):
    """Cut a face along the exact intersection lines of its own sub-planes."""
    cx, cy = poly.centroid.x, poly.centroid.y
    cells = [poly]
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            pa, pb = planes[i], planes[j]
            A, B = pa[0] - pb[0], pa[1] - pb[1]
            n = float(np.hypot(A, B))
            if n < 1e-3:
                continue                     # parallel: a step, not a fold
            C = ((pa[0] * cx + pa[1] * cy + pa[2]) - (pb[0] * cx + pb[1] * cy + pb[2]))
            d = np.array([-B, A]) / n
            pt0 = np.array([cx, cy]) - np.array([A, B]) * (C / (n ** 2))
            span = max(poly.bounds[2] - poly.bounds[0],
                       poly.bounds[3] - poly.bounds[1]) * 2 + 10
            line = LineString([pt0 - d * span, pt0 + d * span])
            nxt = []
            for c in cells:
                try:
                    parts = [q for q in shapely_split(c, line).geoms
                             if isinstance(q, Polygon) and q.area >= 1.0]
                except Exception:
                    parts = []
                nxt.extend(parts if len(parts) >= 2 else [c])
            cells = nxt
            if len(cells) > 60:
                return cells
    return cells


# A compact feature has to be cut out as a REGION, not sliced with a line.
#
# This is what the line-cutting approach structurally cannot do, and it is why
# four separate attempts at 7 Anderson Heights' central feature all failed. A
# line through a face yields two half-planes; a raised hip, a dormer or a plant
# housing is an enclosed shape in the middle of one. Cutting the best available
# line there isolates a 7.4 m2 sliver of the feature's edge and leaves the fit
# unchanged at 84%, because the 103 deviating points are a tenth of the face's
# area and an area-weighted score cannot see them.
#
# What the local normals DO see is unambiguous: inside that face, 671 points lie
# at 24.8 degrees and 103 sit at 74.3 degrees. Near-vertical inside a roof face
# is the side of something standing on it. Take those points, take the ground
# they cover, and lift it out as its own region.
FEATURE_MIN_AREA_M2 = 2.0
FEATURE_MAX_AREA_SHARE = 0.45   # more than this is not a feature ON the roof
FEATURE_BUFFER_M = 0.45         # close the gaps between returns off one object
FEATURE_MIN_COMPACTNESS = 0.25  # area / (its own bounding box) -- a strip hugging
# an edge is a plane-fit artefact, not an object


def extract_features(face_poly, pts, plane):
    """Compact regions inside a face whose surface points elsewhere.

    Returns [(polygon, plane), ...] for the feature regions only -- the caller
    subtracts them from the face. [] when the face carries no such region."""
    if face_poly.is_empty or face_poly.area < TRIGGER_MIN_AREA_M2:
        return []
    inside = pts[shapely.contains_xy(face_poly, pts[:, 0], pts[:, 1])] \
        if len(pts) else pts
    if len(inside) < TRIGGER_MIN_POINTS:
        return []
    if _inlier(inside, plane) >= TRIGGER_INLIER:
        return []

    normals = _local_normals(inside)
    groups = _cluster_by_normal(normals, max(MIN_CLUSTER_POINTS,
                                             int(MIN_CLUSTER_SHARE * len(inside))))
    if len(groups) < 2:
        return []
    main = max(groups, key=lambda g: g.sum())
    main_plane = _fit(inside[main])

    out = []
    for g in groups:
        if g is main or g.sum() < 8:
            continue
        pl = _fit(inside[g])
        if _plane_angle(main_plane, pl) < CLUSTER_ANGLE_DEG:
            continue                       # same surface, just noisy
        # The ENCLOSED region, not the sloping band itself. A raised or recessed
        # feature is bounded by steep surface; the thing to remove is what those
        # sides surround. On 7 Anderson Heights the steep points alone come to
        # 4.2 m2 -- just the sides -- while what they enclose is 12.2 m2 sitting
        # 0.77 m BELOW the roof with 2.79 m of height spread. That is the
        # recessed valley Josh drew in the middle of that roof, and panels were
        # being laid across it.
        pos = inside[g][:, :2]
        # Hull the steep points TOGETHER. Hulling each buffered blob separately
        # only recovers the sides again -- on 7 Anderson that gives 3.3 m2 where
        # the sides collectively enclose 12.2 m2. The sides of one feature are
        # several disconnected arcs; what matters is what they surround.
        blobs = unary_union([Point(x, y).buffer(FEATURE_BUFFER_M) for x, y in pos])
        clusters = list(blobs.geoms) if blobs.geom_type == "MultiPolygon" else [blobs]
        clusters = [c for c in clusters if c.area >= FEATURE_MIN_AREA_M2 * 0.5]
        if not clusters:
            continue
        whole = unary_union(clusters)
        hull = MultiPoint([(x, y) for x, y in pos
                           if whole.contains(Point(x, y))]).convex_hull
        region = (hull if hull.geom_type == "Polygon" else whole).intersection(face_poly)
        for q in (region.geoms if region.geom_type == "MultiPolygon" else [region]):
            if q.is_empty or q.geom_type != "Polygon":
                continue
            if not (FEATURE_MIN_AREA_M2 <= q.area <= FEATURE_MAX_AREA_SHARE * face_poly.area):
                continue
            mrr = q.minimum_rotated_rectangle
            if mrr.is_empty or q.area / mrr.area < FEATURE_MIN_COMPACTNESS:
                continue                   # a long thin strip is not an object
            sub = inside[shapely.contains_xy(q, inside[:, 0], inside[:, 1])]
            out.append((q, _fit(sub) if len(sub) >= 8 else pl))
    return out


def _plane_angle(pa, pb):
    na = np.array([-pa[0], -pa[1], 1.0]); na /= np.linalg.norm(na)
    nb = np.array([-pb[0], -pb[1], 1.0]); nb /= np.linalg.norm(nb)
    return float(np.degrees(np.arccos(np.clip(abs(na @ nb), -1.0, 1.0))))


def subdivide_face(face_poly, pts, plane):
    """Split one face into its real sub-planes, or return None to leave it alone.

    Returns [(polygon, plane), ...] or None."""
    if face_poly.is_empty or face_poly.area < TRIGGER_MIN_AREA_M2:
        return None
    inside = pts[shapely.contains_xy(face_poly, pts[:, 0], pts[:, 1])] \
        if len(pts) else pts
    if len(inside) < TRIGGER_MIN_POINTS:
        return None
    if _inlier(inside, plane) >= TRIGGER_INLIER:
        return None                          # the face already is one plane

    normals = _local_normals(inside)
    groups = _cluster_by_normal(normals, max(MIN_CLUSTER_POINTS,
                                             int(MIN_CLUSTER_SHARE * len(inside))))
    if len(groups) < 2:
        return None                          # turns only one way: not a feature

    planes = [_fit(inside[g]) for g in groups if g.sum() >= 8]
    if len(planes) < 2:
        return None

    cells = _cut_on_planes(face_poly, planes)
    if len(cells) < 2:
        return None

    # Each cell goes to whichever sub-plane its own points sit on.
    labelled = {}
    for cell in cells:
        sub = inside[shapely.contains_xy(cell, inside[:, 0], inside[:, 1])]
        if len(sub) < 6:
            best = min(range(len(planes)),
                       key=lambda k: abs(planes[k][0] * cell.centroid.x
                                         + planes[k][1] * cell.centroid.y
                                         + planes[k][2]))
        else:
            best = int(np.argmin([np.median(np.abs(
                sub[:, 2] - (p[0] * sub[:, 0] + p[1] * sub[:, 1] + p[2]))) for p in planes]))
        labelled.setdefault(best, []).append(cell)

    out = []
    for k, polys in labelled.items():
        merged = unary_union(polys)
        for q in (merged.geoms if merged.geom_type == "MultiPolygon" else [merged]):
            if q.area < MIN_SUBFACE_M2:
                continue
            sub = inside[shapely.contains_xy(q, inside[:, 0], inside[:, 1])]
            out.append((Polygon(q.exterior, [r for r in q.interiors]),
                        _fit(sub) if len(sub) >= 8 else planes[k]))
    if len(out) < 2:
        return None

    # Only worth it if the sub-planes genuinely explain the face better.
    before = _inlier(inside, plane)
    tot = num = 0.0
    for q, pl in out:
        sub = inside[shapely.contains_xy(q, inside[:, 0], inside[:, 1])]
        if len(sub) < 8:
            continue
        num += _inlier(sub, pl) * q.area
        tot += q.area
    after = (num / tot) if tot else 0.0
    return out if after > before + 0.05 else None
