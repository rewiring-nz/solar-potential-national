"""
Reconstruct a roof as a set of planes joined along real edges, instead of
tracing each facet independently out of a raster.

Why (measured on the pilot, 26 Aug): panels escaping their facet -- 0. Panels
crossing the building outline -- 0. Panels on a drawn obstruction -- 1 in
60,562. But 7.2% of panels sit more than 0.35m off the plane they were placed
on, and on the worst buildings that is 60-92% of the roof. Panel fitting is
not what is wrong. The roof model underneath it is: a barrel-vaulted building
comes through as ONE flat facet, a stepped terrace as planes spanning the
steps between levels.

The approach here builds the roof the way a roof is actually made:

1. Planes, from the point cloud rather than the DSM raster. Iterative RANSAC:
   take the best-supported plane, remove its inliers, repeat. Sawtooth bays
   separate naturally -- same normal, different offset is a different plane
   equation. A curved roof comes out as a fan of narrow strips, which is the
   correct piecewise-planar answer rather than one wrong plane.

2. Edges, analytically. Two planes that meet do so along their intersection
   line -- that IS the ridge/hip/valley, exact, straight, and shared by both
   faces. Near-parallel neighbours don't intersect; they are a step in height,
   so that boundary is fitted to where the point support actually changes.
   Outer edges come from the LINZ outline, which is surveyed and already
   straight. No boundary comes from tracing pixels.

3. Facets, by arrangement. Every edge line is cut against the outline and the
   whole set polygonized, so the roof is partitioned into cells whose borders
   are straight by construction and exactly shared between neighbours. Each
   cell goes to whichever plane the points inside it actually support, and
   cells with the same winner are merged.

Emits the same facet dicts as roof_segmentation (building_id, plane_a/b/c,
slope_deg, aspect_deg, area_m2, point_count, geometry) so it can be dropped
in behind a flag once it is proven.

Prototype: NOT wired into the pipeline, and it should not be re-enabled in its
current form. Josh reviewed ten before/after layouts on 26 Aug and called the
reconstruction worse on all ten. See src/compare_reconstruct.py and
src/compare_layouts.py.

WHY IT FAILED, and the design rule any next attempt has to obey:

Josh: "They need to be large and blocky most of the time like real rooftops.
It's a lot more common for rooftops to be clear large flat surfaces on a few
different angles and slopes, than it is to have lots of small changes." And:
"you've massively overcomplicated panel placement, I think it's because you are
drawing lots of tiny facet outlines and you might have a rule a panel can't
overlap a facet outline."

Both correct. panel_fitting does exactly that -- surface_ridge =
surface_poly.buffer(-RIDGE_SETBACK_M) -- so every facet is eroded by the ridge
setback and panels must fit inside the result. Fragmenting a roof therefore
costs area N times over, and no panel can span two facets:

     6 m2 facet ->  57% of it usable
    25 m2 facet ->  77%
   150 m2 facet ->  90%
   400 m2 facet ->  94%

So this module optimising for "planes that fit the points" is optimising the
wrong thing. A roof is better modelled as FEW LARGE faces even where the points
would support splitting one -- the split has to earn back the setback area it
costs, and small ones never do. The merge thresholds here (MERGE_SLOPE_DEG,
SPLIT_MIN_*) are far too eager to divide.

The plane counts also said this was fine: it scored level with the shipped
segmenter on Josh's 20 labelled roofs. Neither plane count nor off-plane
residual can see the cost of a split. Judge any future version on layouts.
"""

import math
import sys
from pathlib import Path

import numpy as np
import shapely.vectorized
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, split, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RANSAC_TOL_M = 0.15        # a point this close to a plane is on it
RANSAC_ITERS = 400
MIN_PLANE_PTS = 25         # ~4.5 m2 at pilot density (5.7 pts/m2)
MAX_PLANES = 40
WALL_SLOPE_DEG = 72        # steeper than this is a wall, not a roof face (== config.MAX_ROOF_SLOPE_DEG)
GRID_M = 0.30              # label raster step, for adjacency and step edges
PARALLEL_GRAD_TOL = 0.06   # gradient difference below this can only be a step
# Two planes always intersect somewhere unless they are parallel -- but that
# line is only the ROOF's edge if the two faces actually fold together there.
# 4750866 is a house with a wing 4m lower: opposite aspects, so the gradients
# differ and the analytic line was trusted, but it falls 1.7m OUTSIDE the
# building. Nothing got cut and the whole roof came back as one facet fitting
# 13% of its own points. So the analytic line is only used when it lands on the
# boundary the points actually show; otherwise that boundary is fitted.
FOLD_MAX_OFFSET_M = 1.0
MIN_FACET_M2 = 6.0         # smaller than ~3 panels is not a face worth racking
MIN_CELL_PTS = 4           # a cell with fewer points has no say in its own label
# Coplanar merge. Splitting a roof into more planes ALWAYS lowers residual, so
# without this the reconstruction scores well by shattering: 93 Beach St came
# out as 82 speckled fragments at "5% off-plane". Two neighbouring cells that
# describe the same physical surface have to end up as one facet.
# 4 degrees, not 6: the barrel vault's adjacent strips are ~6 apart and must
# stay separate, while a flat deck's sag shows up as 1-3 and must not.
MERGE_SLOPE_DEG = 4.0
MERGE_ASPECT_DEG = 14.0
MERGE_FLAT_DEG = 6.0       # below this slope, aspect is meaningless
MERGE_STEP_M = 0.25        # ...and they must not be parallel faces at different heights
# The angle tolerances above only decide what is WORTH trying to merge. What
# decides it is whether one plane still fits both sets of points: on the barrel
# vault, adjacent strips differ by ~5 degrees and merging on angle alone put
# the curve back under a single wrong plane (49% -> 33% off-plane, still bad).
# A merge has to keep this share of the combined points on the plane.
# Was 0.90 and was the PRIMARY gate, which is why 5 Isle St stayed split: a
# real roof deck sags 0.2m across its span, so one honest plane over it never
# reaches 90% of points within 0.15m. The evidence test above decides now --
# pitch, bearing, or a step -- and this only catches gross mistakes.
MERGE_MIN_INLIER_FRAC = 0.35
# A plane and a polygon are not the same thing. One flat deck interrupted by a
# lift overrun comes back as three polygons, and merge_coplanar cannot fuse
# them because it only considers facets that TOUCH. 1 Memorial St returned ten
# facets against Josh's count of seven, and three of those ten were the same
# deck at 329.25/329.25/329.26 m. So facets carry a plane_id: same surface,
# possibly several pieces. Layout still works per polygon; counting works per
# plane.
SAME_PLANE_SLOPE_DEG = 3.0
SAME_PLANE_ASPECT_DEG = 12.0
SAME_PLANE_STEP_M = 0.20
SMOOTH_ROUNDS = 3
# A cell is only allowed to belong to one plane if its own points mostly agree.
# 111 Hallenstein came out of the arrangement with a 109 m2 cell straddling the
# ridge, votes split 245/213/319 between three faces -- winner-takes-all handed
# the lot to one of them and the refit plane was meaningless (32% off-plane on
# a roof the shipped model got to 15%). Where the line set is incomplete, the
# cell gets cut by the two planes competing for it.
MIXED_MAX_SHARE = 0.75
# Cutting a roof up always lowers the residual, so residual alone cannot decide
# whether to cut. Josh, on 5 Isle St: "this one is better as one plane, not two
# with one diagonal". That roof is one large flat surface with drainage falls;
# two near-identical planes were separated by a line corresponding to nothing.
# A split has to be justified by a real difference -- a change of pitch, of
# bearing, or a step in height -- not by the arithmetic improving.
SPLIT_MIN_SLOPE_DEG = 5.0
SPLIT_MIN_ASPECT_DEG = 20.0
SPLIT_MIN_STEP_M = 0.40
REFINE_ROUNDS = 4
# Things ON the roof, as distinct from faces OF it. Without this a big AC unit
# or a lift overrun becomes a "facet" -- which is most of why 93 Beach St still
# came out as 28 pieces after the merge. A plant deck is small, sits above the
# face it is embedded in, and is surrounded by it.
OBST_MAX_AREA_M2 = 30.0      # larger than this is another roof level, not plant
OBST_MIN_HEIGHT_M = 0.25     # above the parent plane
# 0.35, from a clean sweep against the 18 labelled roofs: total plane error 24,
# against 26 at 0.55 and 27 at 0.20. Equipment sitting at the junction of two
# faces shares its border with both, so demanding most of it touch ONE parent
# refused the cases that matter -- but dropping the bar to 0.20 over-fires and
# is worse again.
#
# An earlier sweep reported 0.20 as best at 23. That sweep was WRONG: its
# except/continue skipped buildings that crashed on the MultiPolygon bug fixed
# below, and a skipped building contributed zero error, so whichever settings
# crashed most looked best. Sweeps here now charge a crash as a failure.
OBST_ENCLOSURE = 0.35        # share of its border shared with one parent face
OBST_CLUSTER_EPS_M = 0.60    # ~1.4x the 0.42m point spacing at pilot density
OBST_MIN_PTS = 5
OBST_MIN_AREA_M2 = 0.35      # smaller than this is noise, not equipment
# LiDAR draws a box bigger than it is. Josh, on 28 Rees St: obstructions "show
# up wider in the lidar than they really are, because they often have vertical
# edges but don't necessarily show as vertical edges in the lidar". A return
# near a vertical face lands at an intermediate height, so the above-plane
# cluster is dilated by roughly half the point spacing in every direction --
# 0.21 m at the pilot's 5.7 pts/m2. Un-dilating it is the difference between a
# clean array around a duct and a hole punched through one.
OBST_EDGE_SMEAR_M = 0.21
# Two bounds on how much of a roof may be declared "things on the roof".
# Without them a 161 m2 house with 10 small faces had 9 of them reclassified as
# plant and came back covering 4% of its own outline.
OBST_PARENT_RATIO = 2.5      # the face beneath must be this much bigger
OBST_MAX_ROOF_SHARE = 0.25   # obstructions may not claim more roof than this


def fit_plane(pts):
    """Least squares z = a*x + b*y + c, centred for conditioning."""
    x0, y0 = pts[:, 0].mean(), pts[:, 1].mean()
    A = np.column_stack([pts[:, 0] - x0, pts[:, 1] - y0, np.ones(len(pts))])
    coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    a, b, c0 = coef
    return np.array([a, b, c0 - a * x0 - b * y0])


def residuals(plane, pts):
    a, b, c = plane
    return a * pts[:, 0] + b * pts[:, 1] + c - pts[:, 2]


def plane_slope_aspect(plane):
    a, b, _ = plane
    slope = math.degrees(math.atan(math.hypot(a, b)))
    aspect = math.degrees(math.atan2(-a, -b)) % 360
    return slope, aspect


def ransac_planes(pts, rng):
    """Strongest plane first, remove its inliers, repeat. Returns planes and
    the index of the plane each point was claimed by (-1 = unclaimed)."""
    n = len(pts)
    owner = np.full(n, -1)
    live = np.arange(n)
    planes = []
    while len(live) >= MIN_PLANE_PTS and len(planes) < MAX_PLANES:
        sub = pts[live]
        best_inl, best_plane = None, None
        for _ in range(RANSAC_ITERS):
            idx = rng.choice(len(sub), 3, replace=False)
            tri = sub[idx]
            v1, v2 = tri[1] - tri[0], tri[2] - tri[0]
            nx, ny, nz = np.cross(v1, v2)
            if abs(nz) < 1e-6:          # vertical sample triple -- no z = f(x,y)
                continue
            plane = np.array([-nx / nz, -ny / nz,
                              (nx * tri[0, 0] + ny * tri[0, 1] + nz * tri[0, 2]) / nz])
            inl = np.abs(residuals(plane, sub)) < RANSAC_TOL_M
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl, best_plane = inl, plane
        if best_inl is None or best_inl.sum() < MIN_PLANE_PTS:
            break
        # Refit on the consensus set, then re-select: the 3-point seed plane is
        # noisy, and one refit typically pulls in another 10-20% of the face.
        plane = fit_plane(sub[best_inl])
        inl = np.abs(residuals(plane, sub)) < RANSAC_TOL_M
        if inl.sum() < MIN_PLANE_PTS:
            break
        plane = fit_plane(sub[inl])
        planes.append(plane)
        owner[live[inl]] = len(planes) - 1
        live = live[~inl]
    return planes, owner


def label_raster(outline, pts, owner, planes):
    """Nearest claimed point wins each grid cell -- only used to find which
    planes are neighbours and where the steps are, never to make a boundary."""
    minx, miny, maxx, maxy = outline.bounds
    xs = np.arange(minx, maxx + GRID_M, GRID_M)
    ys = np.arange(miny, maxy + GRID_M, GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    inside = shapely.vectorized.contains(outline, gx, gy)
    claimed = owner >= 0
    if claimed.sum() == 0:
        return None, None, None
    tree = cKDTree(pts[claimed][:, :2])
    _, nn = tree.query(np.column_stack([gx[inside], gy[inside]]))
    lab = np.full(gx.shape, -1)
    lab[inside] = owner[claimed][nn]
    return lab, gx, gy


def adjacent_pairs(lab):
    """Label pairs that touch, so only real neighbours contribute an edge --
    every plane pair would be O(n^2) lines and shatter the arrangement."""
    pairs = set()
    for A, B in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        d = (A != B) & (A >= 0) & (B >= 0)
        for i, j in zip(A[d], B[d]):
            pairs.add((min(i, j), max(i, j)))
    return pairs


def _line_from_coeffs(A, B, C, bounds):
    """A*x + B*y + C = 0 as a segment long enough to cross the building."""
    minx, miny, maxx, maxy = bounds
    norm = math.hypot(A, B)
    if norm < 1e-12:
        return None
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) + 20.0
    t = -(A * cx + B * cy + C) / (norm * norm)
    px, py = cx + A * t, cy + B * t
    dx, dy = -B / norm, A / norm
    return LineString([(px - dx * half, py - dy * half), (px + dx * half, py + dy * half)])


def _fit_line_tls(P):
    """Total least squares through 2D points -> A*x + B*y + C = 0."""
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    dx, dy = Vt[0]
    A, B = -dy, dx
    return A, B, -(A * c[0] + B * c[1])


def _boundary_points(lab, gx, gy, i, j):
    """Grid midpoints where label i meets label j."""
    bnd = []
    for Aa, Bb, axis in ((lab[:, :-1], lab[:, 1:], 0), (lab[:-1, :], lab[1:, :], 1)):
        d = ((Aa == i) & (Bb == j)) | ((Aa == j) & (Bb == i))
        if not d.any():
            continue
        yy, xx = np.nonzero(d)
        if axis == 0:
            bnd.append(np.column_stack([gx[yy, xx] + GRID_M / 2, gy[yy, xx]]))
        else:
            bnd.append(np.column_stack([gx[yy, xx], gy[yy, xx] + GRID_M / 2]))
    return np.vstack(bnd) if bnd else np.empty((0, 2))


def edge_lines(planes, pairs, lab, gx, gy, bounds):
    """The edge between two faces. Where they fold together -- a ridge, hip or
    valley -- that is their intersection line, exact and straight. Where one
    simply stands above the other, they never meet, so the edge is fitted to
    where the point support actually changes."""
    lines = []
    for i, j in pairs:
        ai, bi, ci = planes[i]
        aj, bj, cj = planes[j]
        A, B, C = ai - aj, bi - bj, ci - cj
        P = _boundary_points(lab, gx, gy, i, j)
        if len(P) and not genuinely_different(planes[i], planes[j],
                                              type("p", (), {"x": P[:, 0].mean(),
                                                             "y": P[:, 1].mean()})()):
            continue     # no real edge between them; merge_coplanar will fuse them
        analytic = _line_from_coeffs(A, B, C, bounds) if math.hypot(A, B) >= PARALLEL_GRAD_TOL else None
        if analytic is not None and len(P) >= 3:
            # Distance from each observed boundary point to the analytic line.
            dist = np.abs(A * P[:, 0] + B * P[:, 1] + C) / math.hypot(A, B)
            if float(np.median(dist)) <= FOLD_MAX_OFFSET_M:
                lines.append(analytic)      # a real fold: prefer the exact line
                continue
        if len(P) >= 3:
            ln = _line_from_coeffs(*_fit_line_tls(P), bounds)
            if ln is not None:
                lines.append(ln)
        elif analytic is not None:
            lines.append(analytic)
    return lines


def _circ_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _same_surface(f, g):
    """Do these two facets describe one physical plane, or two?"""
    if abs(f["slope_deg"] - g["slope_deg"]) > MERGE_SLOPE_DEG:
        return False
    both_flat = f["slope_deg"] < MERGE_FLAT_DEG and g["slope_deg"] < MERGE_FLAT_DEG
    if not both_flat and _circ_diff(f["aspect_deg"], g["aspect_deg"]) > MERGE_ASPECT_DEG:
        return False
    # Same tilt and bearing still leaves the sawtooth case: parallel faces one
    # bay apart in height. Compare the two planes where they actually meet.
    shared = f["geometry"].intersection(g["geometry"].buffer(0.05))
    if shared.is_empty:
        return False
    c = shared.centroid
    zf = f["plane_a"] * c.x + f["plane_b"] * c.y + f["plane_c"]
    zg = g["plane_a"] * c.x + g["plane_b"] * c.y + g["plane_c"]
    return abs(zf - zg) <= MERGE_STEP_M


def merge_coplanar(facets, pts):
    """Repeatedly fuse touching facets that are the same surface."""
    changed = True
    while changed and len(facets) > 1:
        changed = False
        for i in range(len(facets)):
            for j in range(i + 1, len(facets)):
                f, g = facets[i], facets[j]
                if not f["geometry"].buffer(0.05).intersects(g["geometry"]):
                    continue
                if not _same_surface(f, g):
                    continue
                geom = unary_union([f["geometry"], g["geometry"]]).buffer(0.02).buffer(-0.02)
                if geom.geom_type != "Polygon":
                    continue
                inside = pts[shapely.vectorized.contains(geom, pts[:, 0], pts[:, 1])]
                if len(inside) < 8:
                    continue
                plane = fit_plane(inside)
                # Does one plane still describe both faces? If not, they are
                # genuinely different surfaces however similar their angles.
                if (np.abs(residuals(plane, inside)) < RANSAC_TOL_M).mean() < MERGE_MIN_INLIER_FRAC:
                    continue
                slope, aspect = plane_slope_aspect(plane)
                facets[i] = {**f, "geometry": geom, "area_m2": float(geom.area),
                             "plane_a": float(plane[0]), "plane_b": float(plane[1]),
                             "plane_c": float(plane[2]), "slope_deg": float(slope),
                             "aspect_deg": float(aspect), "point_count": int(len(inside))}
                facets.pop(j)
                changed = True
                break
            if changed:
                break
    return facets


def genuinely_different(p1, p2, where):
    """Are these two planes different SURFACES, or one surface twice? Judged at
    `where` (a point or geometry), because two planes that fold together are
    identical at the fold and only diverge away from it."""
    s1, a1 = plane_slope_aspect(p1)
    s2, a2 = plane_slope_aspect(p2)
    if abs(s1 - s2) > SPLIT_MIN_SLOPE_DEG:
        return True
    both_flat = s1 < MERGE_FLAT_DEG and s2 < MERGE_FLAT_DEG
    if not both_flat and _circ_diff(a1, a2) > SPLIT_MIN_ASPECT_DEG:
        return True
    c = where.centroid if hasattr(where, "centroid") else where
    dz = abs((p1[0] * c.x + p1[1] * c.y + p1[2]) - (p2[0] * c.x + p2[1] * c.y + p2[2]))
    return dz > SPLIT_MIN_STEP_M


def _cell_votes(cell, planes, pts):
    inside = pts[shapely.vectorized.contains(cell, pts[:, 0], pts[:, 1])]
    if len(inside) < MIN_CELL_PTS:
        return inside, []
    return inside, [int((np.abs(residuals(p, inside)) < RANSAC_TOL_M).sum()) for p in planes]


def refine_mixed_cells(cells, planes, pts, bounds):
    """Split any cell whose own points disagree about which plane it is, along
    the intersection of the two planes competing for it. Each split strictly
    reduces the mixing, so this terminates."""
    for _ in range(REFINE_ROUNDS):
        out, split_any = [], False
        for cell in cells:
            inside, votes = _cell_votes(cell, planes, pts)
            if not votes or sum(votes) == 0:
                out.append(cell)
                continue
            order = np.argsort(votes)[::-1]
            top, second = int(order[0]), int(order[1]) if len(order) > 1 else None
            if second is None or votes[second] < MIN_CELL_PTS \
                    or votes[top] / max(sum(votes), 1) >= MIXED_MAX_SHARE:
                out.append(cell)
                continue
            if not genuinely_different(planes[top], planes[second], cell):
                out.append(cell)     # one surface fitted twice -- leave it whole
                continue
            a1, b1, c1 = planes[top]
            a2, b2, c2 = planes[second]
            ln = _line_from_coeffs(a1 - a2, b1 - b2, c1 - c2, bounds)
            if ln is None or not ln.intersects(cell):
                # They do not fold together inside this cell, so split on where
                # their points part company: the perpendicular bisector of the
                # two groups' centroids.
                g1 = inside[np.abs(residuals(planes[top], inside)) < RANSAC_TOL_M][:, :2]
                g2 = inside[np.abs(residuals(planes[second], inside)) < RANSAC_TOL_M][:, :2]
                if len(g1) < 3 or len(g2) < 3:
                    out.append(cell)
                    continue
                m1, m2 = g1.mean(axis=0), g2.mean(axis=0)
                d = m2 - m1
                if float(np.hypot(*d)) < 1e-6:
                    out.append(cell)
                    continue
                mid = (m1 + m2) / 2
                ln = _line_from_coeffs(d[0], d[1], -(d[0] * mid[0] + d[1] * mid[1]), bounds)
            if ln is None:
                out.append(cell)
                continue
            try:
                pieces = [g for g in split(cell, ln).geoms if g.area > 0.5]
            except Exception:
                pieces = []
            if len(pieces) < 2:
                out.append(cell)
                continue
            out.extend(pieces)
            split_any = True
        cells = out
        if not split_any:
            break
    return cells


def _cluster_xy(P, eps, min_pts):
    """Connected components of a radius graph. Stands in for DBSCAN, which
    would mean adding scikit-learn for one call."""
    if len(P) == 0:
        return np.empty(0, int)
    tree = cKDTree(P)
    pairs = np.array(list(tree.query_pairs(eps)), dtype=int)
    if len(pairs) == 0:
        return np.full(len(P), -1)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    n, lab = connected_components(g, directed=False)
    counts = np.bincount(lab, minlength=n)
    return np.where(counts[lab] >= min_pts, lab, -1)


def _box_from_points(P, base_plane, pts3):
    """Oriented footprint + how far it stands off the face beneath it."""
    from shapely.geometry import MultiPoint
    hull = MultiPoint([tuple(q) for q in P]).convex_hull
    if hull.geom_type != "Polygon":
        hull = hull.buffer(0.15)
    if hull.is_empty or hull.area < OBST_MIN_AREA_M2:
        return None
    # A rotated rectangle, not the ragged hull: equipment is boxes, and a clean
    # rectangle is also what panel fitting can set back from cleanly.
    rect = hull.minimum_rotated_rectangle
    # Shrink back the edge smear. Keep the dilated box only if eroding would
    # erase it entirely -- a real 0.5 m vent is better modelled slightly too
    # large than not at all.
    shrunk = rect.buffer(-OBST_EDGE_SMEAR_M)
    if shrunk.geom_type == "Polygon" and not shrunk.is_empty \
            and shrunk.area >= OBST_MIN_AREA_M2:
        rect = shrunk.minimum_rotated_rectangle
    height = float(np.max(-residuals(base_plane, pts3))) if len(pts3) else 0.0
    return {"geometry": rect, "height_m": round(height, 2),
            "area_m2": float(rect.area), "point_count": int(len(P))}


def assign_plane_ids(facets):
    """Group facets that describe ONE physical surface, whether or not their
    polygons touch. Compared where they actually are -- midway between the two
    centroids -- because two planes that differ slightly in tilt diverge with
    distance and would otherwise never look equal."""
    ids = [None] * len(facets)
    nxt = 0
    for i, f in enumerate(facets):
        if ids[i] is not None:
            continue
        ids[i] = nxt
        for j in range(i + 1, len(facets)):
            if ids[j] is not None:
                continue
            g = facets[j]
            if abs(f["slope_deg"] - g["slope_deg"]) > SAME_PLANE_SLOPE_DEG:
                continue
            both_flat = (f["slope_deg"] < MERGE_FLAT_DEG
                         and g["slope_deg"] < MERGE_FLAT_DEG)
            if not both_flat and _circ_diff(f["aspect_deg"], g["aspect_deg"]) > SAME_PLANE_ASPECT_DEG:
                continue
            c1, c2 = f["geometry"].centroid, g["geometry"].centroid
            mx, my = (c1.x + c2.x) / 2, (c1.y + c2.y) / 2
            z1 = f["plane_a"] * mx + f["plane_b"] * my + f["plane_c"]
            z2 = g["plane_a"] * mx + g["plane_b"] * my + g["plane_c"]
            if abs(z1 - z2) <= SAME_PLANE_STEP_M:
                ids[j] = nxt
        nxt += 1
    for f, pid in zip(facets, ids):
        f["plane_id"] = int(pid)
    return facets


def n_planes(facets):
    """How many distinct surfaces the roof has -- the thing a label counts."""
    return len({f.get("plane_id", i) for i, f in enumerate(facets)})


def extract_obstructions(facets, pts):
    """Separate faces OF the roof from things standing ON it. Two sources:
    whole facets that turn out to be elevated islands, and points sitting above
    the face they belong to."""
    obstructions = []
    keep = []
    budget = OBST_MAX_ROOF_SHARE * sum(f["area_m2"] for f in facets)
    spent = 0.0
    # Smallest first, so if the budget binds it is spent on the things most
    # likely to actually be equipment.
    for i, f in sorted(enumerate(facets), key=lambda kv: kv[1]["area_m2"]):
        if f["area_m2"] > OBST_MAX_AREA_M2 or spent + f["area_m2"] > budget:
            keep.append(f)
            continue
        # .boundary, not .exterior: absorbing an island can leave the parent a
        # MultiPolygon, and only Polygon has .exterior. Latent until the
        # enclosure threshold dropped and absorption started firing often.
        per = f["geometry"].boundary.length
        best, best_share = None, 0.0
        for j, g in enumerate(facets):
            if i == j:
                continue
            shared = f["geometry"].buffer(0.05).intersection(g["geometry"].boundary).length
            if shared > best_share:
                best, best_share = j, shared
        if best is None or per <= 0 or best_share / per < OBST_ENCLOSURE:
            keep.append(f)
            continue
        parent = facets[best]
        if parent["area_m2"] < OBST_PARENT_RATIO * f["area_m2"]:
            keep.append(f)      # comparable size: two faces, not a face and a box
            continue
        pplane = np.array([parent["plane_a"], parent["plane_b"], parent["plane_c"]])
        pp = pts[shapely.vectorized.contains(f["geometry"], pts[:, 0], pts[:, 1])]
        if len(pp) < OBST_MIN_PTS:
            keep.append(f)
            continue
        # Height measured AT THE SHARED EDGE, not over the footprint. Two faces
        # of a gable each sit "above" the other's extended plane further away,
        # which would make every second face of every house into plant.
        edge = f["geometry"].buffer(0.05).intersection(parent["geometry"].boundary)
        if edge.is_empty:
            keep.append(f)
            continue
        ec = edge.centroid
        step = abs(float(f["plane_a"] * ec.x + f["plane_b"] * ec.y + f["plane_c"])
                   - float(pplane[0] * ec.x + pplane[1] * ec.y + pplane[2]))
        if step < OBST_MIN_HEIGHT_M:
            keep.append(f)
            continue
        box = _box_from_points(pp[:, :2], pplane, pp)
        if box is None:
            keep.append(f)
            continue
        box["source"] = "island"
        obstructions.append(box)
        spent += f["area_m2"]
        # The face underneath does not stop existing because something sits on
        # it: the parent absorbs the footprint and the obstruction is carved
        # out later, the same way detected obstructions already are.
        parent["geometry"] = unary_union([parent["geometry"], f["geometry"]]).buffer(0.02).buffer(-0.02)
        parent["area_m2"] = float(parent["geometry"].area)
    facets = keep

    for f in facets:
        plane = np.array([f["plane_a"], f["plane_b"], f["plane_c"]])
        inside = pts[shapely.vectorized.contains(f["geometry"], pts[:, 0], pts[:, 1])]
        if len(inside) < OBST_MIN_PTS:
            continue
        sel = inside[-residuals(plane, inside) > OBST_MIN_HEIGHT_M]
        if len(sel) < OBST_MIN_PTS:
            continue
        lab = _cluster_xy(sel[:, :2], OBST_CLUSTER_EPS_M, OBST_MIN_PTS)
        for c in set(lab.tolist()) - {-1}:
            grp = sel[lab == c]
            box = _box_from_points(grp[:, :2], plane, grp)
            if box is None or box["area_m2"] > OBST_MAX_AREA_M2:
                continue
            box["source"] = "points"
            obstructions.append(box)
    return facets, obstructions


# A split has to earn back the setback area it costs. This is the rule the
# module docstring says any next attempt must obey, and it was never
# implemented -- the first version optimised "planes that fit the points",
# which is not the same thing as "panels fit on the result".
#
# panel_fitting erodes every facet by RIDGE_SETBACK_M along shared boundaries,
# so cutting one face into two loses a strip down the middle FOREVER:
#
#     6 m2 facet ->  57% of it usable
#    25 m2 facet ->  77%
#   150 m2 facet ->  90%
#   400 m2 facet ->  94%
#
# So two faces are only worth keeping apart when a panel genuinely cannot lie
# across the join. A panel is 1.7 m long; over that span a 5 degree fold rises
# 15 cm, which no rigid frame bridges, while 2-3 degrees is within the slack a
# real mounting rail takes up. Below the bridge angle the split is invisible on
# the roof and merging is strictly better -- it recovers the setback strip and
# removes an edge panels had to stop at.
#
# This also answers Josh on 29 Park St: "treating a gradually curving sloping
# roof as two planes when really it is just a light curve across the whole
# roof". A light curve is exactly a sequence of faces that differ by less than
# a panel can bridge, so it now comes back as one face.
BRIDGE_ANGLE_DEG = 5.0      # a fold shallower than this, a panel lies across
BRIDGE_MAX_STEP_M = 0.10    # ...provided the faces are not offset in height too
EARN_MIN_GAIN_M2 = 0.5      # ignore merges that win only rounding


def _plane_angle(f, g):
    """Angle between two facets' plane normals, in degrees."""
    na = np.array([-f["plane_a"], -f["plane_b"], 1.0])
    nb = np.array([-g["plane_a"], -g["plane_b"], 1.0])
    na /= np.linalg.norm(na)
    nb /= np.linalg.norm(nb)
    return float(np.degrees(np.arccos(np.clip(abs(na @ nb), -1.0, 1.0))))


def _step_at_join(f, g):
    """Height difference between two planes where they actually adjoin.

    Compared at the shared boundary, not at the origin: two parallel faces at
    different levels have IDENTICAL normals, so an angle test alone reads 0
    degrees and merges a step. That bug was found once already in
    roof_segmentation.merge_uneconomic_splits."""
    shared = f["geometry"].buffer(0.3).intersection(g["geometry"].buffer(0.3))
    if shared.is_empty:
        return float("inf")
    c = shared.centroid
    za = f["plane_a"] * c.x + f["plane_b"] * c.y + f["plane_c"]
    zb = g["plane_a"] * c.x + g["plane_b"] * c.y + g["plane_c"]
    return abs(float(za - zb))


def _usable(poly, setback=None):
    setback = config.RIDGE_SETBACK_M if setback is None else setback
    if poly.is_empty:
        return 0.0
    return float(poly.buffer(-setback).area)


def merge_to_earn_setback(facets, pts):
    """Merge adjacent faces a panel could lie across, when merging wins area.

    Greedy and repeated: each pass takes the merge with the largest gain, so a
    fan of narrow strips across a gently curved roof collapses from the middle
    outward rather than by whichever pair came first in the list."""
    if len(facets) < 2:
        return facets
    facets = list(facets)
    for _ in range(len(facets)):
        best = None
        for i in range(len(facets)):
            for j in range(i + 1, len(facets)):
                f, g = facets[i], facets[j]
                if not f["geometry"].buffer(0.3).intersects(g["geometry"]):
                    continue
                if _plane_angle(f, g) > BRIDGE_ANGLE_DEG:
                    continue
                if _step_at_join(f, g) > BRIDGE_MAX_STEP_M:
                    continue
                union = unary_union([f["geometry"], g["geometry"]])
                if union.geom_type != "Polygon":
                    continue
                gain = _usable(union) - _usable(f["geometry"]) - _usable(g["geometry"])
                if gain > EARN_MIN_GAIN_M2 and (best is None or gain > best[0]):
                    best = (gain, i, j, union)
        if best is None:
            break
        _, i, j, union = best
        f, g = facets[i], facets[j]
        member = pts[np.abs(residuals((f["plane_a"], f["plane_b"], f["plane_c"]), pts)) < RANSAC_TOL_M]
        inside = member[shapely.vectorized.contains(union, member[:, 0], member[:, 1])] \
            if len(member) else member
        plane = fit_plane(inside) if len(inside) >= 8 else (f["plane_a"], f["plane_b"], f["plane_c"])
        slope, aspect = plane_slope_aspect(plane)
        merged = dict(f)
        merged.update({"geometry": Polygon(union.exterior, [r for r in union.interiors]),
                       "plane_a": float(plane[0]), "plane_b": float(plane[1]),
                       "plane_c": float(plane[2]), "slope_deg": float(slope),
                       "aspect_deg": float(aspect), "area_m2": float(union.area),
                       "point_count": int(len(inside))})
        facets = [x for k, x in enumerate(facets) if k not in (i, j)] + [merged]
    return facets


def reconstruct(building_id, outline, pts, seed=0, with_obstructions=True):
    """Point cloud + surveyed outline -> straight-edged, plane-backed facets."""
    if len(pts) < MIN_PLANE_PTS:
        return []
    rng = np.random.default_rng(seed)
    planes, owner = ransac_planes(pts, rng)
    if not planes:
        return []
    lab, gx, gy = label_raster(outline, pts, owner, planes)
    if lab is None:
        return []
    pairs = adjacent_pairs(lab)
    lines = edge_lines(planes, pairs, lab, gx, gy, outline.bounds)

    clipped = []
    for ln in lines:
        piece = ln.intersection(outline)
        if piece.is_empty:
            continue
        clipped.extend(piece.geoms if piece.geom_type == "MultiLineString" else [piece])
    cells = list(polygonize(unary_union([outline.boundary] + clipped)))
    if not cells:
        cells = [outline]
    cells = refine_mixed_cells(cells, planes, pts, outline.bounds)

    # Each cell goes to the plane its own points support. Cells with too few
    # points to vote inherit from the nearest labelled point instead of being
    # guessed at.
    tree = cKDTree(pts[:, :2])
    claimed = owner >= 0
    ntree = cKDTree(pts[claimed][:, :2]) if claimed.any() else None
    assigned, cell_label = {}, []
    for cell in cells:
        if cell.area < 0.5:
            continue
        idx = tree.query_ball_point(np.array(cell.centroid.coords[0]),
                                    r=math.hypot(*(np.array(cell.bounds[2:]) - np.array(cell.bounds[:2]))) / 2 + 1.0)
        if idx:
            sub = pts[idx]
            inside = shapely.vectorized.contains(cell, sub[:, 0], sub[:, 1])
            sub = sub[inside]
        else:
            sub = pts[:0]
        best, best_n = None, 0
        if len(sub) >= MIN_CELL_PTS:
            for pi, plane in enumerate(planes):
                n = int((np.abs(residuals(plane, sub)) < RANSAC_TOL_M).sum())
                if n > best_n:
                    best, best_n = pi, n
        if best is None and ntree is not None:
            _, nn = ntree.query(np.array(cell.centroid.coords[0]))
            best = int(owner[claimed][nn])
        if best is None:
            continue
        cell_label.append((cell, best, sub))

    # Speckle removal: an isolated cell whose neighbours all say otherwise, and
    # whose own points do not strongly disagree, takes the neighbours' label.
    for _ in range(SMOOTH_ROUNDS):
        moved = 0
        for k, (cell, lab_k, sub) in enumerate(cell_label):
            share = {}
            for m, (other, lab_m, _) in enumerate(cell_label):
                if m == k or lab_m == lab_k:
                    continue
                b = cell.buffer(0.05).intersection(other).area
                if b > 0:
                    share[lab_m] = share.get(lab_m, 0.0) + b
            if not share:
                continue
            cand = max(share, key=share.get)
            if len(sub) < MIN_CELL_PTS:
                cell_label[k] = (cell, cand, sub)
                moved += 1
                continue
            n_own = int((np.abs(residuals(planes[lab_k], sub)) < RANSAC_TOL_M).sum())
            n_cand = int((np.abs(residuals(planes[cand], sub)) < RANSAC_TOL_M).sum())
            if n_cand >= 0.8 * n_own:
                cell_label[k] = (cell, cand, sub)
                moved += 1
        if not moved:
            break
    for cell, lab_k, _ in cell_label:
        assigned.setdefault(lab_k, []).append(cell)

    facets = []
    for pi, cs in assigned.items():
        geom = unary_union(cs)
        for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            if poly.area < MIN_FACET_M2:
                continue
            member = pts[np.abs(residuals(planes[pi], pts)) < RANSAC_TOL_M]
            inside = member[shapely.vectorized.contains(poly, member[:, 0], member[:, 1])] \
                if len(member) else member
            plane = fit_plane(inside) if len(inside) >= 8 else planes[pi]
            slope, aspect = plane_slope_aspect(plane)
            if slope > WALL_SLOPE_DEG:
                continue
            facets.append({
                "building_id": building_id,
                "plane_a": float(plane[0]), "plane_b": float(plane[1]), "plane_c": float(plane[2]),
                "slope_deg": float(slope), "aspect_deg": float(aspect),
                "area_m2": float(poly.area), "point_count": int(len(inside)),
                "geometry": Polygon(poly.exterior, [r for r in poly.interiors]),
            })
    facets = merge_coplanar(facets, pts)
    facets = merge_to_earn_setback(facets, pts)
    if not with_obstructions:
        return assign_plane_ids(facets), []
    facets, obstructions = extract_obstructions(facets, pts)
    return assign_plane_ids(facets), obstructions
